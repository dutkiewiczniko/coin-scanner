"""Match a coin's two faces across the pair of tray photos.

Each batch is photographed twice: once in tray A, then flipped into tray B to
expose the other face. Both photos rectify into the same metric frame, so the
two faces of one coin land at nearly the same coordinates — which is what makes
matching tractable at all.

Two coins of different denominations cannot be the same coin, so diameter acts
as an independent check on position. Where the two disagree, the pair is
rejected rather than guessed at: an unmatched coin is a visible gap, whereas a
wrong match silently attaches one coin's obverse to another's reverse.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np

from .config import Config, PairConfig
from .stages.calibrate import to_reference_frame
from .types import Coin, ScanResult, format_denomination


@dataclass
class PairedCoin:
    """One physical coin, with whichever of its two faces were found."""

    coin_id: str
    batch: str
    index: int
    face_a: Optional[Coin] = None
    face_b: Optional[Coin] = None
    #: Position in the shared reference frame, in millimetres from the tray's
    #: top-left marker. This is how a coin is located in the tray by hand.
    x_mm: float = 0.0
    y_mm: float = 0.0
    #: How far apart the two faces landed. Small means a confident match.
    offset_mm: float = 0.0
    #: Disagreement between the two faces' measured diameters.
    diameter_delta_mm: float = 0.0

    @property
    def is_paired(self) -> bool:
        return self.face_a is not None and self.face_b is not None

    @property
    def faces_found(self) -> str:
        if self.is_paired:
            return "both"
        return "a" if self.face_a is not None else "b"

    @property
    def coin(self) -> Coin:
        """Whichever face is present, preferring side A."""
        found = self.face_a or self.face_b
        assert found is not None  # a PairedCoin always has at least one face
        return found

    @property
    def diameter_mm(self) -> Optional[float]:
        """Best diameter estimate: the mean of both faces when paired.

        Averaging two independent measurements of the same coin is a real
        improvement, since the per-face error is a few tenths of a millimetre
        against denomination gaps of one.
        """
        values = [
            c.diameter_mm
            for c in (self.face_a, self.face_b)
            if c is not None and c.diameter_mm is not None
        ]
        if not values:
            return None
        return round(sum(values) / len(values), 2)

    @property
    def denomination(self) -> Optional[int]:
        """Denomination from the combined diameter, not one arbitrary face.

        Taking the size from both faces but the label from side A alone lets the
        two contradict each other — a coin captioned "10c" while reporting a
        diameter nearest 2c. Deriving both from the same number prevents that.
        """
        diameter = self.diameter_mm
        if diameter is None:
            return self.coin.denomination
        from .stages.classify import nearest_denomination

        return nearest_denomination(diameter)[0]

    def to_record(self) -> dict[str, Any]:
        return {
            "coin_id": self.coin_id,
            "batch": self.batch,
            "index": self.index,
            "faces_found": self.faces_found,
            "x_mm": round(self.x_mm, 1),
            "y_mm": round(self.y_mm, 1),
            "diameter_mm": self.diameter_mm,
            "denomination": self.denomination,
            "denomination_label": format_denomination(self.denomination),
            "offset_mm": round(self.offset_mm, 2),
            "diameter_delta_mm": round(self.diameter_delta_mm, 2),
            "is_flagged": self.coin.is_flagged,
            "rare_notes": self.coin.rare_notes,
        }


@dataclass
class BatchResult:
    """Everything known about one batch after both faces are matched."""

    batch: str
    coins: list[PairedCoin] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def paired(self) -> list[PairedCoin]:
        return [c for c in self.coins if c.is_paired]

    @property
    def unpaired(self) -> list[PairedCoin]:
        return [c for c in self.coins if not c.is_paired]

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch": self.batch,
            "meta": self.meta,
            "total": len(self.coins),
            "paired": len(self.paired),
            "unpaired": len(self.unpaired),
            "coins": [c.to_record() for c in self.coins],
        }


def _reference_positions(result: ScanResult) -> list[tuple[float, float]]:
    """Each coin's position in the shared frame, in millimetres."""
    side = result.meta.get("side", "a")
    height_px = result.meta["image_size"][1]
    scale = result.meta["pixels_per_mm"]
    positions = []
    for coin in result.coins:
        x, y = to_reference_frame(
            coin.detection.x, coin.detection.y, side, height_px
        )
        positions.append((x / scale, y / scale))
    return positions


def estimate_alignment(
    pos_a: list[tuple[float, float]],
    pos_b: list[tuple[float, float]],
    config: PairConfig,
) -> tuple[float, float, int]:
    """Recover the constant offset between the two trays' frames.

    The trays nest inside one another rather than meeting corner to corner, so
    their marker rectangles sit at a fixed offset. Rather than ask for it to be
    measured, it is read off the coins: every candidate pair votes for the shift
    that would align it, and the true shift is the one every coin agrees on
    while wrong pairings scatter.

    Returns the shift and how many coins supported it. Support below
    `align_min_support` means no offset was convincingly found, and the caller
    should not apply one.
    """
    if not pos_a or not pos_b:
        return 0.0, 0.0, 0

    reach = config.align_search_mm
    votes = [
        (ax - bx, ay - by)
        for ax, ay in pos_a
        for bx, by in pos_b
        if abs(ax - bx) <= reach and abs(ay - by) <= reach
    ]
    if not votes:
        return 0.0, 0.0, 0

    votes_array = np.array(votes)
    tolerance = config.align_tolerance_mm

    # Each vote is itself a candidate shift, so there is no grid to choose.
    best_shift = (0.0, 0.0)
    best_support = 0
    for dx, dy in votes:
        near = np.hypot(votes_array[:, 0] - dx, votes_array[:, 1] - dy) < tolerance
        support = int(near.sum())
        if support > best_support:
            best_support = support
            # Average the agreeing votes rather than taking the seed, which
            # would quantise the estimate to whatever pair happened to win.
            best_shift = (
                float(votes_array[near, 0].mean()),
                float(votes_array[near, 1].mean()),
            )

    return best_shift[0], best_shift[1], best_support


def _aligned(
    pos_a: list[tuple[float, float]],
    pos_b: list[tuple[float, float]],
    config: PairConfig,
) -> tuple[list[tuple[float, float]], tuple[float, float, int]]:
    """Side B's positions shifted into side A's frame, plus the shift used."""
    if not config.auto_align:
        return pos_b, (0.0, 0.0, 0)

    dx, dy, support = estimate_alignment(pos_a, pos_b, config)
    needed = config.align_min_support * min(len(pos_a), len(pos_b))
    if support < needed:
        return pos_b, (0.0, 0.0, support)  # not convincing; leave it alone
    return [(x + dx, y + dy) for x, y in pos_b], (dx, dy, support)


def match_faces(
    result_a: ScanResult, result_b: ScanResult, config: PairConfig
) -> list[tuple[Optional[int], Optional[int], float, float]]:
    """Pair up coin indices between the two sides.

    Returns (index_a, index_b, offset_mm, diameter_delta_mm) tuples, with None
    for a face that had no acceptable partner.

    Candidate pairs are accepted cheapest-first, each coin used once. A globally
    optimal assignment would need scipy; greedy is equivalent here because the
    accepted matches are an order of magnitude closer than any competing one —
    the same coin photographed twice lands within a millimetre or two, while the
    nearest different coin is a whole diameter away.
    """
    pos_a = _reference_positions(result_a)
    pos_b = _reference_positions(result_b)
    pos_b = _aligned(pos_a, pos_b, config)[0]

    candidates = []
    for i, (ax, ay) in enumerate(pos_a):
        da = result_a.coins[i].diameter_mm
        for j, (bx, by) in enumerate(pos_b):
            offset = float(np.hypot(ax - bx, ay - by))
            if offset > config.max_offset_mm:
                continue

            db = result_b.coins[j].diameter_mm
            delta = abs(da - db) if (da is not None and db is not None) else 0.0
            if delta > config.max_diameter_delta_mm:
                continue

            cost = offset + config.diameter_weight * delta
            candidates.append((cost, offset, delta, i, j))

    candidates.sort()
    used_a: set[int] = set()
    used_b: set[int] = set()
    matches: list[tuple[Optional[int], Optional[int], float, float]] = []

    for _, offset, delta, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matches.append((i, j, offset, delta))

    for i in range(len(pos_a)):
        if i not in used_a:
            matches.append((i, None, 0.0, 0.0))
    for j in range(len(pos_b)):
        if j not in used_b:
            matches.append((None, j, 0.0, 0.0))

    return matches


def pair_batch(
    batch: str, result_a: ScanResult, result_b: ScanResult, config: Config
) -> BatchResult:
    """Match both faces and number the coins for tracking."""
    matches = match_faces(result_a, result_b, config.pair)
    pos_a = _reference_positions(result_a)
    pos_b, alignment = _aligned(
        pos_a, _reference_positions(result_b), config.pair
    )

    entries = []
    for index_a, index_b, offset, delta in matches:
        face_a = result_a.coins[index_a] if index_a is not None else None
        face_b = result_b.coins[index_b] if index_b is not None else None
        if index_a is not None:
            x_mm, y_mm = pos_a[index_a]
        else:
            x_mm, y_mm = pos_b[index_b]  # type: ignore[index]
        entries.append((x_mm, y_mm, face_a, face_b, offset, delta))

    # Number in reading order so a coin's id corresponds to where it sits in the
    # tray. Rows are banded before sorting by x, otherwise a coin sitting a few
    # millimetres high would sort into the previous row.
    band = config.pair.row_band_mm
    entries.sort(key=lambda e: (round(e[1] / band), e[0]))

    coins = [
        PairedCoin(
            coin_id=f"{batch}-{i + 1:03d}",
            batch=batch,
            index=i + 1,
            face_a=face_a,
            face_b=face_b,
            x_mm=x_mm,
            y_mm=y_mm,
            offset_mm=offset,
            diameter_delta_mm=delta,
        )
        for i, (x_mm, y_mm, face_a, face_b, offset, delta) in enumerate(entries)
    ]

    result = BatchResult(batch=batch, coins=coins)
    result.meta = {
        "source_a": result_a.source,
        "source_b": result_b.source,
        "found_a": len(result_a.coins),
        "found_b": len(result_b.coins),
        "pixels_per_mm": result_a.meta.get("pixels_per_mm"),
        "alignment_mm": [round(alignment[0], 2), round(alignment[1], 2)],
        "alignment_support": alignment[2],
    }
    return result


# -- output ---------------------------------------------------------------

_BG = (32, 32, 32)
_FG = (235, 235, 235)
_DIM = (150, 150, 150)
_FLAG = (60, 190, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _face_tile(coin: Optional[Coin], size: int, label: str) -> np.ndarray:
    """One face, or a placeholder when that face was not found."""
    tile = np.full((size, size, 3), _BG, dtype=np.uint8)
    if coin is not None and coin.normalised is not None:
        face = cv2.resize(coin.normalised, (size, size), interpolation=cv2.INTER_AREA)
        tile = face
    else:
        text = "no match"
        (tw, _), _ = cv2.getTextSize(text, _FONT, 0.6, 1)
        cv2.putText(tile, text, ((size - tw) // 2, size // 2), _FONT, 0.6, _DIM, 1,
                    cv2.LINE_AA)
    cv2.putText(tile, label, (8, size - 10), _FONT, 0.5, _FG, 1, cv2.LINE_AA)
    return tile


def compose_coin(paired: PairedCoin, config: PairConfig) -> np.ndarray:
    """Both faces side by side, captioned so the file stands alone."""
    size = config.composite_face_px
    pad = 12
    header, footer = 40, 34

    width = size * 2 + pad * 3
    height = header + size + footer + pad
    canvas = np.full((height, width, 3), _BG, dtype=np.uint8)

    canvas[header:header + size, pad:pad + size] = _face_tile(
        paired.face_a, size, "side A"
    )
    canvas[header:header + size, pad * 2 + size:pad * 2 + size * 2] = _face_tile(
        paired.face_b, size, "side B"
    )

    denomination = format_denomination(paired.denomination) or "unknown"
    cv2.putText(canvas, paired.coin_id, (pad, 28), _FONT, 0.8, _FG, 2, cv2.LINE_AA)
    title = f"{denomination}"
    if paired.diameter_mm is not None:
        title += f"   {paired.diameter_mm:.2f} mm"
    (tw, _), _ = cv2.getTextSize(title, _FONT, 0.6, 1)
    cv2.putText(canvas, title, (width - tw - pad, 26), _FONT, 0.6,
                _FLAG if paired.coin.is_flagged else _FG, 1, cv2.LINE_AA)

    where = f"tray {paired.x_mm:.0f},{paired.y_mm:.0f} mm"
    if paired.is_paired:
        where += f"   offset {paired.offset_mm:.1f} mm"
    else:
        where += f"   FACE {paired.faces_found.upper()} ONLY"
    cv2.putText(canvas, where, (pad, header + size + 24), _FONT, 0.5, _DIM, 1,
                cv2.LINE_AA)
    return canvas


def write_batch(result: BatchResult, config: Config) -> dict[str, str]:
    """Write per-coin composites plus a manifest for tracking.

    The manifest is the durable record: it says which coin got which id, where
    it sat in the tray, and what it measured, so a coin can still be identified
    after the tray has been emptied.
    """
    out_dir = Path(config.output_dir) / f"batch_{result.batch}"
    coins_dir = out_dir / "coins"
    coins_dir.mkdir(parents=True, exist_ok=True)

    from .pipeline import save_image

    for paired in result.coins:
        name = f"{paired.coin_id}"
        if paired.coin.is_flagged:
            name += "_flagged"
        if not paired.is_paired:
            name += "_unpaired"
        save_image(coins_dir / f"{name}.png", compose_coin(paired, config.pair))

    written = {"coins": str(coins_dir)}

    records = [c.to_record() for c in result.coins]
    csv_path = out_dir / "manifest.csv"
    if records:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        written["csv"] = str(csv_path)

    json_path = out_dir / "manifest.json"
    json_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    written["json"] = str(json_path)

    return written
