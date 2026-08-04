"""Measure how square-on each coin was photographed.

A coin is a disc, so a camera directly above it records a circle and a camera
off to one side records an ellipse. That projection is the thing worth
filtering on here: an obliquely shot *ordinary* coin is elliptical, and an
elliptical outline is exactly what a geometric striking-error detector is
looking for. Oblique photos therefore do not merely add noise, they teach
the detector that normal coins are deformed.

The measurement is the one asked for: find the coin, take its centre, take
the longest distance from that centre to its edge, draw the circle that
radius describes, and report how much of the circle the coin fails to fill.

    missing = 1 - coin_area / (pi * r_max^2)

For an ellipse with semi-axes a >= b that is 1 - b/a, and since a tilt of t
degrees foreshortens one axis by cos(t), the number converts straight into
an angle: tilt = arccos(1 - missing). So the output is in degrees, which is
a quantity worth arguing about -- "keep everything under 25 degrees" is a
decision a person can make, "keep everything above 0.83 circularity" is not.

The measurement only means anything when the whole coin is inside the frame.
Roughly half this set is macro work -- the coin overflowing the picture, or a
crop of one arc of its rim -- and there the coin has no outline to find, so
the threshold latches onto some brightness feature in the relief instead and
reports a violent tilt for a photo taken straight down. Those are refused
rather than scored: a contour that runs along the frame edge is a cropped
coin, and its angle is simply unknown.

Two further caveats the number cannot separate, both reported not hidden:

  * r_max is a single farthest pixel, so one stray blob welded to the coin
    by the threshold inflates it and fakes a tilt. Measured against hand-
    checked photos this is not a marginal effect: a 1 euro segmented
    perfectly, and circular to the eye, scored 19.9 degrees from r_max and
    9.6 from a least-squares ellipse fit over the whole outline, whose axis
    ratio was 0.986. The headline number is therefore the ellipse fit, with
    the r_max figure kept beside it -- when the two disagree badly the
    segmentation is what is wrong, not the camera.

    Both are reported as a ratio first and an angle second, because the
    angle is arccos of the ratio and arccos is near-vertical at 1: a 6%
    error in the outline becomes "20 degrees of tilt". Ratios are what the
    threshold should be set on; degrees are for intuition only.
  * a coin that is genuinely not round -- a clipped planchet, a broken flan,
    a big off-centre strike -- also fails to fill its circle. No projection
    measurement can tell that from a tilt, so a low score means "not a
    circle" and a human still decides which kind.

Nothing is discarded. Results land in `manifest_judged.json` as `round_*`
columns and in `roundness.json`, and --overlay writes annotated copies (coin
outline in green, the r_max circle in red) for eyeballing the threshold.

    python tools/measure_roundness.py --overlay 40
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

WORK_EDGE = 700


def coin_mask(image: np.ndarray) -> np.ndarray | None:
    """Largest solid blob that is plausibly the coin.

    Segmented by growing the *background* inward from the picture edge
    rather than by thresholding brightness. Brightness fails on the coins
    that matter most: a 1 or 2 euro is bimetallic, its inner disc and outer
    ring sit on opposite sides of any global threshold, and Otsu duly
    returns the inner disc alone -- a good top-down photo then scores as
    violently tilted. What is true regardless of the coin's own colouring is
    that the backdrop touches the frame and the coin does not, so a flood
    fill from the border finds the backdrop whatever the coin is made of,
    and whatever is left standing is the coin entire.

    No roundness test anywhere in here: gating segmentation on circularity
    is what made the previous pass blind to the very photos this script
    exists to find.
    """
    height, width = image.shape[:2]
    frame_area = float(height * width)
    blurred = cv2.GaussianBlur(image, (7, 7), 0)

    # Seed from the midpoint of each edge as well as the corners: a coin
    # sitting in one corner of the frame would otherwise be seeded into.
    seeds = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1),
             (width // 2, 0), (width // 2, height - 1),
             (0, height // 2), (width - 1, height // 2)]

    best, best_fill = None, 0.0
    for tolerance in (12, 24, 40):
        filled = np.zeros((height + 2, width + 2), np.uint8)
        work = blurred.copy()
        for seed in seeds:
            # MASK_ONLY so each seed judges the original picture; without it
            # the previous fill's paint becomes the next seed's neighbour
            # and the background bleeds across the coin.
            cv2.floodFill(work, filled, seed, 255,
                          (tolerance,) * 3, (tolerance,) * 3,
                          4 | cv2.FLOODFILL_FIXED_RANGE
                          | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
        coin = np.where(filled[1:-1, 1:-1] > 0, 0, 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        coin = cv2.morphologyEx(coin, cv2.MORPH_CLOSE, kernel)
        coin = cv2.morphologyEx(coin, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(coin, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < frame_area * 0.02 or area > frame_area * 0.97:
                continue  # speck, or the whole frame
            x, y, w, h = cv2.boundingRect(contour)
            # Fill of its own bounding box: an ellipse at any angle scores
            # ~0.79, a hand or a caption scores far lower.
            fill = area / float(w * h)
            if fill > best_fill:
                best_fill, best = fill, contour
        if best_fill > 0.72:
            break  # a clean disc; looser tolerances would only bleed
    return best


def border_fraction(contour: np.ndarray, shape: tuple[int, int]) -> float:
    """Share of the outline lying on the picture edge."""
    height, width = shape
    points = contour.reshape(-1, 2)
    on_edge = ((points[:, 0] <= 1) | (points[:, 0] >= width - 2)
               | (points[:, 1] <= 1) | (points[:, 1] >= height - 2))
    return float(on_edge.mean()) if len(points) else 1.0


def measure(path: Path) -> dict:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return {"round_ok": False, "round_why": "unreadable"}

    height, width = image.shape[:2]
    scale = WORK_EDGE / max(height, width)
    if scale < 1:
        image = cv2.resize(image, (int(width * scale), int(height * scale)),
                           interpolation=cv2.INTER_AREA)

    contour = coin_mask(image)
    if contour is None:
        return {"round_ok": False, "round_why": "no coin found"}

    # A contour running along the picture edge is not the coin's outline --
    # the coin is bigger than the frame. Nothing about its shape can be
    # measured from a crop of its middle, so say so instead of guessing.
    edge_fraction = border_fraction(contour, image.shape[:2])
    if edge_fraction > 0.02:
        return {"round_ok": False,
                "round_why": "coin cropped by frame — angle unknowable",
                "round_border": round(edge_fraction, 3)}

    # Centre of mass, not the bounding box centre: a foreshortened coin is
    # still symmetric about its own centroid.
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return {"round_ok": False, "round_why": "degenerate contour"}
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    points = contour.reshape(-1, 2).astype(np.float64)
    radii = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
    r_max = float(radii.max())
    area = float(cv2.contourArea(contour))
    missing = 1.0 - area / (math.pi * r_max * r_max)
    missing = min(max(missing, 0.0), 1.0)

    # The same quantity from a robust radius, so a single spur on the
    # outline can be told from a genuinely squashed coin.
    r_p98 = float(np.percentile(radii, 98))
    missing_robust = min(max(1.0 - area / (math.pi * r_p98 * r_p98), 0.0), 1.0)

    # Independent cross-check: least-squares ellipse over the whole outline.
    axis_ratio = None
    if len(points) >= 5:
        (_, _), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
        if max(axis_a, axis_b) > 0:
            axis_ratio = float(min(axis_a, axis_b) / max(axis_a, axis_b))

    out = {
        "round_ok": True,
        "round_missing": round(missing, 4),
        "round_missing_robust": round(missing_robust, 4),
        "round_rmax_tilt_deg": round(math.degrees(math.acos(
            min(max(1.0 - missing_robust, -1.0), 1.0))), 1),
        "round_axis_ratio": None if axis_ratio is None else round(axis_ratio, 3),
        "round_tilt_deg": None if axis_ratio is None else round(
            math.degrees(math.acos(min(axis_ratio, 1.0))), 1),
        "round_centre": [round(cx, 1), round(cy, 1)],
        "round_rmax": round(r_max, 1),
    }
    out["round_verdict"] = verdict_for(axis_ratio)
    return out


#: Set from hand-checked photos: a well segmented, visibly circular coin
#: lands at 0.98-0.99, and a coin the eye calls tilted is below 0.90.
SQUARE_ON = 0.95
SLIGHT = 0.90


def verdict_for(axis_ratio: float | None) -> str:
    """Square-on enough to train a geometric detector on?

    "oblique" means only *not verifiably circular*: a foreshortened photo, a
    coin that genuinely is not round, and a coin the segmenter got wrong all
    land there together, and the three are not separable from the outline.
    """
    if axis_ratio is None:
        return "unknown"
    if axis_ratio >= SQUARE_ON:
        return "square-on"
    if axis_ratio >= SLIGHT:
        return "slight-angle"
    return "oblique"


def write_overlay(path: Path, out: Path) -> None:
    """The coin outline in green and the r_max circle in red, so the gap
    between them is the missing area the score is made of."""
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return
    height, width = image.shape[:2]
    scale = WORK_EDGE / max(height, width)
    if scale < 1:
        image = cv2.resize(image, (int(width * scale), int(height * scale)),
                           interpolation=cv2.INTER_AREA)
    contour = coin_mask(image)
    if contour is None:
        return
    moments = cv2.moments(contour)
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    points = contour.reshape(-1, 2).astype(np.float64)
    r_max = float(np.hypot(points[:, 0] - cx, points[:, 1] - cy).max())

    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
    cv2.circle(image, (int(cx), int(cy)), int(r_max), (0, 0, 255), 2)
    cv2.circle(image, (int(cx), int(cy)), 3, (255, 0, 0), -1)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), image)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("data/errors_ref/emuenzen"))
    parser.add_argument("--overlay", type=int, default=0,
                        help="write this many annotated samples across the range")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    source = args.root / "manifest_judged.json"
    if not source.exists():
        source = args.root / "manifest.json"
    rows = json.loads(source.read_text(encoding="utf-8"))

    cache_path = args.root / "roundness.json"
    cache: dict[str, dict] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    todo = sorted({r["file"] for r in rows} - set(cache))
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} photos to measure", file=sys.stderr)

    started = time.time()
    for i, name in enumerate(todo):
        cache[name] = measure(args.root / "images" / name)
        if (i + 1) % 200 == 0:
            cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
            print(f"  {i + 1}/{len(todo)}  ({time.time() - started:.0f}s)",
                  file=sys.stderr)
    cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")

    for row in rows:
        row.update(cache.get(row["file"], {}))
        row.setdefault("scraper_verdict", row["verdict"])
    (args.root / "manifest_judged.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    from collections import Counter
    good = [c for c in cache.values() if c.get("round_ok")]
    tilts = sorted(c["round_tilt_deg"] for c in good)
    print(f"\n{len(good)} measurable, {len(cache) - len(good)} not:",
          file=sys.stderr)
    for why, count in Counter(c.get("round_why", "?") for c in cache.values()
                              if not c.get("round_ok")).most_common():
        print(f"  {count:4d}  {why}", file=sys.stderr)
    if good:
        print("\nverdicts among the measurable:", file=sys.stderr)
        for verdict, count in Counter(
                c["round_verdict"] for c in good).most_common():
            print(f"  {count:4d}  {verdict}", file=sys.stderr)
        ratios = sorted(c["round_axis_ratio"] for c in good
                        if c.get("round_axis_ratio"))
        print("\naxis ratio (1.000 = a perfect circle):", file=sys.stderr)
        for pct in (10, 25, 50, 75, 90):
            print(f"  p{pct:<3d} {ratios[int(len(ratios) * pct / 100) - 1]:.3f}",
                  file=sys.stderr)
        print("\nkept at each cut-off:", file=sys.stderr)
        for cut in (0.99, 0.97, 0.95, 0.92, 0.90):
            keep = sum(1 for r in ratios if r >= cut)
            print(f"  ratio >= {cut:.2f}  ({math.degrees(math.acos(cut)):4.1f}° "
                  f"tilt)  {keep:4d} of {len(ratios)} "
                  f"({keep / len(ratios):.0%})", file=sys.stderr)

    if args.overlay:
        by_tilt = sorted((c["round_tilt_deg"], n) for n, c in cache.items()
                         if c.get("round_ok"))
        step = max(1, len(by_tilt) // args.overlay)
        out_dir = args.root / "overlays"
        for tilt, name in by_tilt[::step]:
            write_overlay(args.root / "images" / name,
                          out_dir / f"{tilt:05.1f}deg__{Path(name).name}")
        print(f"\noverlays written to {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
