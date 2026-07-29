"""Cut each reference coin out of its listing photo onto a white background.

The collected photographs are whatever the seller happened to take: a coin in
someone's hand, on a kitchen table, on a scanner, two faces side by side. That
is unusable for two different jobs at once --

  eyeballing    sorting a thousand images into "real fault" and "ordinary coin
                someone hoped was valuable" is fast on uniform crops and slow on
                a wall of hands and worktops.
  measuring     the anomaly scorer expects what the pipeline's own normalise
                stage produces: one centred coin, background masked away.

so the reference set is put into the same shape as the pipeline's crops.

Two detection strategies, chosen per image. Sellers who photograph on white --
dealers, mostly -- give a clean silhouette that thresholding finds exactly.
Everything else goes to Hough, which is less accurate but does not need the
background to cooperate. The choice is made by looking at the border pixels
rather than by trying Hough first, because Hough returns *something* on almost
any input and a confident wrong circle is worse than a slow correct one.

Radius here is a crop bound, not a measurement. The pipeline measures diameter
radially from the tray image for good reasons (see `segment.radial_measure`);
none of that applies to a photograph with no scale in it.

    .venv/Scripts/python tools/normalise_refs.py data/errors_ref/ebay_de
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

#: Listing photos are often two faces of one coin side by side. Both are wanted.
MAX_COINS = 2


def border_is_plain(image: np.ndarray, band: int = 12) -> bool:
    """Whether the photo sits on a uniform studio background.

    Sampled from the frame edge rather than the corners: a scanned coin is
    frequently placed off-centre, which leaves one corner holding a shadow and
    makes a four-corner test disagree with itself.
    """
    edges = np.concatenate([
        image[:band].reshape(-1, 3),
        image[-band:].reshape(-1, 3),
        image[:, :band].reshape(-1, 3),
        image[:, -band:].reshape(-1, 3),
    ])
    # Bright and consistent. The threshold is loose on brightness because
    # "white" backgrounds photograph grey, and tight on spread because that is
    # what actually separates a backdrop from a worktop.
    return bool(edges.mean() > 150 and edges.std() < 34)


def circles_by_threshold(gray: np.ndarray) -> list[tuple[int, int, int]]:
    """Coins as the dark blobs on a light background."""
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blurred, 0, 255,
                            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    frame_area = gray.shape[0] * gray.shape[1]
    found = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:6]:
        area = cv2.contourArea(contour)
        if area < frame_area * 0.01:
            continue
        (x, y), radius = cv2.minEnclosingCircle(contour)
        # A coin fills most of the circle that encloses it. A hand, a caption
        # bar or a capsule does not, and this is what rejects them.
        if area / (np.pi * radius * radius) < 0.72:
            continue
        found.append((int(x), int(y), int(radius)))
    return found


def circles_by_hough(gray: np.ndarray) -> list[tuple[int, int, int]]:
    """Coins as circles, for photos whose background will not threshold."""
    short = min(gray.shape[:2])
    # A banner or a sliver of a photo can be a handful of pixels tall, which
    # rounds minDist and minRadius to zero. HoughCircles then tries to allocate
    # an accumulator for every possible centre and radius and dies with "bad
    # allocation", taking the whole run with it.
    if short < 64:
        return []
    min_dist = max(8, int(short * 0.30))
    min_radius = max(4, int(short * 0.12))
    max_radius = max(min_radius + 1, int(short * 0.52))

    blurred = cv2.medianBlur(gray, 5)
    try:
        detected = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=min_dist, param1=110, param2=55,
            minRadius=min_radius, maxRadius=max_radius,
        )
    except cv2.error:
        # Still possible on a pathological image, and one photo is not worth
        # losing the other four hundred over.
        return []
    if detected is None:
        return []
    return [(int(x), int(y), int(r)) for x, y, r in np.round(detected[0]).astype(int)]


def cut_out(image: np.ndarray, x: int, y: int, radius: int, size: int,
            padding: float, rim_slack: float) -> np.ndarray:
    """Square crop centred on the coin, everything outside the rim made white.

    Two separate radii here, and conflating them is what cut the rims off the
    first version's output.

    `keep` is where the mask is drawn, and it is deliberately larger than the
    detected radius. Hough under-reads radius -- the project README says as much
    about the tray pipeline, and on bimetallic euros it frequently locks onto
    the boundary between core and ring rather than the outer edge. Masking at
    exactly the detected radius therefore shaves off the very rim that a
    clipped, broadstruck or off-centre coin shows its fault on. Better to admit
    a thin ring of background than to erase the evidence.

    `reach` is the framing, and only adds white margin around that.
    """
    keep = max(1, int(radius * rim_slack))
    reach = int(keep * (1.0 + padding))
    # Pad first so a coin near the frame edge is not cropped to a rectangle.
    canvas = cv2.copyMakeBorder(image, reach, reach, reach, reach,
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cx, cy = x + reach, y + reach
    patch = canvas[cy - reach:cy + reach, cx - reach:cx + reach]
    if patch.size == 0:
        raise ValueError("empty crop")

    mask = np.zeros(patch.shape[:2], np.uint8)
    cv2.circle(mask, (reach, reach), keep, 255, -1)
    white = np.full_like(patch, 255)
    patch = np.where(mask[:, :, None] == 255, patch, white)

    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)


def dedupe(circles: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Drop circles that sit on top of one another, keeping the larger.

    Two coins photographed side by side barely touch, so genuine neighbours are
    nearly non-overlapping. The earlier threshold here allowed centres only
    0.6 radii apart, which let a near-duplicate of the same coin survive and be
    cropped as a crescent -- the lens-shaped output that made this obvious.
    """
    kept: list[tuple[int, int, int]] = []
    for x, y, r in sorted(circles, key=lambda c: -c[2]):
        if all(np.hypot(x - kx, y - ky) > (r + kr) * 0.75 for kx, ky, kr in kept):
            kept.append((x, y, r))
    return kept


def edge_support(gray: np.ndarray, x: int, y: int, r: int) -> float:
    """How much of the proposed rim lies on an actual brightness step.

    Hough returns a best-scoring circle whether or not a coin is there, so on a
    photo of a worktop it confidently reports the grain. The accumulator score
    is not exposed and is not comparable between images anyway, so the circle is
    re-checked directly: step just inside and just outside the rim at intervals
    and see whether the two differ. On a coin they differ nearly all the way
    round; on wood grain they agree.
    """
    if r < 8:
        return 0.0
    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    inner_r, outer_r = r * 0.90, r * 1.12
    height, width = gray.shape[:2]
    hits = 0
    total = 0
    for angle in angles:
        ix = int(x + inner_r * np.cos(angle))
        iy = int(y + inner_r * np.sin(angle))
        ox = int(x + outer_r * np.cos(angle))
        oy = int(y + outer_r * np.sin(angle))
        if not (0 <= ix < width and 0 <= iy < height):
            continue
        total += 1
        # Outside the frame counts as background, which is what a coin running
        # off the edge of the photo actually looks like.
        outer = float(gray[oy, ox]) if (0 <= ox < width and 0 <= oy < height) else 255.0
        if abs(float(gray[iy, ix]) - outer) > 18.0:
            hits += 1
    return hits / total if total else 0.0


def fill_ratio(image: np.ndarray, x: int, y: int, r: int, band: int = 12) -> float:
    """Fraction of the proposed disc that differs from the background.

    `edge_support` only asks whether the rim sits on a step, which a circle
    straddling a coin's edge satisfies just as well as one centred on it -- that
    is what produced the lens-shaped crops, half coin and half backdrop.

    This asks the other question: once the circle is placed, is it *full*? The
    background colour is taken from the frame border, so the same test works
    whatever it is. A crop of bare worktop matches the border and scores near
    zero; a coin on that worktop does not and scores near one.
    """
    height, width = image.shape[:2]
    border = np.concatenate([
        image[:band].reshape(-1, 3), image[-band:].reshape(-1, 3),
        image[:, :band].reshape(-1, 3), image[:, -band:].reshape(-1, 3),
    ])
    background = np.median(border, axis=0)

    x0, x1 = max(0, x - r), min(width, x + r)
    y0, y1 = max(0, y - r), min(height, y + r)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    patch = image[y0:y1, x0:x1].astype(np.float32)

    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
    if inside.sum() < 64:
        return 0.0

    distance = np.abs(patch - background).sum(axis=2)
    return float((distance[inside] > 60).mean())


def process(path: Path, out_dir: Path, size: int, padding: float,
            min_support: float, min_fill: float,
            detect_size: int = 900, rim_slack: float = 1.12) -> list[dict]:
    image = cv2.imread(str(path))
    if image is None:
        return [{"file": path.name, "status": "unreadable", "method": "", "n": 0}]

    # Detection runs on a downscaled copy; the coin is a large object and the
    # extra pixels only cost time. Coordinates are scaled back up to cut from
    # the original, so nothing is lost from the output.
    scale = float(detect_size) / max(image.shape[:2])
    small = cv2.resize(image, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA) if scale < 1 else image.copy()
    factor = image.shape[1] / small.shape[1]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    if border_is_plain(small):
        method = "threshold"
        circles = circles_by_threshold(gray)
        if not circles:
            method, circles = "hough", circles_by_hough(gray)
    else:
        method = "hough"
        circles = circles_by_hough(gray)
        if not circles:
            method, circles = "threshold", circles_by_threshold(gray)

    if not circles:
        return [{"file": path.name, "status": "no coin found", "method": method,
                 "n": 0}]

    # Rejected here rather than cropped and sorted out by eye later. A crop of a
    # worktop looks like a coin at thumbnail size, and the whole point of this
    # step is that the contact sheet can be trusted.
    scored = [(c, edge_support(gray, *c), fill_ratio(small, *c))
              for c in dedupe(circles)]
    kept = [c for c, support, fill in scored
            if support >= min_support and fill >= min_fill][:MAX_COINS]
    if not kept:
        best_support = max((s for _, s, _ in scored), default=0.0)
        best_fill = max((f for _, _, f in scored), default=0.0)
        return [{"file": path.name,
                 "status": f"rejected (rim {best_support:.2f}, fill {best_fill:.2f})",
                 "method": method, "n": 0}]
    circles = kept
    rows = []
    for index, (x, y, r) in enumerate(circles):
        X, Y, R = int(x * factor), int(y * factor), int(r * factor)
        try:
            crop = cut_out(image, X, Y, R, size, padding, rim_slack)
        except Exception as exc:  # noqa: BLE001
            rows.append({"file": path.name, "status": f"crop failed: {exc}",
                         "method": method, "n": 0})
            continue
        suffix = "" if len(circles) == 1 else f"_{'ab'[index]}"
        target = out_dir / f"{path.stem}{suffix}.png"
        # imwrite signals failure by returning False rather than raising, so an
        # unchecked call lets the report claim a crop that is not on disk.
        if not cv2.imwrite(str(target), crop):
            rows.append({"file": target.name, "status": "write failed",
                         "method": method, "n": 0})
            continue
        rows.append({"file": target.name, "status": "ok", "method": method,
                     "n": len(circles)})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path,
                        help="folder containing images/, e.g. data/errors_ref/ebay_de")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--padding", type=float, default=0.18,
                        help="white margin around the coin, as a fraction")
    parser.add_argument("--rim-slack", type=float, default=1.12,
                        help="mask radius as a multiple of the detected one")
    parser.add_argument("--min-support", type=float, default=0.65,
                        help="fraction of the rim that must sit on an edge")
    parser.add_argument("--min-fill", type=float, default=0.85,
                        help="fraction of the disc that must be coin, not backdrop")
    parser.add_argument("--detect-size", type=int, default=640,
                        help="long edge used for detection; lower is faster")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    source = args.root / "images"
    if not source.is_dir():
        print(f"no images/ under {args.root}", file=sys.stderr)
        return 1
    out_dir = args.root / "normalised"
    out_dir.mkdir(exist_ok=True)

    paths = sorted(p for p in source.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit:
        paths = paths[:args.limit]

    report = args.root / "normalise_report.csv"

    def save_report(rows: list[dict]) -> None:
        with report.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh,
                                    fieldnames=["file", "status", "method", "n"])
            writer.writeheader()
            writer.writerows(rows)

    rows: list[dict] = []
    for index, path in enumerate(paths, 1):
        # One unreadable or pathological photo must not cost the run. OpenCV
        # raises rather than returning on some inputs, and a fifteen-minute
        # sweep that dies on image 300 and writes nothing is the worst outcome
        # available.
        try:
            rows.extend(process(path, out_dir, args.size, args.padding,
                                args.min_support, args.min_fill,
                                args.detect_size, args.rim_slack))
        except Exception as exc:  # noqa: BLE001
            rows.append({"file": path.name, "status": f"error: {exc}",
                         "method": "", "n": 0})
            print(f"  skipped {path.name}: {exc}", file=sys.stderr)
        if index % 50 == 0:
            save_report(rows)
            print(f"  {index}/{len(paths)}", file=sys.stderr)

    save_report(rows)

    from collections import Counter
    ok = [r for r in rows if r["status"] == "ok"]
    print(f"\n{len(paths)} photos -> {len(ok)} crops", file=sys.stderr)
    print(f"  failed: {len(paths) - len({r['file'].split('_a')[0] for r in ok})}",
          file=sys.stderr)
    for method, count in Counter(r["method"] for r in ok).most_common():
        print(f"  {count:4d}  via {method}", file=sys.stderr)
    for status, count in Counter(
            r["status"] for r in rows if r["status"] != "ok").most_common():
        print(f"  {count:4d}  {status}", file=sys.stderr)
    print(f"\nreport: {report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
