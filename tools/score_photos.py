"""Find the coin in each scraped photo, and score the photo on that.

The forum verdicts say what the coin is; nothing so far says whether the
photograph is usable. A confirmed error shot through a coin capsule from
across the desk, or a picture of the whole album page, trains a detector on
nothing. That judgement is the one part of the review a machine can make
without reading German: a usable photo has a coin in it, roughly circular,
big enough to see, and in focus.

So: locate the largest disc, then measure three things about it.

    coverage    the disc's area as a fraction of the frame. Tiny means an
                album page, a coin roll, or a coin lost in the background.
    circularity contour area over enclosing-circle area. A real coin is
                round from above; a low score means the largest blob is
                packaging, a hand, or a caption, not a coin.
    sharpness   variance of the Laplacian *inside the disc*. Measured there
                and not over the frame, because a shallow-focus photo with a
                sharp coin and a soft background is a good photo, and the
                frame-wide number would call it blurry.

None of that may be used to discard a coin, and the reason is the whole
point of the exercise: the errors worth collecting are the ones that are not
round. A clipped planchet, a broadstrike, a badly off-centre strike -- the
entire geometric class this project detects -- fails a circularity test
*because it is the error*. Checked against the scraped set, "no coin-shaped
object" caught a broken half-moon planchet and a macro crop of a die crack
alongside the actual rubbish. A filter that discarded on roundness would
preferentially delete the best training examples in the set.

So roundness only ever routes a photo to a human. Shape cannot discard
either, and that took a second attempt to learn: a wide strip with no disc
in it looked like a safe signature-banner test until three of them turned
out to be edge-on photographs of a coin's rim -- which is what an edge
lettering or plain-edge error has to be photographed as.

What is left that may be discarded unseen is junk identifiable without
looking at the picture at all: a file that will not decode, and a file whose
bytes appear under more than one post. Nobody photographs the same coin
twice to the pixel, so a repeated file is a member's signature image or
avatar riding along in the attachment list.

The tray segmenter in `euro_vision.stages.segment` is not reused here: it is
built for many coins on a calibrated tray, where radius bounds come from
millimetres per pixel. These are single hand-held coins at unknown scale, so
the bounds have to be relative to the frame instead.

Nothing is moved or deleted. Each photo comes out as one of three verdicts,
written into `manifest_judged.json` as `photo_*` columns:

    junk    unreadable, or a signature banner. Safe to discard unseen.
    check   no clean disc found. Could be a dramatic error, a detail crop,
            or rubbish -- a human decides, and `review_images.py` sorts
            these to the front so it happens early.
    ok      a clear, sharp, well-framed coin.

    python tools/score_photos.py                  # score only
    python tools/score_photos.py --auto-discard   # pre-mark `junk` only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

#: Work at this long edge. Big enough to keep a coin's outline honest, small
#: enough that 1,400 photos take a minute.
WORK_EDGE = 700

#: A disc smaller than this fraction of the frame is a coin in a scene rather
#: than a photo of a coin -- an album page, a roll, a coin held at arm's
#: length. Set from the scraped set: real single-coin shots sit above 0.05.
MIN_COVERAGE = 0.030

#: Below this the largest blob is not round enough to be a coin seen from
#: above. Angled shots still pass -- a coin at 45 degrees is an ellipse, and
#: an ellipse's contour still fills ~0.79 of its enclosing circle.
MIN_CIRCULARITY = 0.70

#: Laplacian variance inside the disc. Hand-held forum macro shots are soft;
#: this only catches the genuinely unreadable.
MIN_SHARPNESS = 12.0


def find_disc(image: np.ndarray) -> dict | None:
    """The largest plausible coin-like disc, or None.

    Two detectors, because neither alone covers the range of backgrounds in
    the thread. Hough finds a coin whose edge is a clean gradient -- a coin
    on a plain sheet -- but misses one on a busy or same-toned background.
    The contour pass thresholds instead, which handles those, but is fooled
    by a hand or a capsule; the circularity test is what rejects those.
    """
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    short_edge = min(height, width)

    best: dict | None = None

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=short_edge // 3,
        param1=100, param2=45,
        minRadius=int(short_edge * 0.08),
        maxRadius=int(short_edge * 0.75),
    )
    if circles is not None:
        x, y, r = max(np.round(circles[0]).astype(int), key=lambda c: c[2])
        best = {"x": int(x), "y": int(y), "r": int(r), "circularity": 1.0,
                "how": "hough"}

    # Otsu on a blurred frame: coin against backdrop is a two-class problem
    # nearly everywhere in this set. Which class is the coin is unknown, so
    # both polarities are tried and the rounder winner is kept.
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    for invert in (False, True):
        flags = cv2.THRESH_BINARY + cv2.THRESH_OTSU
        if invert:
            flags = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        _, mask = cv2.threshold(blurred, 0, 255, flags)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < (short_edge * 0.08) ** 2 * np.pi:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if radius > short_edge * 0.75:
                continue  # spans the frame: the background, not a coin
            circularity = area / (np.pi * radius * radius)
            if circularity < MIN_CIRCULARITY:
                continue
            # Prefer the rounder disc; on a tie prefer the bigger one.
            if best is None or best["how"] == "hough" or \
                    (circularity, radius) > (best["circularity"], best["r"]):
                best = {"x": int(cx), "y": int(cy), "r": int(radius),
                        "circularity": float(circularity), "how": "contour"}
    return best


def score(path: Path) -> dict:
    """Raw measurements only. The verdict is `classify`'s job, so that
    changing where the lines fall never means re-running detection."""
    data = np.fromfile(path, dtype=np.uint8)  # handles non-ASCII paths
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return {"readable": False}

    height, width = image.shape[:2]
    full_w, full_h = width, height
    scale = WORK_EDGE / max(height, width)
    if scale < 1:
        image = cv2.resize(image, (int(width * scale), int(height * scale)),
                           interpolation=cv2.INTER_AREA)
        height, width = image.shape[:2]

    disc = find_disc(image)
    if disc is None:
        return {"readable": True, "width": full_w, "height": full_h,
                "disc": False, "photo_coverage": 0.0,
                "photo_circularity": 0.0, "photo_sharpness": 0.0}

    coverage = (np.pi * disc["r"] ** 2) / float(height * width)

    # Sharpness inside the disc only -- a soft background is not a fault.
    mask = np.zeros((height, width), np.uint8)
    cv2.circle(mask, (disc["x"], disc["y"]), int(disc["r"] * 0.8), 255, -1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(lap[mask == 255].var()) if (mask == 255).any() else 0.0

    return {
        "readable": True,
        "width": full_w,
        "height": full_h,
        "disc": True,
        "photo_coverage": round(float(coverage), 4),
        "photo_circularity": round(float(disc["circularity"]), 3),
        "photo_sharpness": round(sharpness, 1),
        "photo_how": disc["how"],
    }


def classify(m: dict, repeated: bool = False) -> tuple[str, str]:
    """(verdict, reason) from the measurements. See the module docstring for
    why neither roundness nor shape may discard."""
    if not m.get("readable", True):
        return "junk", "file will not decode"
    if repeated:
        return "junk", "same file posted under several posts (signature image)"

    if not m.get("disc"):
        return "check", "no clean disc — dramatic error, detail crop, or junk"
    if m.get("photo_coverage", 0) < MIN_COVERAGE:
        return "check", f"coin small in frame ({m['photo_coverage']:.1%})"
    if m.get("photo_circularity", 1) < MIN_CIRCULARITY:
        return "check", f"disc not round ({m['photo_circularity']:.2f})"
    if m.get("photo_sharpness", 999) < MIN_SHARPNESS:
        return "check", f"soft focus ({m['photo_sharpness']:.0f})"
    return "ok", ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("data/errors_ref/emuenzen"))
    parser.add_argument("--auto-discard", action="store_true",
                        help="write discard decisions for failures into review.csv")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    source = args.root / "manifest_judged.json"
    if not source.exists():
        source = args.root / "manifest.json"
    rows = json.loads(source.read_text(encoding="utf-8"))

    cache_path = args.root / "photo_scores.json"
    cache: dict[str, dict] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    todo = [r["file"] for r in rows if r["file"] not in cache]
    todo = sorted(set(todo))
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(rows)} rows, {len(todo)} photos to score", file=sys.stderr)

    started = time.time()
    for i, name in enumerate(todo):
        cache[name] = score(args.root / "images" / name)
        if (i + 1) % 100 == 0:
            cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
            print(f"  {i + 1}/{len(todo)}  ({time.time() - started:.0f}s)",
                  file=sys.stderr)
    cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")

    # Entries written before the verdict split carry photo_ok/photo_why and
    # no dimensions; backfill rather than re-run the detection they came from.
    for name, entry in cache.items():
        if "readable" in entry:
            continue
        entry["readable"] = entry.get("photo_why") != "unreadable file"
        entry["disc"] = bool(entry.get("photo_circularity", 0))
        if entry["readable"]:
            try:
                from PIL import Image  # header read only; no decode
                with Image.open(args.root / "images" / name) as img:
                    entry["width"], entry["height"] = img.size
            except Exception:  # noqa: BLE001
                entry["readable"] = False
        entry.pop("photo_ok", None)
        entry.pop("photo_why", None)
    cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")

    # A file whose bytes appear under more than one post is boilerplate, not
    # a coin. Hashing is cheap next to the detection pass above.
    posts_by_hash: dict[str, set[str]] = {}
    hash_by_file: dict[str, str] = {}
    for row in rows:
        if row["file"] in hash_by_file:
            posts_by_hash[hash_by_file[row["file"]]].add(row["post_id"])
            continue
        path = args.root / "images" / row["file"]
        if not path.exists():
            continue
        digest = hashlib.sha1(path.read_bytes()).hexdigest()
        hash_by_file[row["file"]] = digest
        posts_by_hash.setdefault(digest, set()).add(row["post_id"])

    for row in rows:
        measured = cache.get(row["file"])
        if measured:
            row.update({k: v for k, v in measured.items()
                        if k.startswith("photo_")})
            digest = hash_by_file.get(row["file"], "")
            repeated = len(posts_by_hash.get(digest, ())) > 1
            verdict, why = classify(measured, repeated)
            row["photo_verdict"] = verdict
            row["photo_why"] = why
        row.setdefault("scraper_verdict", row["verdict"])
    (args.root / "manifest_judged.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    scored = [r for r in rows if "photo_verdict" in r]
    failed = [r for r in scored if r["photo_verdict"] == "junk"]
    counts = Counter(r["photo_verdict"] for r in scored)
    print(f"\n{len(scored)} scored:", file=sys.stderr)
    for verdict in ("ok", "check", "junk"):
        print(f"  {counts[verdict]:4d}  {verdict}", file=sys.stderr)
    print("\nreasons a photo needs an eye:", file=sys.stderr)
    for why, count in Counter(
            r["photo_why"].split(" (")[0] for r in scored
            if r["photo_verdict"] == "check").most_common():
        print(f"  {count:4d}  {why}", file=sys.stderr)

    if args.auto_discard and failed:
        review = args.root / "review.csv"
        new_file = not review.exists()
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        with review.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if new_file:
                writer.writerow(["file", "decision", "time"])
            for row in failed:
                writer.writerow([row["file"], "discard", stamp])
        print(f"\n{len(failed)} discard decisions appended to {review}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
