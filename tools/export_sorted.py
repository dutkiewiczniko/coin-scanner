"""Copy scraped photos into nested folders by any of their scored properties.

The scrape's own folders group by what the forum said about the coin. Once a
photo has also been scored for camera angle, or reviewed by hand, it is
useful to see those groupings on disk -- to flick through a folder in an
image viewer rather than one image at a time in the review tool.

Grouping keys nest left to right, so the same photos can be arranged
whichever way the question needs:

    --by angle,verdict    square-on/confirmed/...   "of my square-on shots,
                                                     which are real errors?"
    --by verdict,angle    confirmed/square-on/...   "of my real errors,
                                                     which are usable?"
    --by label            off_centre/, die_crack/   the shape a training
                                                    set is usually read in

Copies, never moves. `review.csv` and every manifest refer to photos by their
path under `images/`, so relocating an original would break the record of
what was decided about it. Disk is the cheaper thing to spend, and the whole
tree is rebuilt on each run because it is a derived view, not a record.

Filenames are prefixed with the axis ratio where one was measured, so every
folder is in roundest-first order in any file browser, and they keep the
original bucket and label so a photo can always be traced back:

    0.984__confirmed__die_crack__812608_69102.jpg
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

#: key -> the folder a row belongs in, or None to leave it out entirely.
GROUPERS = {
    # Not measurable is not a verdict: a cropped coin has no outline, so its
    # angle is unknown rather than bad, and it is kept as its own folder
    # instead of being silently dropped or lumped in with the oblique ones.
    "angle": lambda row, marks: (row["round_verdict"] if row.get("round_ok")
                                 else "angle-unknown"),
    "verdict": lambda row, marks: row.get("verdict"),
    "label": lambda row, marks: row.get("label") or "unsorted",
    "photo": lambda row, marks: row.get("photo_verdict"),
    # Only photos actually reviewed; the rest are not in this grouping.
    "decision": lambda row, marks: marks.get(row["file"]),
}


def load_review(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["file"]: row["decision"] for row in csv.DictReader(fh)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("data/errors_ref/emuenzen"))
    parser.add_argument("--by", default="angle,verdict",
                        help="comma-separated, nested left to right: "
                             + ", ".join(GROUPERS))
    parser.add_argument("--out", type=Path, default=None,
                        help="defaults to <root>/by_<keys>")
    parser.add_argument("--measured-only", action="store_true",
                        help="skip photos whose angle could not be measured")
    args = parser.parse_args()

    keys = [k.strip() for k in args.by.split(",") if k.strip()]
    unknown = [k for k in keys if k not in GROUPERS]
    if unknown:
        print(f"unknown grouping key(s): {', '.join(unknown)}", file=sys.stderr)
        return 1

    out_root = args.out or args.root / ("by_" + "_".join(keys))
    manifest = args.root / "manifest_judged.json"
    if not manifest.exists():
        manifest = args.root / "manifest.json"
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    marks = load_review(args.root / "review.csv")

    if out_root.exists():
        shutil.rmtree(out_root)

    counts: Counter = Counter()
    seen: set[str] = set()
    for row in rows:
        if row["file"] in seen:
            continue
        seen.add(row["file"])
        if args.measured_only and not row.get("round_ok"):
            counts["(skipped: angle unmeasurable)"] += 1
            continue

        parts = [GROUPERS[key](row, marks) for key in keys]
        if any(part is None for part in parts):
            counts["(not in this grouping)"] += 1
            continue

        source = args.root / "images" / row["file"]
        if not source.exists():
            continue
        ratio = row.get("round_axis_ratio")
        prefix = f"{ratio:.3f}" if ratio else "-----"
        origin = Path(row["file"]).parent.name
        target = out_root.joinpath(*parts) / f"{prefix}__{origin}__{source.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        counts["/".join(parts)] += 1

    print(f"copied into {out_root}", file=sys.stderr)
    for folder, count in sorted(counts.items()):
        print(f"  {count:4d}  {folder}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
