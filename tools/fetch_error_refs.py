"""Collect reference photographs of real mint-error coins.

These are the validation set, not training data. A detector for striking errors
has to be built from *normal* coins -- real errors are far too rare to gather a
training set of, and a model fitted to synthetic ones learns the generator
rather than the defect. What real error photographs are good for is measuring
whether a detector built the other way actually fires on the real thing.

Source is Wikimedia Commons, walked from `Category:Error coins`. Everything
there carries an explicit licence, which is why it is preferred over the places
with far more material: eBay listings and Reddit posts are both larger and
better targeted, and both forbid scraping in their terms. See the module notes
at the bottom of this file for what those would take.

The images are mostly US coins, and that is fine. An off-centre strike is the
same geometry on a Lincoln cent as on a 2 euro coin -- the blank sat proud of
the die and the design ran off the edge. Denomination-specific validation needs
Euro material, but the first question is whether the detector responds to the
fault at all.

Usage:
    python tools/fetch_error_refs.py --out data/errors_ref
    python tools/fetch_error_refs.py --out data/errors_ref --limit 50
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests

API = "https://commons.wikimedia.org/w/api.php"
ROOT_CATEGORY = "Category:Error coins"

# Wikimedia asks for a descriptive agent with a contact address, and rate-limits
# hard without one. The pause is deliberately generous: this runs once, and a
# 429 costs more time than the delay does.
USER_AGENT = (
    "euro-vision-research/0.1 (personal ML dataset; "
    "https://github.com/nikodut/coin-scanner; dutkiewiczniko@gmail.com)"
)
REQUEST_PAUSE_S = 1.1
MAX_RETRIES = 5

#: Category name -> the error class it evidences. Commons' tree mixes faults of
#: the blank, faults of the strike and faults of the die, which are different
#: things to detect; keeping the source category means they can be separated
#: later without re-fetching.
CATEGORY_LABELS = {
    "brockage": "brockage",
    "flan defect": "planchet",
    "mule": "mule",
    "overstruck": "overstrike",
    "united states mint errors": "us_mint_error",
    "celtic coins with struck errors": "struck_error",
}

# Filename keywords, applied when the category is uninformative. Deliberately
# crude -- this annotates, it does not decide, and anything unmatched keeps the
# category label rather than being dropped.
FILENAME_PATTERNS = [
    (r"off[- ]?cent(er|re)", "off_centre"),
    (r"clip(ped)?", "clipped_planchet"),
    (r"broadstr(uck|ike)", "broadstrike"),
    (r"double[d]?[- ]?(die|strike)", "doubled_die"),
    (r"brockage", "brockage"),
    (r"mule", "mule"),
    (r"lamination", "lamination"),
    (r"cud\b", "cud"),
    (r"planchet", "planchet"),
    (r"overstr(uck|ike)", "overstrike"),
    (r"fehlpr(a|ä)gung", "error_de"),
]


class Commons:
    """A politely paced Commons API client."""

    def __init__(self, pause: float = REQUEST_PAUSE_S):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.pause = pause
        self._last = 0.0

    def get(self, params: dict) -> dict:
        params = {"format": "json", **params}
        for attempt in range(MAX_RETRIES):
            gap = time.monotonic() - self._last
            if gap < self.pause:
                time.sleep(self.pause - gap)
            response = self.session.get(API, params=params, timeout=30)
            self._last = time.monotonic()
            if response.status_code == 200 and response.text.lstrip().startswith("{"):
                return response.json()
            # 429 is the common one. Back off geometrically rather than giving
            # up -- the whole walk is a few hundred requests and losing it to a
            # transient limit means starting over.
            wait = 3.0 * (2 ** attempt)
            print(
                f"    HTTP {response.status_code}, waiting {wait:.0f}s",
                file=sys.stderr,
            )
            time.sleep(wait)
        raise RuntimeError(f"Commons API kept failing: {params}")

    def paged(self, params: dict, key: str):
        """Yield every member of a list query, following continuations."""
        cont: dict = {}
        while True:
            payload = self.get({**params, **cont})
            yield from payload.get("query", {}).get(key, [])
            if "continue" not in payload:
                return
            cont = dict(payload["continue"])


def walk_categories(api: Commons, root: str, max_depth: int = 3) -> dict[str, str]:
    """Every file under `root`, mapped to the category it was found in.

    Breadth is bounded because Commons' category graph is not a tree -- it has
    cycles, and coin categories connect out to general numismatics within a few
    hops, which would drag in the entire coin collection.
    """
    seen: set[str] = set()
    files: dict[str, str] = {}

    def visit(category: str, depth: int) -> None:
        if category in seen or depth > max_depth:
            return
        seen.add(category)
        print(f"  {'  ' * depth}{category}", file=sys.stderr)
        members = api.paged(
            {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": category,
                "cmtype": "subcat|file",
                "cmlimit": 500,
            },
            "categorymembers",
        )
        subcats = []
        for member in members:
            title = member["title"]
            if title.startswith("File:"):
                files.setdefault(title, category)
            elif title.startswith("Category:"):
                subcats.append(title)
        for subcat in subcats:
            visit(subcat, depth + 1)

    visit(root, 0)
    return files


def label_for(title: str, category: str) -> str:
    """Best guess at which fault a file shows, from its category and name."""
    haystack = title.lower()
    for pattern, label in FILENAME_PATTERNS:
        if re.search(pattern, haystack):
            return label
    plain = category.replace("Category:", "").lower()
    for key, label in CATEGORY_LABELS.items():
        if key in plain:
            return label
    return "unsorted"


def fetch_metadata(api: Commons, titles: list[str]) -> dict[str, dict]:
    """Direct URL, licence and author for each file, 50 at a time."""
    out: dict[str, dict] = {}
    for start in range(0, len(titles), 50):
        chunk = titles[start:start + 50]
        payload = api.get(
            {
                "action": "query",
                "titles": "|".join(chunk),
                "prop": "imageinfo",
                "iiprop": "url|extmetadata|size|mime",
            }
        )
        for page in payload.get("query", {}).get("pages", {}).values():
            info = (page.get("imageinfo") or [{}])[0]
            if not info.get("url"):
                continue
            meta = info.get("extmetadata", {})
            out[page["title"]] = {
                "url": info["url"],
                "mime": info.get("mime", ""),
                "width": info.get("width", 0),
                "height": info.get("height", 0),
                "licence": meta.get("LicenseShortName", {}).get("value", "unknown"),
                "artist": re.sub(
                    r"<[^>]+>", "", meta.get("Artist", {}).get("value", "")
                ).strip(),
                "descriptionurl": info.get("descriptionurl", ""),
            }
        print(f"  metadata {min(start + 50, len(titles))}/{len(titles)}", file=sys.stderr)
    return out


def safe_name(title: str) -> str:
    stem = title[len("File:"):]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem)[:120]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/errors_ref"))
    parser.add_argument("--limit", type=int, default=0, help="stop after N images")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument(
        "--min-pixels", type=int, default=200,
        help="skip images smaller than this on the short edge",
    )
    parser.add_argument("--pause", type=float, default=REQUEST_PAUSE_S)
    args = parser.parse_args()

    api = Commons(pause=args.pause)
    args.out.mkdir(parents=True, exist_ok=True)
    images_dir = args.out / "images"
    images_dir.mkdir(exist_ok=True)

    print("Walking categories...", file=sys.stderr)
    files = walk_categories(api, ROOT_CATEGORY, args.max_depth)
    print(f"Found {len(files)} files\n", file=sys.stderr)

    print("Fetching metadata...", file=sys.stderr)
    meta = fetch_metadata(api, list(files))

    rows = []
    downloaded = skipped = 0
    for title, category in sorted(files.items()):
        entry = meta.get(title)
        if not entry:
            continue
        if not entry["mime"].startswith("image/") or entry["mime"] == "image/svg+xml":
            skipped += 1
            continue
        if min(entry["width"], entry["height"]) < args.min_pixels:
            skipped += 1
            continue

        label = label_for(title, category)
        name = f"{label}__{safe_name(title)}"
        target = images_dir / name

        if not target.exists():
            if args.limit and downloaded >= args.limit:
                break
            try:
                gap = time.monotonic() - api._last
                if gap < api.pause:
                    time.sleep(api.pause - gap)
                blob = api.session.get(entry["url"], timeout=60)
                api._last = time.monotonic()
                blob.raise_for_status()
                target.write_bytes(blob.content)
                downloaded += 1
                print(f"  [{downloaded}] {name}", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
                print(f"  FAILED {title}: {exc}", file=sys.stderr)
                continue

        rows.append(
            {
                "file": name,
                "label": label,
                "category": category,
                "licence": entry["licence"],
                "artist": entry["artist"],
                "width": entry["width"],
                "height": entry["height"],
                "source": entry["descriptionurl"],
            }
        )

    manifest = args.out / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])) if rows else None
        if writer:
            writer.writeheader()
            writer.writerows(rows)

    (args.out / "manifest.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    from collections import Counter
    print(f"\n{len(rows)} images ({downloaded} new, {skipped} skipped)", file=sys.stderr)
    for label, count in Counter(r["label"] for r in rows).most_common():
        print(f"  {count:4d}  {label}", file=sys.stderr)
    print(f"\nManifest: {manifest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
