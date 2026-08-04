"""Collect photographs of real mint-error coins from eBay listings.

The validation set. A striking-error detector has to be built from *normal*
coins -- real errors are too rare to gather a training set of, and a model
fitted to synthetic ones learns the generator rather than the defect. What these
are for is checking whether a detector built the other way actually fires on the
real thing.

eBay is a better source than the licensed alternatives for one reason: the
seller writes the label. "1998 2 Euro Off Centre Strike Mint Error" names the
fault, the denomination and often the year, so a keyword sweep produces sorted
data instead of a pile needing hand-labelling. Wikimedia Commons has a hundred
times fewer, mostly ancient overstrikes -- see `fetch_error_refs.py`.

Seller identity is discarded at fetch time rather than stored and ignored. The
keyset runs under an exemption from eBay's marketplace account deletion
notifications, which is granted on the basis that no eBay user data is
persisted; keeping a seller username would quietly make that false. Item ids,
titles and images are listing data, not user data.

These images are for local model validation only. They are other people's
photographs, and they are neither redistributed nor committed -- `data/**` is
gitignored, and only the manifest is tracked.

    python tools/fetch_ebay_errors.py --limit 400
    python tools/fetch_ebay_errors.py --euro-only --limit 200
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import shutil
import sys
import time
from pathlib import Path

import requests

TOKEN_URL = "https://api.ebay.com/identity/v1/oauth2/token"
SEARCH_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"
SCOPE = "https://api.ebay.com/oauth/api_scope"

#: Searched across several marketplaces because error terminology and coin stock
#: differ by country -- GB and DE surface Euro errors that the US site does not.
MARKETPLACES = ["EBAY_GB", "EBAY_DE", "EBAY_US"]

#: (query, label). The label is the fault the query is meant to surface; it is
#: confirmed against the listing title afterwards, because sellers pad titles
#: with every error word they know.
QUERIES = [
    ("off centre strike error coin", "off_centre"),
    ("off center strike mint error", "off_centre"),
    ("clipped planchet error coin", "clipped_planchet"),
    ("curved clip error coin", "clipped_planchet"),
    ("broadstrike error coin", "broadstrike"),
    ("broadstruck coin error", "broadstrike"),
    ("double struck mint error coin", "double_struck"),
    ("doubled die error coin", "doubled_die"),
    ("brockage error coin", "brockage"),
    ("cud die break error coin", "cud"),
    ("lamination error coin", "lamination"),
    ("wrong planchet error coin", "wrong_planchet"),
    ("mule error coin", "mule"),
    ("blank planchet coin", "blank_planchet"),
]

#: Run alongside the generic sweep. Euro errors are a small fraction of the
#: market, so they need naming directly or they never surface.
#:
#: Half of these are German because that is where euro errors actually trade.
#: "Fehlprägung" is the whole category; "dezentriert" is an off-centre strike,
#: "Schrötling" and "Rohling" are the blank, and a coin struck on the wrong one
#: is the error worth the most. Searching only in English finds the coins that
#: German sellers listed in English, which is a small and unrepresentative slice.
EURO_QUERIES = [
    ("euro coin mint error", "unsorted"),
    ("2 euro error coin", "unsorted"),
    ("1 euro error coin", "unsorted"),
    ("euro cent error coin", "unsorted"),
    ("2 euro off centre strike", "off_centre"),
    ("euro fehlprägung münze", "unsorted"),
    ("2 euro fehlprägung", "unsorted"),
    ("1 euro fehlprägung", "unsorted"),
    ("cent fehlprägung", "unsorted"),
    ("euro münze dezentriert", "off_centre"),
    ("fehlprägung dezentriert geprägt", "off_centre"),
    ("euro auf falschem rohling", "wrong_planchet"),
    ("fehlprägung schrötling euro", "wrong_planchet"),
    ("kursmünze fehlprägung euro", "unsorted"),
    ("2 euro spiegelei fehlprägung", "unsorted"),
    ("euro münze doppelprägung", "double_struck"),
]

#: Applied to the listing title to confirm or correct the query's label.
TITLE_PATTERNS = [
    (r"off[- ]?cent(er|re)", "off_centre"),
    (r"\bclip(ped)?\b", "clipped_planchet"),
    (r"broadstr(uck|ike)", "broadstrike"),
    (r"double[d]?[- ]?die|\bddo\b|\bddr\b", "doubled_die"),
    (r"double[- ]?struck|\bdouble strike\b", "double_struck"),
    (r"brockage", "brockage"),
    (r"\bcud\b|die break", "cud"),
    (r"lamination|peel", "lamination"),
    (r"wrong planchet|off metal|wrong metal", "wrong_planchet"),
    (r"\bmule\b", "mule"),
    (r"blank planchet|\bblank\b", "blank_planchet"),
    # German. Ordered after the English terms so that in a bilingual title the
    # English wins -- "Fehlprägung" names the category, while "off centre"
    # names the actual fault, and the specific one is the more useful label.
    (r"dezentriert|dezentrierung", "off_centre"),
    (r"doppelpr(ä|a)gung", "double_struck"),
    (r"falsche[mnr]? (rohling|schr(ö|o)tling)", "wrong_planchet"),
    (r"spiegelei", "spiegelei"),
    (r"fehlpr(a|ä)gung", "error_de"),
]

#: Titles that are not a coin with a fault. Sellers use "error" loosely, and a
#: book about errors or a lot of fifty mixed coins is noise in a validation set.
REJECT_PATTERNS = [
    r"\bbook\b", r"\bguide\b", r"\bcatalog(ue)?\b", r"\bposter\b",
    r"\breplica\b", r"\bcopy\b", r"\bfantasy\b", r"\btoken\b",
    r"\blot of\b", r"\bbulk\b", r"\bmixed lot\b", r"\bjob lot\b",
    r"\bholder\b", r"\balbum\b", r"\bcapsule\b", r"\bt[- ]?shirt\b",
    # German equivalents. "Konvolut" is the word for a job lot and is the
    # commonest of these by far on the German site.
    r"\bbuch\b", r"\bkatalog\b", r"\bkopie\b", r"\bnachbildung\b",
    r"\bkonvolut\b", r"\bm(ü|u)nzkapsel", r"\bzubeh(ö|o)r\b",
]

EURO_HINTS = re.compile(r"\beuro?\b|\bcent\b|\bcts?\b|€", re.I)


def load_env(path: Path = Path(".env")) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def get_token(client_id: str, client_secret: str) -> str:
    auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    response = requests.post(
        TOKEN_URL,
        headers={
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={"grant_type": "client_credentials", "scope": SCOPE},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def write_manifest(out: Path, rows: list[dict]) -> None:
    """Write both manifests. Called repeatedly, not just at the end."""
    if not rows:
        return
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "manifest.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def classify(title: str, fallback: str) -> str | None:
    """The fault a listing shows, or None if the title says it is not one.

    The query's own label is only a fallback. A search for clipped planchets
    returns off-centre strikes too, and the title is the better evidence.
    """
    low = title.lower()
    for pattern in REJECT_PATTERNS:
        if re.search(pattern, low):
            return None
    for pattern, label in TITLE_PATTERNS:
        if re.search(pattern, low):
            return label
    return fallback


def full_size(url: str) -> str:
    """Ask eBay's image CDN for the large rendition of a thumbnail.

    Search results carry a small image whose size is encoded in the filename
    (`s-l225.jpg`). The same object is served at higher resolution under a
    larger number, which matters here -- a striking fault is a fine detail and
    225 px of coin is not enough to see one.
    """
    return re.sub(r"/s-l\d+\.(jpg|jpeg|png|webp)", r"/s-l1600.\1", url)


def search(token: str, query: str, marketplace: str, limit: int,
           offset: int = 0) -> list[dict]:
    response = requests.get(
        SEARCH_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "X-EBAY-C-MARKETPLACE-ID": marketplace,
        },
        params={"q": query, "limit": min(limit, 200), "offset": offset,
                "category_ids": "11116"},  # Coins & Paper Money
        timeout=30,
    )
    if response.status_code != 200:
        print(f"    HTTP {response.status_code}: {response.text[:160]}",
              file=sys.stderr)
        return []
    return response.json().get("itemSummaries", []) or []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/errors_ref/ebay"))
    parser.add_argument("--limit", type=int, default=400,
                        help="stop after this many downloaded images")
    parser.add_argument("--per-query", type=int, default=100)
    parser.add_argument("--euro-only", action="store_true",
                        help="only listings whose title mentions euro/cent")
    parser.add_argument("--marketplaces", default=",".join(MARKETPLACES),
                        help="comma-separated, in order, e.g. EBAY_DE,EBAY_GB")
    parser.add_argument("--min-free-mb", type=int, default=500,
                        help="stop before free disk falls below this")
    parser.add_argument("--pause", type=float, default=0.25)
    args = parser.parse_args()
    marketplaces = [m.strip() for m in args.marketplaces.split(",") if m.strip()]

    env = {**load_env()}
    client_id = env.get("EBAY_CLIENT_ID")
    client_secret = env.get("EBAY_CLIENT_SECRET")
    if not client_id or not client_secret:
        print("EBAY_CLIENT_ID / EBAY_CLIENT_SECRET missing from .env",
              file=sys.stderr)
        return 1

    token = get_token(client_id, client_secret)
    print("token acquired\n", file=sys.stderr)

    args.out.mkdir(parents=True, exist_ok=True)
    images_dir = args.out / "images"
    images_dir.mkdir(exist_ok=True)

    queries = EURO_QUERIES if args.euro_only else QUERIES + EURO_QUERIES
    rows: list[dict] = []
    seen_items: set[str] = set()
    seen_images: set[str] = set()
    downloaded = 0
    session = requests.Session()

    for query, fallback in queries:
        if downloaded >= args.limit:
            break
        # Checked once per query rather than per image: statvfs is cheap but not
        # free, and a query is at most a couple of hundred megabytes. A run that
        # fills the disk corrupts its own last downloads and takes the rest of
        # the machine down with it, which is how the first sweep ended.
        free_mb = shutil.disk_usage(args.out).free / (1024 * 1024)
        if free_mb < args.min_free_mb:
            print(f"\nstopping: {free_mb:.0f} MB free, below the "
                  f"{args.min_free_mb} MB floor", file=sys.stderr)
            break
        for marketplace in marketplaces:
            if downloaded >= args.limit:
                break
            print(f"  {marketplace}  {query!r}", file=sys.stderr)
            for offset in (0, 100):
                items = search(token, query, marketplace, args.per_query, offset)
                if not items:
                    break
                for item in items:
                    if downloaded >= args.limit:
                        break
                    item_id = item.get("itemId", "")
                    title = item.get("title", "")
                    if not item_id or item_id in seen_items:
                        continue
                    seen_items.add(item_id)

                    if args.euro_only and not EURO_HINTS.search(title):
                        continue
                    label = classify(title, fallback)
                    if label is None:
                        continue

                    image_url = (item.get("image") or {}).get("imageUrl")
                    if not image_url:
                        continue
                    url = full_size(image_url)
                    # The same stock photo appears across a seller's listings.
                    if url in seen_images:
                        continue
                    seen_images.add(url)

                    name = f"{label}__{re.sub(r'[^A-Za-z0-9]+', '_', item_id)}.jpg"
                    target = images_dir / name
                    if not target.exists():
                        try:
                            blob = session.get(url, timeout=45)
                            blob.raise_for_status()
                            if len(blob.content) < 6000:
                                continue  # a placeholder, not a photograph
                            target.write_bytes(blob.content)
                            downloaded += 1
                            time.sleep(args.pause)
                        except Exception as exc:  # noqa: BLE001
                            print(f"    failed {item_id}: {exc}", file=sys.stderr)
                            continue

                    rows.append({
                        "file": name,
                        "label": label,
                        "title": title,
                        "item_id": item_id,
                        "marketplace": marketplace,
                        "query": query,
                        "euro": bool(EURO_HINTS.search(title)),
                        "image_url": url,
                        # Deliberately no seller field. See the module docstring.
                    })
                    # Checkpointed rather than written once at the end. A sweep
                    # that dies partway -- a full disk, a revoked token -- would
                    # otherwise leave the images on disk with no record of what
                    # they are, and the titles cannot be recovered from a
                    # filename.
                    if downloaded % 25 == 0 and downloaded:
                        write_manifest(args.out, rows)
                        print(f"    {downloaded} images", file=sys.stderr)

    write_manifest(args.out, rows)

    from collections import Counter
    print(f"\n{len(rows)} listings, {downloaded} images downloaded",
          file=sys.stderr)
    for label, count in Counter(r["label"] for r in rows).most_common():
        euro = sum(1 for r in rows if r["label"] == label and r["euro"])
        print(f"  {count:4d}  {label:18s} ({euro} euro)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
