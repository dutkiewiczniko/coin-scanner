"""Pull images off a page with a real browser.

`fetch_error_refs.py` talks to an API and `requests` is the right tool for that.
This is for the other case: a page that assembles its images in JavaScript, so
there is nothing in the initial HTML to parse. Numista's coin galleries, auction
archives and most catalogue sites work that way. Playwright drives an actual
Chromium, so the DOM this reads is the one a person would see.

Two things fall out of using a browser that a plain HTTP client makes awkward.
Images are downloaded through the page's own request context, so cookies, the
Referer header and any anti-hotlinking check see a consistent session. And the
size filter reads `naturalWidth`/`naturalHeight` off the decoded element, which
means thumbnails, spacers and tracking pixels are discarded before anything is
fetched rather than after.

Usage:
    python tools/scrape_images.py --url https://example.com/coins --out data/scraped
    python tools/scrape_images.py --url ... --out ... --min-width 400 --scrolls 10
    python tools/scrape_images.py --urls-file pages.txt --out data/scraped --headed

Output is `<out>/<host>/<hash>.<ext>` plus a `manifest.csv` recording where each
file came from, so a scrape can be audited or re-run against the same sources.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import mimetypes
import sys
import time
import urllib.error
import urllib.request
import urllib.robotparser
from pathlib import Path
from urllib.parse import urljoin, urlparse

from playwright.sync_api import sync_playwright

# A real UA string. Not evasion -- a default automation UA is refused outright by
# a lot of CDNs, and being refused is indistinguishable from the page being empty.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

MAX_RETRIES = 4
RETRY_BASE_S = 3.0

# Collects every <img> plus anything carrying a CSS background image, and picks
# the largest srcset candidate where one is offered. Runs in the page, so the
# dimensions reported are post-layout and post-decode.
COLLECT_JS = """
(rootSelector) => {
  const root = rootSelector ? document.querySelector(rootSelector) : document;
  if (!root) return [];
  const out = [];

  const widest = (srcset) => {
    if (!srcset) return null;
    let best = null, bestW = -1;
    for (const part of srcset.split(',')) {
      const [url, size] = part.trim().split(/\\s+/);
      if (!url) continue;
      const w = size && size.endsWith('w') ? parseInt(size) : 0;
      if (w > bestW) { bestW = w; best = url; }
    }
    return best;
  };

  for (const img of root.querySelectorAll('img')) {
    const url = widest(img.getAttribute('srcset')) || img.currentSrc || img.src;
    if (!url) continue;
    out.push({
      url,
      width: img.naturalWidth || 0,
      height: img.naturalHeight || 0,
      alt: (img.alt || '').slice(0, 200),
    });
  }

  for (const el of root.querySelectorAll('*')) {
    const bg = getComputedStyle(el).backgroundImage;
    if (!bg || bg === 'none') continue;
    const match = bg.match(/url\\(["']?(.*?)["']?\\)/);
    if (!match || match[1].startsWith('data:')) continue;
    const box = el.getBoundingClientRect();
    out.push({
      url: match[1],
      width: Math.round(box.width),
      height: Math.round(box.height),
      alt: (el.getAttribute('aria-label') || '').slice(0, 200),
    });
  }
  return out;
}
"""


def robots_allows(url: str, agent: str = "*") -> bool:
    """Whether the site's robots.txt permits fetching `url`.

    Advisory, not enforcement -- but the failure mode of ignoring it is getting
    the address blocked partway through a long collection, which costs more than
    checking. A site with no robots.txt is treated as permitting; that is what
    the standard says absence means.

    The file is fetched by hand rather than with `RobotFileParser.read()`,
    because that sends the default `Python-urllib` agent and a fair number of
    sites -- Wikimedia among them -- answer it with 403. The parser reads a 403
    as "disallow everything", so the naive call blocks the scrape on sites that
    in fact permit it.
    """
    parts = urlparse(url)
    request = urllib.request.Request(
        f"{parts.scheme}://{parts.netloc}/robots.txt",
        headers={"User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            body = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        # 404 means no rules. A 401/403 on robots.txt itself is the standard's
        # "stay out" signal, and is honoured.
        return exc.code not in (401, 403)
    except Exception:
        return True

    parser = urllib.robotparser.RobotFileParser()
    parser.parse(body.splitlines())
    return parser.can_fetch(agent, url)


def autoscroll(page, rounds: int, pause_ms: int = 700) -> None:
    """Scroll to the bottom repeatedly so lazy-loaded images attach.

    Stops early once the page stops growing, because most galleries exhaust
    themselves well before the requested number of rounds and the remaining
    waits are dead time.
    """
    previous = 0
    for _ in range(rounds):
        page.mouse.wheel(0, 4000)
        page.wait_for_timeout(pause_ms)
        height = page.evaluate("document.body.scrollHeight")
        if height == previous:
            break
        previous = height


def extension_for(url: str, content_type: str | None) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".avif"}:
        return suffix
    if content_type:
        guessed = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if guessed:
            return ".jpg" if guessed == ".jpe" else guessed
    return ".jpg"


def download(page, src: str, pause: float) -> tuple[bytes, str | None] | None:
    """Fetch one image through the page's request context, backing off on 429.

    Going through the page rather than a bare HTTP client means cookies, the
    Referer and any session the render established come along, which is what
    anti-hotlinking checks look at. The retry is here because a gallery is a
    burst of requests against one host: Wikimedia in particular starts refusing
    partway through, and without a back-off the tail of every page is lost.
    """
    if pause:
        time.sleep(pause)
    for attempt in range(MAX_RETRIES):
        try:
            response = page.request.get(src, timeout=30_000)
        except Exception as exc:
            print(f"    fetch failed {src}: {exc}", file=sys.stderr)
            return None

        if response.ok:
            body = response.body()
            # Under a kilobyte is a spacer, a tracking pixel or an error page
            # served with a 200, none of which are worth keeping.
            if len(body) < 1024:
                return None
            return body, response.headers.get("content-type")

        if response.status in (429, 503) and attempt < MAX_RETRIES - 1:
            wait = RETRY_BASE_S * (2 ** attempt)
            print(f"    HTTP {response.status}, waiting {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
            continue

        print(f"    HTTP {response.status} {src}", file=sys.stderr)
        return None
    return None


def scrape(
    page,
    url: str,
    out_dir: Path,
    min_width: int,
    min_height: int,
    selector: str | None,
    scrolls: int,
    limit: int | None,
    pause: float,
    seen: set[str],
) -> list[dict]:
    print(f"  {url}", file=sys.stderr)
    page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    try:
        page.wait_for_load_state("networkidle", timeout=15_000)
    except Exception:
        # Sites with polling or open sockets never go idle. The scroll below
        # gives lazy content its chance regardless, so this is not fatal.
        pass
    if scrolls:
        autoscroll(page, scrolls)

    candidates = page.evaluate(COLLECT_JS, selector)
    rows: list[dict] = []

    for item in candidates:
        if limit is not None and len(rows) >= limit:
            break
        src = urljoin(url, item["url"])
        if src.startswith("data:") or src in seen:
            continue
        if item["width"] < min_width or item["height"] < min_height:
            continue
        seen.add(src)

        fetched = download(page, src, pause)
        if fetched is None:
            continue
        body, content_type = fetched

        # Name by content hash: galleries serve the same image under several
        # URLs, and a re-run should overwrite rather than accumulate copies.
        digest = hashlib.sha1(body).hexdigest()[:16]
        host = urlparse(src).netloc.replace(":", "_")
        target = out_dir / host / f"{digest}{extension_for(src, content_type)}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)

        rows.append(
            {
                "file": str(target.relative_to(out_dir)).replace("\\", "/"),
                "source_url": src,
                "page_url": url,
                "width": item["width"],
                "height": item["height"],
                "alt": item["alt"],
                "bytes": len(body),
            }
        )

    print(f"    kept {len(rows)} of {len(candidates)}", file=sys.stderr)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", action="append", default=[], help="page to scrape (repeatable)")
    ap.add_argument("--urls-file", type=Path, help="file of URLs, one per line")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--min-width", type=int, default=200)
    ap.add_argument("--min-height", type=int, default=200)
    ap.add_argument("--selector", help="CSS selector to scope collection to")
    ap.add_argument("--scrolls", type=int, default=5, help="lazy-load scroll rounds")
    ap.add_argument("--limit", type=int, help="max images per page")
    ap.add_argument(
        "--pause",
        type=float,
        default=0.4,
        help="seconds between image downloads (raise it if you see 429s)",
    )
    ap.add_argument("--headed", action="store_true", help="show the browser")
    ap.add_argument("--ignore-robots", action="store_true")
    args = ap.parse_args()

    urls = list(args.url)
    if args.urls_file:
        urls += [
            line.strip()
            for line in args.urls_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
    if not urls:
        ap.error("give at least one --url or a --urls-file")

    if not args.ignore_robots:
        blocked = [u for u in urls if not robots_allows(u)]
        if blocked:
            print("robots.txt disallows:", file=sys.stderr)
            for u in blocked:
                print(f"  {u}", file=sys.stderr)
            print("re-run with --ignore-robots to proceed anyway", file=sys.stderr)
            urls = [u for u in urls if u not in blocked]
        if not urls:
            return 1

    args.out.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    rows: list[dict] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
        context = browser.new_context(user_agent=USER_AGENT, viewport={"width": 1440, "height": 900})
        page = context.new_page()
        for url in urls:
            try:
                rows += scrape(
                    page,
                    url,
                    args.out,
                    args.min_width,
                    args.min_height,
                    args.selector,
                    args.scrolls,
                    args.limit,
                    args.pause,
                    seen,
                )
            except Exception as exc:
                print(f"    page failed: {exc}", file=sys.stderr)
        browser.close()

    manifest = args.out / "manifest.csv"
    fields = ["file", "source_url", "page_url", "width", "height", "alt", "bytes"]
    existing = []
    if manifest.exists():
        with manifest.open(encoding="utf-8", newline="") as fh:
            existing = [r for r in csv.DictReader(fh) if r["file"] not in {x["file"] for x in rows}]
    with manifest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing + rows)

    print(f"\n{len(rows)} images -> {args.out} ({manifest})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
