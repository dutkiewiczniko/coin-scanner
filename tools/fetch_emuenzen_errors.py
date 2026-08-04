"""Collect euro mint-error photographs from the emuenzen.de forum megathread.

"Bilder von Euro-Fehlprägungen" (thread 9830, ~400 pages since 2002) is the
largest concentration of euro error photos anywhere, but it is a conversation,
not a catalogue: veterans posting their finest pieces sit next to novices
asking whether a coin-roll-machine ding is worth a fortune. The image alone
cannot tell those apart -- the *replies* can. Nearly every posted coin gets a
verdict from experienced members within a few posts, in formulaic German
("Münzrollmaschine", "Dezentrierung", "keine Fehlprägung"), so this scraper
reads each post together with the replies that follow it and sorts the images
into three buckets:

    confirmed/  replies (or a veteran self-label) name a real striking error
    rejected/   replies call it damage -- kept deliberately: these are hard
                negatives, the "looks weird but is not an error" distribution
                a detector must learn to say no to
    uncertain/  no verdict found; small enough to review by hand

A reply verdict labels the *coin*, and a post usually carries several photos
of it (obverse, reverse, edge), so images are named by post id to keep them
grouped -- never treat the attachments of one post as independent samples.

Pages before ~150 (roughly pre-2015) link to long-dead external image hosts;
attachments from the XenForo era download fine without an account, which is
why --start defaults to 150.

These images are other people's photographs, for local training only. They
are neither redistributed nor committed -- `data/**` is gitignored, and only
the manifest is tracked.

    python tools/fetch_emuenzen_errors.py --start 390 --end 400 --limit 100
    python tools/fetch_emuenzen_errors.py            # pages 150..last, capped
"""

from __future__ import annotations

import argparse
import csv
import html as htmllib
import json
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import requests

BASE = "https://forum.emuenzen.de"
THREAD = "bilder-von-euro-fehlpraegungen.9830"

#: A browser user agent, because the forum serves plain requests a 403.
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

#: (pattern, label). A specific fault named in the post or a reply both
#: confirms the coin and labels it. Labels shared with fetch_ebay_errors.py
#: where the fault is the same, so the buckets merge downstream.
LABEL_PATTERNS = [
    (r"dezentri", "off_centre"),
    (r"stempeldreh", "rotated_die"),
    (r"stempelriss|stempelbruch|riss im stempel", "die_crack"),
    (r"stempelausbruch|materialausbruch", "cud"),
    (r"zainende", "zainende"),
    (r"doppelpr(ä|ae)gung|doppelschlag|doppelt gepr", "double_struck"),
    (r"falsche[mnr]? (rohling|schr(ö|oe)tling)|fremdrohling", "wrong_planchet"),
    (r"ohne pille|pille fehlt|fehlende pille", "missing_pill"),
    (r"spiegelei", "spiegelei"),
    (r"schr(ö|oe)tlingsriss", "planchet_crack"),
    (r"ohne (riffel|r(ä|ae)ndel)|glatte[rn]? rand", "plain_edge"),
    (r"ungepr(ä|ae)gt|rohling gefunden|blanke[rn]? schr(ö|oe)tling",
     "blank_planchet"),
]

#: Generic confirmations carrying no class label. "Glückwunsch" is how the
#: thread congratulates a genuine find; nobody congratulates a rolling ding.
CONFIRM_GENERIC = re.compile(
    r"(sch(ö|oe)ne|echte|eindeutige|klare|tolle|top|nette) fehlpr"
    r"|gl(ü|ue)ckwunsch|sammelw(ü|ue)rdig|sch(ö|oe)nes st(ü|ue)ck", re.I)

#: The standard let-downs. "Münzrollmaschine" (damage from a coin-rolling
#: machine) is the single most common verdict in the whole thread.
REJECT = re.compile(
    r"m(ü|ue)nzroll|rolliermaschine|rollmaschine|rollierger(ä|ae)t"
    r"|keine fehlpr|kein fehler|keine pr(ä|ae)gefehler"
    r"|besch(ä|ae)dig|nachtr(ä|ae)glich|umlaufs(chaden|puren)"
    r"|korro|wertlos|manipul|gequetscht|zerkratzt|verkratzt"
    r"|s(ä|ae)ure|hitze|gewalteinwirkung|schrott|abgenutzt"
    r"|\bpmd\b|post[- ]mint", re.I)

#: A post that asks is not a post that knows. Any of these in the image post
#: itself blocks the veteran self-label path -- the verdict must then come
#: from the replies.
QUESTION = re.compile(
    r"was meint ihr|was sagt ihr|was ist (das|es)|ist (das|es|dies)"
    r"|k(ö|oe)nnt ihr|wei(ß|ss) jemand|frage|lohnt|wert\s*\?|\?\s*$", re.I)

#: Below this many forum posts a self-label is not trusted. Veterans in this
#: thread have thousands; drive-by "is this rare" accounts have single digits.
VETERAN_POSTS = 500

#: How many following posts count as replies to an image post. Verdicts come
#: fast in an active thread; five covers nearly all of them without bleeding
#: into the next member's coin.
REPLY_WINDOW = 5

#: Safety cap on stored post/reply text -- guards against a pasted wall of
#: text or a copy-pasted table, not a real limit on a normal forum post.
EXCERPT_CAP = 4000


@dataclass
class Post:
    post_id: str
    page: int
    author: str
    author_posts: int
    date: str
    text: str
    attachments: list[tuple[str, str]] = field(default_factory=list)  # (id, url)
    quotes: list[str] = field(default_factory=list)  # post ids this one quotes


def fetch(session: requests.Session, url: str) -> str:
    response = session.get(url, headers={"User-Agent": UA}, timeout=30)
    response.raise_for_status()
    response.encoding = "utf-8"
    return response.text


def strip_tags(chunk: str) -> str:
    text = re.sub(r"<[^>]+>", " ", chunk)
    return re.sub(r"\s+", " ", htmllib.unescape(text)).strip()


def parse_page(html: str, page: int) -> list[Post]:
    """The posts on one thread page, in order.

    Quoted earlier posts are cut before anything else is read: a reply that
    quotes the question would otherwise donate the question's attachments and
    keywords to the reply, which is exactly the confusion the reply is
    supposed to resolve.
    """
    posts = []
    chunks = re.split(r'<article class="message message--post', html)[1:]
    for chunk in chunks:
        # A quote names the post it answers (`data-source="post: 12345"`).
        # That id is the reliable link between a coin and its verdict, so it
        # is read before the quote body is thrown away.
        quotes = re.findall(r'data-source="post:\s*(\d+)"', chunk)
        chunk = re.sub(r"<blockquote.*?</blockquote>", " ", chunk, flags=re.S)

        post_id = _first(r'id="js-post-(\d+)"', chunk)
        author = _first(r'data-author="([^"]*)"', chunk)
        date = _first(r'datetime="([^"]+)"', chunk) or ""
        raw_count = _first(r"<dt>Beitr(?:ä|&auml;)ge</dt>\s*<dd>([\d.,]+)</dd>",
                           chunk) or "0"
        author_posts = int(re.sub(r"[.,]", "", raw_count) or 0)

        body = _first(r'class="bbWrapper">(.*?)(?:<footer|</article)', chunk,
                      flags=re.S) or ""
        # The attachment footer sits outside bbWrapper, so scan the whole
        # (de-quoted) chunk. Each attachment appears several times -- thumb,
        # lightbox, footer list -- hence the dict keyed on attachment id.
        seen: dict[str, str] = {}
        for slug, att_id in re.findall(r'href="/attachments/([^"/]+)\.(\d+)/',
                                       chunk):
            seen.setdefault(att_id, f"{BASE}/attachments/{slug}.{att_id}/")

        if post_id:
            posts.append(Post(post_id, page, author or "", author_posts, date,
                              strip_tags(body), sorted(seen.items()),
                              sorted(set(quotes))))
    return posts


def _first(pattern: str, text: str, flags: int = 0) -> str | None:
    match = re.search(pattern, text, flags)
    return match.group(1) if match else None


def find_labels(text: str) -> list[str]:
    low = text.lower()
    return [label for pattern, label in LABEL_PATTERNS if re.search(pattern, low)]


def judge(post: Post, quote_replies: list[Post],
          following: list[Post]) -> dict:
    """Verdict for one image post, from its own text and the replies.

    This is a show-and-tell thread: the posts after a coin are as likely to
    be the next member's coin as a reply about this one. So a reply counts
    only if it quotes this post (XenForo records which post a quote answers),
    or if it sits in the window right after and carries no attachments of its
    own -- a post with photos is a new coin, not a verdict on the last one.

    Replies outrank the poster: a named fault or a congratulation confirms, a
    damage verdict rejects, and on a tie the coin stays uncertain rather than
    guessing. With no reply verdict at all, a veteran naming the fault in
    their own post -- and not asking a question -- is trusted; that is how
    most of the thread's best material was posted.
    """
    own_labels = find_labels(post.text)
    asks = bool(QUESTION.search(post.text))

    replies = list(quote_replies)
    for reply in following:
        if reply.attachments or reply.author == post.author:
            continue
        # A window reply that quotes some other post is part of a different
        # conversation.
        if reply.quotes and post.post_id not in reply.quotes:
            continue
        if reply.post_id not in {r.post_id for r in replies}:
            replies.append(reply)

    reply_text = " || ".join(r.text for r in replies)
    reply_labels = find_labels(reply_text)
    confirm_hits = len(reply_labels) + len(CONFIRM_GENERIC.findall(reply_text))
    reject_hits = len(REJECT.findall(reply_text))

    if reject_hits > confirm_hits and reject_hits > 0:
        verdict = "rejected"
    elif confirm_hits > reject_hits:
        verdict = "confirmed"
    elif own_labels and not asks and post.author_posts >= VETERAN_POSTS:
        verdict = "confirmed"
    else:
        verdict = "uncertain"

    # The poster saw the coin in hand; a reply names a fault through a photo.
    # Where both name one, the poster's wins.
    labels = own_labels or reply_labels
    return {
        "verdict": verdict,
        "label": labels[0] if labels else "unsorted",
        "asks": asks,
        "confirm_hits": confirm_hits,
        "reject_hits": reject_hits,
        # Full text, not an excerpt: an early version cut these at 200/300
        # chars, which sliced off exactly the reply that settles a long
        # thread. EXCERPT_CAP is a guard against a pasted wall of text, not a
        # real limit -- ordinary posts and replies are far short of it.
        "own_excerpt": post.text[:EXCERPT_CAP],
        "reply_excerpt": reply_text[:EXCERPT_CAP],
        # Same text, kept as separate replies so the review UI can show one
        # bullet per reply instead of one run-on paragraph.
        "replies": [{"author": r.author, "text": r.text[:EXCERPT_CAP]}
                   for r in replies],
    }


def sniff_ext(blob: bytes) -> str:
    if blob[:3] == b"\xff\xd8\xff":
        return "jpg"
    if blob[:4] == b"\x89PNG":
        return "png"
    if blob[:4] == b"RIFF":
        return "webp"
    return "jpg"


def write_manifest(out: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "manifest.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def last_page(html: str) -> int:
    numbers = [int(n) for n in re.findall(r"page-(\d+)", html)]
    return max(numbers) if numbers else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("data/errors_ref/emuenzen"))
    parser.add_argument("--start", type=int, default=150,
                        help="first thread page (earlier pages: dead hosts)")
    parser.add_argument("--end", type=int, default=0,
                        help="last thread page, 0 = detect")
    parser.add_argument("--limit", type=int, default=2000,
                        help="stop after this many downloaded images")
    parser.add_argument("--buckets", default="confirmed,rejected,uncertain",
                        help="which verdicts to download images for")
    parser.add_argument("--page-pause", type=float, default=1.5)
    parser.add_argument("--pause", type=float, default=0.7,
                        help="between image downloads")
    parser.add_argument("--min-free-mb", type=int, default=500)
    args = parser.parse_args()
    wanted = {b.strip() for b in args.buckets.split(",") if b.strip()}

    session = requests.Session()
    args.out.mkdir(parents=True, exist_ok=True)

    first_url = f"{BASE}/threads/{THREAD}/page-{args.start}"
    first_html = fetch(session, first_url)
    end = args.end or last_page(first_html)
    print(f"pages {args.start}..{end}", file=sys.stderr)

    # Pass 1: read every page first, download later. The verdict for the last
    # posts of a page lives on the next page, so judging needs the full
    # stream; a quarter-million posts of metadata is nothing in memory.
    posts: list[Post] = []
    for page in range(args.start, end + 1):
        html = first_html if page == args.start else fetch(
            session, f"{BASE}/threads/{THREAD}/page-{page}")
        posts.extend(parse_page(html, page))
        if page % 10 == 0:
            print(f"  page {page}: {len(posts)} posts read", file=sys.stderr)
        time.sleep(args.page_pause)

    # A quote can answer a coin from many pages back; index them all first.
    quoted_by: dict[str, list[Post]] = {}
    for post in posts:
        for quoted_id in post.quotes:
            quoted_by.setdefault(quoted_id, []).append(post)

    # Pass 2: judge and download.
    rows: list[dict] = []
    downloaded = 0
    for i, post in enumerate(posts):
        if not post.attachments:
            continue
        if downloaded >= args.limit:
            break
        free_mb = shutil.disk_usage(args.out).free / (1024 * 1024)
        if free_mb < args.min_free_mb:
            print(f"stopping: {free_mb:.0f} MB free", file=sys.stderr)
            break

        result = judge(post, quoted_by.get(post.post_id, []),
                       posts[i + 1:i + 1 + REPLY_WINDOW])
        verdict = result["verdict"]
        images_dir = args.out / "images" / verdict
        images_dir.mkdir(parents=True, exist_ok=True)

        for att_id, url in post.attachments:
            name_stem = f"{result['label']}__{post.post_id}_{att_id}"
            existing = list(images_dir.glob(f"{name_stem}.*"))
            if existing:
                name = existing[0].name
            elif verdict in wanted:
                try:
                    blob = session.get(url, headers={"User-Agent": UA},
                                       timeout=45)
                    blob.raise_for_status()
                    if len(blob.content) < 6000:
                        continue  # a placeholder, not a photograph
                    name = f"{name_stem}.{sniff_ext(blob.content)}"
                    (images_dir / name).write_bytes(blob.content)
                    downloaded += 1
                    time.sleep(args.pause)
                except Exception as exc:  # noqa: BLE001
                    print(f"    failed {url}: {exc}", file=sys.stderr)
                    continue
            else:
                continue

            rows.append({
                "file": f"{verdict}/{name}",
                "verdict": verdict,
                "label": result["label"],
                "post_id": post.post_id,
                "page": post.page,
                "author_posts": post.author_posts,
                "date": post.date,
                "asks": result["asks"],
                "confirm_hits": result["confirm_hits"],
                "reject_hits": result["reject_hits"],
                "own_excerpt": result["own_excerpt"],
                "reply_excerpt": result["reply_excerpt"],
                "replies": result["replies"],
                "attachment_url": url,
                "post_url": f"{BASE}/threads/{THREAD}/post-{post.post_id}",
            })
        if downloaded and downloaded % 25 == 0:
            write_manifest(args.out, rows)
            print(f"  {downloaded} images", file=sys.stderr)

    write_manifest(args.out, rows)

    image_posts = {r["post_id"] for r in rows}
    print(f"\n{len(posts)} posts read, {len(image_posts)} with images kept, "
          f"{downloaded} images downloaded", file=sys.stderr)
    for (verdict, label), count in sorted(
            Counter((r["verdict"], r["label"]) for r in rows).items()):
        print(f"  {count:4d}  {verdict:10s} {label}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
