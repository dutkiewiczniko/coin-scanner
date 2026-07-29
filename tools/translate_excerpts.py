"""Add English translations of the forum excerpts to the manifest.

The review tool shows each coin with the poster's German text and the
replies' verdict. For a reviewer who doesn't read German that evidence is
noise, so this pass translates both excerpts (Google Translate's free web
endpoint via deep-translator -- no API key) and stores them as
`own_excerpt_en` / `reply_excerpt_en` in `manifest_judged.json`, which
`review_images.py` already prefers. Individual replies (the `replies` list
fetch_emuenzen_errors.py records per post) each get their own `text_en` too,
so the review UI can show one bullet per reply instead of one joined blob.
The German stays: the error vocabulary (Zainende, Stempeldrehung...) is the
label language and shouldn't vanish.

`manifest.json` (the scraper's own output) is always the source of truth for
the scraped text itself -- `own_excerpt`/`reply_excerpt`/`replies` are
refreshed from it every run before translating, even when
`manifest_judged.json` already exists. Once the judge/photo/roundness passes
have merged their columns onto `manifest_judged.json` those columns are
otherwise left alone; only a rerun of fetch_emuenzen_errors.py (e.g. to
recover text an early version had truncated) changes the text itself.

Every translation is cached in `translations.json` keyed by text hash, so
interrupting and rerunning costs nothing and never re-hits the endpoint.

    python tools/translate_excerpts.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from deep_translator import GoogleTranslator


def key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("data/errors_ref/emuenzen"))
    parser.add_argument("--pause", type=float, default=0.3,
                        help="seconds between requests to the free endpoint")
    args = parser.parse_args()

    source = args.root / "manifest_judged.json"
    if not source.exists():
        source = args.root / "manifest.json"
    rows = json.loads(source.read_text(encoding="utf-8"))

    if source.name == "manifest_judged.json":
        scraped = json.loads(
            (args.root / "manifest.json").read_text(encoding="utf-8"))
        by_file = {r["file"]: r for r in scraped}
        for row in rows:
            fresh = by_file.get(row["file"])
            if fresh:
                row["own_excerpt"] = fresh.get("own_excerpt", "")
                row["reply_excerpt"] = fresh.get("reply_excerpt", "")
                row["replies"] = fresh.get("replies", [])

    cache_path = args.root / "translations.json"
    cache: dict[str, str] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    texts = {t for row in rows
             for t in (row.get("own_excerpt", ""), row.get("reply_excerpt", ""),
                      *(r.get("text", "") for r in row.get("replies", [])))
             if t.strip()}
    todo = sorted(t for t in texts if key(t) not in cache)
    print(f"{len(texts)} unique excerpts, {len(texts) - len(todo)} cached, "
          f"{len(todo)} to translate", file=sys.stderr)

    translator = GoogleTranslator(source="de", target="en")
    for i, text in enumerate(todo):
        # The excerpts are truncated mid-sentence; the translator copes.
        try:
            cache[key(text)] = translator.translate(text) or ""
        except Exception as exc:  # noqa: BLE001
            print(f"  failed ({exc}); retrying once in 5s", file=sys.stderr)
            time.sleep(5)
            try:
                cache[key(text)] = translator.translate(text) or ""
            except Exception as exc2:  # noqa: BLE001
                print(f"  giving up on one excerpt: {exc2}", file=sys.stderr)
                continue
        if (i + 1) % 25 == 0:
            cache_path.write_text(json.dumps(cache, ensure_ascii=False),
                                  encoding="utf-8")
            print(f"  {i + 1}/{len(todo)}", file=sys.stderr)
        time.sleep(args.pause)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False),
                          encoding="utf-8")

    for row in rows:
        row["own_excerpt_en"] = cache.get(key(row.get("own_excerpt", "")), "")
        row["reply_excerpt_en"] = cache.get(key(row.get("reply_excerpt", "")), "")
        # Per-reply translations too, so the review UI can show one bullet
        # per reply in English instead of relying on the joined blob.
        for reply in row.get("replies", []):
            reply["text_en"] = cache.get(key(reply.get("text", "")), "")
        row.setdefault("scraper_verdict", row["verdict"])
    (args.root / "manifest_judged.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"translations written into "
          f"{args.root / 'manifest_judged.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
