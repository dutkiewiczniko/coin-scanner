"""Second opinion on the scraped forum coins, from a vision model.

The scraper's verdicts are keyword matches over German forum text. That pass
is precise where the thread speaks in formulas ("Münzrollmaschine",
"Dezentrierung") and blind everywhere else -- half the haul landed in
`uncertain` because the verdict was phrased as "Ja." or "verschmutzter
Prägestempel". A model that reads the discussion *and looks at the coin*
closes that gap, and also judges what the keywords never could: whether the
photograph itself is usable (in focus, coin fills the frame, not a shot of an
album page).

One request per post -- a verdict is per coin, and a post's photos are that
coin's obverse/reverse/edge, so they are judged together. Claude Haiku 4.5
with a strict JSON schema; images resized to 1024px. The judge is shown the
forum text but not the scraper's verdict, so the two stay independent and
disagreement means something.

Output is `judgement.jsonl` (append-only, resumable) and
`manifest_judged.json` -- the scraped manifest with `ai_*` columns and a
`verdict` recomputed from both passes:

    scraper + judge agree            -> that verdict
    judge confident, scraper wasn't  -> judge's verdict
    the two contradict               -> uncertain (human review)

`review_images.py` prefers `manifest_judged.json` automatically, so after
this run the review tool shows the merged buckets. Files never move.

Needs ANTHROPIC_API_KEY in .env or the environment. Full run is ~570
requests, roughly $2-3 at Haiku pricing.

    python tools/judge_images.py --limit 5    # taste test
    python tools/judge_images.py
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from PIL import Image

MODEL = "claude-haiku-4-5"

#: Long-edge cap before upload. A striking fault is coarse enough to judge at
#: this size, and it keeps a post's photos to ~1k tokens each.
MAX_EDGE = 1024

#: The scraper's label vocabulary, so the two passes merge cleanly.
ERROR_TYPES = ["off_centre", "rotated_die", "die_crack", "cud", "zainende",
               "double_struck", "wrong_planchet", "missing_pill", "spiegelei",
               "planchet_crack", "plain_edge", "blank_planchet", "weak_strike",
               "grease_filled_die", "lamination", "other", "none"]

SYSTEM = """You judge coins posted to a German coin-collecting forum thread \
about euro striking errors (Fehlprägungen). For each post you receive the \
photos of ONE coin plus the poster's text and the replies from other members.

Decide whether the coin shows a genuine minting error, post-mint damage, or \
nothing notable. Weigh the replies heavily: experienced members' verdicts \
outrank the original poster's hopes. "Münzrollmaschine" or "Rolliermaschine" \
means coin-rolling-machine damage, not an error. A verschmutzter/verfetteter \
Prägestempel (grease-filled die) IS a genuine minting error. Judge the coin \
from the images too -- text and image evidence together.

Also rate each photo's usability as machine-learning training data: usable \
means the coin is reasonably sharp and fills a fair share of the frame. \
Photos of packaging, albums, coin rolls, or where the coin is tiny or badly \
blurred are not usable."""

SCHEMA = {
    "type": "object",
    "properties": {
        "coin_verdict": {
            "type": "string",
            "enum": ["error", "damage", "normal", "unclear"],
            "description": "Is this coin a genuine minting error?",
        },
        "error_type": {"type": "string", "enum": ERROR_TYPES},
        "confidence": {
            "type": "number",
            "description": "0 to 1. Above 0.7 only when the reply evidence "
                           "or the image is unambiguous.",
        },
        "reason": {"type": "string",
                   "description": "One sentence, in English."},
        "images": {
            "type": "array",
            "description": "One entry per photo, in the order given.",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "usable": {"type": "boolean"},
                    "problem": {"type": "string",
                                "description": "Empty string if usable."},
                },
                "required": ["index", "usable", "problem"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["coin_verdict", "error_type", "confidence", "reason",
                 "images"],
    "additionalProperties": False,
}


def load_env(path: Path = Path(".env")) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def encode_image(path: Path) -> str | None:
    """Resized JPEG as base64, or None for files PIL cannot open."""
    try:
        img = Image.open(path)
        img = img.convert("RGB")
        if max(img.size) > MAX_EDGE:
            img.thumbnail((MAX_EDGE, MAX_EDGE))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.standard_b64encode(buf.getvalue()).decode()
    except Exception as exc:  # noqa: BLE001
        print(f"    unreadable {path.name}: {exc}", file=sys.stderr)
        return None


def judge_post(client: anthropic.Anthropic, root: Path, post_id: str,
               rows: list[dict]) -> dict:
    content: list[dict] = []
    files = []
    for row in rows:
        data = encode_image(root / "images" / row["file"])
        if data is None:
            continue
        files.append(row["file"])
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg",
                       "data": data},
        })
    if not files:
        return {"post_id": post_id, "error": "no readable images"}

    first = rows[0]
    content.append({"type": "text", "text": (
        f"Photos 0..{len(files) - 1} of one coin.\n"
        f"Poster (forum post count {first['author_posts']}) wrote:\n"
        f"{first['own_excerpt']}\n\n"
        f"Replies from other members:\n{first['reply_excerpt'] or '(none)'}"
    )})

    response = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        system=SYSTEM,
        messages=[{"role": "user", "content": content}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
    )
    result = json.loads(next(b.text for b in response.content
                             if b.type == "text"))
    usable = {files[img["index"]]: img
              for img in result["images"] if img["index"] < len(files)}
    return {
        "post_id": post_id,
        "ai_verdict": result["coin_verdict"],
        "ai_label": result["error_type"],
        "ai_confidence": result["confidence"],
        "ai_reason": result["reason"],
        "files": {f: {"usable": u["usable"], "problem": u["problem"]}
                  for f, u in usable.items()},
    }


#: scraper verdict -> what the judge would have to say to agree.
AGREES = {"confirmed": "error", "rejected": "damage"}


def final_verdict(scraper: str, judgement: dict | None) -> str:
    """Merge the two passes; contradictions go back to a human."""
    if not judgement or "ai_verdict" not in judgement:
        return scraper
    ai = judgement["ai_verdict"]
    confident = judgement["ai_confidence"] >= 0.7
    if ai == "unclear":
        return scraper if scraper != "uncertain" else "uncertain"
    if scraper == "uncertain":
        if not confident:
            return "uncertain"
        return "confirmed" if ai == "error" else "rejected"
    if AGREES[scraper] == ai or (scraper == "rejected" and ai == "normal"):
        return scraper
    return "uncertain"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("data/errors_ref/emuenzen"))
    parser.add_argument("--limit", type=int, default=0,
                        help="judge at most this many posts (0 = all)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    env = load_env()
    api_key = env.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key) if api_key \
        else anthropic.Anthropic()  # falls back to the environment

    rows = json.loads((args.root / "manifest.json").read_text(encoding="utf-8"))
    by_post: dict[str, list[dict]] = {}
    seen_files: set[str] = set()
    for row in rows:
        if row["file"] not in seen_files:  # resumed scrapes duplicate rows
            seen_files.add(row["file"])
            by_post.setdefault(row["post_id"], []).append(row)

    jsonl = args.root / "judgement.jsonl"
    judged: dict[str, dict] = {}
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            judged[entry["post_id"]] = entry

    todo = [(pid, post_rows) for pid, post_rows in by_post.items()
            if pid not in judged]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(by_post)} posts, {len(judged)} already judged, "
          f"{len(todo)} to go", file=sys.stderr)

    if todo:
        with jsonl.open("a", encoding="utf-8") as log, \
                ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(judge_post, client, args.root, pid, pr): pid
                       for pid, pr in todo}
            done = 0
            for future in as_completed(futures):
                try:
                    entry = future.result()
                except anthropic.APIStatusError as exc:
                    print(f"  {futures[future]}: API {exc.status_code} "
                          f"{exc.message}", file=sys.stderr)
                    continue
                except Exception as exc:  # noqa: BLE001
                    print(f"  {futures[future]}: {exc}", file=sys.stderr)
                    continue
                judged[entry["post_id"]] = entry
                log.write(json.dumps(entry, ensure_ascii=False) + "\n")
                log.flush()
                done += 1
                if done % 25 == 0:
                    print(f"  {done}/{len(todo)} judged", file=sys.stderr)

    # Merge into manifest_judged.json. Every row keeps its scraper verdict in
    # scraper_verdict; `verdict` becomes the merged one the review tool uses.
    merged = []
    moves = 0
    for row in rows:
        j = judged.get(row["post_id"])
        out = dict(row)
        out["scraper_verdict"] = row["verdict"]
        out["verdict"] = final_verdict(row["verdict"], j)
        if j and "ai_verdict" in j:
            out["ai_verdict"] = j["ai_verdict"]
            out["ai_label"] = j["ai_label"]
            out["ai_confidence"] = j["ai_confidence"]
            out["ai_reason"] = j["ai_reason"]
            per_file = j["files"].get(row["file"], {})
            out["ai_usable"] = per_file.get("usable", True)
            out["ai_problem"] = per_file.get("problem", "")
            if out["ai_label"] not in ("none", "other") \
                    and out["label"] == "unsorted":
                out["label"] = out["ai_label"]
        if out["verdict"] != out["scraper_verdict"]:
            moves += 1
        merged.append(out)
    (args.root / "manifest_judged.json").write_text(
        json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")

    from collections import Counter
    print(f"\n{moves} images changed bucket; final:", file=sys.stderr)
    for verdict, count in sorted(Counter(r["verdict"] for r in merged).items()):
        print(f"  {count:4d}  {verdict}", file=sys.stderr)
    unusable = sum(1 for r in merged if r.get("ai_usable") is False)
    print(f"  {unusable:4d}  flagged as unusable photos", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
