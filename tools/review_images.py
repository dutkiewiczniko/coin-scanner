"""Keyboard-driven triage of scraped error-coin images.

The forum verdicts are text verdicts: they say what the coin is, not whether
the photograph is usable. A confirmed coin can arrive as a blurry hand-held
shot of a coin album, and a rejected post sometimes carries a perfectly good
error photo of a *different* coin. Only a human eye sorts that, and at ~1,400
images the eye needs to be the bottleneck, not the tooling.

So: one image on screen, one keypress to file it, auto-advance.

    k  keep        j  discard        l  unsure
    K / J / L      the same, applied to every image of the same post
                   (a verdict is per coin; its photos usually stand or
                   fall together)
    ← / →          walk back to fix a mistake
    g              open the forum post to see the discussion

The uncertain bucket comes first: those are the coins the scraper could not
call, so they are the ones a human eye actually adds information to.

Decisions append to review.csv next to the manifest -- one row per keypress,
last row per file wins -- so a session can be killed and resumed at any time
and nothing is ever overwritten. Images are never moved or deleted: the
training loader joins manifest.json with review.csv and takes keep-rows.

    python tools/review_images.py                       # everything unreviewed
    python tools/review_images.py --buckets confirmed   # one bucket at a time
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import webbrowser
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>coin review</title><style>
  body { margin:0; background:#111; color:#ddd; font:14px system-ui, sans-serif;
         display:flex; flex-direction:column; height:100vh; }
  #bar { padding:8px 14px; background:#1c1c1c; display:flex; gap:18px;
         align-items:baseline; flex-wrap:wrap; }
  #bar b { color:#fff; }
  .verdict-confirmed { color:#7c6; } .verdict-rejected { color:#e77; }
  .verdict-uncertain { color:#fb5; }
  #stage { flex:1; display:flex; align-items:center; justify-content:center;
           min-height:0; }
  #img { max-width:98vw; max-height:100%; object-fit:contain; }
  #ctx { padding:8px 14px; background:#1c1c1c; font-size:13px; color:#aaa;
         max-height:38vh; overflow-y:auto; }
  .en { color:#cdc; }
  .de { color:#777; font-size:12px; }
  #ctx div { margin:2px 0; }
  #ctx h4 { margin:10px 0 2px; color:#888; font-size:11px;
            text-transform:uppercase; letter-spacing:.04em; }
  #ctx h4:first-child { margin-top:2px; }
  #replies { margin:0; padding-left:1.2em; }
  #replies li { margin:3px 0; }
  #keys { padding:6px 14px; background:#000; color:#888; font-size:12px; }
  kbd { background:#333; color:#eee; border-radius:3px; padding:1px 5px; }
  .mark { font-weight:bold; }
  .mark-keep { color:#7c6; } .mark-discard { color:#e77; }
  .mark-unsure { color:#fb5; }
  .photo-ok { color:#5a5; } .photo-check { color:#fb5; }
  .photo-junk { color:#e77; }
  #multi { color:#000; background:#fb5; padding:1px 7px; border-radius:3px;
           font-weight:bold; }
  .round-square-on { color:#5a5; } .round-slight-angle { color:#fb5; }
  .round-oblique { color:#e77; } .round-unknown { color:#777; }
</style></head><body>
<div id="bar">
  <span id="progress"></span><span id="verdict"></span><b id="label"></b>
  <span id="post"></span><span id="photo"></span><span id="round"></span>
  <span id="multi"></span><span id="decision"></span>
</div>
<div id="stage"><img id="img"></div>
<div id="ctx"><div id="ai"></div>
<h4>Poster</h4>
<div id="own_en" class="en"></div><div id="own" class="de"></div>
<h4 id="replies_h">Replies</h4>
<ul id="replies"></ul>
</div>
<div id="keys"><kbd>k</kbd> keep &nbsp; <kbd>j</kbd> discard &nbsp;
  <kbd>l</kbd> unsure &nbsp; <kbd>shift</kbd>+key = whole post &nbsp;
  <kbd>&larr;</kbd><kbd>&rarr;</kbd> move &nbsp; <kbd>g</kbd> open post</div>
<script>
let items = [], marks = {}, pos = 0;

function firstUnreviewed() {
  const i = items.findIndex(it => !marks[it.file]);
  return i < 0 ? items.length - 1 : i;
}
function show() {
  const it = items[pos];
  if (!it) return;
  document.getElementById("img").src = "/img/" + it.file;
  const next = items[pos + 1];
  if (next) (new Image()).src = "/img/" + next.file;
  document.getElementById("progress").textContent =
    (pos + 1) + " / " + items.length +
    "  (" + Object.keys(marks).length + " reviewed)";
  const v = document.getElementById("verdict");
  v.textContent = it.verdict; v.className = "verdict-" + it.verdict;
  document.getElementById("label").textContent = it.label;
  document.getElementById("post").textContent =
    "post " + it.post_id + " · author posts: " + it.author_posts +
    " · confirm words: " + it.confirm_hits +
    " · damage words: " + it.reject_hits;
  // Photo quality is a machine opinion about the photograph, never about
  // the coin: "check" most often means the coin isn't round, which is as
  // likely to be a dramatic error as it is to be junk.
  const p = document.getElementById("photo");
  if (it.photo_verdict) {
    p.textContent = "photo: " + it.photo_verdict +
      (it.photo_why ? " (" + it.photo_why + ")" : "");
    p.className = "photo-" + it.photo_verdict;
  } else p.textContent = "";
  // How square-on the camera was, from the coin's outline. Only meaningful
  // when the whole coin is in frame, so most photos read "angle unknown".
  const rd = document.getElementById("round");
  const verdict = it.round_ok ? it.round_verdict : "unknown";
  rd.className = "round-" + verdict;
  rd.textContent = it.round_ok
    ? "angle: " + verdict + " (" + it.round_axis_ratio + " ≈ " +
      it.round_tilt_deg + "°)"
    : "angle: unknown";
  // Three or more photos in a post usually means several different coins,
  // which the thread judges separately — so shift would stamp one verdict
  // across all of them, and the post's own label may describe a different
  // coin than the one on screen.
  const multi = document.getElementById("multi");
  if (it.post_images >= 3) {
    multi.textContent = it.post_images + " photos in this post — likely " +
      "several coins; judge individually, shift hits all " + it.post_images;
    multi.style.display = "";
  } else multi.style.display = "none";
  const d = document.getElementById("decision");
  const m = marks[it.file];
  d.textContent = m ? "→ " + m : "";
  d.className = m ? "mark mark-" + m : "";
  const ai = document.getElementById("ai");
  if (it.ai_verdict) {
    const usable = it.ai_usable === false ? " · PHOTO UNUSABLE: " + it.ai_problem : "";
    ai.textContent = "AI: " + it.ai_verdict + " (" + it.ai_label + ", " +
      it.ai_confidence + ") — " + it.ai_reason + usable;
  } else ai.textContent = "";
  document.getElementById("own_en").textContent = it.own_excerpt_en || "";
  document.getElementById("own").textContent =
    "(de) " + (it.own_excerpt || "");
  renderReplies(it);
}
// One bullet per reply, English first with the German underneath so a
// reviewer who doesn't read German never has to piece a run-on blob back
// together. `replies` (a real list, one entry per reply) is what a
// freshly-scraped post carries; older manifest rows only have the replies
// joined into one string, so those are split back apart on the same " || "
// the scraper used to join them.
function renderReplies(it) {
  const ul = document.getElementById("replies");
  ul.textContent = "";
  let entries = (it.replies || []).map(r => ({
    en: r.text_en || "", de: r.text || "",
  }));
  if (!entries.length && (it.reply_excerpt || it.reply_excerpt_en)) {
    const des = (it.reply_excerpt || "").split(" || ");
    const ens = (it.reply_excerpt_en || "").split(" || ");
    entries = des.map((de, i) => ({ de, en: ens[i] || "" }))
      .filter(e => e.de.trim());
  }
  document.getElementById("replies_h").style.display =
    entries.length ? "" : "none";
  if (!entries.length) {
    const li = document.createElement("li");
    li.className = "de"; li.textContent = "(none)";
    ul.appendChild(li);
    return;
  }
  for (const e of entries) {
    const li = document.createElement("li");
    if (e.en) {
      const en = document.createElement("div");
      en.className = "en"; en.textContent = e.en;
      li.appendChild(en);
    }
    const de = document.createElement("div");
    de.className = "de"; de.textContent = (e.en ? "(de) " : "") + e.de;
    li.appendChild(de);
    ul.appendChild(li);
  }
}
async function mark(decision, wholePost) {
  const it = items[pos];
  const targets = wholePost
    ? items.filter(x => x.post_id === it.post_id) : [it];
  for (const t of targets) marks[t.file] = decision;
  await fetch("/api/mark", { method:"POST",
    body: JSON.stringify(targets.map(t => ({file:t.file, decision}))) });
  // jump past everything just marked
  let p = pos + 1;
  while (p < items.length && marks[items[p].file]) p++;
  pos = Math.min(p, items.length - 1);
  show();
}
document.addEventListener("keydown", e => {
  const k = e.key.toLowerCase();
  if (k === "arrowright") { pos = Math.min(pos + 1, items.length - 1); show(); }
  else if (k === "arrowleft") { pos = Math.max(pos - 1, 0); show(); }
  else if (k === "k") mark("keep", e.shiftKey);
  else if (k === "j") mark("discard", e.shiftKey);
  else if (k === "l") mark("unsure", e.shiftKey);
  else if (k === "g") window.open(items[pos].post_url, "_blank");
});
fetch("/api/items").then(r => r.json()).then(data => {
  items = data.items; marks = data.marks; pos = firstUnreviewed(); show();
});
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    root: Path
    wanted: set[str]
    redo: bool
    review_path: Path

    def log_message(self, *_args) -> None:
        pass

    def _send(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/":
            self._send(PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/items":
            # Rebuilt per request: a translation or judge pass may have
            # rewritten the manifest since the server started, and a browser
            # refresh should pick that up without a restart.
            items = build_items(self.root, self.wanted, self.redo,
                                self.review_path)
            body = json.dumps({"items": items,
                               "marks": load_review(self.review_path)})
            self._send(body.encode(), "application/json")
        elif self.path.startswith("/img/"):
            # The manifest's file field is "<verdict>/<name>"; resolve under
            # images/ only, so the server cannot be walked out of the tree.
            target = (self.root / "images" / self.path[5:]).resolve()
            if target.is_file() and target.is_relative_to(
                    (self.root / "images").resolve()):
                suffix = target.suffix.lstrip(".").replace("jpg", "jpeg")
                self._send(target.read_bytes(), f"image/{suffix}")
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/api/mark":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        decisions = json.loads(self.rfile.read(length))
        new_file = not self.review_path.exists()
        with self.review_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if new_file:
                writer.writerow(["file", "decision", "time"])
            stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
            for d in decisions:
                writer.writerow([d["file"], d["decision"], stamp])
        self._send(b"{}", "application/json")


BUCKET_ORDER = {"uncertain": 0, "confirmed": 1, "rejected": 2}


def load_review(path: Path) -> dict[str, str]:
    """file -> latest decision. Append-only log, last row wins."""
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["file"]: row["decision"] for row in csv.DictReader(fh)}


def build_items(root: Path, wanted: set[str], redo: bool,
                review_path: Path) -> list[dict]:
    """The review queue: one entry per file (a resumed scrape can append
    duplicate rows), the uncertain bucket first (the scraper could not call
    those, so a human eye is worth most there), the images of one post kept
    adjacent, and within a bucket the coins with the strongest evidence
    first -- the easy confirms build calibration for the murky ones at the
    end.

    After a judge or translation pass the merged manifest exists and carries
    the combined verdicts and English text; it wins over the scraper's own.
    """
    manifest = root / "manifest_judged.json"
    if not manifest.exists():
        manifest = root / "manifest.json"
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    marks = load_review(review_path)

    seen: set[str] = set()
    items = []
    for row in rows:
        if row["verdict"] not in wanted or row["file"] in seen:
            continue
        if not redo and row["file"] in marks:
            continue
        seen.add(row["file"])
        items.append(row)
    items.sort(key=lambda r: (BUCKET_ORDER.get(r["verdict"], len(BUCKET_ORDER)),
                              r["verdict"], -int(r["confirm_hits"]),
                              int(r["reject_hits"]), r["post_id"], r["file"]))

    # How many photos the post carries, which decides whether the shift key
    # is safe on it. Two photos are a coin's two faces; five are five coins,
    # and the thread's replies judge those separately ("Nr. 1 und 3 sind
    # schöne Fehlprägungen, Nr. 2 und 5 sind Rohlinge"). The scraper's
    # verdict and label are per post either way, so on a multi-coin post
    # they describe the post, not necessarily the coin on screen.
    per_post = Counter(r["post_id"] for r in items)
    for row in items:
        row["post_images"] = per_post[row["post_id"]]
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("data/errors_ref/emuenzen"))
    parser.add_argument("--buckets", default="confirmed,rejected,uncertain")
    parser.add_argument("--redo", action="store_true",
                        help="include already-reviewed images")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    wanted = {b.strip() for b in args.buckets.split(",") if b.strip()}
    review_path = args.root / "review.csv"
    items = build_items(args.root, wanted, args.redo, review_path)
    if not items:
        print("nothing left to review in", ", ".join(sorted(wanted)))
        return 0

    Handler.root = args.root
    Handler.wanted = wanted
    Handler.redo = args.redo
    Handler.review_path = review_path
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"{len(items)} images to review -- {url}  (Ctrl+C to stop; "
          f"progress is saved on every keypress)")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped;", len(load_review(review_path)), "decisions in",
              review_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
