# Task: rewrite the emuenzen forum judge to read replies properly

## What this project is

`coin-scanner` (euro-vision) detects striking errors on euro coins from
photographs. To train that, we scraped the emuenzen.de forum megathread
"Bilder von Euro-Fehlprägungen" — 1,404 photos across 572 posts, in
`data/errors_ref/emuenzen/`. The thread is a conversation, not a catalogue:
a member posts a coin, and other members reply saying whether it is a genuine
minting error (Fehlprägung) or just post-mint damage.

Those replies are the only ground truth available. The project owner cannot
reliably identify these errors by eye — that is not a gap to work around, it
is the reason the text matters so much, and it is why every verdict this code
produces has to be auditable back to the sentence that caused it.

## The job

Replace the verdict logic in `tools/fetch_emuenzen_errors.py` (the `judge`
function and the constants it uses) with something that reads **each reply
individually, in German and English, with negation handling**, instead of
counting keyword hits over one joined string.

Everything stays offline: **no Anthropic/OpenAI/paid API calls.** The user has
no API credits and has explicitly declined paid usage. This is regex,
heuristics and plain Python. (`tools/judge_images.py` is a vision-model judge
that is written, tested and deliberately unused for exactly this reason —
leave it alone.)

## The data is ready — verify, don't re-scrape

The scrape and translation passes have already been re-run, so the full text
is on disk. Confirm before starting:

```bash
python -c "
import json
j = json.load(open('data/errors_ref/emuenzen/manifest_judged.json', encoding='utf-8'))
print(len(j), 'rows;', sum(1 for r in j if r.get('replies')), 'with replies[]')
print('reply keys:', sorted(next(r for r in j if r.get('replies'))['replies'][0]))
"
# expect: 1404 rows; 1343 with replies[]; ['author', 'text', 'text_en']
```

Each row now carries:

- `own_excerpt` / `own_excerpt_en` — the poster's own text, up to 4,000 chars
- `reply_excerpt` / `reply_excerpt_en` — all replies joined with `" || "`
  (legacy; prefer `replies` below)
- `replies` — **the list to use**: `[{author, text, text_en}, …]`, median 3
  per post, capped at 5 by `REPLY_WINDOW`
- `post_id`, `author_posts`, `page`, `post_url`, plus the current
  `verdict` / `label` / `confirm_hits` / `reject_hits` / `asks`

61 rows have no `replies` (nobody answered) — handle that, don't assume.

If you do need to re-scrape for any reason, note the trap: the enrichment
scripts read `manifest_judged.json` in preference to `manifest.json`, so a
stale judged manifest shadows a fresh scrape. `translate_excerpts.py` already
refreshes the text fields from `manifest.json` on every run; the judge should
do the same or operate on `manifest.json` directly.

`review.csv` holds the user's manual keep/discard decisions, keyed by image
path. Never write to it and never move an image.

## What the current judge does, and what is wrong with it

It gathers replies (those that quote the post, plus the next few posts that
carry no attachments and aren't by the same author), joins their text, then
counts regex hits: confirm-words (error names like `dezentri`, `stempeldreh`,
plus congratulations like `Glückwunsch`) against damage-words (`münzroll`,
`beschädig`, `keine fehlpr`). More damage-words → `rejected`; more
confirm-words → `confirmed`; a tie or nothing → `uncertain`. One extra path:
a member with ≥500 forum posts who names an error in their own post without
asking a question → `confirmed`.

Measured failure modes. These numbers were audited on the current data —
reproduce them as your regression baseline before changing anything:

1. **Negation is invisible.** 12 of 225 confirmed posts (5%) contain a negated
   error word in the replies. Real case, currently filed as
   `confirmed / double_struck`:
   *"…also keine Doppelprägung, sondern nur eine Doppelsenkung"* — "not a
   double strike, but only a doubled die". Note the correct answer is still
   *an* error, just a different one: negation frequently redirects rather than
   rejects, so "found a negation → reject" would be its own bug.
   Do not over-correct either. *"Auch wenn es keine Münze ist, so ist es doch
   eine Fehlprägung"* negates "Münze", not "Fehlprägung", and is correctly
   `confirmed`. Scope negation to its clause.
2. **Questions count as assertions.** 19 of 225 (8%) have the error word
   inside a question — *"ist das eine Dezentrierung?"* currently scores as a
   confirm. A question is a request for a verdict, not a verdict.
3. **The label is far weaker than the bucket.** The bucket (error vs not) is
   roughly 90–95% right because several replies vote on it. The *class* label
   is taken from whichever keyword matched first and is often wrong: a 2 €
   struck on a 1 € blank — plainly `wrong_planchet` — is currently labelled
   `off_centre`. Improving label precision is a first-class goal, and emitting
   no label is better than emitting a guessed one.
4. **Multi-coin posts get a single verdict.** 157 posts carry 3+ photos (800
   images, 57% of the set) and usually show *several different coins*, which
   the thread judges separately:
   *"Nr. 1 und 3 sind schöne Fehlprägungen, Nr. 2 und 5 sind keine Münzen aber
   Rohlinge, Nr. 4 ist…"*. The current code stamps one verdict across all of
   them. Attachments are stored in post order, so parsing enumerated verdicts
   (`Nr. 1`, `No. 2`, `Bild 3`, `die erste`) and mapping them onto that order
   is the single biggest win available. Where the mapping is uncertain, say so
   in the output instead of guessing.

## What to produce

Per post, and per photo where the text supports it:

- `verdict` — confirmed | rejected | uncertain
- `label` — the error class, or `unsorted` when no class is clearly stated
- `confidence` — 0–1, so downstream work can filter on it
- `evidence` — the specific sentence that decided it, quoted, with which reply
  (author and index) it came from

`evidence` is not optional polish. The user cannot check these calls against
the coin, so a verdict without a traceable sentence is unverifiable by anyone.
`tools/review_images.py` renders the manifest and should be extended to show
it.

Keep the three-bucket output and the existing field names.
`review_images.py`, `score_photos.py`, `measure_roundness.py` and
`export_sorted.py` all read them, and the user is midway through a manual
review pass keyed to those buckets. Verdicts may change; the schema should
stay backward-compatible, and `scraper_verdict` should keep recording what the
scraper originally said.

## German the parser must handle

Confirming: `Dezentrierung` / `dezentriert`, `Stempeldrehung`,
`Stempelriss` / `Stempelbruch`, `Zainende`, `Doppelprägung`, `Doppelschlag`,
`Doppelsenkung` (a *doubled die* — distinct from a double strike),
`auf falschem Rohling` / `Schrötling`, `ohne Pille`, `Spiegelei`,
`Schrötlingsriss`, `Materialausbruch`, `verschmutzter` /
`verfetteter Prägestempel` (grease-filled die — a genuine error),
`Glückwunsch`, `schöne Fehlprägung`, `sammelwürdig`.

Rejecting: `Münzrollmaschine` / `Rolliermaschine` (coin-rolling-machine
damage — the commonest verdict in the thread), `Beschädigung`, `nachträglich`,
`keine Fehlprägung`, `Umlaufspuren`, `Korrosion`, `Rostblasen`, `wertlos`,
`manipuliert`, `gequetscht`, `PMD`.

English is available per reply as `text_en`. Use both: the German is
authoritative for the technical vocabulary, the English is easier to parse for
negation and sentence structure. Machine translation mangles the terms
predictably — `Stempel` becomes "stamp", so "stamp rotation" is a rotated die
and "stamp crack" is a die crack — so never key a label off the English alone.

## How to validate it

You cannot check these calls against the images; nobody on this project can
read the coins reliably. Validate against the text instead:

1. Print a confusion matrix of new verdicts against the current ones, and
   hand-inspect a sample from every cell where they disagree. The
   disagreements are the deliverable, not a nuisance.
2. Re-run the negation and question audits above. Both counts should drop
   substantially without the confirmed bucket collapsing.
3. Show worked examples — post text, each reply, and the sentence chosen as
   evidence — for ~20 posts spanning all three verdicts.
4. State the residual error rate you believe remains and how you measured it.
   An honest "I could not verify X" is worth more than a confident number.

## House style

Read `tools/fetch_emuenzen_errors.py` and `tools/measure_roundness.py` first
and match them. Module docstrings explain *why* the approach is what it is,
including approaches that were tried and abandoned and the measurement that
killed them. Comments state constraints the code cannot show; they never
narrate the next line. Scripts are resumable, cache expensive work to disk,
checkpoint as they go, and never move or delete the user's images.

Per `CLAUDE.md`: add a dated entry to `CHANGELOG.md` for anything not
self-evident from the diff, and write findings up as a topic file in `docs/`
rather than a root-level one-off — check for an existing file on the topic
first.

## Approaches already tried that did not work — don't repeat them

- **Circle/roundness detection to filter images by quality.** It rejected the
  most dramatic errors, because the best errors are precisely the least
  circular: a broken half-moon planchet and macro crops of die cracks both
  failed as "no coin-shaped object". It survives only as a camera-angle
  measure in `measure_roundness.py`.
- **Aspect ratio to spot forum signature banners.** Three of the flagged
  "banners" turned out to be edge-on photographs of a coin's rim — the only
  way to shoot an edge-lettering error. Replaced with file-hash duplication
  across posts, which is safe.
- **Brightness thresholding to segment coins.** It splits bimetallic 1 € and
  2 € coins into inner disc and outer ring, i.e. it fails worst on the
  denominations that matter most. Replaced with a flood fill inward from the
  frame border.
