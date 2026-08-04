# euro-vision

Computer vision pipeline for automated Euro coin scanning and rare coin detection. Segments individual coins from tray images, normalises orientation, classifies denomination, and flags potentially valuable coins against a curated database of rare Euro issues.

---

## Overview

euro-vision is a personal project aimed at automating the process of scanning large volumes of Euro coins to identify rare or valuable specimens. Coins are placed in a fixed 3D-printed tray, photographed, and processed end-to-end by the pipeline — from raw tray image to a flagged list of potentially valuable coins.

---

## Pipeline

```
Tray Image
    │
    ▼
1. Coin Segmentation       — YOLO-based object detection to locate and crop individual coins
    │
    ▼
2. Normalisation           — Orientation correction and size normalisation via OpenCV
    │
    ▼
3. Denomination Classification  — CNN classifier to identify coin value (1c → €2)
    │
    ▼
4. Rare Coin Detection     — Image similarity / classification against curated rare coin database
    │
    ▼
Flagged Results
```

---

## Features

- Automated segmentation of coins from a controlled tray image
- Rotation-invariant normalisation for consistent coin orientation
- Denomination classification across all Euro coin values
- Rare coin detection against a hand-curated database of ~50 high-value Euro issues
- Focus on €1 and €2 coins where rare variants have the most value

---

## Target Rare Coins

The rare coin database focuses on the most valuable Euro coins likely to appear in circulation, including:

- Low-mintage commemorative 2 euro coins (Monaco, Vatican, San Marino)
- Early issue Greek and Cypriot standard coins (2004–2015)
- Error coins and misstrikes
- Selected national commemoratives with mintage under ~200,000

---

## Tech Stack

| Component | Technology |
|---|---|
| Coin detection | YOLOv8 |
| Image processing | OpenCV |
| Classification | PyTorch |
| Database | SQLite + local image store |
| Interface | Python CLI / (planned) web UI |

---

## Getting Started

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use source .venv/bin/activate elsewhere
pip install -e ".[dev]"       # add ".[ml]" for the YOLO/PyTorch backends
pytest
```

Name a batch's two photos, working out which tray each shows from its markers:

```bash
euro-vision ingest IMG_0203.jpeg IMG_0205.jpeg --batch 101
# IMG_0205.jpeg  ->  data/raw/101_a.jpeg
# IMG_0203.jpeg  ->  data/raw/101_b.jpeg
```

Match both faces of a batch into one image per coin:

```bash
euro-vision pair 101
```

Writes `data/out/batch_101/`:

- `coins/101-001.png` … one image per coin, both faces side by side, captioned
  with its id, denomination, diameter and position in the tray. Unmatched coins
  are suffixed `_unpaired`, flagged ones `_flagged`.
- `manifest.csv` / `manifest.json` — the durable record. Ids run in reading
  order (`<batch>-<NNN>`) and every coin carries its millimetre position, so a
  coin can still be identified after the tray is emptied.

Build training data. Two shapes, for two different jobs:

```bash
# Whole-tray images with boxes, for a YOLO coin detector
euro-vision export-dataset data/raw/ -o data/dataset

# Per-coin crops in folder-per-class layout, for a denomination classifier
euro-vision export-dataset data/raw/ --crops -o data/crops
```

Labels come from the pipeline's own measurements, so they need correcting before
training — a model fitted to uncorrected output can only repeat its mistakes.
Crops the pipeline cannot label confidently go to `_unsorted/` rather than being
guessed at; sorting those is the labelling work.

The crop layout is one folder per class, matching Keras `flow_from_directory`
and torchvision `ImageFolder`. Public Euro coin datasets use the same shape, so
they merge by copying class folders in — which matters, because a single batch
yields only a few dozen examples of the rarer denominations.

Check the scale is true against coins of known size:

```bash
euro-vision measure data/raw/101_a.jpeg
```

Scan a tray image:

```bash
euro-vision scan data/raw/tray_01.jpg -v
euro-vision scan data/raw/ --config config/default.yaml
```

Results are written to `data/out/<image-name>/` as `results.json`, `results.csv`,
and per-coin crops.

Manage the rare coin database:

```bash
euro-vision db init
euro-vision db seed data/rare_coins_seed.csv
euro-vision db list --denomination 200
```

### Numista — the catalogue of varieties

> Working notes, gotchas and the current database contents:
> [`docs/numista-findings.md`](docs/numista-findings.md)


Three different things get called a rare coin, and only two can be looked up:

| | What it is | Where it comes from |
|---|---|---|
| **Mintage** | How many of an issue were struck | Wikipedia (`db import-commemoratives`), Numista |
| **Variety** | A design fault that ran through a whole batch — a mule, a wrong die pairing | **Numista only** |
| **Mint error** | Off-centre strike, clipped planchet, broadstrike | Nothing to look up — measured (`rare.backend: errors`) |

A variety is an *issue*, so it gets its own catalogue entry. Germany's 2008
Hamburg mule is N# 327881, KM# A261, mintage 600,000 — sitting beside the
ordinary N# 2994 struck 9,000,000 times, because Stuttgart put the old Europe
map on the reverse and the run was stopped. Wikipedia has none of this; it
lists issues as intended, not as struck.

A mint error belongs to one individual coin and no catalogue can list it. That
half stays where it is, in `errors.py`.

```bash
# Free key (2,000 requests/month): log in, then open
#   https://en.numista.com/api/api_key.php
# That page is a <button> on /api/index.php with no href, so it is not
# reachable by browsing or by search — go straight to the URL.
$env:NUMISTA_API_KEY = "<key>"
euro-vision numista check
euro-vision numista search "2 euro Hamburg mule"
euro-vision numista type 327881
euro-vision numista scarce germany --max-types 60 [--save]
```

**Do not parse the descriptions.** The comment field is what a contributor
happened to type, in whatever language and punctuation — N# 327881 writes
`the 'Old' Europe Map`, which a substring search for "old map" misses by an
apostrophe. The reliable signal is structural: the variety is a *separate type*
with its own mintage, so it shows up as an anomalously low figure among the
types sharing an issuer and year. `numista scarce` keys on that ratio and works
regardless of wording. Keyword matching is used only to annotate a candidate
the mintage rule already found.

It returns a shortlist, not a verdict — a proof-only issue and a short
commemorative run look the same from the mintage alone, and both are worth a
glance anyway.

#### Filling the database, in batches

Cherry-picking, not mirroring. Each row of a target file is a *search*, never a
fact — the mintage always comes back from Numista, so nothing enters the
database on the strength of anyone's memory:

```bash
euro-vision numista fetch data/numista_targets_200.csv   --max-mintage 500000 --save
euro-vision numista fetch data/numista_targets_100.csv   --max-mintage 300000 --save
euro-vision numista fetch data/numista_targets_small.csv --varieties-only    --save
```

Three batches because the three denominations need different rules:

| Batch | Rule | Why |
|---|---|---|
| 2 EUR | mintage ≤ 500,000 | where the low-mintage commemoratives are |
| 1 EUR | mintage ≤ 300,000 | no commemoratives exist; scarcity is micro-state and first-year |
| 1c–50c | `--varieties-only` | a 1c struck 5,000 times is still worth a cent. Only the fault has value |

Two filters do nearly all the work, and both were learned from the data rather
than guessed:

- **Collector issues are excluded.** Not a field — it has to be read out of the
  issue comment. Slovenia's 2007 2 EUR lists 21,250,000 for circulation and
  100,000 as a "BU set", and only the first can ever reach a bank bag. Without
  this the results are swamped: batch 1 dropped 347 collector-only issues to
  keep 47. `--include-sets` turns it off. Plain `BU` is *kept*, because Monaco's
  2007 Grace Kelly — 20,001 struck, the most sought-after euro coin there is —
  is marked exactly that.
- **A variety is kept at any mintage, including unknown.** The Hamburg mule ran
  to 600,000, common by mintage and still the coin worth finding. Italy's 2002
  1c "Error - missing R mintmark" has no figure at all.

Evidence strength is tracked. `variant='error'` requires the *issue* comment to
describe a fault. Finland's N# 95 says "Die error on the 2000 coin" at type
level, which is a claim about one year — reading it as the type's property
labelled nine ordinary years as errors.

**Numista does not have everything.** The most valuable euro coin in existence,
Italy's 2002 1c struck with the 2c Mole Antonelliana reverse, has no entry at
all — with perhaps a hundred known it is a mint error on individual coins, not a
variety with a production run. That is the boundary in the table above, and no
catalogue crosses it.

**Budget and terms.** The free tier is 2,000 requests a month and a sweep costs
one request per type, so `scarce` is capped and prompts before spending.
Responses are cached under `data/cache/numista` (gitignored) so a repeat costs
nothing. Search by image — the endpoint that would identify a coin straight
from a crop — is a paid plan (€100/month floor plus €0.03 a call) and returns
403 on a free key; the client method exists for when that becomes worth it.

Numista's terms of use forbid extracting a substantial part of their database,
with no personal-use exemption, which is why `catalogue.py` sources mintages
from Wikipedia instead. Using the sanctioned API is a different thing from
mirroring the catalogue, and this stays on the right side of the line: bounded
sweeps, caching to avoid re-fetching rather than to accumulate, and only the
handful of coins that come back scarce written locally, each with its N# and
source URL.

### Swappable backends

Every stage runs a backend chosen in `config/default.yaml`, so the pipeline is
runnable end to end before any model is trained:

| Stage | Backends |
|---|---|
| Calibration | off · four coloured corner markers |
| Segmentation | `watershed` (splits touching coins) · `hough` (spaced coins only) · `yolo` (trained weights) |
| Normalisation | rotation `none` · `gradient` |
| Classification | `stub` · `diameter` (physical size) · `cnn` (trained) |
| Rare detection | `none` · `shortlist` · `embedding`\* · `metadata`\* |

\* Not implemented — the matching strategy is still an open decision.

### Calibration

Each tray carries four coloured dots at its corners. Warping that rectangle to a
fixed pixel grid removes perspective and tilt and makes the scale exact and
uniform across the whole tray — a ruler in one spot cannot do that, because a
phone's wide lens renders coins near the frame edge smaller and slightly
elliptical.

Check a photo before committing to a scan:

```bash
euro-vision calibrate data/raw/tray_001_a.jpg -o /tmp/rectified.png
```

It reports the detected corners, how square-on the shot is, and the resulting
scale. Then set `calibrate.enabled: true` and measure `tray_width_mm` /
`tray_height_mm` **centre to centre between the dots**, not across the tray's
outer edge.

With calibration on, `segment.min_radius`, `segment.max_radius`,
`segment.min_distance` and `normalise.pixels_per_mm` are all derived from
millimetres automatically and stop needing to be tuned by hand.

#### Marking the two trays

Coins land face-up at random, so each tray is photographed, then sandwiched with
a second tray and flipped to expose the other side.

1. Mark tray A's four corners with four distinct colours.
2. Sandwich the trays exactly as you will when flipping them.
3. Mark each corner of tray B with the colour of the A corner it is touching.

Tray A's arrangement is free — pick any order round the corners. Tray B's is
then fully determined by step 3; you do not get to choose both.

**The two trays necessarily photograph with opposite winding.** When the trays
are face to face their coin-side surfaces have opposite chirality, so if tray A
reads red → green → blue → magenta clockwise, tray B reads those same colours
anticlockwise. No marking scheme avoids this. Tray B's markers are therefore
read in reverse order, which keeps the rectification orientation-preserving —
mirroring it would flip every coin design and make dates read backwards.

Because both frames are defined by the same physical corner points, a coin keeps
consistent coordinates across the pair. Tray B's frame comes out flipped
vertically relative to tray A's, and `Calibration.to_reference_frame` undoes
that, so paired coins land in the same place.

The pipeline reads the side from the `_a` / `_b` filename suffix, and errors out
if a photo's winding does not match the side it was given — that catches a photo
filed under the wrong name, and a tray marked without the touch rule.

Corner markers are chosen as the four blobs enclosing the **largest**
quadrilateral, not the largest blobs. That handles impostors *inside* the tray —
a coin matching a marker's hue is enclosed by the real corners, so it loses even
though it dwarfs a dot.

**It does not handle impostors outside the tray, and cannot.** Anything beyond
the true corners enlarges the quadrilateral and wins regardless of size; a
six-pixel speck is enough. Measured on real photos, yellow markers on an oak
table lost to wood grain. So the requirement is on the colour, not the code:

- **Use colours absent from the background.** Magenta and cyan matched 0.00% of
  real test photos; blue and green 0.01–0.08%. Yellow matched 0.30% (bare wood,
  and Nordic gold coins) and red overlaps copper-plated 1c/2c/5c.
- **Or remove the background problem**: shoot on plain matte neutral card rather
  than a wooden surface. This also steadies the white balance.
- Put the corner dots **outside the coin area** so no coin can cover one.
- Any other marking — a batch number or a tray identifier — must stay **inside**
  the corner rectangle, or use black or white, which are outside the palette.

Marker height above the coin plane is not worth worrying about: the markers only
need to be roughly coplanar with the coins. A 1 mm offset on a 200 mm tray shot
from 400 mm costs about 0.25% of scale, or 0.06 mm on a 2 euro coin. Anything
left over is absorbed by `calibrate.scale_correction` (see `euro-vision
measure`).

Name paired photos `tray_001_a.jpg` / `tray_001_b.jpg` — the suffix is how the
pipeline knows which winding to expect. Photograph **one tray per frame**; two
trays in one shot puts two dots of each colour in view and calibration will
reject it as ambiguous.

### Photography

- **No flash.** A phone's flash is a point source next to the lens; on metal it
  bounces a specular hotspot straight back, blowing out the relief detail that
  identifies the design. Use fixed diffuse lighting at roughly 45° instead.
- **Neutral white (~5000K), not warm.** Colour separates the coins into copper,
  Nordic gold and bimetallic groups, and warm light compresses that distinction.
- **Lock AE, AWB and focus** before each session, and turn off HDR, night mode
  and any scene optimisation. Drifting auto white balance does more damage to
  colour consistency than the choice of lamp.
- Keep the tray in the central part of the frame and shoot from as far back as
  practical. A four-point homography corrects perspective but not lens barrel
  distortion, which is worst at the frame edges.
- Pour coins sparsely. Touching coins share an edge and cannot be separated by
  circle detection.

### Why diameter is measured, not detected

Hough circle detection reports the strongest accumulator peak, which is not a
measurement: it reads a few percent small on a plain disc, and on a bimetallic
1 or 2 euro coin it can lock onto the inner ring and report a radius 20–30%
short — reading a 2 euro coin as roughly 19 mm, i.e. a 10 cent piece. Every
detection is therefore re-measured from the thresholded coin area
(`r = sqrt(A / pi)`), and accepted only if it falls within the physically
possible coin size range.

---

## Hardware

Coins are photographed in a custom 3D-printed tray designed to hold coins flat and evenly spaced, allowing consistent overhead imaging across scans.

---

## Status

> 🚧 Work in progress

- [x] End-to-end pipeline skeleton with swappable stage backends
- [x] Corner-marker calibration and metric rectification
- [x] Sub-millimetre coin diameter measurement
- [x] Normalisation pipeline
- [x] Rare coin database schema
- [x] CLI interface
- [x] Results export (JSON / CSV / crops)
- [ ] Tray segmentation model (Hough baseline in place; YOLO not trained)
- [ ] Denomination classifier (diameter baseline in place; CNN not trained)
- [ ] Rare coin database (initial set — seed file is unverified placeholders)
- [x] Numista client — variety lookup and mintage-outlier sweep (needs an API key)
- [ ] Rare coin detection model (approach undecided)
- [x] Pairing — matching each coin's two faces into one record

### Measured on a real 80-coin tray

| | Hough | Watershed |
|---|---|---|
| Coins found (of ~80) | 99 | 70 |
| Impossible overlapping pairs | 43 | **0** |
| Scale accuracy | — | within 0.4% |

Hough's 99 was inflated by double-detections and straddled pairs. The watershed
misses are almost all worn copper coins, which are as dark as the tray. Closing
that gap needs a trained detector; see `segment.copper_saturation` for an
attempted colour fix and why it backfired.

Pairing on the same batch matched 52 of 72, with 60% of pairs agreeing on
denomination across both faces. Inspection of the composites shows the matches
are largely correct — national side paired with value side, consistent metal and
size — and the 1 and 2 euro coins that matter most for rare detection are
identified correctly.

Getting there needed the two trays' real marker spacing, taken from CAD. They
are deliberately different sizes so one nests inside the other, and treating
them as identical put coins up to 6.5 mm from where their other face landed.
With the correct figures the scale is within 1%.

Beware two misleading metrics here. Agreement between the two faces on
denomination rewards consistent bias — it was higher (59%) when both faces were
wrong in the same direction than after the bias was reduced (40%). And per-coin
nearest-reference matching cannot detect a scale error of about one denomination
step, since every coin still lands near *a* real diameter; that is why `measure`
fits the whole population as well.

Recall and matching are limited by how densely the tray is packed:

- **Recall.** A coin missing from either photo cannot be paired, and at 70 and
  72 of ~80 only about 79% of coins are available to match at all.
- **Movement.** Coins shift a median of 5.1 mm between the two photos, because a
  packed tray lets them roll as they are slid across. No constant offset
  explains it — `pair.auto_align` searches for one and finds too little
  agreement to apply. Lanes in the tray would constrain this to one dimension
  and make pairing a sequence alignment per lane instead of a 2D search.

The thresholds are set to prefer an unmatched coin over a wrong one: a gap is
visible, whereas a bad match silently joins one coin's obverse to another's
reverse.

---

## Motivation

Built as a portfolio project combining computer vision and practical numismatics. The goal is to be able to scan coins obtained from banks or accumulated over time and automatically surface anything worth keeping.
