# Euro mint-error image sources

Researched 2026-07-28. Goal: images (top-down preferred) of euro coins with striking errors,
for training data or as reference for synthetic generation. Ordered by expected yield.

## Tier 1 — high volume, scrapeable

### 1. forum.emuenzen.de — "Bilder von Euro-Fehlprägungen" megathread
- https://forum.emuenzen.de/threads/bilder-von-euro-fehlpraegungen.9830/
- ~400 pages, running since 2002. The single largest concentration of euro error photos online.
- Pages ~1–150 (pre-~2015) mostly link to dead external hosts (fotango.de) — skip.
- Recent pages use native XenForo attachments: `/data/attachments/<nnn>/<id>-<hash>.jpg`
  (thumbnails public; full-size may need a free account → log in and reuse session cookies).
- Amateur photos, but mostly straight-on obverse/reverse shots. German labels in post text
  (Dezentrierung, Stempeldrehung, Stempelriss, Zainende, Doppelprägung...) → weak labels via keyword match.
- Also browse the tag index: https://forum.emuenzen.de/tags/fehlpraegung/
- Estimated yield: 3,000–8,000 usable images.

### 2. eBay (already partly scraped — extend)
- Extend `tools/fetch_ebay_errors.py` beyond ebay.de/ebay.com:
  - ebay.fr: `2 euro fauté`, `pièce euro erreur de frappe`, `décentrée`
  - ebay.it: `euro errore di conio`, `decentrata`, `doppia battitura`
  - ebay.es: `euro error de acuñación`, `desplazada`
  - ebay.de extra terms: `Zainende`, `Stempeldrehung`, `Dezentriert`, `auf falschem Rohling`,
    `Pille dezentriert`, `ohne Pille` (missing bimetal insert), `Spiegelei` (broadstrike slang)
- Sold/completed listings (`LH_Sold=1&LH_Complete=1`) multiply volume and are seller-verified-ish.
- Estimated yield: 1,000–3,000 additional images. Caveat: listings mix true errors with post-mint damage.

### 3. Catawiki — dedicated weekly "Euro Coins (Errors and Defects)" auctions
- https://www.catawiki.com/en/a/th/17947-euro-coins-errors-and-defects
- Professional/expert-reviewed lots, multiple photos per lot, usually clean top-down obverse+reverse.
- Closed lots remain browsable. Cloudflare-protected: WebFetch/requests get 403 →
  scrape with Playwright (browser automation), throttled.
- Estimated yield: 1,000–5,000 high-quality images (best photo quality of all sources).

## Tier 2 — medium volume, good labels

### 4. MA-Shops error-coin category (German dealer marketplace)
- https://www.ma-shops.de/shops/maCategory.php?catid=11909 (Euro Fehlprägungen)
- https://www.ma-shops.com/error-coins/ (all error coins)
- Dealer photos, consistent top-down style, descriptive German titles → good labels.
- Estimated yield: 300–1,000 images.

### 5. NumisBids / Sixbid auction archives
- Dedicated past sessions, e.g. The Coin House Auction 13 "Euro Error Coins" (2019),
  17 Auctions S.L. Auction 8 "Euro Error Coins" (2024):
  - https://www.numisbids.com/n.php?p=sale&sid=3601&cid=109210
  - https://www.numisbids.com/n.php?p=sale&sid=7760&cid=246160
- Archive search: https://www.sixbid-coin-archive.com/ (search "Fehlprägung", "mint error euro")
- Professional photos + expert descriptions (best label quality). 403 on plain HTTP → Playwright.
- Estimated yield: 500–2,000 images.

### 6. sammler.com euro varieties pages
- https://www.sammler.com/mz/euro_abarten.htm + per-country subpages
  (Belgium…Cyprus, each with reported errors + photos).
- Small images but documented/verified. Estimated yield: 300–800 images.

### 7. 2-euromuenzen.de Fehlprägungen catalog
- https://www.2-euromuenzen.de/2-euromuenzen/fehlpraegungen/
- Catalog-style pages for known 2€ errors with photos.

## Tier 3 — reference / taxonomy / small

- **Neugebauer (Battenberg), "Varianten und Fehlprägungen der Euro-Münzen"** — free 38-page PDF,
  66 labeled coin photos, the de-facto German error taxonomy:
  https://www.numis-online.ch/download/Neugebauer-Varianten-und-Fehlpraegungen-opt.pdf
- **Künker "Fehlprägung: ja oder nein?"** PDF (error vs post-mint damage guide — useful for
  cleaning weak labels): https://www.kuenker-numismatik.de/wp-content/uploads/2024/05/Fehlpraegungen.pdf
- **error-ref.com** — most comprehensive error taxonomy with photos (US coins, but the
  physics/classes transfer; use for the label schema).
- **Wikimedia Commons** — already scraped (142 imgs, mostly non-euro/historical). Licensed (CC).
- **wert2euro.de 2€ error catalog** — paid PDF/book (~€20), photos + valuations.
- **Reddit r/coinerrors, r/euro** — occasional euro errors; scrapeable via Reddit API/PRAW.
- **Facebook groups** ("2 Euro Fehlprägungen" etc.) — large but effectively unscrapeable; skip.
- **Numista / uCoin** — catalog die *varieties*, not one-off striking errors; use their stock
  images of NORMAL coins as clean bases for synthetic error generation (already have
  `data/numista_targets_*.csv`).

## Ready-made ML datasets
None found. No euro-error dataset on Roboflow Universe / Kaggle / HuggingFace as of 2026-07.
Generic "coin defect" datasets are tiny (~200 imgs) and not euro. We would be building the
first one.

## Licensing note
Commons images are CC-licensed. Everything else (eBay/Catawiki/dealers/forums/auctions) is
copyrighted — fine for internal model training, do not redistribute the raw images.

## Class balance expectation
Common in the wild: die cracks/Stempelriss, rotated die, weak strike, plating/lamination.
Moderately available: off-centre, broadstrike, Zainende (strip-end), clipped planchet,
double strike. Rare (tens of images total): wrong planchet, missing bimetal pill, mules.
→ geometric classes (off-centre, broadstrike, rotation, clip) are the best candidates for
synthetic augmentation from normal-coin stock images; texture classes (cracks, doubling)
need the real data above.
