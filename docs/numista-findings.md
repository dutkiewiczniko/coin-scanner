# Numista: what it does and does not give us

Working notes from wiring up the Numista API and loading the rare coin
database. Written down because most of it was learned by getting it wrong first,
and none of it is in Numista's documentation.

Date: 2026-07-29. Code: [`numista.py`](../src/euro_vision/numista.py),
`euro-vision numista *`. Target lists: `data/numista_targets_*.csv`.

---

## The short answer

**Numista does not tag coins as scarce.** There is no rarity field, no
scarcity flag, nothing. `tags` is subject matter — the Hamburg mule's tags are
`["Religious building", "Map"]`. Everything below is derived.

What it actually gives, and what each is good for:

| Field | Endpoint | Use |
|---|---|---|
| `mintage` per year + mint letter | `/types/{id}/issues` | the real scarcity signal |
| `comment` on an issue | `/types/{id}/issues` | **where the errors are documented**, and the only reliable place |
| `comments` on a type (HTML) | `/types/{id}` | background prose; claims may apply to only some years |
| `related_types` | `/types/{id}` | links a mule to the normal coin — cheap way to find varieties |
| `value.numeric_value` | `/types/{id}` | authoritative denomination; do not parse the title |
| `obverse/reverse.picture` + licence | `/types/{id}` | images for `reference_image`, with copyright holder recorded |
| `/prices` | separate call | catalogue value by grade |

---

## Three things that cost real time

### 1. Filtering "BU" would have deleted the best coin

Monaco 2007 Grace Kelly — 20,001 struck, the most sought-after euro coin there
is — has an issue comment of exactly `"BU"`. Meanwhile `"BU set"` and
`"Proof set"` are collector products that never reach circulation.

So the test order in `issue_quality()` is load-bearing: **`set` must be tested
before `bu`**, never the reverse. Reversed, every set in the catalogue is kept
and Grace Kelly is thrown out with them.

This is also the single highest-value filter in the whole pipeline. Slovenia's
2007 2 EUR lists **21,250,000** for circulation and **100,000** as a `"BU set"`.
Mintage alone picks the wrong one every time. Batch 1 dropped 347 collector-only
issues to keep 47; batch 2 dropped 1,162 to keep 10.

### 2. Numista does not have the most valuable euro coin

Italy's 2002 1 cent struck with the 2 cent Mole Antonelliana reverse — perhaps
a hundred known, worth thousands — has **no entry at all**. No type, no
`related_types` link, no mention in the 1c type's comments. Searched four ways.

With ~100 known it is a mint error on individual coins, not a variety with a
production run, and Numista catalogues *types*. This is the boundary:

| | Belongs to | Catalogued? | Handled by |
|---|---|---|---|
| Mintage | an issue | yes | `catalogue.py` (Wikipedia), Numista |
| Variety / mule | an issue | **yes, Numista only** | `numista.py` |
| Mint error | one individual coin | **never** | `errors.py`, geometrically |

No catalogue crosses the third row. Do not go looking for a source that does.

### 3. Type-level evidence is not issue-level evidence

Finland's N# 95 has type comments reading *"Die error on the 2000 coin / Die
errors on 2001 coin / Die error on 2005 coin"*. That is a claim about three
years. Reading it as a property of the type labelled **all nine years**
1999–2006 as errors, including years the note explicitly does not mention.

`variant='error'` now requires the **issue's own** comment to describe a fault.
Type-level notes are surfaced as `(type notes errors in some years)` and written
into `notes` as an explicit caveat, never as a variant.

The database is what the rare stage acts on, so it has to carry the narrower
claim. 52 rows had to be deleted and rebuilt after this was spotted.

---

## What is in the database now

664 rows total: **612 from Wikipedia** (2 EUR commemoratives, loaded by
`db import-commemoratives`) and **52 from Numista**.

| Denomination | standard | commemorative | error |
|---|---|---|---|
| 2 EUR | 32 | 6 | 2 |
| 1 EUR | 9 | — | 1 |
| 20c | — | — | 1 |
| 1c | — | — | 1 |

13 countries: Austria, Croatia, Cyprus, Estonia, Finland, Germany, Italy,
Latvia, Lithuania, Malta, Monaco, Portugal, Slovenia.

### The five confirmed varieties

Each documented on its own issue comment, not inferred from anything.

| Mintage | Value | Year | Country | N# | What |
|---|---|---|---|---|---|
| 55,000 | 2 EUR | 2006 | Finland | 259433 | mule, wrong reverse |
| 600,000 | 2 EUR | 2008 | Germany | 327881 | the Hamburg mule, KM# A261 |
| 98,375 | 1 EUR | 2008 | Portugal | 354653 | old map mule; ~98,000 believed still circulating |
| 200,000 | 20c | 2007 | Germany | 109 | map transition |
| unknown | 1c | 2002 | Italy | 129 | missing R mintmark |

The Portugal and Finland mules were **not targeted by name** — the generic
`2 euro mule` / `1 euro mule` searches found them. Worth repeating that trick
for other denominations.

### Known soft spot in the stored rows

**31 of the 52 rows are `bu` quality, 21 are `circulation`.** The BU ones are
mostly Cyprus and Lithuania 2 EUR at 5,000–7,000 mintage. They are kept because
plain `BU` is kept for Grace Kelly's sake, but a Lithuanian BU single is far
less likely to turn up in a bank tray than she is.

If the flag list ever comes out too long, dropping `bu` for everything except
Monaco is the first thing to try. Do not drop it globally.

---

## Batch rules, and why each is different

```bash
euro-vision numista fetch data/numista_targets_200.csv   --max-mintage 500000 --save
euro-vision numista fetch data/numista_targets_100.csv   --max-mintage 300000 --save
euro-vision numista fetch data/numista_targets_small.csv --varieties-only    --save
```

- **2 EUR, ≤ 500,000** — where the low-mintage commemoratives live.
- **1 EUR, ≤ 300,000** — no 1 EUR commemoratives exist (a member state may only
  issue those at 2 EUR), so scarcity is entirely micro-state and first-year.
- **1c–50c, varieties only** — a 1c struck 5,000 times is still worth a cent.
  Only the fault carries value, so the mintage test is dropped rather than
  tightened. This turned 262 rows of noise into 2 real finds.

Two rules cut across all three:

- **A variety is kept at any mintage, including unknown.** The Hamburg mule ran
  to 600,000 — common, and still the coin worth finding. Italy's missing-R has
  no figure at all, and skipping unmintaged issues discarded exactly what we
  were looking for.
- **Every CSV row is a search, never a fact.** Mintages always come back from
  Numista, so nothing enters the database on the strength of anyone's memory,
  and a target that turns out to be common is dropped rather than recorded as
  rare.

---

## API gotchas

- **API key page is unlinkable.** <https://en.numista.com/api/api_key.php>. On
  `/api/index.php` it is a `<button onclick>` with no href, so it appears in no
  navigation and no search index. Log in first or it 302s to login.
- **Header is `Numista-API-Key`.** Base URL `https://api.numista.com/v3`.
- **The comment field on a type is `comments`, plural, and is HTML** — bold
  tags, `&quot;` entities, `<br />`, sometimes an embedded `<img>`. Reading
  `comment` returns nothing at all and fails silently.
- **Issue field is `mint_letter`**, and mintage arrives as an int.
- **Issuer codes are not guessable and a wrong one is a hard HTTP 400**, not an
  empty result. Mostly French slugs, but the separator varies: `saint-marin` and
  `pays-bas` with hyphens, `pays_bas` rejected outright. All 24 euro codes in
  `EURO_ISSUERS` are verified against a live `/issuers` response — do not add
  one by guessing. `/issuers` returns 11,840 entries and is cheap to re-check.
- **Quota: 2,000 requests/month free.** Used ~352 so far for all three batches
  plus exploration. Responses cache under `data/cache/numista` (gitignored, 30
  day TTL), so re-running a batch costs nothing — which is what made rebuilding
  the database after finding bug 3 free.
- **Search by image is a paid plan** — €100 activation, €0.03/call, €100/month
  floor — and 403s on a free key. It is the endpoint that would identify a coin
  straight from a crop. `Numista.search_by_image()` is written and waiting.

### Terms of use

The ToU forbids extracting a substantial part of the database, with **no
personal-use exemption** — which is why `catalogue.py` sources mintages from
Wikipedia. Using the sanctioned API is a different thing from mirroring, and the
current approach stays on the right side: bounded sweeps, caching to avoid
re-fetching rather than to accumulate, and only the handful of coins that come
back scarce written locally, each with its N# and source URL. The cache is
gitignored deliberately — a committed cache would be redistribution.

---

## Bugs found and fixed

Kept because each one failed quietly rather than loudly.

| Bug | Symptom |
|---|---|
| Read `comment`, field is `comments` | type descriptions silently always empty |
| `"50 Euro Cents"` parsed euro-first | read as €50 instead of 50c |
| Variety keyword missed `the 'Old' Europe Map` | apostrophes; now punctuation-flattened |
| One bad issuer code aborted the whole run | ~80 requests spent, all results discarded |
| Commas inside unquoted CSV notes | shifted columns |
| Issues with `mintage: None` skipped | dropped Italy's missing-R error, an actual target |
| Type-level hint applied per issue | nine Finland years stored as errors |
| Tests built "keyless" clients | passed only while `$NUMISTA_API_KEY` was unset; now an autouse fixture |

---

## Next steps

1. **Reference images.** `/types/{id}` carries `obverse.picture` and
   `reverse.picture` with `picture_copyright` and `picture_license_name` (mostly
   CC BY-NC). The `reference_image` table exists and the embedding backend needs
   it. Storing URLs rather than downloading is the ToU-safer starting point.
2. **Per-mintmark splits for the Wikipedia commemoratives.** Wikipedia gives one
   planned figure per issue; Numista splits by mint letter and gives actuals.
3. **`related_types` sweep.** Every mule found so far links back to its normal
   coin. Walking that edge from known types is cheaper than text search and
   would likely surface varieties the queries missed.
4. **Generic mule searches at other denominations.** `2 euro mule` and
   `1 euro mule` each found a mule that was not on the target list. 5c/10c/50c
   have not had the same treatment.
5. Belgium and Luxembourg variety searches matched nothing — either genuinely
   absent or the wording is wrong. Unresolved.
