"""Numista API client — the catalogue of *varieties*, which Wikipedia has not got.

Three different things get called "a rare coin" and only two of them can be
looked up at all. Keeping them apart is the whole reason this module exists
alongside `euro_vision.catalogue` and `euro_vision.errors`:

  mintage    How many of an issue were struck. A property of the issue, so a
             catalogue can state it. `catalogue.py` already gets this for 2 EUR
             commemoratives from Wikipedia. Numista adds the standard
             circulating coins and splits by mint mark, which Wikipedia does
             not.

  variety    A design fault that ran through a whole production batch — a mule,
             a wrong die pairing, a map that should have been replaced. Every
             coin from that run carries it identically, so it *is* an issue and
             Numista gives it its own catalogue entry. Germany's 2008 Hamburg
             mule is N# 327881, KM# A261, mintage 600,000, sitting beside the
             ordinary N# 2994 struck 9,000,000 times. This is the half of the
             problem Numista is here for.

  mint error An off-centre strike, a clipped planchet, a broadstrike. Belongs
             to one individual coin, not to an issue. No catalogue can list
             these and Numista does not try — there is nothing to match
             against. Measured geometrically instead; see `errors.py`.

So the answer to "does the description name the exact mintage issue" is: for a
variety, effectively yes — but do not read it out of the prose. The comment
field is community-written free text in whatever language and phrasing the
contributor chose, and parsing it is a losing game. The reliable signal is
structural: the variety is a *separate type* with its own mintage, so it shows
up as an anomalously low figure among the types sharing an issuer, denomination
and year. `mintage_outliers` below leans on that instead of on wording.

Terms of use. `catalogue.py` avoids Numista because the ToU forbids extracting
a substantial part of the database, with no personal-use exemption. That still
stands and this module does not repeal it. What it does is stay on the right
side of it: requests go through the sanctioned API with a key, responses are
cached to avoid re-fetching rather than to build a mirror, and only the handful
of coins that come back scarce are written to the local database, each carrying
its N# and source URL. Walking every euro type to build a local copy of the
catalogue is exactly the thing the ToU prohibits, so `scarce_varieties` is
capped and asks before it spends.

The free tier is 2,000 requests a month, which one careless sweep will eat.
Search by image — the one endpoint that would identify a coin from a crop —
is a paid plan (100 EUR/month minimum plus 0.03 EUR a call) and returns 403 on
a free key.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

BASE_URL = "https://api.numista.com/v3"
API_KEY_ENV = "NUMISTA_API_KEY"
#: Direct link to key generation. Not discoverable by browsing: on /api/index.php
#: it is a <button onclick> rather than an anchor, so it has no href, appears in
#: no navigation, and is not indexed. Log in first or it redirects to the login
#: page.
KEY_URL = "https://en.numista.com/api/api_key.php"

#: Numista issuer codes for the euro-area states, as used by ?issuer=. Mostly
#: French-language slugs, the site having been built in French originally.
#:
#: Every one of these is verified against a live /issuers response — do not
#: guess a new one. The separator is not predictable: it is `saint-marin` and
#: `pays-bas` with hyphens but `pays_bas` is rejected outright, and an invalid
#: code is a hard HTTP 400 rather than an empty result. `euro-vision numista
#: issuers --check` re-verifies the whole table for one request.
EURO_ISSUERS = {
    "andorra": "andorre", "austria": "autriche", "belgium": "belgique",
    "croatia": "croatie", "cyprus": "chypre", "estonia": "estonie",
    "finland": "finlande", "france": "france", "germany": "allemagne",
    "greece": "grece", "ireland": "irlande", "italy": "italie",
    "latvia": "lettonie", "lithuania": "lituanie", "luxembourg": "luxembourg",
    "malta": "malte", "monaco": "monaco", "netherlands": "pays-bas",
    "portugal": "portugal", "san_marino": "saint-marin", "slovakia": "slovaquie",
    "slovenia": "slovenie", "spain": "espagne", "vatican": "vatican",
}


class NumistaError(RuntimeError):
    """Any failure talking to the API."""


class NumistaAuthError(NumistaError):
    """Missing, wrong, or insufficiently privileged API key."""


class NumistaQuotaError(NumistaError):
    """Rate limit or monthly quota exhausted."""


@dataclass
class TypeSummary:
    """One catalogue type, flattened to the fields this project cares about."""

    id: int
    title: str
    issuer: str
    min_year: int | None
    max_year: int | None
    #: Krause-Mishler and friends. A variety usually gets a suffixed number of
    #: its own — KM# A261 against KM# 261 — which is a useful corroborating hint
    #: but not one to depend on, since not every variety has been assigned one.
    references: list[str] = field(default_factory=list)
    url: str = ""

    @property
    def n_number(self) -> str:
        return f"N# {self.id}"


@dataclass
class Issue:
    """One year/mintmark line of a type, with its mintage."""

    id: int
    year: int | None
    mint_letter: str | None
    mintage: int | None
    #: Free text on the year line. Verified against N# 327881, where this is
    #: where the variety is described ("Error: old map style on reverse (Mule)")
    #: and the type-level comment is empty. Anything reading the prose has to
    #: look in both places.
    comment: str = ""


@dataclass
class Candidate:
    """A type whose mintage is out of line with its siblings.

    `siblings` is the largest mintage among the types it was compared against,
    so the ratio that made it stand out stays visible in the output rather than
    being collapsed into a score.
    """

    type: TypeSummary
    year: int | None
    mintage: int
    sibling_mintage: int
    reason: str

    @property
    def ratio(self) -> float:
        return self.mintage / self.sibling_mintage if self.sibling_mintage else 0.0


class Numista:
    """Client for the Numista REST API v3.

    Uses stdlib urllib rather than requests so the project keeps its current
    dependency set; the API is half a dozen GETs and needs nothing more.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        lang: str = "en",
        cache_dir: str | Path | None = "data/cache/numista",
        cache_days: int = 30,
        min_interval_s: float = 1.0,
        timeout: int = 30,
    ):
        self.api_key = api_key or os.environ.get(API_KEY_ENV, "").strip()
        self.lang = lang
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_days = cache_days
        self.min_interval_s = min_interval_s
        self.timeout = timeout
        #: Requests actually sent, cache hits excluded. This is what counts
        #: against the monthly quota, so it is worth being able to report.
        self.requests_made = 0
        self._last_request = 0.0

    # -- transport --------------------------------------------------------

    def _cache_path(self, path: str, params: dict[str, Any]) -> Path | None:
        if self.cache_dir is None:
            return None
        key = json.dumps([path, sorted(params.items())], sort_keys=True)
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        # Keep the endpoint in the filename so the cache stays inspectable by
        # hand; the digest alone would make a stale entry impossible to find.
        stem = path.strip("/").replace("/", "_")[:48]
        return self.cache_dir / f"{stem}_{digest}.json"

    def _read_cache(self, cache: Path | None) -> Any | None:
        if cache is None or not cache.exists():
            return None
        if self.cache_days > 0:
            age_days = (time.time() - cache.stat().st_mtime) / 86400
            if age_days > self.cache_days:
                return None
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        params = {"lang": self.lang, **(params or {})}
        params = {k: v for k, v in params.items() if v is not None}

        cache = self._cache_path(path, params)
        cached = self._read_cache(cache)
        if cached is not None:
            return cached

        if not self.api_key:
            raise NumistaAuthError(
                f"no Numista API key. Set {API_KEY_ENV}, or put one in "
                f"numista.api_key_env. Generate a free key at {KEY_URL}"
            )

        # Numista enforces a per-second limit as well as the monthly quota, and
        # sweeping an issuer fires requests back to back.
        wait = self.min_interval_s - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)

        url = f"{BASE_URL}{path}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(
            url,
            headers={
                "Numista-API-Key": self.api_key,
                "Accept": "application/json",
                "User-Agent": "euro-vision/0.1 (personal numismatics project)",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise _http_error(exc) from exc
        except urllib.error.URLError as exc:
            raise NumistaError(f"could not reach {BASE_URL}: {exc.reason}") from exc
        finally:
            self._last_request = time.monotonic()
            self.requests_made += 1

        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    # -- endpoints --------------------------------------------------------

    def check(self) -> dict[str, Any]:
        """Cheapest call that proves the key works. One request."""
        return self._get("/types", {"q": "2 euro", "category": "coin", "count": 1})

    def search_types(
        self,
        q: str | None = None,
        *,
        issuer: str | None = None,
        category: str = "coin",
        page: int = 1,
        count: int = 50,
    ) -> tuple[int, list[TypeSummary]]:
        """One page of a type search. Returns (total matches, this page)."""
        payload = self._get(
            "/types",
            {"q": q, "issuer": issuer, "category": category,
             "page": page, "count": count},
        )
        return int(payload.get("count", 0)), [
            _type_summary(t) for t in payload.get("types", [])
        ]

    def all_types(
        self,
        *,
        q: str | None = None,
        issuer: str | None = None,
        category: str = "coin",
        max_requests: int = 20,
        page_size: int = 50,
    ) -> Iterator[TypeSummary]:
        """Page through a search, stopping at `max_requests` pages.

        Bounded on purpose. An unbounded walk of a large issuer is both a quota
        problem and the "substantial part of the database" the ToU forbids.
        """
        seen = 0
        for page in range(1, max_requests + 1):
            total, types = self.search_types(
                q=q, issuer=issuer, category=category, page=page, count=page_size
            )
            if not types:
                return
            yield from types
            seen += len(types)
            if seen >= total:
                return

    def get_type(self, type_id: int | str) -> dict[str, Any]:
        """Full record for one type, including the free-text comment."""
        return self._get(f"/types/{type_id}")

    def get_issues(self, type_id: int | str) -> list[Issue]:
        """Year/mintmark lines for a type. This is where mintage lives."""
        payload = self._get(f"/types/{type_id}/issues")
        return [_issue(row) for row in payload or []]

    def get_prices(self, type_id: int | str, issue_id: int | str) -> dict[str, Any]:
        """Catalogue values by grade, for triage only."""
        return self._get(f"/types/{type_id}/issues/{issue_id}/prices")

    def issuers(self) -> list[dict[str, Any]]:
        return self._get("/issuers").get("issuers", [])

    def search_by_image(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> Any:
        """Identify a coin from a photo. **Paid plans only.**

        This is the endpoint that would close the loop on this project — feed it
        a normalised crop and get back candidate types — but a free key gets 403.
        As of the current pricing it is 100 EUR to activate, 0.03 EUR a search
        and a 100 EUR/month floor, and the terms require displaying the N# of
        every result and cap follow-up `get_type` calls at 15 per search.
        """
        import base64

        body = json.dumps({
            "category": "coin",
            "images": [{
                "mime_type": mime_type,
                "image_data": base64.b64encode(image_bytes).decode("ascii"),
            }],
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{BASE_URL}/search_by_image",
            data=body,
            headers={
                "Numista-API-Key": self.api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 403:
                raise NumistaAuthError(
                    "search by image is not enabled on this key — it is a paid "
                    "plan. See https://en.numista.com/api/pricing.php"
                ) from exc
            raise _http_error(exc) from exc
        finally:
            self.requests_made += 1


# -- variety detection ----------------------------------------------------


def mintage_outliers(
    typed_issues: Iterable[tuple[TypeSummary, list[Issue]]],
    *,
    ratio: float = 0.25,
    max_mintage: int | None = 2_000_000,
) -> list[Candidate]:
    """Types struck far less than their same-year siblings.

    This is the structural version of "read the description". When a mint runs
    a batch with the wrong die, that batch becomes its own type with its own
    mintage, and it is small because it was caught and stopped — 600,000 Hamburg
    mules against 9,000,000 corrected coins. Grouping types by issuer and year
    and looking for the small one finds that pattern without reading a word of
    prose, in any language, including varieties documented only as a comment.

    It is a shortlist, not a verdict. Two legitimate things also produce a low
    sibling mintage: a proof or BU-set-only issue, and a genuinely short
    commemorative run. Both are worth a human glance anyway, which is what the
    output is for.

    `ratio` is how much smaller than the biggest sibling a type must be.
    `max_mintage` suppresses the case where every sibling is enormous and the
    "small" one was still struck in the millions; None disables it.
    """
    by_group: dict[tuple[str, int], list[tuple[TypeSummary, Issue]]] = {}
    for summary, issues in typed_issues:
        for issue in issues:
            if issue.mintage is None or issue.year is None:
                continue
            by_group.setdefault((summary.issuer, issue.year), []).append(
                (summary, issue)
            )

    out: list[Candidate] = []
    for (_, year), members in by_group.items():
        if len(members) < 2:
            continue
        largest = max(m[1].mintage for m in members)
        for summary, issue in members:
            if issue.mintage >= largest * ratio or issue.mintage == largest:
                continue
            if max_mintage is not None and issue.mintage > max_mintage:
                continue
            out.append(Candidate(
                type=summary,
                year=year,
                mintage=issue.mintage,
                sibling_mintage=largest,
                reason=(f"{issue.mintage:,} struck against {largest:,} for another "
                        f"{year} type from the same issuer"),
            ))
    return sorted(out, key=lambda c: c.ratio)


#: Words that show up in the comment of a genuine variety entry. Only ever used
#: to annotate a candidate that the mintage rule already found — never to find
#: one. The comments are contributor-written and multilingual, so absence of a
#: keyword says nothing at all.
#:
#: "old europe map" rather than "old map": the actual N# 327881 comment reads
#: "accidentally issued 600,000 of them with the 'Old' Europe Map", so the
#: shorter phrase never appears. A reminder that the wording is whatever the
#: contributor typed, and why nothing here decides anything on it.
VARIETY_WORDS = (
    "mule", "muling", "error", "variety", "variant", "wrong", "mistake",
    "accidental", "incorrect", "old europe map", "transitional", "die",
)


def plain_text(html: str) -> str:
    """Flatten a Numista comment to readable text.

    The `comments` field is HTML, not prose: bold tags, `&quot;` entities, `<br
    />` line breaks and sometimes an embedded `<img>` of the variety. Left raw
    it is unreadable in a terminal and useless in a database column.
    """
    import html as html_module
    import re

    text = re.sub(r"<br\s*/?>", "\n", html or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_module.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def variety_hint(comment: str) -> str:
    """Which variety words a type's comment contains, if any.

    Punctuation is flattened before matching. N# 327881 writes the map as
    "the 'Old' Europe Map", quotes and all, which a plain substring test misses
    by one apostrophe — a good illustration of why this annotates a candidate
    the mintage rule already found and never produces one.
    """
    import re

    flat = re.sub(r"[^a-z0-9]+", " ", (comment or "").lower())
    return ", ".join(w for w in VARIETY_WORDS if w in flat)


#: Issue qualities, ordered by how likely the coin is to turn up in a tray of
#: bulk circulation coins from a bank — which is the only question that matters
#: for this project.
CIRCULATING_QUALITIES = ("circulation", "bu")


def issue_quality(comment: str) -> str:
    """Whether an issue was struck for circulation or for collectors.

    This is the single most important filter on the data, and it is not a field
    — it has to be read out of the issue comment. Without it the results are
    swamped: Slovenia's 2007 2 EUR shows a 21,250,000 circulation line and a
    100,000 "BU set" line, and only the first can ever appear in a bank bag. A
    mintage-only threshold keeps the wrong one of the two every time.

      circulation  No marker. Struck for spending.
      bu           Brilliant uncirculated, sold singly or in a coincard. Does
                   reach circulation when someone spends it — Monaco's 2007
                   Grace Kelly, 20,001 struck and the most sought-after euro
                   coin there is, is marked exactly "BU" and nothing more.
      set          Sold only as part of a collector set. Occasionally broken up
                   and spent, but not worth carrying in a watchlist.
      proof        Mirror finish, sold cased. Effectively never circulates.

    The order matters: "BU set" is a set, not a BU, so `set` must be tested
    before `bu` or every set in the catalogue is kept.
    """
    text = re.sub(r"[^a-z0-9]+", " ", (comment or "").lower())
    if "proof" in text:
        return "proof"
    if "set" in text or "coincard" in text or "folder" in text or "blister" in text:
        return "set"
    if re.search(r"\bbu\b", text) or "brilliant uncirculated" in text:
        return "bu"
    return "circulation"


def denomination_cents(record: dict[str, Any]) -> int | None:
    """Face value in cents, from the type record's structured value field.

    Read from `value.numeric_value` rather than parsed out of the title, which
    is ambiguous — "50 Euro Cents" and "50 Euros" differ by one word. Returns
    None for anything not denominated in euro, which is how a search that
    wandered onto a pre-euro coin gets caught rather than stored with a
    plausible-looking wrong denomination.
    """
    value = record.get("value") or {}
    numeric = value.get("numeric_value")
    currency = ((value.get("currency") or {}).get("name") or "").lower()
    if numeric is None or "euro" not in currency:
        return None
    try:
        return int(round(float(numeric) * 100))
    except (TypeError, ValueError):
        return None


def variant_of(object_type: str | None, is_variety: bool) -> str:
    """Classify an issue as 'error', 'commemorative' or 'standard'.

    Matches the `variant` column the rare_coin table already uses.

    `is_variety` must come from the *issue* comment, not the type comments.
    Finland's N# 95 says "Die error on the 2000 coin" at type level, which is
    true of that one year and says nothing about 1999 or 2003 — deciding from
    it stored nine ordinary years as errors. What the database records is what
    the rare stage will act on, so it has to be the narrower claim.
    """
    if is_variety:
        return "error"
    if "commemorative" in (object_type or "").lower():
        return "commemorative"
    return "standard"


# -- parsing --------------------------------------------------------------


def _type_summary(raw: dict[str, Any]) -> TypeSummary:
    issuer = raw.get("issuer") or {}
    return TypeSummary(
        id=int(raw["id"]),
        title=raw.get("title", ""),
        issuer=issuer.get("name") or issuer.get("code") or "",
        min_year=_int(raw.get("min_year")),
        max_year=_int(raw.get("max_year")),
        references=[
            f"{r.get('catalogue', {}).get('code', '')}# {r.get('number', '')}".strip()
            for r in raw.get("references", []) or []
        ],
        url=raw.get("url") or f"https://en.numista.com/{raw['id']}",
    )


def _issue(raw: dict[str, Any]) -> Issue:
    return Issue(
        id=int(raw["id"]),
        year=_int(raw.get("year")),
        mint_letter=raw.get("mint_letter") or raw.get("mintmark"),
        mintage=_int(raw.get("mintage")),
        comment=raw.get("comment") or "",
    )


def _int(value: Any) -> int | None:
    """None rather than a guess — a wrong mintage decides whether a coin is rare."""
    if value in (None, "", "unknown"):
        return None
    try:
        return int(str(value).replace(",", "").replace(" ", "").split(".")[0])
    except (TypeError, ValueError):
        return None


def _http_error(exc: urllib.error.HTTPError) -> NumistaError:
    try:
        message = json.loads(exc.read().decode("utf-8")).get("error_message", "")
    except Exception:  # noqa: BLE001 - the body is best-effort context only
        message = ""
    if exc.code == 401:
        return NumistaAuthError(
            f"Numista rejected the API key ({message or 'unauthorised'}). "
            f"Generate one at {KEY_URL}"
        )
    if exc.code == 403:
        return NumistaAuthError(
            f"key is not authorised for this endpoint ({message or 'forbidden'}) "
            "— some endpoints need a paid plan"
        )
    if exc.code == 429:
        return NumistaQuotaError(
            f"rate limited or out of quota ({message or 'too many requests'}). "
            "The free tier is 2,000 requests a month."
        )
    return NumistaError(f"Numista returned HTTP {exc.code}: {message or exc.reason}")
