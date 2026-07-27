"""Build the issue catalogue from Wikipedia's commemorative-coin tables.

Rarity in circulation is decided by mintage: a design struck 15,000 times is
worth looking for, one struck 50 million times is not. So the catalogue stores
every issue with its mintage rather than a hand-picked "rare" list, and rarity
becomes a threshold applied at query time. That way the judgement can be
retuned without re-fetching anything, and an issue is never silently omitted
because someone's shortlist predates it.

Wikipedia is the source because its tables carry the mintage column outright and
the text is CC BY-SA, so it can be stored and redistributed with attribution.
Numista is the broader catalogue but its terms of use forbid extracting a
substantial part of the database, with no personal-use exemption, so it is not
used here.

Mintage figures come from the Official Journal of the European Union, which
publishes each issue before it is struck. Wikipedia notes that actual production
"sometimes deviates considerably" from the published figure -- these are
planned volumes, accurate enough to sort common from scarce but not exact.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Iterable

WIKI_PAGE = "2 euro commemorative coins"
WIKI_API = "https://en.wikipedia.org/w/api.php"
SOURCE_URL = "https://en.wikipedia.org/wiki/2_euro_commemorative_coins"

#: All commemoratives are 2 EUR by definition -- a member state may issue two a
#: year, and only in that denomination.
COMMEMORATIVE_DENOMINATION = 200

#: Wikipedia flag templates -> country. Several countries appear under more than
#: one alias, the page having been edited by many hands over twenty years.
COUNTRY_CODES = {
    "AND": "Andorra", "AUT": "Austria", "BEL": "Belgium", "CRO": "Croatia",
    "HRV": "Croatia", "CYP": "Cyprus", "EST": "Estonia", "FIN": "Finland",
    "FRA": "France", "DEU": "Germany", "GER": "Germany", "GRC": "Greece",
    "GRE": "Greece", "IRL": "Ireland", "ITA": "Italy", "LAT": "Latvia",
    "LVA": "Latvia", "LIT": "Lithuania", "LTU": "Lithuania",
    "LUX": "Luxembourg", "MLT": "Malta", "MCO": "Monaco", "MON": "Monaco",
    "NLD": "Netherlands", "NED": "Netherlands", "POR": "Portugal",
    "PRT": "Portugal", "SMR": "San Marino", "SVK": "Slovakia",
    "SVN": "Slovenia", "SLO": "Slovenia", "ESP": "Spain",
    "VAT": "Vatican City", "EU": "Eurozone (joint issue)",
}

_MULTIPLIER = {"million": 1_000_000, "millions": 1_000_000,
               "billion": 1_000_000_000, "thousand": 1_000}


@dataclass
class Issue:
    """One commemorative issue."""

    country: str
    year: int
    feature: str
    mintage: int | None
    volume_text: str
    date: str
    description: str = ""

    @property
    def name(self) -> str:
        return f"{self.country} {self.year} {self.feature}".strip()


def fetch_wikitext(page: str = WIKI_PAGE, timeout: int = 60) -> str:
    """Fetch a page's wikitext through the MediaWiki API."""
    query = urllib.parse.urlencode({
        "action": "parse", "page": page, "prop": "wikitext",
        "format": "json", "formatversion": "2",
    })
    request = urllib.request.Request(
        f"{WIKI_API}?{query}",
        headers={"User-Agent": "euro-vision/0.1 (personal numismatics project)"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if "error" in payload:
        raise RuntimeError(f"wikipedia: {payload['error'].get('info')}")
    return payload["parse"]["wikitext"]


def strip_markup(text: str) -> str:
    """Reduce wikitext to plain readable text."""
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]|]*)\]\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", text)
    text = re.sub(r"\[https?://\S*\]", "", text)
    text = re.sub(r"</?[^>]+>", "", text)
    text = text.replace("'''", "").replace("''", "")
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_volume(cell: str) -> tuple[int | None, str]:
    """Mintage as an integer, plus the text it came from.

    Returns None rather than a guess when the figure is missing or unreadable:
    a wrong mintage changes whether a coin is called rare, and a missing one is
    easy to spot later while a fabricated one is not.
    """
    raw = strip_markup(cell)
    if not raw or re.search(r"\b(unknown|n/?a|tbd|\?)\b", raw, re.I):
        return None, raw

    match = re.search(r"([\d][\d.,\s]*)\s*(million|millions|billion|thousand)?",
                      raw, re.I)
    if not match:
        return None, raw
    number = match.group(1).replace(",", "").replace(" ", "").rstrip(".")
    if not number:
        return None, raw
    unit = (match.group(2) or "").lower()
    try:
        value = float(number)
    except ValueError:
        return None, raw
    if unit:
        value *= _MULTIPLIER[unit]
    elif "." in number and value < 1000:
        # A bare "1.5" is ambiguous between 1.5 million and one and a half
        # coins. Only accept a decimal when a unit spells out the scale.
        return None, raw
    return int(round(value)), raw


def parse_issues(wikitext: str) -> list[Issue]:
    """Every commemorative issue in the page, in year order."""
    issues: list[Issue] = []
    # Sections are "=== 2014 coinage ===" and "=== 2015 commonly issued coin ===";
    # splitting on the year keeps each table with the year it belongs to, which
    # the rows themselves do not carry.
    sections = re.split(r"^===+\s*(\d{4})[^=]*===+\s*$", wikitext, flags=re.M)
    for i in range(1, len(sections), 2):
        year = int(sections[i])
        for table in re.findall(r"\{\|.*?\|\}", sections[i + 1], flags=re.S):
            issues.extend(_parse_table(table, year))
    return issues


def _parse_table(table: str, year: int) -> list[Issue]:
    out: list[Issue] = []
    pending: Issue | None = None
    for row in table.split("|-")[1:]:
        row = row.strip()
        if not row or row.startswith("!"):
            continue
        cells = _cells(row)
        if not cells:
            continue

        # The description spans all five columns and follows its own data row.
        if len(cells) == 1 and pending is not None:
            text = strip_markup(cells[0])
            if text.lower().startswith("description"):
                pending.description = text.split(":", 1)[-1].strip()
            continue

        code = _country_code(row)
        if code is None:
            continue
        # The image cell carries a rowspan and so is missing from all but the
        # first row of its group. Counting back from the end is stable where
        # counting forward is not.
        mintage, volume_text = parse_volume(cells[-2] if len(cells) >= 2 else "")
        pending = Issue(
            country=COUNTRY_CODES.get(code, code),
            year=year,
            feature=strip_markup(cells[-3]) if len(cells) >= 3 else "",
            mintage=mintage,
            volume_text=volume_text,
            date=strip_markup(cells[-1]),
        )
        out.append(pending)
    return out


def _cells(row: str) -> list[str]:
    cells: list[str] = []
    for line in row.split("\n"):
        line = line.strip()
        if not line.startswith("|"):
            continue
        for part in re.split(r"\|\|", line.lstrip("|")):
            part = part.strip()
            # Drop a leading style attribute, e.g. |style="width:20%;"| Greece
            if "=" in part.split("|")[0] and "|" in part:
                part = part.split("|", 1)[1].strip()
            if part:
                cells.append(part)
    return cells


def _country_code(row: str) -> str | None:
    for code in re.findall(r"\{\{([A-Za-z]{2,3})\}\}", row):
        if code.upper() in COUNTRY_CODES:
            return code.upper()
    return None


def load_commemoratives(page: str = WIKI_PAGE) -> list[Issue]:
    return parse_issues(fetch_wikitext(page))


def rarity_note(mintage: int | None, threshold: int) -> str:
    """Short human-readable reason a coin is or is not worth a look."""
    if mintage is None:
        return "mintage unknown — check by hand"
    if mintage <= threshold:
        return f"low mintage ({mintage:,})"
    return f"common issue ({mintage:,})"


def scarce(issues: Iterable[Issue], threshold: int) -> list[Issue]:
    return [i for i in issues
            if i.mintage is not None and i.mintage <= threshold]
