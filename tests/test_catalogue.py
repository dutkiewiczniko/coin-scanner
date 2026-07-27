from __future__ import annotations

from euro_vision.catalogue import (
    COUNTRY_CODES,
    parse_issues,
    parse_volume,
    rarity_note,
    scarce,
)

# Two rows in the page's own shape: an image cell carrying a rowspan, then a
# second coin whose row has no image cell at all, each followed by a description
# spanning all five columns.
SAMPLE = """
=== 2007 coinage ===

{|class="wikitable"
|-
! Image !! Country !! Feature !! Volume !! Date
|-
| style="width:160px;" rowspan="2"| [https://example.invalid/oj]
| style="width:20%;"| {{MCO}}
| style="width:40%;"| 25th anniversary of the death of [[Grace Kelly]]
| style="width:20%;"| 20,001 coins
| style="width:20%;"| 1 July 2007
|-
| colspan="5" | '''Description:''' A portrait of Grace Kelly.
|-
| style="width:20%;"| {{IRL}}
| style="width:40%;"| 50th anniversary of the Treaty of Rome
| style="width:20%;"| 4.6 million coins
| style="width:20%;"| March 2007
|-
| colspan="5" | '''Description:''' The Treaty document.
|}
"""


def test_parses_both_rows_of_a_rowspan_group():
    issues = parse_issues(SAMPLE)
    assert len(issues) == 2
    assert [i.country for i in issues] == ["Monaco", "Ireland"]


def test_reads_the_year_from_the_section_heading():
    # The rows carry a release date but not the coinage year, so it has to come
    # from the heading the table sits under.
    assert {i.year for i in parse_issues(SAMPLE)} == {2007}


def test_strips_wiki_markup_from_the_feature():
    issues = parse_issues(SAMPLE)
    assert issues[0].feature == "25th anniversary of the death of Grace Kelly"


def test_reads_an_exact_mintage():
    assert parse_issues(SAMPLE)[0].mintage == 20_001


def test_expands_a_millions_figure():
    assert parse_issues(SAMPLE)[1].mintage == 4_600_000


def test_captures_the_description_row():
    assert "Grace Kelly" in parse_issues(SAMPLE)[0].description


def test_volume_parsing_handles_the_page_s_variants():
    assert parse_volume("50 million coins")[0] == 50_000_000
    assert parse_volume("100,000 coins")[0] == 100_000
    assert parse_volume("1.5 million")[0] == 1_500_000
    assert parse_volume("30 000 coins")[0] == 30_000


def test_unreadable_volumes_return_none_rather_than_a_guess():
    """A fabricated mintage silently changes whether a coin is called rare."""
    assert parse_volume("unknown")[0] is None
    assert parse_volume("")[0] is None
    assert parse_volume("n/a")[0] is None


def test_a_bare_decimal_is_refused_as_ambiguous():
    # "1.5" alone could be 1.5 million or one and a half coins.
    assert parse_volume("1.5")[0] is None


def test_country_aliases_resolve_to_one_name():
    # The page has been edited by many hands; the same country appears under
    # several flag templates.
    assert COUNTRY_CODES["MCO"] == COUNTRY_CODES["MON"] == "Monaco"
    assert COUNTRY_CODES["DEU"] == COUNTRY_CODES["GER"] == "Germany"


def test_scarce_selects_on_mintage():
    issues = parse_issues(SAMPLE)
    picked = scarce(issues, threshold=200_000)
    assert [i.country for i in picked] == ["Monaco"]


def test_scarce_skips_issues_with_no_figure():
    issues = parse_issues(SAMPLE)
    issues[0].mintage = None
    assert scarce(issues, threshold=200_000) == []


def test_rarity_note_distinguishes_unknown_from_common():
    assert "unknown" in rarity_note(None, 200_000)
    assert "low mintage" in rarity_note(15_000, 200_000)
    assert "common" in rarity_note(5_000_000, 200_000)
