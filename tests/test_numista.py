from __future__ import annotations

import json

import pytest

from euro_vision.numista import (
    EURO_ISSUERS,
    Issue,
    Numista,
    NumistaAuthError,
    TypeSummary,
    mintage_outliers,
    variety_hint,
    _issue,
    _type_summary,
)

# The real shape of a /types search hit, trimmed to the fields that are read.
HAMBURG_MULE = {
    "id": 327881,
    "title": '2 Euros Bundesländer - "Hamburg" (Mule)',
    "issuer": {"code": "allemagne", "name": "Federal Republic of Germany"},
    "min_year": 2008,
    "max_year": 2008,
    "references": [{"catalogue": {"code": "KM"}, "number": "A261"}],
    "url": "https://en.numista.com/327881",
}
HAMBURG_CORRECT = {
    "id": 2994,
    "title": '2 Euros (Bundesländer - "Hamburg")',
    "issuer": {"code": "allemagne", "name": "Federal Republic of Germany"},
    "min_year": 2008,
    "max_year": 2008,
    "references": [{"catalogue": {"code": "KM"}, "number": "261"}],
    "url": "https://en.numista.com/2994",
}


def _summaries():
    return _type_summary(HAMBURG_MULE), _type_summary(HAMBURG_CORRECT)


# -- parsing --------------------------------------------------------------


def test_type_summary_flattens_issuer_and_references():
    mule = _type_summary(HAMBURG_MULE)
    assert mule.id == 327881
    assert mule.n_number == "N# 327881"
    assert mule.issuer == "Federal Republic of Germany"
    assert mule.references == ["KM# A261"]


def test_mintage_is_none_rather_than_a_guess_when_unreadable():
    # A wrong mintage decides whether a coin is called rare, so an unparseable
    # figure has to come back as absent rather than as zero.
    assert _issue({"id": 1, "year": 2008, "mintage": "unknown"}).mintage is None
    assert _issue({"id": 1, "year": 2008, "mintage": ""}).mintage is None
    assert _issue({"id": 1, "year": 2008}).mintage is None
    assert _issue({"id": 1, "year": 2008, "mintage": "600,000"}).mintage == 600_000


def test_issue_year_survives_a_string():
    assert _issue({"id": 1, "year": "2008", "mintage": 5}).year == 2008


# -- the variety rule -----------------------------------------------------


def test_finds_the_hamburg_mule_by_mintage_alone():
    """The rule has to work without reading the word "mule" anywhere.

    The comments are contributor-written and multilingual; the 600,000 against
    9,000,000 is not.
    """
    mule, correct = _summaries()
    found = mintage_outliers([
        (mule, [Issue(id=1, year=2008, mint_letter="F", mintage=600_000)]),
        (correct, [Issue(id=2, year=2008, mint_letter="F", mintage=9_000_000)]),
    ])
    assert [c.type.id for c in found] == [327881]
    assert found[0].mintage == 600_000
    assert found[0].sibling_mintage == 9_000_000
    assert found[0].ratio == pytest.approx(0.0667, abs=1e-3)


def test_ordinary_year_to_year_variation_is_not_flagged():
    mule, correct = _summaries()
    found = mintage_outliers([
        (mule, [Issue(id=1, year=2008, mint_letter=None, mintage=8_000_000)]),
        (correct, [Issue(id=2, year=2008, mint_letter=None, mintage=9_000_000)]),
    ])
    assert found == []


def test_a_lone_type_in_a_year_has_nothing_to_be_anomalous_against():
    mule, _ = _summaries()
    assert mintage_outliers([
        (mule, [Issue(id=1, year=2008, mint_letter=None, mintage=600)]),
    ]) == []


def test_issues_without_a_mintage_are_skipped_not_treated_as_zero():
    mule, correct = _summaries()
    found = mintage_outliers([
        (mule, [Issue(id=1, year=2008, mint_letter=None, mintage=None)]),
        (correct, [Issue(id=2, year=2008, mint_letter=None, mintage=9_000_000)]),
    ])
    assert found == []


def test_max_mintage_suppresses_a_small_share_of_an_enormous_run():
    """20% of a billion is still not a coin worth looking for."""
    mule, correct = _summaries()
    pair = [
        (mule, [Issue(id=1, year=2002, mint_letter=None, mintage=50_000_000)]),
        (correct, [Issue(id=2, year=2002, mint_letter=None, mintage=900_000_000)]),
    ]
    assert mintage_outliers(pair) == []
    assert len(mintage_outliers(pair, max_mintage=None)) == 1


def test_different_years_are_not_compared_with_each_other():
    mule, correct = _summaries()
    found = mintage_outliers([
        (mule, [Issue(id=1, year=2008, mint_letter=None, mintage=600_000)]),
        (correct, [Issue(id=2, year=2009, mint_letter=None, mintage=9_000_000)]),
    ])
    assert found == []


#: Verbatim from N# 327881, apostrophes included.
MULE_COMMENT = (
    "This Mule Coin was issued in 2008 by the Stuttgart Mint. It was supposed "
    "to have the 'New' Europe Map on the reverse. The Stuttgart Mint (F) "
    "accidentally issued 600,000 of them with the 'Old' Europe Map on the "
    "reverse."
)


def test_variety_hint_only_annotates_never_decides():
    hint = variety_hint(MULE_COMMENT)
    assert "mule" in hint
    # Punctuation is flattened, so "the 'Old' Europe Map" still matches. A plain
    # substring test misses this by one apostrophe.
    assert "old europe map" in hint
    assert "accidental" in hint
    assert variety_hint("Struck at the Karlsruhe mint.") == ""
    assert variety_hint("") == ""


# -- client plumbing ------------------------------------------------------


def test_missing_key_names_the_variable_and_where_to_get_one(tmp_path, monkeypatch):
    monkeypatch.delenv("NUMISTA_API_KEY", raising=False)
    client = Numista(api_key=None, cache_dir=tmp_path)
    with pytest.raises(NumistaAuthError) as exc:
        client.check()
    assert "NUMISTA_API_KEY" in str(exc.value)
    assert "numista.com" in str(exc.value)


@pytest.fixture(autouse=True)
def _no_ambient_key(monkeypatch):
    """Keep a real key in the environment out of the tests.

    `Numista(api_key=None)` falls back to $NUMISTA_API_KEY, so on a machine
    where the key is set the "keyless" tests below quietly acquire one and
    either hit the network or stop testing what they claim to.
    """
    monkeypatch.delenv("NUMISTA_API_KEY", raising=False)


def test_a_cached_response_costs_no_quota_and_needs_no_key(tmp_path):
    """The free tier is 2,000 requests a month, so a repeat must not spend one."""
    client = Numista(api_key="k", cache_dir=tmp_path)
    cache = client._cache_path("/types/327881", {"lang": "en"})
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"id": 327881}), encoding="utf-8")

    keyless = Numista(api_key=None, cache_dir=tmp_path)
    assert keyless.get_type(327881) == {"id": 327881}
    assert keyless.requests_made == 0


def test_expired_cache_is_ignored(tmp_path, monkeypatch):
    import os
    import time

    client = Numista(api_key=None, cache_dir=tmp_path, cache_days=30)
    cache = client._cache_path("/types/1", {"lang": "en"})
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"id": 1}), encoding="utf-8")
    old = time.time() - 40 * 86400
    os.utime(cache, (old, old))

    # Falls through to the network, which without a key raises rather than
    # silently serving stale data.
    with pytest.raises(NumistaAuthError):
        client.get_type(1)


def test_cache_key_separates_different_queries(tmp_path):
    client = Numista(api_key="k", cache_dir=tmp_path)
    a = client._cache_path("/types", {"q": "hamburg", "lang": "en"})
    b = client._cache_path("/types", {"q": "bremen", "lang": "en"})
    assert a != b
    assert a.name.startswith("types_") and b.name.startswith("types_")


def test_euro_issuer_codes_cover_the_states_that_strike_commemoratives():
    for country in ("germany", "monaco", "san_marino", "vatican", "croatia"):
        assert country in EURO_ISSUERS


# -- the real record shape ------------------------------------------------

#: Verbatim from the live API for N# 327881. The comment field is `comments`,
#: plural, and is HTML — reading `comment` returns nothing at all.
LIVE_COMMENTS = (
    'This <span style="font-weight:bold;">Mule Coin</span> was issued in 2008 '
    'by the Stuttgart Mint. It was supposed to have the &quot;New&quot; Europe '
    'Map on the reverse.<br />\r\n<a href="https://en.numista.com/x.jpg">'
    '<img src="https://en.numista.com/y.jpg" alt="" /></a>'
)


def test_plain_text_strips_tags_entities_and_embedded_images():
    from euro_vision.numista import plain_text

    out = plain_text(LIVE_COMMENTS)
    assert "<span" not in out and "<img" not in out and "<br" not in out
    assert "&quot;" not in out
    assert '"New" Europe Map' in out
    assert out.startswith("This Mule Coin was issued in 2008")


def test_plain_text_on_empty_input():
    from euro_vision.numista import plain_text

    assert plain_text("") == ""
    assert plain_text(None) == ""


def test_variety_words_survive_the_html():
    """The mule is only findable in the comments after the tags come off."""
    from euro_vision.numista import plain_text

    assert "mule" in variety_hint(plain_text(LIVE_COMMENTS))


def test_denomination_comes_from_the_structured_value_not_the_title():
    from euro_vision.numista import denomination_cents

    assert denomination_cents({
        "value": {"numeric_value": 2, "currency": {"name": "Euro"}}
    }) == 200
    assert denomination_cents({
        "value": {"numeric_value": 0.5, "currency": {"name": "Euro"}}
    }) == 50
    assert denomination_cents({
        "value": {"numeric_value": 0.01, "currency": {"name": "Euro"}}
    }) == 1


def test_non_euro_denomination_is_none_so_a_stray_search_hit_is_dropped():
    from euro_vision.numista import denomination_cents

    assert denomination_cents({
        "value": {"numeric_value": 2, "currency": {"name": "German mark"}}
    }) is None
    assert denomination_cents({"value": {}}) is None
    assert denomination_cents({}) is None


def test_variant_matches_the_rare_coin_column_values():
    from euro_vision.numista import variant_of

    assert variant_of("Circulating commemorative coins", True) == "error"
    assert variant_of("Standard circulation coins", True) == "error"
    assert variant_of("Circulating commemorative coins", False) == "commemorative"
    assert variant_of("Standard circulation coins", False) == "standard"
    assert variant_of(None, False) == "standard"


def test_a_type_level_error_note_does_not_make_every_year_an_error():
    """Finland's N# 95 says "Die error on the 2000 coin" at type level.

    That is a claim about one year. Deciding `variant` from it stored nine
    ordinary years as errors, and the database is what the rare stage acts on.
    """
    from euro_vision.numista import variant_of, variety_hint

    type_comments = "Die error on the 2000 coin\nDie errors on 2001 coin"
    assert variety_hint(type_comments)          # the type does mention errors
    assert not variety_hint("")                 # ...but this year's issue does not
    assert variant_of("Standard circulation coins",
                      bool(variety_hint(""))) == "standard"


def test_issue_quality_separates_circulation_from_collector_products():
    from euro_vision.numista import issue_quality

    # Verbatim comments from the live API.
    assert issue_quality("") == "circulation"
    assert issue_quality(None) == "circulation"
    assert issue_quality("BU set") == "set"
    assert issue_quality("Proof set") == "proof"
    assert issue_quality("BU; Mint of Finland") == "bu"
    assert issue_quality("Error: old map style on reverse (Mule)") == "circulation"
    assert issue_quality("Mule (Wrong Reverse)") == "circulation"


def test_grace_kelly_survives_the_quality_filter():
    """Monaco 2007, 20,001 struck, comment exactly "BU".

    The most sought-after euro coin there is. A blanket exclude on "BU" would
    drop it, which is why `set` is tested before `bu` rather than the reverse.
    """
    from euro_vision.numista import CIRCULATING_QUALITIES, issue_quality

    assert issue_quality("BU") == "bu"
    assert issue_quality("BU") in CIRCULATING_QUALITIES
    # ...while the set version of the same denomination is dropped.
    assert issue_quality("BU set") not in CIRCULATING_QUALITIES
    assert issue_quality("Proof set") not in CIRCULATING_QUALITIES


def test_bu_is_matched_as_a_word_not_a_substring():
    from euro_vision.numista import issue_quality

    # "Bundesländer" and "debut" must not read as brilliant uncirculated.
    assert issue_quality("Bundeslander series") == "circulation"
    assert issue_quality("debut issue") == "circulation"


def test_a_documented_error_with_no_mintage_is_still_a_variety():
    """Italy's 2002 1c "Error - missing R mintmark" carries no figure at all.

    Nobody counted them. Skipping unmintaged issues would drop exactly the
    coins this is looking for.
    """
    assert variety_hint("Error - missing R mintmark")
    assert _issue({"id": 1, "year": 2002, "comment": "Error - missing R mintmark"}
                  ).mintage is None


def test_target_csv_skips_comments_and_blank_queries(tmp_path):
    from euro_vision.cli import _read_targets

    path = tmp_path / "targets.csv"
    path.write_text(
        "# a comment line\n"
        "denomination,issuer,query,note\n"
        "200,allemagne,2 euro Hamburg mule,KM# A261\n"
        "# another comment\n"
        ",,2 euro mule,any issuer\n"
        "200,monaco,,no query so skipped\n",
        encoding="utf-8",
    )
    rows = _read_targets(path)
    assert len(rows) == 2
    assert rows[0] == {
        "denomination": 200, "issuer": "allemagne",
        "query": "2 euro Hamburg mule", "note": "KM# A261",
    }
    assert rows[1]["denomination"] is None and rows[1]["issuer"] == ""


def test_denomination_read_from_a_numista_title():
    from euro_vision.cli import _denomination_from_title

    assert _denomination_from_title('2 Euros Bundesländer - "Hamburg"') == 200
    assert _denomination_from_title("1 Euro (1st map)") == 100
    assert _denomination_from_title("50 Euro Cents") == 50
    assert _denomination_from_title("2 Cents (2nd map)") == 2
    # Unreadable gives 0, which is visibly wrong, rather than a plausible guess
    # that would quietly drop the coin out of the rare stage's watchlist.
    assert _denomination_from_title("Medal - Hamburg") == 0
