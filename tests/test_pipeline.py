from __future__ import annotations

import json
from pathlib import Path

import pytest

from euro_vision.config import Config
from euro_vision.pipeline import Pipeline
from euro_vision.results import write_results
from euro_vision.stages.classify import DIAMETERS_MM
from euro_vision.types import format_denomination


@pytest.fixture
def config(tmp_path) -> Config:
    cfg = Config()
    cfg.output_dir = str(tmp_path / "out")
    cfg.segment.min_radius = 30
    cfg.segment.max_radius = 70
    cfg.segment.min_distance = 100
    return cfg


def test_pipeline_finds_every_coin(config, tray_image_path):
    result = Pipeline(config).run(tray_image_path)
    assert len(result.coins) == 6  # 2 rows x 3 cols from the synthetic tray


def test_normalisation_produces_uniform_crops(config, tray_image_path):
    result = Pipeline(config).run(tray_image_path)
    size = config.normalise.size
    for coin in result.coins:
        assert coin.normalised is not None
        assert coin.normalised.shape == (size, size, 3)


def test_coins_are_ordered_top_left_first(config, tray_image_path):
    result = Pipeline(config).run(tray_image_path)
    first, last = result.coins[0], result.coins[-1]
    assert first.detection.y <= last.detection.y
    assert [c.index for c in result.coins] == list(range(len(result.coins)))


def test_stub_classifier_leaves_denomination_unset(config, tray_image_path):
    result = Pipeline(config).run(tray_image_path)
    assert all(coin.denomination is None for coin in result.coins)


def _calibrate_to(config, image_path, cents: int) -> None:
    """Set pixels_per_mm so the detected coins measure as `cents`.

    Hough's radius estimate is a few percent under the true edge, so the scale
    has to come from what detection actually reports, not from the drawn radius.
    """
    measured = Pipeline(config).run(image_path).coins[0].detection.radius
    config.normalise.pixels_per_mm = (measured * 2) / DIAMETERS_MM[cents]


def test_diameter_classifier_matches_known_sizes(config, tray_image_path):
    _calibrate_to(config, tray_image_path, 200)
    config.classify.backend = "diameter"

    result = Pipeline(config).run(tray_image_path)
    assert result.coins
    for coin in result.coins:
        assert coin.denomination == 200
        assert coin.denomination_confidence > 0.5


def test_shortlist_backend_flags_only_watchlist_denominations(config, tray_image_path):
    _calibrate_to(config, tray_image_path, 200)
    config.classify.backend = "diameter"
    config.rare.backend = "shortlist"
    config.rare.denominations = [1, 2]  # deliberately excludes the 2 euro coins

    result = Pipeline(config).run(tray_image_path)
    assert result.flagged == []

    config.rare.denominations = [200]
    result = Pipeline(config).run(tray_image_path)
    assert len(result.flagged) == len(result.coins)


def test_rare_stage_rejects_unknown_backend(config):
    config.rare.backend = "nonsense"
    with pytest.raises(ValueError, match="unknown rare backend"):
        Pipeline(config)


def test_results_are_written_to_disk(config, tray_image_path):
    result = Pipeline(config).run(tray_image_path)
    written = write_results(result, config)

    payload = json.loads(Path(written["json"]).read_text(encoding="utf-8"))
    assert payload["coin_count"] == len(result.coins)
    assert Path(written["csv"]).exists()
    assert len(list(Path(written["crops"]).glob("*.png"))) == len(result.coins)


def test_missing_image_raises_clear_error(config):
    with pytest.raises(FileNotFoundError):
        Pipeline(config).run("does/not/exist.jpg")


@pytest.mark.parametrize(
    "cents,expected", [(1, "1c"), (50, "50c"), (100, "1 EUR"), (200, "2 EUR"), (None, None)]
)
def test_denomination_formatting(cents, expected):
    assert format_denomination(cents) == expected
