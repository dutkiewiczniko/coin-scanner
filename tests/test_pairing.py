from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import TRAY_H_MM, TRAY_W_MM, make_marked_tray

from euro_vision.config import CalibrateConfig, Config
from euro_vision.pairing import compose_coin, pair_batch, write_batch
from euro_vision.pipeline import Pipeline
from euro_vision.stages.classify import DIAMETERS_MM

COINS = [
    (30.0, 30.0, DIAMETERS_MM[1]),
    (90.0, 32.0, DIAMETERS_MM[200]),
    (150.0, 30.0, DIAMETERS_MM[10]),
    (40.0, 90.0, DIAMETERS_MM[50]),
    (110.0, 95.0, DIAMETERS_MM[100]),
]


@pytest.fixture
def calibrate_config() -> CalibrateConfig:
    return CalibrateConfig(
        enabled=True,
        tray_width_mm=TRAY_W_MM,
        tray_height_mm=TRAY_H_MM,
        pixels_per_mm=8.0,
    )


@pytest.fixture
def config(calibrate_config, tmp_path) -> Config:
    cfg = Config(calibrate=calibrate_config)
    cfg.output_dir = str(tmp_path / "out")
    cfg.classify.backend = "diameter"
    return cfg


def _write_pair(tmp_path, coins=COINS, jitter=None):
    """Write a batch's two photos.

    The side-b photo is the side-a layout mirrored, which is what flipping the
    sandwich physically does. `jitter` shifts coins in side b, as real coins
    shift a little when the trays are flipped.
    """
    image_a, _ = make_marked_tray(coins, pixels_per_mm=6.0)
    moved = coins if jitter is None else [
        (x + dx, y + dy, d) for (x, y, d), (dx, dy) in zip(coins, jitter)
    ]
    image_b, _ = make_marked_tray(moved, pixels_per_mm=6.0)

    path_a = tmp_path / "77_a.png"
    path_b = tmp_path / "77_b.png"
    cv2.imwrite(str(path_a), image_a)
    cv2.imwrite(str(path_b), cv2.flip(image_b, 1))
    return path_a, path_b


def _run(config, tmp_path, **kwargs):
    path_a, path_b = _write_pair(tmp_path, **kwargs)
    pipeline = Pipeline(config)
    return pair_batch(
        "77",
        pipeline.run(str(path_a), side="a"),
        pipeline.run(str(path_b), side="b"),
        config,
    )


def test_every_coin_is_matched(config, tmp_path):
    batch = _run(config, tmp_path)
    assert len(batch.coins) == len(COINS)
    assert len(batch.paired) == len(COINS)
    assert batch.unpaired == []


def test_matched_faces_agree_on_position_and_size(config, tmp_path):
    batch = _run(config, tmp_path)
    for coin in batch.paired:
        assert coin.offset_mm < 1.5
        assert coin.diameter_delta_mm < 0.5


def test_matching_survives_coins_shifting_during_the_flip(config, tmp_path):
    """Coins move a little when the trays are inverted; matching must cope."""
    jitter = [(2.0, -1.5), (-1.0, 2.0), (1.5, 1.0), (-2.0, -1.0), (1.0, -2.0)]
    batch = _run(config, tmp_path, jitter=jitter)

    assert len(batch.paired) == len(COINS)
    for coin in batch.paired:
        # Same denomination on both faces is the real check that the right two
        # faces were joined.
        assert coin.face_a.denomination == coin.face_b.denomination


def test_ids_run_in_reading_order(config, tmp_path):
    batch = _run(config, tmp_path)
    ids = [c.coin_id for c in batch.coins]
    assert ids == sorted(ids)
    assert ids[0] == "77-001"
    assert all(c.batch == "77" for c in batch.coins)

    top_row = [c for c in batch.coins if c.y_mm < 60]
    assert [c.index for c in top_row] == sorted(c.index for c in top_row)
    assert [c.x_mm for c in top_row] == sorted(c.x_mm for c in top_row)


def test_a_coin_missing_from_one_side_is_reported_not_invented(config, tmp_path):
    """A missed detection must surface as unpaired, never as a wrong match."""
    image_a, _ = make_marked_tray(COINS, pixels_per_mm=6.0)
    image_b, _ = make_marked_tray(COINS[:-1], pixels_per_mm=6.0)  # one coin absent
    path_a, path_b = tmp_path / "77_a.png", tmp_path / "77_b.png"
    cv2.imwrite(str(path_a), image_a)
    cv2.imwrite(str(path_b), cv2.flip(image_b, 1))

    pipeline = Pipeline(config)
    batch = pair_batch(
        "77",
        pipeline.run(str(path_a), side="a"),
        pipeline.run(str(path_b), side="b"),
        config,
    )

    assert len(batch.paired) == len(COINS) - 1
    assert len(batch.unpaired) == 1
    assert batch.unpaired[0].faces_found == "a"


def test_distant_coins_are_not_matched(config, tmp_path):
    """Beyond the offset limit a pairing is a guess, so it must be refused."""
    config.pair.max_offset_mm = 1.0
    config.pair.auto_align = False  # otherwise the shift below is corrected away
    jitter = [(20.0, 20.0)] * len(COINS)
    batch = _run(config, tmp_path, jitter=jitter)

    assert batch.paired == []
    assert len(batch.unpaired) == 2 * len(COINS)


def test_auto_align_recovers_a_constant_offset(config, tmp_path):
    """The trays nest rather than meeting corner to corner, so their frames sit
    at a fixed offset. It must be recovered from the coins, not measured."""
    config.pair.max_offset_mm = 2.0  # far tighter than the offset applied
    shift = [(9.0, -7.0)] * len(COINS)

    config.pair.auto_align = False
    assert _run(config, tmp_path, jitter=shift).paired == []

    config.pair.auto_align = True
    batch = _run(config, tmp_path, jitter=shift)
    assert len(batch.paired) == len(COINS)

    dx, dy = batch.meta["alignment_mm"]
    assert dx == pytest.approx(-9.0, abs=1.0)
    assert dy == pytest.approx(7.0, abs=1.0)


def test_auto_align_ignores_an_unconvincing_shift(config, tmp_path):
    """With too little agreement the estimate is noise and must not be applied."""
    config.pair.align_min_support = 0.95
    jitter = [(1.0, 0.0), (-8.0, 6.0), (7.0, -9.0), (-3.0, 8.0), (9.0, 2.0)]
    batch = _run(config, tmp_path, jitter=jitter)
    assert batch.meta["alignment_mm"] == [0.0, 0.0]


def test_diameter_disagreement_blocks_a_match(config, tmp_path):
    config.pair.max_diameter_delta_mm = 0.05
    # Same positions, but every coin is a different denomination on side b.
    swapped = [(x, y, DIAMETERS_MM[200] if d != DIAMETERS_MM[200]
                else DIAMETERS_MM[1]) for x, y, d in COINS]
    image_a, _ = make_marked_tray(COINS, pixels_per_mm=6.0)
    image_b, _ = make_marked_tray(swapped, pixels_per_mm=6.0)
    path_a, path_b = tmp_path / "77_a.png", tmp_path / "77_b.png"
    cv2.imwrite(str(path_a), image_a)
    cv2.imwrite(str(path_b), cv2.flip(image_b, 1))

    pipeline = Pipeline(config)
    batch = pair_batch(
        "77",
        pipeline.run(str(path_a), side="a"),
        pipeline.run(str(path_b), side="b"),
        config,
    )
    assert batch.paired == []


def test_paired_diameter_averages_both_faces(config, tmp_path):
    batch = _run(config, tmp_path)
    for coin in batch.paired:
        expected = (coin.face_a.diameter_mm + coin.face_b.diameter_mm) / 2
        assert coin.diameter_mm == pytest.approx(expected, abs=0.01)


def test_denomination_follows_the_combined_diameter(config, tmp_path):
    """The reported size and label must come from the same measurement."""
    from euro_vision.stages.classify import DIAMETERS_MM, nearest_denomination

    batch = _run(config, tmp_path)
    for coin in batch.paired:
        assert coin.denomination == nearest_denomination(coin.diameter_mm)[0]
        reference = DIAMETERS_MM[coin.denomination]
        assert abs(coin.diameter_mm - reference) < 0.6


def test_composite_shows_both_faces(config, tmp_path):
    batch = _run(config, tmp_path)
    image = compose_coin(batch.coins[0], config.pair)
    size = config.pair.composite_face_px
    assert image.shape[1] > size * 2  # two faces side by side
    assert image.shape[0] > size      # plus caption space


def test_manifest_records_every_coin(config, tmp_path):
    batch = _run(config, tmp_path)
    written = write_batch(batch, config)

    payload = json.loads(Path(written["json"]).read_text(encoding="utf-8"))
    assert payload["batch"] == "77"
    assert payload["total"] == len(COINS)
    assert payload["paired"] == len(COINS)

    with open(written["csv"], newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(COINS)
    assert {r["coin_id"] for r in rows} == {c.coin_id for c in batch.coins}
    # The manifest has to locate a coin after the tray is emptied.
    assert all(r["x_mm"] and r["y_mm"] for r in rows)

    images = list(Path(written["coins"]).glob("*.png"))
    assert len(images) == len(COINS)


def test_unpaired_coins_are_marked_in_the_filename(config, tmp_path):
    image_a, _ = make_marked_tray(COINS, pixels_per_mm=6.0)
    image_b, _ = make_marked_tray(COINS[:-1], pixels_per_mm=6.0)
    path_a, path_b = tmp_path / "77_a.png", tmp_path / "77_b.png"
    cv2.imwrite(str(path_a), image_a)
    cv2.imwrite(str(path_b), cv2.flip(image_b, 1))

    pipeline = Pipeline(config)
    batch = pair_batch(
        "77",
        pipeline.run(str(path_a), side="a"),
        pipeline.run(str(path_b), side="b"),
        config,
    )
    written = write_batch(batch, config)
    names = [p.name for p in Path(written["coins"]).glob("*.png")]
    assert sum("_unpaired" in n for n in names) == 1
