from __future__ import annotations

import numpy as np
import pytest

from euro_vision.metal import (
    METAL_GROUP,
    GroupModel,
    MetalFeatures,
    denominations_in,
    metal_features,
)
from euro_vision.stages.classify import nearest_denomination


def make_coin(bgr: tuple[int, int, int],
              centre_bgr: tuple[int, int, int] | None = None,
              size: int = 224) -> np.ndarray:
    """A flat disc, optionally with a differently coloured centre disc."""
    cv2 = pytest.importorskip("cv2")
    image = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(image, (size // 2, size // 2), size // 2 - 2, bgr, -1)
    if centre_bgr is not None:
        cv2.circle(image, (size // 2, size // 2), int(size * 0.28), centre_bgr, -1)
    return image


def test_group_restriction_widens_the_decision_boundary():
    """23.0 mm is nearest 1 EUR overall but nearest 20c among gold coins.

    This is the whole reason the alloy is worth predicting: across all eight
    denominations the boundaries sit 0.5 mm apart, which is inside the
    measurement's own spread.
    """
    assert nearest_denomination(23.0)[0] == 100
    assert nearest_denomination(23.0, "gold")[0] == 20
    assert nearest_denomination(23.0, "copper")[0] == 5


def test_group_restriction_leaves_more_margin():
    _, gap_any = nearest_denomination(21.0)
    _, gap_gold = nearest_denomination(21.0, "gold")
    # Fewer candidates can only ever leave the measurement further from the
    # nearest rival, never closer.
    assert gap_gold >= gap_any


def test_unknown_group_falls_back_to_all_denominations():
    assert nearest_denomination(23.0, None)[0] == nearest_denomination(23.0)[0]


def test_denominations_in_partitions_every_coin():
    covered = [d for group in ("copper", "gold", "bimetal")
               for d in denominations_in(group)]
    assert sorted(covered) == sorted(METAL_GROUP)


def test_copper_reads_redder_than_gold():
    """Ratio, not hue: this must hold even though both are warm-toned.

    Separating these two on hue alone was tried first and failed — under warm
    light copper spanned 12-21 and gold 13-20, almost total overlap.
    """
    copper = metal_features(make_coin((40, 90, 190)))
    gold = metal_features(make_coin((90, 165, 205)))
    assert copper.redness > gold.redness


def test_two_tone_detects_a_bimetallic_centre():
    plain = metal_features(make_coin((90, 165, 205)))
    bimetal = metal_features(make_coin((90, 165, 205), centre_bgr=(190, 190, 190)))
    assert bimetal.two_tone > plain.two_tone


def test_two_tone_ignores_a_brightness_gradient():
    """A centre that is merely lighter is not a second alloy.

    Each zone is normalised by its own brightness first, so glare in the middle
    of a coin must not read as a bimetallic boundary.

    The lit centre is the same colour scaled by 1.2, chosen so no channel
    reaches 255 — a clipped channel is a genuine hue change, not a brightness
    one, and would fail this for the right reason.
    """
    flat = metal_features(make_coin((90, 165, 205)))
    lit = metal_features(make_coin((90, 165, 205), centre_bgr=(108, 198, 246)))
    assert lit.two_tone == pytest.approx(flat.two_tone, abs=0.02)


def test_model_round_trips_through_a_dict():
    features = [MetalFeatures(1.4, 0.01), MetalFeatures(1.1, 0.005),
                MetalFeatures(1.1, 0.05)]
    groups = ["copper", "gold", "bimetal"]
    model = GroupModel.fit(features, groups)
    restored = GroupModel.from_dict(model.to_dict())
    for f in features:
        assert restored.predict(f)[0] == model.predict(f)[0]


def test_model_recovers_the_group_it_was_fitted_on():
    features, groups = [], []
    for redness, two_tone, group in [
        (1.40, 0.010, "copper"), (1.38, 0.008, "copper"),
        (1.15, 0.004, "gold"), (1.16, 0.003, "gold"),
        (1.11, 0.045, "bimetal"), (1.10, 0.040, "bimetal"),
    ]:
        features.append(MetalFeatures(redness, two_tone))
        groups.append(group)

    model = GroupModel.fit(features, groups)
    assert model.predict(MetalFeatures(1.39, 0.009))[0] == "copper"
    assert model.predict(MetalFeatures(1.15, 0.004))[0] == "gold"
    assert model.predict(MetalFeatures(1.10, 0.043))[0] == "bimetal"


def test_confidence_falls_when_a_coin_sits_between_two_alloys():
    features = [MetalFeatures(1.40, 0.01), MetalFeatures(1.10, 0.01)]
    model = GroupModel.fit(features, ["copper", "gold"])
    _, sure = model.predict(MetalFeatures(1.40, 0.01))
    _, unsure = model.predict(MetalFeatures(1.25, 0.01))
    assert unsure < sure


def test_features_survive_an_empty_crop():
    assert metal_features(np.zeros((0, 0, 3), np.uint8)) == MetalFeatures(0.0, 0.0)
    assert metal_features(None) == MetalFeatures(0.0, 0.0)
