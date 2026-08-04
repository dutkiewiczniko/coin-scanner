from __future__ import annotations

import numpy as np
import pytest

from euro_vision.errors import ErrorReport, _broadstrike_score, measure


def round_profile(radius: float = 100.0, rays: int = 64) -> np.ndarray:
    return np.full(rays, radius, dtype=np.float64)


def bite(profile: np.ndarray, arc: float = 0.20, depth: float = 0.30) -> np.ndarray:
    """Take a tapered bite out of the rim, as a clipped planchet has."""
    out = profile.copy()
    n = len(out)
    width = max(2, int(n * arc))
    for k in range(width):
        taper = np.sin(np.pi * (k + 0.5) / width)
        out[(n // 3 + k) % n] *= 1.0 - depth * taper
    return out


def overshoot(profile: np.ndarray, fraction: float = 0.45) -> np.ndarray:
    """Let rays run on into a touching neighbour, which reads too far."""
    out = profile.copy()
    out[: int(len(out) * fraction)] *= 1.35
    return out


def test_round_coin_is_not_flagged():
    assert measure(radial_profile=round_profile()).clip == 0.0


def test_clipped_planchet_is_detected():
    report = measure(radial_profile=bite(round_profile()))
    assert report.clip >= 0.5
    assert "clipped planchet" in " ".join(report.flagged(0.5))


def test_touching_neighbours_are_declined_not_flagged():
    """A blocked outline must report "cannot tell", never "clipped".

    Rays that meet a neighbouring coin overshoot, which drags the reference
    radius out and makes every honest ray look short. On a full tray this is the
    single biggest source of false alarms, so it is checked rather than scored.
    """
    report = measure(radial_profile=overshoot(round_profile()))
    assert report.clip == 0.0
    assert not report.valid
    assert "neighbour" in report.notes


def test_a_clip_survives_the_neighbour_guard():
    """The guard keys off the upper tail, so it must not swallow real clips."""
    report = measure(radial_profile=bite(round_profile(), arc=0.20, depth=0.30))
    assert report.valid
    assert report.clip >= 0.5


def test_scattered_short_rays_are_not_a_clip():
    """Shadow and noise hit isolated rays; a clip removes a continuous arc."""
    profile = round_profile()
    profile[::7] *= 0.75
    assert measure(radial_profile=profile).clip == 0.0


def test_broadstrike_only_counts_oversize():
    # A coin reading small is worn or short-measured, not struck without a collar.
    assert _broadstrike_score(24.0, 25.75) == 0.0
    assert _broadstrike_score(25.75, 25.75) == 0.0
    assert _broadstrike_score(27.5, 25.75) > 0.5


def test_broadstrike_ignores_measurement_noise():
    """0.45 mm of spread on a 23 mm coin is about 2%, so 2% cannot mean anything."""
    assert _broadstrike_score(23.25 * 1.015, 23.25) == 0.0


def test_crop_alone_cannot_judge_shape():
    """Without a tray-side profile, clip must stay silent rather than reassure.

    The normalised crop is masked to a circle, so its outline is round no matter
    what the coin was; a zero here would be an artefact of the mask.
    """
    crop = np.zeros((64, 64, 3), np.uint8)
    crop[16:48, 16:48] = 200
    assert measure(crop=crop).clip == 0.0


def test_empty_input_is_marked_invalid():
    report = measure()
    assert not report.valid
    assert report.worst == 0.0


def test_flagged_lists_each_fault_separately():
    report = ErrorReport(clip=0.9, off_centre=0.1, broadstrike=0.8)
    found = report.flagged(0.5)
    assert len(found) == 2
    assert report.worst == pytest.approx(0.9)
