from __future__ import annotations

import cv2
import numpy as np
import pytest
from conftest import TRAY_H_MM, TRAY_W_MM, apply_perspective, make_marked_tray

from euro_vision.config import CalibrateConfig, Config
from euro_vision.pipeline import Pipeline
from euro_vision.stages import CalibrationError, compute_calibration
from euro_vision.stages.classify import DIAMETERS_MM

# One coin of each denomination, spread across the tray so any scale error that
# varies with position shows up. The layout is deliberately asymmetric about both
# axes: a symmetric one would map onto itself under a mirror, letting a wrong
# side-b transform pass the pairing test unnoticed.
COINS = [
    (25.0, 30.0, DIAMETERS_MM[1]),
    (85.0, 30.0, DIAMETERS_MM[5]),
    (155.0, 30.0, DIAMETERS_MM[10]),
    (35.0, 80.0, DIAMETERS_MM[20]),
    (100.0, 80.0, DIAMETERS_MM[50]),
    (160.0, 80.0, DIAMETERS_MM[100]),
    (70.0, 125.0, DIAMETERS_MM[200]),
]


@pytest.fixture
def calibrate_config() -> CalibrateConfig:
    return CalibrateConfig(
        enabled=True,
        tray_width_mm=TRAY_W_MM,
        tray_height_mm=TRAY_H_MM,
        pixels_per_mm=8.0,
        min_marker_area=40,
    )


@pytest.fixture
def marked_tray_path(tmp_path) -> str:
    image, _ = make_marked_tray(COINS, pixels_per_mm=6.0)
    path = tmp_path / "tray_marked.png"
    cv2.imwrite(str(path), image)
    return str(path)


@pytest.fixture
def warped_tray_path(tmp_path) -> str:
    image, _ = make_marked_tray(COINS, pixels_per_mm=6.0)
    path = tmp_path / "tray_warped.png"
    cv2.imwrite(str(path), apply_perspective(image))
    return str(path)


def _pipeline_config(calibrate_config, tmp_path) -> Config:
    config = Config(calibrate=calibrate_config)
    config.output_dir = str(tmp_path / "out")
    config.classify.backend = "diameter"
    return config


# -- marker detection -----------------------------------------------------


def test_finds_all_four_markers(calibrate_config, marked_tray_path):
    from euro_vision.pipeline import load_image

    calibration = compute_calibration(load_image(marked_tray_path), calibrate_config)
    assert [m.colour for m in calibration.markers] == [
        "red", "green", "blue", "magenta"
    ]


def test_square_on_image_reports_no_tilt(calibrate_config, marked_tray_path):
    from euro_vision.pipeline import load_image

    calibration = compute_calibration(load_image(marked_tray_path), calibrate_config)
    assert calibration.tilt_ratio() == pytest.approx(1.0, abs=0.01)


def test_warped_image_reports_tilt(calibrate_config, warped_tray_path):
    from euro_vision.pipeline import load_image

    calibration = compute_calibration(load_image(warped_tray_path), calibrate_config)
    assert calibration.tilt_ratio() > 1.02


def test_coins_matching_marker_colours_do_not_hijack_calibration(calibrate_config):
    """Copper coins fall near the red band and gold coins near yellow.

    A coin is far larger than a corner dot, so choosing the biggest blob of each
    colour would pick the coin and silently produce a wrong calibration. The
    corners must win on position instead.
    """
    from euro_vision.pipeline import load_image

    image, offset = make_marked_tray(COINS, pixels_per_mm=6.0)
    reference = compute_calibration(image, calibrate_config)

    scale = 6.0
    # Big red and green discs well inside the marker rectangle, each far larger
    # than a corner dot.
    cv2.circle(image, (int(70 * scale + offset), int(60 * scale + offset)),
               int(13 * scale), (0, 0, 220), -1)
    cv2.circle(image, (int(130 * scale + offset), int(100 * scale + offset)),
               int(13 * scale), (0, 200, 0), -1)

    hijacked = compute_calibration(image, calibrate_config)

    for before, after in zip(reference.markers, hijacked.markers):
        assert after.colour == before.colour
        assert after.x == pytest.approx(before.x, abs=2.0)
        assert after.y == pytest.approx(before.y, abs=2.0)


def test_background_impostors_defeat_corner_selection(calibrate_config):
    """Documents a real limitation, so the workaround is not mistaken for a fix.

    Largest-quadrilateral selection assumes impostor blobs lie INSIDE the true
    corners. That holds for coins on the tray, but not for the surface the tray
    sits on. Measured on real photos: a yellow marker on an oak table lost to
    wood grain once enough candidates were considered.

    The mitigation is a marker colour absent from the background, not a cleverer
    search — this test exists to keep that distinction honest.
    """
    image, offset = make_marked_tray(COINS, pixels_per_mm=6.0)
    reference = compute_calibration(image, calibrate_config)
    true_magenta = next(m for m in reference.markers if m.colour == "magenta")

    # A magenta speck OUTSIDE the tray, beyond the bottom-left marker, where it
    # enlarges the quadrilateral rather than shrinking it.
    h, w = image.shape[:2]
    cv2.circle(image, (8, h - 8), 6, (200, 0, 200), -1)
    contaminated = compute_calibration(image, calibrate_config)
    picked = next(m for m in contaminated.markers if m.colour == "magenta")

    drift = np.hypot(picked.x - true_magenta.x, picked.y - true_magenta.y)
    assert drift > 50, (
        "expected background contamination to capture the corner; if this now "
        "passes, corner selection has been made robust and the docs in "
        "stages/calibrate.py should be updated"
    )


def test_tiny_marker_quadrilateral_is_rejected(calibrate_config):
    """Four dots clustered in the middle of the frame are not tray corners."""
    image = np.full((900, 1200, 3), 90, dtype=np.uint8)
    for offset_x, colour in enumerate(
        [(0, 0, 220), (0, 200, 0), (220, 0, 0), (200, 0, 200)]
    ):
        cv2.circle(image, (580 + (offset_x % 2) * 40, 440 + (offset_x // 2) * 40),
                   10, colour, -1)

    with pytest.raises(CalibrationError, match="too little to be the tray"):
        compute_calibration(image, calibrate_config)


def test_missing_marker_gives_actionable_error(calibrate_config, tmp_path):
    image, _ = make_marked_tray(COINS, pixels_per_mm=6.0)
    image, offset = make_marked_tray(COINS, pixels_per_mm=6.0)
    # Cover the magenta corner, as a coin resting on it would.
    cv2.circle(image, (int(offset), int(150 * 6.0 + offset)), 40, (90, 90, 90), -1)

    with pytest.raises(CalibrationError, match="no magenta corner marker"):
        compute_calibration(image, calibrate_config)


def test_side_a_rejects_an_anticlockwise_photo(calibrate_config):
    """A side-b photo passed as side a would mirror every coin design."""
    image, _ = make_marked_tray(COINS, pixels_per_mm=6.0)

    with pytest.raises(CalibrationError, match="anticlockwise in this side-a"):
        compute_calibration(cv2.flip(image, 1), calibrate_config, "a")


def test_side_b_rejects_a_clockwise_photo(calibrate_config):
    """Two trays winding the same way means one was not marked by touch."""
    image, _ = make_marked_tray(COINS, pixels_per_mm=6.0)

    with pytest.raises(CalibrationError, match="wind the same way"):
        compute_calibration(image, calibrate_config, "b")


def test_side_b_accepts_the_flipped_tray(calibrate_config):
    """Flipping the sandwich mirrors the layout; that is the expected side-b photo."""
    image, _ = make_marked_tray(COINS, pixels_per_mm=6.0)
    calibration = compute_calibration(cv2.flip(image, 1), calibrate_config, "b")

    # Reversed reading order, and orientation-preserving so designs are not
    # mirrored — a negative determinant would flip every coin.
    assert [m.colour for m in calibration.markers] == [
        "magenta", "blue", "green", "red"
    ]
    assert np.linalg.det(calibration.homography) > 0


def test_paired_coins_land_together_in_the_reference_frame(calibrate_config, tmp_path):
    """The whole point of the marking scheme: a coin keeps its position.

    A side-b photo is geometrically the side-a photo mirrored, since flipping the
    sandwich mirrors the layout. After rectifying each with its own side and
    mapping into the reference frame, a coin must land in the same place.
    """
    image, _ = make_marked_tray(COINS, pixels_per_mm=6.0)
    path_a = tmp_path / "tray_009_a.png"
    path_b = tmp_path / "tray_009_b.png"
    cv2.imwrite(str(path_a), image)
    cv2.imwrite(str(path_b), cv2.flip(image, 1))

    config = _pipeline_config(calibrate_config, tmp_path)
    pipeline = Pipeline(config)

    result_a = pipeline.run(str(path_a))
    result_b = pipeline.run(str(path_b))
    assert result_a.meta["side"] == "a"
    assert result_b.meta["side"] == "b"

    cal_b = compute_calibration(
        cv2.imread(str(path_b)), calibrate_config, "b"
    )

    def key(point: tuple[float, float]) -> tuple[int, int]:
        return int(round(point[0] / 10)), int(round(point[1] / 10))

    points_a = sorted(key((c.detection.x, c.detection.y)) for c in result_a.coins)
    points_b = sorted(
        key(cal_b.to_reference_frame(c.detection.x, c.detection.y))
        for c in result_b.coins
    )

    assert len(points_a) == len(COINS)
    for (ax, ay), (bx, by) in zip(points_a, points_b):
        assert abs(ax - bx) <= 1
        assert abs(ay - by) <= 1


def test_duplicate_marker_colours_rejected():
    from euro_vision.stages import CalibrateStage

    config = CalibrateConfig(enabled=True, markers=["red", "red", "blue", "green"])
    with pytest.raises(ValueError, match="all be different"):
        CalibrateStage(config)


def test_unknown_marker_colour_rejected():
    from euro_vision.stages import CalibrateStage

    config = CalibrateConfig(enabled=True, markers=["red", "puce", "blue", "green"])
    with pytest.raises(ValueError, match="unknown marker colour"):
        CalibrateStage(config)


# -- rectification and measurement ---------------------------------------


def test_rectified_output_has_expected_metric_size(
    calibrate_config, marked_tray_path, tmp_path
):
    config = _pipeline_config(calibrate_config, tmp_path)
    result = Pipeline(config).run(marked_tray_path)

    assert result.meta["image_size"] == [
        int(TRAY_W_MM * calibrate_config.pixels_per_mm),
        int(TRAY_H_MM * calibrate_config.pixels_per_mm),
    ]
    assert result.meta["pixels_per_mm"] == calibrate_config.pixels_per_mm


def test_scale_correction_changes_measured_diameters(
    calibrate_config, marked_tray_path, tmp_path
):
    """A 2% correction must move every measured diameter by 2%.

    It must not change the rectified output size, which is only a rendering
    choice — if it did, the two effects would cancel and the knob would do
    nothing.
    """
    baseline = Pipeline(_pipeline_config(calibrate_config, tmp_path)).run(
        marked_tray_path
    )

    corrected_config = _pipeline_config(calibrate_config, tmp_path)
    corrected_config.calibrate.scale_correction = 1.02
    corrected = Pipeline(corrected_config).run(marked_tray_path)

    assert corrected.meta["image_size"] == baseline.meta["image_size"]

    before = sorted(c.diameter_mm for c in baseline.coins)
    after = sorted(c.diameter_mm for c in corrected.coins)
    assert len(before) == len(after)
    for b, a in zip(before, after):
        assert a == pytest.approx(b * 1.02, rel=0.005)


def test_segment_radii_derived_from_millimetres(calibrate_config, tmp_path):
    config = _pipeline_config(calibrate_config, tmp_path)
    Pipeline(config)  # derivation happens at construction

    scale = calibrate_config.pixels_per_mm
    assert config.segment.min_radius == int(15.0 * scale / 2)
    assert config.segment.max_radius == int(27.0 * scale / 2)
    assert config.normalise.pixels_per_mm == scale


def test_denominations_recovered_from_square_on_photo(
    calibrate_config, marked_tray_path, tmp_path
):
    config = _pipeline_config(calibrate_config, tmp_path)
    result = Pipeline(config).run(marked_tray_path)

    assert len(result.coins) == len(COINS)
    found = sorted(coin.denomination for coin in result.coins)
    assert found == sorted(int(d) for d in [1, 5, 10, 20, 50, 100, 200])


def test_denominations_survive_a_tilted_photo(
    calibrate_config, warped_tray_path, tmp_path
):
    """Rectification is the whole point: an off-axis shot must still measure true."""
    config = _pipeline_config(calibrate_config, tmp_path)
    result = Pipeline(config).run(warped_tray_path)

    assert len(result.coins) == len(COINS)
    found = sorted(coin.denomination for coin in result.coins)
    assert found == sorted([1, 5, 10, 20, 50, 100, 200])


def test_measured_diameters_are_accurate(
    calibrate_config, marked_tray_path, tmp_path
):
    config = _pipeline_config(calibrate_config, tmp_path)
    result = Pipeline(config).run(marked_tray_path)

    expected = sorted(d for _, _, d in COINS)
    measured = sorted(coin.diameter_mm for coin in result.coins)
    for want, got in zip(expected, measured):
        # The tightest gap between any two Euro denominations is 1 mm, so the
        # measurement has to stay comfortably inside half of that.
        assert got == pytest.approx(want, abs=0.3)


def test_refinement_recovers_from_an_inner_ring(calibrate_config, tmp_path):
    """Hough can lock onto a coin's inner circle; the measurement must not.

    Bimetallic 1 and 2 euro coins have a strong ring/centre boundary at roughly
    0.75 of their radius. Detecting that instead of the rim would read a 2 euro
    coin as about 19 mm — a 10 cent piece.
    """
    from euro_vision.pipeline import load_image
    from euro_vision.stages.segment import SegmentStage
    from euro_vision.types import Detection

    image, _ = make_marked_tray([(100.0, 75.0, DIAMETERS_MM[200])], pixels_per_mm=6.0)
    path = tmp_path / "bimetal.png"
    cv2.imwrite(str(path), image)

    config = _pipeline_config(calibrate_config, tmp_path)
    calibration = compute_calibration(load_image(str(path)), calibrate_config)
    rectified = cv2.warpPerspective(
        load_image(str(path)),
        calibration.homography,
        (calibration.width_px, calibration.height_px),
    )

    from euro_vision.pipeline import _apply_calibrated_scale

    _apply_calibrated_scale(config)
    stage = SegmentStage(config.segment)

    scale = calibrate_config.pixels_per_mm
    true_radius = DIAMETERS_MM[200] / 2 * scale
    centre_x = int(100.0 * scale)
    centre_y = int(75.0 * scale)

    # Simulate the detector having found the inner ring rather than the rim.
    inner = Detection(x=centre_x, y=centre_y, radius=int(true_radius * 0.75))
    refined = stage._measure_only(rectified, inner)

    assert refined.radius == pytest.approx(true_radius, abs=0.3 * scale / 2)
