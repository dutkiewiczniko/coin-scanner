"""Pipeline orchestration — tray image in, ScanResult out."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .stages import (
    SIDE_A,
    SIDE_B,
    CalibrateStage,
    ClassifyStage,
    NormaliseStage,
    RareStage,
    SegmentStage,
    Stage,
)
from .types import ScanResult

log = logging.getLogger(__name__)


class Pipeline:
    """The stages, wired together."""

    def __init__(self, config: Config):
        self.config = config

        self.stages: list[Stage] = []
        if config.calibrate.enabled:
            self.stages.append(CalibrateStage(config.calibrate))
            _apply_calibrated_scale(config)

        self.stages += [
            SegmentStage(config.segment),
            NormaliseStage(config.normalise),
            ClassifyStage(config.classify),
            RareStage(config.rare),
        ]

    def run(self, image_path: str | Path, side: str | None = None) -> ScanResult:
        image_path = Path(image_path)
        image = load_image(image_path)

        result = ScanResult(source=str(image_path))
        result.meta["side"] = side or side_from_path(image_path)
        result.meta["image"] = image
        result.meta["image_size"] = [image.shape[1], image.shape[0]]
        result.meta["timings"] = {}

        for stage in self.stages:
            started = time.perf_counter()
            result = stage.run(result)
            elapsed = time.perf_counter() - started
            result.meta["timings"][stage.name] = round(elapsed, 3)
            log.info("%s: %d coins (%.3fs)", stage.describe(), len(result.coins), elapsed)

        # The full-size tray image is large and not part of the result payload.
        result.meta.pop("image", None)
        return result

    def describe(self) -> str:
        return " -> ".join(stage.describe() for stage in self.stages)


def side_from_path(path: str | Path) -> str:
    """Which tray a photo shows, from a `tray_001_a` / `tray_001_b` filename.

    The two trays are marked with opposite winding, so calibration has to know
    which it is looking at. Anything without a recognised suffix is treated as
    the first tray.
    """
    stem = Path(path).stem.lower()
    if stem.endswith(f"_{SIDE_B}"):
        return SIDE_B
    return SIDE_A


def _apply_calibrated_scale(config: Config) -> None:
    """Derive pixel-space parameters from physical sizes once the scale is known.

    After rectification the scale is exact, so the segmentation radii and the
    normalisation scale should follow from millimetres rather than being tuned
    by hand against a particular camera height.
    """
    scale = config.calibrate.pixels_per_mm / config.calibrate.scale_correction

    if config.normalise.pixels_per_mm is None:
        config.normalise.pixels_per_mm = scale

    config.segment.min_radius = max(1, int(config.segment.min_diameter_mm * scale / 2))
    config.segment.max_radius = int(config.segment.max_diameter_mm * scale / 2)
    # Two coins cannot have centres closer than the smallest coin's diameter.
    config.segment.min_distance = max(1, int(config.segment.min_diameter_mm * scale))

    log.info(
        "calibrated: %.3g px/mm, coin radius %d-%d px",
        scale,
        config.segment.min_radius,
        config.segment.max_radius,
    )


def load_image(path: str | Path) -> np.ndarray:
    """Read an image as BGR uint8, raising a clear error if it cannot be read."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")
    # imread cannot handle non-ASCII paths on Windows; go through numpy instead.
    buffer = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode image: {path}")
    return image


def save_image(path: str | Path, image: np.ndarray) -> None:
    """Write an image, tolerating non-ASCII paths on Windows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"could not encode image for {path}")
    encoded.tofile(str(path))
