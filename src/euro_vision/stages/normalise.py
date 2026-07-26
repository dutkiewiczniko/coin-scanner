"""Stage 2 — bring every crop into a common frame.

Downstream models see coins at one fixed size, centred, with the tray background
masked away. Rotation alignment is optional: Euro reverses have no reliable
single orientation cue, so the default is to leave rotation alone and rely on
rotation-invariant training instead.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import NormaliseConfig
from ..types import ScanResult
from .base import Stage


class NormaliseStage(Stage):
    name = "normalise"

    def __init__(self, config: NormaliseConfig):
        self.config = config

    def describe(self) -> str:
        return f"normalise[{self.config.size}px, rotation={self.config.rotation_method}]"

    def run(self, result: ScanResult) -> ScanResult:
        cfg = self.config
        for coin in result.coins:
            if coin.image is None or coin.image.size == 0:
                continue

            square = _to_square(coin.image)
            resized = cv2.resize(
                square, (cfg.size, cfg.size), interpolation=cv2.INTER_AREA
            )

            if cfg.rotation_method == "gradient":
                coin.rotation = _dominant_gradient_angle(resized)
                resized = _rotate(resized, -coin.rotation)
            elif cfg.rotation_method != "none":
                raise ValueError(f"unknown rotation_method: {cfg.rotation_method}")

            if cfg.circular_mask:
                resized = _apply_circular_mask(resized)

            coin.normalised = resized

            if cfg.pixels_per_mm:
                coin.diameter_mm = round(
                    (coin.detection.radius * 2) / cfg.pixels_per_mm, 2
                )
        return result


def _to_square(image: np.ndarray) -> np.ndarray:
    """Pad the shorter axis so resizing does not distort the coin."""
    h, w = image.shape[:2]
    if h == w:
        return image
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    return cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )


def _apply_circular_mask(image: np.ndarray) -> np.ndarray:
    size = image.shape[0]
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), size // 2, 255, -1)
    return cv2.bitwise_and(image, image, mask=mask)


def _rotate(image: np.ndarray, degrees: float) -> np.ndarray:
    size = image.shape[0]
    matrix = cv2.getRotationMatrix2D((size / 2, size / 2), degrees, 1.0)
    return cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR)


def _dominant_gradient_angle(image: np.ndarray) -> float:
    """Angle of the strongest gradient direction, in degrees.

    A cheap, deterministic orientation estimate. It is repeatable rather than
    semantically correct — it does not find "up" on the coin design, it just
    maps a given coin to a consistent angle across scans.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(gx, gy)
    angle = np.rad2deg(np.arctan2(gy, gx)) % 180.0

    histogram, edges = np.histogram(
        angle, bins=36, range=(0, 180), weights=magnitude
    )
    peak = int(np.argmax(histogram))
    return float((edges[peak] + edges[peak + 1]) / 2)
