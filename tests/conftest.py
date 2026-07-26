from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="opencv-python not installed")


def make_tray_image(rows: int = 2, cols: int = 3, radius: int = 50) -> np.ndarray:
    """A synthetic tray: evenly spaced light discs on a dark background.

    Enough to exercise segmentation and the stage wiring without real photos.
    """
    spacing = radius * 3
    margin = spacing
    height = margin * 2 + spacing * (rows - 1)
    width = margin * 2 + spacing * (cols - 1)

    image = np.full((height, width, 3), 30, dtype=np.uint8)
    centres = []
    for r in range(rows):
        for c in range(cols):
            cx = margin + c * spacing
            cy = margin + r * spacing
            cv2.circle(image, (cx, cy), radius, (200, 200, 205), -1)
            cv2.circle(image, (cx, cy), radius, (120, 120, 125), 3)
            centres.append((cx, cy))
    return image


#: BGR values matching the HSV bands in stages/calibrate.py.
MARKER_BGR = {
    "red": (0, 0, 220),
    "green": (0, 200, 0),
    "blue": (220, 0, 0),
    "magenta": (200, 0, 200),
}

TRAY_W_MM = 200.0
TRAY_H_MM = 150.0


def make_marked_tray(
    coins_mm: list[tuple[float, float, float]],
    pixels_per_mm: float = 4.0,
    margin_mm: float = 12.0,
    markers: tuple[str, ...] = ("red", "green", "blue", "magenta"),
) -> tuple[np.ndarray, float]:
    """A top-down tray with four coloured corner dots and coins at known sizes.

    `coins_mm` is a list of (x_mm, y_mm, diameter_mm) in tray coordinates, where
    (0, 0) is the centre of the first marker. Returns the image and the offset
    in pixels between tray origin and image origin.
    """
    offset_px = margin_mm * pixels_per_mm
    width = int((TRAY_W_MM + 2 * margin_mm) * pixels_per_mm)
    height = int((TRAY_H_MM + 2 * margin_mm) * pixels_per_mm)
    image = np.full((height, width, 3), 90, dtype=np.uint8)  # mid-grey tray

    def to_px(x_mm: float, y_mm: float) -> tuple[int, int]:
        return int(round(x_mm * pixels_per_mm + offset_px)), int(
            round(y_mm * pixels_per_mm + offset_px)
        )

    # Markers clockwise from the top-left corner.
    corners_mm = [(0, 0), (TRAY_W_MM, 0), (TRAY_W_MM, TRAY_H_MM), (0, TRAY_H_MM)]
    for colour, (mx, my) in zip(markers, corners_mm):
        cv2.circle(image, to_px(mx, my), int(4 * pixels_per_mm), MARKER_BGR[colour], -1)

    for x_mm, y_mm, diameter_mm in coins_mm:
        centre = to_px(x_mm, y_mm)
        radius = int(round(diameter_mm / 2 * pixels_per_mm))
        # A real coin's rim is raised and catches the light, so the coin stays
        # bright right out to its edge; any shadow falls outside it. Drawing a
        # dark ring inside the radius would understate the true diameter.
        cv2.circle(image, centre, radius, (200, 200, 205), -1)
        cv2.circle(image, centre, int(radius * 0.78), (170, 170, 176), -1)  # field
        cv2.circle(image, centre, int(radius * 0.35), (195, 195, 200), -1)  # relief

    return image, offset_px


def apply_perspective(image: np.ndarray, strength: float = 0.06) -> np.ndarray:
    """Warp an image as if the camera were off-axis, for testing rectification."""
    h, w = image.shape[:2]
    source = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    shift = w * strength
    # Keystone: the far edge of the tray renders shorter than the near edge.
    destination = np.float32(
        [[shift, shift * 0.4], [w - shift, 0], [w, h - shift * 0.3], [0, h]]
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(image, matrix, (w, h), borderValue=(90, 90, 90))


@pytest.fixture
def tray_image() -> np.ndarray:
    return make_tray_image()


@pytest.fixture
def tray_image_path(tmp_path, tray_image) -> str:
    path = tmp_path / "tray_test.png"
    cv2.imwrite(str(path), tray_image)
    return str(path)
