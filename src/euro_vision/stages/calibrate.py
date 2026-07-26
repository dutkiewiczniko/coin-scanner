"""Stage 0 — rectify the tray to a metric top-down view.

Four coloured dots at the tray corners define a rectangle of known physical
size. Warping that rectangle to a fixed pixel grid removes perspective and tilt,
and makes the scale exact and uniform across the whole tray — a ruler in one
corner cannot do that, because a phone's wide lens renders coins near the edge
smaller and slightly elliptical.

The same scheme handles the two-tray flip. Tray B's corners are marked with the
colour of the tray A corner they touch.

An unavoidable consequence: when the trays are face to face their coin-side
surfaces have opposite chirality, so if tray A reads clockwise in a given colour
order, tray B necessarily reads *anticlockwise* in that same order. No marking
scheme avoids this. The second tray's markers are therefore read in reverse
order, which keeps the homography orientation-preserving — mirroring the
rectified image would flip every coin design, making dates read backwards and
breaking any design matching.

The cost is that tray B's rectified frame is flipped vertically relative to tray
A's. `Calibration.to_reference_frame` gives the transform that undoes it, so the
pairing stage can put both faces in one coordinate system.

Caveat: a four-point homography corrects perspective, not lens barrel
distortion. Keep the tray within the central part of the frame.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

log = logging.getLogger(__name__)

from ..config import CalibrateConfig
from ..types import ScanResult
from .base import Stage

#: Inclusive HSV bounds per colour name, OpenCV ranges (H 0-179, S/V 0-255).
#: Red wraps the hue origin, so it needs two bands.
#:
#: Coin metal overlaps some of these: copper-plated 1c/2c/5c sit near the red
#: band, and Nordic gold 10c/20c/50c near yellow. That is tolerated rather than
#: avoided, because corner markers are chosen by position (see
#: `_select_corner_markers`) rather than by being the biggest blob of their
#: colour. Blue, green, magenta and cyan have no coin equivalent at all, so a
#: palette drawn from those needs the least from the detector.
COLOUR_RANGES: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "red": [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    # Orange sits between the red and yellow bands, and close to Nordic gold
    # coins in hue — but not in saturation. Measured on real photos: orange
    # paint H=13 S=191, gold coins H=19-20 S~102. The high saturation floor is
    # what separates them, so it matters more here than for other colours.
    "orange": [((11, 140, 40), (18, 255, 255))],
    "green": [((40, 80, 60), (85, 255, 255))],
    "blue": [((95, 100, 60), (130, 255, 255))],
    "magenta": [((140, 90, 70), (168, 255, 255))],
    "cyan": [((85, 90, 70), (98, 255, 255))],
    "yellow": [((20, 120, 90), (35, 255, 255))],
}


class CalibrationError(ValueError):
    """Raised when the tray corners cannot be located or make no sense."""


def to_reference_frame(
    x: float, y: float, side: str, height_px: int
) -> tuple[float, float]:
    """Map a rectified point into tray A's frame.

    Tray B's frame is flipped vertically relative to tray A's, because its
    markers are read in reverse order to keep coin designs unmirrored. Applying
    this to both faces puts every coin in one coordinate system, where the two
    faces of the same coin land in nearly the same place.

    This lives at module level so the pairing stage and the calibration share one
    definition; two copies of this transform would be a silent correctness bug.
    """
    if side == SIDE_A:
        return x, y
    return x, height_px - y


@dataclass
class Marker:
    colour: str
    x: float
    y: float
    area: float


#: Photo of the first tray; markers read clockwise in the configured order.
SIDE_A = "a"
#: Photo of the second tray; markers necessarily read anticlockwise.
SIDE_B = "b"


@dataclass
class Calibration:
    """The mapping from one photo to the metric tray frame."""

    markers: list[Marker]
    homography: np.ndarray
    width_px: int
    height_px: int
    pixels_per_mm: float
    side: str = SIDE_A
    #: How many blobs of each colour were in the running. More than one means
    #: something else in shot shares that hue — coins, or the background.
    candidate_counts: dict[str, int] = field(default_factory=dict)

    def to_reference_frame(self, x: float, y: float) -> tuple[float, float]:
        """Map a point in this photo's rectified frame into tray A's frame."""
        return to_reference_frame(x, y, self.side, self.height_px)

    def edge_lengths_px(self) -> list[float]:
        """Marker-to-marker distances in the original photo, clockwise.

        Opposite edges should be close to equal. A large difference means the
        camera is tilted or off-centre relative to the tray.
        """
        points = [(m.x, m.y) for m in self.markers]
        return [
            float(np.hypot(points[i][0] - points[(i + 1) % 4][0],
                           points[i][1] - points[(i + 1) % 4][1]))
            for i in range(4)
        ]

    def tilt_ratio(self) -> float:
        """Worst opposite-edge length ratio; 1.0 is a perfectly square-on shot."""
        top, right, bottom, left = self.edge_lengths_px()
        horizontal = max(top, bottom) / max(min(top, bottom), 1e-6)
        vertical = max(left, right) / max(min(left, right), 1e-6)
        return max(horizontal, vertical)


class CalibrateStage(Stage):
    name = "calibrate"

    def __init__(self, config: CalibrateConfig):
        self.config = config
        if len(config.markers) != 4:
            raise ValueError(
                f"calibrate.markers needs exactly 4 colours, got {len(config.markers)}"
            )
        unknown = [c for c in config.markers if c not in COLOUR_RANGES]
        if unknown:
            raise ValueError(
                f"unknown marker colour(s): {', '.join(unknown)} "
                f"(available: {', '.join(sorted(COLOUR_RANGES))})"
            )
        if len(set(config.markers)) != 4:
            raise ValueError("calibrate.markers must all be different colours")

    def describe(self) -> str:
        return f"calibrate[{self.config.pixels_per_mm:g}px/mm]"

    def run(self, result: ScanResult) -> ScanResult:
        image = result.meta.get("image")
        if image is None:
            raise ValueError("calibrate stage requires result.meta['image']")

        side = result.meta.get("side", SIDE_A)
        calibration = compute_calibration(image, self.config, side)
        result.meta["image"] = cv2.warpPerspective(
            image,
            calibration.homography,
            (calibration.width_px, calibration.height_px),
            flags=cv2.INTER_LINEAR,
        )
        result.meta["image_size"] = [calibration.width_px, calibration.height_px]
        result.meta["pixels_per_mm"] = calibration.pixels_per_mm
        result.meta["side"] = side
        result.meta["tilt_ratio"] = round(calibration.tilt_ratio(), 4)
        result.meta["markers"] = {
            m.colour: [round(m.x, 1), round(m.y, 1)] for m in calibration.markers
        }
        return result


def compute_calibration(
    image: np.ndarray, config: CalibrateConfig, side: str = SIDE_A
) -> Calibration:
    """Locate the corner markers and derive the rectifying homography.

    `side` says which of the two trays this photo shows. The second tray's
    markers are read in reverse, because face-to-face trays are necessarily
    marked with opposite winding.
    """
    if side not in (SIDE_A, SIDE_B):
        raise ValueError(f"side must be '{SIDE_A}' or '{SIDE_B}', got {side!r}")

    order = list(config.markers)
    if side == SIDE_B:
        order.reverse()

    candidates = {
        colour: _find_candidates(image, colour, config.min_marker_area)
        for colour in order
    }
    markers = _select_corner_markers(candidates, order, image.shape[:2])
    _check_winding(markers, side)

    # Each tray is rectified against its own marker rectangle. Both frames then
    # measure millimetres from the corner the trays share when sandwiched, so
    # positions correspond even if the two trays are not the same size.
    tray_width = config.tray_width_mm
    tray_height = config.tray_height_mm
    if side == SIDE_B:
        tray_width = config.tray_b_width_mm or tray_width
        tray_height = config.tray_b_height_mm or tray_height

    # The output resolution is just a rendering choice and takes no correction.
    width_px = int(round(tray_width * config.pixels_per_mm))
    height_px = int(round(tray_height * config.pixels_per_mm))

    # The correction belongs on the measurement scale alone. Markers projecting
    # outward make the marker rectangle span more of the coin plane than its
    # measured width, so coins render smaller than life and read small; dividing
    # the scale by the correction scales every measured diameter back up.
    effective_scale = config.pixels_per_mm / config.scale_correction

    source = np.array([[m.x, m.y] for m in markers], dtype=np.float32)
    destination = np.array(
        [[0, 0], [width_px, 0], [width_px, height_px], [0, height_px]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(source, destination)

    return Calibration(
        markers=markers,
        homography=homography,
        width_px=width_px,
        height_px=height_px,
        pixels_per_mm=effective_scale,
        side=side,
        candidate_counts={c: len(candidates[c]) for c in order},
    )


#: How many blobs per colour to keep as possible corners.
#:
#: This is a genuinely awkward knob, and the reason marker colour matters more
#: than any amount of cleverness here. Selecting the largest quadrilateral
#: assumes impostors lie INSIDE the true corners, which holds for coins on the
#: tray but fails for anything in the background — that is outside the corners,
#: so it enlarges the quad and wins.
#:
#: Measured on real photos: yellow markers on an oak table produced 35-53 wood
#: blobs. At this cap the true marker was still found; raising it to 16 made the
#: selection pick wood ~900 px from the real marker. Keeping the cap low papers
#: over the problem rather than solving it — the actual fix is a marker colour
#: absent from the background, so no impostor exists to rank at all. Blue,
#: green, magenta and cyan matched 0.00-0.08% of those photos; yellow matched
#: 0.30% and red overlaps copper coins.
_MAX_CANDIDATES = 6

#: The marker quadrilateral must cover at least this share of the frame. The
#: tray fills most of the shot, so anything smaller means the corners were not
#: what got detected.
_MIN_QUAD_COVERAGE = 0.05


def _find_candidates(image: np.ndarray, colour: str, min_area: int) -> list[Marker]:
    """All plausible blobs of `colour`, largest first.

    More than one is expected and normal: copper 1c/2c/5c coins fall close to
    the red band and Nordic gold 10c/20c/50c coins close to yellow, so a tray
    full of coins can easily produce extra matches. Choosing between them is
    left to `_select_corner_markers`, which uses geometry rather than size — a
    coin is much bigger than a dot, so picking the largest blob would reliably
    pick the wrong thing.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low, high in COLOUR_RANGES[colour]:
        mask |= cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))

    # Remove speckle before measuring, so a few stray pixels cannot win.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not candidates:
        raise CalibrationError(
            f"no {colour} corner marker found "
            f"(no blob of at least {min_area} px). Check lighting and that the "
            f"dot is not covered by a coin."
        )

    candidates.sort(key=cv2.contourArea, reverse=True)
    if len(candidates) > _MAX_CANDIDATES:
        log.warning(
            "%s matched %d blobs; only the %d largest are considered. Corner "
            "selection is only reliable while impostors sit inside the true "
            "corners — background ones do not. Use a colour absent from the "
            "background (magenta or cyan).",
            colour, len(candidates), _MAX_CANDIDATES,
        )

    markers = []
    for contour in candidates[:_MAX_CANDIDATES]:
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        markers.append(
            Marker(
                colour=colour,
                x=moments["m10"] / moments["m00"],
                y=moments["m01"] / moments["m00"],
                area=float(cv2.contourArea(contour)),
            )
        )
    return markers


def _signed_area(points: Sequence[Marker]) -> float:
    """Shoelace area of the quadrilateral through these points, in order."""
    total = 0.0
    for i in range(len(points)):
        a, b = points[i], points[(i + 1) % len(points)]
        total += a.x * b.y - b.x * a.y
    return total / 2.0


def _select_corner_markers(
    candidates: dict[str, list[Marker]],
    order: list[str],
    image_shape: tuple[int, int],
) -> list[Marker]:
    """Pick the one blob per colour that forms the largest quadrilateral.

    The corner dots sit on the tray lip, outside the coin area, so they are
    always the outermost blobs of their colour — any coin that happens to match
    a marker's hue lies inside them. Maximising the enclosed area therefore
    picks the true corners regardless of how big the impostors are.

    It also self-corrects the ordering: taking the points in the configured
    cyclic order, a wrong assignment produces a self-intersecting bowtie whose
    area is smaller than the correct convex quadrilateral.
    """
    missing = [colour for colour in order if not candidates[colour]]
    if missing:
        raise CalibrationError(
            f"no {', '.join(missing)} corner marker(s) found. Check the lighting, "
            f"that no coin is covering a dot, and that the dot colours match "
            f"calibrate.markers."
        )

    best: tuple[Marker, ...] | None = None
    best_area = -1.0
    for combination in itertools.product(*(candidates[colour] for colour in order)):
        area = abs(_signed_area(combination))
        if area > best_area:
            best_area, best = area, combination

    assert best is not None  # every colour has at least one candidate

    height, width = image_shape
    coverage = best_area / float(width * height)
    if coverage < _MIN_QUAD_COVERAGE:
        raise CalibrationError(
            f"the four corner markers enclose only {coverage:.1%} of the frame, "
            f"which is too little to be the tray. The dots found are probably "
            f"not the corner markers — check that each corner has one clear, "
            f"saturated dot and that nothing else in shot shares those colours."
        )

    return list(best)


def _check_winding(markers: list[Marker], side: str) -> None:
    """Reject an order that would make the homography a reflection.

    In image coordinates (y downwards) a visually clockwise quadrilateral has a
    positive shoelace area. By this point the marker list has already been
    reversed for the second tray, so either side should come out clockwise. A
    negative area means the rectification would mirror every coin design, so
    fail loudly rather than silently corrupt the crops.
    """
    if _signed_area(markers) > 0:
        return

    order = " -> ".join(m.colour for m in markers)
    if side == SIDE_A:
        raise CalibrationError(
            f"corner markers run anticlockwise in this side-{SIDE_A} photo "
            f"({order}). Rectifying would mirror every coin. Either this is "
            f"actually a photo of the second tray — check the '_{SIDE_B}' "
            f"filename suffix — or calibrate.markers lists the colours in the "
            f"wrong order for the first tray."
        )
    raise CalibrationError(
        f"corner markers run anticlockwise in this side-{SIDE_B} photo even "
        f"after reversing ({order}), so they wind the same way as the first "
        f"tray. Face-to-face trays must wind oppositely. Either this is really "
        f"a side-{SIDE_A} photo, or the second tray's corners were not marked "
        f"by touch. To mark it: sandwich the trays as you would to flip them, "
        f"and give each corner the colour of the corner it touches."
    )
