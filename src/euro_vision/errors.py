"""Find striking errors by geometry rather than by matching a catalogue.

No database can help here. A catalogue says which *issues* are scarce; an error
is a fault in one particular coin, so it has to be measured. The errors worth
money that are also visible at phone resolution are all geometric:

  clipped planchet   the blank was punched overlapping a previous hole, so the
                     coin has a straight or curved bite out of its edge.
  off-centre strike  the blank sat proud of the die, so the design and its rim
                     sit off to one side, leaving a blank crescent.
  broadstrike        struck without the retaining collar, so the coin spread
                     wider and thinner than it should be.

Doubled dies, repunched mintmarks and small die chips are deliberately not
attempted. They need magnification this rig does not have, and a detector that
claims to find them would only produce noise.

Each measure is scored so that 0 is a perfectly ordinary coin and 1 is blatant,
and the three are reported separately rather than as one number -- a clipped
planchet and an off-centre strike are different finds, and a coin that trips two
of them is more likely to be a segmentation failure than a jackpot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ErrorReport:
    """Geometric anomaly scores for one coin face, each in [0, 1]."""

    clip: float = 0.0
    off_centre: float = 0.0
    broadstrike: float = 0.0
    #: Set when the crop was unusable, e.g. all one colour. The scores are then
    #: meaningless and should not be treated as "no error found".
    valid: bool = True
    notes: str = ""

    @property
    def worst(self) -> float:
        return max(self.clip, self.off_centre, self.broadstrike)

    def flagged(self, threshold: float) -> list[str]:
        found = []
        if self.clip >= threshold:
            found.append(f"clipped planchet ({self.clip:.2f})")
        if self.off_centre >= threshold:
            found.append(f"off-centre strike ({self.off_centre:.2f})")
        if self.broadstrike >= threshold:
            found.append(f"broadstrike ({self.broadstrike:.2f})")
        return found


def measure(
    crop: np.ndarray | None = None,
    diameter_mm: float | None = None,
    expected_mm: float | None = None,
    radial_profile: list[float] | np.ndarray | None = None,
) -> ErrorReport:
    """Score one coin for striking errors.

    `radial_profile` is the per-ray edge distance measured on the tray image by
    the segmentation stage. It is the only input a clip can be read from: the
    normalised crop is masked to a circle, which makes every coin round no matter
    what shape it really was, so passing only a crop leaves `clip` at zero rather
    than reporting a reassuring but meaningless number.

    `expected_mm` is the reference diameter of the denomination the coin was
    classified as; without it broadstrike cannot be judged, since a wide coin and
    a misclassified one look the same from size alone.
    """
    report = ErrorReport()
    measured_anything = False

    if radial_profile is not None:
        radii = np.asarray(radial_profile, dtype=np.float64)
        if radii.size >= 8 and np.isfinite(radii).all() and radii.max() > 0:
            if _outline_is_contaminated(radii):
                report.notes = ("outline blocked by touching neighbours — "
                                "shape not judged")
            else:
                report.clip = _clip_score(radii)
                measured_anything = True
        else:
            report.notes = "radial profile too short to judge shape"

    if crop is not None and crop.size:
        gray = (crop if crop.ndim == 2 else crop[:, :, :3].mean(axis=2))
        gray = gray.astype(np.float32)
        if float(gray.max()) - float(gray.min()) >= 5.0:
            report.off_centre = _off_centre_score(gray)
            measured_anything = True
        else:
            report.notes = "flat crop, nothing to measure"

    if diameter_mm is not None and expected_mm:
        report.broadstrike = _broadstrike_score(diameter_mm, expected_mm)
        measured_anything = True

    report.valid = measured_anything
    if not measured_anything and not report.notes:
        report.notes = "nothing measurable supplied"
    return report


def _outline_is_contaminated(radii: np.ndarray) -> bool:
    """Whether neighbouring coins have spoiled the outline measurement.

    A ray that meets a touching neighbour never crosses back to tray level and
    runs on into it, so it reports too far. On a densely filled tray that happens
    on most of a coin's perimeter, which drags the median out past the real edge
    and leaves every honest ray looking short — the exact signature of a clip.

    The two are told apart by which tail is long. Overshoot stretches the
    distribution upward; a clip only ever cuts downward. Checked on 145 real
    faces from a full tray, this is what separated 12 spurious "clips" from the
    coins genuinely worth a look.

    The comparison is against the 75th percentile rather than the median because
    the median moves under a clip: a deep bite drags it down, inflating the ratio
    and rejecting exactly the coin worth finding. Any clip covering less than a
    quarter of the rim leaves the 75th percentile sitting on intact metal.

    From the numbers alone the two are genuinely ambiguous once enough rays are
    involved -- half the rays short of the rest reads equally as "half of them
    overshot" or "half the coin is missing". Physics breaks the tie: a coin
    missing a third of its edge is not a coin any more, while a third of the rim
    touching neighbours is an ordinary crowded tray. So too large a deficit is
    read as contamination, not as a spectacular find.
    """
    upper_quartile = float(np.percentile(radii, 75))
    if upper_quartile <= 0:
        return True
    # Partial overshoot: a few rays running past an otherwise clean rim.
    overshoot = (float(np.percentile(radii, 95)) - upper_quartile) / upper_quartile
    if overshoot > 0.05:
        return True
    # Wholesale overshoot: so much of the perimeter reads short of the reference
    # that the reference itself must be wrong.
    deficit = float(np.mean(radii < upper_quartile * 0.90))
    return deficit > 0.28


def _clip_score(radii: np.ndarray) -> float:
    """How much of the rim is bitten away.

    The reference is the 75th percentile, taken as the coin's true radius: a clip
    removes a contiguous arc and leaves the rest intact, so the upper quartile
    still sits on undamaged rim while the median is already being pulled down by
    the defect.

    Only a sustained *inward* run counts. Rays that fall short in ones and twos
    are shadow; a real clip is both deep and continuous, so the score needs an
    arc rather than a scatter.
    """
    reference = float(np.percentile(radii, 75))
    if reference <= 0:
        return 0.0
    shortfall = (reference - radii) / reference
    # 0.18 rather than something tighter: swept against 77 real coins with a
    # measurable outline and clips injected into them, this was where false
    # alarms stopped falling (6.5%, and the residue is segmentation rather than
    # metal) while a pronounced clip was still caught 94% of the time.
    deep = shortfall > 0.18
    longest = _longest_run(deep)
    # Below a sixteenth of the circumference this is shadow or a segmentation
    # wobble, not a bite out of the metal.
    if longest < max(4, len(radii) // 16):
        return 0.0
    arc_fraction = longest / len(radii)
    depth = float(np.mean(shortfall[deep]))
    # Scaled so a bite an eighth of the way round and 20% deep reads about 1.0.
    return float(np.clip((arc_fraction / 0.125) * min(depth / 0.20, 1.5), 0.0, 1.0))


def _off_centre_score(gray: np.ndarray) -> float:
    """How far the struck design sits from the middle of the blank.

    The design's own centre is estimated from where the detail is: relief casts
    light and shade, so the variance of brightness is high across the design and
    low across an unstruck crescent. Weighting each pixel by local contrast and
    taking the centroid finds the design without needing to recognise it.
    """
    height, width = gray.shape[:2]
    cy, cx = height / 2.0, width / 2.0
    # The crop is square and centred on the coin, so its own half-width is the
    # radius; the normalise stage's padding is inside that.
    radius = min(cy, cx)
    if radius <= 4:
        return 0.0

    yy, xx = np.mgrid[0:height, 0:width]
    inside = np.hypot(yy - cy, xx - cx) < radius * 0.94
    if inside.sum() < 64:
        return 0.0

    # Local contrast via the difference from a coarse blur, done with cumulative
    # sums so the module keeps working without OpenCV.
    detail = np.abs(gray - _box_blur(gray, max(3, int(radius * 0.18))))
    weights = np.where(inside, detail, 0.0)
    total = float(weights.sum())
    if total <= 1e-6:
        return 0.0

    centroid_y = float((weights * yy).sum() / total)
    centroid_x = float((weights * xx).sum() / total)
    offset = float(np.hypot(centroid_y - cy, centroid_x - cx)) / radius
    # Ordinary coins are not perfectly balanced -- a portrait fills one side more
    # than the other -- so a few percent of drift is normal and only larger
    # offsets mean the design missed the blank.
    return float(np.clip((offset - 0.06) / 0.24, 0.0, 1.0))


def _broadstrike_score(diameter_mm: float, expected_mm: float) -> float:
    """How far outside tolerance the coin's diameter is.

    Only oversize counts. A coin reading small is far more likely to be a worn
    coin or a short measurement than a striking error, whereas metal cannot
    spread beyond the collar unless the collar was missing.
    """
    if expected_mm <= 0:
        return 0.0
    excess = (diameter_mm - expected_mm) / expected_mm
    # 0.45 mm of measurement spread on a 23 mm coin is about 2%, so nothing
    # below that can be told apart from noise.
    return float(np.clip((excess - 0.02) / 0.06, 0.0, 1.0))


def _box_blur(image: np.ndarray, size: int) -> np.ndarray:
    """Mean filter via a summed-area table, so numpy alone suffices."""
    size = max(1, int(size) | 1)
    pad = size // 2
    padded = np.pad(image, pad, mode="edge")
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode="constant")
    h, w = image.shape
    total = (integral[size:size + h, size:size + w]
             - integral[0:h, size:size + w]
             - integral[size:size + h, 0:w]
             + integral[0:h, 0:w])
    return total / float(size * size)


def _longest_run(mask: np.ndarray) -> int:
    """Longest run of True, treating the array as a closed ring."""
    if not mask.any():
        return 0
    if mask.all():
        return len(mask)
    doubled = np.concatenate([mask, mask])
    best = run = 0
    for value in doubled:
        run = run + 1 if value else 0
        best = max(best, run)
    return min(best, len(mask))
