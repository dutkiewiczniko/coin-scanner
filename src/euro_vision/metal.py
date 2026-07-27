"""Tell a coin's alloy from its colour, so size only has to choose within it.

Euro denominations are spaced about 1 mm apart, but the radial edge measurement
has a spread of roughly 0.45 mm, so a nearest-diameter rule has to resolve a
half-millimetre boundary and gets 84% of graded faces right. The eight
denominations fall into three alloys, though, and *within* an alloy the sizes are
2 mm or more apart:

    copper   1c 16.25   2c 18.75   5c 21.25
    gold     10c 19.75  20c 22.25  50c 24.25
    bimetal  1 EUR 23.25          2 EUR 25.75

Deciding the alloy first turns a half-millimetre problem into a one-millimetre
one, which the same measurement answers correctly 98% of the time. That is the
whole reason this module exists: it is not a denomination classifier, it just
removes five of the eight options.

Two features do the work, both chosen to survive a change of lighting:

    redness   median red over median green across the coin's face. A ratio
              between channels rather than an absolute hue, because an earlier
              attempt to separate copper from gold on hue alone failed outright
              -- under warm light the two overlapped almost completely.
    two_tone  how far the centre disc's normalised colour sits from the outer
              annulus. Bimetallic coins have a real material boundary there;
              copper and gold coins are one alloy across.

Each zone is normalised by its own brightness before comparison, so the feature
measures hue difference rather than how the light fell across the coin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Alloy of each denomination, in cents.
METAL_GROUP: dict[int, str] = {
    1: "copper", 2: "copper", 5: "copper",
    10: "gold", 20: "gold", 50: "gold",
    100: "bimetal", 200: "bimetal",
}

GROUPS = ("copper", "gold", "bimetal")

#: Fraction of the crop radius bounding the centre disc and the outer annulus.
#: The annulus stops at 0.88 so the coin's own rim and the crop padding stay out
#: of it, and the centre stops at 0.45 to sit inside a 1 EUR coin's inner disc.
_CENTRE_MAX = 0.45
_ANNULUS = (0.62, 0.88)

#: Relief casts shadow, and those pixels carry the shadow's colour rather than
#: the alloy's. Dropping the darkest fifth of each zone keeps them out.
_DARK_PERCENTILE = 20


@dataclass(frozen=True)
class MetalFeatures:
    redness: float
    two_tone: float

    def as_array(self) -> np.ndarray:
        return np.array([self.redness, self.two_tone], dtype=np.float64)


def metal_features(crop: np.ndarray) -> MetalFeatures:
    """Colour features of a normalised coin crop (BGR, as OpenCV reads it)."""
    if crop is None or crop.size == 0:
        return MetalFeatures(0.0, 0.0)

    bgr = crop.astype(np.float32)
    if bgr.ndim != 3 or bgr.shape[2] < 3:
        return MetalFeatures(0.0, 0.0)

    height, width = bgr.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    radius = np.hypot(yy - height / 2, xx - width / 2) / (height / 2)

    centre = radius < _CENTRE_MAX
    annulus = (radius > _ANNULUS[0]) & (radius < _ANNULUS[1])
    if not centre.any() or not annulus.any():
        return MetalFeatures(0.0, 0.0)

    def lit(mask: np.ndarray) -> np.ndarray:
        pixels = bgr[mask]
        if pixels.size == 0:
            return pixels
        cutoff = np.percentile(pixels[:, 1], _DARK_PERCENTILE)
        keep = pixels[pixels[:, 1] > cutoff]
        return keep if keep.size else pixels

    centre_px, annulus_px = lit(centre), lit(annulus)
    if centre_px.size == 0 or annulus_px.size == 0:
        return MetalFeatures(0.0, 0.0)

    interior = np.concatenate([centre_px, annulus_px])
    blue, green, red = (np.median(interior[:, i]) for i in range(3))
    redness = float(red / max(green, 1e-6))

    centre_med = np.median(centre_px, axis=0)
    annulus_med = np.median(annulus_px, axis=0)
    centre_norm = centre_med / max(centre_med.sum(), 1e-6)
    annulus_norm = annulus_med / max(annulus_med.sum(), 1e-6)
    two_tone = float(np.abs(centre_norm - annulus_norm).sum())

    return MetalFeatures(redness, two_tone)


def denominations_in(group: str) -> list[int]:
    """Denominations belonging to an alloy, smallest first."""
    return sorted(d for d, g in METAL_GROUP.items() if g == group)


class GroupModel:
    """Nearest-centroid classifier over the two colour features.

    Deliberately the smallest thing that works. Scored leave-one-out on 48
    graded coins it matched scikit-learn's logistic regression exactly -- 47/48
    alloys, 46/48 denominations once size was applied within the alloy -- so the
    extra dependency bought nothing. With five copper coins in the training set,
    anything with real capacity would be fitting noise.

    Features are standardised before comparison because redness and two_tone
    differ by an order of magnitude in scale, and an unscaled distance would let
    redness decide everything.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray,
                 centroids: dict[str, np.ndarray], n_train: int = 0):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.centroids = {k: np.asarray(v, dtype=np.float64)
                          for k, v in centroids.items()}
        self.n_train = n_train

    @classmethod
    def fit(cls, features: list[MetalFeatures], groups: list[str]) -> "GroupModel":
        X = np.array([f.as_array() for f in features], dtype=np.float64)
        mean, std = X.mean(axis=0), X.std(axis=0) + 1e-9
        Z = (X - mean) / std
        names = np.array(groups)
        centroids = {g: Z[names == g].mean(axis=0)
                     for g in GROUPS if (names == g).any()}
        return cls(mean, std, centroids, n_train=len(groups))

    def predict(self, features: MetalFeatures) -> tuple[str, float]:
        """Alloy, and a confidence in [0, 1].

        Confidence is the relative margin between the nearest centroid and the
        next nearest, so a coin sitting midway between copper and gold reports
        near zero rather than an arbitrary winner.
        """
        z = (features.as_array() - self.mean) / self.std
        distances = sorted(
            ((float(np.linalg.norm(z - c)), g) for g, c in self.centroids.items())
        )
        if len(distances) < 2:
            return distances[0][1], 1.0
        (best, group), (second, _) = distances[0], distances[1]
        total = best + second
        confidence = 0.0 if total <= 1e-9 else (second - best) / total
        return group, float(confidence)

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "centroids": {k: v.tolist() for k, v in self.centroids.items()},
            "n_train": self.n_train,
            "features": ["redness", "two_tone"],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GroupModel":
        return cls(data["mean"], data["std"], data["centroids"],
                   int(data.get("n_train", 0)))
