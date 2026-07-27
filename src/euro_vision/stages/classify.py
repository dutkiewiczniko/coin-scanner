"""Stage 3 — identify the denomination of each coin.

Backends, in increasing order of what they need from you:
  stub     — does nothing. Keeps the pipeline runnable end-to-end with no models.
  diameter — nearest physical diameter. Needs `normalise.pixels_per_mm` set, but
             no training at all.
  metal    — alloy from colour, then nearest diameter within that alloy. Same
             measurement as `diameter`, but a coin only has to be told from the
             two others of its own alloy rather than all seven.
  cnn      — trained PyTorch classifier over the eight denominations.

Measured on 48 hand-graded coins from batch 101, deciding each coin once from
both its faces:

    diameter   44/48   91.7%
    metal      46/48   95.8%   (alloy itself right 47/48)

The gap is entirely about how much room the measurement has. Across all eight
denominations the nearest boundary is 0.5 mm away and the measurement's spread is
0.45 mm; within one alloy the nearest boundary is 1.0 mm away. Same numbers, far
more margin.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import ClassifyConfig
from ..metal import METAL_GROUP, GroupModel, denominations_in, metal_features
from ..types import DENOMINATIONS, ScanResult
from .base import Stage

#: Official Euro coin diameters in millimetres, keyed by value in cents.
DIAMETERS_MM = {
    1: 16.25,
    2: 18.75,
    5: 21.25,
    10: 19.75,
    20: 22.25,
    50: 24.25,
    100: 23.25,
    200: 25.75,
}

#: Beyond this gap (mm) between measured and reference diameter, the match is
#: reported with low confidence rather than accepted.
_DIAMETER_TOLERANCE_MM = 1.0


def nearest_denomination(
    diameter_mm: float, group: str | None = None
) -> tuple[int, float]:
    """Closest denomination to a measured diameter, and how far off it is.

    Shared by the diameter classifier and the pairing stage, so a coin cannot be
    labelled from one face's measurement while its size is reported from
    another.

    Restricting to an alloy widens the nearest decision boundary from 0.5 mm to
    at least 1.0 mm, which is what makes the same measurement reliable.
    """
    candidates = DIAMETERS_MM
    if group is not None:
        allowed = denominations_in(group)
        if allowed:
            candidates = {d: DIAMETERS_MM[d] for d in allowed}
    return min(
        ((d, abs(diameter_mm - mm)) for d, mm in candidates.items()),
        key=lambda pair: pair[1],
    )


class ClassifyStage(Stage):
    name = "classify"

    def __init__(self, config: ClassifyConfig):
        self.config = config
        self._model = None
        self._group_model = None

    def describe(self) -> str:
        return f"classify[{self.config.backend}]"

    def run(self, result: ScanResult) -> ScanResult:
        backend = self.config.backend
        if backend == "stub":
            return result
        if backend == "diameter":
            return self._classify_by_diameter(result)
        if backend == "metal":
            return self._classify_by_metal(result)
        if backend == "cnn":
            return self._classify_by_cnn(result)
        raise ValueError(f"unknown classify backend: {backend}")

    # -- backends ---------------------------------------------------------

    def _classify_by_diameter(self, result: ScanResult) -> ScanResult:
        missing = [c.index for c in result.coins if c.diameter_mm is None]
        if missing:
            raise ValueError(
                "diameter backend needs normalise.pixels_per_mm to be set "
                f"(no diameter for coin(s) {missing[:5]})"
            )

        for coin in result.coins:
            best, gap = nearest_denomination(coin.diameter_mm)
            coin.denomination = best
            # Confidence falls off linearly with the size mismatch.
            coin.denomination_confidence = max(
                0.0, 1.0 - gap / _DIAMETER_TOLERANCE_MM
            )
        return result

    def _classify_by_metal(self, result: ScanResult) -> ScanResult:
        missing = [c.index for c in result.coins if c.diameter_mm is None]
        if missing:
            raise ValueError(
                "metal backend needs normalise.pixels_per_mm to be set "
                f"(no diameter for coin(s) {missing[:5]})"
            )

        model = self._load_group_model()
        for coin in result.coins:
            group = confidence = None
            if coin.normalised is not None:
                group, confidence = model.predict(metal_features(coin.normalised))

            best, gap = nearest_denomination(coin.diameter_mm, group)
            coin.denomination = best
            size_confidence = max(0.0, 1.0 - gap / _DIAMETER_TOLERANCE_MM)
            # A coin is only as well identified as the weaker of the two steps:
            # a confident alloy paired with a size that landed between two
            # references is still a guess, and so is the reverse.
            coin.denomination_confidence = (
                size_confidence if confidence is None
                else min(size_confidence, confidence * 2.0, 1.0)
            )
            if group is not None:
                coin.metal_group = group
        return result

    def _load_group_model(self) -> GroupModel:
        if self._group_model is None:
            path = Path(self.config.metal_model)
            if not path.exists():
                raise FileNotFoundError(
                    f"metal backend needs a fitted alloy model at {path}; "
                    "build one with `euro-vision fit-metal`"
                )
            self._group_model = GroupModel.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
        return self._group_model

    def _classify_by_cnn(self, result: ScanResult) -> ScanResult:
        import torch  # optional dependency, imported on use

        model = self._load_cnn()
        batch = [c for c in result.coins if c.normalised is not None]
        if not batch:
            return result

        tensor = torch.stack([_to_tensor(c.normalised) for c in batch])
        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1)

        confidences, indices = probabilities.max(dim=1)
        for coin, idx, conf in zip(batch, indices.tolist(), confidences.tolist()):
            coin.denomination = DENOMINATIONS[idx]
            coin.denomination_confidence = float(conf)
        return result

    def _load_cnn(self):
        if self._model is None:
            import torch

            model = torch.load(self.config.weights, map_location="cpu")
            model.eval()
            self._model = model
        return self._model


def _to_tensor(image):
    """HWC BGR uint8 array -> CHW RGB float tensor in [0, 1]."""
    import torch

    rgb = image[:, :, ::-1].copy()
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
