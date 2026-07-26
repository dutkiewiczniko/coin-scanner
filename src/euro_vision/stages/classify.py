"""Stage 3 — identify the denomination of each coin.

Backends, in increasing order of what they need from you:
  stub     — does nothing. Keeps the pipeline runnable end-to-end with no models.
  diameter — nearest physical diameter. Needs `normalise.pixels_per_mm` set, but
             no training at all; every Euro denomination has a distinct diameter,
             so this is a strong baseline in a fixed-height tray rig.
  cnn      — trained PyTorch classifier over the eight denominations.
"""

from __future__ import annotations

from ..config import ClassifyConfig
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


def nearest_denomination(diameter_mm: float) -> tuple[int, float]:
    """Closest denomination to a measured diameter, and how far off it is.

    Shared by the diameter classifier and the pairing stage, so a coin cannot be
    labelled from one face's measurement while its size is reported from
    another.
    """
    return min(
        ((d, abs(diameter_mm - mm)) for d, mm in DIAMETERS_MM.items()),
        key=lambda pair: pair[1],
    )


class ClassifyStage(Stage):
    name = "classify"

    def __init__(self, config: ClassifyConfig):
        self.config = config
        self._model = None

    def describe(self) -> str:
        return f"classify[{self.config.backend}]"

    def run(self, result: ScanResult) -> ScanResult:
        backend = self.config.backend
        if backend == "stub":
            return result
        if backend == "diameter":
            return self._classify_by_diameter(result)
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
