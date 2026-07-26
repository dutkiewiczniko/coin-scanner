"""Configuration loading.

Config is a YAML file mapped onto dataclasses so stages get typed access and
unknown keys fail loudly rather than being silently ignored.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


@dataclass
class CalibrateConfig:
    #: When enabled, every image is rectified to a metric top-down view before
    #: segmentation, and the physical scale becomes known exactly.
    enabled: bool = False
    #: Corner marker colours, in clockwise order as they appear in the photo,
    #: starting from the corner that should map to the top-left of the output.
    markers: list[str] = field(
        default_factory=lambda: ["red", "green", "blue", "magenta"]
    )
    #: Centre-to-centre distances of the marker rectangle, in millimetres.
    tray_width_mm: float = 200.0
    tray_height_mm: float = 150.0
    #: Resolution of the rectified image. 10 px/mm resolves a 2 euro coin at
    #: roughly 258 px across, which is ample for classification.
    pixels_per_mm: float = 10.0
    #: Smallest blob, in pixels, accepted as a corner marker.
    min_marker_area: int = 40
    #: Multiplier applied to every measured diameter. Above 1 means the raw
    #: measurement reads small. Obtain it from `euro-vision measure`.
    #:
    #: The homography assumes the four markers lie in the same plane as the
    #: coins. Marks on a rim — especially on its outer wall rather than its top
    #: face — sit above and outside that plane, so they project outward by
    #: roughly (rim height x distance from the optical axis / camera height).
    #: The marker rectangle then spans more of the coin plane than its measured
    #: width says, and coins read small by that ratio. It is a systematic error,
    #: constant for a fixed camera, so it is measured once against coins of
    #: known denomination rather than modelled.
    scale_correction: float = 1.0


@dataclass
class SegmentConfig:
    #: "hough" (classical, no training needed) or "yolo" (trained detector).
    backend: str = "hough"
    #: Path to a trained YOLO weights file, used when backend == "yolo".
    weights: str = "models/segment.pt"
    confidence: float = 0.25
    #: Expected coin radius range in pixels, for the Hough backend.
    min_radius: int = 30
    max_radius: int = 120
    #: Minimum centre-to-centre distance between coins, in pixels.
    min_distance: int = 60
    #: Padding added around each detection when cropping, as a fraction of radius.
    crop_padding: float = 0.08
    #: Re-measure each radius from the coin's edge. Hough alone reads ~6% small,
    #: which is enough to confuse denominations 1 mm apart. Not used by the
    #: watershed backend, which measures from the coin's own region.
    refine_edges: bool = True
    #: Grey level separating coin from tray, for the watershed backend.
    #: None uses Otsu, which reads high on a full tray and loses dark coppers.
    #: With a fixed rig and lighting, a measured constant does better.
    coin_threshold: int | None = None
    #: Minimum saturation for a dark pixel to count as a copper coin rather than
    #: tray. None disables the colour test and uses brightness alone.
    #:
    #: Off by default because it backfires on a densely packed tray: it does
    #: recover worn coppers, but every pixel it adds also thickens the joins
    #: between touching coins, and the extra merges cost more than the recovered
    #: coins gain. Measured on a real tray: 70 coins found without it, 48 with.
    #: Worth revisiting only for sparsely filled trays.
    copper_saturation: int | None = None
    #: Copper coins still need some brightness; this rejects deep shadow.
    copper_min_value: int = 35
    #: Watershed seeding level, as a fraction of the smallest coin's radius.
    #: Too low and two touching coins share one seed and merge; too high and
    #: small coins get no seed and vanish.
    watershed_seed_ratio: float = 0.62
    #: Coin size bounds in millimetres. When calibration is enabled these replace
    #: the pixel radii above, which removes the guesswork from tuning them.
    #: The real range is 16.25 mm (1c) to 25.75 mm (2 euro).
    min_diameter_mm: float = 15.0
    max_diameter_mm: float = 27.0


@dataclass
class NormaliseConfig:
    #: Output edge length in pixels for every normalised coin crop.
    size: int = 224
    #: Mask out the square corners so background does not reach the model.
    circular_mask: bool = True
    #: "none" or "gradient" (dominant-gradient alignment, rotation-invariance aid).
    rotation_method: str = "none"
    #: Known physical tray scale; enables diameter_mm. None disables it.
    pixels_per_mm: float | None = None


@dataclass
class ClassifyConfig:
    #: "stub" (returns nothing), "diameter" (physical size heuristic), or "cnn".
    backend: str = "stub"
    weights: str = "models/denomination.pt"
    #: Below this the denomination is recorded but treated as unreliable.
    min_confidence: float = 0.5


@dataclass
class RareConfig:
    #: "none", "metadata" (database lookup), or "embedding" (nearest neighbour).
    backend: str = "none"
    weights: str = "models/embedding.pt"
    database: str = "data/rare_coins.sqlite"
    #: Similarity above which a coin is flagged for manual review.
    flag_threshold: float = 0.80
    #: Only run rare detection on these denominations (cents). Empty = all.
    denominations: list[int] = field(default_factory=lambda: [100, 200])


@dataclass
class Config:
    calibrate: CalibrateConfig = field(default_factory=CalibrateConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    normalise: NormaliseConfig = field(default_factory=NormaliseConfig)
    classify: ClassifyConfig = field(default_factory=ClassifyConfig)
    rare: RareConfig = field(default_factory=RareConfig)
    #: Where crops and result files are written.
    output_dir: str = "data/out"
    #: Save per-coin crops alongside the results file.
    save_crops: bool = True


_SECTIONS = {f.name: f.type for f in dataclasses.fields(Config)}


def _build_section(cls: Any, values: dict[str, Any], name: str) -> Any:
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(
            f"unknown key(s) in config section '{name}': {', '.join(sorted(unknown))}"
        )
    return cls(**values)


def load_config(path: str | Path | None = None) -> Config:
    """Load config from YAML, falling back to the packaged defaults."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        if path == DEFAULT_CONFIG_PATH:
            return Config()
        raise FileNotFoundError(f"config file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping, got {type(raw).__name__}")

    unknown = set(raw) - set(_SECTIONS) - {"output_dir", "save_crops"}
    if unknown:
        raise ValueError(f"unknown config section(s): {', '.join(sorted(unknown))}")

    return Config(
        calibrate=_build_section(CalibrateConfig, raw.get("calibrate", {}), "calibrate"),
        segment=_build_section(SegmentConfig, raw.get("segment", {}), "segment"),
        normalise=_build_section(NormaliseConfig, raw.get("normalise", {}), "normalise"),
        classify=_build_section(ClassifyConfig, raw.get("classify", {}), "classify"),
        rare=_build_section(RareConfig, raw.get("rare", {}), "rare"),
        output_dir=raw.get("output_dir", "data/out"),
        save_crops=raw.get("save_crops", True),
    )
