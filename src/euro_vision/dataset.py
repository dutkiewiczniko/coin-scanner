"""Export rectified trays and their detections as a YOLO training set.

Hand-labelling a tray of eighty coins is miserable, and unnecessary: the
watershed backend already finds most of them with no overlapping boxes. Exporting
those as labels turns the job from drawing eighty boxes into correcting a few,
which is the difference between a dataset that gets built and one that does not.

The labels are a starting point, not ground truth. Anything trained on
uncorrected output can only reproduce the current pipeline's mistakes — the point
is to fix the misses and errors in a labelling tool first.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2

from .config import Config
from .pipeline import Pipeline, load_image, save_image, side_from_path
from .stages import compute_calibration
from .types import DENOMINATIONS, format_denomination

#: Single-class mode: everything is just "coin".
CLASSES_COIN = ["coin"]
#: Eight-class mode, one per denomination.
CLASSES_DENOMINATION = [format_denomination(d) or str(d) for d in DENOMINATIONS]


#: Folder-per-class layout, as used by Keras `flow_from_directory` and
#: torchvision's `ImageFolder`. Chosen so that public coin datasets — which
#: overwhelmingly use it — can be merged in by copying folders, which matters
#: when the local data is a few hundred crops and thin in the rare classes.
CROP_LAYOUT = "imagefolder"


@dataclass
class ExportStats:
    images: int = 0
    instances: int = 0
    per_class: dict[str, int] = None  # type: ignore[assignment]
    skipped: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.per_class is None:
            self.per_class = {}
        if self.skipped is None:
            self.skipped = []


def export_crops(
    paths: Iterable[Path],
    out_dir: Path,
    config: Config,
    val_split: float = 0.2,
    seed: int = 0,
    unlabelled: str = "_unsorted",
) -> ExportStats:
    """Write individual coin crops into folder-per-class layout.

    This is the dataset a denomination classifier trains on, and it is a
    different shape from the detection export: one file per coin face rather
    than one per tray. Both faces are written, because whichever face carries
    the printed value is the one that settles the denomination — the cue that
    measuring diameters cannot reach.

    Coins the pipeline could not confidently classify go to `_unsorted` rather
    than being guessed at. Sorting those by hand is the labelling work, and
    putting a guess in a class folder would quietly poison the training set.
    """
    out_dir = Path(out_dir)
    stats = ExportStats()
    paths = list(paths)

    rng = random.Random(seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)
    split_at = max(1, int(len(shuffled) * (1 - val_split))) if len(shuffled) > 1 else 1
    validation = set(shuffled[split_at:])

    pipeline = Pipeline(config)
    tolerance = config.classify.crop_label_tolerance_mm

    for path in paths:
        subset = "val" if path in validation else "train"
        try:
            result = pipeline.run(path, side=side_from_path(path))
        except Exception as exc:  # noqa: BLE001 - report and continue
            stats.skipped.append(f"{path.name}: {exc}")
            continue

        for coin in result.coins:
            if coin.normalised is None:
                continue

            label = unlabelled
            if coin.denomination is not None and coin.diameter_mm is not None:
                from .stages.classify import DIAMETERS_MM

                gap = abs(coin.diameter_mm - DIAMETERS_MM[coin.denomination])
                if gap <= tolerance:
                    label = format_denomination(coin.denomination) or unlabelled

            folder = out_dir / subset / label.replace(" ", "_")
            folder.mkdir(parents=True, exist_ok=True)
            save_image(folder / f"{path.stem}_{coin.index:03d}.png", coin.normalised)
            stats.per_class[label] = stats.per_class.get(label, 0) + 1
            stats.instances += 1

        stats.images += 1

    return stats


def export_dataset(
    paths: Iterable[Path],
    out_dir: Path,
    config: Config,
    by_denomination: bool = False,
    val_split: float = 0.2,
    seed: int = 0,
) -> ExportStats:
    """Write a YOLO dataset from tray photos.

    Images are exported rectified rather than raw. The model then only ever sees
    coins at one scale and one viewpoint, which is a much easier problem than
    learning to cope with the camera moving — and rectification is already
    reliable, so there is no reason to make the network relearn it.
    """
    out_dir = Path(out_dir)
    classes = CLASSES_DENOMINATION if by_denomination else CLASSES_COIN
    stats = ExportStats()

    paths = list(paths)
    rng = random.Random(seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)
    split_at = max(1, int(len(shuffled) * (1 - val_split))) if len(shuffled) > 1 else 1
    validation = set(shuffled[split_at:])

    for subset in ("train", "val"):
        (out_dir / "images" / subset).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / subset).mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(config)

    for path in paths:
        subset = "val" if path in validation else "train"
        try:
            result = pipeline.run(path, side=side_from_path(path))
        except Exception as exc:  # noqa: BLE001 - report and continue
            stats.skipped.append(f"{path.name}: {exc}")
            continue

        image = load_image(path)
        if config.calibrate.enabled:
            calibration = compute_calibration(
                image, config.calibrate, result.meta.get("side", "a")
            )
            rectified = cv2.warpPerspective(
                image,
                calibration.homography,
                (calibration.width_px, calibration.height_px),
            )
        else:
            # No calibration: export the photo as shot. Detection labels are
            # still valid, they just describe an unrectified view.
            rectified = image
        height, width = rectified.shape[:2]

        lines = []
        for coin in result.coins:
            if by_denomination:
                if coin.denomination is None:
                    continue
                index = DENOMINATIONS.index(coin.denomination)
            else:
                index = 0

            det = coin.detection
            # YOLO wants a normalised centre and box size.
            box = det.radius * 2
            lines.append(
                f"{index} {det.x / width:.6f} {det.y / height:.6f} "
                f"{box / width:.6f} {box / height:.6f}"
            )
            name = classes[index]
            stats.per_class[name] = stats.per_class.get(name, 0) + 1

        stem = path.stem
        save_image(out_dir / "images" / subset / f"{stem}.png", rectified)
        (out_dir / "labels" / subset / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        stats.images += 1
        stats.instances += len(lines)

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(classes))
    (out_dir / "data.yaml").write_text(
        f"# Generated by `euro-vision export-dataset`.\n"
        f"# Labels are the pipeline's own detections and need correcting before\n"
        f"# training — a model fitted to them can only repeat their mistakes.\n"
        f"path: {out_dir.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )
    return stats
