from __future__ import annotations

from pathlib import Path

import cv2
import pytest
from conftest import TRAY_H_MM, TRAY_W_MM, make_marked_tray

from euro_vision.config import CalibrateConfig, Config
from euro_vision.dataset import export_crops, export_dataset
from euro_vision.stages.classify import DIAMETERS_MM

COINS = [
    (30.0, 30.0, DIAMETERS_MM[1]),
    (90.0, 32.0, DIAMETERS_MM[200]),
    (150.0, 30.0, DIAMETERS_MM[10]),
    (40.0, 90.0, DIAMETERS_MM[50]),
]


@pytest.fixture
def config(tmp_path) -> Config:
    cfg = Config(
        calibrate=CalibrateConfig(
            enabled=True,
            tray_width_mm=TRAY_W_MM,
            tray_height_mm=TRAY_H_MM,
            pixels_per_mm=8.0,
        )
    )
    cfg.output_dir = str(tmp_path / "out")
    cfg.classify.backend = "diameter"
    return cfg


@pytest.fixture
def images(tmp_path) -> list[Path]:
    paths = []
    for i in range(3):
        image, _ = make_marked_tray(COINS, pixels_per_mm=6.0)
        path = tmp_path / f"{i}_a.png"
        cv2.imwrite(str(path), image)
        paths.append(path)
    return paths


# -- detection export -----------------------------------------------------


def test_export_writes_yolo_layout(config, images, tmp_path):
    out = tmp_path / "ds"
    stats = export_dataset(images, out, config, val_split=0.34)

    assert stats.images == len(images)
    assert stats.instances == len(images) * len(COINS)
    assert (out / "data.yaml").exists()

    labels = list((out / "labels").rglob("*.txt"))
    assert len(labels) == len(images)
    train = list((out / "images" / "train").glob("*.png"))
    val = list((out / "images" / "val").glob("*.png"))
    assert train and val and len(train) + len(val) == len(images)


def test_labels_are_normalised_yolo_boxes(config, images, tmp_path):
    out = tmp_path / "ds"
    export_dataset(images, out, config)

    text = next((out / "labels").rglob("*.txt")).read_text(encoding="utf-8")
    rows = [line.split() for line in text.strip().splitlines()]
    assert len(rows) == len(COINS)
    for row in rows:
        assert len(row) == 5
        assert row[0] == "0"  # single "coin" class by default
        assert all(0.0 <= float(v) <= 1.0 for v in row[1:])


def test_denomination_mode_uses_one_class_per_value(config, images, tmp_path):
    out = tmp_path / "ds"
    stats = export_dataset(images, out, config, by_denomination=True)

    assert set(stats.per_class) <= {"1c", "2c", "5c", "10c", "20c", "50c",
                                    "1 EUR", "2 EUR"}
    text = next((out / "labels").rglob("*.txt")).read_text(encoding="utf-8")
    classes = {line.split()[0] for line in text.strip().splitlines()}
    assert len(classes) > 1  # the coins differ, so the labels must too


def test_unreadable_image_is_skipped_not_fatal(config, images, tmp_path):
    broken = tmp_path / "broken_a.png"
    broken.write_bytes(b"not an image")
    out = tmp_path / "ds"

    stats = export_dataset(images + [broken], out, config)
    assert stats.images == len(images)
    assert any("broken_a" in s for s in stats.skipped)


# -- crop export ----------------------------------------------------------


def test_crops_are_written_one_folder_per_class(config, images, tmp_path):
    out = tmp_path / "crops"
    stats = export_crops(images, out, config, val_split=0.34)

    assert stats.instances == len(images) * len(COINS)
    subsets = {p.name for p in out.iterdir()}
    assert subsets == {"train", "val"}

    files = list(out.rglob("*.png"))
    assert len(files) == stats.instances
    # Every crop sits inside a directory named for its class.
    assert all(f.parent.name for f in files)


def test_uncertain_crops_go_to_unsorted_not_a_guess(config, tmp_path):
    """A guessed label written into a class folder is a hidden bad example.

    20.50 mm sits 0.75 mm from both 10c (19.75) and 5c (21.25), so no honest
    label exists for it and it must not be filed under either.
    """
    between = [(60.0, 60.0, 20.50)]
    image, _ = make_marked_tray(between, pixels_per_mm=6.0)
    path = tmp_path / "amb_a.png"
    cv2.imwrite(str(path), image)

    out = tmp_path / "crops"
    stats = export_crops([path], out, config)

    assert stats.instances == 1
    assert set(stats.per_class) == {"_unsorted"}


def test_generous_tolerance_labels_crops(config, images, tmp_path):
    config.classify.crop_label_tolerance_mm = 5.0
    out = tmp_path / "crops"
    stats = export_crops(images, out, config)

    assert "_unsorted" not in stats.per_class
    assert set(stats.per_class) <= {"1c", "2c", "5c", "10c", "20c", "50c",
                                    "1 EUR", "2 EUR"}


def test_class_folder_names_have_no_spaces(config, images, tmp_path):
    """Folder names become class labels, so spaces invite tooling trouble."""
    config.classify.crop_label_tolerance_mm = 5.0
    out = tmp_path / "crops"
    export_crops(images, out, config)

    for folder in {p.parent for p in out.rglob("*.png")}:
        assert " " not in folder.name
