"""Writing scan results to disk."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import Config
from .pipeline import save_image
from .types import ScanResult


def write_results(result: ScanResult, config: Config) -> dict[str, str]:
    """Write JSON, CSV and (optionally) per-coin crops.

    Returns a mapping of artefact name -> path.
    """
    out_dir = Path(config.output_dir) / Path(result.source).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}

    json_path = out_dir / "results.json"
    json_path.write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8"
    )
    written["json"] = str(json_path)

    csv_path = out_dir / "results.csv"
    records = [coin.to_record() for coin in result.coins]
    if records:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        written["csv"] = str(csv_path)

    if config.save_crops:
        crops_dir = out_dir / "crops"
        for coin in result.coins:
            image = coin.normalised if coin.normalised is not None else coin.image
            if image is None:
                continue
            suffix = "_flagged" if coin.is_flagged else ""
            save_image(crops_dir / f"coin_{coin.index:03d}{suffix}.png", image)
        written["crops"] = str(crops_dir)

    return written
