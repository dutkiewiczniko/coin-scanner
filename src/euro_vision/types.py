"""Data structures passed between pipeline stages.

Each stage takes the output of the previous one and returns an enriched copy,
so a `Coin` accumulates fields as it moves down the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np

# Denominations in cents, smallest to largest.
DENOMINATIONS = [1, 2, 5, 10, 20, 50, 100, 200]


@dataclass
class Detection:
    """A coin located in the tray image."""

    x: int
    y: int
    radius: int
    confidence: float = 1.0

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """(x1, y1, x2, y2), clamped to non-negative."""
        return (
            max(0, self.x - self.radius),
            max(0, self.y - self.radius),
            self.x + self.radius,
            self.y + self.radius,
        )


@dataclass
class Coin:
    """A single coin as it moves through the pipeline."""

    index: int
    detection: Detection
    image: Optional[np.ndarray] = None

    # Filled by the normalisation stage.
    normalised: Optional[np.ndarray] = None
    rotation: float = 0.0
    diameter_mm: Optional[float] = None

    # Filled by the classification stage.
    denomination: Optional[int] = None  # cents
    denomination_confidence: float = 0.0

    # Filled by the rare detection stage.
    is_flagged: bool = False
    rare_match: Optional[str] = None
    rare_score: float = 0.0
    rare_notes: str = ""

    def to_record(self) -> dict[str, Any]:
        """JSON/CSV-safe view — drops the image arrays."""
        return {
            "index": self.index,
            "x": self.detection.x,
            "y": self.detection.y,
            "radius": self.detection.radius,
            "detection_confidence": round(self.detection.confidence, 4),
            "rotation": round(self.rotation, 2),
            "diameter_mm": self.diameter_mm,
            "denomination": self.denomination,
            "denomination_label": format_denomination(self.denomination),
            "denomination_confidence": round(self.denomination_confidence, 4),
            "is_flagged": self.is_flagged,
            "rare_match": self.rare_match,
            "rare_score": round(self.rare_score, 4),
            "rare_notes": self.rare_notes,
        }


@dataclass
class ScanResult:
    """Everything produced from one tray image."""

    source: str
    coins: list[Coin] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def flagged(self) -> list[Coin]:
        return [c for c in self.coins if c.is_flagged]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "meta": self.meta,
            "coin_count": len(self.coins),
            "flagged_count": len(self.flagged),
            "coins": [c.to_record() for c in self.coins],
        }


def format_denomination(cents: Optional[int]) -> Optional[str]:
    """1 -> '1c', 200 -> '2 EUR'."""
    if cents is None:
        return None
    if cents < 100:
        return f"{cents}c"
    return f"{cents // 100} EUR"


__all__ = [
    "DENOMINATIONS",
    "Detection",
    "Coin",
    "ScanResult",
    "format_denomination",
    "asdict",
]
