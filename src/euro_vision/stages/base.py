"""Stage interface.

Every pipeline stage is a callable that takes a ScanResult and returns it
enriched. Keeping the signature uniform means backends can be swapped (Hough
for YOLO, stub classifier for a trained CNN) without touching the orchestrator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import ScanResult


class Stage(ABC):
    """Base class for pipeline stages."""

    #: Human-readable name used in logs and result metadata.
    name: str = "stage"

    @abstractmethod
    def run(self, result: ScanResult) -> ScanResult:
        """Process the scan in place and return it."""

    def __call__(self, result: ScanResult) -> ScanResult:
        return self.run(result)

    def describe(self) -> str:
        """Short description of the active backend, for `--verbose` output."""
        return self.name
