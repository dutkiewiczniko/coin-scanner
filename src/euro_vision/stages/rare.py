"""Stage 4 — flag coins that may be rare.

The matching approach is deliberately not fixed yet. Backends register
themselves here and are selected by name in config, so the decision can be made
once the earlier stages are producing clean crops:

  none      — flags nothing. Default until a strategy is chosen.
  shortlist — flags every coin in the watchlist denominations for manual review.
              No model. Useful immediately: it turns a full tray into the ~handful
              of 1 and 2 euro coins actually worth eyeballing.
  embedding — nearest-neighbour against reference images of known rare issues.
              Scales to new rare coins without retraining. NOT IMPLEMENTED.
  metadata  — read country/year from the design, look up rarity in the database.
              NOT IMPLEMENTED.
"""

from __future__ import annotations

from typing import Callable

from ..config import RareConfig
from ..types import ScanResult
from .base import Stage

Backend = Callable[[ScanResult, RareConfig], ScanResult]

_BACKENDS: dict[str, Backend] = {}


def register(name: str) -> Callable[[Backend], Backend]:
    """Register a rare-detection backend under a config-selectable name."""

    def decorator(fn: Backend) -> Backend:
        _BACKENDS[name] = fn
        return fn

    return decorator


class RareStage(Stage):
    name = "rare"

    def __init__(self, config: RareConfig):
        self.config = config
        if config.backend not in _BACKENDS:
            available = ", ".join(sorted(_BACKENDS))
            raise ValueError(
                f"unknown rare backend: {config.backend} (available: {available})"
            )

    def describe(self) -> str:
        return f"rare[{self.config.backend}]"

    def run(self, result: ScanResult) -> ScanResult:
        return _BACKENDS[self.config.backend](result, self.config)


def _in_scope(coin, config: RareConfig) -> bool:
    """Whether this coin is one the rare stage should consider at all."""
    if not config.denominations:
        return True
    return coin.denomination in config.denominations


# -- backends -------------------------------------------------------------


@register("none")
def _none(result: ScanResult, config: RareConfig) -> ScanResult:
    return result


@register("shortlist")
def _shortlist(result: ScanResult, config: RareConfig) -> ScanResult:
    for coin in result.coins:
        if _in_scope(coin, config):
            coin.is_flagged = True
            coin.rare_score = 0.0
            coin.rare_notes = "watchlist denomination — manual review"
    return result


@register("embedding")
def _embedding(result: ScanResult, config: RareConfig) -> ScanResult:
    raise NotImplementedError(
        "embedding backend not implemented yet — see stages/rare.py. "
        "Needs: an embedding model at rare.weights and reference embeddings "
        "for the coins in rare.database."
    )


@register("metadata")
def _metadata(result: ScanResult, config: RareConfig) -> ScanResult:
    raise NotImplementedError(
        "metadata backend not implemented yet — see stages/rare.py. "
        "Needs a country/year recogniser before the database lookup is useful."
    )
