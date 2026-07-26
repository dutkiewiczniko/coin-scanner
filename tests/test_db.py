from __future__ import annotations

from pathlib import Path

import pytest

from euro_vision.db import RareCoinStore
from euro_vision.types import Coin, Detection, ScanResult


@pytest.fixture
def store(tmp_path):
    with RareCoinStore(tmp_path / "rare.sqlite") as s:
        s.init_schema()
        yield s


def test_schema_starts_empty(store):
    assert store.count() == 0


def test_add_rare_coin_is_idempotent(store):
    fields = dict(
        country="Monaco",
        denomination=200,
        year=2007,
        variant="commemorative",
        name="Grace Kelly",
    )
    first = store.add_rare_coin(**fields)
    second = store.add_rare_coin(**fields)
    assert first == second
    assert store.count() == 1


def test_filter_by_denomination(store):
    store.add_rare_coin(country="Monaco", denomination=200, name="A")
    store.add_rare_coin(country="Cyprus", denomination=100, name="B")
    assert len(store.rare_coins(200)) == 1
    assert len(store.rare_coins()) == 2


def test_record_flags_persists_only_flagged_coins(store):
    result = ScanResult(source="tray_01.jpg")
    for i in range(3):
        coin = Coin(index=i, detection=Detection(x=i * 10, y=0, radius=50))
        coin.denomination = 200
        coin.is_flagged = i == 1
        coin.rare_notes = "watchlist"
        result.coins.append(coin)

    written = store.record_flags(result, crops={1: "crops/coin_001.png"})
    assert written == 1

    pending = store.pending_flags()
    assert len(pending) == 1
    assert pending[0]["coin_index"] == 1
    assert pending[0]["crop_path"] == "crops/coin_001.png"


def test_seed_csv_loads(tmp_path):
    from euro_vision.cli import _read_seed_csv

    rows = _read_seed_csv(Path("data/rare_coins_seed.csv"))
    assert rows
    assert all(isinstance(row["denomination"], int) for row in rows)
    assert all(row["country"] for row in rows)
