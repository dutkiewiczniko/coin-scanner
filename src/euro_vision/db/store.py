"""SQLite access for the rare coin database."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable

from ..types import Coin, ScanResult

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class RareCoinStore:
    """Thin wrapper over the SQLite database.

    Usable as a context manager; commits on clean exit, rolls back on error.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")

    def __enter__(self) -> "RareCoinStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()
        self.close()

    def close(self) -> None:
        self.connection.close()

    def init_schema(self) -> None:
        self.connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.connection.commit()

    # -- rare coins -------------------------------------------------------

    def add_rare_coin(self, **fields: Any) -> int:
        """Insert an issue, ignoring exact duplicates. Returns its row id."""
        columns = ", ".join(fields)
        placeholders = ", ".join("?" for _ in fields)
        cursor = self.connection.execute(
            f"INSERT OR IGNORE INTO rare_coin ({columns}) VALUES ({placeholders})",
            tuple(fields.values()),
        )
        if cursor.lastrowid:
            return cursor.lastrowid
        # Already present — find the existing row.
        where = " AND ".join(f"{k} IS ?" for k in fields)
        row = self.connection.execute(
            f"SELECT id FROM rare_coin WHERE {where}", tuple(fields.values())
        ).fetchone()
        return int(row["id"])

    def rare_coins(self, denomination: int | None = None) -> list[sqlite3.Row]:
        if denomination is None:
            return self.connection.execute(
                "SELECT * FROM rare_coin ORDER BY denomination, country, year"
            ).fetchall()
        return self.connection.execute(
            "SELECT * FROM rare_coin WHERE denomination = ? ORDER BY country, year",
            (denomination,),
        ).fetchall()

    def count(self) -> int:
        return int(
            self.connection.execute("SELECT COUNT(*) FROM rare_coin").fetchone()[0]
        )

    # -- scan history -----------------------------------------------------

    def record_flags(self, result: ScanResult, crops: dict[int, str] | None = None) -> int:
        """Persist the flagged coins from a scan. Returns the number written."""
        crops = crops or {}
        rows: Iterable[tuple] = [
            (
                result.source,
                coin.index,
                crops.get(coin.index),
                coin.denomination,
                coin.rare_score,
                coin.rare_notes,
            )
            for coin in result.flagged
        ]
        cursor = self.connection.executemany(
            """
            INSERT INTO scan_flag
                (source_image, coin_index, crop_path, denomination, score, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.connection.commit()
        return cursor.rowcount

    def pending_flags(self, limit: int = 50) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT * FROM scan_flag
            WHERE reviewed = 0
            ORDER BY score DESC, scanned_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
