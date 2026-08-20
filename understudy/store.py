"""sqlite persistence. One row per listing, the model JSON in a text column."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from understudy.models import Listing

SCHEMA = """
CREATE TABLE IF NOT EXISTS listings (
    id TEXT PRIMARY KEY,
    sku_id TEXT,
    sold INTEGER NOT NULL DEFAULT 0,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sku ON listings(sku_id, sold);
"""


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: FastAPI serves sync handlers from a threadpool,
        # so the connection outlives the thread that opened it. Writes are
        # serialised by _lock; reads are safe under SQLite's default isolation.
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    def upsert(self, listings: list[Listing]) -> None:
        rows = [
            (l.id, l.sku_id, 1 if l.sold_price is not None else 0, l.model_dump_json())
            for l in listings
        ]
        with self._lock:
            self.conn.executemany(
                "INSERT INTO listings (id, sku_id, sold, data) VALUES (?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "sku_id=excluded.sku_id, sold=excluded.sold, data=excluded.data",
                rows,
            )
            self.conn.commit()

    @staticmethod
    def _row(r) -> Listing:
        return Listing.model_validate_json(r[0])

    def get(self, listing_id: str) -> Listing | None:
        r = self.conn.execute("SELECT data FROM listings WHERE id=?", (listing_id,)).fetchone()
        return self._row(r) if r else None

    def all(self) -> list[Listing]:
        return [self._row(r) for r in self.conn.execute("SELECT data FROM listings")]

    def by_sku(self, sku_id: str, sold: bool | None = None) -> list[Listing]:
        q = "SELECT data FROM listings WHERE sku_id=?"
        args: list = [sku_id]
        if sold is not None:
            q += " AND sold=?"
            args.append(1 if sold else 0)
        return [self._row(r) for r in self.conn.execute(q, args)]

    def sku_ids(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT sku_id FROM listings WHERE sku_id IS NOT NULL"
        )
        return [r[0] for r in rows]

    def export_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = [json.loads(l.model_dump_json()) for l in self.all()]
        Path(path).write_text(json.dumps(payload, indent=2))

    def import_json(self, path: str | Path) -> None:
        raw = json.loads(Path(path).read_text())
        self.upsert([Listing.model_validate(r) for r in raw])
