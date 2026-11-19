"""SQLite job/recording tracking. One shared connection guarded by a lock (WAL mode)."""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS recordings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    path          TEXT NOT NULL UNIQUE,
    filename      TEXT NOT NULL,
    library_path  TEXT,                 -- copy inside RAW GAMEPLAY when import_mode=copy
    source        TEXT DEFAULT 'unknown', -- shadowplay | obs | unknown
    status        TEXT NOT NULL,        -- waiting | retry | imported | duplicate | ignored | failed | missing
    size_bytes    INTEGER,
    mtime         REAL,
    fingerprint   TEXT,
    duration      REAL,
    width         INTEGER,
    height        INTEGER,
    fps           REAL,
    video_codec   TEXT,
    audio_codec   TEXT,
    container     TEXT,
    bit_rate      INTEGER,
    duplicate_of  INTEGER REFERENCES recordings(id),
    attempts      INTEGER NOT NULL DEFAULT 0,
    error         TEXT,
    detected_at   TEXT NOT NULL,
    imported_at   TEXT,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_recordings_fp ON recordings(fingerprint);
CREATE INDEX IF NOT EXISTS idx_recordings_status ON recordings(status);
CREATE INDEX IF NOT EXISTS idx_recordings_library ON recordings(library_path);

CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT NOT NULL,
    level        TEXT NOT NULL,
    recording_id INTEGER,
    message      TEXT NOT NULL
);
"""

DONE_STATUSES = ("imported", "duplicate", "ignored", "failed")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(SCHEMA)
            self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- helpers -------------------------------------------------------
    def _one(self, sql: str, args: tuple = ()) -> dict | None:
        with self._lock:
            r = self._conn.execute(sql, args).fetchone()
        return dict(r) if r else None

    def _all(self, sql: str, args: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, args).fetchall()]

    # ---- recordings ----------------------------------------------------
    def get_recording(self, rec_id: int) -> dict | None:
        return self._one("SELECT * FROM recordings WHERE id=?", (rec_id,))

    def get_by_path(self, path: str) -> dict | None:
        return self._one("SELECT * FROM recordings WHERE path=?", (path,))

    def is_library_path(self, path: str) -> bool:
        return self._one("SELECT 1 AS x FROM recordings WHERE library_path=?", (path,)) is not None

    def find_imported_by_fingerprint(self, fp: str, exclude_id: int | None = None) -> dict | None:
        return self._one(
            "SELECT * FROM recordings WHERE fingerprint=? AND status='imported' AND id<>? ORDER BY id LIMIT 1",
            (fp, exclude_id or -1),
        )

    def imported_in_dir_with_stem(self, directory: str, stem: str, exclude_id: int) -> list[dict]:
        rows = self._all("SELECT * FROM recordings WHERE status='imported' AND id<>?", (exclude_id,))
        out = []
        for r in rows:
            p = Path(r["path"])
            if str(p.parent) == directory and p.stem == stem:
                out.append(r)
        return out

    def insert_waiting(self, path: str, filename: str, size: int, mtime: float, source: str) -> int:
        ts = now_iso()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO recordings(path, filename, source, status, size_bytes, mtime, detected_at, updated_at)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (path, filename, source, "waiting", size, mtime, ts, ts),
            )
            return int(cur.lastrowid)

    def reset_to_waiting(self, rec_id: int, size: int, mtime: float) -> None:
        self.update_recording(
            rec_id, status="waiting", size_bytes=size, mtime=mtime, attempts=0, error=None,
            fingerprint=None, duplicate_of=None, imported_at=None,
        )

    def update_recording(self, rec_id: int, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = now_iso()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            self._conn.execute(f"UPDATE recordings SET {cols} WHERE id=?", (*fields.values(), rec_id))

    def list_recordings(self, status: str | None = None, limit: int = 200, offset: int = 0) -> list[dict]:
        if status:
            return self._all(
                "SELECT * FROM recordings WHERE status=? ORDER BY id DESC LIMIT ? OFFSET ?", (status, limit, offset)
            )
        return self._all("SELECT * FROM recordings ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset))

    def status_counts(self) -> dict[str, int]:
        rows = self._all("SELECT status, COUNT(*) AS n FROM recordings GROUP BY status")
        return {r["status"]: r["n"] for r in rows}

    # ---- events --------------------------------------------------------
    def add_event(self, message: str, level: str = "info", recording_id: int | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO events(ts, level, recording_id, message) VALUES(?,?,?,?)",
                (now_iso(), level, recording_id, message),
            )

    def list_events(self, limit: int = 100) -> list[dict]:
        return self._all("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))
