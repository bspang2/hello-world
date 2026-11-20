"""SQLite job/recording tracking. One shared connection guarded by a lock (WAL mode)."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2

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

CREATE TABLE IF NOT EXISTS jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id    INTEGER REFERENCES recordings(id),
    kind            TEXT NOT NULL,          -- analyze (render in Phase 3)
    status          TEXT NOT NULL,          -- queued | running | done | failed | cancelled
    progress        REAL NOT NULL DEFAULT 0,
    stage           TEXT,
    attempts        INTEGER NOT NULL DEFAULT 0,
    not_before      REAL NOT NULL DEFAULT 0,
    error           TEXT,
    created_at      TEXT NOT NULL,
    started_at      TEXT,
    finished_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);

CREATE TABLE IF NOT EXISTS highlights (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id     INTEGER NOT NULL REFERENCES recordings(id),
    start            REAL NOT NULL,
    end              REAL NOT NULL,
    peak             REAL NOT NULL,
    type             TEXT NOT NULL,
    types            TEXT NOT NULL DEFAULT '[]',   -- JSON list
    confidence       REAL NOT NULL,
    local_confidence REAL,
    ai_confidence    REAL,
    description      TEXT,
    ai_description   TEXT,
    signals          TEXT NOT NULL DEFAULT '{}',   -- JSON
    source           TEXT NOT NULL DEFAULT 'local', -- local | ai | manual
    status           TEXT NOT NULL,                 -- selected | below_threshold | approved | rejected
    user_edited      INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_highlights_rec ON highlights(recording_id);

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
            self._migrate()
            self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def _migrate(self) -> None:
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(recordings)")}
        for name in ("analysis_status", "analyzed_at"):
            if name not in cols:  # v1 -> v2
                self._conn.execute(f"ALTER TABLE recordings ADD COLUMN {name} TEXT")

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

    _REC_SELECT = ("SELECT r.*, (SELECT COUNT(*) FROM highlights h WHERE h.recording_id=r.id "
                   "AND h.status IN ('selected','approved')) AS highlight_count, "
                   "(SELECT j.progress FROM jobs j WHERE j.recording_id=r.id AND j.status='running' "
                   "ORDER BY j.id DESC LIMIT 1) AS analysis_progress FROM recordings r")

    def list_recordings(self, status: str | None = None, limit: int = 200, offset: int = 0) -> list[dict]:
        if status:
            return self._all(self._REC_SELECT + " WHERE r.status=? ORDER BY r.id DESC LIMIT ? OFFSET ?",
                             (status, limit, offset))
        return self._all(self._REC_SELECT + " ORDER BY r.id DESC LIMIT ? OFFSET ?", (limit, offset))

    def recordings_needing_analysis(self) -> list[dict]:
        return self._all("SELECT * FROM recordings WHERE status='imported' AND analysis_status IS NULL ORDER BY id")

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

    # ---- jobs ----------------------------------------------------------
    def enqueue_job(self, recording_id: int, kind: str = "analyze") -> int:
        with self._lock:
            r = self._conn.execute(
                "SELECT id FROM jobs WHERE recording_id=? AND kind=? AND status IN ('queued','running')",
                (recording_id, kind)).fetchone()
            if r:
                return int(r["id"])
            cur = self._conn.execute(
                "INSERT INTO jobs(recording_id, kind, status, created_at) VALUES(?,?,?,?)",
                (recording_id, kind, "queued", now_iso()))
            return int(cur.lastrowid)

    def claim_next_job(self, now: float) -> dict | None:
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM jobs WHERE status='queued' AND not_before<=? ORDER BY id LIMIT 1", (now,)).fetchone()
            if not r:
                return None
            self._conn.execute(
                "UPDATE jobs SET status='running', attempts=attempts+1, started_at=?, progress=0, error=NULL "
                "WHERE id=?", (now_iso(), r["id"]))
            return self.get_job(r["id"])

    def get_job(self, job_id: int) -> dict | None:
        return self._one("SELECT * FROM jobs WHERE id=?", (job_id,))

    def update_job(self, job_id: int, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            self._conn.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))

    def list_jobs(self, limit: int = 100, status: str | None = None) -> list[dict]:
        if status:
            return self._all("SELECT * FROM jobs WHERE status=? ORDER BY id DESC LIMIT ?", (status, limit))
        return self._all("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,))

    def requeue_interrupted_jobs(self) -> int:
        with self._lock:
            return self._conn.execute(
                "UPDATE jobs SET status='queued', attempts=MAX(attempts-1,0) WHERE status='running'").rowcount

    # ---- highlights ----------------------------------------------------
    _HL_JSON = ("types", "signals")

    def _hl_out(self, r: dict | None) -> dict | None:
        if r:
            for k in self._HL_JSON:
                r[k] = json.loads(r[k] or ("[]" if k == "types" else "{}"))
        return r

    def list_highlights(self, recording_id: int | None = None, status: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM highlights WHERE 1=1", []
        if recording_id is not None:
            sql += " AND recording_id=?"
            args.append(recording_id)
        if status:
            sql += " AND status=?"
            args.append(status)
        rows = self._all(sql + " ORDER BY recording_id, start", tuple(args))
        return [self._hl_out(r) for r in rows]

    def get_highlight(self, hl_id: int) -> dict | None:
        return self._hl_out(self._one("SELECT * FROM highlights WHERE id=?", (hl_id,)))

    def insert_highlight(self, recording_id: int, h: dict) -> int:
        ts = now_iso()
        row = {
            "recording_id": recording_id, "start": h["start"], "end": h["end"], "peak": h.get("peak", h["start"]),
            "type": h["type"], "types": json.dumps(h.get("types", [h["type"]])), "confidence": h["confidence"],
            "local_confidence": h.get("local_confidence"), "ai_confidence": h.get("ai_confidence"),
            "description": h.get("description"), "ai_description": h.get("ai_description"),
            "signals": json.dumps(h.get("signals", {})), "source": h.get("source", "local"),
            "status": h["status"], "user_edited": int(h.get("user_edited", 0)),
            "created_at": ts, "updated_at": ts,
        }
        cols = ", ".join(row)
        with self._lock:
            cur = self._conn.execute(f"INSERT INTO highlights({cols}) VALUES({','.join('?' * len(row))})",
                                     tuple(row.values()))
            return int(cur.lastrowid)

    def update_highlight(self, hl_id: int, **fields: Any) -> None:
        for k in self._HL_JSON:
            if k in fields and not isinstance(fields[k], str):
                fields[k] = json.dumps(fields[k])
        fields["updated_at"] = now_iso()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            self._conn.execute(f"UPDATE highlights SET {cols} WHERE id=?", (*fields.values(), hl_id))

    def delete_highlight(self, hl_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM highlights WHERE id=?", (hl_id,))

    def replace_auto_highlights(self, recording_id: int, highlights: list[dict]) -> list[int]:
        """Replace machine-made highlights; keep anything the user created or edited.

        A new highlight overlapping a user-touched one by >50% is dropped (the user's version wins).
        """
        with self._lock:
            self._conn.execute(
                "DELETE FROM highlights WHERE recording_id=? AND user_edited=0 AND source<>'manual'",
                (recording_id,))
            kept = self.list_highlights(recording_id)
            ids = []
            for h in highlights:
                if any(_overlap(h, k) > 0.5 for k in kept):
                    continue
                ids.append(self.insert_highlight(recording_id, h))
            return ids

    def highlight_count(self, statuses: tuple[str, ...] = ("selected", "approved")) -> int:
        q = ",".join("?" * len(statuses))
        r = self._one(f"SELECT COUNT(*) AS n FROM highlights WHERE status IN ({q})", statuses)
        return r["n"] if r else 0


def _overlap(a: dict, b: dict) -> float:
    inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    shorter = max(1e-6, min(a["end"] - a["start"], b["end"] - b["start"]))
    return inter / shorter
