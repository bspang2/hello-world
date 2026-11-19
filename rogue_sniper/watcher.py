"""Polling folder monitor that only hands *finished* recordings to the importer.

Why polling instead of filesystem events: ShadowPlay/OBS write multi-GB files for many
minutes; events fire constantly while writing and are unreliable on network/external
drives. A cheap poll (stat only) every few seconds is robust and still near-instant.

A file is considered finished when ALL hold:
  * size > 0 and size + mtime unchanged for `stable_seconds`
  * it is not locked by another process (Windows exclusive-open check)
  * ffprobe can read it (checked in the importer; failures are retried with backoff)
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from .config import RAW_DIR, WORKSPACE_DIRS, Settings
from .db import DONE_STATUSES, Database
from .importer import Importer
from .media import detect_source, is_file_locked

log = logging.getLogger(__name__)

# Temp/partial names written by recorders/copy tools.
_SKIP_SUFFIXES = (".partial", ".part", ".tmp", ".crdownload")


@dataclass
class _Tracked:
    rec_id: int
    size: int
    mtime: float
    last_change: float
    not_before: float = 0.0  # retry backoff


@dataclass
class ScanReport:
    seen: int = 0
    new: int = 0
    waiting: int = 0
    imported: list[int] = field(default_factory=list)
    other: dict[str, int] = field(default_factory=dict)


class FolderWatcher:
    def __init__(self, settings: Settings, db: Database, importer: Importer,
                 clock: Callable[[], float] = time.monotonic,
                 lock_check: Callable[[Path], bool] = is_file_locked):
        self.settings = settings
        self.db = db
        self.importer = importer
        self.clock = clock
        self.lock_check = lock_check
        self._tracked: dict[str, _Tracked] = {}

    @property
    def pending_count(self) -> int:
        return len(self._tracked)

    # ---- discovery -----------------------------------------------------
    def _excluded_roots(self) -> list[str]:
        ws = self.settings.workspace
        return [os.path.normcase(os.path.abspath(ws / d)) for d in WORKSPACE_DIRS if d != RAW_DIR]

    def iter_candidates(self) -> Iterator[Path]:
        exts = set(self.settings.extensions)
        excluded = self._excluded_roots()
        visited: set[str] = set()  # overlapping watch folders must not yield a file twice
        for root in self.settings.monitored_folders():
            if not root.is_dir():
                continue
            walker = os.walk(root) if self.settings.recursive else [(str(root), [], os.listdir(root))]
            for dirpath, dirnames, filenames in walker:
                norm = os.path.normcase(os.path.abspath(dirpath))
                if any(norm == ex or norm.startswith(ex + os.sep) for ex in excluded):
                    dirnames[:] = []
                    continue
                if norm in visited:
                    dirnames[:] = []
                    continue
                visited.add(norm)
                for name in filenames:
                    low = name.lower()
                    if low.endswith(_SKIP_SUFFIXES) or name.startswith("."):
                        continue
                    if os.path.splitext(low)[1] in exts:
                        yield Path(dirpath) / name

    # ---- main loop body ------------------------------------------------
    def scan(self) -> ScanReport:
        rep = ScanReport()
        now = self.clock()
        present: set[str] = set()

        for p in self.iter_candidates():
            key = str(p.resolve())
            try:
                st = p.stat()
            except OSError:
                continue  # vanished mid-scan
            rep.seen += 1
            present.add(key)
            if self.db.is_library_path(key):
                continue  # our own copy inside RAW GAMEPLAY

            row = self.db.get_by_path(key)
            if row is None:
                rec_id = self.db.insert_waiting(key, p.name, st.st_size, st.st_mtime, detect_source(p))
                self.db.add_event(f"Detected new recording {p.name}", recording_id=rec_id)
                self._tracked[key] = _Tracked(rec_id, st.st_size, st.st_mtime, now)
                rep.new += 1
                continue

            if row["status"] in DONE_STATUSES or row["status"] == "missing":
                changed = row["size_bytes"] != st.st_size or abs((row["mtime"] or 0) - st.st_mtime) > 1e-6
                if not changed and row["status"] != "missing":
                    continue
                # File was overwritten / re-recorded under the same name: start again.
                self.db.reset_to_waiting(row["id"], st.st_size, st.st_mtime)
                self.db.add_event(f"{p.name} changed on disk - re-importing", recording_id=row["id"])
                self._tracked[key] = _Tracked(row["id"], st.st_size, st.st_mtime, now)
                continue

            # waiting / retry
            t = self._tracked.get(key)
            if t is None:  # e.g. app restarted while file was pending
                t = self._tracked[key] = _Tracked(row["id"], st.st_size, st.st_mtime, now)
                continue
            if st.st_size != t.size or st.st_mtime != t.mtime:
                t.size, t.mtime, t.last_change = st.st_size, st.st_mtime, now
                self.db.update_recording(t.rec_id, size_bytes=st.st_size, mtime=st.st_mtime)
                continue
            if st.st_size == 0 or now - t.last_change < self.settings.stable_seconds or now < t.not_before:
                continue
            if self.lock_check(p):
                t.last_change = now  # still being written; wait another full window
                continue

            res = self.importer.import_recording(t.rec_id)
            if res.status == "retry":
                attempt = (self.db.get_recording(t.rec_id) or {}).get("attempts", 1)
                t.not_before = now + self.settings.stable_seconds * attempt  # linear backoff
            else:
                self._tracked.pop(key, None)
                if res.status == "imported":
                    rep.imported.append(t.rec_id)
                else:
                    rep.other[res.status] = rep.other.get(res.status, 0) + 1

        # Files that disappeared while still pending
        for key in list(self._tracked):
            if key not in present:
                t = self._tracked.pop(key)
                self.db.update_recording(t.rec_id, status="missing", error="file removed before import")
        rep.waiting = len(self._tracked)
        return rep

    def force_retry(self, rec_id: int) -> bool:
        row = self.db.get_recording(rec_id)
        if not row:
            return False
        self.db.update_recording(rec_id, status="retry", attempts=0, error=None)
        self._tracked.pop(row["path"], None)  # re-tracked (fresh stability window) on next scan
        return True
