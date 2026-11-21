"""Background monitor thread so the API/UI stays responsive during scans and imports."""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone

from .config import Settings, ensure_workspace
from .db import Database
from .importer import Importer
from .watcher import FolderWatcher, ScanReport

log = logging.getLogger(__name__)


class MonitorService:
    def __init__(self, settings: Settings, db: Database, job_runner=None, **watcher_kwargs):
        self.db = db
        self._scan_lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_scan_at: str | None = None
        self.last_error: str | None = None
        self.last_report: ScanReport | None = None
        self._watcher_kwargs = watcher_kwargs
        self.apply_settings(settings)
        self.jobs = job_runner if job_runner is not None else self._default_runner()

    def _default_runner(self):
        from .analysis.pipeline import analyze_recording
        from .jobs import JobRunner

        def analyze(job, progress, cancel):
            return analyze_recording(self.settings, self.db, job["recording_id"], progress, cancel)

        def pause_reason():
            from .gamecheck import game_running
            s = self.settings
            if s.pause_while_gaming and game_running(s.game_processes):
                return "game running - resumes when you close it"
            return None

        return JobRunner(self.db, {"analyze": analyze}, lambda: self.settings, should_pause=pause_reason)

    def apply_settings(self, settings: Settings) -> None:
        with self._scan_lock:
            old = getattr(self, "watcher", None)
            self.settings = settings
            ensure_workspace(settings)
            self.importer = Importer(settings, db=self.db)
            self.watcher = FolderWatcher(settings, self.db, self.importer, **self._watcher_kwargs)
            if old is not None:
                self.watcher._tracked = old._tracked  # keep stability timers across setting changes

    def scan_once(self) -> ScanReport:
        with self._scan_lock:
            try:
                rep = self.watcher.scan()
                self.last_report, self.last_error = rep, None
                if self.settings.auto_analyze:
                    for rid in rep.imported:
                        self.jobs.enqueue(rid)
                return rep
            except Exception as e:  # keep the loop alive; surface error in status
                log.exception("scan failed")
                self.last_error = str(e)
                self.db.add_event(f"Scan error: {e}", level="error")
                raise
            finally:
                self.last_scan_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def request_scan(self) -> None:
        self._wake.set()

    # ---- thread lifecycle ---------------------------------------------
    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="rogue-monitor", daemon=True)
        self._thread.start()
        self.jobs.start()
        if self.settings.auto_analyze:  # backlog: imported before Phase 2 / while analysis was off
            for r in self.db.recordings_needing_analysis():
                self.jobs.enqueue(r["id"])
        self.db.add_event("Folder monitor started: " + ", ".join(str(p) for p in self.settings.monitored_folders()))

    def stop(self, timeout: float = 10) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout)
        self.jobs.stop()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.scan_once()
            except Exception:
                pass
            self._wake.wait(self.settings.poll_interval_seconds)
            self._wake.clear()

    def status(self) -> dict:
        return {
            "running": self.running,
            "last_scan_at": self.last_scan_at,
            "last_error": self.last_error,
            "pending_files": self.watcher.pending_count,
            "jobs_running": self.jobs.running,
            "jobs_paused": getattr(self.jobs, "paused_reason", None),
            "monitored_folders": [str(p) for p in self.settings.monitored_folders()],
        }


def run_forever(service: MonitorService) -> None:
    service.start()
    try:
        while service.running:
            time.sleep(1)
    except KeyboardInterrupt:
        service.stop()
