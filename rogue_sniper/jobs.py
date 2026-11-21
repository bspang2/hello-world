"""Background job runner: one worker thread, DB-backed queue, progress, cancel, retries.

Jobs survive restarts: anything 'running' when the app stopped is re-queued on start.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from .config import FAILED_DIR, Settings, unique_path
from .db import Database, now_iso

log = logging.getLogger(__name__)
Handler = Callable[[dict, Callable[[float, str], None], threading.Event], dict | None]


class JobRunner:
    def __init__(self, db: Database, handlers: dict[str, Handler], settings_getter: Callable[[], Settings],
                 max_attempts: int = 3, poll: float = 1.0, retry_delay: float = 30.0,
                 should_pause: Callable[[], str | None] | None = None):
        self.db = db
        self.handlers = handlers
        self.settings = settings_getter
        self.max_attempts = max_attempts
        self.poll = poll
        self.retry_delay = retry_delay
        self.should_pause = should_pause or (lambda: None)  # returns a reason string while paused
        self.paused_reason: str | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._current: tuple[int, threading.Event] | None = None
        self._lock = threading.Lock()

    # ---- control -------------------------------------------------------
    def enqueue(self, recording_id: int, kind: str = "analyze") -> int:
        jid = self.db.enqueue_job(recording_id, kind)
        if kind == "analyze":
            self.db.update_recording(recording_id, analysis_status="queued")
        self._wake.set()
        return jid

    def cancel(self, job_id: int) -> bool:
        job = self.db.get_job(job_id)
        if not job or job["status"] not in ("queued", "running"):
            return False
        with self._lock:
            if self._current and self._current[0] == job_id:
                self._current[1].set()
                return True
        self.db.update_job(job_id, status="cancelled", finished_at=now_iso())
        self.db.update_recording(job["recording_id"], analysis_status="cancelled")
        return True

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> None:
        if self.running:
            return
        n = self.db.requeue_interrupted_jobs()
        if n:
            self.db.add_event(f"Resumed {n} interrupted job(s)")
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="rogue-jobs", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 15) -> None:
        self._stop.set()
        self._wake.set()
        with self._lock:
            if self._current:
                self._current[1].set()
        if self._thread:
            self._thread.join(timeout)

    def run_pending_once(self) -> bool:
        """Run one queued job synchronously (tests / CLI). Returns False if none was ready."""
        job = self.db.claim_next_job(time.time())
        if not job:
            return False
        self._execute(job)
        return True

    # ---- worker --------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.paused_reason = self.should_pause()
                if self.paused_reason:
                    self._wake.wait(max(self.poll, 5.0))
                    self._wake.clear()
                    continue
                if self.run_pending_once():
                    continue
            except Exception:
                log.exception("job loop error")
            self._wake.wait(self.poll)
            self._wake.clear()

    def _execute(self, job: dict) -> None:
        from .analysis.decode import Cancelled
        cancel = threading.Event()
        with self._lock:
            self._current = (job["id"], cancel)
        last = [0.0, None]

        def progress(frac: float, stage: str) -> None:
            # Pause mid-job (e.g. you started Fortnite): block here; FFmpeg waits on its pipe.
            while not cancel.is_set() and not self._stop.is_set():
                reason = self.should_pause()
                self.paused_reason = reason
                if not reason:
                    break
                self.db.update_job(job["id"], stage=f"paused: {reason}")
                last[1] = None
                cancel.wait(min(5.0, max(self.poll, 0.05)))
            now = time.monotonic()
            if stage != last[1] or now - last[0] > 0.5 or frac >= 1:
                self.db.update_job(job["id"], progress=round(float(frac), 3), stage=stage)
                last[0], last[1] = now, stage

        rec_id = job["recording_id"]
        try:
            handler = self.handlers[job["kind"]]
            self.db.update_job(job["id"], stage="starting")
            handler(job, progress, cancel)
            self.db.update_job(job["id"], status="done", progress=1.0, stage="done", finished_at=now_iso())
        except Cancelled:
            if self._stop.is_set():  # app shutting down: resume next start
                self.db.update_job(job["id"], status="queued", attempts=max(0, job["attempts"] - 1))
                self.db.update_recording(rec_id, analysis_status="queued")
            else:
                self.db.update_job(job["id"], status="cancelled", finished_at=now_iso(), stage="cancelled")
                self.db.update_recording(rec_id, analysis_status="cancelled")
                self.db.add_event(f"Job #{job['id']} cancelled", recording_id=rec_id)
        except Exception as e:
            log.exception("job %s failed", job["id"])
            err = f"{type(e).__name__}: {e}"[:500]
            if job["attempts"] < self.max_attempts:
                self.db.update_job(job["id"], status="queued", error=err,
                                   not_before=time.time() + self.retry_delay * job["attempts"])
                self.db.update_recording(rec_id, analysis_status="queued")
                self.db.add_event(f"Analysis of recording #{rec_id} failed ({err}); retrying "
                                  f"({job['attempts']}/{self.max_attempts})", level="warning", recording_id=rec_id)
            else:
                self.db.update_job(job["id"], status="failed", error=err, finished_at=now_iso())
                self.db.update_recording(rec_id, analysis_status="failed")
                self.db.add_event(f"Analysis of recording #{rec_id} failed: {err}", level="error",
                                  recording_id=rec_id)
                self._failure_report(rec_id, err)
        finally:
            with self._lock:
                self._current = None

    def _failure_report(self, rec_id: int, err: str) -> None:
        try:
            rec = self.db.get_recording(rec_id) or {}
            d = self.settings().dir(FAILED_DIR)
            d.mkdir(parents=True, exist_ok=True)
            name = (rec.get("filename") or f"recording {rec_id}").rsplit(".", 1)[0]
            unique_path(d / f"{name} - analysis error.txt").write_text(
                f"Recording: {rec.get('path')}\nError: {err}\nTime: {now_iso()}\n\n"
                "The original recording was NOT modified. Re-run analysis from the dashboard or with\n"
                f"`python -m rogue_sniper analyze {rec_id}`.\n", encoding="utf-8")
        except OSError:
            log.exception("could not write failure report")
