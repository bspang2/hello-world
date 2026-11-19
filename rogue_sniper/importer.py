"""Import a stable recording: fingerprint, validate with ffprobe, dedupe, optionally copy.

Originals are only ever *read*. Nothing here writes to, moves or deletes the source file.
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import FAILED_DIR, RAW_DIR, Settings, unique_path
from .db import Database, now_iso
from .media import ProbeError, probe, quick_fingerprint

log = logging.getLogger(__name__)

# Container preference when OBS remuxes X.mkv -> X.mp4 next to it.
_CONTAINER_PREFERENCE = {".mp4": 0, ".mkv": 1}


@dataclass
class ImportResult:
    status: str          # imported | duplicate | ignored | retry | failed
    message: str = ""


class Importer:
    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db

    def import_recording(self, rec_id: int) -> ImportResult:
        rec = self.db.get_recording(rec_id)
        if rec is None:
            return ImportResult("failed", "record vanished")
        path = Path(rec["path"])
        attempts = rec["attempts"] + 1
        self.db.update_recording(rec_id, attempts=attempts)

        if not path.exists():
            self.db.update_recording(rec_id, status="missing", error="file disappeared before import")
            return ImportResult("missing", "file disappeared")

        st = path.stat()
        try:
            fp = quick_fingerprint(path)
            info = probe(path, self.settings.ffprobe_path)
        except (ProbeError, OSError) as e:
            return self._fail_or_retry(rec_id, path, attempts, str(e))

        fields = dict(
            fingerprint=fp, size_bytes=st.st_size, mtime=st.st_mtime, duration=info.duration,
            width=info.width, height=info.height, fps=info.fps, video_codec=info.video_codec,
            audio_codec=info.audio_codec, container=info.container, bit_rate=info.bit_rate, error=None,
        )

        # 1) exact duplicate (same bytes already imported, e.g. a copied file)
        dup = self.db.find_imported_by_fingerprint(fp, exclude_id=rec_id)
        if dup:
            self.db.update_recording(rec_id, status="duplicate", duplicate_of=dup["id"], **fields)
            msg = f"Duplicate of #{dup['id']} ({dup['filename']}) - skipped"
            self.db.add_event(msg, recording_id=rec_id)
            return ImportResult("duplicate", msg)

        # 2) too short to contain a highlight
        if info.duration < self.settings.min_duration_seconds:
            fields["error"] = f"shorter than {self.settings.min_duration_seconds}s"
            self.db.update_recording(rec_id, status="ignored", **fields)
            self.db.add_event(f"Ignored {path.name}: only {info.duration:.1f}s long", recording_id=rec_id)
            return ImportResult("ignored", "too short")

        # 3) OBS auto-remux: same folder, same stem, other container, same duration
        superseded = self._resolve_remux_pair(rec_id, path, info.duration)
        if superseded is not None:
            self.db.update_recording(rec_id, status="duplicate", duplicate_of=superseded, **fields)
            msg = f"{path.name} is a remux of recording #{superseded} - skipped"
            self.db.add_event(msg, recording_id=rec_id)
            return ImportResult("duplicate", msg)

        # 4) optional copy into RAW GAMEPLAY
        library_path = None
        if self.settings.import_mode == "copy" and not self._inside_workspace(path):
            try:
                library_path = str(self._copy_to_library(path))
            except OSError as e:
                return self._fail_or_retry(rec_id, path, attempts, f"copy failed: {e}")

        self.db.update_recording(rec_id, status="imported", imported_at=now_iso(),
                                 library_path=library_path, **fields)
        self.db.add_event(
            f"Imported {path.name}: {info.duration:.0f}s, {info.width}x{info.height} @ {info.fps:g} fps",
            recording_id=rec_id,
        )
        log.info("imported #%s %s", rec_id, path)
        return ImportResult("imported")

    # ---- internals -----------------------------------------------------
    def _resolve_remux_pair(self, rec_id: int, path: Path, duration: float) -> int | None:
        """Return id of the recording this one duplicates, or None.

        If the *other* file is the less preferred container (mkv), it is marked duplicate
        instead and this one is imported.
        """
        for other in self.db.imported_in_dir_with_stem(str(path.parent), path.stem, rec_id):
            other_path = Path(other["path"])
            if other_path.suffix.lower() == path.suffix.lower():
                continue
            if abs((other["duration"] or 0) - duration) > 1.5:
                continue
            mine = _CONTAINER_PREFERENCE.get(path.suffix.lower(), 9)
            theirs = _CONTAINER_PREFERENCE.get(other_path.suffix.lower(), 9)
            if mine < theirs:
                self.db.update_recording(other["id"], status="duplicate", duplicate_of=rec_id)
                self.db.add_event(f"{other_path.name} superseded by remuxed {path.name}",
                                  recording_id=other["id"])
                return None
            return other["id"]
        return None

    def _inside_workspace(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.settings.workspace.resolve())
            return True
        except ValueError:
            return False

    def _copy_to_library(self, path: Path) -> Path:
        dest_dir = self.settings.dir(RAW_DIR)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = unique_path(dest_dir / path.name)
        tmp = dest.with_name(dest.name + ".partial")
        shutil.copy2(path, tmp)
        if tmp.stat().st_size != path.stat().st_size:
            tmp.unlink(missing_ok=True)
            raise OSError("size mismatch after copy")
        tmp.rename(dest)
        return dest

    def _fail_or_retry(self, rec_id: int, path: Path, attempts: int, error: str) -> ImportResult:
        if attempts < self.settings.max_import_attempts:
            self.db.update_recording(rec_id, status="retry", error=error)
            self.db.add_event(f"{path.name}: not readable yet ({error}); will retry "
                              f"({attempts}/{self.settings.max_import_attempts})",
                              level="warning", recording_id=rec_id)
            return ImportResult("retry", error)
        self.db.update_recording(rec_id, status="failed", error=error)
        self.db.add_event(f"{path.name}: import failed after {attempts} attempts: {error}",
                          level="error", recording_id=rec_id)
        self._write_failure_report(path, error, attempts)
        return ImportResult("failed", error)

    def _write_failure_report(self, path: Path, error: str, attempts: int) -> None:
        try:
            d = self.settings.dir(FAILED_DIR)
            d.mkdir(parents=True, exist_ok=True)
            report = unique_path(d / f"{path.stem} - import error.txt")
            report.write_text(
                f"Recording: {path}\nAttempts: {attempts}\nError: {error}\nTime: {now_iso()}\n\n"
                "The original recording was NOT modified. Use 'Retry' in the dashboard or\n"
                "`python -m rogue_sniper retry <id>` after fixing the problem.\n",
                encoding="utf-8",
            )
        except OSError:
            log.exception("could not write failure report")
