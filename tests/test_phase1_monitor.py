"""Phase 1: folder monitoring + import, using real FFmpeg-generated recordings."""
import hashlib
import os
import shutil
from pathlib import Path

from conftest import make_clip

from rogue_sniper.config import FAILED_DIR, RAW_DIR, READY_DIR, unique_path
from rogue_sniper.db import Database
from rogue_sniper.importer import Importer
from rogue_sniper.media import ProbeError, detect_source, probe, quick_fingerprint
from rogue_sniper.watcher import FolderWatcher

import pytest


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def settle(watcher, clock, seconds=11):
    """Scan, let the stability window pass, scan again."""
    watcher.scan()
    clock.advance(seconds)
    return watcher.scan()


# ---- media ---------------------------------------------------------------
def test_probe_reads_mp4_and_mkv(tmp_path):
    mp4 = make_clip(tmp_path / "a.mp4", seconds=2, fps=60)
    mkv = make_clip(tmp_path / "b.mkv", seconds=2, fps=30)
    a, b = probe(mp4), probe(mkv)
    assert (a.width, a.height) == (320, 180) and a.fps == 60 and 1.9 < a.duration < 2.2
    assert a.video_codec == "h264" and a.audio_codec == "aac"
    assert b.fps == 30 and "matroska" in b.container


def test_probe_rejects_truncated_mp4(tmp_path):
    """A ShadowPlay MP4 mid-recording has no moov atom; it must not be imported."""
    src = make_clip(tmp_path / "full.mp4", seconds=3)
    data = src.read_bytes()
    part = tmp_path / "recording.mp4"
    part.write_bytes(data[: len(data) // 2])
    with pytest.raises(ProbeError):
        probe(part)


def test_fingerprint_matches_copies_only(tmp_path):
    a = make_clip(tmp_path / "a.mp4", seed=1)
    b = make_clip(tmp_path / "b.mp4", seed=2)
    c = tmp_path / "copy.mp4"
    shutil.copy(a, c)
    assert quick_fingerprint(a) == quick_fingerprint(c)
    assert quick_fingerprint(a) != quick_fingerprint(b)


def test_detect_source():
    assert detect_source("Fortnite 2026.11.19 - 20.15.33.02.mp4") == "shadowplay"
    assert detect_source("Fortnite 2026.11.19 - 20.15.33.02.DVR.mp4") == "shadowplay"
    assert detect_source("2026-11-19 20-15-33.mkv") == "obs"
    assert detect_source("Replay 2026-11-19 20-15-33.mkv") == "obs"
    assert detect_source("my clip.mp4") == "unknown"


def test_unique_path_never_overwrites(tmp_path):
    p = tmp_path / "final_video.mp4"
    assert unique_path(p) == p
    p.write_text("x")
    assert unique_path(p).name == "final_video (2).mp4"
    (tmp_path / "final_video (2).mp4").write_text("x")
    assert unique_path(p).name == "final_video (3).mp4"


# ---- watcher / importer --------------------------------------------------
def test_new_recording_waits_for_stability_then_imports(settings, db, watcher, clock):
    clip = make_clip(settings.dir(RAW_DIR) / "Fortnite 2026.11.19 - 20.15.33.02.mp4")
    rep = watcher.scan()
    assert rep.new == 1 and not rep.imported
    rec = db.list_recordings()[0]
    assert rec["status"] == "waiting" and rec["source"] == "shadowplay"

    clock.advance(5)                       # not yet stable long enough
    assert not watcher.scan().imported
    clock.advance(6)
    rep = watcher.scan()
    assert len(rep.imported) == 1
    rec = db.get_recording(rep.imported[0])
    assert rec["status"] == "imported" and rec["fps"] == 60 and rec["fingerprint"]
    assert rec["path"] == str(clip.resolve())


def test_growing_file_is_not_imported_until_writes_stop(settings, db, watcher, clock):
    full = make_clip(settings.workspace.parent / "src.mkv", seconds=3).read_bytes()
    target = settings.dir(RAW_DIR) / "2026-11-19 20-15-33.mkv"
    target.write_bytes(full[:2000])
    watcher.scan()
    for i in range(2, 6):                   # keeps growing every 8s (< 10s window)
        clock.advance(8)
        target.write_bytes(full[: min(len(full), 2000 * i * 20)])
        os.utime(target, (1000 + i, 1000 + i))
        assert not watcher.scan().imported
    target.write_bytes(full)
    os.utime(target, (2000, 2000))
    clock.advance(8)
    assert not watcher.scan().imported      # just changed
    clock.advance(11)
    assert len(watcher.scan().imported) == 1


def test_locked_file_is_skipped(settings, db, clock):
    locked = {"v": True}
    w = FolderWatcher(settings, db, Importer(settings, db), clock=clock, lock_check=lambda p: locked["v"])
    make_clip(settings.dir(RAW_DIR) / "a.mp4")
    assert not settle(w, clock).imported
    locked["v"] = False
    clock.advance(11)
    assert len(w.scan().imported) == 1


def test_originals_are_never_modified(settings, db, watcher, clock, tmp_path):
    shadow = tmp_path / "Videos" / "Fortnite"
    clip = make_clip(shadow / "Fortnite 2026.11.19 - 21.00.00.01.mp4")
    before, mtime = sha(clip), clip.stat().st_mtime
    settings.watch_folders = [str(shadow)]
    settings.import_mode = "copy"
    assert len(settle(watcher, clock).imported) == 1
    assert sha(clip) == before and clip.stat().st_mtime == mtime
    rec = db.list_recordings()[0]
    lib = Path(rec["library_path"])
    assert lib.parent == settings.dir(RAW_DIR) and sha(lib) == before
    # our own copy in RAW GAMEPLAY must not be re-imported as a new recording
    clock.advance(11)
    watcher.scan()
    assert len(db.list_recordings()) == 1


def test_duplicate_content_is_skipped(settings, db, watcher, clock, tmp_path):
    a = make_clip(settings.dir(RAW_DIR) / "a.mp4", seed=3)
    shutil.copy(a, settings.dir(RAW_DIR) / "a copy.mp4")
    rep = settle(watcher, clock)
    assert len(rep.imported) == 1 and rep.other == {"duplicate": 1}
    dup = db.list_recordings(status="duplicate")[0]
    assert dup["duplicate_of"] == rep.imported[0]


def test_already_imported_files_are_not_reprocessed(settings, db, watcher, clock):
    make_clip(settings.dir(RAW_DIR) / "a.mp4")
    settle(watcher, clock)
    for _ in range(3):
        clock.advance(30)
        rep = watcher.scan()
        assert rep.new == 0 and not rep.imported
    assert db.get_recording(1)["attempts"] == 1


def test_state_survives_restart(settings, tmp_path, clock):
    dbp = tmp_path / "persist.db"
    make_clip(settings.dir(RAW_DIR) / "a.mp4")
    db1 = Database(dbp)
    w1 = FolderWatcher(settings, db1, Importer(settings, db1), clock=clock, lock_check=lambda p: False)
    assert len(settle(w1, clock).imported) == 1
    db1.close()
    db2 = Database(dbp)
    w2 = FolderWatcher(settings, db2, Importer(settings, db2), clock=clock, lock_check=lambda p: False)
    rep = settle(w2, clock)
    assert rep.new == 0 and not rep.imported
    db2.close()


def test_obs_remux_pair_keeps_mp4_only(settings, db, watcher, clock):
    raw = settings.dir(RAW_DIR)
    mkv = make_clip(raw / "2026-11-19 20-15-33.mkv", seconds=3, seed=5)
    assert len(settle(watcher, clock).imported) == 1
    # OBS remuxes to MP4 afterwards (different bytes, same content/duration)
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mkv), "-c", "copy",
                    str(raw / "2026-11-19 20-15-33.mp4")], check=True)
    rep = settle(watcher, clock)
    assert len(rep.imported) == 1
    imported = db.list_recordings(status="imported")
    assert [Path(r["path"]).suffix for r in imported] == [".mp4"]
    assert Path(db.list_recordings(status="duplicate")[0]["path"]).suffix == ".mkv"


def test_short_clip_is_ignored(settings, db, watcher, clock):
    settings.min_duration_seconds = 5
    make_clip(settings.dir(RAW_DIR) / "short.mp4", seconds=2)
    rep = settle(watcher, clock)
    assert rep.other == {"ignored": 1}


def test_corrupt_file_retries_then_fails_with_report(settings, db, watcher, clock):
    bad = settings.dir(RAW_DIR) / "broken.mp4"
    bad.write_bytes(b"\x00" * 50_000)
    original = sha(bad)
    watcher.scan()
    statuses = []
    for _ in range(12):
        clock.advance(31)
        watcher.scan()
        statuses.append(db.get_recording(1)["status"])
    assert "retry" in statuses and statuses[-1] == "failed"
    assert db.get_recording(1)["attempts"] == settings.max_import_attempts
    reports = list(settings.dir(FAILED_DIR).glob("broken - import error.txt"))
    assert len(reports) == 1 and "NOT modified" in reports[0].read_text()
    assert sha(bad) == original  # original kept in place, untouched


def test_retry_after_failure(settings, db, watcher, clock):
    bad = settings.dir(RAW_DIR) / "fixme.mp4"
    bad.write_bytes(b"\x00" * 1000)
    watcher.scan()
    for _ in range(12):
        clock.advance(31)
        watcher.scan()
    assert db.get_recording(1)["status"] == "failed"
    # user fixes the file (e.g. re-exports it) -> changed on disk -> re-imported automatically
    good = make_clip(bad.parent / "tmp_good.mp4")
    os.replace(good, bad)
    rep = settle(watcher, clock)
    assert rep.imported == [1]


def test_output_folders_are_never_scanned(settings, db, watcher, clock, tmp_path):
    """Watching a parent folder (e.g. Videos) must not re-import our own exports."""
    settings.watch_folders = [str(settings.workspace.parent)]
    make_clip(settings.dir(READY_DIR) / "clip1" / "final_video.mp4")
    make_clip(settings.dir(FAILED_DIR) / "x.mkv", seed=9)
    make_clip(settings.workspace.parent / "Fortnite 2026.11.19 - 22.00.00.01.mp4", seed=4)
    rep = settle(watcher, clock)
    names = [r["filename"] for r in db.list_recordings()]
    assert names == ["Fortnite 2026.11.19 - 22.00.00.01.mp4"] and len(rep.imported) == 1


def test_partial_and_hidden_names_ignored(settings, db, watcher, clock):
    raw = settings.dir(RAW_DIR)
    (raw / "x.mp4.partial").write_bytes(b"1")
    (raw / ".hidden.mp4").write_bytes(b"1")
    (raw / "notes.txt").write_text("x")
    assert watcher.scan().seen == 0


def test_deleted_pending_file_marked_missing(settings, db, watcher, clock):
    p = make_clip(settings.dir(RAW_DIR) / "gone.mp4")
    watcher.scan()
    p.unlink()
    clock.advance(11)
    watcher.scan()
    assert db.get_recording(1)["status"] == "missing"
    assert watcher.pending_count == 0
