import shutil
import subprocess
from pathlib import Path

import pytest

from rogue_sniper.config import Settings, ensure_workspace
from rogue_sniper.db import Database
from rogue_sniper.importer import Importer
from rogue_sniper.watcher import FolderWatcher

if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
    pytest.skip("FFmpeg/FFprobe required for tests", allow_module_level=True)


def make_clip(path: Path, seconds: float = 3, fps: int = 60, size: str = "320x180",
              audio: bool = True, seed: int = 0) -> Path:
    """Generate a real test clip (video + optional tone) with FFmpeg."""
    path.parent.mkdir(parents=True, exist_ok=True)
    args = ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
            "-i", f"testsrc2=size={size}:rate={fps}:duration={seconds}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency={440 + seed * 10}:duration={seconds}"]
    args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if audio:
        args += ["-c:a", "aac", "-shortest"]
    # make content unique per seed so fingerprints differ
    args += ["-metadata", f"comment=seed{seed}", str(path)]
    subprocess.run(args, check=True)
    return path


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ROGUE_SNIPER_HOME", str(tmp_path / "apphome"))


@pytest.fixture
def settings(tmp_path):
    s = Settings(workspace_root=str(tmp_path / "ROGUE SNIPER"), stable_seconds=10,
                 min_duration_seconds=1, max_import_attempts=3)
    ensure_workspace(s)
    return s


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    yield d
    d.close()


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def watcher(settings, db, clock):
    return FolderWatcher(settings, db, Importer(settings, db), clock=clock, lock_check=lambda p: False)
