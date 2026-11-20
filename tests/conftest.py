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


# ---- Phase 2 synthetic "gameplay" -------------------------------------------
# Known ground truth:
#   8.0s   single "elimination" pop-up (1.5s)
#   20-23s sniper scope view, one loud shot at 21.5s
#   30.0s, 34.0s two "elimination" pop-ups  -> multi elimination
#   40-44s gunfire burst, 5 shots/s          -> high-action combat
GAME_TRUTH = {"elims": [8.0, 30.0, 34.0], "scope": (20.0, 23.0), "sniper_shot": 21.5, "burst": (40.0, 44.0)}
POPUP_BOX = (520, 560, 760, 620)   # x1, y1, x2, y2 at 1280x720


def _assets(d: Path):
    import cv2
    import numpy as np
    d.mkdir(parents=True, exist_ok=True)
    w, h = 1280, 720
    yy, xx = np.mgrid[0:h, 0:w]
    scope = np.zeros((h, w, 4), np.uint8)              # black outside a circle, transparent inside
    scope[..., 3] = np.where(np.hypot(xx - w / 2, yy - h / 2) > h * 0.48, 255, 0)
    cv2.imwrite(str(d / "scope.png"), scope)
    rng = np.random.default_rng(7)                    # high-texture "ELIMINATED" pop-up stand-in
    pw, ph = POPUP_BOX[2] - POPUP_BOX[0], POPUP_BOX[3] - POPUP_BOX[1]
    block = rng.integers(0, 2, (ph // 10, pw // 10)).repeat(10, 0).repeat(10, 1)
    popup = np.dstack([np.where(block, 255, 30)] * 3 + [np.full(block.shape, 255)]).astype(np.uint8)
    popup[..., 2] = np.where(block, 255, 0)
    cv2.imwrite(str(d / "popup.png"), popup)
    return d / "scope.png", d / "popup.png"


def make_game_clip(path: Path, seconds: float = 60, audio: bool = True) -> Path:
    scope, popup = _assets(path.parent / "_assets")
    e = GAME_TRUTH["elims"]
    pop_en = "+".join(f"between(t,{t},{t + 1.5})" for t in e)
    x1, y1 = POPUP_BOX[:2]
    shot = GAME_TRUTH["sniper_shot"]
    b0, b1 = GAME_TRUTH["burst"]
    aexpr = (f"0.02*(random(0)*2-1)"
             f"+0.8*(random(1)*2-1)*exp(-40*mod(t,0.2))*between(t,{b0},{b1})"
             f"+0.9*(random(2)*2-1)*exp(-30*(t-{shot}))*between(t,{shot},{shot + 0.7})")
    args = ["ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate=30:duration={seconds}",
            "-loop", "1", "-i", str(scope), "-loop", "1", "-i", str(popup)]
    if audio:
        args += ["-f", "lavfi", "-i", f"aevalsrc='{aexpr}':s=16000:d={seconds}"]
    s0, s1 = GAME_TRUTH["scope"]
    args += ["-filter_complex",
             f"[0:v][1:v]overlay=0:0:enable='between(t,{s0},{s1})'[a];"
             f"[a][2:v]overlay={x1}:{y1}:enable='{pop_en}':shortest=1[v]",
             "-map", "[v]", "-t", str(seconds), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if audio:
        args += ["-map", "3:a", "-c:a", "aac"]
    args += [str(path)]
    subprocess.run(args, check=True)
    return path
