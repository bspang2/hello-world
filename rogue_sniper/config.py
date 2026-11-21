"""Settings, app-data locations and the ROGUE SNIPER workspace layout."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

APP_NAME = "RogueSniperAI"

RAW_DIR = "RAW GAMEPLAY"
PROCESSING_DIR = "PROCESSING"
READY_DIR = "READY TO POST"
ARCHIVE_DIR = "ARCHIVE"
FAILED_DIR = "FAILED"
WORKSPACE_DIRS = (RAW_DIR, PROCESSING_DIR, READY_DIR, ARCHIVE_DIR, FAILED_DIR)


def app_home() -> Path:
    """Directory for settings, the SQLite DB and logs (not for videos).

    Windows: %LOCALAPPDATA%\\RogueSniperAI. Override with ROGUE_SNIPER_HOME.
    """
    override = os.environ.get("ROGUE_SNIPER_HOME")
    if override:
        return Path(override)
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / APP_NAME
    return Path.home() / ".rogue_sniper"


def default_workspace() -> Path:
    return Path.home() / "Videos" / "ROGUE SNIPER"


class Settings(BaseModel):
    workspace_root: str = Field(default_factory=lambda: str(default_workspace()))
    # Extra folders to monitor (ShadowPlay / OBS output). RAW GAMEPLAY is always monitored.
    watch_folders: list[str] = Field(default_factory=list)
    recursive: bool = True
    extensions: list[str] = Field(default_factory=lambda: [".mp4", ".mkv"])
    # A file must keep the same size + mtime for this long (and not be locked) before import.
    stable_seconds: float = Field(default=15.0, ge=1.0, le=3600)
    poll_interval_seconds: float = Field(default=5.0, ge=0.5, le=600)
    min_duration_seconds: float = Field(default=10.0, ge=0)
    max_import_attempts: int = Field(default=5, ge=1, le=50)
    # reference: analyse the original in place (never modified).
    # copy: also copy it into RAW GAMEPLAY (original still untouched).
    import_mode: Literal["reference", "copy"] = "reference"
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"

    # ---- Phase 2: highlight detection ---------------------------------
    auto_analyze: bool = True
    analysis_fps: float = Field(default=4.0, ge=1.0, le=15.0)       # frames sampled per second
    analysis_width: int = Field(default=640, ge=320, le=1920)       # analysis frame width (px)
    analysis_hwaccel: Literal["auto", "none", "cuda", "d3d11va", "dxva2", "qsv"] = "auto"
    sensitivity: Literal["low", "medium", "high"] = "medium"
    # Optional per-type confidence thresholds, e.g. {"high_action_combat": 0.6}
    type_thresholds: dict[str, float] = Field(default_factory=dict)
    max_highlights_per_recording: int = Field(default=15, ge=1, le=100)
    max_clip_seconds: float = Field(default=45.0, ge=5, le=180)
    scope_threshold: float = Field(default=0.5, ge=0.05, le=1.0)
    combat_min_shots: dict[str, int] = Field(default_factory=lambda: {"low": 8, "medium": 6, "high": 4})

    audio_sensitivity_k: float = Field(default=6.0, ge=1.5, le=12.0)  # lower = hears more transients
    # Don't analyse while you play (keeps FPS up); resumes when the game closes.
    pause_while_gaming: bool = True
    game_processes: list[str] = Field(default_factory=lambda: ["FortniteClient-Win64-Shipping.exe"])

    # ---- Phase 2: optional cloud AI review (Claude) --------------------
    ai_enabled: bool = False
    ai_model: str = "claude-opus-5-5"
    ai_effort: Literal["low", "medium", "high"] = "low"
    ai_max_candidates: int = Field(default=8, ge=1, le=40)          # cost guard per recording
    ai_frames_per_candidate: int = Field(default=6, ge=2, le=12)

    @field_validator("extensions")
    @classmethod
    def _norm_ext(cls, v: list[str]) -> list[str]:
        out = []
        for e in v:
            e = e.strip().lower()
            if not e:
                continue
            out.append(e if e.startswith(".") else "." + e)
        return out or [".mp4", ".mkv"]

    # ---- derived paths -------------------------------------------------
    @property
    def workspace(self) -> Path:
        return Path(self.workspace_root)

    def dir(self, name: str) -> Path:
        return self.workspace / name

    def monitored_folders(self) -> list[Path]:
        folders = [self.dir(RAW_DIR)] + [Path(p) for p in self.watch_folders if p.strip()]
        seen, out = set(), []
        for f in folders:
            key = os.path.normcase(os.path.abspath(f))
            if key not in seen:
                seen.add(key)
                out.append(f)
        return out


def settings_path() -> Path:
    return app_home() / "settings.json"


def db_path() -> Path:
    return app_home() / "rogue_sniper.db"


def load_settings() -> Settings:
    p = settings_path()
    if p.exists():
        return Settings.model_validate(json.loads(p.read_text(encoding="utf-8")))
    s = Settings()
    save_settings(s)
    return s


def save_settings(s: Settings) -> None:
    p = settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(s.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp, p)  # atomic on Windows and POSIX


def ensure_workspace(s: Settings) -> dict[str, Path]:
    """Create ROGUE SNIPER/ and its five sub-folders. Safe to call repeatedly."""
    out = {}
    for name in WORKSPACE_DIRS:
        d = s.dir(name)
        d.mkdir(parents=True, exist_ok=True)
        out[name] = d
    return out


def unique_path(path: Path) -> Path:
    """Return `path`, or `name (2).ext`, `name (3).ext`... so nothing is ever overwritten."""
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    n = 2
    while True:
        cand = path.with_name(f"{stem} ({n}){suffix}")
        if not cand.exists():
            return cand
        n += 1
