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
