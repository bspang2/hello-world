"""Evaluate the PC: FFmpeg, working GPU encoders, GPU, Python packages, ShadowPlay/OBS setup."""
from __future__ import annotations

import configparser
import importlib
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .config import Settings
from .media import run_tool

# Encoders in order of preference for the Phase 3 renderer.
ENCODERS = [
    ("h264_nvenc", "NVIDIA NVENC H.264"),
    ("hevc_nvenc", "NVIDIA NVENC HEVC"),
    ("h264_amf", "AMD AMF H.264"),
    ("h264_qsv", "Intel Quick Sync H.264"),
    ("libx264", "CPU x264 (always-works fallback)"),
]
PY_PACKAGES = ["fastapi", "uvicorn", "pydantic", "numpy", "cv2"]
VIDEO_EXTS = {".mp4", ".mkv"}


def _which(tool: str) -> str | None:
    if os.path.isabs(tool) or os.sep in tool:
        return tool if Path(tool).exists() else None
    return shutil.which(tool)


def check_ffmpeg(ffmpeg: str, ffprobe: str) -> dict:
    out: dict = {"ffmpeg_path": _which(ffmpeg), "ffprobe_path": _which(ffprobe), "version": None,
                 "hwaccels": [], "encoders": {}}
    if not out["ffmpeg_path"]:
        return out
    try:
        v = run_tool([ffmpeg, "-hide_banner", "-version"], timeout=20)
        out["version"] = (v.stdout.splitlines() or [""])[0]
        hw = run_tool([ffmpeg, "-hide_banner", "-hwaccels"], timeout=20)
        out["hwaccels"] = [l.strip() for l in hw.stdout.splitlines()[1:] if l.strip()]
        listed = run_tool([ffmpeg, "-hide_banner", "-encoders"], timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        out["error"] = str(e)
        return out
    for name, label in ENCODERS:
        is_listed = re.search(rf"\s{re.escape(name)}\s", listed) is not None
        works, err = (test_encoder(ffmpeg, name) if is_listed else (False, "not compiled into this FFmpeg"))
        out["encoders"][name] = {"label": label, "listed": is_listed, "works": works, "error": err}
    return out


def test_encoder(ffmpeg: str, encoder: str) -> tuple[bool, str | None]:
    """Really encode 1 s of 1080p60 test video. 'Listed' != 'usable' (no GPU / old driver)."""
    args = [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
            "-i", "testsrc2=size=1920x1080:rate=60", "-t", "1", "-pix_fmt", "yuv420p",
            "-c:v", encoder, "-f", "null", "-"]
    try:
        cp = run_tool(args, timeout=60)
    except subprocess.TimeoutExpired:
        return False, "timed out"
    if cp.returncode == 0:
        return True, None
    lines = [l for l in (cp.stderr or "").splitlines() if l.strip()]
    # The first line from the encoder itself is the real cause (e.g. "Cannot load nvcuda.dll").
    own = [l for l in lines if f"{encoder} @" in l and "vost#" not in l]
    return False, ((own or lines or ["failed"])[0])[:300]


def check_gpu() -> list[dict]:
    gpus: list[dict] = []
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            cp = run_tool([smi, "--query-gpu=name,driver_version,memory.total",
                           "--format=csv,noheader"], timeout=15)
            for line in cp.stdout.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    gpus.append({"vendor": "NVIDIA", "name": parts[0], "driver": parts[1], "memory": parts[2]})
        except (OSError, subprocess.TimeoutExpired):
            pass
    if not gpus and sys.platform == "win32":
        try:
            cp = run_tool(["powershell", "-NoProfile", "-Command",
                           "Get-CimInstance Win32_VideoController | "
                           "ForEach-Object { $_.Name + '|' + $_.DriverVersion }"], timeout=30)
            for line in cp.stdout.splitlines():
                if "|" in line:
                    name, drv = line.split("|", 1)
                    vendor = next((v for v in ("NVIDIA", "AMD", "Intel") if v.lower() in name.lower()), "?")
                    gpus.append({"vendor": vendor, "name": name.strip(), "driver": drv.strip(), "memory": None})
        except (OSError, subprocess.TimeoutExpired):
            pass
    return gpus


def check_python_packages() -> dict:
    out = {}
    for name in PY_PACKAGES:
        try:
            mod = importlib.import_module(name)
            out[name] = getattr(mod, "__version__", "installed")
        except Exception:
            out[name] = None
    return out


def _count_videos(folder: Path, limit: int = 5000) -> tuple[int, str | None]:
    n, newest, newest_t = 0, None, 0.0
    try:
        for i, p in enumerate(folder.rglob("*")):
            if i > limit:
                break
            if p.suffix.lower() in VIDEO_EXTS and p.is_file():
                n += 1
                t = p.stat().st_mtime
                if t > newest_t:
                    newest, newest_t = p.name, t
    except OSError:
        pass
    return n, newest


def find_obs_profiles(appdata: Path | None) -> list[dict]:
    """Parse OBS profiles: %APPDATA%\\obs-studio\\basic\\profiles\\<name>\\basic.ini."""
    if not appdata:
        return []
    root = appdata / "obs-studio" / "basic" / "profiles"
    out = []
    if not root.is_dir():
        return out
    for ini in sorted(root.glob("*/basic.ini")):
        cp = configparser.ConfigParser(interpolation=None, strict=False)
        cp.optionxform = str
        try:
            cp.read(ini, encoding="utf-8-sig")
        except (configparser.Error, OSError):
            continue
        mode = cp.get("Output", "Mode", fallback="Simple")
        if mode.lower() == "advanced":
            folder = cp.get("AdvOut", "RecFilePath", fallback=None)
            fmt = cp.get("AdvOut", "RecFormat2", fallback=cp.get("AdvOut", "RecFormat", fallback=None))
        else:
            folder = cp.get("SimpleOutput", "FilePath", fallback=None)
            fmt = cp.get("SimpleOutput", "RecFormat2", fallback=cp.get("SimpleOutput", "RecFormat", fallback=None))
        out.append({"profile": ini.parent.name, "mode": mode, "folder": folder, "format": fmt,
                    "auto_remux": cp.get("Video", "AutoRemux", fallback=None)})
    return out


def check_recording_setup(home: Path | None = None, appdata: Path | None = None) -> dict:
    home = home or Path.home()
    if appdata is None and os.environ.get("APPDATA"):
        appdata = Path(os.environ["APPDATA"])
    videos = home / "Videos"
    candidates = [videos / "Fortnite", videos / "NVIDIA" / "Fortnite", videos / "NVIDIA", videos / "Captures", videos]
    folders = []
    for c in candidates:
        if c.is_dir():
            n, newest = _count_videos(c) if c != videos else (None, None)
            folders.append({"path": str(c), "videos": n, "newest": newest,
                            "likely": "shadowplay" if "fortnite" in c.name.lower() or "nvidia" in str(c).lower()
                            else "windows-game-bar" if c.name == "Captures" else "generic"})
    obs = find_obs_profiles(appdata)
    return {"shadowplay_or_capture_folders": folders, "obs_profiles": obs}


def run_env_check(settings: Settings | None = None, home: Path | None = None,
                  appdata: Path | None = None, test_encoders: bool = True) -> dict:
    s = settings or Settings()
    ff = check_ffmpeg(s.ffmpeg_path, s.ffprobe_path) if test_encoders else {
        "ffmpeg_path": _which(s.ffmpeg_path), "ffprobe_path": _which(s.ffprobe_path), "encoders": {}}
    ws = s.workspace
    disk_root = ws if ws.exists() else next((p for p in ws.parents if p.exists()), Path.cwd())
    du = shutil.disk_usage(disk_root)
    report = {
        "system": {"os": platform.platform(), "python": sys.version.split()[0],
                   "is_windows": sys.platform == "win32", "cpu_count": os.cpu_count()},
        "ffmpeg": ff,
        "gpus": check_gpu(),
        "python_packages": check_python_packages(),
        "recording": check_recording_setup(home, appdata),
        "workspace": {"path": str(ws), "free_gb": round(du.free / 1e9, 1)},
    }
    report["recommended_encoder"] = next(
        (n for n, _ in ENCODERS if ff.get("encoders", {}).get(n, {}).get("works")), None)
    report["recommendations"] = _recommendations(report)
    return report


def _recommendations(r: dict) -> list[str]:
    rec = []
    if not r["ffmpeg"].get("ffmpeg_path") or not r["ffmpeg"].get("ffprobe_path"):
        rec.append("Install FFmpeg (includes ffprobe): `winget install Gyan.FFmpeg`, then reopen the terminal.")
    if r["ffmpeg"].get("encoders") and not any(
            e["works"] for n, e in r["ffmpeg"]["encoders"].items() if n != "libx264"):
        rec.append("No working GPU encoder found - rendering will use CPU x264 (slower). "
                   "Update your GPU driver if you have an NVIDIA/AMD/Intel GPU.")
    missing = [k for k, v in r["python_packages"].items() if v is None]
    if missing:
        rec.append("Missing Python packages: " + ", ".join(missing) + " - run scripts\\windows\\install.bat")
    for p in r["recording"]["obs_profiles"]:
        fmt = (p.get("format") or "").lower()
        if fmt == "mp4":
            rec.append(f"OBS profile '{p['profile']}' records plain MP4: a crash loses the whole file. "
                       "Use MKV, fragmented MP4 or Hybrid MP4 (OBS 30.2+).")
    rec_folders = [f for f in r["recording"]["shadowplay_or_capture_folders"] if f["likely"] != "generic"]
    if not rec_folders and not r["recording"]["obs_profiles"]:
        rec.append("No ShadowPlay/OBS recording folder found. Add your recording folder in Settings.")
    if r["workspace"]["free_gb"] < 50:
        rec.append(f"Only {r['workspace']['free_gb']} GB free on the workspace drive; recordings are large.")
    return rec


def format_report(r: dict) -> str:
    L = []
    s = r["system"]
    L.append(f"System      : {s['os']} | Python {s['python']} | {s['cpu_count']} CPU threads")
    ff = r["ffmpeg"]
    L.append(f"FFmpeg      : {ff.get('version') or 'NOT FOUND'}")
    L.append(f"FFprobe     : {ff.get('ffprobe_path') or 'NOT FOUND'}")
    if ff.get("hwaccels"):
        L.append(f"HW decode   : {', '.join(ff['hwaccels'])}")
    for name, e in ff.get("encoders", {}).items():
        state = "WORKS" if e["works"] else ("listed, FAILED" if e["listed"] else "not available")
        L.append(f"  encoder {name:<11} {state}" + (f"  ({e['error']})" if e["error"] and e["listed"] else ""))
    L.append(f"Recommended : {r.get('recommended_encoder') or 'none'}")
    if r["gpus"]:
        for g in r["gpus"]:
            L.append(f"GPU         : {g['vendor']} {g['name']} (driver {g['driver']}"
                     + (f", {g['memory']}" if g.get("memory") else "") + ")")
    else:
        L.append("GPU         : none detected")
    L.append("Python pkgs : " + ", ".join(f"{k}={'missing' if v is None else v}"
                                          for k, v in r["python_packages"].items()))
    rec = r["recording"]
    for f in rec["shadowplay_or_capture_folders"]:
        extra = f"  {f['videos']} videos, newest: {f['newest']}" if f["videos"] is not None else ""
        L.append(f"Rec folder  : [{f['likely']}] {f['path']}{extra}")
    for p in rec["obs_profiles"]:
        L.append(f"OBS profile : {p['profile']} -> {p['folder']} (format: {p['format']}, mode: {p['mode']})")
    L.append(f"Workspace   : {r['workspace']['path']} ({r['workspace']['free_gb']} GB free)")
    if r["recommendations"]:
        L.append("\nRecommendations:")
        L += [f"  - {x}" for x in r["recommendations"]]
    return "\n".join(L)


def detected_recording_folders(r: dict) -> list[str]:
    out = [f["path"] for f in r["recording"]["shadowplay_or_capture_folders"] if f["likely"] == "shadowplay"]
    # If a game-specific folder (NVIDIA\Fortnite) exists, don't also monitor its parent
    # (NVIDIA\), which would import recordings of other games.
    norm = [os.path.normcase(os.path.abspath(p)) for p in out]
    out = [p for p, n in zip(out, norm)
           if not any(o != n and o.startswith(n.rstrip("\\/") + os.sep) for o in norm)]
    out += [p["folder"] for p in r["recording"]["obs_profiles"] if p.get("folder")]
    return list(dict.fromkeys(out))
