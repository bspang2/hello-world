"""FFprobe wrapper, quick content fingerprints, Windows file-lock check, source detection."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

# Hide console windows spawned by ffmpeg/ffprobe on Windows.
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


class ProbeError(RuntimeError):
    pass


@dataclass
class MediaInfo:
    duration: float
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: str | None
    container: str
    bit_rate: int | None
    size_bytes: int

    def as_dict(self) -> dict:
        return asdict(self)


def _parse_rate(rate: str | None) -> float:
    if not rate or rate in ("0/0", "0"):
        return 0.0
    if "/" in rate:
        num, den = rate.split("/", 1)
        try:
            return float(num) / float(den) if float(den) else 0.0
        except ValueError:
            return 0.0
    try:
        return float(rate)
    except ValueError:
        return 0.0


def run_tool(args: list[str], timeout: float = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, capture_output=True, text=True, timeout=timeout,
        creationflags=CREATE_NO_WINDOW, encoding="utf-8", errors="replace",
    )


def probe(path: Path | str, ffprobe: str = "ffprobe", timeout: float = 60) -> MediaInfo:
    """Read container/stream info. Raises ProbeError for unreadable / incomplete files.

    An MP4 that is still being written (ShadowPlay, OBS non-fragmented MP4) has no `moov`
    atom yet and fails here, which is exactly what we want.
    """
    path = Path(path)
    try:
        cp = run_tool(
            [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
            timeout=timeout,
        )
    except FileNotFoundError as e:
        raise ProbeError(f"ffprobe not found ({ffprobe}). Install FFmpeg and add it to PATH.") from e
    except subprocess.TimeoutExpired as e:
        raise ProbeError("ffprobe timed out") from e
    if cp.returncode != 0:
        raise ProbeError((cp.stderr or "ffprobe failed").strip().splitlines()[-1][:500])
    try:
        data = json.loads(cp.stdout or "{}")
    except json.JSONDecodeError as e:
        raise ProbeError("ffprobe returned invalid JSON") from e

    streams = data.get("streams", [])
    fmt = data.get("format", {})
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not s.get("disposition", {}).get("attached_pic")), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise ProbeError("no video stream")

    duration = float(fmt.get("duration") or video.get("duration") or 0)
    if duration <= 0:
        raise ProbeError("unknown or zero duration (file may still be recording)")

    fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate"))
    br = fmt.get("bit_rate")
    return MediaInfo(
        duration=round(duration, 3),
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=round(fps, 3),
        video_codec=video.get("codec_name", "unknown"),
        audio_codec=audio.get("codec_name") if audio else None,
        container=fmt.get("format_name", "unknown"),
        bit_rate=int(br) if br and str(br).isdigit() else None,
        size_bytes=path.stat().st_size,
    )


def quick_fingerprint(path: Path | str, chunk: int = 4 * 1024 * 1024) -> str:
    """SHA-256 over size + head + middle + tail chunks.

    Reads at most ~12 MiB, so a 20 GB recording is fingerprinted in milliseconds,
    yet identical copies (same file re-saved / copied elsewhere) still match.
    """
    path = Path(path)
    size = path.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with open(path, "rb") as f:
        if size <= chunk * 3:
            h.update(f.read())
        else:
            for off in (0, size // 2 - chunk // 2, size - chunk):
                f.seek(off)
                h.update(f.read(chunk))
    return h.hexdigest()


def is_file_locked(path: Path | str) -> bool:
    """True if another process (OBS, ShadowPlay) still holds the file open.

    Windows only: tries an exclusive CreateFileW (share mode 0). Elsewhere returns False and
    we rely on the size/mtime stability window plus ffprobe validation.
    """
    if os.name != "nt":
        return False
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    GENERIC_READ, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL = 0x80000000, 3, 0x80
    handle = k32.CreateFileW(str(path), GENERIC_READ, 0, None, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, None)
    if handle == wintypes.HANDLE(-1).value or handle is None:
        err = ctypes.get_last_error()
        return err in (32, 33)  # ERROR_SHARING_VIOLATION, ERROR_LOCK_VIOLATION
    k32.CloseHandle(handle)
    return False


# "Fortnite 2026.11.19 - 20.15.33.02.mp4" / "...02.DVR.mp4" (ShadowPlay / NVIDIA App)
_SHADOWPLAY_RE = re.compile(r"^.+ \d{4}\.\d{2}\.\d{2} - \d{2}\.\d{2}\.\d{2}\.\d{2}(\.DVR)?$", re.I)
# "2026-11-19 20-15-33.mkv" (OBS default %CCYY-%MM-%DD %hh-%mm-%ss), "Replay 2026-11-19 20-15-33"
_OBS_RE = re.compile(r"^(Replay )?\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}$", re.I)


def detect_source(path: Path | str) -> str:
    stem = Path(path).stem
    if _SHADOWPLAY_RE.match(stem):
        return "shadowplay"
    if _OBS_RE.match(stem):
        return "obs"
    return "unknown"
