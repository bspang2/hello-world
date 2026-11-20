"""Stream audio samples and downscaled frames out of FFmpeg without temp files."""
from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from ..media import CREATE_NO_WINDOW


class Cancelled(Exception):
    """Raised when a job's cancel event is set while decoding."""


class DecodeError(RuntimeError):
    pass


def _check(cancel) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


class _Proc:
    """FFmpeg subprocess with stderr drained on a thread (prevents pipe deadlock)."""

    def __init__(self, args: list[str]):
        self.p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  stdin=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
        self._err = bytearray()
        self._t = threading.Thread(target=self._drain, daemon=True)
        self._t.start()

    def _drain(self) -> None:
        for chunk in iter(lambda: self.p.stderr.read(4096), b""):
            self._err += chunk
            del self._err[:-8192]

    def read_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.p.stdout.read(n - len(buf))
            if not chunk:
                break
            buf += chunk
        return bytes(buf)

    def kill(self) -> None:
        if self.p.poll() is None:
            self.p.kill()
        self.p.wait()

    def finish(self) -> tuple[int, str]:
        rc = self.p.wait()
        self._t.join(2)
        return rc, self._err.decode("utf-8", "replace").strip()


def read_audio(path: Path | str, ffmpeg: str = "ffmpeg", sr: int = 16000, cancel=None) -> np.ndarray:
    """Mono float32 samples in [-1, 1] at `sr` Hz."""
    proc = _Proc([ffmpeg, "-v", "error", "-nostdin", "-i", str(path), "-vn", "-ac", "1",
                  "-ar", str(sr), "-f", "s16le", "-"])
    chunks = []
    try:
        while True:
            _check(cancel)
            b = proc.p.stdout.read(1 << 20)
            if not b:
                break
            chunks.append(b)
    except Cancelled:
        proc.kill()
        raise
    rc, err = proc.finish()
    if rc != 0:
        raise DecodeError(f"audio decode failed: {err[-300:]}")
    data = b"".join(chunks)
    data = data[: len(data) // 2 * 2]
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def analysis_size(src_w: int, src_h: int, width: int) -> tuple[int, int]:
    width = min(width, src_w) if src_w else width
    width -= width % 2
    h = int(round(width * (src_h / src_w) / 2) * 2) if src_w and src_h else int(width * 9 / 16)
    return width, max(h, 2)


def iter_frames(path: Path | str, fps: float, size: tuple[int, int], ffmpeg: str = "ffmpeg",
                hwaccel: str = "none", cancel=None) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (time_seconds, BGR uint8 frame) sampled at `fps`.

    With a hardware decoder that fails before producing any frame, retries on the CPU.
    """
    try:
        yield from _iter_frames(path, fps, size, ffmpeg, hwaccel, cancel)
    except DecodeError as e:
        if hwaccel == "none" or getattr(e, "frames", 0) > 0:
            raise
        yield from _iter_frames(path, fps, size, ffmpeg, "none", cancel)


def _iter_frames(path, fps, size, ffmpeg, hwaccel, cancel):
    w, h = size
    args = [ffmpeg, "-v", "error", "-nostdin"]
    if hwaccel and hwaccel != "none":
        args += ["-hwaccel", hwaccel]
    args += ["-i", str(path), "-an", "-sn", "-dn", "-vf", f"fps={fps}:start_time=0:round=up,scale={w}:{h}:flags=area",
             "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    # start_time=0 + round=up: frame i is exactly the frame `extract_frame(i/fps)` returns, even
    # when the video track starts after the audio (common in NVIDIA/OBS files). Templates depend on it.
    proc = _Proc(args)
    n = w * h * 3
    i = 0
    try:
        while True:
            _check(cancel)
            buf = proc.read_exact(n)
            if len(buf) < n:
                break
            yield i / fps, np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
            i += 1
    except BaseException:
        proc.kill()
        raise
    rc, err = proc.finish()
    if rc != 0 or i == 0:
        e = DecodeError(f"video decode failed ({hwaccel}): {err[-300:] or 'no frames'}")
        e.frames = i
        raise e


def extract_frame(path: Path | str, t: float, width: int | None = None, ffmpeg: str = "ffmpeg",
                  size: tuple[int, int] | None = None) -> np.ndarray:
    """One BGR frame at time `t` (full resolution unless `width` or exact `size` is given).

    Pass `size` to get pixels identical to what `iter_frames` produces (same scaler).
    """
    if size:
        vf = ["-vf", f"scale={size[0]}:{size[1]}:flags=area"]
    else:
        vf = ["-vf", f"scale={width}:-2"] if width else []
    cp = subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-ss", f"{max(0.0, t):.3f}", "-i", str(path),
                         "-frames:v", "1", *vf, "-f", "image2pipe", "-vcodec", "png", "-"],
                        capture_output=True, creationflags=CREATE_NO_WINDOW, timeout=60)
    if cp.returncode != 0 or not cp.stdout:
        raise DecodeError(f"could not read frame at {t:.2f}s: {cp.stderr.decode('utf-8', 'replace')[-300:]}")
    img = cv2.imdecode(np.frombuffer(cp.stdout, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise DecodeError("frame decode failed")
    return img


def extract_jpeg(path: Path | str, t: float, width: int = 768, ffmpeg: str = "ffmpeg", quality: int = 85) -> bytes:
    img = extract_frame(path, t, width=width, ffmpeg=ffmpeg)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise DecodeError("jpeg encode failed")
    return buf.tobytes()
