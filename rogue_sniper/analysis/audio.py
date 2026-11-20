"""Audio signals: loudness and sharp transients (gunshots, hits, impacts).

Generic signal processing - it does not know Fortnite's specific sound effects.
Transient density is used as a combat indicator; thresholds are tuned per sensitivity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SR = 16000
WIN = 512
HOP = 320  # 20 ms


@dataclass
class AudioFeatures:
    hop_s: float
    rms_db: np.ndarray                     # per hop
    transient_times: np.ndarray            # seconds
    transient_strength: np.ndarray         # robust z-score of spectral flux
    noise_floor_db: float = -60.0
    loud_db: float = -20.0
    meta: dict = field(default_factory=dict)

    def per_second(self, duration: float) -> dict[str, np.ndarray]:
        n = max(1, int(np.ceil(duration)))
        shots = np.zeros(n)
        idx = np.clip(self.transient_times.astype(int), 0, n - 1)
        np.add.at(shots, idx, 1)
        per = max(1, int(round(1 / self.hop_s)))
        db = self.rms_db[: n * per]
        if len(db) < n * per:
            db = np.pad(db, (0, n * per - len(db)), constant_values=self.noise_floor_db)
        loud = db.reshape(n, per).mean(axis=1)
        span = max(1e-6, self.loud_db - self.noise_floor_db)
        return {"shots": shots, "loudness": np.clip((loud - self.noise_floor_db) / span, 0, 1)}


def _frames(x: np.ndarray, start: int, count: int) -> np.ndarray:
    idx = start * HOP + np.arange(count)[:, None] * HOP + np.arange(WIN)[None, :]
    return x[np.minimum(idx, len(x) - 1)]


def analyze_audio(x: np.ndarray, sr: int = SR, k: float = 6.0, min_gap_s: float = 0.07) -> AudioFeatures:
    if sr != SR:
        raise ValueError("analyze_audio expects 16 kHz input")
    hop_s = HOP / sr
    n = max(0, (len(x) - WIN) // HOP + 1)
    if n < 3:
        return AudioFeatures(hop_s, np.full(max(n, 1), -90.0), np.array([]), np.array([]))

    window = np.hanning(WIN).astype(np.float32)
    lo_bin = int(1000 / (sr / WIN))  # ignore < 1 kHz (music bass, rumble)
    rms_db = np.empty(n, np.float32)
    flux = np.zeros(n, np.float32)
    prev = None
    block = 20000
    for s in range(0, n, block):
        c = min(block, n - s)
        fr = _frames(x, s, c)
        rms_db[s:s + c] = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-6)
        mag = np.log1p(10 * np.abs(np.fft.rfft(fr * window, axis=1))[:, lo_bin:]).astype(np.float32)
        if prev is not None:
            mag_prev = np.vstack([prev[None, :], mag[:-1]])
        else:
            mag_prev = np.vstack([mag[:1], mag[:-1]])
        flux[s:s + c] = np.maximum(0, mag - mag_prev).sum(axis=1)
        prev = mag[-1]

    # Robust adaptive threshold: local median (1.5 s) + k * global MAD.
    med_win = 75
    pad = med_win // 2
    padded = np.pad(flux, pad, mode="edge")
    local_med = np.median(np.lib.stride_tricks.sliding_window_view(padded, med_win), axis=1)[:n]
    mad = float(np.median(np.abs(flux - np.median(flux)))) * 1.4826 + 1e-6
    z = (flux - local_med) / mad

    noise_floor = float(np.percentile(rms_db, 20))
    loud = float(np.percentile(rms_db, 98))
    gate = noise_floor + 6.0
    min_gap = max(1, int(round(min_gap_s / hop_s)))

    times, strengths = [], []
    last = -10 ** 9
    for i in np.flatnonzero(z > k):
        lo, hi = max(0, i - 3), min(n, i + 4)
        if flux[i] < flux[lo:hi].max() or i - last < min_gap:
            continue
        if rms_db[min(n - 1, i + 1)] < gate and rms_db[i] < gate:
            continue
        times.append(i * hop_s + WIN / sr / 2)
        strengths.append(float(z[i]))
        last = i
    return AudioFeatures(hop_s, rms_db, np.array(times), np.array(strengths), noise_floor, loud,
                         {"mad": mad, "k": k})
