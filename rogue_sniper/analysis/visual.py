"""Per-frame visual signals: motion, sniper-scope view, and HUD template matches."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DARK = 28  # gray level treated as "black" scope surround


def _dark(g: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> float:
    h, w = g.shape
    r = g[int(y0 * h):max(int(y0 * h) + 1, int(y1 * h)), int(x0 * w):max(int(x0 * w) + 1, int(x1 * w))]
    return float((r < DARK).mean()) if r.size else 0.0


def scope_score(gray: np.ndarray) -> float:
    """0..1: how much the frame looks like a sniper scope (black surround, round lens).

    Checks that the screen sides are black, the top corners *beside* the lens are black
    (a circle narrows towards the top - this rejects plain black pillarbox bars), and the
    lens itself (mid-row, just inside the edge, and the center) is not black.
    """
    sides = min(_dark(gray, 0.0, 0.16, 0.35, 0.65), _dark(gray, 0.84, 1.0, 0.35, 0.65))
    corners = min(_dark(gray, 0.20, 0.30, 0.02, 0.10), _dark(gray, 0.70, 0.80, 0.02, 0.10))
    lens_edge = max(_dark(gray, 0.25, 0.32, 0.42, 0.58), _dark(gray, 0.68, 0.75, 0.42, 0.58))
    center = _dark(gray, 0.40, 0.60, 0.35, 0.65)
    return float(min(sides, corners) * (1 - lens_edge) * (1 - center))


class MotionMeter:
    def __init__(self):
        self.prev: np.ndarray | None = None

    def __call__(self, gray: np.ndarray) -> float:
        small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA).astype(np.int16)
        v = 0.0 if self.prev is None else float(np.abs(small - self.prev).mean() / 255.0)
        self.prev = small
        return v


@dataclass
class Template:
    event: str
    name: str
    image: np.ndarray                         # gray, cropped at `frame_width` scale
    region: tuple[float, float, float, float]  # x0, y0, x1, y1 relative to frame
    frame_width: int
    threshold: float = 0.75


class TemplateMatcher:
    """Find saved HUD templates (e.g. your ELIMINATED pop-up) in analysis frames."""

    MARGIN = 0.03

    def __init__(self, templates: list[Template], frame_w: int, frame_h: int):
        self.items = []
        for t in templates:
            scale = frame_w / t.frame_width
            img = t.image if abs(scale - 1) < 1e-3 else cv2.resize(
                t.image, (max(1, round(t.image.shape[1] * scale)), max(1, round(t.image.shape[0] * scale))),
                interpolation=cv2.INTER_AREA)
            x0, y0, x1, y1 = t.region
            sx0 = max(0, int((x0 - self.MARGIN) * frame_w))
            sy0 = max(0, int((y0 - self.MARGIN) * frame_h))
            sx1 = min(frame_w, int((x1 + self.MARGIN) * frame_w) + 1)
            sy1 = min(frame_h, int((y1 + self.MARGIN) * frame_h) + 1)
            if sx1 - sx0 < img.shape[1] or sy1 - sy0 < img.shape[0]:
                continue
            self.items.append((t, img, (sx0, sy0, sx1, sy1)))

    def __len__(self) -> int:
        return len(self.items)

    def match(self, gray: np.ndarray) -> list[tuple[Template, float]]:
        out = []
        for t, img, (x0, y0, x1, y1) in self.items:
            res = cv2.matchTemplate(gray[y0:y1, x0:x1], img, cv2.TM_CCOEFF_NORMED)
            score = float(np.nan_to_num(res.max(), nan=0.0))
            out.append((t, score))
        return out
