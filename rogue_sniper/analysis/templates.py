"""Saved HUD templates: crops of Fortnite on-screen messages taken from YOUR footage.

Generic AI and color heuristics can't reliably read Fortnite's HUD, but an exact crop of
your own "ELIMINATED" / "VICTORY ROYALE" pop-up matches very reliably on later recordings
(same resolution/UI scale). Create them with `python -m rogue_sniper make-template`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np

from ..config import app_home, unique_path
from .decode import analysis_size, extract_frame
from .visual import Template

EVENTS = ("elimination", "knocked", "victory_royale", "custom")


def templates_dir() -> Path:
    return app_home() / "templates"


def load_templates() -> list[Template]:
    out = []
    root = templates_dir()
    if not root.is_dir():
        return out
    for meta_path in sorted(root.glob("*/*.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            img = cv2.imread(str(meta_path.with_suffix(".png")), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            out.append(Template(event=meta["event"], name=meta_path.stem, image=img,
                                region=tuple(meta["region"]), frame_width=int(meta["frame_width"]),
                                threshold=float(meta.get("threshold", 0.75))))
        except (OSError, ValueError, KeyError):
            continue
    return out


def list_templates() -> list[dict]:
    return [{"event": t.event, "name": t.name, "region": t.region, "threshold": t.threshold,
             "size": [int(t.image.shape[1]), int(t.image.shape[0])]} for t in load_templates()]


def delete_template(event: str, name: str) -> bool:
    d = templates_dir() / event
    found = False
    for ext in (".png", ".json"):
        p = d / f"{name}{ext}"
        if p.exists():
            p.unlink()
            found = True
    return found


def make_template(recording: Path | str, t: float, region_px: tuple[int, int, int, int],
                  src_w: int, src_h: int, event: str, analysis_width: int, name: str | None = None,
                  threshold: float = 0.75, ffmpeg: str = "ffmpeg") -> Path:
    """Crop `region_px` (x1,y1,x2,y2 in the recording's full-resolution pixels) at time `t`."""
    if event not in EVENTS:
        raise ValueError(f"event must be one of {EVENTS}")
    x1, y1, x2, y2 = region_px
    if not (0 <= x1 < x2 <= src_w and 0 <= y1 < y2 <= src_h):
        raise ValueError(f"region must be inside the {src_w}x{src_h} frame")
    rel = (x1 / src_w, y1 / src_h, x2 / src_w, y2 / src_h)
    # Same size + scaler as the analysis frames, so the template matches pixel-for-pixel.
    frame = extract_frame(recording, t, ffmpeg=ffmpeg, size=analysis_size(src_w, src_h, analysis_width))
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    crop = g[int(rel[1] * h):int(np.ceil(rel[3] * h)), int(rel[0] * w):int(np.ceil(rel[2] * w))]
    if crop.shape[0] < 6 or crop.shape[1] < 6:
        raise ValueError("region is too small - select at least the whole word/icon")
    if float(crop.std()) < 8:
        raise ValueError("region is almost a flat color - choose the text/icon itself so it can be recognised")
    d = templates_dir() / event
    d.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", name or event).strip("_") or event
    png = unique_path(d / f"{stem}.png")
    cv2.imwrite(str(png), crop)
    png.with_suffix(".json").write_text(json.dumps({
        "event": event, "region": rel, "frame_width": w, "threshold": threshold,
        "source": str(recording), "time": t}, indent=2), encoding="utf-8")
    return png


def set_threshold(event: str, name: str, threshold: float) -> bool:
    if not 0.3 <= threshold <= 0.99:
        raise ValueError("threshold must be between 0.3 and 0.99")
    meta_path = templates_dir() / event / f"{name}.json"
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["threshold"] = threshold
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return True
