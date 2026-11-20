"""Analyse one imported recording: audio + sampled frames -> events -> highlights -> DB."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..config import Settings, app_home
from ..db import Database, now_iso
from .audio import SR, analyze_audio
from .decode import Cancelled, analysis_size, extract_jpeg, iter_frames, read_audio
from .detect import Timeline, apply_status, build_highlights, detect_events, primary_type
from .templates import load_templates
from .visual import MotionMeter, TemplateMatcher, scope_score

log = logging.getLogger(__name__)
Progress = Callable[[float, str], None]


def build_timeline(path: Path, info: dict, s: Settings, progress: Progress, cancel=None) -> Timeline:
    duration = float(info["duration"])
    audio = None
    if info.get("audio_codec"):
        progress(0.02, "reading audio")
        samples = read_audio(path, s.ffmpeg_path, SR, cancel)
        progress(0.15, "finding gunshots and impacts")
        audio = analyze_audio(samples)
        del samples

    size = analysis_size(int(info.get("width") or 1920), int(info.get("height") or 1080), s.analysis_width)
    matcher = TemplateMatcher(load_templates(), *size)
    motion_m = MotionMeter()
    times, motion, scope, hits = [], [], [], []
    expected = max(1, int(duration * s.analysis_fps))
    for i, (t, frame) in enumerate(iter_frames(path, s.analysis_fps, size, s.ffmpeg_path,
                                                s.analysis_hwaccel, cancel)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        times.append(t)
        motion.append(motion_m(gray))
        scope.append(scope_score(gray))
        for tpl, score in matcher.match(gray):
            if score >= tpl.threshold:
                hits.append((t, tpl.event, tpl.name, score))
        if i % 20 == 0:
            progress(0.2 + 0.6 * min(1.0, i / expected), f"scanning video {int(100 * min(1, i / expected))}%")
    return Timeline(duration, s.analysis_fps, np.array(times), np.array(motion), np.array(scope),
                    audio, hits, len(matcher))


def ai_review(path: Path, highlights: list[dict], s: Settings, progress: Progress, cancel=None,
              client=None) -> dict:
    from .ai import AIUnavailable, make_client, merge_review, review_highlight
    summary = {"enabled": True, "reviewed": 0, "errors": []}
    try:
        client = client or make_client(s)
    except AIUnavailable as e:
        summary.update(enabled=False, reason=str(e))
        return summary
    cands = sorted(highlights, key=lambda h: -h["confidence"])[: s.ai_max_candidates]
    for i, h in enumerate(cands):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        progress(0.82 + 0.15 * i / max(1, len(cands)), f"AI review {i + 1}/{len(cands)}")
        k = s.ai_frames_per_candidate
        ts = np.linspace(h["start"], max(h["start"], h["end"] - 0.2), k)
        try:
            frames = [(float(t), extract_jpeg(path, float(t), 768, s.ffmpeg_path)) for t in ts]
            merge_review(h, review_highlight(client, s, h, frames))
            summary["reviewed"] += 1
        except AIUnavailable as e:
            summary.update(enabled=False, reason=str(e))
            break
        except Exception as e:  # keep the local result for this candidate
            log.warning("AI review failed for highlight at %.1fs: %s", h["start"], e)
            summary["errors"].append(str(e)[:200])
    for h in highlights:
        h["type"] = primary_type([t for t in h["types"]])
    apply_status(highlights, s)
    return summary


def analyze_recording(s: Settings, db: Database, rec_id: int, progress: Progress | None = None,
                      cancel=None, use_ai: bool | None = None, ai_client=None) -> dict:
    progress = progress or (lambda f, st: None)
    rec = db.get_recording(rec_id)
    if not rec or rec["status"] != "imported":
        raise ValueError(f"recording #{rec_id} is not imported")
    path = Path(rec["library_path"] or rec["path"])
    if not path.exists():
        raise FileNotFoundError(f"recording file is missing: {path}")
    t0 = time.monotonic()
    db.update_recording(rec_id, analysis_status="analyzing")

    tl = build_timeline(path, rec, s, progress, cancel)
    progress(0.8, "scoring highlights")
    events = detect_events(tl, s)
    highlights = build_highlights(events, tl.duration, s)

    ai = {"enabled": False}
    if (s.ai_enabled if use_ai is None else use_ai) and highlights:
        ai = ai_review(path, highlights, s, progress, cancel, ai_client)

    if cancel is not None and cancel.is_set():
        raise Cancelled()
    ids = db.replace_auto_highlights(rec_id, highlights)
    selected = sum(h["status"] == "selected" for h in highlights)
    db.update_recording(rec_id, analysis_status="done", analyzed_at=now_iso())
    summary = {
        "recording_id": rec_id, "seconds": round(time.monotonic() - t0, 1), "events": len(events),
        "highlights": len(ids), "selected": selected, "templates_loaded": tl.templates_loaded,
        "has_audio": tl.audio is not None, "ai": ai,
    }
    _export(rec_id, tl, events, highlights, summary)
    msg = f"Analysed {rec['filename']}: {selected} highlight(s) selected, {len(highlights) - selected} below threshold"
    if ai.get("enabled") is False and ai.get("reason"):
        msg += f" (AI skipped: {ai['reason']})"
    db.add_event(msg, recording_id=rec_id)
    progress(1.0, "done")
    return summary


def _export(rec_id: int, tl: Timeline, events, highlights, summary) -> None:
    """Per-recording analysis dump for tuning (small: per-second signals only)."""
    try:
        from dataclasses import asdict
        d = app_home() / "analysis"
        d.mkdir(parents=True, exist_ok=True)
        ps = tl.per_second()
        scope_ps = np.zeros(len(ps["motion"]))
        if len(tl.times):
            np.maximum.at(scope_ps, np.clip(tl.times.astype(int), 0, len(scope_ps) - 1), tl.scope)
        data = {
            "summary": summary,
            "per_second": {k: [round(float(x), 3) for x in v] for k, v in {**ps, "scope": scope_ps}.items()},
            "template_hits": [[round(t, 2), e, n, round(sc, 3)] for t, e, n, sc in tl.template_hits],
            "events": [asdict(e) for e in events],
            "highlights": highlights,
        }
        (d / f"recording_{rec_id}.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    except Exception:
        log.exception("could not write analysis export")
