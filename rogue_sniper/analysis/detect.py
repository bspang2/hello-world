"""Turn raw signals into events, then into scored highlight windows.

Every number here is a heuristic. Confidence means "how strongly the local signals point
to this", not a calibrated probability; thresholds are adjustable in Settings and should
be tuned on your own footage.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..config import Settings
from .audio import AudioFeatures

# Higher = more important when several events merge into one highlight.
PRIORITY = ["victory_royale", "multi_elimination", "sniper_shot", "elimination", "clutch",
            "funny_failure", "unusual", "knocked", "high_action_combat"]
SENSITIVITY_THRESHOLD = {"low": 0.65, "medium": 0.5, "high": 0.35}
# seconds of footage before/after the key moment
WINDOWS = {"sniper_shot": (5, 3), "elimination": (7, 3), "knocked": (5, 3), "multi_elimination": (6, 3),
           "victory_royale": (12, 5), "high_action_combat": (2, 2)}


@dataclass
class Timeline:
    duration: float
    fps: float
    times: np.ndarray                 # frame sample times
    motion: np.ndarray
    scope: np.ndarray
    audio: AudioFeatures | None
    template_hits: list[tuple[float, str, str, float]] = field(default_factory=list)  # t, event, name, score
    templates_loaded: int = 0
    template_scores: dict[str, np.ndarray] = field(default_factory=dict)  # "event/name" -> per-frame score

    def per_second(self) -> dict[str, np.ndarray]:
        n = max(1, int(np.ceil(self.duration)))
        motion = np.zeros(n)
        if len(self.times):
            idx = np.clip(self.times.astype(int), 0, n - 1)
            sums, counts = np.zeros(n), np.zeros(n)
            np.add.at(sums, idx, self.motion)
            np.add.at(counts, idx, 1)
            motion = np.divide(sums, counts, out=np.zeros(n), where=counts > 0)
        p95 = float(np.percentile(motion, 95)) if motion.any() else 1.0
        out = {"motion": np.clip(motion / max(p95, 1e-6), 0, 1)}
        if self.audio is not None:
            out.update(self.audio.per_second(self.duration))
        else:
            out.update({"shots": np.zeros(n), "loudness": np.zeros(n)})
        return out


@dataclass
class Event:
    type: str
    time: float
    start: float
    end: float
    confidence: float
    description: str
    signals: dict = field(default_factory=dict)


def fmt_t(s: float) -> str:
    s = max(0, int(round(s)))
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


# ---- event detectors -------------------------------------------------------
def detect_combat(tl: Timeline, ps: dict, min_shots: int) -> list[Event]:
    shots = ps["shots"]
    n = len(shots)
    win = 5
    if n == 0 or not shots.any():
        return []
    w = np.convolve(shots, np.ones(win), mode="full")[win - 1:win - 1 + n]  # w[s] = shots in [s, s+5)
    taken = np.zeros(n, bool)
    events = []
    times = tl.audio.transient_times if tl.audio is not None else np.array([])
    for s in np.argsort(-w, kind="stable"):
        if w[s] < min_shots:
            break
        if taken[max(0, s - 8):min(n, s + win + 8)].any():
            continue
        taken[s:min(n, s + win)] = True
        in_win = times[(times >= s) & (times < s + win)]
        start, end = (float(in_win.min()), float(in_win.max())) if len(in_win) else (float(s), float(s + win))
        sl = slice(s, min(n, s + win))
        motion, loud = float(ps["motion"][sl].mean()), float(ps["loudness"][sl].mean())
        conf = min(0.9, 0.4 + 0.04 * (w[s] - min_shots) + 0.15 * motion + 0.1 * loud)
        events.append(Event("high_action_combat", (start + end) / 2, start, end, round(conf, 3),
                            f"Heavy fighting {fmt_t(start)}-{fmt_t(end)}: {int(w[s])} shots/impacts in {win}s",
                            {"shots": int(w[s]), "motion": round(motion, 2), "loudness": round(loud, 2)}))
    return events


def _runs(mask: np.ndarray, max_gap: int = 1) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    runs, a, b = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - b <= max_gap + 1:
            b = i
        else:
            runs.append((a, b))
            a = b = i
    runs.append((a, b))
    return runs


def detect_sniper(tl: Timeline, threshold: float) -> list[Event]:
    if not len(tl.scope):
        return []
    events = []
    times = tl.audio.transient_times if tl.audio is not None else np.array([])
    strength = tl.audio.transient_strength if tl.audio is not None else np.array([])
    min_frames = max(2, int(round(0.5 * tl.fps)))
    for a, b in _runs(tl.scope >= threshold, max_gap=1):
        if b - a + 1 < min_frames:
            continue
        t0, t1 = float(tl.times[a]), float(tl.times[b])
        mean_scope = float(tl.scope[a:b + 1].mean())
        sel = (times >= t0) & (times <= t1 + 0.75)
        if sel.any():
            k = int(np.argmax(np.where(sel, strength, -np.inf)))
            z = float(strength[k])
            conf = 0.5 + 0.25 * mean_scope + 0.15 * min(1.0, z / 10)
            events.append(Event("sniper_shot", float(times[k]), t0, max(t1, float(times[k])), round(conf, 3),
                                f"Scoped in {t1 - t0 + 1 / tl.fps:.1f}s and fired at {fmt_t(times[k])}",
                                {"scope": round(mean_scope, 2), "shot_strength": round(z, 1),
                                 "shots_while_scoped": int(sel.sum())}))
        else:
            events.append(Event("sniper_shot", t1, t0, t1, round(0.3 + 0.1 * mean_scope, 3),
                                f"Scoped in at {fmt_t(t0)} (no clear shot heard)", {"scope": round(mean_scope, 2)}))
    return events


def detect_templates(tl: Timeline) -> list[Event]:
    """Template hits -> events; consecutive hits of the same pop-up count once."""
    events: list[Event] = []
    by_event: dict[str, list[tuple[float, float, float]]] = {}
    gap = 1.5
    for t, ev, name, score in sorted(tl.template_hits):
        runs = by_event.setdefault(ev, [])
        if runs and t - runs[-1][1] <= gap:
            s0, _, best = runs[-1]
            runs[-1] = (s0, t, max(best, score))
        else:
            runs.append((t, t, score))
    for ev, runs in by_event.items():
        for s0, s1, best in runs:
            conf = round(min(0.97, 0.6 + 0.37 * max(0.0, (best - 0.7) / 0.3)), 3)
            label = {"elimination": "Elimination", "knocked": "Knocked an enemy",
                     "victory_royale": "VICTORY ROYALE"}.get(ev, ev)
            events.append(Event(ev, s0, s0, s1, conf, f"{label} at {fmt_t(s0)} (HUD match {best:.2f})",
                                {"template_score": round(best, 3)}))
    return events


def group_multi_eliminations(events: list[Event], window: float = 20.0) -> list[Event]:
    elims = sorted((e for e in events if e.type == "elimination"), key=lambda e: e.time)
    others = [e for e in events if e.type != "elimination"]
    clusters: list[list[Event]] = []
    for e in elims:
        if clusters and e.time - clusters[-1][-1].time <= window:
            clusters[-1].append(e)
        else:
            clusters.append([e])
    out = others
    for c in clusters:
        if len(c) == 1:
            out.append(c[0])
            continue
        conf = min(0.98, max(e.confidence for e in c) + 0.05 * (len(c) - 1))
        label = {2: "Double", 3: "Triple", 4: "Quad"}.get(len(c), f"{len(c)}x")
        out.append(Event("multi_elimination", c[0].time, c[0].time, c[-1].time, round(conf, 3),
                         f"{label} elimination {fmt_t(c[0].time)}-{fmt_t(c[-1].time)}",
                         {"count": len(c), "times": [round(e.time, 2) for e in c]}))
    return out


def detect_events(tl: Timeline, s: Settings) -> list[Event]:
    ps = tl.per_second()
    events = detect_combat(tl, ps, s.combat_min_shots.get(s.sensitivity, 6))
    events += detect_sniper(tl, s.scope_threshold)
    events += group_multi_eliminations(detect_templates(tl))
    for e in events:  # a win is near the end of a recording
        if e.type == "victory_royale" and tl.duration - e.time < 90:
            e.confidence = round(min(0.98, e.confidence + 0.05), 3)
    return sorted(events, key=lambda e: e.time)


# ---- highlights --------------------------------------------------------------
def threshold_for(s: Settings, typ: str) -> float:
    return float(s.type_thresholds.get(typ, SENSITIVITY_THRESHOLD[s.sensitivity]))


def primary_type(types: list[str]) -> str:
    return min(types, key=lambda t: PRIORITY.index(t) if t in PRIORITY else len(PRIORITY))


MERGE_GAP = 4.0  # key moments closer than this become one clip


def _should_merge(cluster: list, sp: list, max_len: float) -> bool:
    """Merge only genuinely connected moments, so separate plays stay separate videos.

    Padded windows touching is not enough: the key moments must be within MERGE_GAP seconds,
    or the windows must overlap by more than half, and the result must fit max_clip_seconds.
    """
    if max(cluster[1], sp[1]) - cluster[0] > max_len:
        return False
    key_end = max(e.end for e in cluster[2])
    key_start = min(e.start for e in sp[2])
    if key_start - key_end <= MERGE_GAP:
        return True
    inter = min(cluster[1], sp[1]) - max(cluster[0], sp[0])
    return inter > 0.5 * min(cluster[1] - cluster[0], sp[1] - sp[0])


def build_highlights(events: list[Event], duration: float, s: Settings) -> list[dict]:
    spans = []
    for e in events:
        pre, post = WINDOWS.get(e.type, (5, 3))
        spans.append([max(0.0, e.start - pre), min(duration, e.end + post), [e]])
    spans.sort(key=lambda x: x[0])
    merged: list[list] = []
    for sp in spans:
        if merged and _should_merge(merged[-1], sp, s.max_clip_seconds):
            merged[-1][1] = max(merged[-1][1], sp[1])
            merged[-1][2].extend(sp[2])
        else:
            merged.append(sp)

    out = []
    for start, end, evs in merged:
        evs.sort(key=lambda e: -e.confidence)
        best = evs[0]
        # noisy-OR of independent evidence, extra events damped
        miss = (1 - best.confidence) * np.prod([1 - 0.5 * e.confidence for e in evs[1:]])
        conf = round(float(min(0.98, 1 - miss)), 3)
        if end - start > s.max_clip_seconds:
            start = max(start, best.time - 0.6 * s.max_clip_seconds)
            end = min(duration, start + s.max_clip_seconds)
        types = sorted({e.type for e in evs}, key=lambda t: PRIORITY.index(t) if t in PRIORITY else 99)
        typ = primary_type(types)
        out.append({
            "start": round(start, 2), "end": round(end, 2), "peak": round(best.time, 2),
            "type": typ, "types": types, "confidence": conf, "local_confidence": conf,
            "description": "; ".join(e.description for e in sorted(evs, key=lambda e: e.time)),
            "signals": {"events": [asdict(e) for e in sorted(evs, key=lambda e: e.time)]},
            "source": "local",
        })
    apply_status(out, s)
    return out


def apply_status(highlights: list[dict], s: Settings) -> None:
    """selected = above its type's threshold and among the top N; otherwise below_threshold."""
    ranked = sorted(highlights, key=lambda h: -h["confidence"])
    n_sel = 0
    for h in ranked:
        if h["confidence"] >= threshold_for(s, h["type"]) and n_sel < s.max_highlights_per_recording:
            h["status"] = "selected"
            n_sel += 1
        else:
            h["status"] = "below_threshold"
