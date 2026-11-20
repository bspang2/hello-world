"""Optional Claude vision review of candidate highlights.

Only short candidate windows are sent (a few downscaled JPEG frames each, plus the local
signal summary) - never the whole recording. Requires an Anthropic API key; without one,
everything else still works and this step is skipped.

Cost guard: at most `ai_max_candidates` x `ai_frames_per_candidate` images per recording.
"""
from __future__ import annotations

import base64
import logging
from typing import Literal

from pydantic import BaseModel

from ..config import Settings
from .detect import fmt_t

log = logging.getLogger(__name__)

AI_TYPES = Literal["elimination", "multi_elimination", "sniper_shot", "victory_royale", "clutch",
                   "funny_failure", "unusual", "high_action_combat", "knocked", "none"]


class AIReview(BaseModel):
    is_highlight: bool
    highlight_score: float          # 0..1 how strong a short-form highlight this is
    event_types: list[AI_TYPES]
    description: str                # one or two sentences, what actually happens
    reasoning: str                  # short: what in the frames supports this


NEGATIVE_CAP = 0.45


class AIUnavailable(RuntimeError):
    """AI step can't run at all (no key, bad key, SDK missing) - skip it for this recording."""


SYSTEM = """You review short excerpts of Fortnite gameplay recorded by a creator who posts highlights \
to TikTok. You get a few frames in time order plus signals from a local detector (audio transients, \
scope detection, HUD template matches), which can be wrong.

Judge only what the frames show. Fortnite cues: the "ELIMINATED <name>" / knocked messages and the \
elimination counter in the HUD, the kill feed, the circular black-bordered sniper scope, damage \
numbers, the "#1 VICTORY ROYALE" screen, storm, building fights. Say "none" if nothing notable is \
visible. Do not invent player names, numbers or events you cannot see. A funny failure is something \
like falling to death, missing an easy shot or an awkward elimination of the creator; clutch means \
winning a fight while clearly at a disadvantage (low health, outnumbered)."""


def make_client(settings: Settings):
    from ..credentials import get_api_key
    try:
        import anthropic
    except ImportError as e:
        raise AIUnavailable("the 'anthropic' package is not installed (run install.bat)") from e
    key = get_api_key()
    if not key:
        raise AIUnavailable("no Anthropic API key configured (Settings -> AI, or `python -m rogue_sniper set-api-key`)")
    return anthropic.Anthropic(api_key=key, max_retries=3, timeout=120.0)


def review_highlight(client, settings: Settings, highlight: dict, frames: list[tuple[float, bytes]]) -> AIReview:
    import anthropic

    signals = "\n".join(f"- {e['type']} at {fmt_t(e['time'])}: {e['description']} "
                        f"(local confidence {e['confidence']:.2f})"
                        for e in highlight.get("signals", {}).get("events", []))
    content: list[dict] = [{"type": "text", "text":
                            f"Excerpt {fmt_t(highlight['start'])}-{fmt_t(highlight['end'])} of a recording.\n"
                            f"Local detector signals:\n{signals or '- none'}\n\nFrames follow in time order."}]
    for t, jpg in frames:
        content.append({"type": "text", "text": f"Frame at {fmt_t(t)}:"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": base64.standard_b64encode(jpg).decode("ascii")}})
    content.append({"type": "text", "text": "Classify this excerpt."})
    try:
        resp = client.beta.messages.parse(
            model=settings.ai_model,
            max_tokens=8000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": settings.ai_effort},
            system=SYSTEM,
            messages=[{"role": "user", "content": content}],
            output_format=AIReview,
        )
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
        raise AIUnavailable(f"API key rejected: {e.message}") from e
    except anthropic.NotFoundError as e:
        raise AIUnavailable(f"model '{settings.ai_model}' not available: {e.message}") from e
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        raise RuntimeError(f"no usable answer (stop_reason={resp.stop_reason})")
    r = resp.parsed_output
    r.highlight_score = max(0.0, min(1.0, float(r.highlight_score)))
    return r


def merge_review(h: dict, r: AIReview) -> None:
    """Blend AI judgement into a highlight dict in place."""
    local = h.get("local_confidence", h["confidence"])
    ai_score = r.highlight_score if r.is_highlight else min(r.highlight_score, 0.3)
    h["ai_confidence"] = round(ai_score, 3)
    blended = 0.5 * local + 0.5 * ai_score
    if not r.is_highlight:  # a clear "no" drops it below the default threshold (still visible for review)
        blended = min(blended, NEGATIVE_CAP)
    h["confidence"] = round(blended, 3)
    h["ai_description"] = r.description
    extra = [t for t in r.event_types if t != "none"]
    h["types"] = list(dict.fromkeys(h["types"] + extra))
    h["signals"]["ai"] = {"is_highlight": r.is_highlight, "event_types": list(r.event_types),
                          "reasoning": r.reasoning}
    h["source"] = "ai"
