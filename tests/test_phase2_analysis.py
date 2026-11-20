"""Phase 2: highlight detection, manual correction, jobs, optional AI review."""
import json
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from conftest import GAME_TRUTH, POPUP_BOX, make_clip, make_game_clip
from fastapi.testclient import TestClient

from rogue_sniper import cli, credentials
from rogue_sniper.analysis.ai import AIReview
from rogue_sniper.analysis.audio import AudioFeatures, analyze_audio
from rogue_sniper.analysis.decode import extract_frame, read_audio
from rogue_sniper.analysis.detect import Event, Timeline, build_highlights, detect_combat, detect_events
from rogue_sniper.analysis.pipeline import analyze_recording, build_timeline
from rogue_sniper.analysis.templates import list_templates, make_template
from rogue_sniper.analysis.visual import scope_score
from rogue_sniper.api import create_app
from rogue_sniper.config import FAILED_DIR, RAW_DIR, Settings, app_home, ensure_workspace
from rogue_sniper.db import Database
from rogue_sniper.jobs import JobRunner
from rogue_sniper.media import probe
from rogue_sniper.service import MonitorService


@pytest.fixture(scope="session")
def game_clip(tmp_path_factory):
    return make_game_clip(tmp_path_factory.mktemp("game") / "Fortnite 2026.11.20 - 21.00.00.01.mp4")


@pytest.fixture(scope="session")
def silent_game_clip(tmp_path_factory):
    return make_game_clip(tmp_path_factory.mktemp("game_na") / "noaudio.mp4", seconds=30, audio=False)


def import_clip(db, settings, path) -> int:
    """Register an already-finished file as imported (Phase 1 is tested separately)."""
    info = probe(path)
    rid = db.insert_waiting(str(path), path.name, path.stat().st_size, path.stat().st_mtime, "shadowplay")
    db.update_recording(rid, status="imported", **{k: v for k, v in info.as_dict().items()})
    return rid


def elim_template(clip, settings):
    return make_template(clip, 30.5, POPUP_BOX, 1280, 720, "elimination", settings.analysis_width)


def by_type(highlights, typ):
    return [h for h in highlights if h["type"] == typ]


# ---- signals -----------------------------------------------------------------
def test_audio_finds_every_shot_and_nothing_else(game_clip):
    feats = analyze_audio(read_audio(game_clip))
    t = feats.transient_times
    b0, b1 = GAME_TRUTH["burst"]
    burst = t[(t >= b0 - 0.05) & (t <= b1 + 0.1)]
    assert 19 <= len(burst) <= 22                         # 5 shots/s for 4 s
    assert np.min(np.abs(t - GAME_TRUTH["sniper_shot"])) < 0.05
    assert len(t) - len(burst) == 1                       # only the sniper shot outside the burst


def test_scope_detector(game_clip):
    gray = lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert scope_score(gray(extract_frame(game_clip, 21.0, width=640))) > 0.8
    assert scope_score(gray(extract_frame(game_clip, 10.0, width=640))) < 0.05
    # Black pillarbox bars (straight edges) must not look like a scope
    pillar = np.full((360, 640), 120, np.uint8)
    pillar[:, :110] = 0
    pillar[:, -110:] = 0
    assert scope_score(pillar) < 0.05
    assert scope_score(np.zeros((360, 640), np.uint8)) == 0       # black loading screen
    assert scope_score(np.full((360, 640), 20, np.uint8)) == 0    # very dark scene


def test_template_matches_only_when_popup_visible(game_clip, settings):
    elim_template(game_clip, settings)
    assert list_templates()[0]["event"] == "elimination"
    tl = build_timeline(game_clip, probe(game_clip).as_dict(), settings, lambda f, s: None)
    hit_secs = sorted({int(h[0]) for h in tl.template_hits})
    assert hit_secs == [8, 9, 30, 31, 34, 35]


def test_template_rejects_flat_region(game_clip, settings):
    with pytest.raises(ValueError):
        make_template(game_clip, 21.0, (0, 300, 100, 400), 1280, 720, "elimination", 640)  # black scope edge


def test_hwaccel_falls_back_to_cpu(game_clip, settings):
    settings.analysis_hwaccel = "cuda"  # no GPU in CI: must fall back, not fail
    tl = build_timeline(game_clip, probe(game_clip).as_dict(), settings, lambda f, s: None)
    assert len(tl.times) >= 59 * settings.analysis_fps


# ---- detection logic ---------------------------------------------------------
def _tl(shot_times, duration=60.0, scope=None):
    feats = AudioFeatures(0.02, np.full(int(duration / 0.02), -30.0), np.array(shot_times, float),
                          np.full(len(shot_times), 12.0), -60, -10)
    times = np.arange(0, duration, 0.25)
    return Timeline(duration, 4.0, times, np.zeros(len(times)),
                    scope if scope is not None else np.zeros(len(times)), feats)


def test_combat_threshold_follows_sensitivity():
    tl = _tl([10, 10.8, 11.6, 12.4, 13.2])  # 5 shots in 5 s
    ps = tl.per_second()
    s = Settings()
    assert detect_combat(tl, ps, s.combat_min_shots["medium"]) == []
    assert len(detect_combat(tl, ps, s.combat_min_shots["high"])) == 1


def test_separate_plays_stay_separate_and_close_ones_merge():
    s = Settings()
    evs = [Event("sniper_shot", 20, 20, 20, 0.9, "a"), Event("elimination", 23, 23, 23, 0.8, "b"),
           Event("high_action_combat", 50, 48, 52, 0.7, "c")]
    hl = build_highlights(evs, 120, s)
    assert len(hl) == 2
    assert hl[0]["type"] == "sniper_shot" and set(hl[0]["types"]) == {"sniper_shot", "elimination"}
    assert hl[0]["confidence"] > 0.9            # two signals agree
    s.max_clip_seconds = 10
    for h in build_highlights([Event("high_action_combat", 50, 30, 70, 0.7, "long")], 120, s):
        assert h["end"] - h["start"] <= 10


def test_max_highlights_cap():
    s = Settings(max_highlights_per_recording=2)
    evs = [Event("sniper_shot", t, t, t, 0.9 - t / 1000, "x") for t in (20, 60, 100, 140)]
    hl = build_highlights(evs, 200, s)
    assert [h["status"] for h in hl].count("selected") == 2


# ---- full pipeline -------------------------------------------------------------
def test_pipeline_finds_ground_truth(game_clip, settings, db):
    elim_template(game_clip, settings)
    rid = import_clip(db, settings, game_clip)
    stages = []
    summary = analyze_recording(settings, db, rid, lambda f, s: stages.append((f, s)), use_ai=False)
    hl = db.list_highlights(rid)
    sel = [h for h in hl if h["status"] == "selected"]
    assert summary["selected"] == len(sel) == 4

    sniper = by_type(sel, "sniper_shot")[0]
    assert abs(sniper["peak"] - GAME_TRUTH["sniper_shot"]) < 0.1 and sniper["start"] < 21.5 < sniper["end"]
    multi = by_type(sel, "multi_elimination")[0]
    assert multi["signals"]["events"][0]["signals"]["count"] == 2
    assert multi["start"] <= 30 and multi["end"] >= 34
    combat = by_type(sel, "high_action_combat")[0]
    assert combat["start"] <= 40 and combat["end"] >= 44
    single = by_type(sel, "elimination")[0]
    assert single["start"] <= 8 <= single["end"]

    assert stages[-1] == (1.0, "done") and all(a[0] <= b[0] for a, b in zip(stages, stages[1:]))
    rec = db.get_recording(rid)
    assert rec["analysis_status"] == "done" and rec["analyzed_at"]
    dump = json.loads((app_home() / "analysis" / f"recording_{rid}.json").read_text())
    assert len(dump["per_second"]["shots"]) == 60 and dump["summary"]["selected"] == 4


def test_without_templates_no_eliminations_claimed(game_clip, settings, db):
    rid = import_clip(db, settings, game_clip)
    analyze_recording(settings, db, rid, use_ai=False)
    types = {t for h in db.list_highlights(rid) for t in h["types"]}
    assert "sniper_shot" in types and "high_action_combat" in types
    assert not types & {"elimination", "multi_elimination"}


def test_quiet_footage_gives_no_highlights(tmp_path, settings, db):
    clip = make_clip(tmp_path / "quiet.mp4", seconds=20, fps=30, size="640x360")  # steady tone, no events
    rid = import_clip(db, settings, clip)
    analyze_recording(settings, db, rid, use_ai=False)
    assert db.list_highlights(rid, "selected") == []


def test_no_audio_recording(silent_game_clip, settings, db):
    rid = import_clip(db, settings, silent_game_clip)
    summary = analyze_recording(settings, db, rid, use_ai=False)
    assert summary["has_audio"] is False
    hl = db.list_highlights(rid)
    sn = by_type(hl, "sniper_shot")
    assert sn and sn[0]["status"] == "below_threshold"   # scoped, but no shot to confirm it


def test_sensitivity_changes_selection(game_clip, settings, db):
    rid = import_clip(db, settings, game_clip)
    tl = build_timeline(game_clip, probe(game_clip).as_dict(), settings, lambda f, s: None)
    counts = {}
    for sens in ("low", "medium", "high"):
        s = settings.model_copy(update={"sensitivity": sens})
        counts[sens] = sum(h["status"] == "selected" for h in build_highlights(detect_events(tl, s), 60, s))
    assert counts["low"] <= counts["medium"] <= counts["high"]


def test_manual_corrections_survive_reanalysis(game_clip, settings, db):
    rid = import_clip(db, settings, game_clip)
    analyze_recording(settings, db, rid, use_ai=False)
    combat = by_type(db.list_highlights(rid), "high_action_combat")[0]
    db.update_highlight(combat["id"], status="rejected", user_edited=1)
    manual = db.insert_highlight(rid, {"start": 50, "end": 58, "type": "funny_failure", "confidence": 1.0,
                                       "source": "manual", "status": "approved", "user_edited": 1})
    analyze_recording(settings, db, rid, use_ai=False)
    hl = db.list_highlights(rid)
    combats = by_type(hl, "high_action_combat")
    assert len(combats) == 1 and combats[0]["id"] == combat["id"] and combats[0]["status"] == "rejected"
    assert db.get_highlight(manual)["status"] == "approved"
    assert len(by_type(hl, "sniper_shot")) == 1  # machine highlights replaced, not duplicated


# ---- jobs ----------------------------------------------------------------------
def test_import_triggers_analysis_job(game_clip, settings, db, clock):
    import shutil
    shutil.copy(game_clip, settings.dir(RAW_DIR) / game_clip.name)
    svc = MonitorService(settings, db, clock=clock, lock_check=lambda p: False)
    svc.scan_once()
    clock.advance(11)
    rep = svc.scan_once()
    assert len(rep.imported) == 1
    job = db.list_jobs()[0]
    assert job["status"] == "queued" and db.get_recording(rep.imported[0])["analysis_status"] == "queued"
    assert svc.jobs.run_pending_once()
    job = db.get_job(job["id"])
    assert job["status"] == "done" and job["progress"] == 1.0
    assert db.list_highlights(rep.imported[0], "selected")


def test_enqueue_is_idempotent(settings, db):
    rid = db.insert_waiting("x.mp4", "x.mp4", 1, 1, "obs")
    r = JobRunner(db, {}, lambda: settings)
    assert r.enqueue(rid) == r.enqueue(rid)


def test_job_cancel(settings, db):
    rid = db.insert_waiting("x.mp4", "x.mp4", 1, 1, "obs")
    started = threading.Event()

    def slow(job, progress, cancel):
        from rogue_sniper.analysis.decode import Cancelled
        started.set()
        for i in range(500):
            progress(i / 500, "working")
            if cancel.wait(0.01):
                raise Cancelled()

    r = JobRunner(db, {"analyze": slow}, lambda: settings, poll=0.05)
    jid = r.enqueue(rid)
    r.start()
    try:
        assert started.wait(5)
        assert r.cancel(jid)
        for _ in range(100):
            if db.get_job(jid)["status"] == "cancelled":
                break
            time.sleep(0.05)
        assert db.get_job(jid)["status"] == "cancelled"
        assert db.get_recording(rid)["analysis_status"] == "cancelled"
    finally:
        r.stop()


def test_job_retries_then_fails_with_report(settings, db):
    rid = db.insert_waiting("C:/x/Fortnite clip.mp4", "Fortnite clip.mp4", 1, 1, "obs")
    calls = []

    def boom(job, progress, cancel):
        calls.append(1)
        raise RuntimeError("decoder exploded")

    r = JobRunner(db, {"analyze": boom}, lambda: settings, max_attempts=3, retry_delay=0)
    jid = r.enqueue(rid)
    while r.run_pending_once():
        pass
    job = db.get_job(jid)
    assert len(calls) == 3 and job["status"] == "failed" and "decoder exploded" in job["error"]
    assert (settings.dir(FAILED_DIR) / "Fortnite clip - analysis error.txt").exists()


def test_retry_succeeds_second_time(settings, db):
    rid = db.insert_waiting("x.mp4", "x.mp4", 1, 1, "obs")
    calls = []

    def flaky(job, progress, cancel):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("file busy")

    r = JobRunner(db, {"analyze": flaky}, lambda: settings, retry_delay=0)
    jid = r.enqueue(rid)
    while r.run_pending_once():
        pass
    assert db.get_job(jid)["status"] == "done"


def test_interrupted_job_resumes_after_restart(settings, db):
    rid = db.insert_waiting("x.mp4", "x.mp4", 1, 1, "obs")
    jid = db.enqueue_job(rid)
    db.claim_next_job(time.time())  # simulate crash while running
    done = []
    r = JobRunner(db, {"analyze": lambda j, p, c: done.append(j["id"])}, lambda: settings)
    assert db.requeue_interrupted_jobs() == 1
    r.run_pending_once()
    assert done == [jid] and db.get_job(jid)["status"] == "done"


# ---- AI review (fake client: no network, no key) -------------------------------
class FakeClient:
    def __init__(self, review=None, error=None):
        self.calls = []
        self.review, self.error = review, error
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._parse))

    def _parse(self, **kw):
        self.calls.append(kw)
        if self.error:
            raise self.error
        return SimpleNamespace(stop_reason="end_turn", parsed_output=self.review.model_copy())


def test_ai_review_blends_into_highlights(game_clip, settings, db):
    rid = import_clip(db, settings, game_clip)
    settings.ai_max_candidates = 2
    fake = FakeClient(AIReview(is_highlight=True, highlight_score=0.95, event_types=["clutch", "sniper_shot"],
                               description="Hits a long-range snipe while scoped.", reasoning="scope + hit"))
    summary = analyze_recording(settings, db, rid, use_ai=True, ai_client=fake)
    assert summary["ai"]["reviewed"] == 2 and len(fake.calls) == 2
    call = fake.calls[0]
    assert call["model"] == "claude-opus-5-5" and call["fallbacks"] == "default"
    assert call["output_format"] is AIReview
    images = [b for b in call["messages"][0]["content"] if b["type"] == "image"]
    assert len(images) == settings.ai_frames_per_candidate
    reviewed = [h for h in db.list_highlights(rid) if h["ai_confidence"] is not None]
    assert len(reviewed) == 2
    h = reviewed[0]
    assert h["source"] == "ai" and "clutch" in h["types"] and h["ai_description"].startswith("Hits")
    assert h["confidence"] == round(0.5 * h["local_confidence"] + 0.5 * 0.95, 3)


def test_ai_negative_verdict_lowers_confidence(game_clip, settings, db):
    rid = import_clip(db, settings, game_clip)
    settings.ai_max_candidates = 10
    fake = FakeClient(AIReview(is_highlight=False, highlight_score=0.1, event_types=["none"],
                               description="Nothing happens.", reasoning="static"))
    analyze_recording(settings, db, rid, use_ai=True, ai_client=fake)
    assert db.list_highlights(rid, "selected") == []


def test_ai_errors_keep_local_results(game_clip, settings, db):
    rid = import_clip(db, settings, game_clip)
    summary = analyze_recording(settings, db, rid, use_ai=True, ai_client=FakeClient(error=RuntimeError("503")))
    assert summary["ai"]["errors"] and len(db.list_highlights(rid, "selected")) == 2


def test_ai_skipped_without_key(game_clip, settings, db, monkeypatch):
    monkeypatch.setattr(credentials, "get_api_key", lambda: None)
    rid = import_clip(db, settings, game_clip)
    summary = analyze_recording(settings, db, rid, use_ai=True)
    assert summary["ai"]["enabled"] is False and "API key" in summary["ai"]["reason"]
    assert db.list_highlights(rid, "selected")


# ---- credentials ---------------------------------------------------------------
class FakeKeyring:
    def __init__(self):
        self.store = {}

    def get_password(self, s, k):
        return self.store.get((s, k))

    def set_password(self, s, k, v):
        self.store[(s, k)] = v

    def delete_password(self, s, k):
        self.store.pop((s, k), None)


def test_api_key_never_exposed(settings, db, monkeypatch):
    kr = FakeKeyring()
    monkeypatch.setattr(credentials, "_keyring", lambda: kr)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = create_app(settings, db, start_monitor=False)
    secret = "sk-ant-test-0123456789abcdef"
    with TestClient(app) as c:
        assert c.get("/api/ai/status").json()["configured"] is False
        r = c.put("/api/ai/key", json={"api_key": secret})
        assert r.json() == {"configured": True, "source": "credential_manager"}
        for path in ("/api/ai/status", "/api/settings", "/api/health"):
            assert secret not in c.get(path).text
        assert c.delete("/api/ai/key").json()["configured"] is False
    assert not (app_home() / "settings.json").exists() or secret not in (app_home() / "settings.json").read_text()


# ---- API + CLI -----------------------------------------------------------------
def test_highlight_api(game_clip, settings, db):
    rid = import_clip(db, settings, game_clip)
    svc = MonitorService(settings, db)
    app = create_app(settings, db, start_monitor=False, service=svc)
    with TestClient(app) as c:
        job = c.post(f"/api/recordings/{rid}/analyze").json()["job_id"]
        assert c.get("/api/jobs").json()[0]["status"] == "queued"
        svc.jobs.run_pending_once()
        assert db.get_job(job)["status"] == "done"
        hl = c.get(f"/api/recordings/{rid}/highlights").json()
        assert hl and c.get("/api/stats").json()["highlights_identified"] == 2
        hid = hl[0]["id"]
        assert c.patch(f"/api/highlights/{hid}", json={"type": "made_up"}).status_code == 422
        assert c.patch(f"/api/highlights/{hid}", json={"start": 30, "end": 20}).status_code == 422
        assert c.patch(f"/api/highlights/{hid}", json={"end": 999}).status_code == 422
        r = c.patch(f"/api/highlights/{hid}", json={"status": "approved", "type": "clutch", "start": 1.0})
        assert r.json()["user_edited"] == 1 and r.json()["type"] == "clutch" and r.json()["start"] == 1.0
        r = c.post(f"/api/recordings/{rid}/highlights",
                   json={"start": 50, "end": 57, "type": "funny_failure", "description": "fell off a cliff"})
        assert r.json()["source"] == "manual" and r.json()["status"] == "approved"
        assert c.delete(f"/api/highlights/{r.json()['id']}").json()["ok"]
        assert c.post("/api/recordings/999/analyze").status_code == 404
        assert c.post(f"/api/jobs/{job}/cancel").status_code == 409  # already done
        assert c.get("/api/recordings").json()[0]["highlight_count"] >= 1


def test_cli_phase2(game_clip, settings, tmp_path, capsys, monkeypatch):
    from rogue_sniper.config import save_settings
    from rogue_sniper.db import Database as DB
    from rogue_sniper.config import db_path
    save_settings(settings)
    db = DB(db_path())
    rid = import_clip(db, settings, game_clip)
    db.close()
    out_png = tmp_path / "snap.png"
    assert cli.main(["snapshot", str(rid), "--time", "0:30.5", "--out", str(out_png)]) == 0
    assert cv2.imread(str(out_png)).shape == (720, 1280, 3)
    region = ",".join(map(str, POPUP_BOX))
    assert cli.main(["make-template", str(rid), "--time", "30.5", "--region", region, "--event", "elimination",
                     "--name", "my elim"]) == 0
    assert cli.main(["templates"]) == 0
    assert "my_elim" in capsys.readouterr().out
    assert cli.main(["analyze", str(rid), "--no-ai"]) == 0
    out = capsys.readouterr().out
    assert "4 highlights selected" in out and "multi_elimination" in out and "sniper_shot" in out
    assert cli.main(["highlight-add", str(rid), "--start", "0:50", "--end", "0:56", "--type", "funny_failure"]) == 0
    assert cli.main(["highlights", str(rid)]) == 0
    assert "funny_failure" in capsys.readouterr().out
    assert cli.main(["make-template", str(rid), "--time", "21", "--region", "0,300,100,400",
                     "--event", "elimination"]) == 1  # flat black region refused


def test_db_upgrades_from_phase1(tmp_path):
    p = tmp_path / "old.db"
    con = sqlite3.connect(p)
    con.executescript("""CREATE TABLE recordings (id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT NOT NULL UNIQUE,
        filename TEXT NOT NULL, library_path TEXT, source TEXT, status TEXT NOT NULL, size_bytes INTEGER,
        mtime REAL, fingerprint TEXT, duration REAL, width INTEGER, height INTEGER, fps REAL, video_codec TEXT,
        audio_codec TEXT, container TEXT, bit_rate INTEGER, duplicate_of INTEGER, attempts INTEGER NOT NULL DEFAULT 0,
        error TEXT, detected_at TEXT NOT NULL, imported_at TEXT, updated_at TEXT NOT NULL);
        INSERT INTO recordings(path, filename, status, detected_at, updated_at) VALUES('a.mp4','a.mp4','imported','t','t');
        PRAGMA user_version=1;""")
    con.close()
    db = Database(p)
    assert db.recordings_needing_analysis()[0]["filename"] == "a.mp4"
    assert db.list_recordings()[0]["highlight_count"] == 0
    db.close()


# ---- tuning fixes from first real-footage run -------------------------------
def test_template_self_match_is_exact(game_clip, settings):
    """Template pixels must come from the same scaler as analysis frames."""
    elim_template(game_clip, settings)
    tl = build_timeline(game_clip, probe(game_clip).as_dict(), settings, lambda f, s: None)
    series = next(iter(tl.template_scores.values()))
    i = int(np.argmin(np.abs(tl.times - 30.5)))
    assert series[i] > 0.99


def test_overlapping_templates_keep_only_best(game_clip, settings):
    """KNOCKED! and ELIMINATION! share a screen spot: one frame can't be both."""
    elim_template(game_clip, settings)
    make_template(game_clip, 30.5, (POPUP_BOX[0] + 10, POPUP_BOX[1], POPUP_BOX[2], POPUP_BOX[3]),
                  1280, 720, "knocked", settings.analysis_width, threshold=0.3)
    tl = build_timeline(game_clip, probe(game_clip).as_dict(), settings, lambda f, s: None)
    per_frame = {}
    for t, ev, name, sc in tl.template_hits:
        per_frame.setdefault(t, []).append(ev)
    assert per_frame and all(len(v) == 1 for v in per_frame.values())


def test_cli_template_tuning(game_clip, settings, capsys):
    from rogue_sniper.config import db_path, save_settings
    save_settings(settings)
    db = Database(db_path())
    rid = import_clip(db, settings, game_clip)
    db.close()
    elim_template(game_clip, settings)
    name = list_templates()[0]["name"]
    assert cli.main(["template-threshold", "elimination", name, "0.9"]) == 0
    assert list_templates()[0]["threshold"] == 0.9
    assert cli.main(["template-threshold", "elimination", name, "1.5"]) == 1
    assert cli.main(["analyze", str(rid), "--no-ai"]) == 0
    capsys.readouterr()
    assert cli.main(["template-scores", str(rid), "--top", "5"]) == 0
    out = capsys.readouterr().out
    assert "0:08" in out and "0:30" in out and "0:34" in out and "MATCH" in out
