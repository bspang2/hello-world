"""FastAPI backend. Phase 1 endpoints: health, environment, stats, recordings, events, settings."""
from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ValidationError

from . import __version__
from .config import Settings, save_settings
from .db import Database
from .env_check import run_env_check
from .service import MonitorService


def create_app(settings: Settings, db: Database, start_monitor: bool = True,
               service: MonitorService | None = None) -> FastAPI:
    svc = service or MonitorService(settings, db)
    env_cache: dict[str, Any] = {}
    env_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if start_monitor:
            svc.start()
        yield
        svc.stop()

    app = FastAPI(title="Rogue Sniper AI", version=__version__, lifespan=lifespan)
    app.state.service = svc
    # The React dashboard (Phase 5) runs on the Vite dev server during development.
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["*"], allow_headers=["*"])

    @app.get("/api/health")
    def health():
        return {"ok": True, "version": __version__, "monitor": svc.status()}

    @app.get("/api/environment")
    def environment(refresh: bool = False):
        # Encoder tests take a few seconds; cache the result. FastAPI runs sync handlers in a threadpool.
        with env_lock:
            if refresh or "report" not in env_cache:
                env_cache["report"] = run_env_check(svc.settings)
            return env_cache["report"]

    @app.get("/api/stats")
    def stats():
        c = db.status_counts()
        return {
            "recordings_detected": sum(c.values()),
            "recordings_imported": c.get("imported", 0),
            "recordings_waiting": c.get("waiting", 0) + c.get("retry", 0),
            "duplicates_skipped": c.get("duplicate", 0),
            "ignored": c.get("ignored", 0),
            "processing_failures": c.get("failed", 0),
            "highlights_identified": db.highlight_count(),
            "highlights_below_threshold": db.highlight_count(("below_threshold",)),
            # Filled in by later phases:
            "videos_generated": 0,
            "videos_ready_to_post": 0,
            "by_status": c,
        }

    @app.get("/api/recordings")
    def recordings(status: str | None = None, limit: int = 200, offset: int = 0):
        return db.list_recordings(status=status, limit=min(limit, 1000), offset=offset)

    @app.get("/api/recordings/{rec_id}")
    def recording(rec_id: int):
        r = db.get_recording(rec_id)
        if not r:
            raise HTTPException(404, "recording not found")
        return r

    @app.post("/api/recordings/{rec_id}/retry")
    def retry(rec_id: int):
        if not svc.watcher.force_retry(rec_id):
            raise HTTPException(404, "recording not found")
        svc.request_scan()
        return {"ok": True}

    @app.post("/api/scan")
    def scan():
        rep = svc.scan_once()
        return {"seen": rep.seen, "new": rep.new, "waiting": rep.waiting,
                "imported": rep.imported, "other": rep.other}

    @app.get("/api/events")
    def events(limit: int = 100):
        return db.list_events(min(limit, 1000))

    @app.get("/api/settings")
    def get_settings():
        return svc.settings.model_dump()

    @app.put("/api/settings")
    def put_settings(patch: dict[str, Any]):
        merged = {**svc.settings.model_dump(), **patch}
        try:
            new = Settings.model_validate(merged)
        except ValidationError as e:
            raise HTTPException(422, e.errors(include_url=False)) from e
        bad = [p for p in new.watch_folders if p.strip() and not Path(p).is_dir()]
        if bad:
            raise HTTPException(422, f"folder does not exist: {bad[0]}")
        save_settings(new)
        svc.apply_settings(new)
        env_cache.clear()
        db.add_event("Settings updated")
        return new.model_dump()

    # ---- Phase 2: analysis jobs -------------------------------------
    @app.post("/api/recordings/{rec_id}/analyze")
    def analyze(rec_id: int):
        r = db.get_recording(rec_id)
        if not r:
            raise HTTPException(404, "recording not found")
        if r["status"] != "imported":
            raise HTTPException(409, f"recording is '{r['status']}', not imported")
        return {"job_id": svc.jobs.enqueue(rec_id)}

    @app.get("/api/jobs")
    def jobs(limit: int = 50, status: str | None = None):
        return db.list_jobs(min(limit, 500), status)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: int):
        if not svc.jobs.cancel(job_id):
            raise HTTPException(409, "job is not queued or running")
        return {"ok": True}

    # ---- Phase 2: highlights + manual correction --------------------
    @app.get("/api/recordings/{rec_id}/highlights")
    def rec_highlights(rec_id: int, status: str | None = None):
        if not db.get_recording(rec_id):
            raise HTTPException(404, "recording not found")
        return db.list_highlights(rec_id, status)

    @app.get("/api/highlights")
    def all_highlights(status: str | None = None):
        return db.list_highlights(None, status)

    @app.patch("/api/highlights/{hl_id}")
    def edit_highlight(hl_id: int, patch: HighlightPatch):
        h = db.get_highlight(hl_id)
        if not h:
            raise HTTPException(404, "highlight not found")
        fields = patch.model_dump(exclude_none=True)
        start, end = fields.get("start", h["start"]), fields.get("end", h["end"])
        _check_range(db, h["recording_id"], start, end)
        if "type" in fields:
            fields["types"] = list(dict.fromkeys([fields["type"]] + h["types"]))
        db.update_highlight(hl_id, user_edited=1, **fields)
        return db.get_highlight(hl_id)

    @app.post("/api/recordings/{rec_id}/highlights")
    def add_highlight(rec_id: int, body: HighlightCreate):
        _check_range(db, rec_id, body.start, body.end)
        hid = db.insert_highlight(rec_id, {
            "start": body.start, "end": body.end, "peak": (body.start + body.end) / 2, "type": body.type,
            "types": [body.type], "confidence": 1.0, "description": body.description or "Added manually",
            "source": "manual", "status": "approved", "user_edited": 1})
        return db.get_highlight(hid)

    @app.delete("/api/highlights/{hl_id}")
    def delete_highlight(hl_id: int):
        if not db.get_highlight(hl_id):
            raise HTTPException(404, "highlight not found")
        db.delete_highlight(hl_id)
        return {"ok": True}

    # ---- Phase 2: HUD templates + AI key ----------------------------
    @app.get("/api/templates")
    def templates():
        from .analysis.templates import list_templates
        return list_templates()

    @app.delete("/api/templates/{event}/{name}")
    def del_template(event: str, name: str):
        from .analysis.templates import delete_template
        if not delete_template(event, name):
            raise HTTPException(404, "template not found")
        return {"ok": True}

    @app.get("/api/ai/status")
    def ai_status():
        from .credentials import key_status
        return {**key_status(), "enabled": svc.settings.ai_enabled, "model": svc.settings.ai_model}

    @app.put("/api/ai/key")
    def ai_set_key(body: ApiKey):
        from .credentials import SecretStoreError, key_status, set_api_key
        try:
            set_api_key(body.api_key)
        except SecretStoreError as e:
            raise HTTPException(400, str(e)) from e
        return key_status()

    @app.delete("/api/ai/key")
    def ai_del_key():
        from .credentials import delete_api_key, key_status
        delete_api_key()
        return key_status()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _STATUS_PAGE

    return app


HL_TYPES = r"^(elimination|multi_elimination|sniper_shot|victory_royale|clutch|funny_failure|unusual|high_action_combat|knocked)$"


class HighlightPatch(BaseModel):
    status: str | None = Field(default=None, pattern="^(selected|approved|rejected|below_threshold)$")
    type: str | None = Field(default=None, pattern=HL_TYPES)
    start: float | None = Field(default=None, ge=0)
    end: float | None = Field(default=None, gt=0)
    description: str | None = Field(default=None, max_length=500)


class HighlightCreate(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    type: str = Field(pattern=HL_TYPES)
    description: str | None = Field(default=None, max_length=500)


class ApiKey(BaseModel):
    api_key: str = Field(min_length=10, max_length=500)


def _check_range(db: Database, rec_id: int, start: float, end: float) -> None:
    r = db.get_recording(rec_id)
    if not r:
        raise HTTPException(404, "recording not found")
    if end <= start:
        raise HTTPException(422, "end must be after start")
    if r["duration"] and end > r["duration"] + 0.5:
        raise HTTPException(422, f"end is past the end of the recording ({r['duration']:.1f}s)")


# Minimal status page until the React dashboard lands in Phase 5.
_STATUS_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Rogue Sniper AI</title>
<style>body{background:#0b0f0c;color:#e6efe8;font:14px system-ui;margin:24px}h1{color:#39ff14}
.c{display:inline-block;background:#141a16;border:1px solid #1f2b22;border-radius:8px;padding:12px 16px;margin:4px}
.c b{display:block;font-size:24px;color:#39ff14}table{border-collapse:collapse;margin-top:16px;width:100%}
td,th{border-bottom:1px solid #1f2b22;padding:6px;text-align:left}button{background:#39ff14;border:0;padding:8px 14px;
border-radius:6px;font-weight:600;cursor:pointer}</style></head><body>
<h1>Rogue Sniper AI</h1><p>Status page. The full dashboard arrives in Phase 5.</p>
<div id="s"></div><p><button onclick="fetch('/api/scan',{method:'POST'}).then(load)">Scan now</button></p>
<table><thead><tr><th>#</th><th>File</th><th>Status</th><th>Source</th><th>Duration</th><th>Res / FPS</th><th>Highlights</th></tr></thead>
<tbody id="r"></tbody></table>
<script>
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function load(){const s=await (await fetch('/api/stats')).json();
document.getElementById('s').innerHTML=[['Detected',s.recordings_detected],['Imported',s.recordings_imported],
['Waiting',s.recordings_waiting],['Highlights',s.highlights_identified],['Duplicates',s.duplicates_skipped],['Failures',s.processing_failures]]
.map(([k,v])=>`<div class=c>${k}<b>${v}</b></div>`).join('');
const r=await (await fetch('/api/recordings')).json();
document.getElementById('r').innerHTML=r.map(x=>`<tr><td>${x.id}</td><td>${esc(x.filename)}</td><td>${esc(x.status)}${x.error?' - '+esc(x.error):''}</td>
<td>${esc(x.source)}</td><td>${x.duration?Math.round(x.duration)+'s':''}</td><td>${x.width?x.width+'x'+x.height+' @ '+x.fps:''}</td>
<td>${x.analysis_progress!=null?'analyzing '+Math.round(x.analysis_progress*100)+'%':(x.analysis_status==='done'?x.highlight_count:esc(x.analysis_status||''))}</td></tr>`).join('');}
load();setInterval(load,3000);
</script></body></html>"""
