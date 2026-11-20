"""Command line: python -m rogue_sniper <command>"""
from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import sys

from .config import app_home, db_path, ensure_workspace, load_settings, save_settings
from .db import Database


def setup_logging(verbose: bool = False) -> None:
    logs = app_home() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(logs / "rogue_sniper.log", maxBytes=5_000_000,
                                              backupCount=5, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.handlers[:] = [fh, sh]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rogue_sniper", description="Rogue Sniper AI")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("env-check", help="evaluate FFmpeg, GPU encoders, recording setup")
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-encode-test", action="store_true")

    p = sub.add_parser("init", help="create the ROGUE SNIPER workspace and settings")
    p.add_argument("--workspace", help="workspace root (default: Videos\\ROGUE SNIPER)")
    p.add_argument("--add-detected", action="store_true",
                   help="also monitor detected ShadowPlay/OBS recording folders")
    p.add_argument("--watch", action="append", default=[], help="extra folder to monitor (repeatable)")

    sub.add_parser("scan", help="scan once (files must already be stable) and exit")
    sub.add_parser("watch", help="monitor folders until Ctrl+C")
    p = sub.add_parser("list", help="list recordings")
    p.add_argument("--status")
    p = sub.add_parser("retry", help="retry a failed recording")
    p.add_argument("id", type=int)

    # ---- Phase 2 ----
    p = sub.add_parser("analyze", help="find highlights in an imported recording (runs now, in this window)")
    p.add_argument("id", type=int)
    p.add_argument("--ai", dest="ai", action="store_true", default=None, help="force Claude review on")
    p.add_argument("--no-ai", dest="ai", action="store_false", help="local detection only")
    p = sub.add_parser("highlights", help="list highlights")
    p.add_argument("id", type=int, nargs="?", help="recording id (default: all)")
    p.add_argument("--all", action="store_true", help="include below-threshold and rejected")
    p = sub.add_parser("highlight-edit", help="correct a highlight")
    p.add_argument("hid", type=int)
    p.add_argument("--status", choices=["approved", "rejected", "selected", "below_threshold"])
    p.add_argument("--type")
    p.add_argument("--start", help="m:ss or seconds")
    p.add_argument("--end", help="m:ss or seconds")
    p = sub.add_parser("highlight-add", help="add a highlight the detector missed")
    p.add_argument("id", type=int)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--type", required=True)
    p.add_argument("--description")
    p = sub.add_parser("snapshot", help="save a full-resolution frame as PNG (to pick a template region)")
    p.add_argument("id", type=int)
    p.add_argument("--time", required=True, help="m:ss or seconds")
    p.add_argument("--out", help="output .png (default: workspace PROCESSING folder)")
    p = sub.add_parser("make-template", help="teach the detector a HUD message from your footage")
    p.add_argument("id", type=int)
    p.add_argument("--time", required=True, help="m:ss or seconds where the message is on screen")
    p.add_argument("--region", required=True, help="x1,y1,x2,y2 in full-resolution pixels (from snapshot)")
    p.add_argument("--event", required=True, choices=["elimination", "knocked", "victory_royale", "custom"])
    p.add_argument("--name")
    p.add_argument("--threshold", type=float, default=0.75)
    sub.add_parser("templates", help="list HUD templates")
    p = sub.add_parser("template-scores", help="show the best-matching moments per template (for tuning)")
    p.add_argument("id", type=int)
    p.add_argument("--top", type=int, default=15)
    p = sub.add_parser("template-threshold", help="change a template's match threshold")
    p.add_argument("event")
    p.add_argument("name")
    p.add_argument("threshold", type=float)
    sub.add_parser("set-api-key", help="store your Anthropic API key in Windows Credential Manager")
    sub.add_parser("ai-status", help="show whether an API key is configured")

    p = sub.add_parser("serve", help="run API + background monitor")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)

    a = ap.parse_args(argv)
    setup_logging(a.verbose)
    settings = load_settings()

    if a.cmd == "env-check":
        from .env_check import format_report, run_env_check
        r = run_env_check(settings, test_encoders=not a.no_encode_test)
        print(json.dumps(r, indent=2) if a.json else format_report(r))
        return 0

    if a.cmd == "init":
        from pathlib import Path
        from .env_check import detected_recording_folders, run_env_check
        if a.workspace:
            settings.workspace_root = str(Path(a.workspace).expanduser())
        folders = list(settings.watch_folders) + [str(Path(w).expanduser()) for w in a.watch]
        r = run_env_check(settings, test_encoders=False)
        detected = detected_recording_folders(r)
        if a.add_detected:
            folders += detected
        settings.watch_folders = list(dict.fromkeys(f for f in folders if Path(f).is_dir()))
        save_settings(settings)
        dirs = ensure_workspace(settings)
        print(f"Workspace : {settings.workspace}")
        for name, d in dirs.items():
            print(f"  {name:<14} {d}")
        print("Monitoring:")
        for f in settings.monitored_folders():
            print(f"  {f}")
        if detected and not a.add_detected:
            print("Detected recording folders (add with --add-detected or --watch):")
            for f in detected:
                print(f"  {f}")
        print(f"Settings  : {app_home() / 'settings.json'}")
        return 0

    db = Database(db_path())
    from .service import MonitorService, run_forever

    if a.cmd == "scan":
        # Single pass with stability window disabled: only use on files you know are finished.
        settings_once = settings.model_copy(update={"stable_seconds": 1.0})
        svc = MonitorService(settings_once, db, lock_check=lambda p: False)
        import time
        rep = svc.scan_once()          # register files
        time.sleep(1.1)
        rep2 = svc.scan_once()         # import those unchanged for 1s
        print(f"seen {rep.seen}, new {rep.new}, imported {len(rep2.imported)}, other {rep2.other}, "
              f"still waiting {rep2.waiting}")
        return 0

    if a.cmd == "watch":
        svc = MonitorService(settings, db)
        print("Monitoring (Ctrl+C to stop):")
        for f in settings.monitored_folders():
            print(f"  {f}")
        run_forever(svc)
        return 0

    if a.cmd == "list":
        for r in db.list_recordings(status=a.status, limit=1000):
            dur = f"{r['duration']:.0f}s" if r["duration"] else "-"
            res = f"{r['width']}x{r['height']}@{r['fps']:g}" if r["width"] else "-"
            print(f"#{r['id']:<4} {r['status']:<9} {dur:>6} {res:<16} {r['source']:<10} {r['filename']}"
                  + (f"  [{r['error']}]" if r["error"] else ""))
        return 0

    if a.cmd == "retry":
        row = db.get_recording(a.id)
        if not row:
            print("no such recording", file=sys.stderr)
            return 1
        db.update_recording(a.id, status="retry", attempts=0, error=None)
        print(f"#{a.id} queued for retry (picked up by the running monitor)")
        return 0

    if a.cmd == "analyze":
        from .analysis.pipeline import analyze_recording
        from .analysis.detect import fmt_t

        def show(frac, stage):
            print(f"\r[{frac * 100:5.1f}%] {stage:<40}", end="", flush=True)
        try:
            summary = analyze_recording(settings, db, a.id, show, use_ai=a.ai)
        except (ValueError, FileNotFoundError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print()
        ai = summary["ai"]
        print(f"Done in {summary['seconds']}s: {summary['events']} events, {summary['selected']} highlights selected"
              f" ({summary['highlights'] - summary['selected']} below threshold). HUD templates: "
              f"{summary['templates_loaded']}. AI: "
              + (f"reviewed {ai.get('reviewed', 0)}" if ai.get("enabled") else ai.get("reason", "off")))
        _print_highlights(db.list_highlights(a.id), show_all=True)
        print(f"Signal dump for tuning: {app_home() / 'analysis' / f'recording_{a.id}.json'}")
        return 0

    if a.cmd == "highlights":
        rows = db.list_highlights(a.id)
        _print_highlights(rows, show_all=a.all)
        return 0

    if a.cmd == "highlight-edit":
        h = db.get_highlight(a.hid)
        if not h:
            print("no such highlight", file=sys.stderr)
            return 1
        fields = {}
        if a.status:
            fields["status"] = a.status
        if a.type:
            fields["type"] = a.type
            fields["types"] = list(dict.fromkeys([a.type] + h["types"]))
        if a.start:
            fields["start"] = parse_time(a.start)
        if a.end:
            fields["end"] = parse_time(a.end)
        if fields.get("end", h["end"]) <= fields.get("start", h["start"]):
            print("end must be after start", file=sys.stderr)
            return 1
        db.update_highlight(a.hid, user_edited=1, **fields)
        _print_highlights([db.get_highlight(a.hid)], show_all=True)
        return 0

    if a.cmd == "highlight-add":
        start, end = parse_time(a.start), parse_time(a.end)
        if end <= start or not db.get_recording(a.id):
            print("invalid recording or time range", file=sys.stderr)
            return 1
        hid = db.insert_highlight(a.id, {"start": start, "end": end, "peak": (start + end) / 2, "type": a.type,
                                         "confidence": 1.0, "description": a.description or "Added manually",
                                         "source": "manual", "status": "approved", "user_edited": 1})
        _print_highlights([db.get_highlight(hid)], show_all=True)
        return 0

    if a.cmd in ("snapshot", "make-template"):
        from pathlib import Path
        rec = db.get_recording(a.id)
        if not rec or rec["status"] != "imported":
            print("recording not found or not imported", file=sys.stderr)
            return 1
        src = rec["library_path"] or rec["path"]
        t = parse_time(a.time)
        if a.cmd == "snapshot":
            import cv2
            from .analysis.decode import extract_frame
            from .config import PROCESSING_DIR, unique_path
            img = extract_frame(src, t, ffmpeg=settings.ffmpeg_path)
            out = Path(a.out) if a.out else unique_path(
                settings.dir(PROCESSING_DIR) / f"snapshot rec{a.id} {int(t // 60)}m{int(t % 60):02d}s.png")
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), img)
            print(f"Saved {img.shape[1]}x{img.shape[0]} frame: {out}")
            print("Open it in Paint, hover the corners of the message to read x,y (bottom-left of Paint).")
            return 0
        from .analysis.templates import make_template
        try:
            x1, y1, x2, y2 = (int(v) for v in a.region.replace(" ", "").split(","))
            out = make_template(src, t, (x1, y1, x2, y2), rec["width"], rec["height"], a.event,
                                settings.analysis_width, a.name, a.threshold, settings.ffmpeg_path)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(f"Template saved: {out}\nRe-run `analyze` on your recordings to use it.")
        return 0

    if a.cmd == "templates":
        from .analysis.templates import list_templates
        for t in list_templates():
            print(f"{t['event']:<15} {t['name']:<20} threshold {t['threshold']}  size {t['size']}")
        return 0

    if a.cmd == "template-scores":
        import json as _json
        from .analysis.detect import fmt_t
        from .analysis.templates import list_templates
        dump = app_home() / "analysis" / f"recording_{a.id}.json"
        if not dump.exists():
            print(f"run `analyze {a.id}` first", file=sys.stderr)
            return 1
        scores = _json.loads(dump.read_text(encoding="utf-8")).get("template_scores", {})
        thr = {f"{t['event']}/{t['name']}": t["threshold"] for t in list_templates()}
        if not scores:
            print("no template scores in the last analysis - re-run analyze")
        for key, series in scores.items():
            print(f"\n{key}  (threshold {thr.get(key, '?')})")
            peaks, used = [], []
            for t, v in sorted(series, key=lambda x: -x[1]):
                if all(abs(t - u) > 3 for u in used):
                    used.append(t)
                    peaks.append((t, v))
                if len(peaks) >= a.top:
                    break
            for t, v in sorted(peaks):
                mark = "MATCH" if v >= thr.get(key, 1) else ""
                print(f"  {fmt_t(t):>6}  {v:.2f}  {mark}")
        return 0

    if a.cmd == "template-threshold":
        from .analysis.templates import set_threshold
        try:
            ok = set_threshold(a.event, a.name, a.threshold)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print("updated" if ok else "template not found (see `templates`)")
        return 0 if ok else 1

    if a.cmd == "set-api-key":
        import getpass
        from .credentials import SecretStoreError, set_api_key
        key = getpass.getpass("Paste your Anthropic API key (input hidden): ")
        try:
            set_api_key(key)
        except SecretStoreError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        settings.ai_enabled = True
        save_settings(settings)
        print("Saved to Windows Credential Manager. AI review is now enabled.")
        return 0

    if a.cmd == "ai-status":
        from .credentials import key_status
        st = key_status()
        print(f"API key: {'configured (' + st['source'] + ')' if st['configured'] else 'not configured'}; "
              f"AI review {'enabled' if settings.ai_enabled else 'disabled'}; model {settings.ai_model}")
        return 0

    if a.cmd == "serve":
        import uvicorn
        from .api import create_app
        app = create_app(settings, db)
        print(f"Rogue Sniper AI running at http://{a.host}:{a.port}")
        uvicorn.run(app, host=a.host, port=a.port, log_level="info")
        return 0
    return 1


def parse_time(v: str) -> float:
    """'200', '3:20', '1:02:03', '3:20.5' -> seconds."""
    parts = [float(p) for p in str(v).strip().split(":")]
    t = 0.0
    for p in parts:
        t = t * 60 + p
    return t


def _print_highlights(rows: list[dict], show_all: bool = False) -> None:
    from .analysis.detect import fmt_t
    shown = [h for h in rows if show_all or h["status"] in ("selected", "approved")]
    if not shown:
        print("(no highlights)")
        return
    for h in shown:
        ai = f" AI {h['ai_confidence']:.2f}" if h.get("ai_confidence") is not None else ""
        print(f"#{h['id']:<4} rec {h['recording_id']:<3} {fmt_t(h['start'])}-{fmt_t(h['end'])} "
              f"{h['type']:<19} conf {h['confidence']:.2f}{ai}  [{h['status']}]  "
              f"{h.get('ai_description') or h.get('description') or ''}")
