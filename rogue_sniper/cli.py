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

    if a.cmd == "serve":
        import uvicorn
        from .api import create_app
        app = create_app(settings, db)
        print(f"Rogue Sniper AI running at http://{a.host}:{a.port}")
        uvicorn.run(app, host=a.host, port=a.port, log_level="info")
        return 0
    return 1
