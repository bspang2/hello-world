"""Phase 1: API, background service, environment check, CLI."""
import json
import time
from pathlib import Path

from conftest import make_clip
from fastapi.testclient import TestClient

from rogue_sniper import cli
from rogue_sniper.api import create_app
from rogue_sniper.config import RAW_DIR, WORKSPACE_DIRS, Settings, load_settings, save_settings
from rogue_sniper.env_check import detected_recording_folders, find_obs_profiles, format_report, run_env_check
from rogue_sniper.service import MonitorService


def test_api_endpoints(settings, db, clock, tmp_path):
    svc = MonitorService(settings, db, clock=clock, lock_check=lambda p: False)
    app = create_app(settings, db, start_monitor=False, service=svc)
    make_clip(settings.dir(RAW_DIR) / "a.mkv")
    with TestClient(app) as c:
        assert c.get("/api/health").json()["ok"]
        assert c.post("/api/scan").json()["new"] == 1
        clock.advance(11)
        assert len(c.post("/api/scan").json()["imported"]) == 1
        stats = c.get("/api/stats").json()
        assert stats["recordings_detected"] == 1 and stats["recordings_imported"] == 1
        recs = c.get("/api/recordings").json()
        assert recs[0]["filename"] == "a.mkv" and recs[0]["duration"] > 2
        assert c.get(f"/api/recordings/{recs[0]['id']}").status_code == 200
        assert c.get("/api/recordings/999").status_code == 404
        assert any("Imported" in e["message"] for e in c.get("/api/events").json())
        assert "Rogue Sniper" in c.get("/").text


def test_api_settings_validation_and_persist(settings, db):
    app = create_app(settings, db, start_monitor=False)
    extra = Path(settings.workspace_root).parent / "ShadowPlay"
    extra.mkdir()
    with TestClient(app) as c:
        assert c.put("/api/settings", json={"stable_seconds": 0}).status_code == 422
        assert c.put("/api/settings", json={"watch_folders": ["/no/such/dir"]}).status_code == 422
        r = c.put("/api/settings", json={"watch_folders": [str(extra)], "stable_seconds": 20})
        assert r.status_code == 200
        assert str(extra) in c.get("/api/health").json()["monitor"]["monitored_folders"]
    assert load_settings().stable_seconds == 20


def test_background_service_imports_without_blocking(settings, db):
    settings.stable_seconds = 1
    settings.poll_interval_seconds = 0.5
    svc = MonitorService(settings, db, lock_check=lambda p: False)
    svc.start()
    try:
        make_clip(settings.dir(RAW_DIR) / "bg.mp4")
        deadline = time.time() + 20
        while time.time() < deadline and db.status_counts().get("imported", 0) == 0:
            time.sleep(0.2)
        assert db.status_counts().get("imported") == 1
        assert svc.status()["running"]
    finally:
        svc.stop()
    assert not svc.running


def test_settings_roundtrip(tmp_path):
    s = Settings(workspace_root=str(tmp_path / "ws"), extensions=["MP4", "mkv", ""])
    assert s.extensions == [".mp4", ".mkv"]
    save_settings(s)
    assert load_settings() == s


def test_obs_profile_parsing(tmp_path):
    appdata = tmp_path / "AppData" / "Roaming"
    prof = appdata / "obs-studio" / "basic" / "profiles" / "Fortnite"
    prof.mkdir(parents=True)
    (prof / "basic.ini").write_text(
        "\ufeff[General]\nName=Fortnite\n[Output]\nMode=Advanced\n"
        "[AdvOut]\nRecFilePath=D:\\Clips\\OBS\nRecFormat2=mp4\n", encoding="utf-8")
    simple = appdata / "obs-studio" / "basic" / "profiles" / "Untitled"
    simple.mkdir(parents=True)
    (simple / "basic.ini").write_text("[SimpleOutput]\nFilePath=C:\\Users\\me\\Videos\nRecFormat2=mkv\n")
    profs = {p["profile"]: p for p in find_obs_profiles(appdata)}
    assert profs["Fortnite"]["folder"] == "D:\\Clips\\OBS" and profs["Fortnite"]["format"] == "mp4"
    assert profs["Untitled"]["mode"] == "Simple" and profs["Untitled"]["format"] == "mkv"


def test_env_check_report(tmp_path, settings):
    home = tmp_path / "home"
    make_clip(home / "Videos" / "Fortnite" / "Fortnite 2026.11.19 - 20.00.00.01.mp4")
    appdata = tmp_path / "appdata"
    prof = appdata / "obs-studio" / "basic" / "profiles" / "P"
    prof.mkdir(parents=True)
    (prof / "basic.ini").write_text("[SimpleOutput]\nFilePath=X:\\obs\nRecFormat2=mp4\n")
    r = run_env_check(settings, home=home, appdata=appdata)
    assert r["ffmpeg"]["ffprobe_path"]
    assert r["ffmpeg"]["encoders"]["libx264"]["works"] is True
    assert r["recommended_encoder"] is not None
    sp = [f for f in r["recording"]["shadowplay_or_capture_folders"] if f["likely"] == "shadowplay"]
    assert sp and sp[0]["videos"] == 1
    assert any("plain MP4" in x for x in r["recommendations"])
    assert set(detected_recording_folders(r)) == {str(home / "Videos" / "Fortnite"), "X:\\obs"}
    text = format_report(r)
    assert "encoder libx264" in text and "WORKS" in text


def test_cli_init_scan_list(tmp_path, capsys):
    ws = tmp_path / "My ROGUE SNIPER"
    assert cli.main(["init", "--workspace", str(ws)]) == 0
    for d in WORKSPACE_DIRS:
        assert (ws / d).is_dir()
    make_clip(ws / RAW_DIR / "Fortnite 2026.11.19 - 20.15.33.02.mp4", seconds=12)
    capsys.readouterr()
    assert cli.main(["scan"]) == 0
    assert "imported 1" in capsys.readouterr().out
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "imported" in out and "shadowplay" in out
    assert cli.main(["env-check", "--json", "--no-encode-test"]) == 0
    assert "ffmpeg" in json.loads(capsys.readouterr().out)


def test_detected_folders_prefer_game_subfolder(tmp_path, settings):
    home = tmp_path / "home"
    (home / "Videos" / "NVIDIA" / "Fortnite").mkdir(parents=True)
    (home / "Videos" / "NVIDIA" / "Minecraft").mkdir(parents=True)
    r = run_env_check(settings, home=home, appdata=tmp_path / "none", test_encoders=False)
    assert detected_recording_folders(r) == [str(home / "Videos" / "NVIDIA" / "Fortnite")]
