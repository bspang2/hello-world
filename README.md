# Rogue Sniper AI

An automated Fortnite content studio that runs locally on a Windows gaming PC.
You play and record. It finds the best moments and edits them into vertical
1080x1920 videos with captions, then saves them to `READY TO POST`. You review
them and upload to TikTok yourself.

## Status

| Phase | Scope | Status |
|---|---|---|
| 1 | Environment check, folder monitoring, safe import, SQLite tracking, API | **Done and tested** (26 automated tests) |
| 2 | Highlight detection (audio + OpenCV + optional multimodal AI), confidence, manual correction | Not started |
| 3 | FFmpeg rendering: 9:16, zooms, hooks, text, loudness, 3 presets, NVENC/AMF/QSV/x264 | Not started |
| 4 | Hooks, titles, captions, hashtags, cover, posting window | Not started |
| 5 | React dashboard + per-video export folders | Not started (Phase 1 ships a basic status page) |
| 6 | End-to-end tests and optimization on real footage | Not started |

## Install (Windows 10/11)

1. Install Python 3.11+ and FFmpeg. In PowerShell:
   ```powershell
   winget install Python.Python.3.12
   winget install Gyan.FFmpeg
   ```
   Close and reopen the terminal afterwards so `ffmpeg` is on PATH.
2. Get the code: `git clone` this repo, or download the ZIP.
3. Double-click **`scripts\windows\install.bat`**. It:
   - creates `.venv` and installs the dependencies
   - creates `Videos\ROGUE SNIPER\` with `RAW GAMEPLAY`, `PROCESSING`, `READY TO POST`, `ARCHIVE`, `FAILED`
   - finds your ShadowPlay (`Videos\Fortnite`) and OBS recording folders and adds them to the monitor
   - runs the environment check, which test-encodes with every GPU encoder
4. Double-click **`scripts\windows\start.bat`**. It starts the monitor and the API and opens http://127.0.0.1:8765.

## Commands

```
python -m rogue_sniper env-check [--json]     # FFmpeg, GPU encoders (really tested), GPU, OBS/ShadowPlay setup
python -m rogue_sniper init [--workspace D:\ROGUE SNIPER] [--add-detected] [--watch "D:\Clips"]
python -m rogue_sniper serve                  # API + background monitor (what start.bat runs)
python -m rogue_sniper watch                  # monitor only, in the console
python -m rogue_sniper scan                   # one pass, for files you know are finished
python -m rogue_sniper list [--status failed]
python -m rogue_sniper retry <id>
python -m pytest                              # test suite (needs ffmpeg on PATH)
```

Settings, the database and logs are stored in `%LOCALAPPDATA%\RogueSniperAI\`
(`settings.json`, `rogue_sniper.db`, `logs\`). To use a different location, set `ROGUE_SNIPER_HOME`.

## How Phase 1 detects a finished recording

A recording is imported only when all of these are true:

1. **Size and modified time haven't changed for `stable_seconds`** (default 15 s). The folder is checked every 5 s; checking a file's size is cheap even for 20 GB recordings.
2. **No other process has the file open.** On Windows, the app tries to open the file exclusively, so OBS or ShadowPlay still writing it blocks the import.
3. **FFprobe can read it.** An MP4 that is still recording has no index yet and fails this check. Failures are retried with backoff, up to `max_import_attempts`. After that, the recording is marked `failed` and a report is written to `FAILED\`.

**Preventing duplicates**
- A file that's already imported is never processed again, and this survives restarts because it's stored in SQLite.
- Identical files saved under a different name or folder are detected by content: a fingerprint built from the size plus the start, middle and end of the file.
- When OBS auto-remuxes `X.mkv` into `X.mp4`, only the MP4 is kept.
- The output folders (`PROCESSING`, `READY TO POST`, `ARCHIVE`, `FAILED`) are never scanned, even if you monitor `Videos` as a whole.
- A file that changes on disk is imported again.

**Originals are read-only.** The app never moves, edits or deletes a recording. With `"import_mode": "copy"`, it also copies the file into `RAW GAMEPLAY` (via a `.partial` temp file and a size check) and leaves the original where it was.

## Recording tips

- **OBS:** record to MKV, Fragmented MP4 or Hybrid MP4 (OBS 30.2+). If OBS crashes, a plain MP4 is lost completely. The env-check warns you about this.
- **ShadowPlay / NVIDIA App:** by default it saves to `Videos\Fortnite`. Both Instant Replay (`.DVR.mp4`) and full recordings are picked up.
- Record at 1080p60 or higher. Phase 3 crops to 9:16, so higher source resolution means sharper vertical videos.

## API (Phase 1)

`GET /api/health` · `GET /api/environment` · `GET /api/stats` · `GET /api/recordings` ·
`GET /api/recordings/{id}` · `POST /api/recordings/{id}/retry` · `POST /api/scan` ·
`GET /api/events` · `GET|PUT /api/settings`

The server listens only on `127.0.0.1`, so it can't be reached from other machines.
