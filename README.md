# Rogue Sniper AI

An automated Fortnite content studio that runs locally on a Windows gaming PC.
You play and record. It finds the best moments and edits them into vertical
1080x1920 videos with captions, then saves them to `READY TO POST`. You review
them and upload to TikTok yourself.

## Status

| Phase | Scope | Status |
|---|---|---|
| 1 | Environment check, folder monitoring, safe import, SQLite tracking, API | **Done and tested**, verified on Windows with a real NVIDIA App recording |
| 2 | Highlight detection (audio + OpenCV + HUD templates + optional Claude review), confidence, manual correction, background jobs | **Done and tested on synthetic footage** (55 tests total). Needs tuning on your real recordings |
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

## Highlight detection (Phase 2)

Every imported recording is analysed automatically in the background. You can watch progress on the status page.

| Signal | How it's detected | Reliability |
|---|---|---|
| **Heavy combat** | Clusters of sharp sounds (gunshots, hits) per 5 s, plus on-screen motion | Generic: the number of shots that counts as combat needs tuning on your audio |
| **Sniper shot** | Sniper-scope view (black surround with a round lens) followed by a shot sound | Good on synthetic footage; **check it on your snipes** |
| **Elimination / knock / Victory Royale** | **HUD templates**: a crop of that on-screen message taken from *your own* footage | Very reliable once a template exists. Without templates, none are claimed |
| **Multi elimination** | 2 or more eliminations within 20 s | Same as eliminations |
| **Clutch / funny fail / unusual** | Only the optional Claude review can label these | Needs an API key |

Each highlight gets a **confidence** (0-1) and a status:
- `selected`: above the threshold for your sensitivity setting
- `below_threshold`: kept, so you can promote it
- `approved` / `rejected`: your manual decision

Re-running analysis never overwrites what you edited or added by hand.

### Teach it your elimination pop-up (recommended, takes 2 minutes)
1. Find a moment in a recording where `ELIMINATED <name>` is on screen, for example at 3:20 in recording #1.
2. Save that frame: `python -m rogue_sniper snapshot 1 --time 3:20`
3. Open the PNG in **Paint**. Hover over the top-left and bottom-right corners of the message; Paint shows the x, y position in its bottom-left corner.
4. Create the template: `python -m rogue_sniper make-template 1 --time 3:20 --region 820,700,1100,760 --event elimination`
5. Re-run analysis: `python -m rogue_sniper analyze 1`

You can do the same with `--event victory_royale` and `--event knocked`.

### Optional Claude review
`python -m rogue_sniper set-api-key` stores your Anthropic API key in **Windows Credential Manager** and turns on the review. The key is never written to disk in plain text and never returned by the API.
- **What's sent:** for the top candidates (8 by default), a few downscaled frames each plus the local signals. Your full recording is never uploaded.
- **What it does:** Claude labels clutch moments and funny fails, writes descriptions, and adjusts confidence. A clear "not a highlight" pushes a clip below the threshold.
- **Model:** `claude-opus-5-5` at low effort, with automatic refusal fallback enabled.
- **Cost:** roughly 6 images x 8 clips per recording, about $0.10 per recording at current prices.
- **Without a key:** everything else still works.

### Commands
```
python -m rogue_sniper analyze <id> [--no-ai]       # run now, prints highlights
python -m rogue_sniper highlights [<id>] [--all]
python -m rogue_sniper highlight-edit <hid> --status rejected | --type clutch | --start 3:10 --end 3:30
python -m rogue_sniper highlight-add <id> --start 5:02 --end 5:15 --type funny_failure
python -m rogue_sniper snapshot <id> --time 3:20
python -m rogue_sniper make-template <id> --time 3:20 --region x1,y1,x2,y2 --event elimination
python -m rogue_sniper set-api-key | ai-status
```
Settings such as `sensitivity` (low/medium/high), `type_thresholds`, `max_highlights_per_recording`, `analysis_fps` and `ai_max_candidates` live in `settings.json`.

After every analysis, a per-second signal dump is saved to `%LOCALAPPDATA%\RogueSniperAI\analysis\recording_<id>.json`, for tuning.

## API

`GET /api/health` · `GET /api/environment` · `GET /api/stats` · `GET /api/recordings` ·
`GET /api/recordings/{id}` · `POST /api/recordings/{id}/retry` · `POST /api/scan` ·
`GET /api/events` · `GET|PUT /api/settings` ·
`POST /api/recordings/{id}/analyze` · `GET /api/jobs` · `POST /api/jobs/{id}/cancel` ·
`GET /api/recordings/{id}/highlights` · `GET /api/highlights` · `PATCH|DELETE /api/highlights/{id}` ·
`POST /api/recordings/{id}/highlights` · `GET /api/templates` · `GET /api/ai/status` · `PUT|DELETE /api/ai/key`

The server listens only on `127.0.0.1`, so it can't be reached from other machines.
