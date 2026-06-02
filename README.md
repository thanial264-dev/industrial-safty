# SafeGuard AI — Industrial Accident Detector

A Flask-based web application that uses Claude's vision capabilities to analyze industrial video footage for accidents and safety hazards, with real-time hardware integration via ESP8266 to automatically shut down machinery when a threat is detected.

---

## Features

- **AI-powered video analysis** — Sends sampled video frames to Claude (claude-opus-4-5) for industrial accident detection: falls, fire, explosions, equipment failure, chemical spills, structural collapse, and PPE violations.
- **Reference video library** — Upload known-accident videos to build a visual profile. Subsequent footage is first compared against these references using OpenCV histogram matching before falling back to the Claude API, saving API calls.
- **ESP8266 relay control** — On accident detection the server fires an HTTP GET to an ESP8266 microcontroller, cutting power to connected machinery. The signal runs in a background thread so it never blocks analysis.
- **Manual re-analysis** — Operators can request a fresh Claude analysis on any specific timestamp from the web UI.
- **Real-time progress streaming** — The frontend polls `/api/progress/<job_id>` and updates a progress bar and incident log as frames are processed.
- **Severity classification** — Each detected event is classified as `low`, `medium`, `high`, or `critical`. The highest severity across all frames is surfaced in the summary.
- **Built-in web UI** — The full frontend (HTML/CSS/JS) is embedded in the Python file and served from the root route, so no separate static file server is needed.

---

## Architecture

```
Browser (embedded HTML/JS)
        │  upload video + accident ranges
        ▼
Flask app (app-16.py, port 5000)
        │
        ├─ OpenCV — frame extraction & histogram comparison
        │
        ├─ Anthropic API — Claude vision analysis per frame
        │
        └─ ESP8266 HTTP relay — machine shutoff signal
```

### Analysis pipeline (per frame)

1. Extract frame at ~2 fps (up to 60 frames per video).
2. If reference profiles exist → compare via HSV color histogram (Pearson correlation, threshold 0.80).
3. If no reference match → send frame to Claude as base64 image with a structured safety prompt.
4. If accident confidence ≥ 0.50 → trigger ESP8266 relay OFF in background thread.

---

## Requirements

- Python 3.8+
- An [Anthropic API key](https://console.anthropic.com/)
- An ESP8266 running a simple HTTP server (optional — app works without it)

### Python dependencies

```
flask
flask-cors
opencv-python
numpy
anthropic
```

Install with:

```bash
pip install flask flask-cors opencv-python numpy anthropic
```

---

## Configuration

### API key

Set your Anthropic API key as an environment variable before starting the server:

```bash
# macOS / Linux
export ANTHROPIC_API_KEY=sk-ant-...

# Windows (Command Prompt)
set ANTHROPIC_API_KEY=sk-ant-...
```

The app will warn on startup if the key is missing and all analysis will fail without it.

### ESP8266 relay (optional)

Edit the constants near **line 24** of `app-16.py`:

| Constant | Default | Description |
|---|---|---|
| `ESP8266_IP` | `10.146.170.117` | IP shown on the ESP8266 Serial Monitor |
| `ESP8266_PORT` | `80` | Must match the port in your Arduino sketch |
| `ESP8266_PATH` | `/accident_on` | Route that sets the relay HIGH (machine OFF) |
| `ESP8266_TIMEOUT_SEC` | `3` | Seconds to wait before giving up on the request |

Make sure your PC and ESP8266 are on the **same Wi-Fi network**. The ESP8266 must expose `/accident_on` (relay off → machine stops) and `/accident_off` (relay on → machine resumes).

---

## Running the app

```bash
python app-16.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

The app also binds to `0.0.0.0`, so it is reachable from other devices on the same network.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the web UI |
| `GET` | `/api/status` | Returns whether the API key is configured |
| `POST` | `/api/upload` | Upload a video for analysis. Form fields: `video` (file), `accident_ranges` (JSON array of `{start, end}` seconds) |
| `GET` | `/api/progress/<job_id>` | Poll analysis progress (`status`, `progress` 0–100, `message`) |
| `GET` | `/api/results/<job_id>` | Fetch full results once analysis is complete |
| `GET` | `/api/video/<filename>` | Stream the uploaded video file |
| `POST` | `/api/reanalyze` | Re-analyze a single frame. Body: `{"job_id": "...", "timestamp": "MM:SS or seconds"}` |
| `POST` | `/api/reference/upload` | Upload a reference accident video |
| `GET` | `/api/reference/progress/<ref_id>` | Poll reference build progress |
| `GET` | `/api/reference/list` | List all reference profiles |
| `POST` | `/api/esp8266/relay/off` | Manually trigger relay OFF. Optional body: `{"reason": "..."}` |
| `GET` | `/api/esp8266/status` | Returns current ESP8266 configuration |
| `POST` | `/api/esp8266/reset` | Sends `/accident_off` to ESP8266 (relay ON — machine safe to run) |

---

## File structure

```
app-16.py          # Full application (Flask server + embedded frontend)
uploads/           # Temporary storage for uploaded analysis videos (auto-created)
reference_videos/  # Reference accident video files and JSON profiles (auto-created)
```

---

## Notes & limitations

- Videos are sampled at roughly **2 fps**, capped at **60 frames** per job. Long videos will have lower temporal resolution.
- `analysis_results` and `analysis_progress` are stored in-memory; they are lost on server restart.
- The reference matching uses simple color histogram correlation. Lighting changes or camera angle shifts may affect match accuracy.
- The ESP8266 signal is fire-and-forget; if the device is unreachable, analysis continues normally and a warning is printed to the console.
