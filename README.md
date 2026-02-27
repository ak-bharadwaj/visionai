# VisionTalk

**Offline AI spatial assistant for the visually impaired.**

VisionTalk runs entirely on a local machine — no cloud APIs, no internet required. It uses a phone camera as a live feed, performs real-time object detection, depth estimation, and OCR, then narrates the scene aloud so visually impaired users can navigate safely and independently.

---

## Features

- **NAVIGATE mode** — continuous spatial narration: detects objects, estimates distance and direction, speaks warnings for approaching hazards
- **READ mode** — OCR via PaddleOCR; reads text in the camera frame aloud
- **ASK mode** — answers free-form questions about the scene using a local LLM (Ollama) or rule-based brain
- **Scene memory** — snapshot and diff: remembers what was in view and tells you what changed
- **Approaching alert** — warns when an object is getting closer across frames
- **Demo Mode** — bounding box overlay + live detective panel for hackathon judges
- **PWA** — installable, offline-capable via service worker
- **Push-to-talk voice** — hold mic button to speak commands or ask questions

---

## Architecture

```
Phone camera (IP stream :8080)
        │
        ▼
backend/camera.py      ← MJPEG frame reader (threading)
        │
        ▼
backend/pipeline.py    ← Main loop: detect → depth → spatial → narrate → TTS
        │
   ┌────┴───────────────────────────────────┐
   │                                        │
backend/detector.py    YOLOv8n             backend/ocr.py      PaddleOCR
backend/depth.py       MiDaS              backend/color_sense.py
backend/spatial.py     distance/direction  backend/brain.py    LLM / rules
backend/scene_memory.py                   backend/narrator.py
backend/tts.py         pyttsx3 (offline)
        │
        ▼
backend/main.py        FastAPI + WebSocket broadcast
        │
        ▼
frontend/              Vanilla JS PWA (no React/Vue)
  index.html / style.css / app.js / overlay.js
  camera.js / voice.js / audio.js / sw.js / manifest.json
```

---

## Requirements

- Python 3.10+
- A phone running an MJPEG camera app (e.g. IP Webcam for Android) on the same LAN
- (Optional) [Ollama](https://ollama.ai) running locally for LLM-powered ASK answers

---

## Quick Start

### Windows

```bat
scripts\start.bat
```

### Linux / macOS

```bash
chmod +x scripts/start.sh
./scripts/start.sh
```

Both scripts:
1. Create a virtual environment
2. Install all dependencies from `requirements.txt`
3. Copy `.env.example` → `.env` if missing
4. Start the server at `http://0.0.0.0:8000`

---

## Configuration

Edit `.env` (created automatically on first run):

```env
CAMERA_SOURCE=http://192.168.1.x:8080/video   # Your phone's IP stream URL
OLLAMA_MODEL=llama3:8b                         # Local LLM model name (Ollama)
TTS_RATE=175                                   # Speech rate (words/min)
TTS_VOLUME=1.0                                 # TTS volume (0.0–1.0)
INFERENCE_FPS=10                               # Max frames per second for inference
INFERENCE_SIZE=320                             # YOLO input resolution (px)
DEPTH_EVERY_N_FRAMES=3                         # Run MiDaS every N frames
LOG_LEVEL=INFO                                 # Logging verbosity (DEBUG/INFO/WARNING)
```

---

## Pre-demo Warmup

Run this once before the demo to pre-load all models into memory:

```bash
python scripts/warmup.py
```

This loads YOLOv8n, MiDaS, and PaddleOCR and confirms each is ready.

---

## File Structure

```
.
├── backend/
│   ├── __init__.py
│   ├── modes.py          # ModeManager (NAVIGATE / READ / ASK)
│   ├── camera.py         # CameraStream — MJPEG reader
│   ├── detector.py       # ObjectDetector — YOLOv8n, 80 COCO classes
│   ├── depth.py          # DepthEstimator — MiDaS
│   ├── ocr.py            # OCRReader — PaddleOCR
│   ├── color_sense.py    # Dominant color of detected objects
│   ├── spatial.py        # SpatialResult — distance + direction
│   ├── scene_memory.py   # SceneMemory — history, approaching alert, snapshot/diff
│   ├── narrator.py       # Narration templates + approaching alert phrasing
│   ├── brain.py          # LLM / rule-based question answering
│   ├── tts.py            # TTSEngine — pyttsx3, threading.Queue
│   ├── pipeline.py       # PipelineRunner — main processing loop
│   └── main.py           # FastAPI app — REST + WebSocket
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── app.js            # WS client, mode buttons, detective panel
│   ├── overlay.js        # Canvas bounding box overlay
│   ├── camera.js         # Camera feed + QR code display
│   ├── voice.js          # Push-to-talk speech recognition
│   ├── audio.js          # Stub (TTS is server-side)
│   ├── manifest.json     # PWA manifest
│   └── sw.js             # Service worker (never intercepts /ws/ or :8080)
├── scripts/
│   ├── warmup.py
│   ├── start.bat
│   └── start.sh
├── models/               # Auto-downloaded model weights (gitignored)
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## WebSocket Protocol

Connect to `ws://<host>:8000/ws/guidance`.

**Server → Client messages:**

| `type`      | Key fields                                      |
|-------------|-------------------------------------------------|
| `init`      | `mode` (current mode state)                     |
| `narration` | `text`, `severity` (0-3), `fps`, `detections`  |
| `answer`    | `question`, `answer`, `fps`                     |
| `reading`   | `text`, `fps`, `detections`                     |
| `system`    | `text`                                          |

**Client → Server commands (JSON):**

```json
{ "type": "command", "action": "set_mode", "mode": "NAVIGATE" }
{ "type": "command", "action": "ask", "question": "What is in front of me?" }
{ "type": "command", "action": "snapshot" }
{ "type": "command", "action": "scene_diff" }
{ "type": "command", "action": "toggle_overlay" }
{ "type": "command", "action": "clear_history" }
```

---

## Inviolable Design Rules

- All queues use `threading.Queue` — never `asyncio.Queue`
- All locks use `threading.Lock`
- No cloud API calls — fully offline
- No React, Vue, or jQuery — pure Vanilla JS
- Camera feed via `<img src="http://IP:8080/video">` — never `getUserMedia()`
- Service worker never intercepts `/ws/`, `/api/`, `/qr`, or `:8080`

---

## License

MIT
