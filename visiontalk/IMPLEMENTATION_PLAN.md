# VisionTalk — Full Implementation Plan (Verified v2)

> Saved from architecture review. This is the full system design reference.
> See AI_BUILDER_PLAN.md for the step-by-step AI coding guide.

## Problem Statement

Build a hackathon-winning assistive navigation system that:
- Uses only a smartphone camera (no additional sensors/hardware)
- Runs 100% offline on a laptop over local WiFi
- Provides continuous directional voice guidance to visually impaired users
- Demonstrates real engineering depth: spatial reasoning, CV pipeline, voice UX

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│  PHONE (Android/iOS)                                                 │
│  ┌─────────────────────┐    ┌──────────────────────────────────────┐ │
│  │  IP Webcam App      │    │  Browser (Chrome) — PWA              │ │
│  │  MJPEG stream ──────┼───►│  Modes: Obstacle / Finder / OCR      │ │
│  │  http://IP:8080/    │WiFi│  Canvas bounding box overlay         │ │
│  │  video              │    │  Guidance text banner + TTS audio     │ │
│  └─────────────────────┘    │  Voice commands (SpeechRecognition)  │ │
│                             │  QR code to connect phone            │ │
│                             └──────────┬─────────────────────────  ┘ │
└────────────────────────────────────────│─────────────────────────────┘
                                         │ WebSocket ws://IP:8000
                                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│  LAPTOP — Python Backend (FastAPI + uvicorn)                         │
│                                                                      │
│  main.py (lifespan + WebSocket hub + static serving)                 │
│  pipeline.py (background thread @ 10fps)                             │
│    ├── camera.py        MJPEG → BGR ndarray                          │
│    ├── detector.py      YOLOv8n ONNX → List[Detection]              │
│    ├── depth.py         MiDaS depth heatmap (every 3rd frame)        │
│    ├── ocr.py           PaddleOCR (OCR mode only)                    │
│    ├── spatial.py       Detection + depth → SpatialResult            │
│    ├── scene_memory.py  TTL dedup → new alerts only                  │
│    ├── decision.py      Risk score + NLG + path clear check          │
│    ├── tts.py           pyttsx3 (threading.Queue, thread-safe)       │
│    └── modes.py         ModeManager singleton (threading.Lock)       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
visiontalk/
├── backend/
│   ├── __init__.py
│   ├── main.py, pipeline.py, camera.py, detector.py
│   ├── depth.py, ocr.py, spatial.py, scene_memory.py
│   ├── decision.py, tts.py, modes.py
├── frontend/
│   ├── index.html, style.css, app.js, camera.js
│   ├── audio.js, voice.js, overlay.js
│   ├── manifest.json, sw.js
├── models/           (gitignored, auto-downloaded)
├── scripts/
│   ├── start.bat, start.sh, warmup.py
├── docs/DEMO_GUIDE.md
├── requirements.txt, .env.example, .gitignore, README.md
```

---

## Tech Stack

| Layer | Tool | Version |
|---|---|---|
| Object Detection | YOLOv8n ONNX | ultralytics 8.1 |
| ONNX Runtime | onnxruntime | 1.17.1 |
| Depth | MiDaS v2.1-small | torch.hub + timm 0.9 |
| OCR | PaddleOCR | 2.7.3 |
| Backend | FastAPI + uvicorn | 0.110 / 0.27 |
| Camera | OpenCV | 4.9 |
| TTS | pyttsx3 | 2.90 |
| Frontend | Vanilla JS PWA | — |
| Phone Cam | IP Webcam App (Android) | — |

---

## Key Issues Resolved in This Plan

1. SpatialResult has class_name + key fields
2. depth_map initialized to None before pipeline loop
3. msg initialized to "" before pipeline loop
4. threading.Queue used in TTS (not asyncio.Queue)
5. ModeManager uses threading.Lock
6. Audio homing specced in decision.py (get_finder_proximity_tone)
7. safe_passage via check_path_clear()
8. Camera display uses MJPEG img src (no HTTPS needed)
9. Cross-thread broadcast via asyncio.run_coroutine_threadsafe()
10. MiDaS via torch.hub (auto-caches, not local file)
11. camera.js fully specced
12. sw.js fully specced
13. PaddleOCR Windows: tuna mirror + timeout guard
14. start.bat includes venv activation
15. FastAPI uses lifespan() not deprecated @on_event
16. /qr endpoint for phone connection via QR scan

---

## See AI_BUILDER_PLAN.md for the full feature-by-feature coding guide.
