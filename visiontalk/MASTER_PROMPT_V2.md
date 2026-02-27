# ████████████████████████████████████████████████████████████████
# VISIONTALK — MASTER PROMPT V2 (FINAL, AUTHORITATIVE)
# ALL OLD PLAN FILES HAVE BEEN DELETED. ONLY USE FILES LISTED BELOW.
# ████████████████████████████████████████████████████████████████

## ⚠️ READ THIS FIRST — CRITICAL DEMO UNDERSTANDING

The demo works like this:
- **User physically closes their eyes**. Uses only voice commands. Hears guidance via speaker.
- **Judges watch the laptop/projector screen** showing live camera + bounding boxes.
- The live bounding boxes PROVE the AI is real — coordinates change every frame.
- **There is NO CSS black overlay feature**. That was wrong. Remove it if you coded it.
- The "Demo Mode" button = **PRESENTATION MODE**: forces overlay ON + big text + detective panel.



**You have one primary reference file:**
👉 **`FINAL_SPEC.md`** — Contains corrected, verified Python for every backend module.

**Supporting file for frontend:**
👉 **`03_FRONTEND_SPEC.md`** — Complete HTML/CSS/JS.

**Supporting file for demo UI:**
👉 **`05_DEMO_ADDITIONS.md`** — Blind mode, detective panel, conversation panel additions.

**DO NOT mix code from older files** (AI_BUILDER_PLAN_BACKEND.md, AI_BUILDER_PLAN_FRONTEND.md). Those have bugs. Use only the files above.

---

## PRE-CODING CHECKLIST — DO ALL 3 BEFORE WRITING ANY CODE

```
[ ] 1. Read FINAL_SPEC.md completely. Understand every module.
[ ] 2. Read 03_FRONTEND_SPEC.md completely. Know all 8 files.
[ ] 3. Read 05_DEMO_ADDITIONS.md. Know exactly what to patch in app.js.
```

If you have not checked all 3 boxes — STOP and read them.

---

## WHAT VISIONTALK DOES — PLAIN ENGLISH

VisionTalk is an offline AI companion for visually impaired people.
A smartphone camera streams video to a laptop over WiFi.
The laptop runs AI on every frame and speaks what it sees via laptop speakers.

Everything is OFFLINE. No cloud. No internet. No API keys.

**How it works in 5 words per feature:**

| Feature | 5-word description |
|---|---|
| NAVIGATE | "Sees objects, speaks their location" |
| Approaching Alert | "Warns if objects get closer" |
| ASK | "Answers questions about your scene" |
| Color Sense | "Tells you colors of things" |
| READ | "Reads any text in view" |
| Scene Snapshot | "Remembers scene, notices changes later" |

---

## MODULE MAP — WHAT EACH FILE DOES

```
backend/modes.py        — Thread-safe mode state (NAVIGATE/ASK/READ)
backend/camera.py       — Background thread reads MJPEG camera
backend/detector.py     — YOLOv8n ONNX → List[Detection]
backend/depth.py        — MiDaS → depth map + stair detection
backend/ocr.py          — PaddleOCR → center-priority text list
backend/color_sense.py  — HSV color identification (no LLM)
backend/spatial.py      — Detection + depth → SpatialResult
backend/scene_memory.py — TTL dedup + approaching alert tracking
backend/narrator.py     — SpatialResult → natural language string
backend/brain.py        — Ollama Llama-3 + color routing + fallback
backend/tts.py          — pyttsx3 TTS (threading.Queue, dedup)
backend/pipeline.py     — Orchestrates all modules, WS broadcast
backend/main.py         — FastAPI app, WS hub, QR, static
```

---

## DATA FLOW — EXACT TYPES AT EACH STEP

```python
# Step 1: Camera
frame: np.ndarray        # shape (480, 640, 3) BGR uint8

# Step 2: YOLO
detections: List[Detection]
# Detection fields: class_id:int, class_name:str, confidence:float, x1,y1,x2,y2:int

# Step 3: MiDaS (every 3rd frame)
depth_map: np.ndarray | None  # shape (480, 640) float32, 0.0=far, 1.0=close

# Step 4: Spatial
spatial_results: List[SpatialResult]
# SpatialResult fields:
#   class_name:str, confidence:float
#   direction: "far left"|"left"|"ahead"|"right"|"far right"
#   distance: "very close"|"nearby"|"ahead"|"far"
#   distance_level: 1|2|3|4   (1=very close, 4=far)
#   zone: "ground"|"mid"|"aerial"
#   depth_score: float (0.0 if MiDaS unavailable)
#   x1,y1,x2,y2: int
#   key: str (property = f"{class_name}_{direction}")
#   bbox_area: int (property = (x2-x1)*(y2-y1))

# Step 5: OCR
texts: List[str]   # sorted by distance from frame center

# Step 6: Narration string
msg: str   # e.g. "Chair nearby, to your left"

# Step 7: WS payload
data: dict   # see WEBSOCKET PROTOCOL section
```

---

## THE 18 INVIOLABLE RULES

Numbered. Named. Final. Break any = wrong output.

```
RULE 01 [QUEUE]     threading.Queue in TTS and pipeline. Never asyncio.Queue.
RULE 02 [LOCK]      threading.Lock in: ModeManager, CameraStream, OCRReader, TTSEngine, SceneMemory.
RULE 03 [DEPTH_NULL] depth_map = None — declare ABOVE the while loop in pipeline._run().
RULE 04 [MSG_NULL]  nav_msg = "" — declare ABOVE the while loop in pipeline._run().
RULE 05 [DOTENV]    load_dotenv() — LINE 1 of main.py. Nothing above it. Not even a comment.
RULE 06 [ASYNC]     asyncio.run_coroutine_threadsafe(broadcast(data), self._loop) — only bridge from thread to event loop.
RULE 07 [LIFESPAN]  @asynccontextmanager async def lifespan(app: FastAPI). NEVER @app.on_event.
RULE 08 [EXPIRE]    Scene memory expire:
          expired = [k for k, e in self._entries.items() if ...]   # list first
          for k in expired: del self._entries[k]                    # THEN delete
          DO NOT iterate and delete in same loop.
RULE 09 [BRAIN]     brain.py catches exactly: FileNotFoundError, subprocess.TimeoutExpired, Exception.
          Never bare except. Always logger.warning/error with message.
RULE 10 [VANILLAJS] No framework. Pure Vanilla JS. Never React, Vue, Angular, jQuery.
RULE 11 [CAMERA]    <img src="http://IP:8080/video"> for camera. NEVER getUserMedia().
RULE 12 [WSNAME]    sendCommand() — no space. Not 'send Command'. One word.
RULE 13 [CANVAS]    position:absolute; inset:0; width:100%; height:100%; pointer-events:none.
RULE 14 [SW]        Service worker must NOT intercept /ws/, /api/, /qr, or :8080. Check url first.
RULE 15 [COCO]      COCO_CLASSES[0]="person", COCO_CLASSES[79]="toothbrush". Exact 80-item order.
RULE 16 [SPATIAL]   SpatialResult MUST have .bbox_area property: return (x2-x1)*(y2-y1).
RULE 17 [PIPELINE]  Do NOT create det_list in pipeline.py. Pass spatial_results directly to brain.answer().
RULE 18 [WS_CHECK]  Always check ws && ws.readyState === WebSocket.OPEN before ws.send().
```

---

## BUILD ORDER — 26 STEPS

### Phase 1 — Scaffold
```
requirements.txt    .env.example    .gitignore    backend/__init__.py    models/
```

### Phase 2 — Backend (FINAL_SPEC.md, strict order)
```
01 modes.py → 02 camera.py → 03 detector.py → 04 depth.py → 05 ocr.py →
06 color_sense.py → 07 spatial.py → 08 scene_memory.py → 09 narrator.py →
10 brain.py → 11 tts.py → 12 pipeline.py → 13 main.py
```

### Phase 3 — Frontend (03_FRONTEND_SPEC.md + 05_DEMO_ADDITIONS.md)
```
14 index.html → 15 style.css → 16 camera.js → 17 overlay.js →
18 voice.js → 19 audio.js → 20 app.js → 21 manifest.json → 22 sw.js
```

### Phase 4 — Scripts + Docs
```
23 warmup.py → 24 start.bat → 25 start.sh → 26 README.md
```

---

## APPLYING DEMO ADDITIONS (05_DEMO_ADDITIONS.md)

After completing base frontend, make these SURGICAL edits:

### `index.html` — ADD 4 blocks INSIDE `<main>` AFTER `.secondary-controls`:
1. `<section class="demo-controls">` — Blind Mode + Detective buttons
2. `<div id="blind-overlay">` — black overlay with green narration text
3. `<div id="detective-panel">` — live detection table
4. `<div id="conversation-panel">` — chat Q&A history

### `style.css` — APPEND at end (do not replace anything):
- All CSS from 05_DEMO_ADDITIONS.md after existing rules

### `app.js` — 4 SURGICAL CHANGES:
1. REPLACE old `handleNarration()` → patched version (feeds blind overlay + detective)
2. REPLACE old `handleAnswer()` → patched version (adds chat bubble + blind overlay)
3. ADD after replacement: `updateBlindNarration()`, `updateDetectivePanel()`, `addConversationTurn()`
4. ADD button handlers: blind mode toggle, detective toggle

### Also from FINAL_SPEC.md — in `app.js`:
5. ADD snapshot button handler: `sendCommand({action:"snapshot"})`
6. ADD in `ws.onmessage`: handle `data.type === "system"` → `setBanner(data.text, 0)`

### In `voice.js` — ADD to `rec.onresult` handler:
```javascript
if (lower.includes('remember') || lower === 'remember this') {
  window.sendCommand?.({ type:'command', action:'snapshot' });
} else if (lower.includes('what changed')) {
  window.sendCommand?.({ type:'command', action:'scene_diff' });
}
```

---

## ZERO TOLERANCE LIST — SEARCH AND DELETE

After writing all code, search for these strings. If found, fix before submitting:

```
asyncio.Queue           → threading.Queue
@app.on_event           → @asynccontextmanager lifespan()
getUserMedia            → remove (camera is img tag)
import React            → remove. Vanilla JS only.
import Vue              → remove.
send Command(           → sendCommand( (no space)
for k in self._entries: del self._entries[k]  → use expire list pattern
except:\n    pass       → add logger.error and message
det_list                → remove. Never use this variable.
_pending_frame          → remove. Never declare this.
# TODO                  → every function must be complete
requests.get(           → remove. Offline only.
import openai           → remove. Never.
fetch("https://api      → remove. Offline only.
```

---

## WEBSOCKET PROTOCOL (final)

### Server → Client
```json
{"type":"narration","mode":"NAVIGATE","text":"...","severity":1|2|3|4,
 "show_overlay":true|false,"fps":9.8,
 "detections":[{"class_name":"chair","direction":"left","distance":"nearby",
                "distance_level":2,"confidence":0.87,"x1":142,"y1":241,"x2":318,"y2":479}]}

{"type":"answer","question":"What is this?","answer":"I can see a chair.",
 "context":["chair (left, nearby)"],"fps":9.2}

{"type":"reading","text":"Reading: Paracetamol 500mg.","fps":9.5,"detections":[...]}
{"type":"system","text":"Scene snapshot saved."}
{"type":"init","mode":{"current_mode":"NAVIGATE","show_overlay":false}}
```

### Client → Server
```json
{"type":"command","action":"set_mode","mode":"NAVIGATE"|"ASK"|"READ"}
{"type":"command","action":"ask","question":"..."}
{"type":"command","action":"toggle_overlay"}
{"type":"command","action":"snapshot"}
{"type":"command","action":"scene_diff"}
```

---

## VERIFICATION COMMANDS (run in order)

```bash
# 1. Quick import check
python -c "
from backend.spatial import SpatialResult
r = SpatialResult('chair',0.9,'left','nearby',2,'mid',0.5,100,100,200,200)
assert r.bbox_area == 10000
assert r.key == 'chair_left'
print('SpatialResult: OK')
"

# 2. Scene memory expire safety
python -c "
from backend.scene_memory import SceneMemory, scene_memory
from backend.spatial import SpatialResult
r = SpatialResult('chair',0.9,'ahead','nearby',2,'mid',0.5,100,100,300,300)
scene_memory.update([r])
scene_memory.update([])
print('SceneMemory expire: OK')
"

# 3. Brain fallback (no Ollama needed)
python -c "
from backend.brain import Brain
b = Brain()
import numpy as np
frame = np.zeros((480,640,3),dtype=np.uint8)
result = b.answer('What is this?', frame, [], [])
assert isinstance(result, str) and len(result) > 0
print('Brain fallback: OK')
"

# 4. Model warmup
python scripts/warmup.py

# 5. Server
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 6. API checks
curl http://localhost:8000/api/status
# → {"ok":true,"mode":{"current_mode":"NAVIGATE","show_overlay":false},"fps":...}

curl -I http://localhost:8000/qr
# → HTTP/1.1 200 OK, content-type: image/png
```

---

## FILE COUNT: 30 TOTAL

```
backend/    14 files (__init__.py + 13 Python modules)
frontend/    9 files (index, style, camera, overlay, voice, audio, app, manifest, sw)
scripts/     3 files (warmup.py, start.bat, start.sh)
root         4 files (requirements.txt, .env.example, .gitignore, README.md)
─────────────────────────────────
TOTAL       30 files

Count yours. If ≠ 30, find and create the missing file.
```

---

## CLOSING STATEMENT

This is the final, authoritative prompt. The code in FINAL_SPEC.md has been:
- Audited line by line
- All bugs documented and corrected
- Verified against Python semantics (dict iteration safety, threading, type consistency)
- New features added and proven real with exact implementation

Produce 30 complete files. Start with Phase 1 scaffold. Output each in order.
No abbreviations. No `# ... rest of code`. No stubs.
Every function must be completely implemented.
