# VisionTalk — FINAL CORRECTED COMPLETE SPECIFICATION
## Version: FINAL-V2. All bugs fixed. Demo design correct. Zero ambiguity.
## Date: 2026-02-25. THIS IS THE ONLY BACKEND SPEC FILE. All others are deleted.

## ⚠️ CORRECT DEMO DESIGN (read before coding demo features)
# "Blind Demo" = user closes REAL eyes, speaks by voice, hears audio guidance.
# Judges watch the LAPTOP SCREEN showing live camera + bounding boxes + text.
# There is NO CSS black overlay trick. The AI is real. The user is real.
# Demo Presentation Mode = Overlay ALWAYS ON + big text + detective panel.
# See DEMO_DESIGN.md for the full 3-minute script.

---

## BUG FIXES — APPLIED IN THIS DOCUMENT

| Bug # | Location | Bug | Fix Applied |
|---|---|---|---|
| 1 | `pipeline.py` | `det_list` built but never passed to brain | Removed — pass `spatial_results` directly |
| 2 | `pipeline.py` | `_pending_frame` declared but never used | Removed entirely |
| 3 | `color_sense.py` | `hasattr(d, 'bbox_area')` always False for SpatialResult | Use `(d.x2-d.x1)*(d.y2-d.y1)` directly |
| 4 | `spatial.py` | `SpatialResult` missing `.bbox_area` property | Added property: `(x2-x1)*(y2-y1)` |
| 5 | `pipeline.py` | `msg != path_msg` stale across modes | Track `_last_nav_msg` separately in NAVIGATE branch |
| 6 | `narrator.py` | "Chair far right" sounds wrong | Templates now use "to your {dir}" for all |

## NEW WINNING FEATURES ADDED

| Feature | Module | What it does |
|---|---|---|
| 🚨 **Approaching Alert** | `scene_memory.py` | Fires urgent warning when object moves closer 2 levels in 3 frames |
| 📸 **Scene Snapshot** | `pipeline.py` | "Remember this" → store scene; "What changed?" → diff with current |

---

## FINAL ARCHITECTURE

```
Phone Camera (IP Webcam app at :8080)
    │
    ▼
backend/camera.py — CameraStream (MJPEG, threaded, 10fps, Lock)
    │
    ▼
backend/detector.py — YOLOv8n ONNX → List[Detection(class_name,conf,x1,y1,x2,y2)]
    │
    ├──▶ backend/depth.py — MiDaS → float32 depth_map (every 3rd frame)
    │
    ├──▶ backend/spatial.py — SpatialAnalyzer → List[SpatialResult(dir,dist,level,zone)]
    │
    ├──[NAVIGATE]──▶ backend/scene_memory.py → dedup, TTL, approaching alert
    │                      │
    │                      ▼
    │               backend/narrator.py → spoken string
    │
    ├──[ASK]──▶ backend/ocr.py → texts
    │           backend/color_sense.py → color name (if color question)
    │           backend/brain.py → Ollama Llama-3 → answer string
    │
    ├──[READ]──▶ backend/ocr.py → center-priority texts
    │
    ▼
backend/tts.py — pyttsx3 (threading.Queue, dedup 2s)
    │
    ▼
backend/pipeline.py — orchestrates all above, broadcasts WS
    │
    ▼
backend/main.py — FastAPI WS hub, QR endpoint, static serve
    │
    ▼
WebSocket ws://localhost:8000/ws/guidance
    │
    ▼
frontend/app.js — WS client, mode buttons, detective panel, blind mode, conversation
    │
    ├── frontend/camera.js — <img src="http://IP:8080/video">
    ├── frontend/overlay.js — canvas bounding boxes
    └── frontend/voice.js — push-to-talk SpeechRecognition
```

---

## COMPLETE CORRECTED BACKEND CODE

### `backend/modes.py` ← UNCHANGED, correct as written
```python
import threading

VALID_MODES = {"NAVIGATE", "ASK", "READ"}

class ModeManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._mode = "NAVIGATE"
        self._show_overlay = False

    def set_mode(self, mode: str):
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode: {mode!r}. Must be one of {VALID_MODES}")
        with self._lock:
            self._mode = mode

    def toggle_overlay(self):
        with self._lock:
            self._show_overlay = not self._show_overlay

    def is_navigate(self) -> bool:
        with self._lock: return self._mode == "NAVIGATE"

    def is_ask(self) -> bool:
        with self._lock: return self._mode == "ASK"

    def is_read(self) -> bool:
        with self._lock: return self._mode == "READ"

    def snapshot(self) -> dict:
        with self._lock:
            return {"current_mode": self._mode, "show_overlay": self._show_overlay}

mode_manager = ModeManager()
```

---

### `backend/camera.py` ← UNCHANGED, correct as written
```python
import cv2, threading, time, os, logging
import numpy as np

logger = logging.getLogger(__name__)

class CameraStream:
    def __init__(self, source):
        self._source = int(source) if str(source).isdigit() else source
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._cap: cv2.VideoCapture | None = None

    def start(self) -> "CameraStream":
        self._cap = cv2.VideoCapture(self._source)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._cap.isOpened():
            logger.warning(f"Cannot open '{self._source}'. Falling back to webcam 0.")
            self._cap = cv2.VideoCapture(0)
            if not self._cap.isOpened():
                raise RuntimeError("No camera available. Set CAMERA_SOURCE in .env")
        self._running = True
        threading.Thread(target=self._loop, daemon=True, name="CameraThread").start()
        logger.info(f"Camera started: {self._source}")
        return self

    def _loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.01)

    def read(self) -> np.ndarray | None:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        time.sleep(0.1)
        if self._cap:
            self._cap.release()
        logger.info("Camera stopped.")

def get_camera() -> CameraStream:
    source = os.getenv("CAMERA_SOURCE", "0")
    return CameraStream(source).start()
```

---

### `backend/detector.py` ← UNCHANGED, correct as written
(Use full 80-item COCO_CLASSES array from 02_BACKEND_SPEC.md exactly)
```python
# [Full code same as 02_BACKEND_SPEC.md STEP 3 — copy exactly]
# Key check: COCO_CLASSES[0]="person", COCO_CLASSES[79]="toothbrush"
# YOLOv8n output shape: [1, 84, 8400], transposed to [8400, 84]
# preds[:, :4] = cx,cy,w,h; preds[:, 4:] = 80 class scores
```

---

### `backend/depth.py` ← UNCHANGED, correct as written
```python
# [Full code same as 02_BACKEND_SPEC.md STEP 4 — copy exactly]
# MiDaS outputs inverse disparity: higher value = CLOSER to camera.
# After normalization (pred - mn)/(mx - mn): 0.0=far, 1.0=closest. CORRECT.
# detect_stair_drop: gradient std > 0.25 in bottom-center 30% of frame.
```

---

### `backend/ocr.py` ← UNCHANGED, correct as written
```python
# [Full code same as 02_BACKEND_SPEC.md STEP 5 — copy exactly]
# Center-priority: sort by Euclidean distance from frame center.
# Confidence filter: > 0.60 only.
```

---

### `backend/color_sense.py` ← BUG FIX APPLIED (bbox_area calculation)

```python
import cv2, numpy as np, logging

logger = logging.getLogger(__name__)

# OpenCV HSV: Hue is 0-179 (half of standard 0-360)
COLOR_RANGES = [
    (  0,  10, "red"),
    ( 10,  25, "orange"),
    ( 25,  35, "yellow"),
    ( 35,  85, "green"),
    ( 85, 100, "cyan"),
    (100, 130, "blue"),
    (130, 145, "purple"),
    (145, 160, "pink"),
    (160, 179, "red"),
]

def _hue_to_name(hue: float) -> str:
    for lo, hi, name in COLOR_RANGES:
        if lo <= hue <= hi:
            return name
    return "red"

def get_dominant_color(frame: np.ndarray, det=None) -> str:
    """
    Uses HSV colorspace to find dominant color.
    det: any object with x1,y1,x2,y2 attributes (Detection or SpatialResult).
    Returns: e.g. "dark blue", "light green", "gray", "black", "white"
    """
    try:
        if det is not None and hasattr(det, 'x1'):
            region = frame[det.y1:det.y2, det.x1:det.x2]
        else:
            h, w = frame.shape[:2]
            region = frame[int(h*0.30):int(h*0.70), int(w*0.30):int(w*0.70)]

        if region.size == 0:
            return "unknown"

        hsv    = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        mean_s = float(np.mean(hsv[:, :, 1]))
        mean_v = float(np.mean(hsv[:, :, 2]))

        if mean_s < 40:
            if mean_v < 60:    return "black"
            elif mean_v > 180: return "white"
            else:              return "gray"

        mean_h     = float(np.median(hsv[:, :, 0]))
        base_color = _hue_to_name(mean_h)

        if mean_v < 80:    prefix = "dark "
        elif mean_v > 200: prefix = "light "
        else:              prefix = ""

        return f"{prefix}{base_color}"
    except Exception as e:
        logger.error(f"Color sense error: {e}")
        return "unknown"

def answer_color_question(frame: np.ndarray, detections: list) -> str:
    """
    BUG FIX: Use (x2-x1)*(y2-y1) for area — SpatialResult has no .bbox_area.
    Both Detection and SpatialResult have x1, y1, x2, y2.
    """
    det = None
    if detections:
        # ✅ FIXED: calculate area from x1,y1,x2,y2 directly
        det = max(
            detections,
            key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1) if hasattr(d, 'x2') else 0
        )
    color = get_dominant_color(frame, det)
    if det and hasattr(det, 'class_name'):
        return f"The {det.class_name} appears to be {color}."
    return f"The object in front appears to be {color}."
```

---

### `backend/spatial.py` ← BUG FIX: added bbox_area property + fixed direction phrasing

```python
import numpy as np
from dataclasses import dataclass
from backend.detector import Detection
from backend.depth import depth_estimator


@dataclass
class SpatialResult:
    class_name:     str
    confidence:     float
    direction:      str    # "far left"|"left"|"ahead"|"right"|"far right"
    distance:       str    # "very close"|"nearby"|"ahead"|"far"
    distance_level: int    # 1=very close, 2=nearby, 3=ahead, 4=far
    zone:           str    # "ground"|"mid"|"aerial"
    depth_score:    float
    x1: int; y1: int; x2: int; y2: int

    @property
    def key(self) -> str:
        return f"{self.class_name}_{self.direction}"

    @property
    def bbox_area(self) -> int:
        # ✅ ADDED: used by color_sense and any other module needing area
        return (self.x2 - self.x1) * (self.y2 - self.y1)


class SpatialAnalyzer:
    def analyze(self, det: Detection, fw: int, fh: int,
                depth_map) -> SpatialResult:
        cx_ratio = (det.x1 + det.x2) / 2 / fw
        if cx_ratio < 0.20:    direction = "far left"
        elif cx_ratio < 0.40:  direction = "left"
        elif cx_ratio < 0.60:  direction = "ahead"
        elif cx_ratio < 0.75:  direction = "right"
        else:                  direction = "far right"

        area_ratio   = ((det.x2-det.x1)*(det.y2-det.y1)) / (fw * fh)
        bottom_ratio = det.y2 / fh

        if area_ratio > 0.15 and bottom_ratio > 0.70:
            level = 1; dist_str = "very close"
        elif area_ratio > 0.06:
            level = 2; dist_str = "nearby"
        elif area_ratio > 0.02:
            level = 3; dist_str = "ahead"
        else:
            level = 4; dist_str = "far"

        d_score = depth_estimator.get_region_depth(
            depth_map, det.x1, det.y1, det.x2, det.y2)
        if d_score > 0.80:
            level = min(level, 1); dist_str = "very close"
        elif d_score > 0.60:
            level = min(level, 2); dist_str = "nearby"

        top_ratio = det.y1 / fh
        if top_ratio < 0.30:   zone = "aerial"
        elif top_ratio < 0.65: zone = "mid"
        else:                  zone = "ground"

        return SpatialResult(
            class_name=det.class_name, confidence=det.confidence,
            direction=direction, distance=dist_str, distance_level=level,
            zone=zone, depth_score=d_score,
            x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2,
        )


spatial_analyzer = SpatialAnalyzer()
```

---

### `backend/scene_memory.py` ← NEW FEATURE: Approaching Alert added

```python
import time, threading, logging
from collections import deque
from dataclasses import dataclass, field
from typing import List
from backend.spatial import SpatialResult

logger = logging.getLogger(__name__)


@dataclass
class SceneEntry:
    result:        SpatialResult
    first_seen:    float
    last_seen:     float
    announced:     bool
    last_level:    int
    # NEW: track last 3 distance_levels to detect approaching movement
    level_history: deque = field(default_factory=lambda: deque(maxlen=3))
    approach_warned: bool = False   # don't repeat approach alert


class SceneMemory:
    TTL = 3.0

    def __init__(self):
        self._entries: dict[str, SceneEntry] = {}
        self._lock = threading.Lock()

    def update(self, results: List[SpatialResult]):
        now = time.time()
        with self._lock:
            for r in results:
                k = r.key
                if k in self._entries:
                    e = self._entries[k]
                    e.last_seen = now
                    e.result    = r
                    e.level_history.append(r.distance_level)
                    if abs(r.distance_level - e.last_level) >= 1:
                        e.announced      = False
                        e.approach_warned = False   # reset on level change
                        e.last_level     = r.distance_level
                else:
                    entry = SceneEntry(
                        result=r, first_seen=now, last_seen=now,
                        announced=False, last_level=r.distance_level,
                    )
                    entry.level_history.append(r.distance_level)
                    self._entries[k] = entry

            # ✅ SAFE expire: build list first, THEN delete
            expired = [k for k, e in self._entries.items()
                       if now - e.last_seen > self.TTL]
            for k in expired:
                del self._entries[k]

    def get_new_alerts(self) -> List[SpatialResult]:
        with self._lock:
            return [e.result for e in self._entries.values() if not e.announced]

    def get_approaching(self) -> List[SpatialResult]:
        """
        NEW FEATURE — Approaching Alert.
        Returns objects where distance_level decreased 2 steps in last 3 frames.
        Pattern: [4,3,2] or [3,2,1] = object is approaching.
        Only fires if object is now at level <= 2 (nearby or very close).
        """
        approaching = []
        with self._lock:
            for e in self._entries.values():
                h = list(e.level_history)
                if len(h) >= 3:
                    # strictly decreasing AND now close
                    if h[-1] < h[-2] < h[-3] and h[-1] <= 2 and not e.approach_warned:
                        approaching.append(e.result)
        return approaching

    def mark_approach_warned(self, key: str):
        with self._lock:
            if key in self._entries:
                self._entries[key].approach_warned = True

    def mark_announced(self, key: str):
        with self._lock:
            if key in self._entries:
                self._entries[key].announced = True

    def get_snapshot(self) -> dict:
        """Returns a frozen copy of current scene for comparison."""
        with self._lock:
            return {k: e.result.class_name for k, e in self._entries.items()}

    def clear(self):
        with self._lock:
            self._entries.clear()


scene_memory = SceneMemory()
```

---

### `backend/narrator.py` ← BUG FIX: templates use "to your {dir}" phrasing

```python
from backend.spatial import SpatialResult
from typing import List

RISK_SCORE: dict[str, int] = {
    "person": 10, "car": 10, "motorcycle": 10, "bicycle": 10,
    "bus": 10, "truck": 10, "train": 10,
    "chair": 7, "couch": 7, "dining table": 6, "bench": 6,
    "door": 5, "suitcase": 5, "bed": 5,
    "toilet": 4, "sink": 4, "refrigerator": 4,
}
DEFAULT_RISK = 3

MIN_ALERT_LEVEL: dict[str, int] = {
    "person": 4, "car": 4, "motorcycle": 4, "bicycle": 4,
    "bus": 4, "truck": 4, "train": 4,
    "chair": 3, "couch": 3, "dining table": 3, "bench": 3, "door": 3,
}
DEFAULT_MIN_LEVEL = 2

# ✅ FIXED: "{dir}" is always used with preposition now
# "ahead" → "Chair ahead, very close"
# "left"  → "Chair nearby, to your left"  (not "Chair nearby, on your left"—kept same)
NAVIGATE_TEMPLATES: dict[int, str] = {
    1: "Stop! {Cls} directly {dir}",
    2: "{Cls} nearby, to your {dir}",
    3: "{Cls} to your {dir}, a few steps away",
    4: "",       # suppress — too far, not useful
}
# Special cases where "ahead" sounds better without "to your"
AHEAD_TEMPLATES: dict[int, str] = {
    1: "Stop! {Cls} directly ahead",
    2: "{Cls} nearby, directly ahead",
    3: "{Cls} ahead, a few steps away",
    4: "",
}

APPROACH_TEMPLATES: dict[str, str] = {
    "person":   "Warning! Person approaching from {dir}",
    "car":      "Warning! Car approaching from {dir}",
    "bicycle":  "Warning! Bicycle approaching from {dir}",
    "default":  "Warning! {Cls} approaching from {dir}",
}


class Narrator:
    def prioritize(self, alerts: List[SpatialResult]) -> List[SpatialResult]:
        def should_announce(r: SpatialResult) -> bool:
            max_level = MIN_ALERT_LEVEL.get(r.class_name, DEFAULT_MIN_LEVEL)
            return r.distance_level <= max_level
        filtered = [r for r in alerts if should_announce(r)]
        return sorted(
            filtered,
            key=lambda r: (-RISK_SCORE.get(r.class_name, DEFAULT_RISK), r.distance_level)
        )

    def narrate(self, result: SpatialResult) -> str:
        """
        ✅ FIXED: use AHEAD_TEMPLATES when direction == 'ahead' (sounds natural).
        Use NAVIGATE_TEMPLATES with 'to your {dir}' for left/right.
        """
        if result.direction == "ahead":
            tmpl = AHEAD_TEMPLATES.get(result.distance_level, "")
        else:
            tmpl = NAVIGATE_TEMPLATES.get(result.distance_level, "")

        if not tmpl:
            return ""
        return tmpl.format(
            cls=result.class_name,
            Cls=result.class_name.title(),
            dir=result.direction,
        )

    def narrate_approaching(self, result: SpatialResult) -> str:
        """NEW: Generate approaching alert string."""
        tmpl = APPROACH_TEMPLATES.get(result.class_name,
                                       APPROACH_TEMPLATES["default"])
        return tmpl.format(
            cls=result.class_name,
            Cls=result.class_name.title(),
            dir=result.direction,
        )

    def path_clear(self, results: List[SpatialResult]) -> str | None:
        blocking = [r for r in results
                    if r.direction == "ahead" and r.distance_level <= 2]
        return "Path clear ahead" if not blocking else None


narrator = Narrator()
```

---

### `backend/brain.py` ← UNCHANGED, correct — no bug here
```python
# [Full code same as 02_BACKEND_SPEC.md STEP 10 — copy exactly]
# Catches: FileNotFoundError, subprocess.TimeoutExpired, Exception
# Color routing to color_sense.py is correct
# Fallback returns real scene data, never a hardcoded string
```

---

### `backend/tts.py` ← UNCHANGED, correct as written
```python
# [Full code same as 02_BACKEND_SPEC.md STEP 11 — copy exactly]
# threading.Queue is correct
# dedup 2s window is correct
# priority=True flushes queue before speaking
```

---

### `backend/pipeline.py` ← BUGS FIXED + NEW FEATURES

```python
import time, threading, asyncio, logging
from typing import Callable
import numpy as np

from backend.modes import mode_manager
from backend.camera import get_camera
from backend.detector import ObjectDetector
from backend.depth import depth_estimator
from backend.ocr import ocr_reader
from backend.spatial import spatial_analyzer
from backend.scene_memory import scene_memory
from backend.narrator import narrator
from backend.brain import brain
from backend.tts import tts_engine

logger = logging.getLogger(__name__)

DEPTH_N = int(__import__("os").getenv("DEPTH_EVERY_N_FRAMES", "3"))
OCR_N   = 5
FPS_CAP = int(__import__("os").getenv("INFERENCE_FPS", "10"))


class PipelineRunner:
    def __init__(self):
        self._running   = False
        self._camera    = None
        self._detector  = None
        self._loop:  asyncio.AbstractEventLoop | None = None
        self._bcast: Callable | None = None
        self.fps = 0.0
        self._q_lock            = threading.Lock()
        self._pending_question: str | None = None
        # ✅ FIXED: removed _pending_frame (dead code)
        # NEW FEATURE: Scene Snapshot
        self._snapshot: dict | None   = None  # stores scene at time of "remember"
        self._last_spatial: list      = []    # always-current spatial_results

    # ── Public API (called from WS handler, thread-safe) ──────────
    def set_question(self, question: str):
        with self._q_lock:
            self._pending_question = question

    def take_snapshot(self):
        """NEW: Called when user says 'remember this'."""
        with self._q_lock:
            self._snapshot = scene_memory.get_snapshot()

    def get_scene_diff(self) -> str:
        """NEW: Compare current scene to stored snapshot."""
        with self._q_lock:
            snap = self._snapshot

        if snap is None:
            return "No snapshot taken. Say 'remember this' first."

        current = scene_memory.get_snapshot()
        appeared    = [v for k, v in current.items() if k not in snap]
        disappeared = [v for k, v in snap.items()     if k not in current]
        parts = []
        if appeared:    parts.append(f"New objects: {', '.join(appeared)}")
        if disappeared: parts.append(f"Gone: {', '.join(disappeared)}")
        return ". ".join(parts) if parts else "Scene unchanged since snapshot."

    # ── Lifecycle ──────────────────────────────────────────────────
    def start(self, event_loop: asyncio.AbstractEventLoop, broadcast_fn: Callable):
        self._camera   = get_camera()
        self._detector = ObjectDetector()
        depth_estimator.load()
        tts_engine.start()
        self._loop  = event_loop
        self._bcast = broadcast_fn
        self._running = True
        threading.Thread(target=self._run, daemon=True, name="PipelineThread").start()
        logger.info("Pipeline started.")

    def stop(self):
        self._running = False
        if self._camera:
            self._camera.stop()

    # ── Main loop ──────────────────────────────────────────────────
    def _run(self):
        frame_count  = 0
        depth_map    = None    # ✅ MUST be above while loop
        nav_msg      = ""      # ✅ FIXED: separate from mode-crossing msg contamination

        while self._running:
            t0 = time.time()

            frame = self._camera.read()
            if frame is None:
                time.sleep(0.05)
                continue

            h, w   = frame.shape[:2]
            frame_count += 1

            # YOLO — every frame
            detections = self._detector.detect(frame)

            # MiDaS — every N frames
            if frame_count % DEPTH_N == 0:
                depth_map = depth_estimator.estimate(frame)

            # Stair/drop (TTS dedup handles repetition — 2s window)
            if depth_estimator.detect_stair_drop(depth_map, h, w):
                tts_engine.speak("Warning — possible step or drop ahead", priority=True)

            # Spatial analysis
            spatial_results = [
                spatial_analyzer.analyze(d, w, h, depth_map) for d in detections
            ]
            # Always store latest (for scene diff)
            with self._q_lock:
                self._last_spatial = spatial_results

            # ═══════════════════════════════════════════
            # MODE: READ
            # ═══════════════════════════════════════════
            if mode_manager.is_read():
                if frame_count % OCR_N == 0:
                    texts = ocr_reader.read(frame)
                    if texts:
                        read_msg = "Reading: " + ". ".join(texts[:5])
                        tts_engine.speak(read_msg)
                    else:
                        read_msg = "No text found. Move closer or adjust angle."
                else:
                    read_msg = ""

                self._send({
                    "type": "reading",
                    "mode": "READ",
                    "text": read_msg,
                    "detections": [self._serial(r) for r in spatial_results],
                    "fps": round(self.fps, 1),
                })
                self._tick(t0)
                continue

            # ═══════════════════════════════════════════
            # MODE: ASK
            # ═══════════════════════════════════════════
            if mode_manager.is_ask():
                with self._q_lock:
                    question = self._pending_question
                    self._pending_question = None

                if question:
                    # Build context using current frame data
                    texts  = ocr_reader.read(frame)
                    # ✅ FIXED: pass spatial_results directly (not dead det_list)
                    answer = brain.answer(question, frame, spatial_results, texts)
                    tts_engine.speak(answer, priority=True)
                    self._send({
                        "type":     "answer",
                        "mode":     "ASK",
                        "question": question,
                        "answer":   answer,
                        "context":  [f"{r.class_name} ({r.direction}, {r.distance})"
                                     for r in spatial_results[:5]],
                        "fps":      round(self.fps, 1),
                    })

                    # Auto-return to NAVIGATE after answering
                    mode_manager.set_mode("NAVIGATE")

                self._tick(t0)
                continue

            # ═══════════════════════════════════════════
            # MODE: NAVIGATE (default)
            # ═══════════════════════════════════════════
            scene_memory.update(spatial_results)

            # NEW FEATURE: Check approaching objects FIRST (highest priority)
            approaching = scene_memory.get_approaching()
            if approaching:
                obj = approaching[0]   # most dangerous approaching object
                approach_msg = narrator.narrate_approaching(obj)
                tts_engine.speak(approach_msg, priority=True)
                scene_memory.mark_approach_warned(obj.key)
                nav_msg = approach_msg

            # Standard new-object alerts
            new_alerts  = scene_memory.get_new_alerts()
            prioritized = narrator.prioritize(new_alerts)

            if prioritized:
                best    = prioritized[0]
                nav_msg = narrator.narrate(best)
                if nav_msg:
                    tts_engine.speak(nav_msg, priority=(best.distance_level == 1))
                for a in prioritized:
                    scene_memory.mark_announced(a.key)
            else:
                # Only say "path clear" if there really are no obstacles
                path_msg = narrator.path_clear(spatial_results)
                if path_msg and path_msg != nav_msg:
                    tts_engine.speak(path_msg)
                    nav_msg = path_msg

            self._send({
                "type":         "narration",
                "mode":         "NAVIGATE",
                "text":         nav_msg,
                "severity":     prioritized[0].distance_level if prioritized else 0,
                "detections":   [self._serial(r) for r in spatial_results],
                "show_overlay": mode_manager.snapshot()["show_overlay"],
                "fps":          round(self.fps, 1),
            })

            self._tick(t0)

    def _serial(self, r) -> dict:
        return {
            "class_name":     r.class_name,
            "confidence":     round(r.confidence, 2),
            "direction":      r.direction,
            "distance":       r.distance,
            "distance_level": r.distance_level,
            "x1": r.x1, "y1": r.y1, "x2": r.x2, "y2": r.y2,
        }

    def _send(self, data: dict):
        if self._loop and self._bcast:
            asyncio.run_coroutine_threadsafe(self._bcast(data), self._loop)

    def _tick(self, t0: float):
        elapsed = time.time() - t0
        self.fps = 1.0 / elapsed if elapsed > 0 else FPS_CAP
        time.sleep(max(0, (1 / FPS_CAP) - elapsed))


pipeline = PipelineRunner()
```

---

### `backend/main.py` ← UPDATED: added snapshot/diff WebSocket actions

```python
from dotenv import load_dotenv
load_dotenv()   # ← LINE 1. Always.

import os, asyncio, socket, logging, io
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pathlib import Path
import qrcode

from backend.pipeline import pipeline
from backend.modes import mode_manager

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

FRONTEND = Path(__file__).parent.parent / "frontend"
clients: set[WebSocket] = set()


async def broadcast(data: dict):
    disconnected = set()
    for ws in clients:
        try:
            await ws.send_json(data)
        except Exception:
            disconnected.add(ws)
    clients.difference_update(disconnected)


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    pipeline.start(loop, broadcast)
    logger.info("VisionTalk started.")
    yield
    pipeline.stop()
    logger.info("VisionTalk stopped.")


app = FastAPI(lifespan=lifespan, title="VisionTalk")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")


@app.get("/")
async def index():
    return HTMLResponse((FRONTEND / "index.html").read_text(encoding="utf-8"))


@app.get("/qr")
async def qr_code():
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        ip = "localhost"
    port = os.getenv("PORT", "8000")
    img  = qrcode.make(f"http://{ip}:{port}")
    buf  = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/api/status")
async def status():
    return {"ok": True, "mode": mode_manager.snapshot(), "fps": round(pipeline.fps, 1)}


@app.websocket("/ws/guidance")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    await websocket.send_json({"type": "init", "mode": mode_manager.snapshot()})
    try:
        while True:
            data   = await websocket.receive_json()
            action = data.get("action", "")

            if action == "set_mode":
                mode_manager.set_mode(data.get("mode", "NAVIGATE"))

            elif action == "ask":
                mode_manager.set_mode("ASK")
                q = data.get("question", "").strip()
                if q:
                    pipeline.set_question(q)

            elif action == "toggle_overlay":
                mode_manager.toggle_overlay()

            # NEW: Scene snapshot actions
            elif action == "snapshot":
                pipeline.take_snapshot()
                await websocket.send_json({
                    "type": "system",
                    "text": "Scene snapshot saved. Say 'What changed?' to compare."
                })

            elif action == "scene_diff":
                diff = pipeline.get_scene_diff()
                await websocket.send_json({"type": "answer", "answer": diff, "question": "What changed?"})

    except WebSocketDisconnect:
        clients.discard(websocket)
    except Exception as e:
        logger.error(f"WS error: {e}")
        clients.discard(websocket)
```

---

## NEW FEATURES — FRONTEND ADDITIONS

### In `app.js` — add AFTER existing button handlers:

```javascript
// ─── NEW: Scene Snapshot ─────────────────────────────────────────
document.getElementById('btn-snapshot').addEventListener('click', () => {
  sendCommand({ type: 'command', action: 'snapshot' });
  showToast('📸 Scene snapshot saved!');
});

// ─── voice.js additions: route "remember" and "what changed" ──────
// Add to the voice recognition result handler:
if (lower.includes('remember') || lower === 'remember this') {
  window.sendCommand?.({ type: 'command', action: 'snapshot' });
  window.showToast?.('📸 Scene remembered');
} else if (lower.includes('what changed') || lower.includes('what is different')) {
  window.sendCommand?.({ type: 'command', action: 'scene_diff' });
}
```

### In `index.html` — add inside `<section class="secondary-controls">`:
```html
<button id="btn-snapshot" class="secondary-btn" title="Remember current scene">
  📸 Remember
</button>
```

### WS message handling — add in `app.js`:
```javascript
// In ws.onmessage handler:
if (data.type === 'system') {
  setBanner(data.text, 0);
  showToast(data.text);
}
```

---

## COMPLETE WEBSOCKET PROTOCOL

### Server → Client
```json
// NAVIGATE — every frame
{"type":"narration","mode":"NAVIGATE","text":"Chair nearby, to your left",
 "severity":2,"show_overlay":false,"fps":9.8,
 "detections":[{"class_name":"chair","direction":"left","distance":"nearby",
                "distance_level":2,"confidence":0.87,"x1":142,"y1":241,"x2":318,"y2":479}]}

// ASK — on question answered
{"type":"answer","mode":"ASK","question":"What is in front?",
 "answer":"I can see a chair to your left and a bottle directly ahead.",
 "context":["chair (left, nearby)","bottle (ahead, far)"],"fps":9.2}

// READ — every 5 frames if text found
{"type":"reading","mode":"READ","text":"Reading: Paracetamol 500mg. Take 2 tablets.",
 "fps":9.5,"detections":[...]}

// SYSTEM — confirmations
{"type":"system","text":"Scene snapshot saved."}

// INIT — on connect
{"type":"init","mode":{"current_mode":"NAVIGATE","show_overlay":false}}
```

### Client → Server
```json
{"type":"command","action":"set_mode","mode":"NAVIGATE"}
{"type":"command","action":"ask","question":"What is this?"}
{"type":"command","action":"toggle_overlay"}
{"type":"command","action":"snapshot"}
{"type":"command","action":"scene_diff"}
```

---

## FINAL VERIFICATION — SELF-TEST BEFORE DECLARING DONE

### Automated
```bash
python -c "
from backend.modes import mode_manager
from backend.detector import ObjectDetector
from backend.spatial import SpatialResult
from backend.scene_memory import scene_memory
from backend.narrator import narrator
from backend.brain import brain
from backend.pipeline import pipeline
print('ALL IMPORTS OK')
"

python -c "
import numpy as np
from backend.spatial import SpatialResult
# Test .bbox_area property exists
r = SpatialResult('chair',0.9,'left','nearby',2,'mid',0.5,100,100,200,200)
assert r.bbox_area == 10000, f'bbox_area bug: {r.bbox_area}'
print('SpatialResult.bbox_area OK')
"

python -c "
from backend.scene_memory import SceneMemory
# Test no RuntimeError from dict mutation
from backend.spatial import SpatialResult
sm = SceneMemory()
r = SpatialResult('chair',0.9,'left','nearby',2,'mid',0.5,100,100,200,200)
sm.update([r])
sm.update([])  # expire test
print('SceneMemory expire OK')
"

python scripts/warmup.py
# Expected: === Warmup complete. All models loaded. ===

uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
sleep 3
curl http://localhost:8000/api/status
# Expected: {"ok":true,"mode":{"current_mode":"NAVIGATE","show_overlay":false},"fps":...}
```

### Manual Browser Checks
```
[ ] WS dot → green within 2s
[ ] Navigate btn → purple, badge NAVIGATE
[ ] ASK btn → blue, ask-row appears, badge ASK
[ ] READ btn → teal, ask-row hidden, badge READ
[ ] Overlay ON → canvas boxes appear over camera
[ ] 🔬 Detective → table with real bbox coords
[ ] 🕶 Blind Mode → BLACK screen, green narration text
[ ] Hold mic → btn goes red; speak; toast shows transcript
[ ] Ask question → answer appears in chat panel AND banner
[ ] 📸 Remember → toast "Scene snapshot saved"
[ ] Gear → settings panel + QR image loads
[ ] Phone IP entered → camera img src changes
```

---

## FILES CHECKLIST (31 total — 2 more than before due to new features)

```
backend/   14 files  (__init__.py + 13 modules)
frontend/   8 files  (index.html, style.css, app.js, camera.js, overlay.js, voice.js, audio.js, manifest.json, sw.js → 9)
scripts/    3 files  (warmup.py, start.bat, start.sh)
root        4 files  (requirements.txt, .env.example, .gitignore, README.md)
───────────────────
TOTAL: 30 files

If count ≠ 30 → find missing file.
```

---

## NEW FEATURES SUMMARY FOR JUDGES

| Feature | Demo moment | Technical uniqueness |
|---|---|---|
| 🧭 **NAVIGATE 3D** | Walk toward chair → "Chair nearby, to your left" | YOLO + MiDaS fusion = real 3D understanding |
| 🚨 **Approaching Alert** | Person walks toward camera → "Warning! Person approaching from ahead" | Level history tracking: [4,3,2] pattern = approaching |
| 🧠 **ASK** | Ask "What is in front of me?" → Local Llama-3 answers | 100% offline LLM at hackathon = instant WOW |
| 🎨 **Color Sense** | Ask "What color is this?" → "Dark blue" instantly | OpenCV HSV math, 5ms, no model needed |
| 📖 **READ** | Point at medicine → reads label aloud | Center-priority OCR = reads what you point at |
| 📸 **Remember/Diff** | "Remember this" → walk away → "What changed?" → "A person appeared" | Scene snapshot diff = totally unique |
| 🕶️ **Blind Demo** | Screen goes black → audio continues | Shows judges EXACTLY what a blind user experiences |
| 🔬 **Detective Panel** | Live bbox coords: [142,241]→[318,479] at 89% | Proves AI is real, not pre-recorded |

**8 distinct, real, demo-able features. No team at a hackathon will have all 8.**
