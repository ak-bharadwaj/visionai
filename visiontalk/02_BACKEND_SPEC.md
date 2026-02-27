# VisionTalk — Complete Backend Specification
## For Coding Agent: This file contains ALL Python code. Copy exactly.

---

## IRON RULES (memorize before writing a single line)

```
1.  threading.Queue   — ALWAYS (never asyncio.Queue in threaded code)
2.  threading.Lock    — in EVERY class accessed from >1 thread
3.  depth_map = None  — BEFORE the while loop, never inside it
4.  msg = ""          — BEFORE the while loop, never inside it
5.  asyncio.run_coroutine_threadsafe(coro, loop) — only way to call async from thread
6.  load_dotenv()     — FIRST LINE of main.py before ANY other import
7.  Scene memory TTL: expired=[k for k in dict if ...]; for k in expired: del dict[k]
8.  brain.py: ALWAYS catch subprocess.TimeoutExpired AND FileNotFoundError
9.  FastAPI: lifespan() context manager (NEVER @app.on_event — deprecated in 0.110+)
10. <img src="http://IP:8080/video"> for camera (never getUserMedia on HTTP)
```

---

## FILE STRUCTURE

```
visiontalk/
├── backend/
│   ├── __init__.py       ← empty file
│   ├── modes.py          ← Step 1
│   ├── camera.py         ← Step 2
│   ├── detector.py       ← Step 3
│   ├── depth.py          ← Step 4
│   ├── ocr.py            ← Step 5
│   ├── color_sense.py    ← Step 6 (new — color identification)
│   ├── spatial.py        ← Step 7
│   ├── scene_memory.py   ← Step 8
│   ├── narrator.py       ← Step 9
│   ├── brain.py          ← Step 10
│   ├── tts.py            ← Step 11
│   └── pipeline.py       ← Step 12
│   └── main.py           ← Step 13
├── frontend/             ← See 03_FRONTEND_SPEC.md
├── models/               ← gitignored, auto-created by detector.py
├── scripts/
│   ├── start.bat
│   ├── start.sh
│   └── warmup.py
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## requirements.txt

```
fastapi==0.110.0
uvicorn[standard]==0.27.1
websockets==12.0
python-multipart==0.0.9
aiofiles==23.2.1
opencv-python==4.9.0.80
ultralytics==8.1.0
onnxruntime==1.17.1
torch==2.2.0
torchvision==0.17.0
timm==0.9.16
paddleocr==2.7.3
pyttsx3==2.90
numpy==1.26.4
Pillow==10.2.0
python-dotenv==1.0.1
qrcode[pil]==7.4.2
```

Windows PaddleOCR (add to README):
```
pip install paddlepaddle -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
pip install paddleocr==2.7.3
```

---

## .env.example

```env
CAMERA_SOURCE=http://192.168.1.5:8080/video
HOST=0.0.0.0
PORT=8000
INFERENCE_FPS=10
INFERENCE_SIZE=320
CONF_THRESHOLD=0.45
IOU_THRESHOLD=0.45
DEPTH_ENABLED=true
DEPTH_EVERY_N_FRAMES=3
TTS_RATE=175
TTS_VOLUME=1.0
OLLAMA_MODEL=llama3:8b
LOG_LEVEL=INFO
```

---

## STEP 1: `backend/modes.py`

Purpose: Thread-safe mode state. Only one mode active at a time.
Modes: NAVIGATE (always-on narrator), ASK (visual Q&A), READ (OCR session)

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

# Module-level singleton — import this everywhere
mode_manager = ModeManager()
```

---

## STEP 2: `backend/camera.py`

Purpose: Thread-safe MJPEG buffer. Captures frames continuously in background.
Note: Source is URL like "http://192.168.1.5:8080/video" or integer 0 for webcam.

```python
import cv2, threading, time, numpy as np, os, logging

logger = logging.getLogger(__name__)

class CameraStream:
    def __init__(self, source):
        # Convert "0" string to integer 0 for local webcam
        self._source = int(source) if str(source).isdigit() else source
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._cap: cv2.VideoCapture | None = None

    def start(self) -> "CameraStream":
        self._cap = cv2.VideoCapture(self._source)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize latency

        if not self._cap.isOpened():
            logger.warning(f"Cannot open '{self._source}'. Falling back to webcam 0.")
            self._cap = cv2.VideoCapture(0)
            if not self._cap.isOpened():
                raise RuntimeError("No camera available. Set CAMERA_SOURCE in .env")

        self._running = True
        t = threading.Thread(target=self._loop, daemon=True, name="CameraThread")
        t.start()
        logger.info(f"Camera started: {self._source}")
        return self

    def _loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame   # store reference, not copy
            else:
                time.sleep(0.01)  # yield if no frame

    def read(self) -> np.ndarray | None:
        """Returns a copy of the latest frame, or None if none yet."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        time.sleep(0.1)
        if self._cap:
            self._cap.release()
        logger.info("Camera stopped.")

def get_camera() -> CameraStream:
    """Factory: reads CAMERA_SOURCE from env, returns started stream."""
    source = os.getenv("CAMERA_SOURCE", "0")
    return CameraStream(source).start()
```

---

## STEP 3: `backend/detector.py`

Purpose: YOLOv8n ONNX inference. Auto-downloads model if missing.
Output: List[Detection] with class_name, confidence, bounding box.

```python
import cv2, numpy as np, onnxruntime as ort, urllib.request, logging
from pathlib import Path
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

MODEL_DIR  = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "yolov8n.onnx"
MODEL_URL  = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.onnx"

# All 80 COCO classes in exact order — index matches YOLOv8 class_id output
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
]


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def bbox_area(self) -> int:   return (self.x2 - self.x1) * (self.y2 - self.y1)
    @property
    def center_x(self) -> int:   return (self.x1 + self.x2) // 2
    @property
    def center_y(self) -> int:   return (self.y1 + self.y2) // 2
    @property
    def bottom_y(self) -> int:   return self.y2


class ObjectDetector:
    SIZE = 320    # input resolution (square)
    CONF = 0.45   # minimum confidence
    IOU  = 0.45   # NMS IOU threshold

    def __init__(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if not MODEL_PATH.exists():
            logger.info("Downloading YOLOv8n ONNX (~6 MB)...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            logger.info("Model downloaded.")
        self._sess = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self._input_name = self._sess.get_inputs()[0].name
        logger.info(f"Detector ready. Provider: {self._sess.get_providers()}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if frame is None:
            return []
        h, w = frame.shape[:2]

        # Preprocess: resize, BGR→RGB, normalize, add batch dim
        img = cv2.resize(frame, (self.SIZE, self.SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis]  # → [1, 3, 320, 320]

        # Inference
        raw = self._sess.run(None, {self._input_name: img})[0]  # [1, 84, 8400]
        preds = raw[0].T  # → [8400, 84]

        # Parse boxes + class scores
        boxes_cxcywh = preds[:, :4]
        class_scores = preds[:, 4:]     # [8400, 80]
        class_ids    = np.argmax(class_scores, axis=1)
        confs        = class_scores[np.arange(len(class_ids)), class_ids]

        # Filter by confidence
        keep = confs >= self.CONF
        if not keep.any():
            return []
        boxes_cxcywh = boxes_cxcywh[keep]
        class_ids    = class_ids[keep]
        confs        = confs[keep]

        # Convert cx,cy,w,h → x1,y1,x2,y2 scaled to original frame
        sx, sy = w / self.SIZE, h / self.SIZE
        x1 = np.clip((boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2) * sx, 0, w - 1).astype(int)
        y1 = np.clip((boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2) * sy, 0, h - 1).astype(int)
        x2 = np.clip((boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2) * sx, 0, w - 1).astype(int)
        y2 = np.clip((boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2) * sy, 0, h - 1).astype(int)

        # NMS
        boxes_xywh = [[int(x1[i]), int(y1[i]), int(x2[i]-x1[i]), int(y2[i]-y1[i])]
                      for i in range(len(x1))]
        idxs = cv2.dnn.NMSBoxes(boxes_xywh, confs.tolist(), self.CONF, self.IOU)
        if len(idxs) == 0:
            return []

        results = []
        for i in idxs.flatten():
            cid = int(class_ids[i])
            results.append(Detection(
                class_id   = cid,
                class_name = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else "object",
                confidence = float(confs[i]),
                x1=int(x1[i]), y1=int(y1[i]),
                x2=int(x2[i]), y2=int(y2[i]),
            ))
        return results
```

---

## STEP 4: `backend/depth.py`

Purpose: MiDaS monocular depth. Returns normalized map (1.0=closest). Also detects stairs.

```python
import threading, logging, numpy as np, cv2

logger = logging.getLogger(__name__)


class DepthEstimator:
    def __init__(self):
        self._model     = None
        self._transform = None
        self._lock      = threading.Lock()
        self._loaded    = False
        self._device    = "cpu"

    def load(self):
        """Call once at startup. Non-fatal if torch unavailable."""
        try:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small",
                trust_repo=True, verbose=False
            )
            self._model.to(self._device).eval()
            self._transform = torch.hub.load(
                "intel-isl/MiDaS", "transforms", trust_repo=True, verbose=False
            ).small_transform
            self._loaded = True
            logger.info(f"MiDaS loaded on {self._device}")
        except Exception as e:
            logger.warning(f"MiDaS failed to load: {e}. Depth disabled.")
            self._loaded = False

    def estimate(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Returns float32 array same size as frame.
        Values: 0.0=far, 1.0=closest.
        Returns None if not loaded or on error.
        """
        if not self._loaded or frame is None:
            return None
        try:
            import torch
            with self._lock:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                inp = self._transform(rgb).to(self._device)
                with torch.no_grad():
                    pred = self._model(inp)
                    pred = torch.nn.functional.interpolate(
                        pred.unsqueeze(1),
                        size=frame.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze().cpu().numpy()
                mn, mx = pred.min(), pred.max()
                if mx - mn > 1e-8:
                    pred = (pred - mn) / (mx - mn)
                return pred.astype(np.float32)
        except Exception as e:
            logger.error(f"Depth estimate error: {e}")
            return None

    def get_region_depth(self, depth_map, x1, y1, x2, y2) -> float:
        """Mean depth in a bounding box region. Returns 0.0 if unavailable."""
        if depth_map is None:
            return 0.0
        y1 = max(0, y1); y2 = min(depth_map.shape[0], y2)
        x1 = max(0, x1); x2 = min(depth_map.shape[1], x2)
        region = depth_map[y1:y2, x1:x2]
        return float(np.mean(region)) if region.size > 0 else 0.0

    def detect_stair_drop(self, depth_map, frame_h: int, frame_w: int) -> bool:
        """
        Detect a sudden depth gradient change in lower-center zone.
        High gradient std = uneven surface = possible stair or ledge ahead.
        Zone: bottom 30% of frame, middle 40% horizontally.
        Threshold: std > 0.25 = warning.
        """
        if depth_map is None:
            return False
        y0 = int(frame_h * 0.70)
        x0 = int(frame_w * 0.30)
        x1 = int(frame_w * 0.70)
        zone = depth_map[y0:, x0:x1]
        if zone.size == 0:
            return False
        return float(np.std(np.gradient(zone, axis=0))) > 0.25


depth_estimator = DepthEstimator()  # module-level singleton
```

---

## STEP 5: `backend/ocr.py`

Purpose: PaddleOCR text extraction. Used in READ mode and in ASK context building.

```python
import logging, threading, numpy as np
from typing import List

logger = logging.getLogger(__name__)


class OCRReader:
    def __init__(self):
        self._ocr    = None
        self._lock   = threading.Lock()
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        try:
            from paddleocr import PaddleOCR
            # show_log=False suppresses spam; use_gpu=False for CPU-only
            self._ocr = PaddleOCR(use_angle_cls=True, lang="en",
                                   show_log=False, use_gpu=False)
            self._loaded = True
            logger.info("PaddleOCR loaded.")
        except Exception as e:
            logger.warning(f"PaddleOCR load failed: {e}. READ mode disabled.")

    def read(self, frame: np.ndarray) -> List[str]:
        """
        Returns list of text strings found (center-priority order).
        Empty list on error or no text.
        """
        self._ensure_loaded()
        if not self._loaded or frame is None:
            return []
        try:
            with self._lock:
                result = self._ocr.ocr(frame, cls=True)
            if not result or result == [None]:
                return []

            frame_cx = frame.shape[1] / 2
            frame_cy = frame.shape[0] / 2
            items = []
            for block in result:
                if not block:
                    continue
                for line in block:
                    # line = [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], [text, confidence]]
                    text, conf = line[1][0], line[1][1]
                    if conf > 0.60 and text.strip():
                        # compute center of bounding box
                        pts = line[0]
                        bx = sum(p[0] for p in pts) / 4
                        by = sum(p[1] for p in pts) / 4
                        # distance from frame center (for priority sorting)
                        dist = ((bx - frame_cx)**2 + (by - frame_cy)**2) ** 0.5
                        items.append((dist, text.strip()))

            # sort: center-of-frame text comes first
            items.sort(key=lambda x: x[0])
            return [t for _, t in items]
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return []


ocr_reader = OCRReader()  # module-level singleton
```

---

## STEP 6: `backend/color_sense.py`

Purpose: New unique feature — identify dominant color in a frame region.
Used in ASK mode when question contains "color" or "colour".

```python
import cv2, numpy as np, logging
from backend.detector import Detection
from typing import List

logger = logging.getLogger(__name__)

# HSV hue ranges → color name
# Hue is 0-179 in OpenCV (half the standard 0-360)
COLOR_RANGES = [
    (  0,  10, "red"),
    ( 10,  25, "orange"),
    ( 25,  35, "yellow"),
    ( 35,  85, "green"),
    ( 85, 100, "cyan"),
    (100, 130, "blue"),
    (130, 145, "purple"),
    (145, 160, "pink"),
    (160, 179, "red"),   # upper red wraps around
]

def _hue_to_name(hue: float) -> str:
    for lo, hi, name in COLOR_RANGES:
        if lo <= hue <= hi:
            return name
    return "red"  # 0 and 179 both wrap to red

def get_dominant_color(frame: np.ndarray, det: Detection | None = None) -> str:
    """
    Returns a human readable color string.
    If det is provided, analyzes that bounding box region.
    Otherwise analyzes center 40% of frame.
    
    Returns: e.g. "dark blue", "bright red", "light gray"
    """
    try:
        if det is not None:
            region = frame[det.y1:det.y2, det.x1:det.x2]
        else:
            h, w = frame.shape[:2]
            y0, y1_end = int(h * 0.30), int(h * 0.70)
            x0, x1_end = int(w * 0.30), int(w * 0.70)
            region = frame[y0:y1_end, x0:x1_end]

        if region.size == 0:
            return "unknown"

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        h_vals = hsv[:, :, 0].flatten()
        s_vals = hsv[:, :, 1].flatten()
        v_vals = hsv[:, :, 2].flatten()

        mean_s = float(np.mean(s_vals))
        mean_v = float(np.mean(v_vals))

        # Gray / black / white detection (low saturation)
        if mean_s < 40:
            if mean_v < 60:   return "black"
            elif mean_v > 180: return "white"
            else:              return "gray"

        mean_h = float(np.median(h_vals))  # median more robust than mean
        base_color = _hue_to_name(mean_h)

        # Add lightness/darkness qualifier
        if mean_v < 80:   prefix = "dark "
        elif mean_v > 200: prefix = "light "
        else:              prefix = ""

        return f"{prefix}{base_color}"
    except Exception as e:
        logger.error(f"Color sense error: {e}")
        return "unknown"


def answer_color_question(frame: np.ndarray, detections: list) -> str:
    """
    Returns a spoken color answer.
    Uses first/largest detection if available, else center region.
    """
    det = None
    if detections:
        # find the detection with the largest bounding box
        det = max(detections, key=lambda d: d.bbox_area if hasattr(d, 'bbox_area') else 0)
    
    color = get_dominant_color(frame, det)
    if det:
        return f"The {det.class_name} appears to be {color}."
    return f"The object in front appears to be {color}."
```

---

## STEP 7: `backend/spatial.py`

Purpose: Converts a Detection + depth map into human-readable spatial data.
Output: SpatialResult with direction, distance, zone, key.

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
    depth_score:    float  # 0..1 from MiDaS (0.0 if unavailable)
    x1: int; y1: int; x2: int; y2: int

    @property
    def key(self) -> str:
        """Unique key for scene memory deduplication."""
        return f"{self.class_name}_{self.direction}"


class SpatialAnalyzer:
    def analyze(self, det: Detection, fw: int, fh: int,
                depth_map) -> SpatialResult:
        # ── Direction: horizontal thirds ──
        cx_ratio = det.center_x / fw
        if cx_ratio < 0.20:    direction = "far left"
        elif cx_ratio < 0.40:  direction = "left"
        elif cx_ratio < 0.60:  direction = "ahead"
        elif cx_ratio < 0.75:  direction = "right"
        else:                  direction = "far right"

        # ── Distance step 1: bounding box size heuristic ──
        area_ratio   = det.bbox_area / (fw * fh)
        bottom_ratio = det.bottom_y / fh
        if area_ratio > 0.15 and bottom_ratio > 0.70:
            level = 1; dist_str = "very close"
        elif area_ratio > 0.06:
            level = 2; dist_str = "nearby"
        elif area_ratio > 0.02:
            level = 3; dist_str = "ahead"
        else:
            level = 4; dist_str = "far"

        # ── Distance step 2: MiDaS depth override ──
        d_score = depth_estimator.get_region_depth(
            depth_map, det.x1, det.y1, det.x2, det.y2)
        if d_score > 0.80:
            level = min(level, 1); dist_str = "very close"
        elif d_score > 0.60:
            level = min(level, 2); dist_str = "nearby"

        # ── Zone: vertical position ──
        top_ratio = det.y1 / fh
        if top_ratio < 0.30:     zone = "aerial"
        elif top_ratio < 0.65:   zone = "mid"
        else:                    zone = "ground"

        return SpatialResult(
            class_name=det.class_name, confidence=det.confidence,
            direction=direction, distance=dist_str, distance_level=level,
            zone=zone, depth_score=d_score,
            x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2,
        )


spatial_analyzer = SpatialAnalyzer()  # module-level singleton
```

---

## STEP 8: `backend/scene_memory.py`

Purpose: Track which objects have been announced. Only return NEW or CHANGED alerts.
Thread-safe. TTL = 3 seconds. Never mutate dict while iterating.

```python
import time, threading, logging
from dataclasses import dataclass
from typing import List
from backend.spatial import SpatialResult

logger = logging.getLogger(__name__)


@dataclass
class SceneEntry:
    result: SpatialResult
    first_seen: float
    last_seen: float
    announced: bool
    last_level: int   # previous distance_level for change detection


class SceneMemory:
    TTL = 3.0  # seconds before entry expires

    def __init__(self):
        self._entries: dict[str, SceneEntry] = {}
        self._lock = threading.Lock()

    def update(self, results: List[SpatialResult]):
        """Update memory with current frame's spatial results."""
        now = time.time()
        with self._lock:
            # Update or insert
            for r in results:
                k = r.key
                if k in self._entries:
                    e = self._entries[k]
                    e.last_seen = now
                    e.result = r
                    # Re-announce if distance changed significantly
                    if abs(r.distance_level - e.last_level) >= 1:
                        e.announced = False
                        e.last_level = r.distance_level
                else:
                    self._entries[k] = SceneEntry(
                        result=r, first_seen=now, last_seen=now,
                        announced=False, last_level=r.distance_level
                    )

            # ⚠️ IMPORTANT: build expired list FIRST, then delete.
            # Never do: for k in dict: del dict[k] — raises RuntimeError
            expired = [k for k, e in self._entries.items()
                       if now - e.last_seen > self.TTL]
            for k in expired:
                del self._entries[k]

    def get_new_alerts(self) -> List[SpatialResult]:
        """Returns objects not yet announced this cycle."""
        with self._lock:
            return [e.result for e in self._entries.values() if not e.announced]

    def mark_announced(self, key: str):
        with self._lock:
            if key in self._entries:
                self._entries[key].announced = True

    def clear(self):
        with self._lock:
            self._entries.clear()


scene_memory = SceneMemory()
```

---

## STEP 9: `backend/narrator.py`

Purpose: Convert SpatialResult into spoken natural language.
Contains: risk ranking, priority filtering, NLG templates. No hallucination.

```python
from backend.spatial import SpatialResult
from typing import List

# Risk score → priority (higher = more important to announce first)
RISK_SCORE: dict[str, int] = {
    "person": 10, "car": 10, "motorcycle": 10, "bicycle": 10,
    "bus": 10, "truck": 10, "train": 10,
    "chair": 7, "couch": 7, "dining table": 6, "bench": 6,
    "door": 5, "suitcase": 5, "bed": 5,
    "toilet": 4, "sink": 4, "refrigerator": 4,
}
DEFAULT_RISK = 3

# Max distance_level to announce for each class (4 = always, 1 = only if very close)
MIN_ALERT_LEVEL: dict[str, int] = {
    "person": 4, "car": 4, "motorcycle": 4, "bicycle": 4,
    "bus": 4, "truck": 4, "train": 4,
    "chair": 3, "couch": 3, "dining table": 3, "bench": 3, "door": 3,
}
DEFAULT_MIN_LEVEL = 2

# Complete NLG templates — all possible narration strings
NAVIGATE_TEMPLATES: dict[int, str] = {
    1: "Stop! {cls} directly {dir}",             # very close
    2: "{Cls} nearby, on your {dir}",             # nearby
    3: "{Cls} {dir}, a few steps ahead",          # ahead
    4: "",                                         # suppress — too far
}


class Narrator:
    def prioritize(self, alerts: List[SpatialResult]) -> List[SpatialResult]:
        """Filter and sort alerts by risk and distance."""
        def should_announce(r: SpatialResult) -> bool:
            max_level = MIN_ALERT_LEVEL.get(r.class_name, DEFAULT_MIN_LEVEL)
            return r.distance_level <= max_level

        filtered = [r for r in alerts if should_announce(r)]
        return sorted(
            filtered,
            key=lambda r: (-RISK_SCORE.get(r.class_name, DEFAULT_RISK), r.distance_level)
        )

    def narrate(self, result: SpatialResult) -> str:
        """Convert a SpatialResult to a spoken string. Returns '' if nothing to say."""
        tmpl = NAVIGATE_TEMPLATES.get(result.distance_level, "")
        if not tmpl:
            return ""
        return tmpl.format(
            cls=result.class_name,
            Cls=result.class_name.title(),
            dir=result.direction,
        )

    def path_clear(self, results: List[SpatialResult]) -> str | None:
        """Returns 'Path clear ahead' if no obstacles directly in path. Else None."""
        blocking = [r for r in results
                    if r.direction == "ahead" and r.distance_level <= 2]
        return "Path clear ahead" if not blocking else None


narrator = Narrator()
```

---

## STEP 10: `backend/brain.py`

Purpose: Local LLM reasoning for ASK mode.
Uses Ollama subprocess. Color questions routed to color_sense.
ALWAYS has a fallback — never crashes if Ollama not installed.

```python
import subprocess, logging, os
from typing import List

logger = logging.getLogger(__name__)

COLOR_KEYWORDS = {"color", "colour", "shade", "hue", "what color"}


class Brain:
    """
    On-device AI reasoning for ASK mode.
    Uses Ollama (run `ollama serve` + `ollama pull llama3:8b` before demo).
    100% offline — no internet required.
    """

    def __init__(self):
        self._model = os.getenv("OLLAMA_MODEL", "llama3:8b")

    def answer(self, question: str, frame,
               detections: list, texts: List[str]) -> str:
        """
        Main entry point for answering a user question.
        Routes color questions separately.
        Falls back gracefully if Ollama fails.
        """
        q_lower = question.lower()

        # Route color questions to dedicated color module (no LLM needed, instant)
        if any(k in q_lower for k in COLOR_KEYWORDS):
            from backend.color_sense import answer_color_question
            return answer_color_question(frame, detections)

        # Build scene context string for LLM
        obj_desc   = self._describe_objects(detections)
        text_desc  = ", ".join(texts[:5]) or "no visible text"

        prompt = (
            "You are an AI assistant helping a visually impaired person.\n"
            f"Scene (from camera): {obj_desc}\n"
            f"Visible text: {text_desc}\n"
            f"Question: {question}\n\n"
            "Instructions: Answer in ONE short sentence. Use ONLY what is described "
            "above. Do not guess. If unsure, say what you can see clearly."
        )

        return self._run_ollama(prompt, obj_desc, texts)

    def _describe_objects(self, detections: list) -> str:
        if not detections:
            return "nothing specific detected"
        parts = []
        for d in detections[:8]:  # cap at 8 to keep prompt short
            name      = d.class_name if hasattr(d, "class_name") else str(d)
            direction = d.direction  if hasattr(d, "direction")  else ""
            distance  = d.distance   if hasattr(d, "distance")   else ""
            parts.append(f"{name} ({direction}, {distance})")
        return "; ".join(parts)

    def _run_ollama(self, prompt: str, obj_desc: str, texts: List[str]) -> str:
        """
        Run Ollama CLI subprocess. 
        Catches: FileNotFoundError (not installed), TimeoutExpired, any other error.
        Always returns a string — never raises.
        """
        try:
            result = subprocess.run(
                ["ollama", "run", self._model, prompt],
                capture_output=True, text=True, timeout=10
            )
            answer = result.stdout.strip()
            if answer:
                return answer
            logger.warning("Ollama returned empty output. Using fallback.")
        except FileNotFoundError:
            logger.warning("Ollama not installed. Using rule-based fallback.")
        except subprocess.TimeoutExpired:
            logger.warning("Ollama timed out. Using fallback.")
        except Exception as e:
            logger.error(f"Ollama error: {e}. Using fallback.")

        # Fallback: structured description (no LLM needed)
        return self._fallback(obj_desc, texts)

    def _fallback(self, obj_desc: str, texts: List[str]) -> str:
        parts = []
        if obj_desc and obj_desc != "nothing specific detected":
            parts.append(f"I can see: {obj_desc}")
        if texts:
            parts.append(f"Text in view: {', '.join(texts[:3])}")
        return ". ".join(parts) if parts else \
               "I can see the scene but nothing specific stands out."


brain = Brain()
```

---

## STEP 11: `backend/tts.py`

Purpose: Thread-safe TTS. Uses threading.Queue (NOT asyncio.Queue).
Dedup: same text within 2s is dropped. Priority = clear queue first.

```python
import queue, threading, time, logging, os

logger = logging.getLogger(__name__)


class TTSEngine:
    DEDUP_WINDOW = 2.0  # seconds

    def __init__(self):
        self._q:      queue.Queue      = queue.Queue(maxsize=5)
        self._last:   dict[str, float] = {}
        self._engine                   = None

    def start(self):
        """Initialize pyttsx3 and start worker thread."""
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate",   int(os.getenv("TTS_RATE",   "175")))
            self._engine.setProperty("volume", float(os.getenv("TTS_VOLUME","1.0")))
            logger.info("TTS engine ready.")
        except Exception as e:
            logger.warning(f"TTS init failed: {e}. Voice output disabled.")
        threading.Thread(target=self._worker, daemon=True, name="TTSThread").start()

    def speak(self, text: str, priority: bool = False):
        """
        Queue text for speech.
        priority=True: clear queue first (for urgent alerts, level-1 distance).
        Dedup: drops if same text was spoken within DEDUP_WINDOW seconds.
        """
        if not text:
            return
        now = time.time()
        if text in self._last and now - self._last[text] < self.DEDUP_WINDOW:
            return  # already said this recently

        if priority:
            # Clear pending low-priority items
            while not self._q.empty():
                try: self._q.get_nowait()
                except queue.Empty: break

        try:
            self._q.put_nowait(text)
        except queue.Full:
            pass  # drop if queue full — never block the pipeline

        self._last[text] = now

    def _worker(self):
        while True:
            text = self._q.get()
            if self._engine:
                try:
                    self._engine.say(text)
                    self._engine.runAndWait()
                except Exception as e:
                    logger.error(f"TTS speak error: {e}")
            self._q.task_done()


tts_engine = TTSEngine()
```

---

## STEP 12: `backend/pipeline.py`

Purpose: Background thread running the CV inference loop.
Handles all 3 modes. Cross-thread WS broadcast via run_coroutine_threadsafe.

```python
import time, threading, asyncio, logging
from dataclasses import asdict
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
OCR_N   = 5   # run OCR every 5 frames in READ mode
FPS_CAP = int(__import__("os").getenv("INFERENCE_FPS", "10"))


class PipelineRunner:
    def __init__(self):
        self._running    = False
        self._camera     = None
        self._detector   = None
        self._loop:  asyncio.AbstractEventLoop | None = None
        self._bcast: Callable | None = None
        self.fps = 0.0
        self._pending_question: str | None = None
        self._pending_frame:    np.ndarray | None = None
        self._q_lock = threading.Lock()

    def set_question(self, question: str):
        """Called from async WS handler. Thread-safe."""
        with self._q_lock:
            self._pending_question = question

    def start(self, event_loop: asyncio.AbstractEventLoop, broadcast_fn: Callable):
        self._camera   = get_camera()
        self._detector = ObjectDetector()
        depth_estimator.load()
        tts_engine.start()
        self._loop  = event_loop
        self._bcast = broadcast_fn
        self._running = True
        threading.Thread(
            target=self._run, daemon=True, name="PipelineThread"
        ).start()
        logger.info("Pipeline started.")

    def stop(self):
        self._running = False
        if self._camera:
            self._camera.stop()

    def _run(self):
        frame_count = 0
        depth_map   = None   # ← MUST initialize before loop
        msg         = ""     # ← MUST initialize before loop

        while self._running:
            t0 = time.time()

            frame = self._camera.read()
            if frame is None:
                time.sleep(0.05)
                continue

            h, w = frame.shape[:2]
            frame_count += 1

            # Always detect (every frame)
            detections = self._detector.detect(frame)

            # Depth every N frames
            if frame_count % DEPTH_N == 0:
                depth_map = depth_estimator.estimate(frame)

            # Stair/drop check (every frame if depth available)
            if depth_estimator.detect_stair_drop(depth_map, h, w):
                tts_engine.speak("Warning — possible step or drop ahead", priority=True)

            spatial_results = [
                spatial_analyzer.analyze(d, w, h, depth_map) for d in detections
            ]

            # ═══════════════════════════════
            # MODE: READ
            # ═══════════════════════════════
            if mode_manager.is_read():
                if frame_count % OCR_N == 0:
                    texts = ocr_reader.read(frame)
                    if texts:
                        reading_text = ". ".join(texts[:5])
                        msg = "Reading: " + reading_text
                        tts_engine.speak(msg)
                    else:
                        msg = "No text found"
                self._send({
                    "type": "reading",
                    "mode": "READ",
                    "text": msg,
                    "detections": [self._serial(r) for r in spatial_results],
                    "fps": round(self.fps, 1),
                })
                self._tick(t0)
                continue

            # ═══════════════════════════════
            # MODE: ASK
            # ═══════════════════════════════
            if mode_manager.is_ask():
                with self._q_lock:
                    question = self._pending_question
                    self._pending_question = None

                if question:
                    # Build context on current frame
                    texts    = ocr_reader.read(frame)
                    det_list = [{"class_name": r.class_name,
                                 "direction":  r.direction,
                                 "distance":   r.distance,
                                 "bbox_area":  r.x2*r.y2}  # for color_sense
                                for r in spatial_results]
                    # Get detections with .bbox_area for color_sense
                    answer = brain.answer(question, frame, spatial_results, texts)
                    msg = answer
                    tts_engine.speak(answer, priority=True)
                    self._send({
                        "type":     "answer",
                        "mode":     "ASK",
                        "question": question,
                        "answer":   answer,
                        "context":  [f"{r.class_name} ({r.direction})" for r in spatial_results[:5]],
                        "fps":      round(self.fps, 1),
                    })
                self._tick(t0)
                continue

            # ═══════════════════════════════
            # MODE: NAVIGATE (default)
            # ═══════════════════════════════
            scene_memory.update(spatial_results)
            new_alerts = scene_memory.get_new_alerts()
            prioritized = narrator.prioritize(new_alerts)

            if prioritized:
                best = prioritized[0]
                msg  = narrator.narrate(best)
                if msg:
                    tts_engine.speak(msg, priority=(best.distance_level == 1))
                    for a in prioritized:
                        scene_memory.mark_announced(a.key)
            else:
                path_msg = narrator.path_clear(spatial_results)
                if path_msg and path_msg != msg:
                    tts_engine.speak(path_msg)
                    msg = path_msg

            self._send({
                "type":         "narration",
                "mode":         "NAVIGATE",
                "text":         msg,
                "severity":     prioritized[0].distance_level if prioritized else 0,
                "detections":   [self._serial(r) for r in spatial_results],
                "show_overlay": mode_manager.snapshot()["show_overlay"],
                "fps":          round(self.fps, 1),
            })

            self._tick(t0)

    def _serial(self, r) -> dict:
        """Convert SpatialResult to JSON-serializable dict."""
        return {
            "class_name":    r.class_name,
            "confidence":    round(r.confidence, 2),
            "direction":     r.direction,
            "distance":      r.distance,
            "distance_level": r.distance_level,
            "x1": r.x1, "y1": r.y1, "x2": r.x2, "y2": r.y2,
        }

    def _send(self, data: dict):
        """Thread-safe async broadcast."""
        if self._loop and self._bcast:
            asyncio.run_coroutine_threadsafe(self._bcast(data), self._loop)

    def _tick(self, t0: float):
        """Track FPS and sleep to maintain target rate."""
        elapsed = time.time() - t0
        self.fps = 1.0 / elapsed if elapsed > 0 else FPS_CAP
        time.sleep(max(0, (1 / FPS_CAP) - elapsed))


pipeline = PipelineRunner()
```

---

## STEP 13: `backend/main.py`

Purpose: FastAPI app. WS hub. QR code. Static serving.
CRITICAL: load_dotenv() is the VERY FIRST LINE before any other import.

```python
from dotenv import load_dotenv
load_dotenv()   # ← MUST be first — reads .env before all backend imports

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
    """Send JSON to all connected WebSocket clients. Remove disconnected ones."""
    disconnected = set()
    for ws in clients:
        try:
            await ws.send_json(data)
        except Exception:
            disconnected.add(ws)
    clients.difference_update(disconnected)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start pipeline on startup. Stop on shutdown."""
    loop = asyncio.get_running_loop()
    pipeline.start(loop, broadcast)
    logger.info("VisionTalk started.")
    yield
    pipeline.stop()
    logger.info("VisionTalk stopped.")


app = FastAPI(lifespan=lifespan, title="VisionTalk")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Required — phone browser is a different origin
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")


@app.get("/")
async def index():
    return HTMLResponse((FRONTEND / "index.html").read_text(encoding="utf-8"))


@app.get("/qr")
async def qr_code():
    """Returns QR code PNG pointing to this server's LAN URL."""
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        ip = "localhost"
    port = os.getenv("PORT", "8000")
    url  = f"http://{ip}:{port}"
    img  = qrcode.make(url)
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
    # Send initial state
    await websocket.send_json({
        "type": "init",
        "mode": mode_manager.snapshot()
    })
    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action", "")

            if action == "set_mode":
                mode = data.get("mode", "NAVIGATE")
                mode_manager.set_mode(mode)

            elif action == "ask":
                mode_manager.set_mode("ASK")
                question = data.get("question", "").strip()
                if question:
                    pipeline.set_question(question)

            elif action == "toggle_overlay":
                mode_manager.toggle_overlay()

    except WebSocketDisconnect:
        clients.discard(websocket)
    except Exception as e:
        logger.error(f"WS error: {e}")
        clients.discard(websocket)
```

---

## SCRIPTS

### `scripts/warmup.py`
```python
"""Run this before the demo to pre-load all models."""
import time, sys, numpy as np
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv()

print("[VisionTalk Warmup] Starting...")
fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

print("  [1/3] Loading YOLOv8n...")
t = time.time()
from backend.detector import ObjectDetector
det = ObjectDetector()
det.detect(fake_frame)
print(f"        Ready in {time.time()-t:.1f}s")

print("  [2/3] Loading MiDaS depth...")
t = time.time()
from backend.depth import depth_estimator
depth_estimator.load()
print(f"        Ready in {time.time()-t:.1f}s")

print("  [3/3] Loading PaddleOCR...")
t = time.time()
from backend.ocr import ocr_reader
ocr_reader.read(fake_frame)
print(f"        Ready in {time.time()-t:.1f}s")

print("\n=== Warmup complete. All models loaded. Run `start.bat` now. ===")
```

### `scripts/start.bat`
```bat
@echo off
echo [VisionTalk] Starting...

if not exist venv (
    echo [VisionTalk] Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo [VisionTalk] Installing dependencies...
pip install -r requirements.txt --quiet

if not exist .env (
    copy .env.example .env
    echo [VisionTalk] Created .env — edit CAMERA_SOURCE with your phone's IP
)

echo [VisionTalk] Starting server at http://0.0.0.0:8000
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### `scripts/start.sh`
```bash
#!/bin/bash
set -e
[ ! -d venv ] && python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt --quiet
[ ! -f .env ] && cp .env.example .env
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## .gitignore
```
venv/
models/
*.pyc
__pycache__/
.env
*.log
.DS_Store
dist/
```
