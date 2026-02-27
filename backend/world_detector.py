"""
world_detector.py — YOLOWorld open-vocabulary secondary detector.
Runs every 5th frame to detect college-room objects missing from COCO-80.
Falls back gracefully if YOLOWorld is not installed.
"""
import logging, threading
import numpy as np

logger = logging.getLogger(__name__)

# College room objects NOT in COCO-80
EXTRA_CLASSES = [
    "door",
    "open door",
    "window",
    "whiteboard",
    "blackboard",
    "staircase",
    "stairs",
    "projector",
    "ceiling fan",
    "light switch",
    "notice board",
    "trash bin",
    "fire extinguisher",
    "exit sign",
]

WORLD_N = 5   # run every N frames (same cadence as OCR)


class WorldDetector:
    """
    YOLOWorld open-vocabulary second-pass detector.
    Detects college-room/indoor objects missing from primary YOLO COCO-80.
    Thread-safe. Loads lazily in pipeline.start().
    """

    def __init__(self):
        self._model  = None
        self._lock   = threading.Lock()
        self._loaded = False

    def load(self):
        """Load YOLOWorld if available, otherwise silently disable extra detection."""
        if self._loaded:
            return
        from ultralytics import YOLO as _YOLO
        # Try YOLOWorld class first (ultralytics >= 8.1.2)
        loaded = False
        try:
            from ultralytics import YOLOWorld as _YW
            self._model = _YW("yolov8s-worldv2.pt")
            if hasattr(self._model, "set_classes"):
                self._model.set_classes(EXTRA_CLASSES)
            loaded = True
        except Exception:
            pass

        # Try loading worldv2 weights with plain YOLO (architecture may still work)
        if not loaded:
            try:
                self._model = _YOLO("yolov8s-worldv2.pt")
                if hasattr(self._model, "set_classes"):
                    self._model.set_classes(EXTRA_CLASSES)
                loaded = True
            except Exception:
                pass

        if loaded:
            self._loaded = True
            logger.info("👁 [VisionTalk] YOLOWorld ready — %d extra classes.", len(EXTRA_CLASSES))
        else:
            # Extra detection disabled — primary YOLO still handles COCO-80
            logger.info("👁 [VisionTalk] YOLOWorld unavailable on this ultralytics version — extra detection disabled.")

    def detect(self, frame: np.ndarray, conf: float = 0.35) -> list[dict]:
        """
        Returns a list of dicts with keys:
          class_name, confidence, x1, y1, x2, y2
        Returns [] if model not loaded or on any error.
        """
        if not self._loaded or self._model is None or frame is None:
            return []
        try:
            with self._lock:
                results = self._model.predict(frame, conf=conf, verbose=False)
            out = []
            for r in results:
                for box in r.boxes:
                    cid = int(box.cls)
                    if cid >= len(EXTRA_CLASSES):
                        continue
                    xyxy = box.xyxy[0].tolist()
                    out.append({
                        "class_name": EXTRA_CLASSES[cid],
                        "confidence": float(box.conf),
                        "x1": int(xyxy[0]),
                        "y1": int(xyxy[1]),
                        "x2": int(xyxy[2]),
                        "y2": int(xyxy[3]),
                    })
            return out
        except Exception as exc:
            logger.error("YOLOWorld detect error: %s", exc)
            return []


world_detector = WorldDetector()
