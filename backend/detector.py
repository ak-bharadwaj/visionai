"""
detector.py — YOLOv8m object detector for VisionTalk.

Changes from original:
  - Model upgraded to YOLOv8m (higher accuracy than s/n).
  - Hard confidence gate raised to 0.60 (was 0.35).
    Fewer false positives. Safety-critical systems prefer precision over recall.
  - Input resize always 640 (YOLOv8 native training resolution).
  - NMS IOU threshold 0.50 per spec (reduces duplicate boxes on same object).
  - Error-safe: any model exception returns [] instead of crashing the pipeline.
  - Stateless per-call design — pipeline controls cadence (every N frames).

Debug mode (set env var DEBUG_DETECTIONS=1):
  Logs every raw YOLO detection BEFORE confidence gating and class whitelist.
  This lets you distinguish:
    - Objects detected but dropped by conf gate  → raise CONF_THRESHOLD
    - Objects detected but dropped by whitelist  → check ALLOWED_CLASSES
    - Objects never detected at all              → model/preprocessing issue

  Example output (at DEBUG level):
    Raw YOLO detections (3 total, conf >= 0.10):
      person       conf=0.82  [dropped: passed]
      chair        conf=0.42  [dropped: below_conf_gate]
      refrigerator conf=0.37  [dropped: not_in_whitelist]
    Inference time: 47ms  returned 1 detections
"""

import time
import numpy as np
import logging
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

# Set DEBUG_DETECTIONS=1 to log all raw YOLO output before any filtering.
# Use this to diagnose whether objects are being seen by the model at all.
DEBUG_DETECTIONS: bool = os.getenv("DEBUG_DETECTIONS", "0").strip() == "1"
# Minimum confidence to show in debug output (avoids noise from near-zero scores).
DEBUG_RAW_CONF_MIN: float = float(os.getenv("DEBUG_RAW_CONF_MIN", "0.10"))

MODEL_DIR  = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "yolov8m.pt"

# ── Allowed detection classes (spec-exact: 12 navigation-relevant classes) ────
# Only these classes pass through in NAVIGATE mode.
# ASK/FIND mode uses apply_whitelist=False for maximum recall.
ALLOWED_CLASSES: frozenset = frozenset({
    "person", "chair", "table", "car", "bus", "truck",
    "bicycle", "motorcycle", "door", "stairs", "wall", "pole",
})

# All 80 COCO classes in exact order
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


@dataclass
class Detection:
    class_id:   int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def bbox_area(self) -> int:
        return max(0, (self.x2 - self.x1) * (self.y2 - self.y1))

    @property
    def center_x(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def center_y(self) -> int:
        return (self.y1 + self.y2) // 2

    @property
    def bottom_y(self) -> int:
        return self.y2


class ObjectDetector:
    # 640 is the native YOLOv8s training resolution.
    SIZE = 640
    # Hard confidence gate. Objects below this NEVER reach downstream stages.
    # 0.60 → higher precision, fewer false narrations. Correct for safety-critical use.
    CONF = float(os.getenv("CONF_THRESHOLD", "0.60"))
    # NMS IOU — per spec: 0.50 to suppress duplicate boxes.
    IOU  = float(os.getenv("IOU_THRESHOLD",  "0.50"))

    def __init__(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        from ultralytics import YOLO
        model_path = str(MODEL_PATH) if MODEL_PATH.exists() else "yolov8m.pt"
        self._model = YOLO(model_path)
        # Cache downloaded weights to models/ for future offline starts
        if not MODEL_PATH.exists():
            import shutil
            downloaded = Path("yolov8m.pt")
            if downloaded.exists():
                shutil.copy(downloaded, MODEL_PATH)
        self._model.overrides["verbose"] = False
        logger.info(
            "Detector ready: YOLOv8m conf=%.2f iou=%.2f size=%d",
            self.CONF, self.IOU, self.SIZE,
        )

    def detect(self, frame: np.ndarray, conf: float | None = None,
               apply_whitelist: bool = True) -> List[Detection]:
        """
        Run YOLOv8m on frame. Returns list of Detection objects.
        Only detections with confidence >= conf (default: self.CONF) are returned.
        Pass a lower conf (e.g. 0.35) for ASK/FIND query mode to improve recall.
        Set apply_whitelist=False in ASK/FIND mode to allow all 80 COCO classes.
        Returns [] on any error — never raises.

        When DEBUG_DETECTIONS=1 (env var), logs every raw YOLO result before
        filtering, including the reason each detection was kept or dropped.
        """
        if frame is None:
            return []
        gate = conf if conf is not None else self.CONF
        t_infer = time.time()
        try:
            results = self._model.predict(
                source=frame,
                imgsz=self.SIZE,
                conf=DEBUG_RAW_CONF_MIN if DEBUG_DETECTIONS else gate,
                iou=self.IOU,
                verbose=False,
            )
        except Exception as exc:
            logger.error("Detector predict error: %s", exc)
            return []
        elapsed_ms = (time.time() - t_infer) * 1000

        detections: List[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        raw_lines: List[str] = []   # accumulated only when DEBUG_DETECTIONS=True

        for box in boxes:
            cid   = int(box.cls[0])
            conf_ = float(box.conf[0])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            name = (
                COCO_CLASSES[cid]
                if cid < len(COCO_CLASSES)
                else results[0].names.get(cid, "object")
            )

            # Determine drop reason for debug logging
            if conf_ < gate:
                drop = "below_conf_gate"
            elif apply_whitelist and name not in ALLOWED_CLASSES:
                drop = "not_in_whitelist"
            else:
                drop = "passed"

            if DEBUG_DETECTIONS:
                raw_lines.append(
                    f"  {name:<20} conf={conf_:.2f}  [{drop}]"
                )

            if drop != "passed":
                continue

            # belt-and-suspenders: gate already applied above
            detections.append(
                Detection(
                    class_id=cid, class_name=name, confidence=conf_,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                )
            )

        if DEBUG_DETECTIONS:
            header = (
                f"Raw YOLO detections ({len(raw_lines)} total, "
                f"conf >= {DEBUG_RAW_CONF_MIN:.2f}):"
            )
            body = "\n".join(raw_lines) if raw_lines else "  (none)"
            logger.debug(
                "%s\n%s\n  Inference: %.0fms  returned %d detection(s)",
                header, body, elapsed_ms, len(detections),
            )
        elif logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Detector: %.0fms  returned %d detection(s)  gate=%.2f",
                elapsed_ms, len(detections), gate,
            )

        return detections
