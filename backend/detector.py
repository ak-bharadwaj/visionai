"""
detector.py — YOLOv8s object detector for VisionTalk.

Changes from original:
  - Hard confidence gate raised to 0.60 (was 0.35).
    Fewer false positives. Safety-critical systems prefer precision over recall.
  - Input resize always 640 (YOLOv8s native training resolution).
  - NMS IOU threshold tightened to 0.40 (reduces duplicate boxes on same object).
  - Error-safe: any model exception returns [] instead of crashing the pipeline.
  - Stateless per-call design — pipeline controls cadence (every N frames).
"""

import numpy as np
import logging
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

MODEL_DIR  = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "yolov8s.pt"

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
    # NMS IOU — tighter than YOLO default 0.45 to suppress duplicate boxes.
    IOU  = float(os.getenv("IOU_THRESHOLD",  "0.40"))

    def __init__(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        from ultralytics import YOLO
        model_path = str(MODEL_PATH) if MODEL_PATH.exists() else "yolov8s.pt"
        self._model = YOLO(model_path)
        # Cache downloaded weights to models/ for future offline starts
        if not MODEL_PATH.exists():
            import shutil
            downloaded = Path("yolov8s.pt")
            if downloaded.exists():
                shutil.copy(downloaded, MODEL_PATH)
        self._model.overrides["verbose"] = False
        logger.info(
            "Detector ready: YOLOv8s conf=%.2f iou=%.2f size=%d",
            self.CONF, self.IOU, self.SIZE,
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLOv8s on frame. Returns list of Detection objects.
        Only detections with confidence >= CONF are returned.
        Returns [] on any error — never raises.
        """
        if frame is None:
            return []
        try:
            results = self._model.predict(
                source=frame,
                imgsz=self.SIZE,
                conf=self.CONF,
                iou=self.IOU,
                verbose=False,
            )
        except Exception as exc:
            logger.error("Detector predict error: %s", exc)
            return []

        detections: List[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        for box in boxes:
            cid  = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < self.CONF:
                continue  # belt-and-suspenders
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            name = (
                COCO_CLASSES[cid]
                if cid < len(COCO_CLASSES)
                else results[0].names.get(cid, "object")
            )
            detections.append(
                Detection(
                    class_id=cid, class_name=name, confidence=conf,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                )
            )
        return detections
