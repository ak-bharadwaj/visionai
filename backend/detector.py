import numpy as np, logging, os
from pathlib import Path
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

MODEL_DIR  = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "yolov8s.pt"   # auto-downloads ~22 MB on first use

# All 80 COCO classes in exact order — kept as reference
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
    SIZE = int(os.getenv("INFERENCE_SIZE", "640"))
    CONF = float(os.getenv("CONF_THRESHOLD", "0.35"))
    IOU  = float(os.getenv("IOU_THRESHOLD",  "0.45"))

    def __init__(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        from ultralytics import YOLO
        # If the .pt file is already in models/, use it; otherwise let ultralytics
        # download it from its CDN (saved to models/ via the name override).
        self._model = YOLO(str(MODEL_PATH) if MODEL_PATH.exists() else "yolov8s.pt")
        # Copy downloaded weights into models/ so subsequent starts are offline-safe
        if not MODEL_PATH.exists():
            import shutil
            downloaded = Path("yolov8s.pt")
            if downloaded.exists():
                shutil.copy(downloaded, MODEL_PATH)
        self._model.overrides["verbose"] = False
        logger.info("Detector ready: YOLOv8s (ultralytics).")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if frame is None:
            return []
        results = self._model.predict(
            source=frame,
            imgsz=self.SIZE,
            conf=self.CONF,
            iou=self.IOU,
            verbose=False,
        )
        detections: List[Detection] = []
        if not results:
            return detections
        boxes = results[0].boxes
        if boxes is None:
            return detections
        for box in boxes:
            cid  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            name = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else (
                results[0].names.get(cid, "object"))
            detections.append(Detection(
                class_id=cid, class_name=name, confidence=conf,
                x1=x1, y1=y1, x2=x2, y2=y2,
            ))
        return detections
