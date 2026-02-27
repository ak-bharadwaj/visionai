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
    # 640 detects small objects (cups, bottles, pens, books) — 320 is faster but misses small items
    SIZE = int(__import__("os").getenv("INFERENCE_SIZE", "640"))  # input resolution (square)
    # Lower threshold = catches small/partial objects. 0.35 is safe for demo use.
    CONF = float(__import__("os").getenv("CONF_THRESHOLD", "0.35"))
    IOU  = float(__import__("os").getenv("IOU_THRESHOLD",  "0.45"))

    def __init__(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if not MODEL_PATH.exists():
            logger.info("Downloading YOLOv8n ONNX (~12 MB)...")
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
