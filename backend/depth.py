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
        Zone: bottom 28% of frame, middle 30% horizontally.
        Requires BOTH std > 0.40 AND max abs gradient > 0.55 to reduce false positives.
        """
        if depth_map is None:
            return False
        y0 = int(frame_h * 0.72)
        x0 = int(frame_w * 0.35)
        x1 = int(frame_w * 0.65)
        zone = depth_map[y0:, x0:x1]
        if zone.size == 0:
            return False
        grad = np.gradient(zone, axis=0)
        return (float(np.std(grad)) > 0.40 and float(np.max(np.abs(grad))) > 0.55)


depth_estimator = DepthEstimator()  # module-level singleton
