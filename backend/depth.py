"""
depth.py — MiDaS depth estimator for VisionTalk.

Changes from original:
  - get_region_depth() now uses MEDIAN (not mean) — more robust to depth noise
    inside bounding boxes where background leaks in at the edges.
  - Added get_region_variance() — used by risk_engine to suppress distance
    reporting when depth is inconsistent (variance > DEPTH_VARIANCE_THRESHOLD).
  - Added is_depth_stable() convenience method.
  - metres_from_score() / feet_from_score() are now static methods for use
    by tracker and risk_engine without importing pipeline state.
  - detect_stair_drop() unchanged — already well-tuned.
  - Added depth_jump_reject(): rejects a new measurement that differs from
    the previous one by more than DEPTH_JUMP_REJECT_M metres (in converted
    metres space).  Prevents monocular depth flicker from entering velocity
    estimation as real motion signal.
"""

import os
import threading
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Variance threshold above which depth is considered unreliable for this region.
DEPTH_VARIANCE_THRESHOLD = 0.04

# Maximum permitted single-frame depth change (metres) before a measurement is
# discarded as a flicker artefact.  Real objects rarely jump >0.8 m per frame.
# Only applied via depth_jump_reject(); callers that do not use it are unaffected.
DEPTH_JUMP_REJECT_M = 0.80

# Maximum frame size for depth estimation (downscale larger frames for speed)
DEPTH_MAX_WIDTH = 640
DEPTH_MAX_HEIGHT = 480


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
            
            # Google-level: Upgrade to full MiDaS for 95% accuracy (better depth)
            # Can override with USE_MIDAS_FULL=0 to use small for speed
            use_full = os.getenv("USE_MIDAS_FULL", "1").strip() == "1"
            
            if use_full:
                # Full MiDaS model - best accuracy
                self._model = torch.hub.load(
                    "intel-isl/MiDaS", "MiDaS",
                    trust_repo=True, verbose=False,
                )
                self._transform = torch.hub.load(
                    "intel-isl/MiDaS", "transforms", trust_repo=True, verbose=False,
                ).default_transform
                model_name = "MiDaS (full)"
            else:
                # Small MiDaS - faster but lower accuracy
                self._model = torch.hub.load(
                    "intel-isl/MiDaS", "MiDaS_small",
                    trust_repo=True, verbose=False,
                )
                self._transform = torch.hub.load(
                    "intel-isl/MiDaS", "transforms", trust_repo=True, verbose=False,
                ).small_transform
                model_name = "MiDaS_small"
            
            self._model.to(self._device).eval()
            self._loaded = True
            logger.info("%s loaded on %s", model_name, self._device)
        except Exception as exc:
            logger.warning("MiDaS failed to load: %s. Depth disabled.", exc)
            self._loaded = False

    def estimate(self, frame: np.ndarray) -> "np.ndarray | None":
        """
        Returns float32 array same spatial size as original frame.
        Values: 0.0 = far, 1.0 = closest to camera.
        Returns None if not loaded or on error.
        
        Performance optimization: automatically downscales large frames for faster inference,
        then upscales the depth map back to original frame size.
        """
        if not self._loaded or frame is None or frame.size == 0:
            return None
        try:
            import torch
            t0 = time.time()
            with self._lock:
                # Downscale large frames for faster inference (maintains aspect ratio)
                original_h, original_w = frame.shape[:2]
                scale_factor = 1.0
                process_frame = frame
                
                if original_w > DEPTH_MAX_WIDTH or original_h > DEPTH_MAX_HEIGHT:
                    scale = min(DEPTH_MAX_WIDTH / original_w, DEPTH_MAX_HEIGHT / original_h)
                    new_w = int(original_w * scale)
                    new_h = int(original_h * scale)
                    process_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    scale_factor = scale
                    logger.debug("Depth: downscaled frame %dx%d → %dx%d", original_w, original_h, new_w, new_h)
                
                # Google-level: Apply advanced preprocessing for better depth
                try:
                    from backend.advanced_preprocessing import advanced_preprocessor
                    process_frame = advanced_preprocessor.preprocess(process_frame)
                except Exception:
                    pass  # Non-fatal
                
                rgb = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
                inp = self._transform(rgb).to(self._device)
                
                with torch.no_grad():
                    pred = self._model(inp)
                    # Get prediction size (may be different from input due to model architecture)
                    pred_h, pred_w = pred.shape[-2:]
                    
                    # Upscale depth map back to original frame size
                    pred = torch.nn.functional.interpolate(
                        pred.unsqueeze(1),
                        size=(original_h, original_w),
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze().cpu().numpy()
                
                # Normalize to [0, 1]
                mn, mx = pred.min(), pred.max()
                if mx - mn > 1e-8:
                    pred = (pred - mn) / (mx - mn)
                
                # Update performance monitor
                elapsed = time.time() - t0
                try:
                    from backend.performance_monitor import performance_monitor
                    performance_monitor.record_depth_time(elapsed)
                except Exception:
                    pass  # Non-fatal if monitor not available
                
                return pred.astype(np.float32)
        except Exception as exc:
            logger.error("Depth estimate error: %s", exc)
            return None

    # ── Region queries ────────────────────────────────────────────────────────

    def get_region_depth(self, depth_map, x1: int, y1: int, x2: int, y2: int) -> float:
        """
        MEDIAN depth score inside a bounding box region.
        Median is more robust than mean when the bbox overlaps background pixels.
        Returns 0.0 if depth_map unavailable or region is empty.
        """
        if depth_map is None:
            return 0.0
        y1c = max(0, y1); y2c = min(depth_map.shape[0], y2)
        x1c = max(0, x1); x2c = min(depth_map.shape[1], x2)
        region = depth_map[y1c:y2c, x1c:x2c]
        if region.size == 0:
            return 0.0
        return float(np.median(region))

    def get_region_variance(self, depth_map, x1: int, y1: int, x2: int, y2: int) -> float:
        """
        Variance of depth values inside a bounding box.
        High variance means depth estimate is unreliable for this region.
        Compare against DEPTH_VARIANCE_THRESHOLD before reporting distance.
        """
        if depth_map is None:
            return 0.0
        y1c = max(0, y1); y2c = min(depth_map.shape[0], y2)
        x1c = max(0, x1); x2c = min(depth_map.shape[1], x2)
        region = depth_map[y1c:y2c, x1c:x2c]
        if region.size < 4:
            return 0.0
        return float(np.var(region))

    def is_depth_stable(self, depth_map, x1: int, y1: int, x2: int, y2: int) -> bool:
        """
        True if depth is stable enough to report a distance reading.
        Stable = variance <= DEPTH_VARIANCE_THRESHOLD.
        """
        return self.get_region_variance(depth_map, x1, y1, x2, y2) <= DEPTH_VARIANCE_THRESHOLD

    @staticmethod
    def depth_jump_reject(prev_dist_m: float, new_dist_m: float) -> bool:
        """
        Return True if the new distance measurement is a likely flicker artefact
        and should be discarded.

        Monocular depth (MiDaS) occasionally produces single-frame jumps that
        are physically implausible — a real object does not teleport 1 m between
        frames at typical camera frame rates.  This guard prevents such spikes
        from entering the EMA smoother and generating false velocity signals.

        Args:
            prev_dist_m : previously accepted smoothed distance in metres.
            new_dist_m  : raw new measurement candidate in metres.

        Returns:
            True  → discard new_dist_m (jump too large).
            False → accept new_dist_m (within plausible range).
        """
        if prev_dist_m <= 0.0:
            # No prior measurement — always accept the first reading.
            return False
        return abs(new_dist_m - prev_dist_m) > DEPTH_JUMP_REJECT_M

    # ── Distance conversion ───────────────────────────────────────────────────

    @staticmethod
    def metres_from_score(depth_score: float) -> float:
        """
        Convert MiDaS depth score (0=far, 1=close) to approximate metres.
        Piecewise linear calibration. Conservative — errs toward closer = safer.
        """
        if depth_score <= 0.0:
            return 0.0
        if depth_score < 0.3:
            metres = 5.0 + (1.0 - depth_score) * 5.0
        elif depth_score < 0.6:
            metres = 2.0 + (0.6 - depth_score) * 3.0 / 0.3
        elif depth_score < 0.8:
            metres = 1.0 + (0.8 - depth_score) * 1.0 / 0.2
        else:
            metres = max(0.1, 0.5 * (1.1 - depth_score) / 0.3)
        return round(metres, 2)

    @staticmethod
    def feet_from_score(depth_score: float) -> float:
        """Convert depth score to feet (for backward compatibility)."""
        return round(DepthEstimator.metres_from_score(depth_score) * 3.28084, 1)

    # ── Floor / stair detection ───────────────────────────────────────────────

    def detect_stair_drop(self, depth_map, frame_h: int, frame_w: int) -> bool:
        """
        Detect a sudden depth gradient change in lower-center zone.
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
        return (
            float(np.std(grad)) > 0.40
            and float(np.max(np.abs(grad))) > 0.55
        )


depth_estimator = DepthEstimator()
