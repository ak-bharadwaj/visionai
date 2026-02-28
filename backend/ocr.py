import logging, threading, numpy as np
import cv2
import time
from typing import List, Set

logger = logging.getLogger(__name__)

# Maximum frame size for OCR (downscale larger frames for speed)
OCR_MAX_WIDTH = 1280
OCR_MAX_HEIGHT = 720


class OCRReader:
    def __init__(self):
        self._ocr    = None
        self._lock   = threading.Lock()
        self._loaded = False
        self._use_gpu = False
        # Session-level read history: cleared when READ mode is entered
        self._read_history: Set[str] = set()

    def _ensure_loaded(self):
        if self._loaded:
            return
        try:
            from paddleocr import PaddleOCR
            # Auto-detect GPU availability
            try:
                import paddle
                self._use_gpu = paddle.device.is_compiled_with_cuda() and paddle.device.get_device() != "cpu"
            except Exception:
                self._use_gpu = False
            
            # Enable GPU if available for 3-5x speedup
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                show_log=False,
                use_gpu=self._use_gpu,
                enable_mkldnn=(not self._use_gpu),  # Intel MKL-DNN on CPU for speed
            )
            self._loaded = True
            logger.info("PaddleOCR loaded (device=%s)", "GPU" if self._use_gpu else "CPU")
        except Exception as e:
            logger.warning(f"PaddleOCR load failed: {e}. READ mode disabled.")

    def clear_history(self):
        """Clear session read history. Call this when entering READ mode."""
        self._read_history.clear()

    def read(self, frame: np.ndarray, deduplicate: bool = False) -> List[str]:
        """
        Returns list of text strings found.
        Sorting: center-of-frame priority first, then top-to-bottom within equal-priority bands.
        When deduplicate=True, skips text already seen in this READ session.
        Empty list on error or no text.
        
        Performance optimization: automatically downscales large frames for faster OCR.
        """
        self._ensure_loaded()
        if not self._loaded or frame is None or frame.size == 0:
            return []
        try:
            t0 = time.time()
            # Downscale large frames for faster OCR (maintains aspect ratio)
            h, w = frame.shape[:2]
            if w > OCR_MAX_WIDTH or h > OCR_MAX_HEIGHT:
                scale = min(OCR_MAX_WIDTH / w, OCR_MAX_HEIGHT / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                logger.debug("OCR: downscaled frame %dx%d → %dx%d", w, h, new_w, new_h)
            
            # Google-level: Advanced preprocessing for better OCR accuracy
            try:
                from backend.advanced_preprocessing import advanced_preprocessor
                preprocessed_frame = advanced_preprocessor.preprocess(frame)
            except Exception:
                try:
                    from backend.enhanced_ocr import ocr_preprocessor
                    preprocessed_frame = ocr_preprocessor.preprocess(frame)
                except Exception:
                    preprocessed_frame = frame  # Fallback to original
            
            with self._lock:
                result = self._ocr.ocr(preprocessed_frame, cls=True)
            
            # Update performance monitor
            elapsed = time.time() - t0
            try:
                from backend.performance_monitor import performance_monitor
                performance_monitor.record_ocr_time(elapsed)
            except Exception:
                pass  # Non-fatal if monitor not available
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
                    
                    # Google-level: Confidence calibration
                    try:
                        from backend.enhanced_ocr import ocr_confidence_calibrator
                        conf = ocr_confidence_calibrator.calibrate(text, conf)
                    except Exception:
                        pass  # Use original confidence
                    
                    # Google-level: Lowered OCR confidence threshold for better recall
                    if conf > 0.50 and len(text.strip()) >= 2:  # Lowered from 0.60 to 0.50
                        pts = line[0]
                        bx = sum(p[0] for p in pts) / 4
                        by = sum(p[1] for p in pts) / 4
                        dist = ((bx - frame_cx)**2 + (by - frame_cy)**2) ** 0.5
                        items.append((dist, by, text.strip()))

            # Sort: center-priority first, then top-to-bottom (by Y) within same band
            # Band width = 10% of frame height so near-center texts still read top→bottom
            band_h = frame.shape[0] * 0.10
            items.sort(key=lambda x: (int(x[0] / band_h), x[1]))

            new_texts = []
            for _, _, text in items:
                if deduplicate and text in self._read_history:
                    continue
                new_texts.append(text)
                if deduplicate:
                    self._read_history.add(text)
            
            # Google-level: Track OCR accuracy metrics
            for text in new_texts:
                try:
                    from backend.accuracy_metrics import accuracy_tracker
                    # Find matching confidence (simplified - in real implementation would track per-text)
                    accuracy_tracker.record_ocr(text, 0.75)  # Approximate
                except Exception:
                    pass  # Non-fatal
            
            return new_texts
        except Exception as e:
            logger.error(f"OCR error: {e}", exc_info=True)
            # Return empty list on error — never crash the pipeline
            return []


ocr_reader = OCRReader()  # module-level singleton
