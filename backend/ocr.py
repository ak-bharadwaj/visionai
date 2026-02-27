import logging, threading, numpy as np
from typing import List, Set

logger = logging.getLogger(__name__)


class OCRReader:
    def __init__(self):
        self._ocr    = None
        self._lock   = threading.Lock()
        self._loaded = False
        # Session-level read history: cleared when READ mode is entered
        self._read_history: Set[str] = set()

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

    def clear_history(self):
        """Clear session read history. Call this when entering READ mode."""
        self._read_history.clear()

    def read(self, frame: np.ndarray, deduplicate: bool = False) -> List[str]:
        """
        Returns list of text strings found.
        Sorting: center-of-frame priority first, then top-to-bottom within equal-priority bands.
        When deduplicate=True, skips text already seen in this READ session.
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
                    if conf > 0.60 and len(text.strip()) >= 2:
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
            return new_texts
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return []


ocr_reader = OCRReader()  # module-level singleton
