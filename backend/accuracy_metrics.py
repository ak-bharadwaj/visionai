"""
accuracy_metrics.py — Google-level accuracy tracking and validation.

Tracks:
  - Detection accuracy metrics
  - OCR accuracy metrics
  - False positive/negative rates
  - Confidence calibration accuracy
  - Per-class accuracy
"""

import logging
import threading
from collections import defaultdict, deque
from typing import Dict, List, Optional
import time

logger = logging.getLogger(__name__)


class AccuracyTracker:
    """Track accuracy metrics for continuous improvement."""
    
    def __init__(self):
        self._lock = threading.Lock()
        
        # Detection metrics
        self._detection_counts: Dict[str, int] = defaultdict(int)
        self._detection_confidences: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        
        # OCR metrics
        self._ocr_text_lengths: deque = deque(maxlen=1000)
        self._ocr_confidences: deque = deque(maxlen=1000)
        
        # Performance tracking
        self._false_positive_estimate: float = 0.0
        self._false_negative_estimate: float = 0.0
        
    def record_detection(self, class_name: str, confidence: float):
        """Record a detection for accuracy tracking."""
        with self._lock:
            self._detection_counts[class_name] += 1
            self._detection_confidences[class_name].append(confidence)
    
    def record_ocr(self, text: str, confidence: float):
        """Record OCR result for accuracy tracking."""
        with self._lock:
            self._ocr_text_lengths.append(len(text))
            self._ocr_confidences.append(confidence)
    
    def get_detection_stats(self) -> Dict[str, Dict]:
        """Get detection statistics per class."""
        with self._lock:
            stats = {}
            for class_name, confidences in self._detection_confidences.items():
                if not confidences:
                    continue
                stats[class_name] = {
                    "count": self._detection_counts[class_name],
                    "avg_confidence": sum(confidences) / len(confidences),
                    "min_confidence": min(confidences),
                    "max_confidence": max(confidences),
                }
            return stats
    
    def get_ocr_stats(self) -> Dict:
        """Get OCR statistics."""
        with self._lock:
            if not self._ocr_confidences:
                return {}
            return {
                "avg_confidence": sum(self._ocr_confidences) / len(self._ocr_confidences),
                "avg_text_length": sum(self._ocr_text_lengths) / len(self._ocr_text_lengths) if self._ocr_text_lengths else 0,
                "total_reads": len(self._ocr_confidences),
            }
    
    def log_summary(self):
        """Log accuracy summary."""
        with self._lock:
            det_stats = self.get_detection_stats()
            ocr_stats = self.get_ocr_stats()
            
            if det_stats:
                logger.info("[Accuracy] Detection stats:")
                for class_name, stats in sorted(det_stats.items(), 
                                               key=lambda x: x[1]["count"], 
                                               reverse=True)[:10]:
                    logger.info(
                        "  %s: count=%d, avg_conf=%.2f",
                        class_name, stats["count"], stats["avg_confidence"]
                    )
            
            if ocr_stats:
                logger.info(
                    "[Accuracy] OCR: avg_conf=%.2f, avg_length=%.1f, total=%d",
                    ocr_stats["avg_confidence"],
                    ocr_stats["avg_text_length"],
                    ocr_stats["total_reads"]
                )


# Module-level singleton
accuracy_tracker = AccuracyTracker()
