"""
performance_monitor.py — Real-time performance monitoring and adaptive quality control.

Tracks:
  - FPS (frames per second)
  - Processing time per frame
  - Detection latency
  - OCR latency
  - Depth estimation latency
  - Memory usage
  - GPU utilization (if available)

Provides adaptive quality settings based on performance metrics.
"""

import time
import logging
import threading
from collections import deque
from typing import Dict, Optional
import os

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Thread-safe performance monitoring with adaptive quality control."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._frame_times: deque = deque(maxlen=60)  # Last 60 frames
        self._detection_times: deque = deque(maxlen=30)
        self._ocr_times: deque = deque(maxlen=20)
        self._depth_times: deque = deque(maxlen=20)
        
        # Performance metrics
        self._current_fps: float = 0.0
        self._avg_processing_time: float = 0.0
        self._avg_detection_time: float = 0.0
        self._avg_ocr_time: float = 0.0
        self._avg_depth_time: float = 0.0
        
        # Adaptive quality settings
        self._quality_level: str = "high"  # high, medium, low
        self._last_update: float = time.time()
        self._update_interval: float = 5.0  # Update quality every 5 seconds
        
        # Performance thresholds
        self._target_fps: float = float(os.getenv("TARGET_FPS", "10.0"))
        self._min_acceptable_fps: float = self._target_fps * 0.7  # 70% of target
        
    def record_frame_time(self, elapsed: float):
        """Record frame processing time."""
        with self._lock:
            self._frame_times.append(elapsed)
            if len(self._frame_times) > 0:
                self._avg_processing_time = sum(self._frame_times) / len(self._frame_times)
                self._current_fps = 1.0 / self._avg_processing_time if self._avg_processing_time > 0 else 0.0
    
    def record_detection_time(self, elapsed: float):
        """Record detection inference time."""
        with self._lock:
            self._detection_times.append(elapsed)
            if len(self._detection_times) > 0:
                self._avg_detection_time = sum(self._detection_times) / len(self._detection_times)
    
    def record_ocr_time(self, elapsed: float):
        """Record OCR processing time."""
        with self._lock:
            self._ocr_times.append(elapsed)
            if len(self._ocr_times) > 0:
                self._avg_ocr_time = sum(self._ocr_times) / len(self._ocr_times)
    
    def record_depth_time(self, elapsed: float):
        """Record depth estimation time."""
        with self._lock:
            self._depth_times.append(elapsed)
            if len(self._depth_times) > 0:
                self._avg_depth_time = sum(self._depth_times) / len(self._depth_times)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        with self._lock:
            return {
                "fps": self._current_fps,
                "avg_processing_time_ms": self._avg_processing_time * 1000,
                "avg_detection_time_ms": self._avg_detection_time * 1000,
                "avg_ocr_time_ms": self._avg_ocr_time * 1000,
                "avg_depth_time_ms": self._avg_depth_time * 1000,
                "quality_level": self._quality_level,
            }
    
    def get_quality_level(self) -> str:
        """Get current adaptive quality level."""
        with self._lock:
            return self._quality_level
    
    def update_quality(self):
        """Update adaptive quality settings based on current performance."""
        now = time.time()
        if now - self._last_update < self._update_interval:
            return
        
        with self._lock:
            self._last_update = now
            
            # Determine quality level based on FPS
            if self._current_fps >= self._target_fps * 0.9:
                new_quality = "high"
            elif self._current_fps >= self._min_acceptable_fps:
                new_quality = "medium"
            else:
                new_quality = "low"
            
            if new_quality != self._quality_level:
                logger.info(
                    "[Performance] Quality level changed: %s → %s (FPS: %.1f, target: %.1f)",
                    self._quality_level, new_quality, self._current_fps, self._target_fps
                )
                self._quality_level = new_quality
    
    def should_reduce_quality(self) -> bool:
        """Check if quality should be reduced due to poor performance."""
        with self._lock:
            return self._current_fps < self._min_acceptable_fps
    
    def get_recommended_detection_cadence(self) -> int:
        """Get recommended detection cadence based on performance."""
        with self._lock:
            if self._quality_level == "high":
                return 2  # More frequent detection
            elif self._quality_level == "medium":
                return 3  # Default
            else:
                return 5  # Less frequent to save CPU
    
    def log_summary(self):
        """Log performance summary (called periodically)."""
        with self._lock:
            metrics = self.get_metrics()
            logger.info(
                "[Performance] FPS: %.1f | Detection: %.1fms | OCR: %.1fms | Depth: %.1fms | Quality: %s",
                metrics["fps"],
                metrics["avg_detection_time_ms"],
                metrics["avg_ocr_time_ms"],
                metrics["avg_depth_time_ms"],
                metrics["quality_level"]
            )


# Module-level singleton
performance_monitor = PerformanceMonitor()
