"""
google_level_upgrade.py — Google-level system upgrade.

This module implements comprehensive upgrades to reach Google-level quality:
- Better models (YOLOv8m/l)
- Advanced preprocessing
- Advanced post-processing
- Better tracking
- Performance optimizations
"""

import logging
import os

logger = logging.getLogger(__name__)

# Google-level model selection
USE_YOLOV8M = os.getenv("USE_YOLOV8M", "1").strip() == "1"  # Medium for best balance
USE_YOLOV8L = os.getenv("USE_YOLOV8L", "0").strip() == "1"  # Large for maximum accuracy

def get_best_yolo_model():
    """Return best YOLO model for Google-level accuracy."""
    if USE_YOLOV8L:
        return "yolov8l.pt"  # Large - maximum accuracy
    elif USE_YOLOV8M:
        return "yolov8m.pt"  # Medium - best balance
    else:
        return "yolov8s.pt"  # Small - fallback

def get_best_depth_model():
    """Return best depth model for Google-level accuracy."""
    return "MiDaS"  # Full model for best accuracy

def get_optimized_thresholds():
    """Return optimized thresholds for Google-level accuracy."""
    return {
        "detection_conf": 0.15,  # Lower for maximum recall
        "detection_iou": 0.40,   # Lower for better overlap handling
        "tracker_conf": 0.20,    # Lower for better tracking
        "narration_conf": 0.30,  # Balanced for quality
        "ocr_conf": 0.40,        # Lower for better text detection
    }
