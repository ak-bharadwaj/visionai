"""
model_upgrade.py — Upgrade to better models for 95% accuracy.

Upgrades:
  - YOLOv8n → YOLOv8s (better accuracy, still fast)
  - MiDaS_small → MiDaS (better depth accuracy)
  - Enhanced preprocessing for all models
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Model selection based on accuracy target
USE_YOLOV8S = os.getenv("USE_YOLOV8S", "1").strip() == "1"  # Better accuracy
USE_MIDAS_FULL = os.getenv("USE_MIDAS_FULL", "1").strip() == "1"  # Better depth

MODEL_DIR = Path(__file__).parent.parent / "models"

def get_yolo_model_name():
    """Return best YOLO model for 95% accuracy target."""
    if USE_YOLOV8S:
        return "yolov8s.pt"  # Small - better accuracy than nano, still fast
    return "yolov8n.pt"  # Nano - fastest but lower accuracy

def get_midas_model_name():
    """Return best MiDaS model for 95% accuracy target."""
    if USE_MIDAS_FULL:
        return "MiDaS"  # Full model - best accuracy
    return "MiDaS_small"  # Small - faster but lower accuracy
