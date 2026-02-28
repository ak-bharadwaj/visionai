"""
model_checker.py — Verify which models exist and are loaded.
Run at startup to diagnose model issues.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
MODEL_DIR = Path(__file__).parent.parent / "models"


def check_models() -> dict:
    """Check which models exist and report status."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    status = {
        "yolo": {"configured": None, "available": [], "loaded": None},
        "depth": {"configured": "MiDaS (downloads on first run)", "loaded": None},
        "ocr": {"configured": "PaddleOCR", "loaded": None},
        "gemini": {"configured": False, "keys": 0},
    }
    
    # YOLO: Check what we have vs what we want
    yolo_models = ["yolov8l.pt", "yolov8m.pt", "yolov8s.pt", "yolov8n.pt"]
    available = [m for m in yolo_models if (MODEL_DIR / m).exists()]
    status["yolo"]["available"] = available
    
    use_m = os.getenv("USE_YOLOV8M", "1").strip() == "1"
    use_l = os.getenv("USE_YOLOV8L", "0").strip() == "1"
    if use_l:
        status["yolo"]["configured"] = "yolov8l.pt"
    elif use_m:
        status["yolo"]["configured"] = "yolov8m.pt"
    else:
        status["yolo"]["configured"] = "yolov8s.pt"
    
    # Gemini keys
    keys = sum(1 for i in range(1, 5) if os.getenv(f"GEMINI_API_KEY_{i}", "").strip())
    if os.getenv("GEMINI_API_KEY", "").strip():
        keys = max(keys, 1)
    status["gemini"]["configured"] = keys > 0
    status["gemini"]["keys"] = keys
    
    return status


def get_best_available_yolo() -> str:
    """
    Return the best YOLO model that EXISTS locally.
    Prefer: yolov8m > yolov8s > yolov8n (use what we HAVE - avoid download failures).
    """
    # Check what actually exists in models/
    preference = ["yolov8m.pt", "yolov8l.pt", "yolov8s.pt", "yolov8n.pt"]
    for name in preference:
        if (MODEL_DIR / name).exists():
            return name
    # None exist - use yolov8s (good balance, smaller download than m/l)
    use_l = os.getenv("USE_YOLOV8L", "0").strip() == "1"
    if use_l:
        return "yolov8l.pt"  # Will download
    return "yolov8s.pt"  # User has this - or will download
