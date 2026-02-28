#!/usr/bin/env python3
"""Run this to verify which models are loaded and working."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

def main():
    from pathlib import Path
    from backend.model_checker import check_models, get_best_available_yolo
    
    MODEL_DIR = Path(__file__).parent / "models"
    print("=" * 50)
    print("VisionTalk Model Check")
    print("=" * 50)
    
    print("\n[Models in models/]:")
    if MODEL_DIR.exists():
        for f in sorted(MODEL_DIR.glob("*")):
            if f.is_file() and not f.name.startswith("."):
                size_mb = f.stat().st_size / 1024 / 1024
                print(f"   OK {f.name} ({size_mb:.1f} MB)")
    else:
        print("   (models/ folder not found)")
    
    status = check_models()
    print("\n[Status]:")
    print(f"   YOLO configured: {status['yolo']['configured']}")
    print(f"   YOLO available:  {status['yolo']['available'] or 'none'}")
    print(f"   Best to use:     {get_best_available_yolo()}")
    print(f"   Gemini keys:     {status['gemini']['keys']}")
    
    print("\n[Testing detector]...")
    try:
        from backend.detector import ObjectDetector
        det = ObjectDetector()
        print("   OK Detector loaded successfully")
        # Quick test
        import numpy as np
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = det.detect(test_frame)
        print(f"   OK Test inference: {len(out)} detections (expected 0 on black frame)")
    except Exception as e:
        print(f"   FAIL Detector failed: {e}")
    
    print("\n[Testing depth (MiDaS)]...")
    try:
        from backend.depth import depth_estimator
        depth_estimator.load()
        print(f"   OK Depth: {'loaded' if depth_estimator._loaded else 'FAILED'}")
    except Exception as e:
        print(f"   FAIL Depth failed: {e}")
    
    print("\n[Testing OCR]...")
    try:
        from backend.ocr import ocr_reader
        ocr_reader._ensure_loaded()
        print(f"   OK OCR: {'loaded' if ocr_reader._loaded else 'FAILED'}")
    except Exception as e:
        print(f"   FAIL OCR failed: {e}")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
