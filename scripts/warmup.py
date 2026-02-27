"""Run this before the demo to pre-load all models."""
import time, sys, os, numpy as np
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv()

print("[VisionTalk Warmup] Starting...")
fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

print("  [1/5] Loading YOLOv8n...")
t = time.time()
from backend.detector import ObjectDetector
det = ObjectDetector()
det.detect(fake_frame)
print(f"        Ready in {time.time()-t:.1f}s")

print("  [2/5] Loading MiDaS depth...")
t = time.time()
from backend.depth import depth_estimator
depth_estimator.load()
print(f"        Ready in {time.time()-t:.1f}s")

print("  [3/5] Loading PaddleOCR...")
t = time.time()
from backend.ocr import ocr_reader
ocr_reader.read(fake_frame)
print(f"        Ready in {time.time()-t:.1f}s")

print("  [4/5] Warming up Ollama LLM (first call is slowest)...")
t = time.time()
try:
    import urllib.request, json
    payload = json.dumps({
        "model": os.getenv("OLLAMA_MODEL", "phi3:mini"),
        "prompt": "Say: ready",
        "stream": False,
        "options": {"num_predict": 5}
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        result = json.loads(r.read())
    print(f"        LLM ready in {time.time()-t:.1f}s")
except Exception as e:
    print(f"        Ollama not running: {e}. Start with: ollama serve")

print("  [5/5] Pre-loading YOLOWorld secondary detector...")
t = time.time()
try:
    from backend.world_detector import world_detector
    world_detector.load()
    world_detector.detect(fake_frame)
    print(f"        Ready in {time.time()-t:.1f}s")
except Exception as e:
    print(f"        YOLOWorld skip: {e}")

print("\n=== Warmup complete. All 5 models loaded. Run `start.bat` now. ===")
