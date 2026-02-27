from dotenv import load_dotenv
load_dotenv()   # LINE 1. Always. Reads .env before all backend imports.

import os, asyncio, socket, logging, io
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pathlib import Path
import qrcode

from backend.pipeline import pipeline
from backend.modes import mode_manager
from backend.brain import brain
from backend.ocr import ocr_reader
from backend.tts import tts_engine

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

FRONTEND = Path(__file__).parent.parent / "frontend"
clients: set[WebSocket] = set()


async def broadcast(data: dict):
    """Send JSON to all connected WebSocket clients. Remove disconnected ones."""
    disconnected = set()
    for ws in clients:
        try:
            await ws.send_json(data)
        except Exception:
            disconnected.add(ws)
    clients.difference_update(disconnected)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start pipeline on startup. Stop on shutdown."""
    loop = asyncio.get_running_loop()
    pipeline.start(loop, broadcast)
    tts_engine.start(broadcast_fn=broadcast, event_loop=loop)
    logger.info("👁 [VisionTalk] Server started — event-driven pipeline ready.")
    yield
    pipeline.stop()
    logger.info("👁 [VisionTalk] Server stopped.")


app = FastAPI(lifespan=lifespan, title="VisionTalk")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Required — phone browser is a different origin
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")


@app.get("/")
async def index():
    return HTMLResponse((FRONTEND / "index.html").read_text(encoding="utf-8"))


@app.get("/qr")
async def qr_code():
    """Returns QR code PNG pointing to this server's LAN URL."""
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        ip = "localhost"
    port = os.getenv("PORT", "8000")
    img  = qrcode.make(f"http://{ip}:{port}")
    buf  = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/api/status")
async def status():
    return {"ok": True, "mode": mode_manager.snapshot(), "fps": round(pipeline.fps, 1)}


@app.websocket("/ws/guidance")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    await websocket.send_json({"type": "init", "mode": mode_manager.snapshot()})
    try:
        while True:
            data   = await websocket.receive_json()

            # ── Frame from browser camera (getUserMedia) ──────────────
            if data.get("type") == "frame":
                import base64
                import cv2
                import numpy as np
                img_b64  = data["data"]
                # Strip data-URI prefix if present ("data:image/jpeg;base64,...")
                if "," in img_b64:
                    img_b64 = img_b64.split(",", 1)[1]
                img_bytes = base64.b64decode(img_b64)
                arr  = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    logger.warning("👁 [VisionTalk] Received unparseable frame — skipping.")
                    continue
                mode = data.get("mode", "NAVIGATE")
                result = await asyncio.get_running_loop().run_in_executor(
                    None, pipeline.process_frame, frame, mode
                )
                if result:
                    await websocket.send_json(result)
                continue

            action = data.get("action", "")

            if action == "set_mode":
                new_mode = data.get("mode", "NAVIGATE")
                # Clear brain conversation history when leaving ASK mode
                old_mode = mode_manager.snapshot().get("current_mode", "NAVIGATE")
                if old_mode == "ASK" and new_mode != "ASK":
                    brain.clear_history()
                mode_manager.set_mode(new_mode)
                logger.info(f"👁 [VisionTalk] Mode switched to {new_mode}")
                # Clear OCR session history when entering READ mode
                if new_mode == "READ":
                    ocr_reader.clear_history()
                # Reset scene inventory announcement when entering NAVIGATE mode
                if new_mode == "NAVIGATE":
                    pass  # no-op: inventory announcement removed

            elif action == "ask":
                mode_manager.set_mode("ASK")
                q   = data.get("question", "").strip()
                src = data.get("input_source", "chat")   # 'voice' or 'chat'
                if q:
                    pipeline.set_question(q, input_source=src)

            elif action == "toggle_overlay":
                mode_manager.toggle_overlay()

            # NEW: Scene snapshot actions
            elif action == "snapshot":
                pipeline.take_snapshot()
                await websocket.send_json({
                    "type": "system",
                    "text": "Scene snapshot saved. Say 'What changed?' to compare."
                })
                tts_engine.speak("Scene saved. Say what changed to compare.", priority=False)

            elif action == "scene_diff":
                diff = pipeline.get_scene_diff()
                src  = data.get("input_source", "chat")
                await websocket.send_json({
                    "type":         "answer",
                    "answer":       diff,
                    "question":     "What changed?",
                    "input_source": src,
                })
                tts_engine.speak(diff, priority=True)

            elif action == "clear_history":
                brain.clear_history()
                ocr_reader.clear_history()

            # NEW: Find Mode actions
            elif action == "find_object":
                target = data.get("target", "").strip()
                if target:
                    pipeline.set_find_target(target)
                    await websocket.send_json({
                        "type": "system",
                        "text": f"Searching for: {target}"
                    })
                    tts_engine.speak(f"Looking for {target}.", priority=False)

            elif action == "set_camera":
                source = data.get("source", "").strip()
                if source:
                    pipeline.set_camera_source(source)
                    logger.info(f"👁 [VisionTalk] Camera source changed to: {source}")
                    await websocket.send_json({
                        "type": "system",
                        "text": f"Camera switched to {source}"
                    })

            elif action == "find_cancel":
                pipeline.clear_find_target()
                await websocket.send_json({
                    "type": "system",
                    "text": "Find mode cancelled."
                })

            # FIND capture flow actions
            elif action == "find_start_capture":
                # Transition to confirming state — ask user for confirmation
                pipeline.find_start_capture()
                await websocket.send_json({
                    "type":  "find_prompt",
                    "state": "confirming",
                    "text":  "Shall I capture what's in front of me?",
                })
                tts_engine.speak("Shall I capture what's in front of me?", priority=True)

            elif action == "find_capture":
                # User confirmed — freeze current frame
                import numpy as np
                # We pass None here; the pipeline will use the live frame on next cycle
                # Actually we store a sentinel and let pipeline grab current frame
                pipeline.find_capture(None)   # None = use next available live frame
                await websocket.send_json({
                    "type":  "find_prompt",
                    "state": "captured",
                    "text":  "What would you like to know?",
                })
                tts_engine.speak("What would you like to know?", priority=True)

            elif action == "find_question":
                q   = data.get("question", "").strip()
                src = data.get("input_source", "voice")
                if q:
                    pipeline.find_ask_question(q, input_source=src)

            # Repeat: client sends last banner text; server broadcasts it for browser TTS
            elif action == "speak":
                text = data.get("text", "").strip()
                if text:
                    await broadcast({"type": "speak", "text": text})

    except WebSocketDisconnect:
        clients.discard(websocket)
    except Exception as e:
        logger.error(f"👁 [VisionTalk] WS error: {e}")
        clients.discard(websocket)
