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
    """Start pipeline on startup. Stop on shutdown. Run FPS watchdog."""
    loop = asyncio.get_running_loop()
    pipeline.start(loop, broadcast)
    logger.info("👁 [VisionTalk] Server started — pipeline running.")
    watchdog_task = asyncio.create_task(_fps_watchdog())
    yield
    watchdog_task.cancel()
    pipeline.stop()
    logger.info("👁 [VisionTalk] Server stopped.")


async def _fps_watchdog():
    """
    Background task that monitors pipeline FPS.
    If FPS stays at 0 for more than 10 consecutive seconds while the pipeline
    is supposed to be running, it attempts a self-heal: stop then restart.
    """
    _STALL_LIMIT = 10.0   # seconds of zero-FPS before restart
    stall_since: float | None = None

    while True:
        await asyncio.sleep(2.0)
        fps = pipeline.fps
        if fps == 0:
            now = asyncio.get_event_loop().time()
            if stall_since is None:
                stall_since = now
            elif now - stall_since >= _STALL_LIMIT:
                logger.critical(
                    "👁 [VisionTalk] WATCHDOG: Pipeline FPS has been 0 for "
                    f"{_STALL_LIMIT:.0f}s — attempting self-heal restart."
                )
                try:
                    loop = asyncio.get_running_loop()
                    pipeline.stop()
                    await asyncio.sleep(1.0)
                    pipeline.start(loop, broadcast)
                    logger.info("👁 [VisionTalk] WATCHDOG: Pipeline restarted successfully.")
                except Exception as e:
                    logger.error(f"👁 [VisionTalk] WATCHDOG: Restart failed: {e}")
                stall_since = None  # reset regardless so we don't tight-loop restarts
        else:
            stall_since = None  # FPS is healthy — clear stall timer


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
                    pipeline.reset_inventory()

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

            # Repeat: client sends last banner text; server just re-speaks it via TTS
            elif action == "speak":
                text = data.get("text", "").strip()
                if text:
                    tts_engine.speak(text, priority=True)

    except WebSocketDisconnect:
        clients.discard(websocket)
    except Exception as e:
        logger.error(f"👁 [VisionTalk] WS error: {e}")
        clients.discard(websocket)
