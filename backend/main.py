from dotenv import load_dotenv
load_dotenv()   # LINE 1. Always. Reads .env before all backend imports.

import os, asyncio, socket, logging, io, time
from contextlib import asynccontextmanager
from enum import Enum, auto
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
from backend.stt import stt_engine

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

FRONTEND = Path(__file__).parent.parent / "frontend"
clients: set[WebSocket] = set()

# Rate-limiting for ASK mode — prevents rapid-fire voice queries from
# flooding the pipeline and causing accessibility announcement storms.
_MIN_ASK_INTERVAL = 2.0   # seconds between processed ASK questions
_last_ask_t: float = 0.0  # monotonic timestamp of last processed ASK


# ── Voice interaction state machine ──────────────────────────────────────────
#
# States model the full lifecycle of a single voice interaction turn:
#
#   IDLE       — mic is ready; not recording (or recording silently)
#   LISTENING  — recording in progress (audio chunk being captured)
#   PROCESSING — Whisper is transcribing; command classifier running
#   SPEAKING   — TTS is actively playing; mic input is logically muted
#
# The state is updated by _stt_dispatcher() and by the TTS engine callback.
# It is broadcast to the frontend via {"type":"system","voice_state":"..."} so
# the UI can show the correct mic/speaker icon at every step.
#
# Thread safety: state changes happen only inside the asyncio event loop
# (awaited in _stt_dispatcher); reads from other threads are tolerated
# because Enum assignments are atomic in CPython.

class VoiceState(Enum):
    IDLE       = auto()
    LISTENING  = auto()
    PROCESSING = auto()
    SPEAKING   = auto()


_voice_state: VoiceState = VoiceState.IDLE
_voice_state_since: float = time.monotonic()   # initialized to now, not 0.0 (avoids false watchdog trigger)


async def broadcast(data: dict):
    """Send JSON to all connected WebSocket clients. Remove disconnected ones."""
    disconnected = set()
    for ws in clients:
        try:
            await ws.send_json(data)
        except Exception:
            disconnected.add(ws)
    clients.difference_update(disconnected)


async def _set_voice_state(state: VoiceState):
    """Update the module-level voice state and broadcast it to all clients."""
    global _voice_state, _voice_state_since
    _voice_state = state
    _voice_state_since = time.monotonic()
    await broadcast({"type": "system", "voice_state": state.name})


async def _stt_dispatcher():
    """
    Continuously drain stt_engine.cmd_q and dispatch voice commands.
    Mirrors the action-handling logic in the WebSocket handler so voice
    commands are identical in effect to chat/button commands.

    Also monitors a secondary "unrecognised" queue (stt_engine.unrecognised_q)
    and gives the user audible feedback when a speech chunk was heard but
    no command was matched.

    Runs as an asyncio background task; cancellation is silent.

    Voice state transitions per turn:
      IDLE → PROCESSING (command arrived from queue)
      PROCESSING → SPEAKING (command dispatched; TTS will speak)
      SPEAKING → IDLE (after dispatch; TTS engine is async)
    """
    from backend.diagnostics import diagnostics

    _last_spoken = ""   # for "repeat" command

    # Cooldown for "didn't catch that" feedback — prevents spam when the
    # recogniser consistently misses in a noisy environment.
    _last_unrecognised_t: float = 0.0
    _UNRECOGNISED_COOLDOWN = 5.0  # seconds

    # Help hint rotation — every Nth unrecognised response (within the cooldown
    # window) rotates to a hint message listing available commands.  This helps
    # new users discover what they can say without reading a manual.
    # The hint fires on every HELP_HINT_EVERY-th unrecognised event that passes
    # the cooldown gate, cycling through a fixed list of example phrases.
    _HELP_HINT_EVERY = 3          # hint fires on 3rd, 6th, 9th … unrecognised
    _unrecognised_count: int = 0  # total unrecognised events that passed cooldown
    _HELP_HINTS = [
        "You can say: navigate, find object, or read text.",
        "Try saying: what's ahead, or is the path clear.",
        "You can say: guide me, locate bottle, or describe this.",
        "Try saying: scan, remember this, or what changed.",
    ]

    while True:
        await asyncio.sleep(0.05)   # poll interval — low CPU cost

        # ── Drain unrecognised transcriptions ────────────────────────────
        while hasattr(stt_engine, "unrecognised_q") and not stt_engine.unrecognised_q.empty():
            try:
                stt_engine.unrecognised_q.get_nowait()
            except Exception:
                break
            diagnostics.voice_command_rejected(reason="unrecognised")
            now = time.monotonic()
            if now - _last_unrecognised_t >= _UNRECOGNISED_COOLDOWN:
                _last_unrecognised_t = now
                _unrecognised_count += 1
                # Every HELP_HINT_EVERY-th event: speak a help hint instead of
                # the plain "didn't catch that" so users learn available commands.
                if _unrecognised_count % _HELP_HINT_EVERY == 0:
                    hint = _HELP_HINTS[(_unrecognised_count // _HELP_HINT_EVERY - 1) % len(_HELP_HINTS)]
                    tts_engine.speak(hint, priority=False)
                    logger.debug("[VisionTalk] STT: unrecognised × %d → help hint.", _unrecognised_count)
                else:
                    tts_engine.speak("Sorry, I didn't catch that.", priority=False)
                    logger.debug("[VisionTalk] STT: unrecognised speech → feedback spoken.")

        # ── Drain classified commands ────────────────────────────────────
        while not stt_engine.cmd_q.empty():
            try:
                cmd = stt_engine.cmd_q.get_nowait()
            except Exception:
                break

            action = cmd.get("action", "")
            logger.info("[VisionTalk] STT command: %s", cmd)
            await _set_voice_state(VoiceState.PROCESSING)

            if action == "set_mode":
                new_mode = cmd.get("mode", "NAVIGATE")
                old_mode = mode_manager.snapshot().get("current_mode", "NAVIGATE")
                if old_mode == "ASK" and new_mode != "ASK":
                    brain.clear_history()
                mode_manager.set_mode(new_mode)
                pipeline.reset_mode()
                if new_mode == "READ":
                    ocr_reader.clear_history()
                diagnostics.voice_command_received(action="set_mode", detail=new_mode)
                diagnostics.mode_changed(old_mode=old_mode, new_mode=new_mode, source="voice")
                # Confirmation: tell user the mode switched.
                confirmations = {
                    "NAVIGATE": "Navigation mode activated.",
                    "READ":     "Reading mode activated.",
                    "SCAN":     "Scan mode activated.",
                    "ASK":      "Ask mode activated.",
                    "FIND":     "Find mode activated.",
                }
                msg = confirmations.get(new_mode, f"Switching to {new_mode.lower()} mode.")
                tts_engine.speak(msg, priority=False)
                await broadcast({"type": "system", "text": f"Voice: switched to {new_mode}"})

            elif action == "ask":
                question = cmd.get("question", "")
                if question:
                    global _last_ask_t
                    now = time.monotonic()
                    if now - _last_ask_t < _MIN_ASK_INTERVAL:
                        logger.debug("[VisionTalk] STT ASK rate-limited — too soon after last question.")
                        diagnostics.voice_command_rejected(reason="rate_limited")
                        # Notify user so the question isn't silently dropped
                        await broadcast({
                            "type":    "system",
                            "message": "Please wait a moment before asking another question.",
                        })
                        await _set_voice_state(VoiceState.IDLE)
                        continue
                    _last_ask_t = now
                    diagnostics.voice_command_received(action="ask", detail=question[:60])
                    mode_manager.set_mode("ASK")
                    last_frame = pipeline.get_last_frame()  # may be None — handled gracefully
                    logger.info(
                        "[VisionTalk] STT ask: q=%r frame=%s",
                        question, "present" if last_frame is not None else "None",
                    )
                    loop = asyncio.get_running_loop()
                    try:
                        result = await loop.run_in_executor(
                            None, pipeline.process_ask_direct,
                            last_frame, question, "voice"
                        )
                    except Exception as exc:
                        logger.exception("[VisionTalk] STT ask executor error: %s", exc)
                        result = None
                    logger.info("[VisionTalk] STT ask result: %s", result)
                    if result:
                        await broadcast(result)
                        _last_spoken = result.get("answer", _last_spoken)

            elif action == "find_object":
                target = cmd.get("target", "")
                if target:
                    diagnostics.voice_command_received(action="find_object", detail=target)
                    pipeline.set_find_target(target)
                    tts_engine.speak(f"Looking for {target}.", priority=False)
                    await broadcast({"type": "system", "text": f"Searching for: {target}"})
                    last_frame = pipeline.get_last_frame()
                    if last_frame is not None:
                        loop = asyncio.get_running_loop()
                        result = await loop.run_in_executor(
                            None, pipeline.process_frame, last_frame, "FIND"
                        )
                        if result:
                            await broadcast(result)

            elif action == "nav_destination":
                dest = cmd.get("destination", "").strip()
                if dest:
                    diagnostics.voice_command_received(action="nav_destination", detail=dest)
                    mode_manager.set_mode("NAVIGATE")  # ensure pipeline runs in NAVIGATE
                    pipeline.set_nav_destination(dest)
                    tts_engine.speak(f"Navigating to {dest}. I'll guide you there.", priority=True)
                    await broadcast({"type": "system", "text": f"Navigating to: {dest}"})
                    logger.info("[VisionTalk] STT nav_destination: %r", dest)

            elif action == "snapshot":
                diagnostics.voice_command_received(action="snapshot")
                pipeline.take_snapshot()
                tts_engine.speak("Scene saved.", priority=False)
                await broadcast({"type": "system", "text": "Scene snapshot saved."})

            elif action == "scene_diff":
                diagnostics.voice_command_received(action="scene_diff")
                diff = pipeline.get_scene_diff()
                tts_engine.speak(diff, priority=True)
                await broadcast({
                    "type":         "answer",
                    "answer":       diff,
                    "question":     "What changed?",
                    "input_source": "voice",
                })
                _last_spoken = diff

            elif action == "repeat":
                diagnostics.voice_command_received(action="repeat")
                if _last_spoken:
                    tts_engine.speak(_last_spoken, priority=True)
                else:
                    tts_engine.speak("Nothing to repeat.", priority=False)

            await _set_voice_state(VoiceState.SPEAKING)
            # Transition back to IDLE after a short delay — actual TTS
            # duration is handled browser-side, so we just leave a brief
            # window before re-enabling mic-state indicators.
            await asyncio.sleep(0.3)
            await _set_voice_state(VoiceState.IDLE)


async def _voice_state_watchdog():
    """
    Background task: if voice state is stuck in PROCESSING or SPEAKING for more
    than 10 seconds (e.g. due to a network error or unhandled exception in the
    dispatcher), reset it to IDLE so the mic becomes usable again.
    """
    _WATCHDOG_TIMEOUT = 10.0   # seconds before forced reset
    while True:
        await asyncio.sleep(2.0)
        if _voice_state in (VoiceState.PROCESSING, VoiceState.SPEAKING):
            stuck_for = time.monotonic() - _voice_state_since
            if stuck_for > _WATCHDOG_TIMEOUT:
                logger.warning(
                    "[VisionTalk] Voice state watchdog: %s stuck for %.1fs — resetting to IDLE.",
                    _voice_state.name, stuck_for,
                )
                await _set_voice_state(VoiceState.IDLE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start pipeline on startup. Stop on shutdown."""
    loop = asyncio.get_running_loop()
    pipeline.start(loop, broadcast)
    tts_engine.start(broadcast_fn=broadcast, event_loop=loop)
    stt_engine.start()
    # Background tasks: drain STT command queue and run voice state watchdog
    stt_task      = asyncio.create_task(_stt_dispatcher())
    watchdog_task = asyncio.create_task(_voice_state_watchdog())
    logger.info("👁 [VisionTalk] Server started — event-driven pipeline ready.")
    yield
    stt_task.cancel()
    watchdog_task.cancel()
    pipeline.stop()
    stt_engine.stop()
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


@app.get("/api/diagnostics")
async def get_diagnostics():
    from backend.diagnostics import diagnostics
    return diagnostics.summary()


@app.get("/api/scene")
async def get_scene():
    from backend.tracker import object_tracker
    from backend.scene_memory import build_scene_graph
    return build_scene_graph(object_tracker.all_tracks())


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
                    await broadcast(result)
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
                # Reset tracker + stability state on every mode change
                pipeline.reset_mode()
                # Clear OCR session history when entering READ mode
                if new_mode == "READ":
                    ocr_reader.clear_history()
                from backend.diagnostics import diagnostics
                diagnostics.mode_changed(old_mode=old_mode, new_mode=new_mode, source="ui")

            elif action == "ask":
                q   = data.get("question", "").strip()
                src = data.get("input_source", "chat")   # 'voice' or 'chat'
                if q:
                    global _last_ask_t
                    now = time.monotonic()
                    # Rate-limit voice ASK to prevent accessibility announcement
                    # storms from continuous voice mode restart loops.
                    if src == "voice" and now - _last_ask_t < _MIN_ASK_INTERVAL:
                        logger.debug("[VisionTalk] WS ASK rate-limited (voice) — too soon.")
                    else:
                        _last_ask_t = now
                        mode_manager.set_mode("ASK")
                        # Use process_ask_direct() to bypass the shared
                        # _pending_question slot and eliminate the race
                        # condition with concurrent NAVIGATE frame processing.
                        # last_frame may be None — process_ask_direct handles it.
                        last_frame = pipeline.get_last_frame()
                        logger.info(
                            "[VisionTalk] WS ask: q=%r src=%r frame=%s",
                            q, src, "present" if last_frame is not None else "None",
                        )
                        try:
                            result = await asyncio.get_running_loop().run_in_executor(
                                None, pipeline.process_ask_direct, last_frame, q, src
                            )
                        except Exception as exc:
                            logger.exception("[VisionTalk] WS ask executor error: %s", exc)
                            result = None
                        logger.info("[VisionTalk] WS ask result: %s", result)
                        if result:
                            await broadcast(result)

            elif action == "toggle_overlay":
                mode_manager.toggle_overlay()

            # NEW: Scene snapshot actions
            elif action == "snapshot":
                pipeline.take_snapshot()
                await broadcast({
                    "type": "system",
                    "text": "Scene snapshot saved. Say 'What changed?' to compare."
                })
                tts_engine.speak("Scene saved. Say what changed to compare.", priority=False)

            elif action == "scene_diff":
                diff = pipeline.get_scene_diff()
                src  = data.get("input_source", "chat")
                await broadcast({
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
                    await broadcast({
                        "type": "system",
                        "text": f"Searching for: {target}"
                    })
                    tts_engine.speak(f"Looking for {target}.", priority=False)
                    # Immediately process with the last cached frame so the
                    # answer arrives without waiting for the next FIND frame.
                    last_frame = pipeline.get_last_frame()
                    if last_frame is not None:
                        result = await asyncio.get_running_loop().run_in_executor(
                            None, pipeline.process_frame, last_frame, "FIND"
                        )
                        if result:
                            await broadcast(result)
                    else:
                        await websocket.send_json({
                            "type": "system",
                            "text": "Camera not ready yet. Please wait for the camera to start.",
                        })

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
                await broadcast({
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
                await broadcast({
                    "type":  "find_prompt",
                    "state": "captured",
                    "text":  "What would you like to know?",
                })
                tts_engine.speak("What would you like to know?", priority=True)

            elif action == "find_question":
                q   = data.get("question", "").strip()
                src = data.get("input_source", "chat")   # default "chat" — find_question is always UI-originated
                if q:
                    # Ensure we are in FIND mode so process_frame dispatches correctly
                    mode_manager.set_mode("FIND")
                    pipeline.find_ask_question(q, input_source=src)
                    # Process immediately using the last cached frame — do not wait
                    # for the next camera frame (which may never arrive in FIND mode)
                    last_frame = pipeline.get_last_frame()
                    logger.info(
                        "[VisionTalk] find_question: q=%r src=%r frame=%s",
                        q, src, "present" if last_frame is not None else "None",
                    )
                    if last_frame is not None:
                        try:
                            result = await asyncio.get_running_loop().run_in_executor(
                                None, pipeline.process_frame, last_frame, "FIND"
                            )
                        except Exception as exc:
                            logger.exception("[VisionTalk] find_question executor error: %s", exc)
                            result = None
                        if result:
                            await broadcast(result)
                    else:
                        # No frame yet — return an answer message so it shows in the
                        # conversation panel (system messages don't show there)
                        await broadcast({
                            "type":         "answer",
                            "answer":       "Camera is not ready yet. Please wait a moment for the camera to start, then try again.",
                            "question":     q,
                            "input_source": src,
                        })

            elif action == "nav_destination":
                dest = data.get("destination", "").strip()
                if dest:
                    mode_manager.set_mode("NAVIGATE")  # ensure pipeline runs in NAVIGATE
                    pipeline.set_nav_destination(dest)
                    await broadcast({
                        "type": "system",
                        "text": f"Navigating to: {dest}",
                    })
                    tts_engine.speak(f"Navigating to {dest}. I'll guide you there.", priority=True)
                    logger.info("[VisionTalk] nav_destination set: %r", dest)

            # Repeat: client sends last banner text; server broadcasts it for browser TTS
            elif action == "speak":
                text = data.get("text", "").strip()
                if text:
                    await broadcast({"type": "speak", "text": text})

            # Push-to-talk / software mute control.
            # The frontend sends these when the user presses/releases the mic
            # button.  Backend mutes/unmutes the STT record thread so no audio
            # is queued while the button is not held.  This prevents background
            # noise from triggering unintended voice commands.
            elif action == "stt_mute":
                stt_engine.muted = True
                await websocket.send_json({"type": "system", "voice_state": "IDLE", "stt_muted": True})
                logger.debug("[VisionTalk] STT muted via WebSocket.")

            elif action == "stt_unmute":
                stt_engine.muted = False
                await websocket.send_json({"type": "system", "voice_state": "LISTENING", "stt_muted": False})
                logger.debug("[VisionTalk] STT unmuted via WebSocket.")

    except WebSocketDisconnect:
        clients.discard(websocket)
    except Exception as e:
        logger.error(f"👁 [VisionTalk] WS error: {e}")
        clients.discard(websocket)
