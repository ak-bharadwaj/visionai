import time, threading, asyncio, logging, queue
from typing import Callable
import numpy as np

from backend.modes import mode_manager
from backend.camera import get_camera
from backend.detector import ObjectDetector
from backend.depth import depth_estimator
from backend.ocr import ocr_reader
from backend.spatial import spatial_analyzer
from backend.scene_memory import scene_memory
from backend.narrator import narrator
from backend.brain import brain
from backend.tts import tts_engine
from backend.world_detector import world_detector, WORLD_N

logger = logging.getLogger(__name__)

DEPTH_N = int(__import__("os").getenv("DEPTH_EVERY_N_FRAMES", "3"))
OCR_N   = 5
FPS_CAP = int(__import__("os").getenv("INFERENCE_FPS", "10"))


class PipelineRunner:
    def __init__(self):
        self._running   = False
        self._camera    = None
        self._detector  = None
        self._loop:  asyncio.AbstractEventLoop | None = None
        self._bcast: Callable | None = None
        self.fps = 0.0
        self._q_lock            = threading.Lock()
        self._pending_question: str | None = None
        self._input_source: str = "chat"   # 'voice' or 'chat'
        # Scene Snapshot
        self._snapshot: dict | None   = None
        # FIND mode: 4-state capture machine
        self._find_capture_state: str = "idle"   # idle | confirming | captured
        self._captured_frame = None              # frozen frame for find Q&A
        # READ mode: last read texts for follow-up Q&A
        self._last_read_texts: list[str] = []
        # READ no-text counter
        self._no_text_count: int = 0
        # FIX #12: MiDaS depth runs in a background thread (stale-ok pattern)
        self._depth_input:  queue.Queue = queue.Queue(maxsize=1)
        self._depth_output: queue.Queue = queue.Queue(maxsize=1)

    # ── Public API (called from WS handler, thread-safe) ──────────
    def set_question(self, question: str, input_source: str = "chat"):
        """input_source: 'voice' or 'chat' — controls whether frontend plays TTS."""
        with self._q_lock:
            self._pending_question = question
            self._input_source = input_source

    def take_snapshot(self):
        """Called when user says 'remember this'."""
        with self._q_lock:
            self._snapshot = scene_memory.get_snapshot()

    def get_scene_diff(self) -> str:
        """Compare current scene to stored snapshot."""
        with self._q_lock:
            snap = self._snapshot

        if snap is None:
            return "No snapshot taken. Say 'remember this' first."

        current = scene_memory.get_snapshot()
        appeared    = [v for k, v in current.items() if k not in snap]
        disappeared = [v for k, v in snap.items()     if k not in current]
        parts = []
        if appeared:    parts.append(f"New objects: {', '.join(appeared)}")
        if disappeared: parts.append(f"Gone: {', '.join(disappeared)}")
        return ". ".join(parts) if parts else "Scene unchanged since snapshot."

    # ── FIND capture state machine ─────────────────────────────────
    def find_start_capture(self):
        """Transition to confirming state — TTS prompt is sent by WS handler."""
        with self._q_lock:
            self._find_capture_state = "confirming"

    def find_capture(self, frame=None):
        """
        Freeze a frame and transition to captured state.
        If frame is None, transitions to 'capturing' and the pipeline
        will freeze the next live frame automatically.
        """
        with self._q_lock:
            if frame is not None:
                self._captured_frame     = frame
                self._find_capture_state = "captured"
            else:
                self._find_capture_state = "capturing"   # pipeline freezes next frame

    def find_ask_question(self, question: str, input_source: str = "chat"):
        """Store question to be answered against the captured frame."""
        with self._q_lock:
            self._pending_question = question
            self._input_source     = input_source

    def reset_find_capture(self):
        """Return FIND capture state machine to idle."""
        with self._q_lock:
            self._find_capture_state = "idle"
            self._captured_frame     = None

    def clear_find_target(self):
        """Alias for reset_find_capture — called by find_cancel WS action."""
        self.reset_find_capture()

    # ── Lifecycle ──────────────────────────────────────────────────
    def start(self, event_loop: asyncio.AbstractEventLoop, broadcast_fn: Callable):
        self._camera   = get_camera()
        self._detector = ObjectDetector()
        depth_estimator.load()
        world_detector.load()   # NEW: load YOLOWorld secondary detector
        # FIX #5: preload PaddleOCR in background so first READ frame doesn't freeze
        threading.Thread(
            target=ocr_reader._ensure_loaded,
            daemon=True,
            name="OCRPreload"
        ).start()
        # FIX #12: depth runs in its own daemon thread — pipeline reads last result (stale-ok)
        threading.Thread(
            target=self._depth_worker,
            daemon=True,
            name="DepthThread"
        ).start()
        tts_engine.start()
        self._loop  = event_loop
        self._bcast = broadcast_fn
        self._running = True
        threading.Thread(target=self._run, daemon=True, name="PipelineThread").start()
        logger.info("👁 [VisionTalk] Pipeline started.")

    def _depth_worker(self):
        """FIX #12: Consumes frames from _depth_input, puts results in _depth_output."""
        while True:
            frame = self._depth_input.get()   # blocks until a frame is queued
            if frame is None:
                break   # sentinel — shut down
            result = depth_estimator.estimate(frame)
            # Drain old result (if pipeline hasn't consumed it yet) before writing new one
            try:
                self._depth_output.get_nowait()
            except queue.Empty:
                pass
            self._depth_output.put(result)

    def set_camera_source(self, source: str):
        """Hot-swap the camera stream source at runtime."""
        if self._camera:
            self._camera.set_source(source)

    def stop(self):
        self._running = False
        if self._camera:
            self._camera.stop()

    # ── Main loop ──────────────────────────────────────────────────
    def _run(self):
        frame_count  = 0
        depth_map    = None    # MUST be above while loop
        nav_msg      = ""      # FIXED: separate from mode-crossing msg contamination
        read_msg     = ""      # FIXED: persists last OCR result between non-OCR frames

        while self._running:
            t0 = time.time()

            frame = self._camera.read()
            if frame is None:
                time.sleep(0.05)
                continue

            h, w   = frame.shape[:2]
            frame_count += 1

            # YOLO — every frame
            detections = self._detector.detect(frame)

            # MiDaS — every N frames (non-blocking: queue frame to depth thread)
            if frame_count % DEPTH_N == 0:
                try:
                    self._depth_input.put_nowait(frame)
                except queue.Full:
                    pass   # depth thread still busy — skip this frame (stale-ok)
            # Pull latest depth result (non-blocking — uses last valid map if none ready)
            try:
                depth_map = self._depth_output.get_nowait()
            except queue.Empty:
                pass   # keep using previous depth_map

            # YOLOWorld — every WORLD_N frames (extra class detection)
            # Guard: only run in NAVIGATE mode — avoids interrupting OCR/LLM TTS
            if mode_manager.is_navigate() and frame_count % WORLD_N == 0:
                extra_dets = world_detector.detect(frame)
                for d in extra_dets:
                    # Compute direction from bounding box centre
                    cx = (d["x1"] + d["x2"]) // 2
                    if cx < w // 3:
                        _dir = "left"
                    elif cx > 2 * w // 3:
                        _dir = "right"
                    else:
                        _dir = "ahead"
                    # Compute rough distance level from bounding box area ratio
                    bb_area    = (d["x2"] - d["x1"]) * (d["y2"] - d["y1"])
                    frame_area = h * w
                    ratio      = bb_area / max(frame_area, 1)
                    dist_level = 1 if ratio > 0.30 else (2 if ratio > 0.10 else 3)
                    dist_label = {1: "very close", 2: "nearby", 3: "a few steps away"}.get(dist_level, "nearby")
                    # Only announce if it's a new detection (scene_memory dedup)
                    key = d["class_name"] + _dir
                    if scene_memory.is_new_by_key(key):
                        msg = (
                            f"{d['class_name'].title()} {dist_label}, "
                            f"{'directly ' if _dir == 'ahead' else 'to your '}"
                            f"{_dir}"
                        )
                        tts_engine.speak(msg, priority=(dist_level == 1))

            # Stair/drop — NAVIGATE only (suppress during READ/ASK to avoid disruption)
            if mode_manager.is_navigate() and depth_estimator.detect_stair_drop(depth_map, h, w):
                tts_engine.speak("Warning — possible step or drop ahead", priority=True)

            # Spatial analysis
            spatial_results = [
                spatial_analyzer.analyze(d, w, h, depth_map) for d in detections
            ]

            # ═══════════════════════════════════════════
            # MODE: FIND (4-state capture flow)
            # ═══════════════════════════════════════════
            if mode_manager.is_find():
                with self._q_lock:
                    find_state   = self._find_capture_state
                    find_frame   = self._captured_frame
                    question     = self._pending_question
                    input_source = self._input_source
                    self._pending_question = None

                # Auto-freeze the live frame when state is 'capturing'
                if find_state == "capturing":
                    with self._q_lock:
                        self._captured_frame     = frame
                        self._find_capture_state = "captured"
                    find_state  = "captured"
                    find_frame  = frame

                if find_state == "captured" and question:
                    # Answer question against the frozen captured frame
                    answer = brain.answer(question, find_frame, spatial_results, [])
                    tts_engine.speak(answer, priority=True)
                    self._send({
                        "type":         "answer",
                        "mode":         "FIND",
                        "question":     question,
                        "answer":       answer,
                        "input_source": input_source,
                        "frame_w":      w,
                        "frame_h":      h,
                        "fps":          round(self.fps, 1),
                    })
                    # Reset back to idle after answering
                    with self._q_lock:
                        self._find_capture_state = "idle"
                        self._captured_frame     = None

                self._tick(t0)
                continue

            # ═══════════════════════════════════════════
            # MODE: READ
            # ═══════════════════════════════════════════
            if mode_manager.is_read():
                # Handle follow-up question about what was just read
                with self._q_lock:
                    question     = self._pending_question
                    input_source = self._input_source
                    self._pending_question = None
                    last_texts   = list(self._last_read_texts)

                if question:
                    answer = brain.answer(question, frame, spatial_results, last_texts)
                    tts_engine.speak(answer, priority=True)
                    self._send({
                        "type":         "answer",
                        "mode":         "READ",
                        "question":     question,
                        "answer":       answer,
                        "input_source": input_source,
                        "frame_w":      w,
                        "frame_h":      h,
                        "fps":          round(self.fps, 1),
                    })
                    self._tick(t0)
                    continue

                if frame_count % OCR_N == 0:
                    texts = ocr_reader.read(frame, deduplicate=True)
                    if texts:
                        self._no_text_count = 0
                        with self._q_lock:
                            self._last_read_texts = texts
                        raw = ". ".join(texts[:3])
                        if len(raw) > 200:
                            raw = raw[:197] + "..."
                        read_msg = "Reading: " + raw
                        tts_engine.speak(read_msg)
                    else:
                        self._no_text_count += 1
                        if self._no_text_count % 5 == 1:
                            tts_engine.speak(
                                "No text in view. Try moving closer or improving the lighting.",
                                priority=False
                            )
                        read_msg = "No text found. Move closer or adjust angle."
                # no else — read_msg persists from last OCR frame

                # FIX #21: don't broadcast on first READ frame before any OCR has run
                if read_msg:
                    self._send({
                        "type": "reading",
                        "mode": "READ",
                        "text": read_msg,
                        "detections": [self._serial(r) for r in spatial_results],
                        "frame_w": w,
                        "frame_h": h,
                        "fps": round(self.fps, 1),
                    })
                self._tick(t0)
                continue

            # ═══════════════════════════════════════════
            # MODE: ASK
            # ═══════════════════════════════════════════
            if mode_manager.is_ask():
                with self._q_lock:
                    question     = self._pending_question
                    input_source = self._input_source
                    self._pending_question = None

                if question:
                    # Only say "Let me check that" when LLM will actually be called.
                    # Instant rule-based answers (color, scene, people, safety) are
                    # fast enough that the holding message would sound out of place.
                    needs_llm = brain.needs_llm(question)
                    if needs_llm:
                        tts_engine.speak("Let me check that.", priority=True)

                    # OCR is expensive (300ms–2s on CPU). Only run it on question
                    # paths that actually use the text: LLM general path, medicine,
                    # and door questions. All other fast-path routes ignore texts=[].
                    q_lower_ocr = question.lower()
                    if (needs_llm
                            or any(k in q_lower_ocr for k in ("door", "open", "closed", "entrance", "exit"))
                            or any(k in q_lower_ocr.split() for k in ("medicine", "medication", "drug",
                                                                       "pill", "tablet", "capsule",
                                                                       "dose", "dosage", "prescription",
                                                                       "label", "bottle", "packet",
                                                                       "paracetamol", "ibuprofen", "aspirin"))):
                        texts = ocr_reader.read(frame)
                    else:
                        texts = []  # instant fast-path — OCR not needed

                    # FIXED: pass spatial_results directly (not dead det_list)
                    answer = brain.answer(question, frame, spatial_results, texts)

                    # Always speak via TTS (server-side voice output)
                    tts_engine.speak(answer, priority=True)

                    self._send({
                        "type":         "answer",
                        "mode":         "ASK",
                        "question":     question,
                        "answer":       answer,
                        "input_source": input_source,   # 'voice' or 'chat'
                        "context":      [f"{r.class_name} ({r.direction}, {r.distance})"
                                         for r in spatial_results[:5]],
                        "frame_w":      w,
                        "frame_h":      h,
                        "fps":          round(self.fps, 1),
                    })

                    # DO NOT auto-return to NAVIGATE — stay in ASK for multi-turn.
                    # User must explicitly switch mode (voice 'navigate' or mode button).

                self._tick(t0)
                continue

            # ═══════════════════════════════════════════
            # MODE: NAVIGATE (default)
            # ═══════════════════════════════════════════
            # Reset nav_msg on every NAVIGATE entry so stale messages
            # from a prior mode switch never bleed through to this frame.
            nav_msg = ""
            scene_memory.update(spatial_results)

            # NAVIGATE in-mode questions: answer without switching to ASK mode
            with self._q_lock:
                nav_question     = self._pending_question
                nav_input_source = self._input_source
                self._pending_question = None
            if nav_question:
                answer = brain.answer(nav_question, frame, spatial_results, [])
                tts_engine.speak(answer, priority=True)
                self._send({
                    "type":         "answer",
                    "mode":         "NAVIGATE",
                    "question":     nav_question,
                    "answer":       answer,
                    "input_source": nav_input_source,
                    "context":      [f"{r.class_name} ({r.direction}, {r.distance})"
                                     for r in spatial_results[:5]],
                    "frame_w":      w,
                    "frame_h":      h,
                    "fps":          round(self.fps, 1),
                })
                self._tick(t0)
                continue

            # Check approaching objects FIRST (highest priority)
            approaching = scene_memory.get_approaching()
            if approaching:
                obj = approaching[0]   # most dangerous approaching object
                approach_msg = narrator.narrate_approaching(obj)
                tts_engine.speak(approach_msg, priority=True)
                scene_memory.mark_approach_warned(obj.key)
                nav_msg = approach_msg

            # Person count / social awareness (level 1 only)
            person_msg = narrator.narrate_persons(spatial_results)
            if person_msg and person_msg != nav_msg:
                tts_engine.speak(person_msg, priority=True)
                nav_msg = person_msg

            # Standard new-object alerts (danger-only filter via prioritize)
            new_alerts  = scene_memory.get_new_alerts()
            prioritized = narrator.prioritize(new_alerts)

            if prioritized:
                best    = prioritized[0]
                nav_msg = narrator.narrate(best)
                if nav_msg:
                    tts_engine.speak(nav_msg, priority=(best.distance_level == 1))
                for a in prioritized:
                    scene_memory.mark_announced(a.key)

            self._send({
                "type":         "narration",
                "mode":         "NAVIGATE",
                "text":         nav_msg,
                "severity":     prioritized[0].distance_level if prioritized else 0,
                "detections":   [self._serial(r) for r in spatial_results],
                "show_overlay": mode_manager.snapshot()["show_overlay"],
                "frame_w":      w,
                "frame_h":      h,
                "fps":          round(self.fps, 1),
            })

            self._tick(t0)

    def _serial(self, r) -> dict:
        return {
            "class_name":     r.class_name,
            "confidence":     round(r.confidence, 2),
            "direction":      r.direction,
            "distance":       r.distance,
            "distance_level": r.distance_level,
            "distance_m":     round(r.distance_m, 1) if r.distance_m else 0.0,
            "x1": r.x1, "y1": r.y1, "x2": r.x2, "y2": r.y2,
        }

    def _send(self, data: dict):
        if self._loop and self._bcast:
            asyncio.run_coroutine_threadsafe(self._bcast(data), self._loop)

    def _tick(self, t0: float):
        elapsed = time.time() - t0
        self.fps = min(1.0 / elapsed, FPS_CAP) if elapsed > 0 else FPS_CAP
        time.sleep(max(0, (1 / FPS_CAP) - elapsed))


pipeline = PipelineRunner()
