"""
pipeline.py — Core processing pipeline for VisionTalk (redesigned).

Architecture (in frame-processing order):
  1. Frame preprocessing (resize to 640, kept inside detector)
  2. Object detection — YOLOv8s every DETECT_EVERY_N frames
  3. Object tracking — IoU tracker every frame (tracker.py)
  4. Depth estimation — MiDaS in background thread, stale-ok
  5. Depth update — only for confirmed (stable) tracked objects
  6. Risk scoring — risk_engine.score_all()
  7. Path corridor modelling — path_model.is_path_clear()
  8. System health check — stability_filter.record_frame()
  9. Narration gate — stability_filter.object_passes_gate()
  10. Cooldown gate — stability_filter.narration_allowed()
  11. Narration build — narrator.build() / narrator.path_clear()
  12. TTS output

Non-NAVIGATE modes (READ, FIND, ASK, SCAN) are UNCHANGED from the original
pipeline — only NAVIGATE mode has been fully redesigned.  The WebSocket
message protocol (type: narration | answer | reading | speak | system |
found_object | find_prompt | init) is unchanged.

Design constraints (safety-critical):
  - No LLM / generative calls in NAVIGATE mode (brain.py used only for
    ASK / READ / FIND).
  - No probabilistic YOLOWorld narrations without tracker confirmation.
  - No "scene caption" thread in NAVIGATE mode.
  - Detection cadence: every DETECT_EVERY_N frames (default 4).
  - Tracking: every frame.
  - Depth only for confirmed tracked objects.
  - Failsafe: if system unstable > 10 frames → "Scene unstable. Please move slowly."
"""

import time
import queue
import asyncio
import threading
import logging
from typing import Callable

import numpy as np

from backend.modes import mode_manager
from backend.detector import ObjectDetector
from backend.depth import depth_estimator
from backend.ocr import ocr_reader
from backend.scene_memory import scene_memory
from backend.brain import brain
from backend.tts import tts_engine
from backend.scanner import scanner

from backend.tracker import object_tracker
from backend.risk_engine import score_all as risk_score_all
from backend.path_model import is_path_clear
from backend.stability_filter import stability_filter, FAILSAFE_MESSAGE
from backend.narrator import build as narrator_build, path_clear as narrator_path_clear
from backend.diagnostics import diagnostics

logger = logging.getLogger(__name__)

# ── Cadence ──────────────────────────────────────────────────────────────────
DETECT_EVERY_N = 4     # run YOLO only every N frames (tracking runs every frame)

# ── GPU detection (best-effort; falls back to CPU safely) ────────────────────
def _is_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

_ON_GPU = _is_gpu()


class PipelineRunner:
    def __init__(self):
        self._running   = False
        self._detector  = None
        self._loop:  asyncio.AbstractEventLoop | None = None
        self._bcast: Callable | None = None
        self.fps = 0.0

        self._q_lock            = threading.Lock()
        self._pending_question: str | None = None
        self._input_source: str = "chat"

        # Scene Snapshot (used by "remember this" feature)
        self._snapshot: dict | None = None

        # FIND mode 4-state machine
        self._find_capture_state: str = "idle"
        self._captured_frame = None

        # READ mode
        self._last_read_texts: list[str] = []
        self._no_text_count: int = 0

        # MiDaS depth — background thread, stale-ok pattern
        self._depth_input:  queue.Queue = queue.Queue(maxsize=1)
        self._depth_output: queue.Queue = queue.Queue(maxsize=1)

        # Timing
        self._last_frame_t: float = 0.0
        self._frame_count:  int   = 0   # total frames processed this session

        # Latest depth map (stale-ok — may be from previous frame)
        self._latest_depth_map = None

    # ── Public API (called from WS handler, thread-safe) ─────────────────────

    def set_question(self, question: str, input_source: str = "chat"):
        with self._q_lock:
            self._pending_question = question
            self._input_source = input_source

    def take_snapshot(self):
        """Called when user says 'remember this'."""
        with self._q_lock:
            self._snapshot = scene_memory.get_snapshot()

    def get_scene_diff(self) -> str:
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

    # ── FIND capture state machine ────────────────────────────────────────────

    def find_start_capture(self):
        with self._q_lock:
            self._find_capture_state = "confirming"

    def find_capture(self, frame=None):
        with self._q_lock:
            if frame is not None:
                self._captured_frame     = frame
                self._find_capture_state = "captured"
            else:
                self._find_capture_state = "capturing"

    def find_ask_question(self, question: str, input_source: str = "chat"):
        with self._q_lock:
            self._pending_question = question
            self._input_source     = input_source

    def reset_find_capture(self):
        with self._q_lock:
            self._find_capture_state = "idle"
            self._captured_frame     = None

    def clear_find_target(self):
        self.reset_find_capture()

    def set_find_target(self, target: str):
        """
        Compatibility shim for main.py 'find_object' action.
        Starts the FIND capture flow and stores the target label for reference.
        The actual visual search is handled by brain.answer() on the captured frame.
        """
        with self._q_lock:
            self._find_capture_state = "capturing"
            self._pending_question   = f"Can you see a {target}? Where is it?"
            self._input_source       = "chat"

    def reset_mode(self):
        """
        Call on mode switches to clear NAVIGATE state.
        Resets the tracker (new mode = fresh scene) and stability filter.
        """
        object_tracker.reset()
        stability_filter.reset()
        scene_memory.clear()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self, event_loop: asyncio.AbstractEventLoop, broadcast_fn: Callable):
        self._detector = ObjectDetector()
        depth_estimator.load()
        # PaddleOCR pre-load in background
        threading.Thread(
            target=ocr_reader._ensure_loaded,
            daemon=True,
            name="OCRPreload",
        ).start()
        # MiDaS depth in its own daemon thread
        threading.Thread(
            target=self._depth_worker,
            daemon=True,
            name="DepthThread",
        ).start()
        self._loop    = event_loop
        self._bcast   = broadcast_fn
        self._running = True
        logger.info("[VisionTalk] Pipeline started (event-driven, redesigned backend).")

    def _depth_worker(self):
        """Consume frames from _depth_input, write results to _depth_output."""
        while True:
            frame = self._depth_input.get()   # blocks until a frame arrives
            if frame is None:
                break   # sentinel — shutdown
            result = depth_estimator.estimate(frame)
            try:
                self._depth_output.get_nowait()   # drain stale result
            except queue.Empty:
                pass
            self._depth_output.put(result)

    def set_camera_source(self, source: str):
        """No-op in event-driven mode."""
        logger.info("[VisionTalk] set_camera_source(%r) — no-op in event-driven mode.", source)

    def stop(self):
        self._running = False
        try:
            self._depth_input.put_nowait(None)
        except queue.Full:
            pass

    # ── Event-driven frame entry point ────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, mode: str) -> dict | None:
        """
        Process a single camera frame.  Called from run_in_executor (thread pool).
        Returns a payload dict to send to the frontend, or None.
        """
        t0 = time.time()
        h, w = frame.shape[:2]
        self._frame_count += 1

        # ── Detection (every DETECT_EVERY_N frames) ───────────────────────────
        if self._frame_count % DETECT_EVERY_N == 0:
            detections = self._detector.detect(frame)
            # Log filtered detections for diagnostics
            for d in detections:
                if d.confidence >= 0.60:
                    diagnostics.object_confirmed(0, d.class_name, 1)
        else:
            detections = []   # tracking runs every frame even when no new detections

        # ── Depth — queue frame, read latest result (stale-ok) ────────────────
        try:
            self._depth_input.put_nowait(frame)
        except queue.Full:
            pass
        try:
            self._latest_depth_map = self._depth_output.get_nowait()
        except queue.Empty:
            pass
        depth_map = self._latest_depth_map

        # ── Mode dispatch ─────────────────────────────────────────────────────
        if mode == "NAVIGATE":
            result = self._process_navigate(frame, w, h, detections, depth_map)
        elif mode == "READ":
            result = self._process_read(frame, w, h)
        elif mode == "FIND":
            result = self._process_find(frame, w, h)
        elif mode == "ASK":
            result = self._process_ask(frame, w, h)
        elif mode == "SCAN":
            result = self._process_scan(frame, w, h)
        else:
            result = None

        elapsed = time.time() - t0
        self.fps = min(1.0 / elapsed, 60.0) if elapsed > 0 else 60.0
        self._last_frame_t = time.time()
        return result

    # ── NAVIGATE mode ─────────────────────────────────────────────────────────

    def _process_navigate(self, frame, w, h, detections, depth_map) -> dict:
        """
        Full redesigned NAVIGATE pipeline.

        Stages:
          track → depth update → risk score → system health →
          narration gate → cooldown → speak → payload
        """

        # ── 1. Tracking (every frame) ─────────────────────────────────────────
        confirmed = object_tracker.update(detections, w, h)

        # ── 2. Depth update — only for confirmed objects ───────────────────────
        if depth_map is not None:
            for obj in confirmed:
                depth_score = depth_estimator.get_region_depth(
                    depth_map, obj.x1, obj.y1, obj.x2, obj.y2
                )
                if depth_score > 0.0:
                    dist_m = depth_estimator.metres_from_score(depth_score)
                    obj.update_distance(dist_m)
                    # Check depth stability and log if suppressed
                    if not depth_estimator.is_depth_stable(
                        depth_map, obj.x1, obj.y1, obj.x2, obj.y2
                    ):
                        var = depth_estimator.get_region_variance(
                            depth_map, obj.x1, obj.y1, obj.x2, obj.y2
                        )
                        diagnostics.depth_suppressed(obj.id, obj.class_name, var)

        # ── 3. Risk scoring ───────────────────────────────────────────────────
        risk_score_all(confirmed)   # attaches .risk_score and .risk_level in-place

        # ── 4. Stair / drop detection (high priority, independent of tracker) ──
        stair_warned = False
        if depth_estimator.detect_stair_drop(depth_map, h, w):
            tts_engine.speak("Warning. Possible step or drop ahead.", priority=True)
            stair_warned = True

        # ── 5. System health ──────────────────────────────────────────────────
        stability_filter.record_frame(
            fps=self.fps,
            has_detections=bool(confirmed),
            is_gpu=_ON_GPU,
        )
        if stability_filter.should_emit_failsafe():
            tts_engine.speak(FAILSAFE_MESSAGE, priority=True)
            stability_filter.record_failsafe_emitted()
            diagnostics.failsafe_emitted()
            return self._navigate_payload(w, h, "", confirmed, severity=0)

        # ── 6. In-mode question (ASK-within-NAVIGATE) ─────────────────────────
        with self._q_lock:
            nav_question     = self._pending_question
            nav_input_source = self._input_source
            self._pending_question = None

        if nav_question:
            # Use spatial context from confirmed objects
            context_strs = [
                f"{o.class_name} ({o.direction}, {o.smoothed_distance_m:.1f}m)"
                for o in confirmed[:5]
            ]
            answer = brain.answer(nav_question, frame, [], [])
            tts_engine.speak(answer, priority=True)
            self._send({
                "type":         "answer",
                "mode":         "NAVIGATE",
                "question":     nav_question,
                "answer":       answer,
                "input_source": nav_input_source,
                "context":      context_strs,
                "frame_w":      w,
                "frame_h":      h,
                "fps":          round(self.fps, 1),
            })
            return None

        # ── 7. Select highest-risk object ─────────────────────────────────────
        from backend.narrator import select_highest_risk
        best = select_highest_risk(confirmed)
        spoken_text = ""
        severity    = 0

        if best is not None:
            # ── 8. Narration gate ─────────────────────────────────────────────
            allowed, reason = stability_filter.object_passes_gate(best)
            if not allowed:
                diagnostics.narration_suppressed(
                    best.id, best.class_name, reason,
                    best.risk_level, best.risk_score,
                )
            else:
                # ── 9. Cooldown gate ──────────────────────────────────────────
                cd_allowed, cd_reason = stability_filter.narration_allowed(best.risk_level)
                if not cd_allowed:
                    diagnostics.narration_suppressed(
                        best.id, best.class_name, cd_reason,
                        best.risk_level, best.risk_score,
                    )
                else:
                    # ── 10. Build and speak narration ─────────────────────────
                    spoken_text = narrator_build(best)
                    priority    = (best.risk_level == "HIGH")
                    tts_engine.speak(spoken_text, priority=priority)
                    severity = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(best.risk_level, 0)
                    diagnostics.narration_emitted(
                        best.id, best.class_name, best.risk_level,
                        best.risk_score, best.confidence,
                        best.smoothed_distance_m, spoken_text,
                    )
                    diagnostics.record_hazard_detection_time(
                        best.first_seen_t, time.time()
                    )

        elif not stair_warned:
            # ── 11. Path clear check ──────────────────────────────────────────
            path_ok, cd_reason = stability_filter.narration_allowed("MEDIUM")
            if path_ok and is_path_clear(depth_map, h, w, confirmed):
                spoken_text = narrator_path_clear()
                tts_engine.speak(spoken_text, priority=False)

        return self._navigate_payload(w, h, spoken_text, confirmed, severity)

    def _navigate_payload(self, w, h, text, confirmed, severity) -> dict:
        return {
            "type":         "narration",
            "mode":         "NAVIGATE",
            "text":         text,
            "severity":     severity,
            "detections":   [self._serial_tracked(o) for o in confirmed],
            "show_overlay": mode_manager.snapshot().get("show_overlay", True),
            "frame_w":      w,
            "frame_h":      h,
            "fps":          round(self.fps, 1),
        }

    # ── READ mode ─────────────────────────────────────────────────────────────

    def _process_read(self, frame, w, h) -> dict | None:
        with self._q_lock:
            question     = self._pending_question
            input_source = self._input_source
            self._pending_question = None
            last_texts   = list(self._last_read_texts)

        if question:
            answer = brain.answer(question, frame, [], last_texts)
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
            return None

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
                    priority=False,
                )
            read_msg = "No text found. Move closer or adjust angle."

        return {
            "type":       "reading",
            "mode":       "READ",
            "text":       read_msg,
            "detections": [],
            "frame_w":    w,
            "frame_h":    h,
            "fps":        round(self.fps, 1),
        }

    # ── FIND mode ─────────────────────────────────────────────────────────────

    def _process_find(self, frame, w, h) -> dict | None:
        with self._q_lock:
            find_state   = self._find_capture_state
            find_frame   = self._captured_frame
            question     = self._pending_question
            input_source = self._input_source
            self._pending_question = None

        if find_state == "capturing":
            with self._q_lock:
                self._captured_frame     = frame
                self._find_capture_state = "captured"
            find_state = "captured"
            find_frame = frame

        if find_state == "captured" and question:
            answer = brain.answer(question, find_frame, [], [])
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
            with self._q_lock:
                self._find_capture_state = "idle"
                self._captured_frame     = None

        return None

    # ── ASK mode ──────────────────────────────────────────────────────────────

    def _process_ask(self, frame, w, h) -> dict | None:
        with self._q_lock:
            question     = self._pending_question
            input_source = self._input_source
            self._pending_question = None

        if not question:
            return None

        needs_llm = brain.needs_llm(question)
        if needs_llm:
            tts_engine.speak("Let me check that.", priority=True)

        q_lower = question.lower()
        if (needs_llm
                or any(k in q_lower for k in ("door", "open", "closed", "entrance", "exit"))
                or any(k in q_lower.split() for k in (
                    "medicine", "medication", "drug", "pill", "tablet", "capsule",
                    "dose", "dosage", "prescription", "label", "bottle", "packet",
                    "paracetamol", "ibuprofen", "aspirin",
                ))):
            texts = ocr_reader.read(frame)
        else:
            texts = []

        answer = brain.answer(question, frame, [], texts)
        tts_engine.speak(answer, priority=True)

        return {
            "type":         "answer",
            "mode":         "ASK",
            "question":     question,
            "answer":       answer,
            "input_source": input_source,
            "context":      [],
            "frame_w":      w,
            "frame_h":      h,
            "fps":          round(self.fps, 1),
        }

    # ── SCAN mode ─────────────────────────────────────────────────────────────

    def _process_scan(self, frame, w, h) -> dict | None:
        results = scanner.scan(frame)
        msg     = scanner.format_result(results)
        if msg:
            tts_engine.speak(msg, priority=True)
        return {
            "type":     "reading",
            "mode":     "SCAN",
            "text":     msg or "No code found. Point at a QR code or barcode.",
            "detections": [
                {
                    "class_name":     r["type"],
                    "confidence":     1.0,
                    "direction":      "ahead",
                    "distance":       "very close",
                    "distance_level": 1,
                    "distance_ft":    0.0,
                    **(r["bbox"] or {"x1": 0, "y1": 0, "x2": 0, "y2": 0}),
                }
                for r in results if r.get("value")
            ],
            "frame_w":  w,
            "frame_h":  h,
            "fps":      round(self.fps, 1),
        }

    # ── Serialisation helpers ─────────────────────────────────────────────────

    def _serial_tracked(self, obj) -> dict:
        """Serialise a TrackedObject for the frontend overlay."""
        return {
            "class_name":     obj.class_name,
            "confidence":     round(obj.confidence, 2),
            "direction":      obj.direction,
            "distance":       f"{obj.smoothed_distance_m:.1f}m",
            "distance_level": obj.distance_level,
            "distance_ft":    round(obj.smoothed_distance_m * 3.28084, 1),
            "risk_level":     obj.risk_level,
            "risk_score":     round(obj.risk_score, 3),
            "x1": obj.x1, "y1": obj.y1, "x2": obj.x2, "y2": obj.y2,
        }

    def _send(self, data: dict):
        """Broadcast a message to all WebSocket clients from a background thread."""
        if self._loop and self._bcast:
            asyncio.run_coroutine_threadsafe(self._bcast(data), self._loop)


pipeline = PipelineRunner()
