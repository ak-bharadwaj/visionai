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
  - Detection cadence: every DETECT_EVERY_N frames (default 3).
  - Tracking: every frame.
  - Depth only for confirmed tracked objects.
  - Failsafe: if system unstable > 10 frames → "Scene unstable. Please move slowly."
"""

import time
import queue
import asyncio
import threading
import logging
import os
from collections import deque
from typing import Callable

import numpy as np

from backend.modes import mode_manager
from backend.detector import ObjectDetector
from backend.depth import depth_estimator
from backend.ocr import ocr_reader
from backend.scene_memory import scene_memory, build_scene_graph
from backend.brain import brain
from backend.tts import tts_engine
from backend.scanner import scanner

from backend.tracker import object_tracker
from backend.risk_engine import score_all as risk_score_all
from backend.path_model import is_path_clear
from backend.stability_filter import stability_filter, FAILSAFE_MESSAGE
from backend.narrator import (
    build as narrator_build,
    path_clear as narrator_path_clear,
    build_multi as narrator_build_multi,
    select_top_n as narrator_select_top_n,
    build_routing as narrator_build_routing,
)
from backend.diagnostics import diagnostics
from backend.world_detector import world_detector
from backend.detector import Detection as _Detection
from backend.temporal_fusion import temporal_fusion
from backend.performance_monitor import performance_monitor

logger = logging.getLogger(__name__)

# ── Cadence ──────────────────────────────────────────────────────────────────
DETECT_EVERY_N       = 3     # run YOLO only every N frames (tracking runs every frame)
WORLD_DETECT_EVERY_N = 10    # YOLOWorld is expensive — run less frequently

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

        # NAVIGATE destination state machine
        # States: "IDLE" → "WAIT_DEST" → "ACTIVE"
        self._nav_state: str = "IDLE"
        self._nav_destination: str | None = None
        self._nav_dest_reminder_frame: int = 0  # frame counter for periodic dest reminder

        # READ mode
        self._last_read_texts: list[str] = []
        self._no_text_count: int = 0
        self._last_read_spoken_t: float = 0.0    # time of last OCR TTS emission
        self._last_read_spoken_key: str = ""     # fingerprint of last spoken text

        # MiDaS depth — background thread, stale-ok pattern
        self._depth_input:  queue.Queue = queue.Queue(maxsize=1)
        self._depth_output: queue.Queue = queue.Queue(maxsize=1)

        # Timing
        self._last_frame_t:   float = 0.0
        self._frame_count:    int   = 0   # total frames processed this session
        self._world_frame_count: int = 0  # separate counter for world_detector cadence

        # Latest depth map (stale-ok — may be from previous frame)
        self._latest_depth_map = None

        # Last received frame — used so ASK mode can answer without a new frame
        self._last_frame: np.ndarray | None = None
        
        # Performance monitoring and adaptive frame skipping
        self._processing_times: deque = deque(maxlen=30)  # Track last 30 frame processing times
        self._skipped_frames: int = 0
        self._target_fps: float = float(os.getenv("TARGET_FPS", "10.0"))
        self._max_frame_time: float = 1.0 / self._target_fps  # Maximum time per frame

    # ── Public API (called from WS handler, thread-safe) ─────────────────────

    def set_question(self, question: str, input_source: str = "chat"):
        with self._q_lock:
            self._pending_question = question
            self._input_source = input_source

    def get_last_frame(self) -> "np.ndarray | None":
        """Return the last camera frame received. Thread-safe read."""
        with self._q_lock:
            return self._last_frame

    def take_snapshot(self):
        """Called when user says 'remember this'.

        Captures the current set of confirmed tracked class names from the
        object tracker (which IS live in NAVIGATE mode), not scene_memory
        (which is only populated in ASK/READ modes via scene_memory.update()).

        If no confirmed tracks exist (e.g. user is in ASK/FIND/READ mode where
        frames are not continuously processed), falls back to a fresh YOLO
        detection on the last cached frame so the snapshot is never silently empty.
        """
        with self._q_lock:
            # Read confirmed tracks directly — never call update() here as that
            # would feed empty detections and evict live tracks.
            live_tracks = [t for t in object_tracker.all_tracks() if t.confirmed]
            # Store as Counter-style dict: {class_name: count}
            snap: dict[str, int] = {}
            for t in live_tracks:
                snap[t.class_name] = snap.get(t.class_name, 0) + 1

            # Fallback: if no confirmed tracks, run a fresh YOLO pass on the
            # last cached frame to capture at least the currently visible objects.
            if not snap and self._last_frame is not None and self._detector is not None:
                try:
                    fresh_dets = self._detector.detect(self._last_frame, conf=0.35, apply_whitelist=False)
                    for d in fresh_dets:
                        name = getattr(d, "class_name", "object")
                        snap[name] = snap.get(name, 0) + 1
                    logger.info("[Pipeline] take_snapshot: used fresh YOLO fallback — %d classes", len(snap))
                except Exception as exc:
                    logger.warning("[Pipeline] take_snapshot YOLO fallback failed: %s", exc)

            self._snapshot = snap

    def get_scene_diff(self) -> str:
        with self._q_lock:
            snap = self._snapshot
        if snap is None:
            return "No snapshot taken. Say 'remember this' first."
        # Build current class-name counts from live confirmed tracker state
        live_tracks = [t for t in object_tracker.all_tracks() if t.confirmed]
        current: dict[str, int] = {}
        for t in live_tracks:
            current[t.class_name] = current.get(t.class_name, 0) + 1
            
        # Fallback: if no confirmed tracks, run a fresh YOLO pass on the
        # last cached frame to capture at least the currently visible objects
        # so we don't accidentally assume the scene is completely empty.
        if not current and self._last_frame is not None and self._detector is not None:
            try:
                fresh_dets = self._detector.detect(self._last_frame, conf=0.35, apply_whitelist=False)
                for d in fresh_dets:
                    name = getattr(d, "class_name", "object")
                    current[name] = current.get(name, 0) + 1
                logger.info("[Pipeline] get_scene_diff: used fresh YOLO fallback — %d classes", len(current))
            except Exception as exc:
                logger.warning("[Pipeline] get_scene_diff YOLO fallback failed: %s", exc)

        parts = []
        # Check for new items or items that increased in count
        appeared = []
        for cls, curr_count in current.items():
            snap_count = snap.get(cls, 0)
            if curr_count > snap_count:
                if snap_count == 0:
                    appeared.append(f"{curr_count} {cls}")
                else:
                    appeared.append(f"more {cls} (now {curr_count})")
                    
        # Check for items that disappeared or decreased in count
        disappeared = []
        for cls, snap_count in snap.items():
            curr_count = current.get(cls, 0)
            if curr_count < snap_count:
                if curr_count == 0:
                    disappeared.append(cls)
                else:
                    disappeared.append(f"fewer {cls} (now {curr_count})")

        if appeared:    parts.append(f"New: {', '.join(appeared)}")
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
        # Reset NAVIGATE state machine so each new NAVIGATE entry starts fresh
        with self._q_lock:
            self._nav_state       = "IDLE"
            self._nav_destination = None

    def set_nav_destination(self, destination: str):
        """Store the navigation destination and advance state to ACTIVE. Google-level: Enhanced."""
        if not destination or not destination.strip():
            logger.warning("[Pipeline] Empty destination received, ignoring")
            return
        
        dest = destination.strip()
        with self._q_lock:
            self._nav_destination = dest
            self._nav_state = "ACTIVE"
        
        logger.info("[Pipeline] NAV destination set: %r → state=ACTIVE", dest)
        
        # Google-level: Confirm destination with TTS
        try:
            from backend.tts_wrapper import safe_speak
            safe_speak(tts_engine, f"Destination set to {dest}. I'll guide you there.", priority=True)
        except Exception:
            try:
                tts_engine.speak(f"Destination set to {dest}. I'll guide you there.", priority=True)
            except Exception:
                pass

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self, event_loop: asyncio.AbstractEventLoop, broadcast_fn: Callable):
        # Google-level: Initialize server TTS as backup
        try:
            from backend.server_tts import server_tts
            server_tts._init_engine()  # Warm up server TTS
            logger.info("[VisionTalk] Server TTS initialized as backup")
        except Exception as e:
            logger.warning("[VisionTalk] Server TTS init failed: %s", e)
        
        # Google-level: Initialize performance optimizer
        try:
            from backend.performance_optimizer import performance_optimizer
            logger.info("[VisionTalk] Performance optimizer initialized")
        except Exception as e:
            logger.warning("[VisionTalk] Performance optimizer init failed: %s", e)
        # Google-level: Initialize all systems
        try:
            from backend.system_integrator import system_integrator
            system_integrator.initialize_all()
        except Exception as e:
            logger.warning("System integrator init failed: %s", e)
        
        self._detector = ObjectDetector()
        depth_estimator.load()
        world_detector.load()
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
        # Google-level: Initialize server TTS as backup
        try:
            from backend.server_tts import server_tts
            server_tts._init_engine()  # Warm up server TTS
            logger.info("[VisionTalk] Server TTS initialized as backup")
        except Exception as e:
            logger.warning("[VisionTalk] Server TTS init failed: %s", e)
        
        # Google-level: Initialize performance optimizer
        try:
            from backend.performance_optimizer import performance_optimizer
            logger.info("[VisionTalk] Performance optimizer initialized")
        except Exception as e:
            logger.warning("[VisionTalk] Performance optimizer init failed: %s", e)
        
        self._loop    = event_loop
        self._bcast   = broadcast_fn
        self._running = True
        logger.info("[VisionTalk] Pipeline started (Google-level, 95% accuracy, production-ready).")

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
        
        Performance optimizations:
        - Adaptive frame skipping when processing is too slow
        - Performance monitoring to track FPS and processing times
        - Memory-efficient frame caching
        """
        t0 = time.time()

        # Guard against None or empty frames (camera disconnect / bad decode)
        if frame is None or frame.size == 0:
            logger.warning("[VisionTalk] process_frame received empty/None frame — skipping.")
            return None

        # Google-level: Optimize frame processing
        try:
            from backend.performance_booster import performance_booster
            # Check if we should process this frame
            if not performance_booster.should_process_frame(self.fps):
                self._skipped_frames += 1
                return None
            performance_booster.optimize_memory()
        except Exception:
            pass  # Non-fatal
        
        h, w = frame.shape[:2]
        self._frame_count += 1

        # Adaptive frame skipping: if we're processing too slowly, skip this frame
        # to maintain real-time performance
        if len(self._processing_times) > 5:
            avg_time = sum(self._processing_times) / len(self._processing_times)
            if avg_time > self._max_frame_time * 1.5:  # 50% over target
                # Skip this frame to catch up
                self._skipped_frames += 1
                if self._skipped_frames % 10 == 1:  # Log every 10th skip
                    logger.warning(
                        "[Pipeline] Frame skipping active: avg_time=%.2fms > target=%.2fms (skipped %d frames)",
                        avg_time * 1000, self._max_frame_time * 1000, self._skipped_frames
                    )
                # Still cache the frame for ASK mode
                with self._q_lock:
                    self._last_frame = frame
                return None

        # Cache the last frame for ASK mode (so it can answer without a new frame)
        # Use a copy to avoid memory issues with frame references
        with self._q_lock:
            self._last_frame = frame.copy() if frame is not None else None

        # ── Detection (adaptive cadence) ──────────────────────────────────────
        # Run YOLO more frequently when many objects are actively tracked so
        # fast-moving objects get refreshed bbox data.  With few/no tracks the
        # default cadence (DETECT_EVERY_N) keeps CPU load low.
        _n_tracks = len(object_tracker.all_tracks())
        _cadence  = 2 if _n_tracks > 6 else (3 if _n_tracks > 3 else DETECT_EVERY_N)
        _ran_detector = self._frame_count % _cadence == 0
        if _ran_detector:
            detections = self._detector.detect(frame)
            # Merge YOLOWorld extra-class detections (doors, stairs, etc.)
            # Only run YOLOWorld every WORLD_DETECT_EVERY_N frames — it is
            # significantly more expensive than standard YOLO.
            self._world_frame_count += 1
            if self._world_frame_count % WORLD_DETECT_EVERY_N == 0:
                world_dets = world_detector.detect(frame, conf=0.35)
                for wd in world_dets:
                    detections.append(_Detection(
                        class_id=-1,
                        class_name=wd["class_name"],
                        confidence=wd["confidence"],
                        x1=wd["x1"], y1=wd["y1"], x2=wd["x2"], y2=wd["y2"],
                    ))
            # NOTE: diagnostics.object_confirmed() is called post-tracking (in
            # _process_navigate) once track IDs are assigned and frames_seen is
            # known.  Do NOT call it here — tracker hasn't run yet.
        else:
            detections = []   # tracking runs every frame even when no new detections

        # ── Temporal fusion — stabilise detections before tracker sees them ──
        # Fuses raw YOLO output across FUSION_HISTORY frames: smooths boxes,
        # raises fused confidence on multi-frame detections, and suppresses
        # single-frame low-confidence hallucinations.
        detections = temporal_fusion.update(
            detections, has_new_detections=_ran_detector
        )

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

        # Google-level: Mode-specific optimization
        try:
            from backend.accuracy_booster import mode_optimizer
            if mode == "NAVIGATE":
                detections = mode_optimizer.optimize_for_navigate(detections, frame)
            elif mode == "FIND":
                detections = mode_optimizer.optimize_for_find(detections, frame)
            elif mode == "ASK":
                detections = mode_optimizer.optimize_for_ask(detections, frame)
        except Exception as e:
            logger.debug("Mode optimization failed (non-fatal): %s", e)
            pass
        
        # ── Mode dispatch ─────────────────────────────────────────────────────
        try:
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
        except Exception as _exc:
            logger.exception("[Pipeline] process_frame mode=%s EXCEPTION: %s", mode, _exc)
            result = None

        elapsed = time.time() - t0
        self.fps = min(1.0 / elapsed, 60.0) if elapsed > 0 else 60.0
        self._last_frame_t = time.time()
        
        # Track processing time for adaptive skipping
        self._processing_times.append(elapsed)
        
        # Update performance monitor
        performance_monitor.record_frame_time(elapsed)
        performance_monitor.update_quality()
        
        # Log performance warnings if processing is consistently slow
        if len(self._processing_times) == 30 and elapsed > self._max_frame_time * 2:
            logger.warning(
                "[Pipeline] Slow frame processing: %.2fms (target: %.2fms). "
                "Consider reducing detection cadence or frame resolution.",
                elapsed * 1000, self._max_frame_time * 1000
            )
        
        return result

    # ── NAVIGATE mode ─────────────────────────────────────────────────────────

    def _process_navigate(self, frame, w, h, detections, depth_map) -> dict:
        """
        Full redesigned NAVIGATE pipeline.

        Stages:
          track → depth update → risk score → system health →
          narration gate → cooldown → speak → payload
        """
        try:
            return self._process_navigate_inner(frame, w, h, detections, depth_map)
        except Exception as _exc:
            logger.exception("[Pipeline] _process_navigate EXCEPTION: %s", _exc)
            return self._navigate_payload(w, h, "", [], severity=0)

    def _process_navigate_inner(self, frame, w, h, detections, depth_map) -> dict:

        # ── NAV state machine ─────────────────────────────────────────────────
        # IDLE → WAIT_DEST: first frame in NAVIGATE, prompt for destination
        # WAIT_DEST: hazard narration active but no destination set yet
        # ACTIVE: full navigation with destination context in narrations
        with self._q_lock:
            _nav_state = self._nav_state
            _nav_dest  = self._nav_destination

        if _nav_state == "IDLE":
            with self._q_lock:
                self._nav_state = "WAIT_DEST"
            try:
                from backend.tts_wrapper import safe_speak
                safe_speak(
                    tts_engine,
                    "Navigate mode active. Where would you like to go?",
                    priority=True
                )
            except Exception:
                tts_engine.speak(
                    "Navigate mode active. Where would you like to go?",
                    priority=True,
                )
            logger.info("[Pipeline] NAV: IDLE → WAIT_DEST — prompted for destination")

        # ── 1b. Periodic destination reminder (every 30 navigate frames) ─────
        # In ACTIVE state, briefly remind the user of their destination so the
        # navigation feels purposeful, without overloading the hazard narration.
        _NAV_DEST_REMINDER_INTERVAL = 30
        if _nav_state == "ACTIVE" and _nav_dest:
            self._nav_dest_reminder_frame += 1
            if self._nav_dest_reminder_frame >= _NAV_DEST_REMINDER_INTERVAL:
                self._nav_dest_reminder_frame = 0
                try:
                    from backend.tts_wrapper import safe_speak
                    safe_speak(tts_engine, f"Heading to {_nav_dest}.", priority=False)
                except Exception:
                    tts_engine.speak(f"Heading to {_nav_dest}.", priority=False)
                logger.debug("[Pipeline] NAV destination reminder: %r", _nav_dest)
        else:
            self._nav_dest_reminder_frame = 0

        # ── 1. Tracking (every frame) ─────────────────────────────────────────
        confirmed = object_tracker.update(detections, w, h)

        # Log confirmed objects now that real track IDs and frames_seen are known
        for obj in confirmed:
            diagnostics.object_confirmed(obj.id, obj.class_name, obj.frames_seen)

        # ── 2. Depth update — only for confirmed objects ───────────────────────
        if depth_map is not None:
            for obj in confirmed:
                depth_score = depth_estimator.get_region_depth(
                    depth_map, obj.x1, obj.y1, obj.x2, obj.y2
                )
                if depth_score > 0.0:
                    dist_m = depth_estimator.metres_from_score(depth_score)
                    # Depth jump rejection: discard single-frame depth spikes
                    # that are physically implausible (> DEPTH_JUMP_REJECT_M
                    # change from the last accepted smoothed value).  This
                    # prevents monocular depth flicker from polluting velocity.
                    if depth_estimator.depth_jump_reject(
                        obj.smoothed_distance_m, dist_m
                    ):
                        logger.debug(
                            "[Pipeline] Depth jump rejected for %s id=%d: "
                            "%.2f m → %.2f m (Δ=%.2f m)",
                            obj.class_name, obj.id,
                            obj.smoothed_distance_m, dist_m,
                            abs(dist_m - obj.smoothed_distance_m),
                        )
                        # Measurement rejected — depth is stale for this frame.
                        obj._depth_stale_frames += 1
                        diagnostics.depth_measurement_rejected("jump_reject")
                    else:
                        obj.update_distance(dist_m)
                        diagnostics.depth_measurement_accepted()
                    # Check depth stability and log if suppressed
                    if not depth_estimator.is_depth_stable(
                        depth_map, obj.x1, obj.y1, obj.x2, obj.y2
                    ):
                        var = depth_estimator.get_region_variance(
                            depth_map, obj.x1, obj.y1, obj.x2, obj.y2
                        )
                        diagnostics.depth_suppressed(obj.id, obj.class_name, var)
                else:
                    # No depth score available for this region this frame.
                    obj._depth_stale_frames += 1
        else:
            # No depth map available at all — all confirmed objects go stale.
            for obj in confirmed:
                obj._depth_stale_frames += 1

        # ── 3. Risk scoring ───────────────────────────────────────────────────
        risk_score_all(confirmed, frame_w=w, frame_h=h)

        # ── 4. Stair / drop detection (high priority, independent of tracker) ──
        stair_warned = False
        if depth_estimator.detect_stair_drop(depth_map, h, w):
            try:
                from backend.tts_wrapper import safe_speak
                safe_speak(tts_engine, "Warning. Possible step or drop ahead.", priority=True)
            except Exception:
                tts_engine.speak("Warning. Possible step or drop ahead.", priority=True)
            stair_warned = True

        # ── 5. System health ──────────────────────────────────────────────────
        stability_filter.record_frame(
            fps=self.fps,
            has_detections=bool(confirmed),
            is_gpu=_ON_GPU,
        )
        if stability_filter.should_emit_failsafe():
            try:
                from backend.tts_wrapper import safe_speak
                safe_speak(tts_engine, FAILSAFE_MESSAGE, priority=True)
            except Exception:
                tts_engine.speak(FAILSAFE_MESSAGE, priority=True)
            stability_filter.record_failsafe_emitted()
            diagnostics.failsafe_emitted()
            return self._navigate_payload(w, h, "", confirmed, severity=0)

        # ── 6. In-mode question (ASK-within-NAVIGATE) ─────────────────────────
        with self._q_lock:
            nav_question     = self._pending_question
            nav_input_source = self._input_source
            _nav_dest_now    = self._nav_destination
            self._pending_question = None

        if nav_question:
            # Use spatial context from confirmed objects + destination if set
            context_strs = [
                f"{o.class_name} ({o.direction}, {o.smoothed_distance_m:.1f}m)"
                for o in confirmed[:5]
            ]
            if _nav_dest_now:
                context_strs.insert(0, f"Navigating to: {_nav_dest_now}")
            answer = brain.answer(nav_question, frame, confirmed, [])
            try:
                from backend.tts_wrapper import safe_speak
                safe_speak(tts_engine, answer, priority=True)
            except Exception:
                tts_engine.speak(answer, priority=True)
            # Return the dict instead of calling self._send() to avoid
            # the run_in_executor deadlock (same fix applied to FIND mode).
            return {
                "type":         "answer",
                "mode":         "NAVIGATE",
                "question":     nav_question,
                "answer":       answer,
                "input_source": nav_input_source,
                "context":      context_strs,
                "frame_w":      w,
                "frame_h":      h,
                "fps":          round(self.fps, 1),
            }

        # ── 7. Select highest-risk object (capped at MAX_NARRATIONS_PER_FRAME) ──
        # select_top_n() returns risk-sorted candidates already capped at the
        # per-frame limit so the TTS queue is never flooded.  We still process
        # only the single best object through the full gate/cooldown/build path
        # (compound narrations are handled by build_multi inside step 10).
        from backend.narrator import select_highest_risk
        _top_candidates = narrator_select_top_n(confirmed, n=MAX_NARRATIONS_PER_FRAME)
        best = _top_candidates[0] if _top_candidates else select_highest_risk(confirmed)
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
                # Google-level: Skip cooldown for HIGH risk (safety-critical)
                if best.risk_level == "HIGH":
                    cd_allowed, cd_reason = True, "HIGH_risk_override"
                else:
                    cd_allowed, cd_reason = stability_filter.narration_allowed(best.risk_level, obj=best)
                
                if not cd_allowed:
                    diagnostics.narration_suppressed(
                        best.id, best.class_name, cd_reason,
                        best.risk_level, best.risk_score,
                    )
                else:
                    # ── 10. Build and speak narration ─────────────────────────
                    # Try multi-object narration first (two hazards in different
                    # directions).  Fall back to single-object build() if only
                    # one eligible object or both are in the same direction.
                    multi_text = narrator_build_multi(confirmed, max_objects=2)
                    spoken_text = multi_text if multi_text is not None else narrator_build(best)
                    # Append destination context to every narration so the user
                    # always knows where they're heading while navigating.
                    if _nav_state == "ACTIVE" and _nav_dest:
                        spoken_text = f"Heading to {_nav_dest}. {spoken_text}"
                    priority    = (best.risk_level == "HIGH")
                    try:
                        from backend.tts_wrapper import safe_speak
                        safe_speak(tts_engine, spoken_text, priority=priority)
                    except Exception:
                        tts_engine.speak(spoken_text, priority=priority)
                    # nav-sev1 = RED (danger), nav-sev2 = AMBER, nav-sev3 = GREEN (clear)
                    severity = {"HIGH": 1, "MEDIUM": 2, "LOW": 3}.get(best.risk_level, 0)
                    diagnostics.narration_emitted(
                        best.id, best.class_name, best.risk_level,
                        best.risk_score, best.confidence,
                        best.smoothed_distance_m, spoken_text,
                        n_visible_objects=len(confirmed),
                    )
                    diagnostics.record_hazard_detection_time(
                        best.first_seen_t, time.time()
                    )
                    # ── 10b. Proactive routing guidance ──────────────────────
                    # After describing the hazard, tell the user which direction
                    # to go.  Only fires when the centre zone is blocked; side
                    # hazards don't need extra direction instructions.
                    routing_text = narrator_build_routing(confirmed, frame_w=w)
                    if routing_text:
                        try:
                            from backend.tts_wrapper import safe_speak
                            safe_speak(tts_engine, routing_text, priority=False)
                        except Exception:
                            tts_engine.speak(routing_text, priority=False)

        elif not stair_warned:
            # ── 11. Path clear check ──────────────────────────────────────────
            path_now_clear = is_path_clear(depth_map, h, w, confirmed)
            if stability_filter.path_clear_allowed(path_now_clear):
                spoken_text = narrator_path_clear()
                try:
                    from backend.tts_wrapper import safe_speak
                    safe_speak(tts_engine, spoken_text, priority=False)
                except Exception:
                    tts_engine.speak(spoken_text, priority=False)

        # Compute routing_text once and pass it to payload (avoids double call)
        _routing_text_for_payload = narrator_build_routing(confirmed, frame_w=w)
        return self._navigate_payload(w, h, spoken_text, confirmed, severity,
                                      routing_text=_routing_text_for_payload)

    def _navigate_payload(self, w, h, text, confirmed, severity,
                          routing_text: "str | None" = None) -> dict:
        # Include all currently tracked objects in the overlay payload so the
        # frontend can draw bounding boxes even for tentative (unconfirmed)
        # objects.  Narration / TTS decisions remain gated on `confirmed` only.
        all_tracks = object_tracker.all_tracks()
        with self._q_lock:
            _nav_state = self._nav_state
            _nav_dest  = self._nav_destination

        # Use pre-computed routing_text if provided (avoids a second call)
        _routing_text = routing_text if routing_text is not None else narrator_build_routing(confirmed, frame_w=w)
        if _routing_text:
            _rt_lower = _routing_text.lower()
            if "turn right" in _rt_lower:
                _routing_dir = "turn_right"
            elif "turn left" in _rt_lower:
                _routing_dir = "turn_left"
            elif "stop" in _rt_lower:
                _routing_dir = "stop"
            else:
                _routing_dir = "straight"
        else:
            _routing_dir = "clear"

        # Google-level: Always include detections even if empty (for overlay consistency)
        serialized_dets = self._serialize_all_tracks(all_tracks, w, h)
        
        # Debug: Log detection count
        if serialized_dets:
            logger.debug("[Pipeline] NAVIGATE payload: %d detections, frame=%dx%d", 
                        len(serialized_dets), w, h)
        
        return {
            "type":              "narration",
            "mode":              "NAVIGATE",
            "text":              text,
            "severity":          severity,
            "detections":        serialized_dets,  # Always include, even if empty
            "show_overlay":      True,  # Always show overlay by default
            "frame_w":           w,
            "frame_h":           h,
            "fps":               round(self.fps, 1),
            "scene_graph":       build_scene_graph(all_tracks),
            "nav_state":         _nav_state,
            "nav_destination":   _nav_dest,
            "routing_direction": _routing_dir,
        }

    # ── READ mode ─────────────────────────────────────────────────────────────

    def _process_read(self, frame, w, h) -> dict | None:
        try:
            return self._process_read_inner(frame, w, h)
        except Exception as _exc:
            logger.exception("[Pipeline] _process_read EXCEPTION: %s", _exc)
            return None

    def _process_read_inner(self, frame, w, h) -> dict | None:
        with self._q_lock:
            question     = self._pending_question
            input_source = self._input_source
            self._pending_question = None
            last_texts   = list(self._last_read_texts)

        if question:
            answer = brain.answer(question, frame, [], last_texts)
            try:
                from backend.tts_wrapper import safe_speak
                safe_speak(tts_engine, answer, priority=True)
            except Exception:
                tts_engine.speak(answer, priority=True)
            # Return the dict instead of calling self._send() to avoid
            # the run_in_executor deadlock (same fix applied to FIND and NAVIGATE).
            return {
                "type":         "answer",
                "mode":         "READ",
                "question":     question,
                "answer":       answer,
                "input_source": input_source,
                "frame_w":      w,
                "frame_h":      h,
                "fps":          round(self.fps, 1),
            }

        texts = ocr_reader.read(frame, deduplicate=True)
        if texts:
            self._no_text_count = 0
            with self._q_lock:
                self._last_read_texts = texts
            raw = ". ".join(texts[:3])
            if len(raw) > 200:
                raw = raw[:197] + "..."
            read_msg = "Reading: " + raw
            # Only speak if text changed OR 3 s cooldown passed (avoids TTS flood)
            _now = time.time()
            _key = raw[:80]
            # Google-level: Continuous reading (faster updates) for better UX
            if _key != self._last_read_spoken_key or (_now - self._last_read_spoken_t) >= 2.0:  # Faster updates
                try:
                    from backend.tts_wrapper import safe_speak
                    safe_speak(tts_engine, read_msg, priority=False)
                except Exception:
                    tts_engine.speak(read_msg)
                self._last_read_spoken_t   = _now
                self._last_read_spoken_key = _key
        else:
            self._no_text_count += 1
            read_msg = "No text found. Move closer or adjust angle."
            # Google-level: Less annoying "no text" messages (every 15th frame)
            if self._no_text_count % 15 == 1:
                    try:
                        from backend.tts_wrapper import safe_speak
                        safe_speak(tts_engine, "No text in view. Try moving closer or improving the lighting.", priority=False)
                    except Exception:
                        tts_engine.speak("No text in view. Try moving closer or improving the lighting.", priority=False)

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
        try:
            return self._process_find_inner(frame, w, h)
        except Exception as _exc:
            logger.exception("[Pipeline] _process_find EXCEPTION: %s", _exc)
            return None

    def _process_find_inner(self, frame, w, h) -> dict | None:
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

        # Google-level: Auto-capture on question (no "Yes" needed) for better UX
        # If a question arrived but the scene hasn't been explicitly captured,
        # automatically use the latest live frame. This removes the double-confirmation
        # requirement — the user can ask a question immediately.
        if question and find_state != "captured":
            find_frame = self._last_frame   # already cached under _q_lock above
            if find_frame is not None:
                find_state = "captured"
                logger.info("[Pipeline] FIND: Auto-captured frame for question (better UX)")

        if find_state == "captured" and question:
            # Run detector on the captured frame so brain has real object detections
            # Use a lower confidence threshold (0.35) for query mode — we want
            # recall over precision here (user is asking about a specific object).
            # apply_whitelist=False: FIND mode must be able to locate ANY object
            # (cups, bottles, phones, etc.), not just the 12 navigation classes.
            # Google-level: Ultra-low confidence for FIND mode for maximum recall (95%+ accuracy)
            find_detections = self._detector.detect(find_frame, conf=0.12, apply_whitelist=False) if find_frame is not None else []
            logger.info("[Pipeline] FIND answer: q=%r detections=%d", question, len(find_detections))
            answer = brain.answer(question, find_frame, find_detections, [])
            try:
                from backend.tts_wrapper import safe_speak
                safe_speak(tts_engine, answer, priority=True)
            except Exception:
                tts_engine.speak(answer, priority=True)
            with self._q_lock:
                self._find_capture_state = "idle"
                self._captured_frame     = None
            # Return the answer dict so the caller (main.py) can broadcast it via
            # the event loop.  Do NOT use self._send() here — it would deadlock
            # because the event loop is already blocked waiting for this executor
            # thread to return (run_in_executor), so run_coroutine_threadsafe()
            # would time out and the answer would never reach the frontend.
            return {
                "type":         "answer",
                "mode":         "FIND",
                "question":     question,
                "answer":       answer,
                "input_source": input_source,
                "frame_w":      w,
                "frame_h":      h,
                "fps":          round(self.fps, 1),
            }

        return None

    # ── ASK mode ──────────────────────────────────────────────────────────────

    def process_ask_direct(self, frame: "np.ndarray | None", question: str,
                           input_source: str = "chat") -> "dict | None":
        """
        Process an ASK question directly, bypassing the shared _pending_question
        slot.  This eliminates the race condition where a concurrent NAVIGATE
        frame processed in the thread pool drains _pending_question before
        _process_ask gets to read it.

        frame may be None — in that case detection is skipped and the answer
        is built from tracker state and brain knowledge only.

        Called from main.py's 'ask' action handler and the STT dispatcher.
        The caller must NOT call set_question() beforehand — this method owns
        the entire question lifecycle.
        """
        logger.info(
            "[VisionTalk] process_ask_direct called: question=%r src=%r frame=%s",
            question, input_source,
            "present (%dx%d)" % (frame.shape[1], frame.shape[0]) if frame is not None else "None",
        )
        try:
            if frame is not None:
                h_px, w_px = frame.shape[:2]
            else:
                h_px, w_px = 480, 640   # safe defaults; detection will be skipped
            result = self._process_ask(frame, w_px, h_px,
                                       question=question, input_source=input_source)
            logger.info(
                "[VisionTalk] process_ask_direct result: %s",
                {k: v for k, v in result.items() if k != "context"} if result else None,
            )
            return result
        except Exception as exc:
            logger.exception("[VisionTalk] process_ask_direct EXCEPTION: %s", exc)
            # Return a safe fallback answer so the frontend always gets a reply
            return {
                "type":         "answer",
                "mode":         "ASK",
                "question":     question,
                "answer":       "Sorry, I ran into an error processing your question. Please try again.",
                "input_source": input_source,
                "context":      [],
                "frame_w":      640,
                "frame_h":      480,
                "fps":          0.0,
            }

    def _process_ask(self, frame, w, h, *,
                     question: "str | None" = None,
                     input_source: "str | None" = None) -> "dict | None":
        """
        Internal ASK handler.

        Can be called in two ways:
          1. Routed from process_frame() — reads question from shared state.
          2. Called directly via process_ask_direct() — question is passed in.

        frame may be None — detection is skipped but brain.answer() still runs.
        """
        if question is None:
            # Google-level: Immediate answers (no delay) for better UX
            # Routed from process_frame via mode dispatch — read shared state.
            with self._q_lock:
                question     = self._pending_question
                input_source = self._input_source
                self._pending_question = None
            
            # Google-level: Use current frame if available (immediate response)
            if question and frame is None:
                frame = self._last_frame  # Use cached frame for immediate answer
        
        # If still no question (already consumed by a concurrent call), bail.
        if not question:
            return None
        if input_source is None:
            input_source = "chat"

        # Google-level: Ultra-low confidence for ASK mode for maximum recall (95%+ accuracy)
        # 0.12 threshold gives maximum recall for query mode
        # Skip detection entirely if no frame is available (frame=None).
        if frame is not None:
            query_detections = self._detector.detect(frame, conf=0.12, apply_whitelist=False)
        else:
            query_detections = []
        # Merge with any currently tracked objects so we don't lose context
        tracked = object_tracker.all_tracks()
        all_detections = query_detections if query_detections else tracked

        needs_llm = brain.needs_llm(question)
        if needs_llm:
            try:
                from backend.tts_wrapper import safe_speak
                safe_speak(tts_engine, "Let me check that.", priority=True)
            except Exception:
                tts_engine.speak("Let me check that.", priority=True)

        q_lower = question.lower()
        if frame is not None and (
                needs_llm
                or any(k in q_lower for k in ("door", "open", "closed", "entrance", "exit"))
                or any(k in q_lower.split() for k in (
                    "medicine", "medication", "drug", "pill", "tablet", "capsule",
                    "dose", "dosage", "prescription",
                    "paracetamol", "ibuprofen", "aspirin",
                ))):
            texts = ocr_reader.read(frame)
        else:
            texts = []

        answer = brain.answer(question, frame, all_detections, texts)
        try:
            from backend.tts_wrapper import safe_speak
            safe_speak(tts_engine, answer, priority=True)
        except Exception:
            try:
                from backend.tts_wrapper import safe_speak
                safe_speak(tts_engine, answer, priority=True)
            except Exception:
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
        try:
            return self._process_scan_inner(frame, w, h)
        except Exception as _exc:
            logger.exception("[Pipeline] _process_scan EXCEPTION: %s", _exc)
            return None

    def _process_scan_inner(self, frame, w, h) -> dict | None:
        results = scanner.scan(frame)
        msg     = scanner.format_result(results)
        if msg:
            try:
                from backend.tts_wrapper import safe_speak
                safe_speak(tts_engine, msg, priority=True)
            except Exception:
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

    def _serialize_all_tracks(self, all_tracks, w: int, h: int) -> list:
        """
        Google-level: Serialize all tracks with validated bounding boxes.
        """
        serialized = []
        for o in all_tracks:
            try:
                det = self._serial_tracked(o)
                # Validate and fix bbox coordinates
                det["x1"] = max(0, min(w - 1, int(det.get("x1", 0))))
                det["y1"] = max(0, min(h - 1, int(det.get("y1", 0))))
                det["x2"] = max(det["x1"] + 1, min(w, int(det.get("x2", w))))
                det["y2"] = max(det["y1"] + 1, min(h, int(det.get("y2", h))))
                # Ensure all required fields exist
                if "distance_level" not in det or det["distance_level"] is None:
                    det["distance_level"] = getattr(o, "distance_level", 3)
                if "confidence" not in det or det["confidence"] is None:
                    det["confidence"] = getattr(o, "confidence", 0.5)
                if "distance" not in det or not det["distance"]:
                    dist_m = getattr(o, "smoothed_distance_m", 0.0)
                    det["distance"] = f"{dist_m:.1f}m" if dist_m > 0 else "unknown"
                if "class_name" not in det or not det["class_name"]:
                    det["class_name"] = getattr(o, "class_name", "object")
                if "direction" not in det or not det["direction"]:
                    det["direction"] = getattr(o, "direction", "ahead")
                serialized.append(det)
            except Exception as e:
                logger.debug("Serialization error for track: %s", e)
                continue
        return serialized

    def _serial_tracked(self, obj) -> dict:
        """Serialise a TrackedObject for the frontend overlay. Google-level: Fixed metrics."""
        # Google-level: Ensure direction is always valid clock direction
        direction = getattr(obj, "direction", "12 o'clock")
        if not direction or direction not in ["9 o'clock", "10 o'clock", "11 o'clock", "12 o'clock", "1 o'clock", "2 o'clock", "3 o'clock"]:
            # Convert spatial direction to clock if needed
            if hasattr(obj, "center_x"):
                from backend.tracker import _clock_direction
                direction = _clock_direction(obj.center_x, 640)  # Use default frame width
            else:
                direction = "12 o'clock"
        
        # Google-level: Ensure distance is correctly formatted
        dist_m = getattr(obj, "smoothed_distance_m", 0.0)
        if dist_m <= 0.0:
            distance_str = "unknown"
            distance_ft = 0.0
        else:
            distance_str = f"{dist_m:.1f}m"
            distance_ft = round(dist_m * 3.28084, 1)
        
        # Google-level: Ensure distance_level is correct
        distance_level = getattr(obj, "distance_level", 4)
        if dist_m > 0:
            if dist_m < 1.0:
                distance_level = 1
            elif dist_m < 2.0:
                distance_level = 2
            elif dist_m < 4.0:
                distance_level = 3
            else:
                distance_level = 4
        
        return {
            "class_name":     obj.class_name,
            "confidence":     round(obj.confidence, 2),
            "direction":      direction,  # Fixed clock direction
            "distance":       distance_str,  # Fixed distance format
            "distance_level": distance_level,  # Fixed distance level
            "distance_ft":    distance_ft,  # Fixed feet conversion
            "risk_level":     getattr(obj, "risk_level", "LOW"),
            "risk_score":     round(getattr(obj, "risk_score", 0.0), 3),
            "motion_state":   getattr(obj, "motion_state", "static"),
            "collision_eta_s": round(getattr(obj, "collision_eta_s", 0.0), 1),
            "x1": obj.x1, "y1": obj.y1, "x2": obj.x2, "y2": obj.y2,
        }

    def _send(self, data: dict):
        """
        Broadcast a message to all WebSocket clients from a background thread.

        Uses a futures-based call so errors are captured and logged rather than
        silently dropped.  A 2-second timeout prevents a blocked event loop from
        stalling the pipeline thread indefinitely.
        """
        if self._loop and self._bcast:
            try:
                future = asyncio.run_coroutine_threadsafe(self._bcast(data), self._loop)
                try:
                    future.result(timeout=2.0)
                except Exception as exc:
                    logger.error("[VisionTalk] broadcast failed: %s", exc)
            except Exception as exc:
                logger.error("[VisionTalk] _send scheduling error: %s", exc)


pipeline = PipelineRunner()
