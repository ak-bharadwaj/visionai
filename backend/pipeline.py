import time, threading, asyncio, logging
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
        # FIXED: removed _pending_frame (dead code)
        # NEW FEATURE: Scene Snapshot
        self._snapshot: dict | None   = None  # stores scene at time of "remember"
        # NEW FEATURE: Find Mode
        self._find_target: str | None = None  # object label user is searching for
        self._find_announced: bool    = False  # prevent repeated "found" TTS
        # NEW FEATURE: Scene Inventory on mode start
        self._inventory_announced: bool = False  # reset each time NAVIGATE activates
        # FIX 5: path-clear cooldown — max once every 10 seconds
        self._last_clear_time: float = 0.0
        # Empty-frame reassurance counter
        self._empty_frame_count: int = 0
        # READ no-text counter
        self._no_text_count: int = 0
        # Find Mode hint counter
        self._find_hint_count: int = 0

    # ── Public API (called from WS handler, thread-safe) ──────────
    def set_question(self, question: str, input_source: str = "chat"):
        """input_source: 'voice' or 'chat' — controls whether frontend plays TTS."""
        with self._q_lock:
            self._pending_question = question
            self._input_source = input_source

    def take_snapshot(self):
        """NEW: Called when user says 'remember this'."""
        with self._q_lock:
            self._snapshot = scene_memory.get_snapshot()

    def get_scene_diff(self) -> str:
        """NEW: Compare current scene to stored snapshot."""
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

    def set_find_target(self, target: str):
        """NEW: Start searching for a named object. Resets found-state."""
        with self._q_lock:
            self._find_target    = target.lower().strip()
            self._find_announced = False
            self._find_hint_count = 0

    def clear_find_target(self):
        """NEW: Cancel active find search."""
        with self._q_lock:
            self._find_target    = None
            self._find_announced = False

    def reset_inventory(self):
        """NEW: Mark inventory as not yet announced (call when entering NAVIGATE mode)."""
        with self._q_lock:
            self._inventory_announced = False

    # ── Lifecycle ──────────────────────────────────────────────────
    def start(self, event_loop: asyncio.AbstractEventLoop, broadcast_fn: Callable):
        self._camera   = get_camera()
        self._detector = ObjectDetector()
        depth_estimator.load()
        world_detector.load()   # NEW: load YOLOWorld secondary detector
        tts_engine.start()
        self._loop  = event_loop
        self._bcast = broadcast_fn
        self._running = True
        threading.Thread(target=self._run, daemon=True, name="PipelineThread").start()
        logger.info("👁 [VisionTalk] Pipeline started.")

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

            # MiDaS — every N frames
            if frame_count % DEPTH_N == 0:
                depth_map = depth_estimator.estimate(frame)

            # YOLOWorld — every WORLD_N frames (extra class detection)
            if frame_count % WORLD_N == 0:
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

            # Stair/drop (TTS dedup handles repetition — 2s window)
            if depth_estimator.detect_stair_drop(depth_map, h, w):
                tts_engine.speak("Warning — possible step or drop ahead", priority=True)

            # Spatial analysis
            spatial_results = [
                spatial_analyzer.analyze(d, w, h, depth_map) for d in detections
            ]

            # NEW FEATURE: Find Mode — check if target object is now in frame
            with self._q_lock:
                find_target    = self._find_target
                find_announced = self._find_announced

            if find_target and not find_announced:
                match = next(
                    (r for r in spatial_results
                     if find_target in r.class_name.lower()),
                    None
                )
                if match:
                    found_msg = (
                        f"Found it! {match.class_name.title()}, "
                        f"{match.distance}, {match.direction}"
                    )
                    tts_engine.speak(found_msg, priority=True)
                    self._send({"type": "found_object", "text": found_msg,
                                "detection": self._serial(match)})
                    with self._q_lock:
                        self._find_announced = True
                else:
                    with self._q_lock:
                        self._find_hint_count += 1
                        hint_count = self._find_hint_count
                    SEARCH_HINTS = [
                        f"Still looking for {find_target}. Try turning slowly.",
                        f"No {find_target} visible. Try looking left.",
                        f"No {find_target} visible. Try looking right.",
                        f"No {find_target} yet. Try looking up or ahead.",
                    ]
                    if hint_count % 30 == 15:
                        hint = SEARCH_HINTS[(hint_count // 30) % len(SEARCH_HINTS)]
                        tts_engine.speak(hint, priority=False)

            # ═══════════════════════════════════════════
            # MODE: READ
            # ═══════════════════════════════════════════
            if mode_manager.is_read():
                if frame_count % OCR_N == 0:
                    texts = ocr_reader.read(frame, deduplicate=True)
                    if texts:
                        self._no_text_count = 0
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

                self._send({
                    "type": "reading",
                    "mode": "READ",
                    "text": read_msg,
                    "detections": [self._serial(r) for r in spatial_results],
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
                        "fps":          round(self.fps, 1),
                    })

                    # DO NOT auto-return to NAVIGATE — stay in ASK for multi-turn.
                    # User must explicitly switch mode (voice 'navigate' or mode button).

                self._tick(t0)
                continue

            # ═══════════════════════════════════════════
            # MODE: NAVIGATE (default)
            # ═══════════════════════════════════════════
            scene_memory.update(spatial_results)

            # Empty frame reassurance — if nothing detected for 10 consecutive frames
            if not spatial_results:
                self._empty_frame_count += 1
                if self._empty_frame_count >= 10:
                    tts_engine.speak("Area looks clear. Move slowly and I'll alert you.", priority=False)
                    self._empty_frame_count = 0
            else:
                self._empty_frame_count = 0

            # NEW FEATURE 4: One-time scene inventory when NAVIGATE mode starts
            with self._q_lock:
                inv_announced = self._inventory_announced
            if not inv_announced and spatial_results:
                from collections import Counter
                counts = Counter(r.class_name for r in spatial_results)
                parts  = [f"{v} {k}" if v > 1 else k for k, v in counts.most_common(5)]
                inv_msg = "Scene loaded: " + ", ".join(parts) + " detected."
                tts_engine.speak(inv_msg)
                with self._q_lock:
                    self._inventory_announced = True

            # NEW FEATURE: Check approaching objects FIRST (highest priority)
            approaching = scene_memory.get_approaching()
            if approaching:
                obj = approaching[0]   # most dangerous approaching object
                approach_msg = narrator.narrate_approaching(obj)
                tts_engine.speak(approach_msg, priority=True)
                scene_memory.mark_approach_warned(obj.key)
                nav_msg = approach_msg

            # NEW FEATURE 2: Person count / social awareness
            person_msg = narrator.narrate_persons(spatial_results)
            if person_msg and person_msg != nav_msg:
                tts_engine.speak(person_msg, priority=False)
                nav_msg = person_msg

            # Standard new-object alerts
            new_alerts  = scene_memory.get_new_alerts()
            prioritized = narrator.prioritize(new_alerts)

            if prioritized:
                best    = prioritized[0]
                nav_msg = narrator.narrate(best)
                if nav_msg:
                    tts_engine.speak(nav_msg, priority=(best.distance_level == 1))
                for a in prioritized:
                    scene_memory.mark_announced(a.key)
            else:
                # Say "path clear" at most once every 10 seconds (avoid spam)
                path_msg = narrator.path_clear(spatial_results)
                now_t = time.time()
                if (path_msg and path_msg != nav_msg
                        and now_t - self._last_clear_time > 10.0):
                    tts_engine.speak(path_msg)
                    nav_msg = path_msg
                    self._last_clear_time = now_t

            self._send({
                "type":         "narration",
                "mode":         "NAVIGATE",
                "text":         nav_msg,
                "severity":     prioritized[0].distance_level if prioritized else 0,
                "detections":   [self._serial(r) for r in spatial_results],
                "show_overlay": mode_manager.snapshot()["show_overlay"],
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
        self.fps = 1.0 / elapsed if elapsed > 0 else FPS_CAP
        time.sleep(max(0, (1 / FPS_CAP) - elapsed))


pipeline = PipelineRunner()
