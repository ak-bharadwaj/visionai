"""
stability_filter.py — System health monitor and narration gate for VisionTalk.

Responsibilities:
  1. Track consecutive frames of detection instability.
     If instability exceeds UNSTABLE_FRAME_LIMIT → emit failsafe warning.
  2. Narration gate: enforce that an object passes ALL criteria before
     its text reaches TTS.
  3. Track FPS stability and flag if below safe thresholds.

Instability criteria (any ONE triggers an unstable frame count increment):
  - No detections for EMPTY_FRAME_LIMIT consecutive frames while in motion.
  - Depth estimation not loaded.
  - FPS drops below FPS_MIN_GPU (GPU mode) or FPS_MIN_CPU (CPU mode).

Narration gate (ALL must pass):
  - obj.confirmed == True (frames_seen >= MIN_FRAMES_CONFIRM = 3)
  - obj.confidence >= CONF_GATE (0.60)
  - obj.smoothed_distance_m > 0.0  (depth available)
  - obj.distance_variance() <= DIST_VARIANCE_GATE  (depth stable)
  - obj.risk_level in ("HIGH", "MEDIUM")  (never narrate LOW)

Design constraints (safety-critical):
  - Thread-safe (single lock around mutable state).
  - Failsafe speech is idempotent within FAILSAFE_COOLDOWN seconds.
  - Never raises; returns safe defaults on error.
"""

import time
import threading
import logging

logger = logging.getLogger(__name__)

# ── Stability thresholds ─────────────────────────────────────────────────────
UNSTABLE_FRAME_LIMIT = 10       # consecutive unstable frames before failsafe
FPS_MIN_GPU          = 25.0     # minimum acceptable FPS on GPU
FPS_MIN_CPU          = 15.0     # minimum acceptable FPS on CPU
FAILSAFE_COOLDOWN    = 5.0      # seconds between repeated failsafe messages

# ── Narration gate constants ─────────────────────────────────────────────────
CONF_GATE            = 0.60
DIST_VARIANCE_GATE   = 0.06     # above this → depth is too noisy to report distance
NARRATION_COOLDOWN   = 4.0      # seconds between non-critical narrations
HIGH_RISK_COOLDOWN   = 0.0      # HIGH risk always interrupts cooldown

FAILSAFE_MESSAGE = "Scene unstable. Please move slowly."


class StabilityFilter:
    """
    Stateful monitor.  One shared instance per pipeline.

    Usage:
        sf = StabilityFilter()

        # Each frame:
        sf.record_frame(fps=current_fps, has_detections=bool, is_gpu=True)
        if sf.is_system_unstable():
            tts.speak(FAILSAFE_MESSAGE, priority=True)

        # Before narrating each candidate object:
        allowed, reason = sf.object_passes_gate(obj)
        if allowed:
            tts.speak(narrator.build(obj))
    """

    def __init__(self):
        self._lock              = threading.Lock()
        self._unstable_frames   = 0
        self._last_failsafe_t   = 0.0
        self._last_narration_t  = 0.0   # time of last non-HIGH narration
        self._last_high_t       = 0.0   # time of last HIGH narration

    def reset(self):
        """Call on mode change to clear accumulated instability."""
        with self._lock:
            self._unstable_frames  = 0
            self._last_failsafe_t  = 0.0
            self._last_narration_t = 0.0
            self._last_high_t      = 0.0

    # ── Frame-level health ───────────────────────────────────────────────────

    def record_frame(self, fps: float, has_detections: bool, is_gpu: bool = False):
        """
        Call once per processed frame to update instability counter.

        Args:
            fps            : current measured frames-per-second.
            has_detections : True if at least one detection returned this frame.
            is_gpu         : whether the pipeline is running on a GPU device.
        """
        fps_min  = FPS_MIN_GPU if is_gpu else FPS_MIN_CPU
        unstable = (not has_detections) or (fps > 0 and fps < fps_min)

        with self._lock:
            if unstable:
                self._unstable_frames += 1
            else:
                # Recovery — reset counter
                self._unstable_frames = max(0, self._unstable_frames - 1)

    def is_system_unstable(self) -> bool:
        """True if the system has been unstable for UNSTABLE_FRAME_LIMIT frames."""
        with self._lock:
            return self._unstable_frames >= UNSTABLE_FRAME_LIMIT

    def should_emit_failsafe(self) -> bool:
        """
        True if the system is unstable AND the failsafe cooldown has expired.
        Consuming code is responsible for calling record_failsafe_emitted().
        """
        if not self.is_system_unstable():
            return False
        with self._lock:
            return (time.time() - self._last_failsafe_t) >= FAILSAFE_COOLDOWN

    def record_failsafe_emitted(self):
        """Call after actually speaking the failsafe message."""
        with self._lock:
            self._last_failsafe_t = time.time()

    # ── Object-level narration gate ──────────────────────────────────────────

    def object_passes_gate(self, obj) -> tuple[bool, str]:
        """
        Check all narration preconditions for a TrackedObject.

        Returns:
            (True, "ok")                       — object may be narrated.
            (False, "<reason>")                — suppressed with logged reason.
        """
        try:
            if not obj.confirmed:
                return False, f"not_confirmed (frames_seen={obj.frames_seen})"

            if obj.confidence < CONF_GATE:
                return False, f"low_conf ({obj.confidence:.2f} < {CONF_GATE})"

            if obj.smoothed_distance_m <= 0.0:
                return False, "no_depth"

            var = obj.distance_variance()
            if var > DIST_VARIANCE_GATE:
                return False, f"depth_unstable (var={var:.4f} > {DIST_VARIANCE_GATE})"

            if obj.risk_level not in ("HIGH", "MEDIUM"):
                return False, f"low_risk ({obj.risk_level})"

            return True, "ok"

        except Exception as exc:
            logger.error("[StabilityFilter] gate error: %s", exc)
            return False, f"exception: {exc}"

    def narration_allowed(self, risk_level: str) -> tuple[bool, str]:
        """
        Check cooldown before speaking.

        HIGH risk always bypasses the 4-second cooldown.
        MEDIUM/LOW respect NARRATION_COOLDOWN.

        Returns (allowed: bool, reason: str).
        """
        now = time.time()
        with self._lock:
            if risk_level == "HIGH":
                # HIGH always allowed — update high timer
                self._last_high_t      = now
                self._last_narration_t = now
                return True, "high_risk_bypass"
            # Non-HIGH: check cooldown
            elapsed = now - self._last_narration_t
            if elapsed >= NARRATION_COOLDOWN:
                self._last_narration_t = now
                return True, "cooldown_ok"
            return False, f"cooldown ({elapsed:.1f}s < {NARRATION_COOLDOWN}s)"

    def record_narration(self, risk_level: str):
        """
        Mark that a narration was emitted.  Call after TTS is triggered.
        (narration_allowed already updates the timer, but this is provided
        for cases where narration is triggered outside the gate.)
        """
        now = time.time()
        with self._lock:
            self._last_narration_t = now
            if risk_level == "HIGH":
                self._last_high_t = now

    @property
    def unstable_frame_count(self) -> int:
        with self._lock:
            return self._unstable_frames


# Module-level singleton
stability_filter = StabilityFilter()
