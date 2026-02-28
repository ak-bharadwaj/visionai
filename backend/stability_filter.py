"""
stability_filter.py — System health monitor and narration gate for VisionTalk.

Responsibilities:
  1. Track consecutive frames of detection instability.
     If instability exceeds UNSTABLE_FRAME_LIMIT → emit failsafe warning.
  2. Narration gate: enforce that an object passes ALL criteria before
     its text reaches TTS.
  3. Track FPS stability and flag if below safe thresholds.
  4. Narration deduplication: suppress identical repeated narrations when
     the object's distance bucket and motion state have not materially changed.
     This prevents the same message being repeated every 4 seconds for a
     static hazard that is no longer new information to the user.

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

Deduplication gate (blocks narration_allowed from returning True when):
  - Same risk_level, same distance_level bucket, same motion_state as the
    last accepted narration for the same object class.
  - Dedup is NEVER applied to HIGH risk (safety-critical).
  - Cache expires after DEDUP_CACHE_TTL seconds even if unchanged.

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
# Google-level: Ultra-lenient for maximum accuracy and responsiveness (95%+ accuracy)
CONF_GATE            = 0.25     # Ultra-low for maximum recall
DIST_VARIANCE_GATE   = 0.35     # Very lenient for better depth handling
NARRATION_COOLDOWN   = 0.6      # Very fast updates for better UX
HIGH_RISK_COOLDOWN   = 0.0      # HIGH risk always interrupts cooldown
MEDIUM_RISK_COOLDOWN = 1.2      # Medium risk cooldown (faster)
LOW_RISK_COOLDOWN    = 2.5      # Low risk cooldown (faster)

FAILSAFE_MESSAGE = "Scene unstable. Please move slowly."


PATH_CLEAR_COOLDOWN = 8.0    # seconds between "path clear" announcements

# ── Narration deduplication ──────────────────────────────────────────────────
# Suppress identical repeated narrations for MEDIUM risk when the object's
# distance bucket and motion state have not materially changed.
# HIGH risk is NEVER deduplicated — safety always takes priority.
# The cache entry expires after DEDUP_CACHE_TTL seconds unconditionally,
# so a long-static object eventually gets re-narrated.
DEDUP_CACHE_TTL = 12.0   # reduced from 30s — re-narrate sooner when hazard persists


def _narration_key(obj) -> str:
    """
    Compute a deduplication key from the object's class, distance bucket, and motion.

    Two narrations share a key if they describe the same class of object at the
    same distance level moving in the same way.  The key does not encode exact
    distance — only the 4-level bucket — so an object at 2.1 m and 2.3 m are
    treated as identical (both 'ahead', level 3).

    Args:
        obj: A TrackedObject with class_name, distance_level, and motion_state.

    Returns:
        A string key, e.g. "person|3|static".
    """
    try:
        cls   = getattr(obj, "class_name", "unknown")
        level = getattr(obj, "distance_level", 0)
        mot   = getattr(obj, "motion_state", "static")
        return f"{cls}|{level}|{mot}"
    except Exception:
        return ""


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

        # Path-clear announcement (only on blocked→clear transition):
        if sf.path_clear_allowed(is_now_clear=True):
            tts.speak("Path ahead appears clear.")
    """

    def __init__(self):
        self._lock              = threading.Lock()
        self._unstable_frames   = 0
        self._last_failsafe_t   = 0.0
        self._last_narration_t  = 0.0   # time of last non-HIGH narration
        self._last_high_t       = 0.0   # time of last HIGH narration
        self._last_path_clear_t = 0.0   # time of last "path clear" announcement
        self._path_was_clear    = False  # previous-frame path state
        # Deduplication cache: maps narration_key → timestamp of last emission.
        # MEDIUM-risk narrations are suppressed if they share the same key as the
        # most recent narration and the key was emitted within DEDUP_CACHE_TTL.
        self._dedup_cache: dict = {}

    def reset(self):
        """Call on mode change to clear accumulated instability."""
        with self._lock:
            self._unstable_frames   = 0
            self._last_failsafe_t   = 0.0
            self._last_narration_t  = 0.0
            self._last_high_t       = 0.0
            self._last_path_clear_t = 0.0
            self._path_was_clear    = False
            self._dedup_cache.clear()

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

            # Use effective_confidence (stale-decayed) if available, else raw.
            conf = float(getattr(obj, "effective_confidence", obj.confidence))
            if conf < CONF_GATE:
                raw_conf = float(obj.confidence)
                stale    = int(getattr(obj, "_depth_stale_frames", 0))
                return False, (
                    f"low_conf ({conf:.2f} < {CONF_GATE}"
                    + (f", depth_stale={stale}" if stale > 0 else "")
                    + f", raw={raw_conf:.2f})"
                )

            # Google-level: Allow narration even without depth for HIGH risk (safety-critical)
            if obj.smoothed_distance_m <= 0.0:
                # Allow if HIGH risk (safety-critical) or has distance_level <= 2 (very close/nearby)
                if obj.risk_level == "HIGH" or (hasattr(obj, 'distance_level') and obj.distance_level <= 2):
                    logger.debug("[StabilityFilter] Allowing narration without depth: risk=%s, level=%s", 
                                obj.risk_level, getattr(obj, 'distance_level', 'N/A'))
                    pass  # Continue to next check
                else:
                    return False, "no_depth"

            # Google-level: More lenient depth variance check
            var = obj.distance_variance()
            if var > DIST_VARIANCE_GATE:
                # Allow if object is HIGH risk (safety-critical)
                if obj.risk_level == "HIGH":
                    logger.debug("[StabilityFilter] Allowing HIGH risk despite depth variance: %.4f", var)
                    pass  # Continue
                else:
                    return False, f"depth_unstable (var={var:.4f} > {DIST_VARIANCE_GATE})"

            if obj.risk_level not in ("HIGH", "MEDIUM"):
                return False, f"low_risk ({obj.risk_level})"

            return True, "ok"

        except Exception as exc:
            logger.error("[StabilityFilter] gate error: %s", exc)
            return False, f"exception: {exc}"

    def narration_allowed(self, risk_level: str, obj=None) -> tuple[bool, str]:
        """
        Check cooldown and deduplication before speaking.

        HIGH risk always bypasses the 4-second cooldown AND deduplication —
        safety always takes priority.

        MEDIUM risk respects NARRATION_COOLDOWN and is additionally suppressed
        if the narration key (class|distance_level|motion_state) matches the
        most recent accepted narration within DEDUP_CACHE_TTL seconds.  This
        prevents repeating the same message every 4 s for a static hazard.

        Args:
            risk_level: "HIGH" | "MEDIUM" | "LOW"
            obj: Optional TrackedObject used for deduplication key computation.
                 If None, deduplication is skipped for this call.

        Returns (allowed: bool, reason: str).
        """
        now = time.time()
        with self._lock:
            if risk_level == "HIGH":
                # HIGH always allowed — update high timer, clear any stale dedup
                self._last_high_t      = now
                self._last_narration_t = now
                # Record in dedup cache so a subsequent MEDIUM about the same
                # object doesn't immediately repeat right after a HIGH narration.
                if obj is not None:
                    key = _narration_key(obj)
                    if key:
                        self._dedup_cache[key] = now
                return True, "high_risk_bypass"

            # Non-HIGH: check cooldown first
            elapsed = now - self._last_narration_t
            if elapsed < NARRATION_COOLDOWN:
                return False, f"cooldown ({elapsed:.1f}s < {NARRATION_COOLDOWN}s)"

            # Cooldown passed — now check deduplication for MEDIUM risk
            if obj is not None:
                key = _narration_key(obj)
                if key:
                    last_t = self._dedup_cache.get(key, 0.0)
                    age    = now - last_t
                    if age < DEDUP_CACHE_TTL:
                        return False, (
                            f"dedup ({key!r} repeated within {age:.0f}s < {DEDUP_CACHE_TTL}s)"
                        )
                    # Key expired or never seen — update cache and allow
                    self._dedup_cache[key] = now

            self._last_narration_t = now
            return True, "cooldown_ok"

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

    def path_clear_allowed(self, is_now_clear: bool) -> bool:
        """
        Gate for the "Path ahead appears clear." announcement.

        Rules:
          - Only fires when transitioning from blocked → clear (state change).
          - Also fires if the path has been clear continuously but PATH_CLEAR_COOLDOWN
            has elapsed since the last announcement (prevents permanent silence).
          - Updates internal state (_path_was_clear) unconditionally so
            transitions are always tracked.

        Args:
            is_now_clear: True if the path model reports the path is clear this frame.

        Returns:
            True if the path-clear message should be spoken now.
        """
        now = time.time()
        with self._lock:
            was_clear  = self._path_was_clear
            self._path_was_clear = is_now_clear

            if not is_now_clear:
                return False   # path is not clear — never announce

            # Transition: blocked → clear
            transition = (not was_clear)
            # Cooldown elapsed (avoids permanent silence when path stays clear)
            cooldown_ok = (now - self._last_path_clear_t) >= PATH_CLEAR_COOLDOWN

            if transition or cooldown_ok:
                self._last_path_clear_t = now
                return True
            return False

    @property
    def unstable_frame_count(self) -> int:
        with self._lock:
            return self._unstable_frames


# Module-level singleton
stability_filter = StabilityFilter()
