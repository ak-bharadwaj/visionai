"""
stability_monitor.py — Frame-level stability tracking for VisionTalk.

Tracks consecutive low-quality or empty detection frames and exposes:
  - is_stable()  → bool: True when pipeline is operating normally.
  - should_warn() → bool: True when fail-safe message should be spoken.
  - record_frame(fps, detection_count): call once per NAVIGATE frame.

Fail-safe trigger: UNSTABLE_FRAME_LIMIT consecutive frames with
  fps < FPS_UNSTABLE_THRESH OR detection_count == 0.
"""

import time
import threading
import logging
from collections import deque

logger = logging.getLogger(__name__)

# ── Tuning constants ─────────────────────────────────────────────────
FPS_UNSTABLE_THRESH  = 2.0    # FPS below this is considered unstable
UNSTABLE_FRAME_LIMIT = 10     # consecutive bad frames before warn
WARN_COOLDOWN_SEC    = 8.0    # minimum seconds between fail-safe warnings
FPS_WINDOW           = 15     # rolling window for FPS smoothing


class StabilityMonitor:
    def __init__(self):
        self._lock             = threading.Lock()
        self._consecutive_bad  = 0
        self._last_warn_t      = 0.0
        self._fps_window: deque = deque(maxlen=FPS_WINDOW)

    def record_frame(self, fps: float, detection_count: int):
        """
        Call once per NAVIGATE frame.

        A frame is "bad" if:
          - fps < FPS_UNSTABLE_THRESH, OR
          - detection_count == 0  (no objects visible, may be camera dropout)
        """
        with self._lock:
            self._fps_window.append(fps)
            is_bad = (fps < FPS_UNSTABLE_THRESH)
            if is_bad:
                self._consecutive_bad += 1
            else:
                self._consecutive_bad = 0

    def is_stable(self) -> bool:
        """True when consecutive bad frame count is below threshold."""
        with self._lock:
            return self._consecutive_bad < UNSTABLE_FRAME_LIMIT

    def should_warn(self) -> bool:
        """
        True if the monitor has been unstable long enough AND enough time
        has passed since the last fail-safe message.

        Resets the warn cooldown timer when it returns True.
        """
        now = time.time()
        with self._lock:
            if self._consecutive_bad < UNSTABLE_FRAME_LIMIT:
                return False
            if now - self._last_warn_t < WARN_COOLDOWN_SEC:
                return False
            self._last_warn_t = now
            return True

    def reset(self):
        """Call on mode change to clear accumulated state."""
        with self._lock:
            self._consecutive_bad = 0
            self._fps_window.clear()

    @property
    def smoothed_fps(self) -> float:
        """Average FPS over the rolling window."""
        with self._lock:
            if not self._fps_window:
                return 0.0
            return sum(self._fps_window) / len(self._fps_window)


# Module-level singleton
stability_monitor = StabilityMonitor()
