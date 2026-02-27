"""
diagnostics.py — Structured logging and metric tracking for VisionTalk.

Logged events:
  - detection_filtered   : detection dropped by confidence gate
  - object_confirmed     : object reaches confirmed state
  - narration_emitted    : text was spoken + why
  - narration_suppressed : text was NOT spoken + reason
  - failsafe_emitted     : "Scene unstable" was spoken
  - depth_suppressed     : distance suppressed due to high variance

Metrics tracked (in-memory, reset on restart):
  - detection_count          : total detections arriving at confidence gate
  - confirmed_count          : detections that passed gate + tracking
  - narration_count          : narrations emitted
  - suppression_count        : narrations suppressed
  - false_narration_estimate : manual counter (incremented externally)
  - failsafe_count           : times failsafe message was spoken
  - distance_errors          : list of |estimated - ground_truth| samples
    (populated externally via record_distance_error())

Measurable targets referenced in design spec:
  - Narration stability rate : narration_count / (narration_count + suppression_count)
  - False narration rate     : false_narration_estimate / narration_count
  - Distance error variance  : np.var(distance_errors)
  - Average time to warn     : tracked via record_hazard_detection_time()
    (should be < 1.0 second for HIGH risk objects)

Design constraints:
  - All methods are non-blocking and thread-safe.
  - Metrics are append-only; never modify past records.
  - Never raises; log errors are swallowed.
"""

import time
import threading
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Structured event logger — goes to same log file but easy to grep
_diag = logging.getLogger("visiontalk.diagnostics")


class Diagnostics:
    def __init__(self):
        self._lock = threading.Lock()

        # Counters
        self.detection_count          = 0
        self.confirmed_count          = 0
        self.narration_count          = 0
        self.suppression_count        = 0
        self.false_narration_estimate = 0
        self.failsafe_count           = 0

        # Samples
        self._distance_errors: list[float]  = []
        self._time_to_warn:    list[float]  = []   # seconds from first-seen to narration

    # ── Event loggers ────────────────────────────────────────────────────────

    def detection_filtered(self, class_name: str, confidence: float):
        """Log a detection dropped by the confidence gate."""
        with self._lock:
            self.detection_count += 1
        _diag.debug(
            "FILTERED class=%s conf=%.3f reason=below_conf_gate",
            class_name, confidence,
        )

    def object_confirmed(self, track_id: int, class_name: str, frames_seen: int):
        """Log when an object reaches confirmed status."""
        with self._lock:
            self.detection_count += 1
            self.confirmed_count += 1
        _diag.info(
            "CONFIRMED id=%d class=%s frames_seen=%d",
            track_id, class_name, frames_seen,
        )

    def narration_emitted(self, track_id: int, class_name: str, risk_level: str,
                          risk_score: float, confidence: float, distance_m: float,
                          text: str, reason: str = ""):
        """Log a narration that was spoken."""
        with self._lock:
            self.narration_count += 1
        _diag.info(
            "NARRATED id=%d class=%s risk=%s score=%.3f conf=%.2f dist=%.2fm "
            "reason=%s text=%r",
            track_id, class_name, risk_level, risk_score, confidence,
            distance_m, reason or "ok", text,
        )

    def narration_suppressed(self, track_id: int, class_name: str, reason: str,
                             risk_level: str = "?", risk_score: float = 0.0):
        """Log a narration that was blocked by the narration gate."""
        with self._lock:
            self.suppression_count += 1
        _diag.debug(
            "SUPPRESSED id=%d class=%s risk=%s score=%.3f reason=%s",
            track_id, class_name, risk_level, risk_score, reason,
        )

    def failsafe_emitted(self):
        """Log when the "Scene unstable" failsafe is spoken."""
        with self._lock:
            self.failsafe_count += 1
        _diag.warning("FAILSAFE scene_unstable")

    def depth_suppressed(self, track_id: int, class_name: str, variance: float):
        """Log when depth reporting is suppressed due to high variance."""
        _diag.debug(
            "DEPTH_SUPPRESSED id=%d class=%s variance=%.4f",
            track_id, class_name, variance,
        )

    def record_distance_error(self, estimated_m: float, ground_truth_m: float):
        """
        Record absolute distance error for offline accuracy analysis.
        Call this with ground-truth measurements when available (e.g., test runs).
        """
        with self._lock:
            self._distance_errors.append(abs(estimated_m - ground_truth_m))

    def record_hazard_detection_time(self, first_seen_t: float, narration_t: float):
        """
        Record the elapsed time from when an object was first seen (first_seen_t)
        to when its first narration was emitted.  Target: < 1.0 s for HIGH risk.
        """
        elapsed = narration_t - first_seen_t
        with self._lock:
            self._time_to_warn.append(elapsed)
        if elapsed > 1.0:
            _diag.warning(
                "SLOW_WARNING time_to_warn=%.3fs (target < 1.0s)", elapsed
            )

    # ── Derived metrics ──────────────────────────────────────────────────────

    def narration_stability_rate(self) -> float:
        """Fraction of candidate narrations that were actually spoken."""
        with self._lock:
            total = self.narration_count + self.suppression_count
            if total == 0:
                return 1.0
            return self.narration_count / total

    def false_narration_rate(self) -> float:
        """Estimated fraction of spoken narrations that were incorrect."""
        with self._lock:
            if self.narration_count == 0:
                return 0.0
            return self.false_narration_estimate / self.narration_count

    def distance_error_variance(self) -> float:
        """Variance of absolute distance errors (metres²)."""
        with self._lock:
            if len(self._distance_errors) < 2:
                return 0.0
            return float(np.var(self._distance_errors))

    def mean_time_to_warn(self) -> float:
        """Average seconds from first detection to narration."""
        with self._lock:
            if not self._time_to_warn:
                return 0.0
            return float(np.mean(self._time_to_warn))

    def summary(self) -> dict:
        """Return a snapshot of all metrics as a plain dict."""
        with self._lock:
            return {
                "detection_count":          self.detection_count,
                "confirmed_count":          self.confirmed_count,
                "narration_count":          self.narration_count,
                "suppression_count":        self.suppression_count,
                "false_narration_estimate": self.false_narration_estimate,
                "failsafe_count":           self.failsafe_count,
                "narration_stability_rate": self.narration_stability_rate(),
                "false_narration_rate":     self.false_narration_rate(),
                "distance_error_variance":  self.distance_error_variance(),
                "mean_time_to_warn_s":      self.mean_time_to_warn(),
                "distance_error_samples":   len(self._distance_errors),
                "time_to_warn_samples":     len(self._time_to_warn),
            }


# Module-level singleton
diagnostics = Diagnostics()
