"""
diagnostics.py — Structured logging and metric tracking for VisionTalk.

Logged events:
  - detection_filtered   : detection dropped by confidence gate
  - object_confirmed     : object reaches confirmed state
  - narration_emitted    : text was spoken + why
  - narration_suppressed : text was NOT spoken + reason
  - failsafe_emitted     : "Scene unstable" was spoken
  - depth_suppressed     : distance suppressed due to high variance
  - voice_command_received : STT classified a voice command
  - voice_command_rejected : STT heard speech but classification failed / was deduped
  - mode_changed         : mode_manager.set_mode() was called
  - tts_interrupted      : high-priority speak() cleared the TTS queue

Metrics tracked (in-memory, reset on restart):
  - detection_count          : total detections arriving at confidence gate
  - confirmed_count          : detections that passed gate + tracking
  - narration_count          : narrations emitted
  - suppression_count        : narrations suppressed
  - false_narration_estimate : manual counter (incremented externally)
  - failsafe_count           : times failsafe message was spoken
  - id_switch_count          : tracker ID reassignments (track fragmentation)
  - tracks_created           : total new track IDs ever spawned
  - tracks_revived           : dormant tracks successfully revived
  - narrations_without_object: narrations emitted when no object was visible
                               in the current frame (proxy for false positives)
  - depth_measurements       : total depth values offered to update_distance()
  - depth_rejections         : depth values discarded (jump reject or variance gate)
  - voice_commands_received  : STT commands successfully classified
  - voice_commands_rejected  : STT results that did not match any command
  - mode_changes             : total mode transitions
  - tts_interruptions        : high-priority TTS queue clears
  - distance_errors          : list of |estimated - ground_truth| samples
    (populated externally via record_distance_error())
  - detection_latency_s      : list of (first_seen → detected) elapsed seconds
    (populated via record_detection_latency())

Measurable targets referenced in design spec:
  - Narration stability rate : narration_count / (narration_count + suppression_count)
  - False narration rate     : false_narration_estimate / narration_count
  - Distance error variance  : np.var(distance_errors)
  - Average time to warn     : tracked via record_hazard_detection_time()
    (should be < 1.0 second for HIGH risk objects)
  - Track fragmentation rate : id_switch_count / max(tracks_created, 1)
  - Revival rate             : tracks_revived / max(tracks_created, 1)
  - Depth failure rate       : depth_rejections / max(depth_measurements, 1)
  - Mean detection latency   : mean(detection_latency_s)
  - Command recognition rate : voice_commands_received / (voice_commands_received + voice_commands_rejected)

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
        self._lock = threading.RLock()

        # Counters
        self.detection_count          = 0
        self.confirmed_count          = 0
        self.narration_count          = 0
        self.suppression_count        = 0
        self.false_narration_estimate = 0
        self.failsafe_count           = 0
        self.id_switch_count          = 0    # tracker reassigned an existing track ID
        self.tracks_created           = 0    # total new track IDs ever spawned
        self.tracks_revived           = 0    # dormant tracks successfully revived
        self.narrations_without_object = 0  # narrations emitted with no visible object

        # Depth measurement counters
        self.depth_measurements       = 0   # total depth values offered
        self.depth_rejections         = 0   # depth values discarded (jump/variance)

        # Interaction / voice counters
        self.voice_commands_received  = 0   # STT commands successfully classified
        self.voice_commands_rejected  = 0   # heard speech but no command matched / deduped
        self.mode_changes             = 0   # total mode transitions (voice + UI)
        self.tts_interruptions        = 0   # high-priority TTS queue clears

        # Samples
        self._distance_errors:    list[float] = []
        self._time_to_warn:       list[float] = []   # seconds from first-seen to narration
        self._detection_latency:  list[float] = []   # seconds from frame_visible to frame_detected

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

    def track_created(self, track_id: int, class_name: str):
        """
        Log when the tracker spawns a brand-new track ID (not a revival).
        Increments tracks_created — used to compute track fragmentation rate:
            fragmentation_rate = id_switch_count / tracks_created
        A high ratio (> 0.10) indicates the IoU matching or dormant revival is
        not working well for a given scene.
        """
        with self._lock:
            self.tracks_created += 1
        _diag.debug("TRACK_CREATED id=%d class=%s", track_id, class_name)

    def track_revived(self, track_id: int, class_name: str):
        """
        Log when the tracker revives a dormant track instead of creating a new ID.
        Increments tracks_revived — used to compute revival_rate():
            revival_rate = tracks_revived / max(tracks_created, 1)
        A healthy revival rate (> 0.0) means dormant revival is actively reducing
        track fragmentation.
        """
        with self._lock:
            self.tracks_revived += 1
        _diag.debug("TRACK_REVIVED id=%d class=%s", track_id, class_name)

    def depth_measurement_accepted(self):
        """Log a depth reading that passed all rejection gates."""
        with self._lock:
            self.depth_measurements += 1

    def depth_measurement_rejected(self, reason: str = ""):
        """
        Log a depth reading discarded by the jump-reject or variance gate.
        Increments both depth_measurements (total offered) and depth_rejections.
        reason: one of 'jump_reject', 'variance_gate', or a custom string.
        """
        with self._lock:
            self.depth_measurements += 1
            self.depth_rejections += 1
        _diag.debug("DEPTH_REJECTED reason=%s", reason or "unknown")

    def narration_emitted(self, track_id: int, class_name: str, risk_level: str,
                          risk_score: float, confidence: float, distance_m: float,
                          text: str, reason: str = "", n_visible_objects: int = -1):
        """
        Log a narration that was spoken.

        Args:
            n_visible_objects: number of confirmed tracked objects in the current
                frame.  If 0, the narration fired with no objects visible — a
                likely false positive.  Pass -1 (default) to skip this check.
        """
        with self._lock:
            self.narration_count += 1
            if n_visible_objects == 0:
                self.narrations_without_object += 1
        _diag.info(
            "NARRATED id=%d class=%s risk=%s score=%.3f conf=%.2f dist=%.2fm "
            "reason=%s text=%r n_visible=%d",
            track_id, class_name, risk_level, risk_score, confidence,
            distance_m, reason or "ok", text, n_visible_objects,
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

    def id_switch(self, old_id: int, new_id: int, class_name: str):
        """
        Log when the tracker cannot match a detection to an existing track and
        issues a new ID for what is likely the same physical object.
        High id_switch_count indicates poor IoU matching or extreme occlusion.
        """
        with self._lock:
            self.id_switch_count += 1
        _diag.debug(
            "ID_SWITCH class=%s old_id=%d new_id=%d", class_name, old_id, new_id,
        )

    def record_distance_error(self, estimated_m: float, ground_truth_m: float):
        """
        Record absolute distance error for offline accuracy analysis.
        Call this with ground-truth measurements when available (e.g., test runs).
        """
        with self._lock:
            self._distance_errors.append(abs(estimated_m - ground_truth_m))

    def record_detection_latency(self, frame_visible_t: float, frame_detected_t: float):
        """
        Record the elapsed time from when an object first appeared in frame
        (frame_visible_t) to when the detector first returned it as a confirmed
        detection (frame_detected_t).

        This measures the pipeline's responsiveness: how many frames / seconds
        pass between an object entering the scene and the system knowing it's
        there.  Target: < 0.3 s (≈ 3 YOLO frames at 10 FPS).

        Call site: tracker.py when a new track is confirmed for the first time
        (frames_seen == MIN_FRAMES_CONFIRM).
        """
        elapsed = frame_detected_t - frame_visible_t
        if elapsed < 0:
            return   # clock skew / bad call — silently ignore
        with self._lock:
            self._detection_latency.append(elapsed)
        if elapsed > 0.5:
            _diag.warning(
                "SLOW_DETECTION latency=%.3fs (target < 0.30s)", elapsed
            )

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

    def voice_command_received(self, action: str, detail: str = ""):
        """
        Log a voice command that was successfully classified and dispatched.

        Args:
            action: the action string (e.g. 'set_mode', 'find_object', 'ask')
            detail: supplementary info (e.g. mode name, find target, question)
        """
        with self._lock:
            self.voice_commands_received += 1
        _diag.info("VOICE_CMD_RECEIVED action=%s detail=%r", action, detail)

    def voice_command_rejected(self, reason: str = "unrecognised"):
        """
        Log a voice transcription that did not map to any command.

        Args:
            reason: 'unrecognised' (no pattern match) or 'deduped' (repetition suppression)
        """
        with self._lock:
            self.voice_commands_rejected += 1
        _diag.debug("VOICE_CMD_REJECTED reason=%s", reason)

    def mode_changed(self, old_mode: str, new_mode: str, source: str = ""):
        """
        Log a mode transition.

        Args:
            old_mode: previous mode string
            new_mode: new mode string
            source: 'voice', 'ui', or '' for unknown
        """
        with self._lock:
            self.mode_changes += 1
        _diag.info("MODE_CHANGED %s→%s source=%s", old_mode, new_mode, source or "?")

    def tts_interrupted(self, reason: str = "high_priority"):
        """Log when a high-priority speak() clears the TTS queue."""
        with self._lock:
            self.tts_interruptions += 1
        _diag.debug("TTS_INTERRUPTED reason=%s", reason)

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

    def tracking_id_switch_rate(self) -> float:
        """Fraction of confirmed objects that involved an ID switch."""
        with self._lock:
            if self.confirmed_count == 0:
                return 0.0
            return self.id_switch_count / self.confirmed_count

    def track_fragmentation_rate(self) -> float:
        """
        Ratio of ID switches to total tracks created.

        A value > 0.10 (10%) means more than 1 in 10 new tracks is actually
        a re-detection of an already-seen object that lost its ID — indicating
        the tracker is fragmenting tracks faster than dormant revival can catch.

        Returns 0.0 when no tracks have been created yet.
        """
        with self._lock:
            if self.tracks_created == 0:
                return 0.0
            return self.id_switch_count / self.tracks_created

    def revival_rate(self) -> float:
        """
        Ratio of revived dormant tracks to total tracks created.

        A value > 0.0 means dormant revival is actively recovering track IDs
        instead of spawning fragmented new ones.  A high revival rate paired
        with a low fragmentation rate indicates healthy tracker continuity.

        Returns 0.0 when no tracks have been created yet.
        """
        with self._lock:
            if self.tracks_created == 0:
                return 0.0
            return self.tracks_revived / self.tracks_created

    def depth_failure_rate(self) -> float:
        """
        Fraction of offered depth measurements that were rejected.

        A value > 0.20 (20%) suggests the depth estimator is producing
        excessive flicker or jump spikes — consider tuning DEPTH_JUMP_REJECT_M
        or increasing the variance gate threshold.

        Returns 0.0 when no depth measurements have been offered yet.
        """
        with self._lock:
            if self.depth_measurements == 0:
                return 0.0
            return self.depth_rejections / self.depth_measurements

    def narration_without_object_rate(self) -> float:
        """
        Fraction of narrations that fired when no object was visible.

        A non-zero value indicates confirmed tracks are persisting (via EMA /
        dormant pool) after the object has left the scene, and the narration
        cooldown has not fully suppressed them.  Target: < 0.05.
        """
        with self._lock:
            if self.narration_count == 0:
                return 0.0
            return self.narrations_without_object / self.narration_count

    def command_recognition_rate(self) -> float:
        """
        Fraction of transcribed speech chunks that matched a voice command.

        voice_commands_received / (voice_commands_received + voice_commands_rejected)

        A low value (< 0.50) indicates either poor STT accuracy for this
        environment or the intent classifier is too narrow.  Target: > 0.70.

        Returns 1.0 when no voice input has been received yet (no data = no failure).
        """
        with self._lock:
            total = self.voice_commands_received + self.voice_commands_rejected
            if total == 0:
                return 1.0
            return self.voice_commands_received / total

    def mean_detection_latency(self) -> float:
        """Mean seconds from frame_visible to frame_detected across all samples."""
        with self._lock:
            if not self._detection_latency:
                return 0.0
            return float(np.mean(self._detection_latency))

    def summary(self) -> dict:
        """Return a snapshot of all metrics as a plain dict."""
        with self._lock:
            return {
                "detection_count":              self.detection_count,
                "confirmed_count":              self.confirmed_count,
                "narration_count":              self.narration_count,
                "suppression_count":            self.suppression_count,
                "false_narration_estimate":     self.false_narration_estimate,
                "failsafe_count":               self.failsafe_count,
                "id_switch_count":              self.id_switch_count,
                "tracks_created":               self.tracks_created,
                "tracks_revived":               self.tracks_revived,
                "narrations_without_object":    self.narrations_without_object,
                "depth_measurements":           self.depth_measurements,
                "depth_rejections":             self.depth_rejections,
                "voice_commands_received":      self.voice_commands_received,
                "voice_commands_rejected":      self.voice_commands_rejected,
                "mode_changes":                 self.mode_changes,
                "tts_interruptions":            self.tts_interruptions,
                "narration_stability_rate":     self.narration_stability_rate(),
                "false_narration_rate":         self.false_narration_rate(),
                "tracking_id_switch_rate":      self.tracking_id_switch_rate(),
                "track_fragmentation_rate":     self.track_fragmentation_rate(),
                "revival_rate":                 self.revival_rate(),
                "depth_failure_rate":           self.depth_failure_rate(),
                "narration_without_object_rate":self.narration_without_object_rate(),
                "command_recognition_rate":     self.command_recognition_rate(),
                "distance_error_variance":      self.distance_error_variance(),
                "mean_time_to_warn_s":          self.mean_time_to_warn(),
                "mean_detection_latency_s":     self.mean_detection_latency(),
                "distance_error_samples":       len(self._distance_errors),
                "time_to_warn_samples":         len(self._time_to_warn),
                "detection_latency_samples":    len(self._detection_latency),
            }


# Module-level singleton
diagnostics = Diagnostics()
