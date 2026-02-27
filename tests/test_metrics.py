"""
test_metrics.py — Measurable performance metrics for the VisionTalk pipeline.

These tests use synthetic data to compute and assert the following KPIs:

  1. False narration rate (FNR)
       Objects that should NOT be narrated (LOW risk / unconfirmed / low conf)
       must never pass the narration gate.
       Target: FNR == 0%

  2. Narration stability rate (NSR)
       Out of all HIGH/MEDIUM confirmed objects with stable depth, the
       fraction that pass the gate.
       Target: NSR >= 95%

  3. Distance error variance
       EMA-smoothed distance should converge within 5 steps for a constant
       input signal.  Residual variance must be < 0.02 m².
       Target: variance < 0.02 m²

  4. Average time-to-warn for HIGH risk
       From the moment an object appears in the tracker to the first HIGH risk
       narration, the pipeline must react within 3 * frame_period seconds
       (the confirmation window).  No artificial delay beyond the 3-frame gate.
       Target: Δt ≤ 3 * (1/15) ≈ 0.2 s  (generous upper bound)

  5. False positive rate — confidence entry gate
       Detections with conf < CONF_DETECT (0.35) must NEVER enter the tracker.
       Detections with CONF_DETECT <= conf < CONF_GATE (0.60) may enter the
       tracker for history accumulation but are blocked from narration by
       stability_filter.object_passes_gate().
       Target: 0 tracks created for conf < CONF_DETECT
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from unittest.mock import MagicMock

from backend.tracker import ObjectTracker, TrackedObject, MIN_FRAMES_CONFIRM, CONF_GATE, CONF_DETECT
from backend.stability_filter import StabilityFilter, DIST_VARIANCE_GATE
from backend.risk_engine import score, score_all
from backend.narrator import select_highest_risk, build
from backend.detector import Detection
from backend.diagnostics import Diagnostics


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_det(class_name="person", conf=0.85, x1=200, y1=100, x2=400, y2=400, class_id=0):
    return Detection(class_id=class_id, class_name=class_name, confidence=conf,
                     x1=x1, y1=y1, x2=x2, y2=y2)


def make_good_obj(risk_level="HIGH", dist_m=1.5, variance=0.01):
    obj = MagicMock()
    obj.confirmed = True
    obj.confidence = 0.85
    obj.effective_confidence = 0.85   # no stale depth penalty
    obj.smoothed_distance_m = dist_m
    obj.distance_variance.return_value = variance
    obj.risk_level = risk_level
    obj.risk_score = 0.9 if risk_level == "HIGH" else 0.6
    obj.motion_state = "approaching"
    obj.path_overlap_ratio = 0.6
    obj.class_name = "person"
    obj.direction = "12 o'clock"
    obj.id = 1
    return obj


def make_bad_obj(reason="low_risk"):
    obj = MagicMock()
    obj.confirmed = True
    obj.confidence = 0.85
    obj.effective_confidence = 0.85   # default full confidence
    obj.smoothed_distance_m = 2.0
    obj.distance_variance.return_value = 0.01
    if reason == "low_risk":
        obj.risk_level = "LOW"
        obj.risk_score = 0.3
    elif reason == "unconfirmed":
        obj.confirmed = False
        obj.frames_seen = 1
        obj.risk_level = "HIGH"
        obj.risk_score = 0.9
    elif reason == "low_conf":
        obj.confidence = 0.40
        obj.effective_confidence = 0.40   # stale or raw — both below CONF_GATE
        obj.risk_level = "HIGH"
        obj.risk_score = 0.9
    elif reason == "no_depth":
        obj.smoothed_distance_m = 0.0
        obj.risk_level = "HIGH"
        obj.risk_score = 0.9
    elif reason == "noisy_depth":
        obj.distance_variance.return_value = DIST_VARIANCE_GATE + 0.05
        obj.risk_level = "HIGH"
        obj.risk_score = 0.9
    return obj


# ── Metric 1: False narration rate ────────────────────────────────────────────

class TestFalseNarrationRate:
    """
    Objects that must NOT be narrated must NEVER pass the gate.
    Expected: 0 / N (0%)
    """

    BAD_REASONS = ["low_risk", "unconfirmed", "low_conf", "no_depth", "noisy_depth"]
    N_TRIALS = 50

    def test_false_narration_rate_is_zero(self):
        sf = StabilityFilter()
        false_narrations = 0

        for reason in self.BAD_REASONS:
            for _ in range(self.N_TRIALS):
                obj = make_bad_obj(reason)
                allowed, _ = sf.object_passes_gate(obj)
                if allowed:
                    false_narrations += 1
                    print(f"FALSE NARRATION: reason={reason}, obj={obj}")

        total = len(self.BAD_REASONS) * self.N_TRIALS
        fnr = false_narrations / total
        print(f"\nFalse Narration Rate: {false_narrations}/{total} = {fnr:.1%}")
        assert false_narrations == 0, (
            f"FNR must be 0%. Got {false_narrations} false narrations "
            f"({fnr:.1%}) out of {total} trials."
        )


# ── Metric 2: Narration stability rate ────────────────────────────────────────

class TestNarrationStabilityRate:
    """
    All fully qualified HIGH/MEDIUM objects (confirmed, conf>=0.60, depth stable)
    must pass the gate.
    Expected: >= 95% (with cooldown effects, first call per test is 100%)
    """

    N_OBJECTS = 200

    def test_stability_rate_for_qualified_objects(self):
        sf = StabilityFilter()
        passed = 0
        for _ in range(self.N_OBJECTS):
            sf_fresh = StabilityFilter()   # fresh instance per object to avoid cooldown
            obj = make_good_obj()
            allowed, _ = sf_fresh.object_passes_gate(obj)
            if allowed:
                passed += 1

        nsr = passed / self.N_OBJECTS
        print(f"\nNarration Stability Rate: {passed}/{self.N_OBJECTS} = {nsr:.1%}")
        assert nsr >= 0.95, (
            f"NSR must be >= 95%. Got {nsr:.1%} ({passed}/{self.N_OBJECTS})"
        )


# ── Metric 3: Distance error variance ────────────────────────────────────────

class TestDistanceVariance:
    """
    After 5+ EMA updates with a constant signal, residual variance must be < 0.02 m².
    """

    def test_ema_converges_on_constant_signal(self):
        obj = TrackedObject(id=1, class_name="chair", confidence=0.80)
        true_distance = 2.5

        for _ in range(10):
            obj.update_distance(true_distance)
            time.sleep(0.001)

        var = obj.distance_variance()
        print(f"\nDistance variance (constant signal, 10 steps): {var:.6f} m²")
        assert var < 0.02, (
            f"Variance should converge to < 0.02 m². Got {var:.6f} m²"
        )

    def test_ema_reduces_noise(self):
        """Smoothed distance should be more stable than raw noisy signal."""
        obj = TrackedObject(id=1, class_name="chair", confidence=0.80)
        noisy_readings = [2.0, 2.5, 1.8, 2.3, 2.1, 1.9, 2.4, 2.0, 2.2, 2.1]

        for d in noisy_readings:
            obj.update_distance(d)
            time.sleep(0.001)

        # Variance of smoothed should be less than variance of raw
        smoothed_var = obj.distance_variance()
        raw_var = float(np.var(noisy_readings))
        print(f"\nRaw variance: {raw_var:.4f} m², Smoothed variance: {smoothed_var:.4f} m²")
        assert smoothed_var < raw_var, (
            "EMA smoothing must reduce variance compared to raw signal."
        )


# ── Metric 4: Time-to-warn for HIGH risk ─────────────────────────────────────

class TestTimeToWarn:
    """
    From first detection to first valid HIGH risk narration, the latency must
    be bounded by the confirmation window (MIN_FRAMES_CONFIRM frames).

    We simulate frame processing at ~30 FPS to bound the clock-time.
    Target: ≤ 0.5 s (generous, for unit test environment variability)
    """

    SIMULATED_FPS = 30
    FRAME_PERIOD  = 1.0 / SIMULATED_FPS

    def test_time_to_warn_under_half_second(self):
        tracker = ObjectTracker()
        sf = StabilityFilter()

        det = make_det(class_name="person", conf=0.90, x1=250, y1=100, x2=390, y2=400)

        # Pre-populate a mocked depth distance via update_distance
        # (In real pipeline depth is async; here we call it directly post-confirm)
        first_warn_t = None
        t_start = time.time()

        for frame_idx in range(MIN_FRAMES_CONFIRM + 2):
            confirmed = tracker.update([det], 640, 480)

            # Inject stable depth for confirmed tracks
            for obj in confirmed:
                for _ in range(5):
                    obj.update_distance(1.5)

            score_all(confirmed)

            for obj in confirmed:
                passed, _ = sf.object_passes_gate(obj)
                if passed:
                    cool_ok, _ = sf.narration_allowed(obj.risk_level)
                    if cool_ok and first_warn_t is None:
                        first_warn_t = time.time()

            # Simulate frame timing (very short for unit test speed)
            time.sleep(self.FRAME_PERIOD * 0.1)   # 10% of real frame time

        elapsed = (first_warn_t - t_start) if first_warn_t else None

        print(f"\nTime-to-warn: {'not reached' if elapsed is None else f'{elapsed:.4f}s'}")
        assert first_warn_t is not None, (
            "HIGH risk object was never narrated — pipeline blocked all narrations"
        )
        assert elapsed < 0.5, (
            f"Time-to-warn ({elapsed:.4f}s) exceeds 0.5s target"
        )


# ── Metric 5: Confidence gate false positive rate ─────────────────────────────

class TestConfidenceGateFalsePositiveRate:
    """
    Two-tier confidence gate verification:

    With the two-tier design (CONF_DETECT=0.35 / CONF_GATE=0.60):
      - Detections with conf < CONF_DETECT must NEVER enter the tracker at all.
      - Detections with CONF_DETECT <= conf < CONF_GATE may enter the tracker
        and accumulate history, but are blocked from narration by
        stability_filter.object_passes_gate() (which checks conf >= CONF_GATE).

    This test verifies tier 1 (tracker entry gate = CONF_DETECT).
    Narration gate integrity is covered by test_stability_filter.py.
    """

    CONF_VALUES_BELOW_DETECT = [0.0, 0.10, 0.20, 0.30, CONF_DETECT - 0.01]
    N_FRAMES = MIN_FRAMES_CONFIRM + 5

    def test_no_tracks_below_conf_detect(self):
        """Detections below CONF_DETECT must never create any track at all."""
        false_positives = 0
        for conf in self.CONF_VALUES_BELOW_DETECT:
            if conf < 0:
                continue   # skip if CONF_DETECT is already at 0
            tracker = ObjectTracker()
            det = make_det(conf=conf)
            for _ in range(self.N_FRAMES):
                tracker.update([det], 640, 480)
                if tracker.all_tracks():
                    false_positives += 1
                    print(f"FALSE POSITIVE: conf={conf} entered tracker")

        assert false_positives == 0, (
            f"Confidence entry gate (CONF_DETECT={CONF_DETECT}) must block all "
            f"detections below it from entering the tracker. "
            f"Got {false_positives} false positives."
        )


class TestDiagnosticsNewMetrics:
    """Tests for the three new diagnostic metrics added this session:
       - track_created / track_fragmentation_rate
       - narration_emitted(n_visible_objects) / narration_without_object_rate
       - record_detection_latency / mean_detection_latency
    """

    def _fresh(self) -> Diagnostics:
        return Diagnostics()

    # ── track_created / track_fragmentation_rate ──────────────────────────────

    def test_track_created_increments_counter(self):
        d = self._fresh()
        assert d.tracks_created == 0
        d.track_created(1, "person")
        assert d.tracks_created == 1
        d.track_created(2, "chair")
        assert d.tracks_created == 2

    def test_track_fragmentation_rate_zero_when_no_tracks(self):
        d = self._fresh()
        assert d.track_fragmentation_rate() == 0.0

    def test_track_fragmentation_rate_computed_correctly(self):
        d = self._fresh()
        d.track_created(1, "person")
        d.track_created(2, "person")
        d.track_created(3, "person")
        d.track_created(4, "person")
        # Simulate 1 ID switch out of 4 tracks → 25 %
        d.id_switch(old_id=1, new_id=5, class_name="person")
        assert d.track_fragmentation_rate() == pytest.approx(0.25)

    def test_track_fragmentation_rate_zero_switches(self):
        d = self._fresh()
        d.track_created(1, "person")
        d.track_created(2, "person")
        # No ID switches
        assert d.track_fragmentation_rate() == pytest.approx(0.0)

    # ── narration_emitted(n_visible_objects) / narration_without_object_rate ──

    def test_narration_emitted_with_zero_objects_increments_counter(self):
        d = self._fresh()
        d.narration_emitted(
            track_id=1, class_name="person", risk_level="HIGH",
            risk_score=0.9, confidence=0.8, distance_m=1.5,
            text="Person 12 o'clock", n_visible_objects=0,
        )
        assert d.narrations_without_object == 1
        assert d.narration_count == 1

    def test_narration_emitted_with_objects_present_does_not_increment(self):
        d = self._fresh()
        d.narration_emitted(
            track_id=1, class_name="person", risk_level="HIGH",
            risk_score=0.9, confidence=0.8, distance_m=1.5,
            text="Person ahead", n_visible_objects=2,
        )
        assert d.narrations_without_object == 0
        assert d.narration_count == 1

    def test_narration_emitted_default_n_visible_skips_check(self):
        """Default n_visible_objects=-1 must not touch narrations_without_object."""
        d = self._fresh()
        d.narration_emitted(
            track_id=1, class_name="person", risk_level="MEDIUM",
            risk_score=0.5, confidence=0.75, distance_m=3.0,
            text="Person 3 o'clock",
            # n_visible_objects not passed — uses default -1
        )
        assert d.narrations_without_object == 0
        assert d.narration_count == 1

    def test_narration_without_object_rate_zero_when_no_narrations(self):
        d = self._fresh()
        assert d.narration_without_object_rate() == 0.0

    def test_narration_without_object_rate_computed_correctly(self):
        d = self._fresh()
        # 2 narrations with objects, 1 without
        for _ in range(2):
            d.narration_emitted(
                track_id=1, class_name="person", risk_level="HIGH",
                risk_score=0.9, confidence=0.8, distance_m=1.5,
                text="ok", n_visible_objects=1,
            )
        d.narration_emitted(
            track_id=2, class_name="chair", risk_level="LOW",
            risk_score=0.2, confidence=0.7, distance_m=5.0,
            text="ghost", n_visible_objects=0,
        )
        assert d.narration_without_object_rate() == pytest.approx(1 / 3)

    # ── record_detection_latency / mean_detection_latency ────────────────────

    def test_record_detection_latency_appended(self):
        d = self._fresh()
        assert len(d._detection_latency) == 0
        t0 = time.monotonic()
        d.record_detection_latency(t0, t0 + 0.1)
        assert len(d._detection_latency) == 1
        assert d._detection_latency[0] == pytest.approx(0.1, abs=1e-6)

    def test_record_detection_latency_negative_ignored(self):
        """Negative elapsed (clock skew) must be silently dropped."""
        d = self._fresh()
        t0 = time.monotonic()
        d.record_detection_latency(t0 + 1.0, t0)  # detected BEFORE visible — impossible
        assert len(d._detection_latency) == 0

    def test_mean_detection_latency_zero_when_no_samples(self):
        d = self._fresh()
        assert d.mean_detection_latency() == 0.0

    def test_mean_detection_latency_computed(self):
        d = self._fresh()
        t0 = time.monotonic()
        d.record_detection_latency(t0, t0 + 0.1)
        d.record_detection_latency(t0, t0 + 0.3)
        assert d.mean_detection_latency() == pytest.approx(0.2, abs=1e-6)

    # ── summary() includes all new keys ──────────────────────────────────────

    def test_summary_includes_new_keys(self):
        d = self._fresh()
        s = d.summary()
        required_new_keys = {
            "tracks_created",
            "tracks_revived",
            "narrations_without_object",
            "depth_measurements",
            "depth_rejections",
            "track_fragmentation_rate",
            "revival_rate",
            "depth_failure_rate",
            "narration_without_object_rate",
            "mean_detection_latency_s",
            "detection_latency_samples",
            # interaction counters
            "voice_commands_received",
            "voice_commands_rejected",
            "mode_changes",
            "tts_interruptions",
            "command_recognition_rate",
        }
        missing = required_new_keys - set(s.keys())
        assert not missing, f"summary() is missing keys: {missing}"


class TestDormantRevivalDiagnostics:
    """Tests for tracks_revived counter and revival_rate() metric."""

    def _fresh(self) -> Diagnostics:
        return Diagnostics()

    def test_tracks_revived_starts_at_zero(self):
        d = self._fresh()
        assert d.tracks_revived == 0

    def test_track_revived_increments_counter(self):
        d = self._fresh()
        d.track_revived(1, "person")
        assert d.tracks_revived == 1
        d.track_revived(1, "person")
        assert d.tracks_revived == 2

    def test_revival_rate_zero_when_no_tracks_created(self):
        d = self._fresh()
        assert d.revival_rate() == 0.0

    def test_revival_rate_computed_correctly(self):
        d = self._fresh()
        # 4 tracks created, 2 of which were revivals
        d.track_created(1, "person")
        d.track_created(2, "person")
        d.track_created(3, "chair")
        d.track_created(4, "chair")
        d.track_revived(1, "person")
        d.track_revived(3, "chair")
        assert d.revival_rate() == pytest.approx(0.5)

    def test_revival_rate_zero_when_no_revivals(self):
        d = self._fresh()
        d.track_created(1, "person")
        d.track_created(2, "chair")
        assert d.revival_rate() == pytest.approx(0.0)

    def test_revival_rate_in_summary(self):
        d = self._fresh()
        d.track_created(1, "person")
        d.track_revived(1, "person")
        s = d.summary()
        assert "revival_rate" in s
        assert s["revival_rate"] == pytest.approx(1.0)
        assert s["tracks_revived"] == 1


class TestDepthFailureDiagnostics:
    """Tests for depth_measurements, depth_rejections, and depth_failure_rate()."""

    def _fresh(self) -> Diagnostics:
        return Diagnostics()

    def test_depth_counters_start_at_zero(self):
        d = self._fresh()
        assert d.depth_measurements == 0
        assert d.depth_rejections == 0

    def test_depth_measurement_accepted_increments_measurements_only(self):
        d = self._fresh()
        d.depth_measurement_accepted()
        assert d.depth_measurements == 1
        assert d.depth_rejections == 0

    def test_depth_measurement_rejected_increments_both(self):
        d = self._fresh()
        d.depth_measurement_rejected("jump_reject")
        assert d.depth_measurements == 1
        assert d.depth_rejections == 1

    def test_depth_failure_rate_zero_when_no_measurements(self):
        d = self._fresh()
        assert d.depth_failure_rate() == 0.0

    def test_depth_failure_rate_zero_when_all_accepted(self):
        d = self._fresh()
        for _ in range(5):
            d.depth_measurement_accepted()
        assert d.depth_failure_rate() == pytest.approx(0.0)

    def test_depth_failure_rate_computed_correctly(self):
        d = self._fresh()
        # 3 accepted, 1 rejected → failure rate = 1/4 = 0.25
        for _ in range(3):
            d.depth_measurement_accepted()
        d.depth_measurement_rejected("jump_reject")
        assert d.depth_failure_rate() == pytest.approx(0.25)

    def test_depth_failure_rate_in_summary(self):
        d = self._fresh()
        d.depth_measurement_accepted()
        d.depth_measurement_rejected("variance_gate")
        s = d.summary()
        assert "depth_failure_rate" in s
        assert s["depth_failure_rate"] == pytest.approx(0.5)
        assert s["depth_measurements"] == 2
        assert s["depth_rejections"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
