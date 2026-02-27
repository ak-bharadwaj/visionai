"""
test_stability_filter.py — Unit tests for stability_filter.py

Covers:
  - record_frame() correctly increments / decrements unstable counter
  - is_system_unstable() triggers at UNSTABLE_FRAME_LIMIT
  - should_emit_failsafe() respects FAILSAFE_COOLDOWN
  - record_failsafe_emitted() resets the failsafe timer
  - object_passes_gate() enforces all narration preconditions:
      • confirmed == True
      • confidence >= 0.60
      • smoothed_distance_m > 0.0
      • distance_variance() <= DIST_VARIANCE_GATE
      • risk_level in ("HIGH", "MEDIUM")
  - narration_allowed() respects 4-second cooldown for MEDIUM
  - narration_allowed() always passes for HIGH
  - reset() clears all counters
  - Thread-safety (basic — instantiation and reset under GIL)
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock, patch

from backend.stability_filter import (
    StabilityFilter,
    UNSTABLE_FRAME_LIMIT,
    FPS_MIN_CPU,
    FPS_MIN_GPU,
    FAILSAFE_COOLDOWN,
    CONF_GATE,
    DIST_VARIANCE_GATE,
    NARRATION_COOLDOWN,
    PATH_CLEAR_COOLDOWN,
    DEDUP_CACHE_TTL,
    _narration_key,
)


# ── Helper ────────────────────────────────────────────────────────────────────

def make_good_obj(risk_level="HIGH"):
    obj = MagicMock()
    obj.confirmed = True
    obj.confidence = 0.85
    obj.effective_confidence = 0.85   # mirrors confidence — no stale depth penalty
    obj.smoothed_distance_m = 2.0
    obj.distance_variance.return_value = 0.01   # well below DIST_VARIANCE_GATE
    obj.risk_level = risk_level
    return obj


def make_sf():
    """Return a fresh StabilityFilter instance."""
    return StabilityFilter()


# ── Frame instability counter ─────────────────────────────────────────────────

class TestRecordFrame:
    def test_fps_below_min_cpu_increments_counter(self):
        sf = make_sf()
        sf.record_frame(fps=FPS_MIN_CPU - 1, has_detections=True, is_gpu=False)
        assert sf.unstable_frame_count == 1

    def test_fps_below_min_gpu_increments_counter(self):
        sf = make_sf()
        sf.record_frame(fps=FPS_MIN_GPU - 1, has_detections=True, is_gpu=True)
        assert sf.unstable_frame_count == 1

    def test_no_detections_increments_counter(self):
        sf = make_sf()
        sf.record_frame(fps=30.0, has_detections=False, is_gpu=False)
        assert sf.unstable_frame_count == 1

    def test_healthy_frame_decrements_counter(self):
        sf = make_sf()
        # Drive counter up to 3
        for _ in range(3):
            sf.record_frame(fps=5.0, has_detections=False, is_gpu=False)
        assert sf.unstable_frame_count == 3
        # Healthy frame should decrement
        sf.record_frame(fps=30.0, has_detections=True, is_gpu=False)
        assert sf.unstable_frame_count == 2

    def test_counter_does_not_go_below_zero(self):
        sf = make_sf()
        # Start at 0 and feed healthy frames
        for _ in range(5):
            sf.record_frame(fps=30.0, has_detections=True, is_gpu=False)
        assert sf.unstable_frame_count == 0


# ── is_system_unstable ────────────────────────────────────────────────────────

class TestIsSystemUnstable:
    def test_not_unstable_below_limit(self):
        sf = make_sf()
        for _ in range(UNSTABLE_FRAME_LIMIT - 1):
            sf.record_frame(fps=0.0, has_detections=False)
        assert sf.is_system_unstable() is False

    def test_unstable_at_limit(self):
        sf = make_sf()
        for _ in range(UNSTABLE_FRAME_LIMIT):
            sf.record_frame(fps=0.0, has_detections=False)
        assert sf.is_system_unstable() is True

    def test_stable_after_recovery(self):
        sf = make_sf()
        for _ in range(UNSTABLE_FRAME_LIMIT):
            sf.record_frame(fps=0.0, has_detections=False)
        assert sf.is_system_unstable() is True
        # Recover
        for _ in range(UNSTABLE_FRAME_LIMIT):
            sf.record_frame(fps=30.0, has_detections=True)
        assert sf.is_system_unstable() is False


# ── should_emit_failsafe / record_failsafe_emitted ───────────────────────────

class TestFailsafe:
    def _drive_unstable(self, sf):
        for _ in range(UNSTABLE_FRAME_LIMIT):
            sf.record_frame(fps=0.0, has_detections=False)

    def test_failsafe_emitted_when_unstable_and_cooldown_expired(self):
        sf = make_sf()
        self._drive_unstable(sf)
        assert sf.should_emit_failsafe() is True

    def test_failsafe_suppressed_when_stable(self):
        sf = make_sf()
        # System is stable — should never emit
        sf.record_frame(fps=30.0, has_detections=True)
        assert sf.should_emit_failsafe() is False

    def test_failsafe_suppressed_within_cooldown(self):
        sf = make_sf()
        self._drive_unstable(sf)
        sf.record_failsafe_emitted()   # just emitted — cooldown starts now
        # Without sleeping, cooldown has NOT expired
        assert sf.should_emit_failsafe() is False

    def test_failsafe_re_emitted_after_cooldown(self):
        sf = make_sf()
        self._drive_unstable(sf)
        # Simulate cooldown already expired by backdating the timestamp
        sf._last_failsafe_t = time.time() - FAILSAFE_COOLDOWN - 1.0
        assert sf.should_emit_failsafe() is True


# ── object_passes_gate ────────────────────────────────────────────────────────

class TestObjectPassesGate:
    def test_good_object_passes(self):
        sf = make_sf()
        obj = make_good_obj()
        allowed, reason = sf.object_passes_gate(obj)
        assert allowed is True
        assert reason == "ok"

    def test_unconfirmed_object_blocked(self):
        sf = make_sf()
        obj = make_good_obj()
        obj.confirmed = False
        obj.frames_seen = 2
        allowed, reason = sf.object_passes_gate(obj)
        assert allowed is False
        assert "not_confirmed" in reason

    def test_low_confidence_blocked(self):
        sf = make_sf()
        obj = make_good_obj()
        obj.confidence = CONF_GATE - 0.01
        obj.effective_confidence = CONF_GATE - 0.01   # must reflect updated confidence
        allowed, reason = sf.object_passes_gate(obj)
        assert allowed is False
        assert "low_conf" in reason

    def test_no_depth_blocked(self):
        sf = make_sf()
        obj = make_good_obj()
        obj.smoothed_distance_m = 0.0
        allowed, reason = sf.object_passes_gate(obj)
        assert allowed is False
        assert "no_depth" in reason

    def test_noisy_depth_blocked(self):
        sf = make_sf()
        obj = make_good_obj()
        obj.distance_variance.return_value = DIST_VARIANCE_GATE + 0.01
        allowed, reason = sf.object_passes_gate(obj)
        assert allowed is False
        assert "depth_unstable" in reason

    def test_low_risk_blocked(self):
        sf = make_sf()
        obj = make_good_obj(risk_level="LOW")
        allowed, reason = sf.object_passes_gate(obj)
        assert allowed is False
        assert "low_risk" in reason

    def test_none_risk_blocked(self):
        sf = make_sf()
        obj = make_good_obj(risk_level="NONE")
        allowed, reason = sf.object_passes_gate(obj)
        assert allowed is False

    def test_medium_risk_passes(self):
        sf = make_sf()
        obj = make_good_obj(risk_level="MEDIUM")
        allowed, _ = sf.object_passes_gate(obj)
        assert allowed is True

    def test_gate_never_raises_on_bad_object(self):
        sf = make_sf()
        bad = MagicMock()
        del bad.confirmed  # force AttributeError
        allowed, reason = sf.object_passes_gate(bad)
        assert allowed is False
        assert "exception" in reason


# ── narration_allowed (cooldown) ──────────────────────────────────────────────

class TestNarrationAllowed:
    def test_first_narration_always_allowed(self):
        sf = make_sf()
        allowed, reason = sf.narration_allowed("MEDIUM")
        assert allowed is True

    def test_medium_blocked_within_cooldown(self):
        sf = make_sf()
        sf.narration_allowed("MEDIUM")   # first call — sets timer
        allowed, reason = sf.narration_allowed("MEDIUM")
        assert allowed is False
        assert "cooldown" in reason

    def test_high_always_bypasses_cooldown(self):
        sf = make_sf()
        sf.narration_allowed("MEDIUM")   # start the cooldown
        # HIGH should bypass immediately
        allowed, reason = sf.narration_allowed("HIGH")
        assert allowed is True
        assert "bypass" in reason

    def test_cooldown_expires(self):
        sf = make_sf()
        sf.narration_allowed("MEDIUM")
        # Backdate the timer so cooldown appears expired
        sf._last_narration_t = time.time() - NARRATION_COOLDOWN - 1.0
        allowed, _ = sf.narration_allowed("MEDIUM")
        assert allowed is True


# ── reset ─────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_instability_counter(self):
        sf = make_sf()
        for _ in range(UNSTABLE_FRAME_LIMIT):
            sf.record_frame(fps=0.0, has_detections=False)
        assert sf.is_system_unstable() is True
        sf.reset()
        assert sf.unstable_frame_count == 0
        assert sf.is_system_unstable() is False

    def test_reset_clears_cooldown(self):
        sf = make_sf()
        sf.narration_allowed("MEDIUM")   # start cooldown
        sf.reset()
        # After reset, cooldown should be gone
        allowed, _ = sf.narration_allowed("MEDIUM")
        assert allowed is True


# ── path_clear_allowed ────────────────────────────────────────────────────────

class TestPathClearAllowed:
    def test_first_clear_fires_immediately(self):
        """First transition to clear (from initial state=False) must announce."""
        sf = make_sf()
        assert sf.path_clear_allowed(True) is True

    def test_path_not_clear_never_fires(self):
        sf = make_sf()
        assert sf.path_clear_allowed(False) is False

    def test_stays_clear_suppressed_within_cooldown(self):
        """Path was clear, still clear, cooldown not expired → suppress."""
        sf = make_sf()
        sf.path_clear_allowed(True)   # first announcement
        # immediately call again — cooldown has NOT elapsed
        assert sf.path_clear_allowed(True) is False

    def test_blocked_to_clear_transition_fires(self):
        """After being blocked, going clear again must fire even within cooldown."""
        sf = make_sf()
        sf.path_clear_allowed(True)   # announce initially
        sf.path_clear_allowed(False)  # now blocked
        # Transition blocked→clear — must fire even within PATH_CLEAR_COOLDOWN
        assert sf.path_clear_allowed(True) is True

    def test_cooldown_expiry_refires(self):
        """After PATH_CLEAR_COOLDOWN, a repeated clear path fires again."""
        sf = make_sf()
        sf.path_clear_allowed(True)
        # Backdate the last path-clear timestamp
        sf._last_path_clear_t = time.time() - PATH_CLEAR_COOLDOWN - 1.0
        sf._path_was_clear    = True   # path stayed clear
        assert sf.path_clear_allowed(True) is True

    def test_narration_cooldown_unaffected(self):
        """path_clear_allowed must NOT consume the MEDIUM narration cooldown."""
        sf = make_sf()
        sf.path_clear_allowed(True)   # fire path clear
        # MEDIUM narration cooldown should still be available
        allowed, _ = sf.narration_allowed("MEDIUM")
        assert allowed is True

    def test_reset_clears_path_state(self):
        sf = make_sf()
        sf.path_clear_allowed(True)    # prime the state
        sf.reset()
        # After reset, the next clear should announce (fresh state)
        assert sf.path_clear_allowed(True) is True


# ── Narration deduplication ───────────────────────────────────────────────────

def make_medium_obj(class_name="person", distance_level=2, motion_state="static"):
    """Make a MEDIUM risk object with explicit dedup-relevant attributes."""
    obj = MagicMock()
    obj.confirmed = True
    obj.confidence = 0.85
    obj.effective_confidence = 0.85
    obj.smoothed_distance_m = 1.5
    obj.distance_variance.return_value = 0.01
    obj.risk_level = "MEDIUM"
    obj.risk_score = 0.65
    obj.class_name = class_name
    obj.distance_level = distance_level
    obj.motion_state = motion_state
    return obj


class TestNarrationDedup:
    """
    Narration deduplication: MEDIUM-risk narrations with the same class,
    distance bucket, and motion state are suppressed within DEDUP_CACHE_TTL.

    HIGH risk is NEVER suppressed by dedup.
    """

    def test_narration_key_encodes_class_level_motion(self):
        obj = make_medium_obj(class_name="chair", distance_level=3, motion_state="static")
        key = _narration_key(obj)
        assert "chair" in key
        assert "3" in key
        assert "static" in key

    def test_narration_key_different_objects_differ(self):
        a = make_medium_obj(class_name="person",  distance_level=2, motion_state="static")
        b = make_medium_obj(class_name="bicycle", distance_level=2, motion_state="static")
        assert _narration_key(a) != _narration_key(b)

    def test_narration_key_same_class_different_level_differ(self):
        a = make_medium_obj(class_name="person", distance_level=2, motion_state="static")
        b = make_medium_obj(class_name="person", distance_level=3, motion_state="static")
        assert _narration_key(a) != _narration_key(b)

    def test_first_medium_narration_allowed(self):
        """First time a key is seen it must pass."""
        sf = make_sf()
        obj = make_medium_obj()
        allowed, reason = sf.narration_allowed("MEDIUM", obj=obj)
        assert allowed is True

    def test_second_medium_narration_within_ttl_suppressed(self):
        """Same key within DEDUP_CACHE_TTL (after cooldown) must be suppressed."""
        sf = make_sf()
        obj = make_medium_obj()
        # First call — allowed, records in dedup cache
        sf.narration_allowed("MEDIUM", obj=obj)
        # Force cooldown to expire
        sf._last_narration_t = 0.0
        # Second call — same key, still within TTL
        allowed, reason = sf.narration_allowed("MEDIUM", obj=obj)
        assert allowed is False
        assert "dedup" in reason

    def test_high_risk_always_bypasses_dedup(self):
        """HIGH risk must never be suppressed by deduplication."""
        sf = make_sf()
        obj = make_medium_obj()
        obj.risk_level = "HIGH"
        # Call twice — both must be allowed
        a1, _ = sf.narration_allowed("HIGH", obj=obj)
        a2, _ = sf.narration_allowed("HIGH", obj=obj)
        assert a1 is True
        assert a2 is True

    def test_different_key_allowed_after_first(self):
        """A different class at the same distance allows a new narration."""
        sf = make_sf()
        obj_a = make_medium_obj(class_name="chair", distance_level=2)
        obj_b = make_medium_obj(class_name="person", distance_level=2)
        # First narration (chair)
        sf.narration_allowed("MEDIUM", obj=obj_a)
        # Force cooldown elapsed
        sf._last_narration_t = 0.0
        # Second narration (person) — different key, must pass
        allowed, reason = sf.narration_allowed("MEDIUM", obj=obj_b)
        assert allowed is True

    def test_dedup_cache_cleared_on_reset(self):
        """reset() must clear the dedup cache."""
        sf = make_sf()
        obj = make_medium_obj()
        sf.narration_allowed("MEDIUM", obj=obj)   # populate cache
        sf.reset()
        sf._last_narration_t = 0.0   # also clear cooldown
        allowed, _ = sf.narration_allowed("MEDIUM", obj=obj)
        assert allowed is True

    def test_narration_without_obj_skips_dedup(self):
        """narration_allowed(risk_level) with no obj must behave as before (no dedup)."""
        sf = make_sf()
        a1, _ = sf.narration_allowed("MEDIUM")
        # Force cooldown elapsed
        sf._last_narration_t = 0.0
        a2, _ = sf.narration_allowed("MEDIUM")
        # Without obj, dedup is skipped so second call must be allowed
        assert a1 is True
        assert a2 is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
