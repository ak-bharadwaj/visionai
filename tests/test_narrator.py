"""
test_narrator.py — Unit tests for narrator.py

Covers:
  - build() produces only the permitted templates
  - Urgent "STOP." prefix when ETA < ETA_URGENT_THRESH
  - "ahead" template when overlap >= 0.50 (blocking centre)
  - "approaching" template when approaching and overlap < 0.50
  - Static/receding fallback template ("[Class] [dir]. [dist]m.")
  - Distance rounded to 0.1 m
  - Class name is capitalised
  - Never returns empty string (even on bad object)
  - path_clear() returns the exact canonical message
  - select_highest_risk() picks the highest risk_score object
  - select_highest_risk() tie-break: closest distance wins
  - select_highest_risk() returns None on empty list
  - select_highest_risk() excludes LOW risk objects
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock

from backend.narrator import (
    build,
    path_clear,
    select_highest_risk,
    select_top_n,
    build_multi,
    PATH_CLEAR_MESSAGE,
    BLOCKING_OVERLAP_THRESH,
    ETA_URGENT_THRESH,
    MAX_NARRATIONS_PER_FRAME,
)


# ── Helper ────────────────────────────────────────────────────────────────────

def make_obj(class_name="person", dist_m=2.0, direction="12 o'clock",
             overlap=0.0, motion="approaching", motion_state=None,
             risk_score=0.8, risk_level="HIGH", collision_eta_s=0.0,
             confidence=0.90, distance_m=None):
    obj = MagicMock()
    obj.class_name = class_name
    obj.smoothed_distance_m = distance_m if distance_m is not None else dist_m
    obj.direction = direction
    obj.path_overlap_ratio = overlap
    obj.motion_state = motion_state if motion_state is not None else motion
    obj.risk_score = risk_score
    obj.risk_level = risk_level
    obj.id = 1
    obj.collision_eta_s = collision_eta_s
    obj.confidence = confidence
    return obj


# ── build() ───────────────────────────────────────────────────────────────────

class TestBuild:
    def test_blocking_template_used_when_overlap_high(self):
        obj = make_obj(overlap=BLOCKING_OVERLAP_THRESH, motion_state="static")
        text = build(obj)
        assert "ahead" in text.lower()

    def test_blocking_threshold_exact(self):
        obj = make_obj(overlap=BLOCKING_OVERLAP_THRESH)
        text = build(obj)
        assert "ahead" in text.lower()

    def test_approaching_template_below_threshold(self):
        obj = make_obj(overlap=BLOCKING_OVERLAP_THRESH - 0.01,
                       motion_state="approaching", direction="11 o'clock")
        text = build(obj)
        assert "approaching" in text.lower()
        assert "11 o'clock" in text

    def test_static_fallback_template(self):
        obj = make_obj(overlap=0.0, motion_state="static", direction="10 o'clock")
        text = build(obj)
        assert "approaching" not in text.lower()
        # Static fallback: "[Class] [dir]. [dist]m."
        assert "10 o'clock" in text

    def test_receding_uses_fallback_template(self):
        obj = make_obj(overlap=0.0, motion_state="receding")
        text = build(obj)
        assert "approaching" not in text.lower()

    def test_distance_rounded_to_1dp(self):
        obj = make_obj(dist_m=1.756)
        text = build(obj)
        assert "1.8" in text

    def test_distance_rounded_exact(self):
        obj = make_obj(dist_m=2.0)
        text = build(obj)
        assert "2.0" in text

    def test_class_name_capitalised(self):
        obj = make_obj(class_name="person")
        text = build(obj)
        assert text[0].isupper()
        assert "Person" in text

    def test_class_multiword_capitalised(self):
        obj = make_obj(class_name="dining table")
        text = build(obj)
        assert text.startswith("Dining table")

    def test_distance_in_output(self):
        obj = make_obj(dist_m=1.5)
        text = build(obj)
        assert "1.5" in text

    def test_never_returns_empty_string(self):
        bad = MagicMock()
        del bad.path_overlap_ratio  # force AttributeError inside build()
        text = build(bad)
        assert isinstance(text, str) and len(text) > 0

    def test_no_adjectives_or_speculation(self):
        """Ensure no prohibited descriptive words appear."""
        prohibited = ["dangerous", "safe", "likely", "probably", "might",
                      "seems", "appears to be", "looks like", "scary"]
        obj = make_obj(overlap=0.6, motion_state="approaching")
        text = build(obj).lower()
        for word in prohibited:
            assert word not in text, f"Prohibited word '{word}' found in: {text!r}"


# ── path_clear() ──────────────────────────────────────────────────────────────

class TestPathClear:
    def test_returns_exact_canonical_string(self):
        assert path_clear() == PATH_CLEAR_MESSAGE

    def test_canonical_string_value(self):
        assert PATH_CLEAR_MESSAGE == "Path ahead appears clear."


# ── select_highest_risk() ─────────────────────────────────────────────────────

class TestSelectHighestRisk:
    def test_empty_list_returns_none(self):
        assert select_highest_risk([]) is None

    def test_single_high_risk_returned(self):
        obj = make_obj(risk_score=0.9, risk_level="HIGH")
        assert select_highest_risk([obj]) is obj

    def test_low_risk_excluded(self):
        obj = make_obj(risk_score=0.3, risk_level="LOW")
        result = select_highest_risk([obj])
        assert result is None

    def test_only_low_risk_returns_none(self):
        objs = [
            make_obj(risk_score=0.4, risk_level="LOW"),
            make_obj(risk_score=0.2, risk_level="LOW"),
        ]
        assert select_highest_risk(objs) is None

    def test_picks_highest_risk_score(self):
        low  = make_obj(risk_score=0.55, risk_level="MEDIUM", dist_m=2.0)
        high = make_obj(risk_score=0.90, risk_level="HIGH",   dist_m=3.0)
        assert select_highest_risk([low, high]) is high

    def test_tiebreak_closest_distance_wins(self):
        obj_a = make_obj(risk_score=0.80, risk_level="HIGH", dist_m=3.0)
        obj_b = make_obj(risk_score=0.80, risk_level="HIGH", dist_m=1.0)
        result = select_highest_risk([obj_a, obj_b])
        assert result is obj_b, "Closer object should win the tiebreak"

    def test_medium_risk_included(self):
        obj = make_obj(risk_score=0.60, risk_level="MEDIUM")
        assert select_highest_risk([obj]) is obj

    def test_none_risk_excluded(self):
        obj = make_obj(risk_score=0.0, risk_level="NONE")
        assert select_highest_risk([obj]) is None

    def test_mixed_list_returns_best(self):
        objs = [
            make_obj(risk_score=0.3,  risk_level="LOW",    dist_m=1.0),
            make_obj(risk_score=0.6,  risk_level="MEDIUM", dist_m=2.0),
            make_obj(risk_score=0.85, risk_level="HIGH",   dist_m=5.0),
            make_obj(risk_score=0.0,  risk_level="NONE",   dist_m=0.5),
        ]
        result = select_highest_risk(objs)
        assert result.risk_level == "HIGH"
        assert result.risk_score == pytest.approx(0.85)


# ── ETA template ──────────────────────────────────────────────────────────────

class TestEtaTemplate:
    def test_eta_under_threshold_uses_stop_prefix(self):
        obj = make_obj(motion_state="approaching", collision_eta_s=2.0)
        text = build(obj)
        assert text.startswith("STOP.")

    def test_eta_under_threshold_has_seconds(self):
        obj = make_obj(motion_state="approaching", collision_eta_s=2.0)
        text = build(obj)
        assert "seconds" in text.lower()

    def test_eta_zero_does_not_use_stop(self):
        obj = make_obj(motion_state="approaching", collision_eta_s=0.0)
        text = build(obj)
        assert not text.startswith("STOP.")

    def test_eta_above_threshold_uses_normal_template(self):
        obj = make_obj(motion_state="approaching", collision_eta_s=ETA_URGENT_THRESH + 1.0)
        text = build(obj)
        assert not text.startswith("STOP.")
        assert "approaching" in text.lower()

    def test_eta_rounds_up_to_minimum_1(self):
        """ETA of 0.4s rounds to max(1, round(0.4)) = 1 second."""
        obj = make_obj(motion_state="approaching", collision_eta_s=0.4)
        text = build(obj)
        assert "1 seconds" in text.lower()

    def test_eta_template_has_class_name(self):
        obj = make_obj(class_name="car", motion_state="approaching", collision_eta_s=1.5)
        text = build(obj)
        assert "car" in text.lower()

    def test_static_object_eta_not_used(self):
        obj = make_obj(motion_state="static", collision_eta_s=1.0)
        text = build(obj)
        assert not text.startswith("STOP.")


# ── build_multi() ─────────────────────────────────────────────────────────────

class TestBuildMulti:
    def test_single_object_returns_none(self):
        obj = make_obj()
        assert build_multi([obj]) is None

    def test_two_objects_same_direction_returns_none(self):
        o1 = make_obj(direction="12 o'clock")
        o2 = make_obj(direction="12 o'clock")
        assert build_multi([o1, o2]) is None

    def test_two_objects_different_directions_returns_string(self):
        o1 = make_obj(class_name="person", direction="12 o'clock", risk_level="HIGH")
        o2 = make_obj(class_name="chair", direction="3 o'clock", risk_level="MEDIUM")
        result = build_multi([o1, o2])
        assert result is not None
        assert "person" in result.lower()
        assert "chair" in result.lower()

    def test_multi_result_ends_with_period(self):
        o1 = make_obj(direction="12 o'clock")
        o2 = make_obj(direction="9 o'clock")
        result = build_multi([o1, o2])
        assert result is not None
        assert result.endswith(".")

    def test_low_risk_excluded(self):
        o1 = make_obj(direction="12 o'clock", risk_level="LOW")
        o2 = make_obj(direction="9 o'clock", risk_level="HIGH")
        result = build_multi([o1, o2])
        # Only 1 eligible (LOW excluded) -> None
        assert result is None

    def test_multi_capitalised(self):
        o1 = make_obj(class_name="person", direction="12 o'clock")
        o2 = make_obj(class_name="chair", direction="3 o'clock")
        result = build_multi([o1, o2])
        assert result is not None
        assert result[0].isupper()

    def test_empty_list_returns_none(self):
        assert build_multi([]) is None

    def test_max_objects_respected(self):
        objects = [
            make_obj(class_name="person", direction="12 o'clock", risk_level="HIGH"),
            make_obj(class_name="chair",  direction="3 o'clock",  risk_level="MEDIUM"),
            make_obj(class_name="car",    direction="9 o'clock",  risk_level="HIGH"),
        ]
        result = build_multi(objects, max_objects=2)
        assert result is not None
        # Only 2 objects max — "car" should not appear
        assert result.lower().count(",") <= 1


# ── Confidence hedging ────────────────────────────────────────────────────────

class TestConfidenceHedging:
    """build() prefixes 'Possible' for obj.confidence < CONF_HEDGE_THRESH (0.75).
    STOP alerts (ETA < 3s) must NEVER be hedged — safety overrides uncertainty.
    """

    def test_high_confidence_no_hedge_static(self):
        obj = make_obj(confidence=0.90, motion="static", overlap=0.0)
        text = build(obj)
        assert not text.startswith("Possible")

    def test_high_confidence_no_hedge_blocking(self):
        obj = make_obj(confidence=0.90, motion="static", overlap=0.8)
        text = build(obj)
        assert not text.startswith("Possible")

    def test_high_confidence_no_hedge_approaching(self):
        obj = make_obj(confidence=0.90, motion="approaching", overlap=0.0)
        text = build(obj)
        assert not text.startswith("Possible")

    def test_low_confidence_hedge_static(self):
        obj = make_obj(confidence=0.62, motion="static", overlap=0.0)
        text = build(obj)
        assert text.lower().startswith("possible")

    def test_low_confidence_hedge_blocking(self):
        obj = make_obj(confidence=0.62, motion="static", overlap=0.8)
        text = build(obj)
        assert text.lower().startswith("possible")

    def test_low_confidence_hedge_approaching(self):
        obj = make_obj(confidence=0.62, motion="approaching", overlap=0.0)
        text = build(obj)
        assert text.lower().startswith("possible")

    def test_stop_alert_never_hedged(self):
        """Imminent collision (ETA < 3s) must use STOP prefix regardless of confidence."""
        from unittest.mock import PropertyMock
        obj = make_obj(confidence=0.61, motion="approaching", overlap=0.0)
        type(obj).collision_eta_s = PropertyMock(return_value=1.5)
        text = build(obj)
        assert text.startswith("STOP.")
        assert not text.lower().startswith("possible")

    def test_boundary_confidence_not_hedged(self):
        """Exactly 0.75 should NOT be hedged (hedge is strictly less than 0.75)."""
        obj = make_obj(confidence=0.75, motion="static", overlap=0.0)
        text = build(obj)
        assert not text.lower().startswith("possible")

    def test_hedge_still_contains_class_and_distance(self):
        obj = make_obj(confidence=0.62, motion="static", overlap=0.0,
                       class_name="chair", distance_m=2.0)
        text = build(obj)
        assert "chair" in text.lower()
        assert "2.0" in text


class TestSelectTopN:
    """Tests for select_top_n() and the MAX_NARRATIONS_PER_FRAME cap."""

    def _obj(self, risk_level="HIGH", risk_score=0.8, dist_m=2.0, eta=0.0):
        obj = make_obj(risk_level=risk_level, risk_score=risk_score,
                       dist_m=dist_m, collision_eta_s=eta)
        return obj

    def test_returns_empty_for_no_candidates(self):
        assert select_top_n([]) == []

    def test_excludes_low_risk_objects(self):
        objs = [
            self._obj(risk_level="LOW", risk_score=0.3),
            self._obj(risk_level="MEDIUM", risk_score=0.65),
        ]
        result = select_top_n(objs)
        assert all(o.risk_level in ("HIGH", "MEDIUM") for o in result)
        assert len(result) == 1

    def test_cap_enforced(self):
        """Result must never exceed n items."""
        objs = [self._obj(risk_score=0.8 - i * 0.05) for i in range(10)]
        result = select_top_n(objs, n=2)
        assert len(result) <= 2

    def test_max_narrations_per_frame_constant_default(self):
        """Default MAX_NARRATIONS_PER_FRAME must be a positive integer."""
        assert isinstance(MAX_NARRATIONS_PER_FRAME, int)
        assert MAX_NARRATIONS_PER_FRAME >= 1

    def test_imminent_collision_first(self):
        """Objects with ETA < ETA_URGENT_THRESH must rank before high-score non-imminent ones."""
        non_imminent = self._obj(risk_level="HIGH", risk_score=0.95, eta=0.0)
        imminent = self._obj(risk_level="MEDIUM", risk_score=0.65,
                              eta=ETA_URGENT_THRESH - 0.5)
        result = select_top_n([non_imminent, imminent], n=2)
        assert result[0] is imminent, (
            "Imminent collision must rank first regardless of risk_score"
        )

    def test_high_before_medium_when_no_imminent(self):
        """Without imminent collisions, HIGH objects precede MEDIUM ones."""
        medium = self._obj(risk_level="MEDIUM", risk_score=0.70)
        high   = self._obj(risk_level="HIGH",   risk_score=0.80)
        result = select_top_n([medium, high], n=2)
        assert result[0] is high

    def test_consistent_with_select_highest_risk(self):
        """select_top_n(n=1)[0] must match select_highest_risk() for simple cases."""
        objs = [
            self._obj(risk_level="HIGH",   risk_score=0.90, dist_m=1.5),
            self._obj(risk_level="MEDIUM", risk_score=0.65, dist_m=2.0),
        ]
        top_n_best = select_top_n(objs, n=1)
        best = select_highest_risk(objs)
        assert top_n_best[0] is best

    def test_does_not_raise_on_bad_input(self):
        """select_top_n must never raise; returns [] on unexpected input."""
        result = select_top_n(None)  # type: ignore[arg-type]
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
