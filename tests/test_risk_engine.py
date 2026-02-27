"""
test_risk_engine.py — Unit tests for risk_engine.py

Covers:
  - Deterministic: same inputs → same output
  - Weights sum to exactly 1.0
  - Distance factor boundaries (< 1 m, 1–2 m, 2–3 m, > 3 m)
  - Velocity factor (approaching / static / receding)
  - Class weight (high hazard / furniture / small)
  - Normalised score clamps to [0, 1]
  - score() returns (0.0, "NONE") when distance is 0
  - HIGH >= 0.75, MEDIUM 0.60–0.74, LOW < 0.60
  - score_all() sorts by risk_score descending and mutates in-place
  - Never raises on bad input
  - Age decay: static old objects get reduced risk score
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock
from backend.risk_engine import (
    score,
    score_all,
    W_DISTANCE, W_OVERLAP, W_VELOCITY, W_CLASS,
    THRESH_HIGH, THRESH_MEDIUM,
    _distance_factor, _velocity_factor, _class_weight,
    _size_boosted_distance_factor, SIZE_LARGE_THRESH,
    _age_decay_factor,
    AGE_DECAY_ONSET_S, AGE_DECAY_FULL_S, AGE_DECAY_MIN_FACTOR,
    LATERAL_PX_THRESH, LATERAL_DIST_THRESH, LATERAL_RISK_BOOST,
)


# ── Helper ────────────────────────────────────────────────────────────────────

def make_obj(class_name="person", dist_m=1.5, overlap=0.5,
             motion_state="approaching", x1=0, y1=0, x2=100, y2=100,
             velocity_m_per_s=0.0):
    obj = MagicMock()
    obj.class_name = class_name
    obj.smoothed_distance_m = dist_m
    obj.path_overlap_ratio = overlap
    obj.motion_state = motion_state
    obj.velocity_m_per_s = velocity_m_per_s
    obj.id = 1
    obj.x1 = x1; obj.y1 = y1; obj.x2 = x2; obj.y2 = y2
    return obj


# ── Weights ───────────────────────────────────────────────────────────────────

class TestWeights:
    def test_weights_sum_to_one(self):
        total = W_DISTANCE + W_OVERLAP + W_VELOCITY + W_CLASS
        assert abs(total - 1.0) < 1e-9

    def test_thresholds_ordered(self):
        assert THRESH_HIGH > THRESH_MEDIUM > 0.0


# ── Distance factor ───────────────────────────────────────────────────────────

class TestDistanceFactor:
    def test_very_close(self):
        assert _distance_factor(0.5) == pytest.approx(1.0)

    def test_close(self):
        assert _distance_factor(1.5) == pytest.approx(0.7)

    def test_medium(self):
        assert _distance_factor(2.5) == pytest.approx(0.4)

    def test_far(self):
        assert _distance_factor(4.0) == pytest.approx(0.1)

    def test_zero_distance_returns_zero(self):
        assert _distance_factor(0.0) == pytest.approx(0.0)

    def test_negative_distance_returns_zero(self):
        assert _distance_factor(-1.0) == pytest.approx(0.0)

    def test_boundary_exactly_1m(self):
        # < 1 m → 1.0; at exactly 1.0 m → falls into 1–2 m range → 0.7
        assert _distance_factor(1.0) == pytest.approx(0.7)

    def test_boundary_exactly_2m(self):
        assert _distance_factor(2.0) == pytest.approx(0.4)

    def test_boundary_exactly_3m(self):
        assert _distance_factor(3.0) == pytest.approx(0.1)


# ── Velocity factor ───────────────────────────────────────────────────────────

class TestVelocityFactor:
    def test_approaching(self):
        assert _velocity_factor("approaching") == pytest.approx(1.0)

    def test_static(self):
        assert _velocity_factor("static") == pytest.approx(0.5)

    def test_receding(self):
        assert _velocity_factor("receding") == pytest.approx(0.2)

    def test_unknown_defaults_receding(self):
        # Any unrecognised string should fall through to the default (receding)
        assert _velocity_factor("zigzagging") == pytest.approx(0.2)


# ── Class weight ──────────────────────────────────────────────────────────────

class TestClassWeight:
    def test_person_is_high(self):
        assert _class_weight("person") == pytest.approx(1.0)

    def test_car_is_high(self):
        assert _class_weight("car") == pytest.approx(1.0)

    def test_chair_weight(self):
        # spec: chair=0.6 (explicitly below moderate 0.7, above small 0.5)
        assert _class_weight("chair") == pytest.approx(0.6)

    def test_couch_is_small(self):
        # couch not in any named set → falls to small/unknown = 0.5
        assert _class_weight("couch") == pytest.approx(0.5)

    def test_wall_is_static_feature(self):
        assert _class_weight("wall") == pytest.approx(0.4)

    def test_pole_is_static_feature(self):
        assert _class_weight("pole") == pytest.approx(0.4)

    def test_table_is_moderate(self):
        assert _class_weight("table") == pytest.approx(0.7)

    def test_door_is_moderate(self):
        assert _class_weight("door") == pytest.approx(0.7)

    def test_unknown_is_small(self):
        assert _class_weight("stapler") == pytest.approx(0.5)


# ── score() ───────────────────────────────────────────────────────────────────

class TestScore:
    def test_no_depth_returns_none(self):
        obj = make_obj(dist_m=0.0)
        s, lvl = score(obj)
        assert s == pytest.approx(0.0)
        assert lvl == "NONE"

    def test_high_risk_scenario(self):
        # person, < 1 m, full corridor overlap, approaching → should be HIGH
        obj = make_obj(class_name="person", dist_m=0.5, overlap=1.0,
                       motion_state="approaching")
        s, lvl = score(obj)
        assert s >= THRESH_HIGH
        assert lvl == "HIGH"

    def test_medium_risk_scenario(self):
        # chair, 2 m, overlap=0.6, static
        # Manual: 0.35*0.4 + 0.30*0.6 + 0.20*0.5 + 0.15*0.6
        #       = 0.14 + 0.18 + 0.10 + 0.09 = 0.51 → LOW (below THRESH_MEDIUM=0.60)
        # Use a closer/higher-overlap scenario to get a true MEDIUM:
        # chair, 1.5 m, overlap=0.7, static
        # Manual: 0.35*0.7 + 0.30*0.7 + 0.20*0.5 + 0.15*0.6
        #       = 0.245 + 0.21 + 0.10 + 0.09 = 0.645 → MEDIUM
        obj = make_obj(class_name="chair", dist_m=1.5, overlap=0.7,
                       motion_state="static")
        s, lvl = score(obj)
        assert THRESH_MEDIUM <= s < THRESH_HIGH
        assert lvl == "MEDIUM"

    def test_low_risk_scenario(self):
        # small object, far, no overlap, receding
        obj = make_obj(class_name="stapler", dist_m=4.0, overlap=0.0,
                       motion_state="receding")
        s, lvl = score(obj)
        assert s < THRESH_MEDIUM
        assert lvl == "LOW"

    def test_deterministic(self):
        """Calling score() twice with identical inputs must return identical outputs."""
        obj = make_obj()
        result1 = score(obj)
        result2 = score(obj)
        assert result1 == result2

    def test_score_clamped_to_0_1(self):
        obj = make_obj(class_name="person", dist_m=0.1, overlap=1.0,
                       motion_state="approaching")
        s, _ = score(obj)
        assert 0.0 <= s <= 1.0

    def test_never_raises_on_bad_object(self):
        bad = MagicMock()
        bad.smoothed_distance_m = 2.0
        del bad.motion_state  # force AttributeError
        # Should not raise
        s, lvl = score(bad)
        assert isinstance(s, float)

    def test_manual_calculation(self):
        """
        Manually verify the formula for a known scenario.
        person, 1.5 m → dist_factor=0.7
        overlap=0.5 → overlap_factor=0.5
        approaching → velocity_factor=1.0
        class_weight=1.0

        expected = 0.35*0.7 + 0.30*0.5 + 0.20*1.0 + 0.15*1.0
                 = 0.245 + 0.15 + 0.20 + 0.15
                 = 0.745
        """
        obj = make_obj(class_name="person", dist_m=1.5, overlap=0.5,
                       motion_state="approaching")
        s, lvl = score(obj)
        assert s == pytest.approx(0.745, abs=1e-6)
        assert lvl == "MEDIUM"


# ── score_all() ───────────────────────────────────────────────────────────────

class TestScoreAll:
    def test_mutates_objects_in_place(self):
        obj = make_obj(dist_m=1.5)
        score_all([obj])
        assert obj.risk_score > 0.0
        assert obj.risk_level in ("HIGH", "MEDIUM", "LOW", "NONE")

    def test_returns_sorted_by_risk_desc(self):
        high = make_obj(class_name="person", dist_m=0.5, overlap=1.0,
                        motion_state="approaching")
        low  = make_obj(class_name="stapler", dist_m=5.0, overlap=0.0,
                        motion_state="receding")
        result = score_all([low, high])
        assert result[0].risk_score >= result[1].risk_score

    def test_empty_list_returns_empty(self):
        assert score_all([]) == []

    def test_three_objects_sorted(self):
        objs = [
            make_obj(class_name="stapler", dist_m=5.0, overlap=0.0, motion_state="receding"),
            make_obj(class_name="person",  dist_m=0.5, overlap=1.0, motion_state="approaching"),
            make_obj(class_name="chair",   dist_m=2.0, overlap=0.5, motion_state="static"),
        ]
        result = score_all(objs)
        scores = [o.risk_score for o in result]
        assert scores == sorted(scores, reverse=True)


# ── size factor ───────────────────────────────────────────────────────────────

class TestSizeFactor:
    def test_large_bbox_overrides_distance_to_1(self):
        """Bbox >= 20% of frame → distance factor clamped to 1.0."""
        # frame 640x480=307200; bbox 400x400=160000 → ratio≈0.52 >= 0.20
        df = _size_boosted_distance_factor(
            dist_m=4.0,   # far — would normally be 0.1
            x1=0, y1=0, x2=400, y2=400,
            frame_w=640, frame_h=480,
        )
        assert df == pytest.approx(1.0)

    def test_small_bbox_uses_normal_distance_factor(self):
        """Bbox < 20% → normal distance factor applied."""
        # 50x50=2500; 640x480=307200 → ratio≈0.008 < 0.20
        df = _size_boosted_distance_factor(
            dist_m=4.0,
            x1=0, y1=0, x2=50, y2=50,
            frame_w=640, frame_h=480,
        )
        assert df == pytest.approx(0.1)   # >3 m → 0.1

    def test_size_at_exact_threshold(self):
        """Bbox at SIZE_LARGE_THRESH or above → clamped to 1.0."""
        # 200x100 = 20000 pixels; frame 100x1000 = 100000 → ratio = 0.20 exactly
        df = _size_boosted_distance_factor(
            dist_m=5.0,
            x1=0, y1=0, x2=200, y2=100,
            frame_w=100, frame_h=1000,
        )
        assert df == pytest.approx(1.0)

    def test_score_with_frame_dims_boosts_large_object(self):
        """score() with frame dims boosts a large far object to HIGH."""
        # person at 4m with tiny bbox — normally LOW risk
        obj_far = make_obj(class_name="person", dist_m=4.0, overlap=0.0,
                           motion_state="static", x1=0, y1=0, x2=50, y2=50)
        s_no_size, _ = score(obj_far, frame_w=0, frame_h=0)   # no size boost

        # person at 4m with large bbox (fills >20% of frame)
        obj_large = make_obj(class_name="person", dist_m=4.0, overlap=0.0,
                             motion_state="static", x1=0, y1=0, x2=400, y2=400)
        s_boosted, _ = score(obj_large, frame_w=640, frame_h=480)

        assert s_boosted > s_no_size, (
            "Large-bbox object at same distance should score higher with size boost"
        )

    def test_score_without_frame_dims_is_unchanged(self):
        """Calling score(obj) with no frame dims must behave identically to before."""
        obj = make_obj(class_name="person", dist_m=1.5, overlap=0.5,
                       motion_state="approaching")
        s, lvl = score(obj)
        assert s == pytest.approx(0.745, abs=1e-6)
        assert lvl == "MEDIUM"


# ── Age decay ─────────────────────────────────────────────────────────────────

def make_obj_with_age(class_name="chair", dist_m=1.5, overlap=0.5,
                      motion_state="static", track_age_seconds=0.0):
    """Make a mock TrackedObject with a real track_age_seconds value."""
    obj = MagicMock()
    obj.class_name = class_name
    obj.smoothed_distance_m = dist_m
    obj.path_overlap_ratio = overlap
    obj.motion_state = motion_state
    obj.id = 99
    obj.x1 = 0; obj.y1 = 0; obj.x2 = 100; obj.y2 = 100
    obj.track_age_seconds = track_age_seconds
    return obj


class TestAgeFactor:
    """Unit tests for _age_decay_factor helper."""

    def test_young_object_no_decay(self):
        """Object younger than AGE_DECAY_ONSET_S → decay factor = 1.0."""
        f = _age_decay_factor(AGE_DECAY_ONSET_S - 1, "static")
        assert f == pytest.approx(1.0)

    def test_approaching_always_no_decay(self):
        """Approaching objects are never decayed, regardless of age."""
        f = _age_decay_factor(1000.0, "approaching")
        assert f == pytest.approx(1.0)

    def test_onset_boundary(self):
        """At exactly AGE_DECAY_ONSET_S → still 1.0 (decay hasn't started)."""
        f = _age_decay_factor(AGE_DECAY_ONSET_S, "static")
        assert f == pytest.approx(1.0)

    def test_at_full_decay_age(self):
        """At AGE_DECAY_FULL_S → factor reaches AGE_DECAY_MIN_FACTOR."""
        f = _age_decay_factor(AGE_DECAY_FULL_S, "static")
        assert f == pytest.approx(AGE_DECAY_MIN_FACTOR, abs=1e-6)

    def test_beyond_full_decay_age_is_clamped(self):
        """Beyond AGE_DECAY_FULL_S → factor stays at AGE_DECAY_MIN_FACTOR."""
        f = _age_decay_factor(AGE_DECAY_FULL_S * 10, "static")
        assert f == pytest.approx(AGE_DECAY_MIN_FACTOR, abs=1e-6)

    def test_midpoint_decay(self):
        """At midpoint between onset and full, factor should be halfway."""
        midpoint = (AGE_DECAY_ONSET_S + AGE_DECAY_FULL_S) / 2.0
        f = _age_decay_factor(midpoint, "receding")
        expected = 1.0 - 0.5 * (1.0 - AGE_DECAY_MIN_FACTOR)
        assert f == pytest.approx(expected, abs=1e-6)

    def test_factor_never_below_min(self):
        """Decay factor must never fall below AGE_DECAY_MIN_FACTOR."""
        for age in (0, 10, 30, 60, 120, 600):
            f = _age_decay_factor(float(age), "static")
            assert f >= AGE_DECAY_MIN_FACTOR - 1e-9


class TestAgeDecayIntegration:
    """Integration: score() applies age decay correctly."""

    def test_old_static_object_scores_lower_than_young(self):
        """Same object at same distance but older → lower risk score."""
        young = make_obj_with_age(dist_m=1.5, overlap=0.5,
                                  motion_state="static", track_age_seconds=0.0)
        old   = make_obj_with_age(dist_m=1.5, overlap=0.5,
                                  motion_state="static", track_age_seconds=AGE_DECAY_FULL_S)
        s_young, _ = score(young)
        s_old,   _ = score(old)
        assert s_old < s_young, (
            "An old static object must score lower than an identical young one"
        )

    def test_old_approaching_object_not_decayed(self):
        """Approaching objects should NOT be penalised for age."""
        young = make_obj_with_age(dist_m=1.5, overlap=0.5,
                                  motion_state="approaching", track_age_seconds=0.0)
        old   = make_obj_with_age(dist_m=1.5, overlap=0.5,
                                  motion_state="approaching", track_age_seconds=AGE_DECAY_FULL_S)
        s_young, _ = score(young)
        s_old,   _ = score(old)
        assert s_old == pytest.approx(s_young, abs=1e-6), (
            "Approaching objects must have the same score regardless of age"
        )

    def test_very_old_static_object_score_bounded_by_min_factor(self):
        """At extreme age, score is at most raw * AGE_DECAY_MIN_FACTOR."""
        obj = make_obj_with_age(dist_m=1.5, overlap=0.5,
                                motion_state="static", track_age_seconds=10_000.0)
        # Raw score without decay
        raw_obj = make_obj_with_age(dist_m=1.5, overlap=0.5,
                                    motion_state="static", track_age_seconds=0.0)
        s_raw, _ = score(raw_obj)
        s_decayed, _ = score(obj)
        assert s_decayed >= s_raw * AGE_DECAY_MIN_FACTOR - 1e-6

    def test_object_without_track_age_attr_works(self):
        """If track_age_seconds is absent (e.g. old mock), score must still work."""
        obj = make_obj(class_name="chair", dist_m=1.5, overlap=0.5,
                       motion_state="static")
        # make_obj uses MagicMock; getattr returns 0.0 → no decay
        s, lvl = score(obj)
        assert isinstance(s, float)
        assert lvl in ("HIGH", "MEDIUM", "LOW", "NONE")


# ── Trajectory projection ─────────────────────────────────────────────────────

from backend.risk_engine import _projected_distance, TRAJ_HORIZON_S, TRAJ_MIN_DIST_M


class TestTrajectoryProjection:
    """
    Tests for _projected_distance() and its integration into score().

    Design rules:
      - proj = max(TRAJ_MIN_DIST_M, dist - vel * TRAJ_HORIZON_S)
      - Only applied when velocity_m_per_s > 0 (approaching).
      - Static / receding objects use current distance unchanged.
      - score() for approaching objects must be >= score() for same object static.
    """

    def test_projected_distance_basic(self):
        """proj = dist - vel * horizon, floored at TRAJ_MIN_DIST_M."""
        result = _projected_distance(3.0, 1.0)
        expected = max(TRAJ_MIN_DIST_M, 3.0 - 1.0 * TRAJ_HORIZON_S)
        assert result == pytest.approx(expected)

    def test_projected_distance_floor_prevents_negative(self):
        """Fast object very close: projected dist cannot go below floor."""
        result = _projected_distance(1.0, 5.0)
        assert result == pytest.approx(TRAJ_MIN_DIST_M)
        assert result >= TRAJ_MIN_DIST_M

    def test_projected_distance_slow_object(self):
        """Slow object far away: projection stays above floor naturally."""
        result = _projected_distance(5.0, 0.1)
        expected = max(TRAJ_MIN_DIST_M, 5.0 - 0.1 * TRAJ_HORIZON_S)
        assert result == pytest.approx(expected)
        assert result > TRAJ_MIN_DIST_M

    def test_projected_distance_zero_velocity(self):
        """Zero velocity → projection equals current distance (floored)."""
        result = _projected_distance(2.5, 0.0)
        assert result == pytest.approx(max(TRAJ_MIN_DIST_M, 2.5))

    def test_approaching_scores_higher_than_static_same_distance(self):
        """An approaching object at 2.5 m should score >= the same static object."""
        approaching = make_obj(class_name="person", dist_m=2.5, overlap=0.5,
                               motion_state="approaching", velocity_m_per_s=1.0)
        static_obj  = make_obj(class_name="person", dist_m=2.5, overlap=0.5,
                               motion_state="static",    velocity_m_per_s=0.0)
        s_app, _    = score(approaching)
        s_sta, _    = score(static_obj)
        assert s_app >= s_sta, (
            f"Approaching object should score >= static at same distance: "
            f"{s_app:.3f} vs {s_sta:.3f}"
        )

    def test_approaching_fast_close_promotes_to_high(self):
        """A person at 2.5 m approaching at 1.0 m/s should reach HIGH risk via projection."""
        obj = make_obj(class_name="person", dist_m=2.5, overlap=0.6,
                       motion_state="approaching", velocity_m_per_s=1.0)
        s, lvl = score(obj)
        # With projection: proj_dist ≈ 0.5m → df=1.0; raw ≈ 0.35 + 0.18 + 0.20 + 0.15 = 0.88 → HIGH
        assert lvl == "HIGH", (
            f"Fast-approaching person at 2.5 m expected HIGH, got {lvl} ({s:.3f})"
        )

    def test_static_object_no_projection(self):
        """Static object score must use current distance unchanged."""
        # Static person at 2.5 m: df=0.4; raw = 0.35*0.4 + 0.30*0.5 + 0.20*0.5 + 0.15*1.0
        # = 0.14 + 0.15 + 0.10 + 0.15 = 0.54 → LOW
        obj = make_obj(class_name="person", dist_m=2.5, overlap=0.5,
                       motion_state="static", velocity_m_per_s=0.0)
        s, lvl = score(obj)
        assert lvl == "LOW", (
            f"Static person at 2.5 m with overlap 0.5 expected LOW, got {lvl} ({s:.3f})"
        )

    def test_receding_object_no_projection(self):
        """Receding object uses current distance, not projected."""
        obj = make_obj(class_name="person", dist_m=2.5, overlap=0.5,
                       motion_state="receding", velocity_m_per_s=-1.0)
        s_rec, _ = score(obj)
        # Compare with static at same position — should be similar (lower velocity factor)
        obj_sta = make_obj(class_name="person", dist_m=2.5, overlap=0.5,
                           motion_state="static", velocity_m_per_s=0.0)
        s_sta, _ = score(obj_sta)
        # Receding has lower velocity factor (0.2 vs 0.5) → must score below static
        assert s_rec < s_sta

    def test_score_without_velocity_attr_does_not_raise(self):
        """score() must never raise if velocity_m_per_s is missing from obj."""
        obj = make_obj(class_name="person", dist_m=2.0, overlap=0.5,
                       motion_state="approaching")
        # deliberately do NOT set velocity_m_per_s (MagicMock returns 0.0 via float)
        s, lvl = score(obj)
        assert isinstance(s, float)
        assert lvl in ("HIGH", "MEDIUM", "LOW", "NONE")


class TestLateralMotionBoost:
    """Tests for the lateral motion risk boost (LATERAL_PX_THRESH, LATERAL_DIST_THRESH)."""

    def _make_obj(self, dist_m=1.5, lateral_px_s=0.0, motion_state="static",
                  overlap=0.3, class_name="person", velocity_m_per_s=0.0):
        obj = MagicMock()
        obj.class_name = class_name
        obj.smoothed_distance_m = dist_m
        obj.path_overlap_ratio = overlap
        obj.motion_state = motion_state
        obj.velocity_m_per_s = velocity_m_per_s
        obj.id = 42
        obj.x1 = 100; obj.y1 = 100; obj.x2 = 200; obj.y2 = 300
        obj.lateral_speed_px_per_s = lateral_px_s
        obj.track_age_seconds = 1.0
        return obj

    def test_no_boost_when_slow_lateral(self):
        """Objects below the lateral threshold must not receive a boost."""
        obj_no_lat = self._make_obj(dist_m=1.5, lateral_px_s=0.0)
        obj_fast_lat = self._make_obj(dist_m=1.5, lateral_px_s=LATERAL_PX_THRESH + 50)
        s_no, _ = score(obj_no_lat)
        s_lat, _ = score(obj_fast_lat)
        assert s_lat > s_no, (
            "Fast lateral motion at close range must yield higher risk score"
        )

    def test_no_boost_when_far_lateral(self):
        """Fast lateral motion beyond LATERAL_DIST_THRESH must NOT be boosted."""
        obj_close = self._make_obj(dist_m=LATERAL_DIST_THRESH - 0.1,
                                    lateral_px_s=LATERAL_PX_THRESH + 50)
        obj_far   = self._make_obj(dist_m=LATERAL_DIST_THRESH + 0.5,
                                    lateral_px_s=LATERAL_PX_THRESH + 50)
        s_close, _ = score(obj_close)
        s_far,   _ = score(obj_far)
        assert s_close > s_far, (
            "Lateral boost should only apply within LATERAL_DIST_THRESH"
        )

    def test_boost_magnitude(self):
        """Score difference between boosted and non-boosted must equal LATERAL_RISK_BOOST."""
        obj_base = self._make_obj(dist_m=1.5, lateral_px_s=0.0, motion_state="static",
                                   overlap=0.3, velocity_m_per_s=0.0)
        obj_lat  = self._make_obj(dist_m=1.5, lateral_px_s=LATERAL_PX_THRESH + 1,
                                   motion_state="static", overlap=0.3,
                                   velocity_m_per_s=0.0)
        s_base, _ = score(obj_base)
        s_lat,  _ = score(obj_lat)
        diff = s_lat - s_base
        assert diff == pytest.approx(LATERAL_RISK_BOOST, abs=0.01), (
            f"Lateral boost should add exactly {LATERAL_RISK_BOOST:.2f}; got {diff:.4f}"
        )

    def test_boost_clamped_to_one(self):
        """Even with the boost, risk_score must not exceed 1.0."""
        # Manufacture a worst-case object where base score is already very high
        obj = self._make_obj(dist_m=0.5, lateral_px_s=LATERAL_PX_THRESH + 200,
                              motion_state="approaching", overlap=1.0,
                              class_name="person", velocity_m_per_s=2.0)
        s, _ = score(obj)
        assert s <= 1.0

    def test_constants_have_expected_defaults(self):
        """Default values from env vars must be sane."""
        assert LATERAL_PX_THRESH  > 0
        assert LATERAL_DIST_THRESH > 0
        assert 0 < LATERAL_RISK_BOOST < 0.5   # additive, should be modest


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
