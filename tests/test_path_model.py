"""
test_path_model.py — Unit tests for path_model.py

Covers:
  - corridor_overlap() matches tracker._corridor_overlap() semantics
  - free_space_fraction() returns 0.0 when depth_map is None
  - free_space_fraction() computes correct fraction for synthetic depth maps
  - free_space_fraction() never raises
  - is_path_clear() returns False when HIGH/MEDIUM object blocks corridor
  - is_path_clear() returns False when free_space_fraction < threshold
  - is_path_clear() returns True when no blocking objects + floor clear
  - floor_zone_exists() returns False for None / tiny frames
  - floor_zone_exists() returns True for adequate depth map
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from unittest.mock import MagicMock

from backend.path_model import (
    corridor_overlap,
    free_space_fraction,
    is_path_clear,
    floor_zone_exists,
    CORRIDOR_LEFT_FRAC,
    CORRIDOR_RIGHT_FRAC,
    FLOOR_ZONE_TOP_FRAC,
    FLOOR_ZONE_LEFT_FRAC,
    FLOOR_ZONE_RIGHT_FRAC,
    FREE_SPACE_DEPTH_THRESH,
    FREE_SPACE_CLEAR_THRESHOLD,
)


# ── Helper ────────────────────────────────────────────────────────────────────

def make_obj(risk_level="HIGH", overlap=0.5):
    obj = MagicMock()
    obj.risk_level = risk_level
    obj.path_overlap_ratio = overlap
    return obj


def make_depth_map(h, w, value=0.5):
    """Uniform depth map filled with `value`."""
    return np.full((h, w), value, dtype=np.float32)


# ── corridor_overlap ──────────────────────────────────────────────────────────

class TestCorridorOverlap:
    def test_fully_inside(self):
        # corridor = 30%–70% of 640 → 192–448
        result = corridor_overlap(200, 400, 640)
        assert result == pytest.approx(1.0)

    def test_fully_outside_left(self):
        result = corridor_overlap(0, 100, 640)
        assert result == pytest.approx(0.0)

    def test_fully_outside_right(self):
        result = corridor_overlap(500, 640, 640)
        assert result == pytest.approx(0.0)

    def test_partial_overlap(self):
        # Corridor starts at 192; obj x1=100, x2=250 → overlap = 250-192=58, obj_w=150
        result = corridor_overlap(100, 250, 640)
        assert result == pytest.approx(58 / 150, rel=1e-3)

    def test_zero_frame_width_returns_zero(self):
        assert corridor_overlap(100, 200, 0) == pytest.approx(0.0)

    def test_zero_object_width_returns_zero(self):
        assert corridor_overlap(200, 200, 640) == pytest.approx(0.0)

    def test_matches_tracker_implementation(self):
        """Ensure path_model.corridor_overlap matches tracker._corridor_overlap."""
        from backend.tracker import _corridor_overlap
        for x1, x2, fw in [(0, 640, 640), (100, 300, 640), (50, 150, 640), (400, 600, 640)]:
            pm_result = corridor_overlap(x1, x2, fw)
            tk_result = _corridor_overlap(x1, x2, fw)
            assert pm_result == pytest.approx(tk_result, abs=1e-6), (
                f"Mismatch for x1={x1},x2={x2},fw={fw}: "
                f"path_model={pm_result}, tracker={tk_result}"
            )


# ── free_space_fraction ───────────────────────────────────────────────────────

class TestFreeSpaceFraction:
    def test_none_depth_returns_zero(self):
        result = free_space_fraction(None, 480, 640)
        assert result == pytest.approx(0.0)

    def test_all_clear_floor(self):
        """A depth map entirely below FREE_SPACE_DEPTH_THRESH → fraction = 1.0."""
        dm = make_depth_map(480, 640, value=0.1)  # all far → clear
        result = free_space_fraction(dm, 480, 640)
        assert result == pytest.approx(1.0)

    def test_all_blocked_floor(self):
        """A depth map entirely above FREE_SPACE_DEPTH_THRESH → fraction = 0.0."""
        dm = make_depth_map(480, 640, value=0.9)  # all close → blocked
        result = free_space_fraction(dm, 480, 640)
        assert result == pytest.approx(0.0)

    def test_half_clear_floor(self):
        """Half the floor zone pixels below threshold → fraction ≈ 0.5."""
        dm = make_depth_map(480, 640, value=0.9)   # start all blocked
        # Floor zone: y from y0=int(480*0.65)=312 downward, x from 160 to 480
        y0 = int(480 * FLOOR_ZONE_TOP_FRAC)
        x0 = int(640 * FLOOR_ZONE_LEFT_FRAC)
        x1 = int(640 * FLOOR_ZONE_RIGHT_FRAC)
        zone_h = 480 - y0
        zone_w = x1 - x0
        # Clear the left half of the floor zone
        half = zone_w // 2
        dm[y0:, x0:x0 + half] = 0.1   # clear (far)
        result = free_space_fraction(dm, 480, 640)
        assert 0.45 < result < 0.55, f"Expected ~0.5, got {result}"

    def test_zero_frame_dims_returns_zero(self):
        dm = make_depth_map(480, 640)
        assert free_space_fraction(dm, 0, 640) == pytest.approx(0.0)
        assert free_space_fraction(dm, 480, 0) == pytest.approx(0.0)

    def test_never_raises(self):
        """Should never raise — return 0.0 for any bizarre input."""
        result = free_space_fraction("not-an-array", 480, 640)
        assert result == pytest.approx(0.0)


# ── is_path_clear ─────────────────────────────────────────────────────────────

class TestIsPathClear:
    def _clear_depth(self):
        """Depth map where the floor zone is fully clear."""
        dm = make_depth_map(480, 640, value=0.1)  # all far → clear
        return dm

    def _blocked_depth(self):
        """Depth map where the floor zone is fully blocked."""
        dm = make_depth_map(480, 640, value=0.9)  # all close → blocked
        return dm

    def test_clear_when_no_objects_and_floor_clear(self):
        result = is_path_clear(self._clear_depth(), 480, 640, [])
        assert result is True

    def test_not_clear_when_high_risk_in_corridor(self):
        obj = make_obj(risk_level="HIGH", overlap=0.5)
        result = is_path_clear(self._clear_depth(), 480, 640, [obj])
        assert result is False

    def test_not_clear_when_medium_risk_in_corridor(self):
        obj = make_obj(risk_level="MEDIUM", overlap=0.4)
        result = is_path_clear(self._clear_depth(), 480, 640, [obj])
        assert result is False

    def test_low_risk_does_not_block_path(self):
        """LOW risk objects must NOT trigger path-blocked."""
        obj = make_obj(risk_level="LOW", overlap=1.0)
        result = is_path_clear(self._clear_depth(), 480, 640, [obj])
        assert result is True

    def test_not_clear_when_floor_blocked(self):
        result = is_path_clear(self._blocked_depth(), 480, 640, [])
        assert result is False

    def test_not_clear_when_no_depth_map(self):
        """Without depth map we cannot confirm floor is clear → conservative = False."""
        result = is_path_clear(None, 480, 640, [])
        assert result is False

    def test_overlap_at_exact_threshold(self):
        """Object with overlap exactly at 0.3 threshold should block."""
        obj = make_obj(risk_level="HIGH", overlap=0.31)
        result = is_path_clear(self._clear_depth(), 480, 640, [obj])
        assert result is False

    def test_overlap_just_below_threshold_does_not_block(self):
        """Object with overlap just below 0.3 should not block (path check only)."""
        obj = make_obj(risk_level="HIGH", overlap=0.29)
        result = is_path_clear(self._clear_depth(), 480, 640, [obj])
        assert result is True


# ── floor_zone_exists ─────────────────────────────────────────────────────────

class TestFloorZoneExists:
    def test_none_depth_returns_false(self):
        assert floor_zone_exists(None, 480, 640) is False

    def test_normal_frame_returns_true(self):
        dm = make_depth_map(480, 640)
        assert floor_zone_exists(dm, 480, 640) is True

    def test_tiny_frame_returns_false(self):
        # Frame too small to yield 64 pixels in floor zone
        dm = make_depth_map(10, 10)
        assert floor_zone_exists(dm, 10, 10) is False

    def test_zero_dims_returns_false(self):
        dm = make_depth_map(480, 640)
        assert floor_zone_exists(dm, 0, 640) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
