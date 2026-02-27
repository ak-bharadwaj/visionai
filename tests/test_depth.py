"""
test_depth.py — Unit tests for depth.py

Covers:
  - depth_jump_reject(): rejects large single-frame depth jumps
  - DEPTH_JUMP_REJECT_M constant
  - metres_from_score() calibration curve (sanity checks)
  - get_region_depth() median behaviour
  - is_depth_stable() / get_region_variance()
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from backend.depth import DepthEstimator, DEPTH_JUMP_REJECT_M, DEPTH_VARIANCE_THRESHOLD


# ── depth_jump_reject ─────────────────────────────────────────────────────────

class TestDepthJumpReject:
    """
    depth_jump_reject(prev_dist_m, new_dist_m) should return True when the
    absolute change exceeds DEPTH_JUMP_REJECT_M, and False otherwise.
    """

    def test_no_prior_measurement_always_accepted(self):
        """prev_dist_m=0.0 → no prior reading → never reject."""
        assert DepthEstimator.depth_jump_reject(0.0, 5.0) is False
        assert DepthEstimator.depth_jump_reject(0.0, 0.5) is False

    def test_small_change_accepted(self):
        """Change well below threshold must be accepted."""
        assert DepthEstimator.depth_jump_reject(2.0, 2.3) is False

    def test_change_exactly_at_threshold_accepted(self):
        """Exactly at the threshold is still accepted (strictly greater rejects)."""
        assert DepthEstimator.depth_jump_reject(2.0, 2.0 + DEPTH_JUMP_REJECT_M) is False

    def test_change_just_above_threshold_rejected(self):
        """Just above DEPTH_JUMP_REJECT_M must be rejected."""
        delta = DEPTH_JUMP_REJECT_M + 0.01
        assert DepthEstimator.depth_jump_reject(2.0, 2.0 + delta) is True

    def test_large_approach_jump_rejected(self):
        """A sudden 2 m drop (depth jump toward camera) is rejected."""
        assert DepthEstimator.depth_jump_reject(3.0, 1.0) is True

    def test_large_recede_jump_rejected(self):
        """A sudden 2 m increase (depth jump away from camera) is rejected."""
        assert DepthEstimator.depth_jump_reject(1.5, 3.5) is True

    def test_negative_prev_treated_as_no_prior(self):
        """Negative prev_dist_m (shouldn't happen, but safe) → never reject."""
        assert DepthEstimator.depth_jump_reject(-1.0, 5.0) is False

    def test_depth_jump_reject_m_constant(self):
        """Default DEPTH_JUMP_REJECT_M must be 0.80 m."""
        assert DEPTH_JUMP_REJECT_M == pytest.approx(0.80, abs=1e-6)

    def test_symmetric(self):
        """Rejection is symmetric — direction of jump doesn't matter."""
        prev, new_far = 2.0, 2.0 + DEPTH_JUMP_REJECT_M + 0.05
        new_close = 2.0 - DEPTH_JUMP_REJECT_M - 0.05
        assert DepthEstimator.depth_jump_reject(prev, new_far) is True
        assert DepthEstimator.depth_jump_reject(prev, new_close) is True


# ── metres_from_score ─────────────────────────────────────────────────────────

class TestMetresFromScore:
    def test_zero_score_returns_zero(self):
        assert DepthEstimator.metres_from_score(0.0) == pytest.approx(0.0)

    def test_high_score_is_close(self):
        """Score near 1.0 → very close object."""
        d = DepthEstimator.metres_from_score(0.95)
        assert d < 1.0, f"Score 0.95 should map to < 1 m, got {d}"

    def test_low_score_is_far(self):
        """Score near 0.1 → far object."""
        d = DepthEstimator.metres_from_score(0.1)
        assert d > 4.0, f"Score 0.1 should map to > 4 m, got {d}"

    def test_monotonically_decreasing(self):
        """Higher depth score (closer object) → smaller metres value."""
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        metres = [DepthEstimator.metres_from_score(s) for s in scores]
        for i in range(1, len(metres)):
            assert metres[i] <= metres[i - 1], (
                f"metres_from_score not monotonically decreasing at "
                f"score={scores[i]}: {metres[i]} > {metres[i-1]}"
            )


# ── get_region_depth / variance / is_depth_stable ────────────────────────────

class TestRegionDepth:
    def _make_depth_map(self, h=100, w=100, fill=0.5):
        return np.full((h, w), fill, dtype=np.float32)

    def test_none_depth_map_returns_zero(self):
        de = DepthEstimator()
        assert de.get_region_depth(None, 0, 0, 10, 10) == pytest.approx(0.0)

    def test_uniform_map_returns_fill_value(self):
        de = DepthEstimator()
        dm = self._make_depth_map(fill=0.7)
        result = de.get_region_depth(dm, 10, 10, 50, 50)
        assert result == pytest.approx(0.7, abs=1e-4)

    def test_empty_region_returns_zero(self):
        de = DepthEstimator()
        dm = self._make_depth_map()
        assert de.get_region_depth(dm, 50, 50, 50, 50) == pytest.approx(0.0)

    def test_region_clips_to_bounds(self):
        """Out-of-bounds coordinates must be clipped, not raise."""
        de = DepthEstimator()
        dm = self._make_depth_map(h=50, w=50, fill=0.3)
        result = de.get_region_depth(dm, -10, -10, 200, 200)
        assert result == pytest.approx(0.3, abs=1e-4)

    def test_variance_uniform_is_zero(self):
        de = DepthEstimator()
        dm = self._make_depth_map(fill=0.5)
        var = de.get_region_variance(dm, 0, 0, 50, 50)
        assert var == pytest.approx(0.0, abs=1e-6)

    def test_variance_noisy_is_positive(self):
        de = DepthEstimator()
        dm = np.random.default_rng(42).random((100, 100)).astype(np.float32)
        var = de.get_region_variance(dm, 10, 10, 90, 90)
        assert var > 0.0

    def test_is_depth_stable_uniform(self):
        de = DepthEstimator()
        dm = self._make_depth_map(fill=0.5)
        assert de.is_depth_stable(dm, 0, 0, 50, 50) is True

    def test_is_depth_stable_noisy(self):
        de = DepthEstimator()
        # Highly variable depth map — should be unstable
        dm = np.tile([0.0, 1.0], 5000).reshape(100, 100).astype(np.float32)
        assert de.is_depth_stable(dm, 0, 0, 100, 100) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
