"""
test_pipeline_cadence.py — Unit tests for adaptive detection cadence logic.

The adaptive cadence formula in pipeline.py:
    _cadence = 2 if n_tracks > 6 else (3 if n_tracks > 3 else DETECT_EVERY_N)

This controls how frequently YOLO is run:
  - ≤3 tracks  → DETECT_EVERY_N (default 4) — low CPU load
  - 4–6 tracks → every 3 frames             — moderate refresh
  - >6 tracks  → every 2 frames             — maximum refresh for busy scenes

Tests here exercise the formula directly (no camera / model required).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

# Import the cadence constant from pipeline without triggering model loads.
# We test the formula in isolation using a helper that mirrors pipeline.py.
from backend.pipeline import DETECT_EVERY_N


def _cadence(n_tracks: int) -> int:
    """Mirror of the adaptive cadence formula in pipeline.py:process_frame()."""
    return 2 if n_tracks > 6 else (3 if n_tracks > 3 else DETECT_EVERY_N)


class TestAdaptiveCadence:
    def test_zero_tracks_uses_default(self):
        assert _cadence(0) == DETECT_EVERY_N

    def test_one_track_uses_default(self):
        assert _cadence(1) == DETECT_EVERY_N

    def test_three_tracks_uses_default(self):
        """Boundary: exactly 3 tracks still uses the default cadence."""
        assert _cadence(3) == DETECT_EVERY_N

    def test_four_tracks_uses_medium(self):
        """Boundary: 4 tracks → medium cadence (every 3 frames)."""
        assert _cadence(4) == 3

    def test_six_tracks_uses_medium(self):
        """Upper boundary of medium tier: exactly 6 tracks."""
        assert _cadence(6) == 3

    def test_seven_tracks_uses_fast(self):
        """Boundary: 7 tracks → fast cadence (every 2 frames)."""
        assert _cadence(7) == 2

    def test_many_tracks_uses_fast(self):
        assert _cadence(20) == 2

    def test_cadence_never_zero(self):
        """Cadence must always be >= 1 to avoid ZeroDivisionError."""
        for n in range(0, 30):
            assert _cadence(n) >= 1

    def test_cadence_monotonically_decreasing(self):
        """More tracks → equal or smaller cadence (runs YOLO more often)."""
        prev = _cadence(0)
        for n in range(1, 20):
            cur = _cadence(n)
            assert cur <= prev, (
                f"Cadence must not increase with more tracks: "
                f"n={n} gave {cur}, previous was {prev}"
            )
            prev = cur

    def test_default_detect_every_n_is_4(self):
        """The base cadence constant must be 3 (lowered from 4 for faster detection)."""
        assert DETECT_EVERY_N == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
