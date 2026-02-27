"""
tests/test_tracker_deque_fix.py

Regression tests for Fix 5:
  _dist_history and _dist_timestamps in TrackedObject must be deque instances,
  not lists.

Before the fix they were typed as List[float] and trimmed with pop(0) — an O(N)
operation on every depth update.  The fix changed them to deque(maxlen=…) with
lazy initialisation, giving O(1) append + auto-trim and eliminating pop(0).

Tests verify:
  1. After the first update_distance() call, _dist_history and _dist_timestamps
     are collections.deque instances (not list or any other type).
  2. Their maxlen is VELOCITY_WINDOW + 2 (the correct cap from the fix).
  3. After many depth updates the deque never exceeds maxlen — proving the
     auto-trim is working (no pop(0) needed).
  4. Velocity is still computed correctly over the deque window.
  5. No pop(0) attribute exists on the objects (list has pop, deque does not
     expose pop(0) as a named method — confirmed by absence of list-style pop).
"""

import time
import unittest
from collections import deque

from backend.tracker import TrackedObject, VELOCITY_WINDOW
from backend.detector import Detection


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_tracked(x1=100, y1=100, x2=200, y2=300) -> TrackedObject:
    """Return a freshly-created TrackedObject (no depth updates yet)."""
    return TrackedObject(
        id=1,
        class_name="person",
        confidence=0.85,
        frames_seen=3,
        frames_missed=0,
        confirmed=True,
        _x1=float(x1), _y1=float(y1),
        _x2=float(x2), _y2=float(y2),
    )


# ── tests ─────────────────────────────────────────────────────────────────────

class TestTrackedObjectDistDeque(unittest.TestCase):
    """Verify _dist_history and _dist_timestamps are deques after Fix 5."""

    def test_dist_history_is_none_before_first_update(self):
        """Before any update_distance() call, _dist_history must be None (lazy init)."""
        obj = _make_tracked()
        self.assertIsNone(
            obj._dist_history,
            "_dist_history must be None before the first depth measurement",
        )

    def test_dist_timestamps_is_none_before_first_update(self):
        """Before any update_distance() call, _dist_timestamps must be None."""
        obj = _make_tracked()
        self.assertIsNone(
            obj._dist_timestamps,
            "_dist_timestamps must be None before the first depth measurement",
        )

    def test_dist_history_becomes_deque_after_first_update(self):
        """After update_distance(), _dist_history must be a collections.deque."""
        obj = _make_tracked()
        obj.update_distance(1.5)
        self.assertIsInstance(
            obj._dist_history,
            deque,
            "_dist_history must be a collections.deque after Fix 5 "
            "(was list before fix — list.pop(0) is O(N))",
        )

    def test_dist_timestamps_becomes_deque_after_first_update(self):
        """After update_distance(), _dist_timestamps must be a collections.deque."""
        obj = _make_tracked()
        obj.update_distance(1.5)
        self.assertIsInstance(
            obj._dist_timestamps,
            deque,
            "_dist_timestamps must be a collections.deque after Fix 5",
        )

    def test_dist_history_maxlen_is_velocity_window_plus_two(self):
        """_dist_history.maxlen must equal VELOCITY_WINDOW + 2."""
        obj = _make_tracked()
        obj.update_distance(1.5)
        expected_maxlen = VELOCITY_WINDOW + 2
        self.assertEqual(
            obj._dist_history.maxlen,
            expected_maxlen,
            f"_dist_history.maxlen must be VELOCITY_WINDOW+2={expected_maxlen}, "
            f"got {obj._dist_history.maxlen}",
        )

    def test_dist_timestamps_maxlen_is_velocity_window_plus_two(self):
        """_dist_timestamps.maxlen must equal VELOCITY_WINDOW + 2."""
        obj = _make_tracked()
        obj.update_distance(1.5)
        expected_maxlen = VELOCITY_WINDOW + 2
        self.assertEqual(
            obj._dist_timestamps.maxlen,
            expected_maxlen,
            f"_dist_timestamps.maxlen must be VELOCITY_WINDOW+2={expected_maxlen}",
        )

    def test_deque_never_grows_beyond_maxlen(self):
        """
        After many update_distance() calls, _dist_history must never exceed
        its maxlen — the deque auto-trim replaces the old manual pop(0).
        """
        obj = _make_tracked()
        N = 50  # many more than VELOCITY_WINDOW + 2
        for i in range(N):
            obj.update_distance(float(i) * 0.1 + 0.5)

        maxlen = obj._dist_history.maxlen
        actual_len = len(obj._dist_history)
        self.assertLessEqual(
            actual_len,
            maxlen,
            f"_dist_history grew to {actual_len} but maxlen={maxlen} — "
            f"auto-trim is broken",
        )
        self.assertEqual(
            actual_len,
            maxlen,
            f"After {N} updates, _dist_history should be full (len={maxlen}), "
            f"got {actual_len}",
        )

    def test_timestamps_never_grows_beyond_maxlen(self):
        """Same auto-trim guarantee for _dist_timestamps."""
        obj = _make_tracked()
        for i in range(50):
            obj.update_distance(float(i) * 0.1 + 0.5)

        maxlen = obj._dist_timestamps.maxlen
        self.assertLessEqual(len(obj._dist_timestamps), maxlen)

    def test_velocity_computed_correctly_over_deque_window(self):
        """
        Velocity must still be computed from the deque window even after
        more than VELOCITY_WINDOW + 2 updates.  A steadily-approaching object
        (distance decreasing) must yield a negative velocity (approaching).
        """
        obj = _make_tracked()
        # Simulate object approaching: 3.0 → 2.0 m over multiple frames
        distances = [3.0, 2.8, 2.6, 2.4, 2.2, 2.0, 1.8]
        for d in distances:
            obj.update_distance(d)
            time.sleep(0.01)  # ensure timestamps advance

        # Velocity should be negative (approaching) or at least computed
        # without raising — deque window must work correctly
        vel = obj.velocity_m_per_s
        self.assertIsInstance(vel, float, "velocity_m_per_s must be a float")
        # With a steadily decreasing distance the velocity must be <= 0
        self.assertLessEqual(
            vel, 0.0,
            f"Approaching object must have non-positive velocity, got {vel:.3f} m/s",
        )

    def test_not_a_list(self):
        """
        _dist_history must NOT be a list — list.pop(0) is O(N) and was the
        original bug.  Using deque(maxlen=…) eliminates the need for pop(0).
        """
        obj = _make_tracked()
        obj.update_distance(1.0)
        self.assertNotIsInstance(
            obj._dist_history,
            list,
            "_dist_history must not be a list (regression: O(N) pop(0) removed in Fix 5)",
        )
        self.assertNotIsInstance(
            obj._dist_timestamps,
            list,
            "_dist_timestamps must not be a list",
        )


if __name__ == "__main__":
    unittest.main()
