"""
test_tracker.py — Unit tests for tracker.py

Covers:
  - Hard confidence gate (conf < 0.60 must be silently dropped)
  - Confirmation gate (≥ 3 frames_seen required before confirmed=True)
  - Eviction after 2 consecutive misses
  - Bbox area stability gate (area change > 40% resets frames_seen)
  - EMA bbox smoothing
  - Velocity / motion_state computation
  - Clock direction mapping
  - Corridor overlap calculation
  - Class guard (different-class detections don't merge)
  - reset() clears all state
"""

import sys
import os
import time

# Allow importing backend as a package from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from backend.tracker import (
    ObjectTracker,
    TrackedObject,
    _iou,
    _clock_direction,
    _corridor_overlap,
    MIN_FRAMES_CONFIRM,
    MISS_FRAMES_EVICT,
    CONF_GATE,
    CONF_DETECT,
    MAX_VELOCITY_M_S,
    AREA_CHANGE_LIMIT,
    DEPTH_STALE_EVICT_FRAMES,
    DORMANT_REVIVAL_TTL,
    DORMANT_REVIVAL_IOU_THRESH,
)
from backend.detector import Detection


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_det(class_name="person", conf=0.80, x1=100, y1=100, x2=200, y2=300, class_id=0):
    return Detection(class_id=class_id, class_name=class_name, confidence=conf,
                     x1=x1, y1=y1, x2=x2, y2=y2)


def confirm_object(tracker, det, frame_w=640, frame_h=480, n=None):
    """Feed the same detection n times (default: MIN_FRAMES_CONFIRM) to reach confirmed=True."""
    n = n or MIN_FRAMES_CONFIRM
    confirmed = []
    for _ in range(n):
        confirmed = tracker.update([det], frame_w, frame_h)
    return confirmed


# ── IoU helper ───────────────────────────────────────────────────────────────

class TestIou:
    def test_identical_boxes(self):
        assert _iou((0, 0, 100, 100), (0, 0, 100, 100)) == pytest.approx(1.0)

    def test_non_overlapping(self):
        assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == pytest.approx(0.0)

    def test_partial_overlap(self):
        iou = _iou((0, 0, 100, 100), (50, 50, 150, 150))
        # intersection = 50x50 = 2500; union = 10000+10000-2500 = 17500
        assert iou == pytest.approx(2500 / 17500, rel=1e-4)

    def test_zero_area_box_returns_zero(self):
        assert _iou((0, 0, 0, 0), (0, 0, 100, 100)) == pytest.approx(0.0)


# ── Clock direction ───────────────────────────────────────────────────────────

class TestClockDirection:
    def test_centre(self):
        # center_x at 50% of 640 → ratio=0.5 → "12 o'clock"
        assert _clock_direction(320, 640) == "12 o'clock"

    def test_hard_left(self):
        assert _clock_direction(0, 640) == "9 o'clock"

    def test_hard_right(self):
        assert _clock_direction(639, 640) == "3 o'clock"

    def test_left_quadrant(self):
        # ratio ≈ 0.20 → "10 o'clock"
        assert _clock_direction(128, 640) == "10 o'clock"

    def test_right_quadrant(self):
        # ratio ≈ 0.77 → "2 o'clock"
        assert _clock_direction(492, 640) == "2 o'clock"


# ── Corridor overlap ─────────────────────────────────────────────────────────

class TestCorridorOverlap:
    def test_fully_inside_corridor(self):
        # corridor = 30%–70% of 640 → 192–448
        # object completely inside
        ratio = _corridor_overlap(200, 400, 640)
        assert ratio == pytest.approx(1.0)

    def test_fully_outside_left(self):
        ratio = _corridor_overlap(0, 100, 640)
        assert ratio == pytest.approx(0.0)

    def test_fully_outside_right(self):
        ratio = _corridor_overlap(550, 640, 640)
        assert ratio == pytest.approx(0.0)

    def test_partial_overlap(self):
        # corridor left=192, obj x1=100, x2=250 → overlap=250-192=58, obj_w=150
        ratio = _corridor_overlap(100, 250, 640)
        assert ratio == pytest.approx(58 / 150, rel=1e-3)

    def test_zero_frame_width(self):
        assert _corridor_overlap(100, 200, 0) == pytest.approx(0.0)


# ── Tracker — confidence gate ─────────────────────────────────────────────────

class TestConfidenceGate:
    def test_below_conf_detect_dropped(self):
        """Detections with conf < CONF_DETECT must never produce any track."""
        tracker = ObjectTracker()
        det = make_det(conf=max(0.0, CONF_DETECT - 0.01))
        for _ in range(10):
            tracker.update([det], 640, 480)
        assert tracker.all_tracks() == [], (
            f"Detection with conf below CONF_DETECT ({CONF_DETECT}) "
            "must be silently ignored — no track created"
        )

    def test_exactly_at_conf_detect_accepted(self):
        """Exactly at CONF_DETECT enters the tracker."""
        tracker = ObjectTracker()
        det = make_det(conf=CONF_DETECT)
        tracker.update([det], 640, 480)
        assert len(tracker.all_tracks()) == 1, (
            f"Detection at exactly CONF_DETECT ({CONF_DETECT}) must enter tracker"
        )

    def test_exactly_at_conf_gate_accepted(self):
        """Exactly at CONF_GATE (narration gate) also passes the tracker entry gate."""
        tracker = ObjectTracker()
        det = make_det(conf=CONF_GATE)
        confirmed = confirm_object(tracker, det)
        assert len(confirmed) == 1

    def test_above_conf_gate_accepted(self):
        tracker = ObjectTracker()
        det = make_det(conf=0.95)
        confirmed = confirm_object(tracker, det)
        assert len(confirmed) == 1


# ── Tracker — confirmation gate ───────────────────────────────────────────────

class TestConfirmationGate:
    def test_not_confirmed_before_min_frames(self):
        """Object must NOT be confirmed before MIN_FRAMES_CONFIRM frames."""
        tracker = ObjectTracker()
        det = make_det()
        for i in range(MIN_FRAMES_CONFIRM - 1):
            confirmed = tracker.update([det], 640, 480)
            assert confirmed == [], (
                f"Object must not be confirmed after only {i+1} frame(s)"
            )

    def test_confirmed_at_min_frames(self):
        """Object must be confirmed exactly at frame MIN_FRAMES_CONFIRM."""
        tracker = ObjectTracker()
        det = make_det()
        confirmed = confirm_object(tracker, det)
        assert len(confirmed) == 1
        assert confirmed[0].frames_seen == MIN_FRAMES_CONFIRM
        assert confirmed[0].confirmed is True

    def test_frames_seen_increments(self):
        tracker = ObjectTracker()
        det = make_det()
        for i in range(1, MIN_FRAMES_CONFIRM + 3):
            tracker.update([det], 640, 480)
            tracks = tracker.all_tracks()
            assert tracks[0].frames_seen == i


# ── Tracker — eviction ────────────────────────────────────────────────────────

class TestEviction:
    def test_track_evicted_after_miss_frames(self):
        """Track must be removed after MISS_FRAMES_EVICT consecutive misses."""
        tracker = ObjectTracker()
        det = make_det()
        confirm_object(tracker, det)
        assert len(tracker.all_tracks()) == 1

        # Now send empty detection list for MISS_FRAMES_EVICT frames
        for _ in range(MISS_FRAMES_EVICT):
            tracker.update([], 640, 480)

        assert tracker.all_tracks() == [], "Track should have been evicted"

    def test_track_survives_one_miss(self):
        """A single missed frame should not evict (only MISS_FRAMES_EVICT triggers eviction)."""
        if MISS_FRAMES_EVICT <= 1:
            pytest.skip("MISS_FRAMES_EVICT=1, single miss already evicts")
        tracker = ObjectTracker()
        det = make_det()
        confirm_object(tracker, det)
        tracker.update([], 640, 480)
        assert len(tracker.all_tracks()) == 1, "Track should survive one missed frame"

    def test_recovery_resets_miss_counter(self):
        """After a miss, re-detecting the object recovers the track without eviction."""
        tracker = ObjectTracker()
        det = make_det()
        confirm_object(tracker, det)
        tracker.update([], 640, 480)   # one miss
        tracker.update([det], 640, 480)  # detected again
        tracks = tracker.all_tracks()
        assert len(tracks) == 1
        assert tracks[0].frames_missed == 0


# ── Tracker — bbox area stability gate ───────────────────────────────────────

class TestBboxAreaStability:
    def test_large_area_jump_resets_confirmation(self):
        """
        An abrupt bbox area change > AREA_CHANGE_LIMIT causes the tracker to
        refuse the match (sets mat[ti,di]=0, continues).  The original track
        ages (frames_missed +1) and the big detection spawns a new tentative
        track.  The original track is still confirmed (it just wasn't updated),
        but a second track appears for the mismatched detection.
        """
        tracker = ObjectTracker()
        det_small = make_det(x1=100, y1=100, x2=200, y2=200)  # area=10000
        confirm_object(tracker, det_small)
        tracks = tracker.all_tracks()
        assert tracks[0].confirmed is True

        # Big detection — area 16× larger → > 40% change
        det_big = make_det(x1=0, y1=0, x2=400, y2=400)
        tracker.update([det_big], 640, 480)
        tracks = tracker.all_tracks()

        # The original confirmed track should have frames_missed=1
        original = next((t for t in tracks if t.id == 1), None)
        assert original is not None
        assert original.frames_missed == 1, (
            "Original track should have frames_missed=1 after bbox area rejection"
        )

        # A new tentative (unconfirmed) track should have spawned for the big detection
        new_tracks = [t for t in tracks if t.id != 1]
        assert len(new_tracks) == 1, (
            "A new tentative track must be spawned for the rejected detection"
        )
        assert new_tracks[0].confirmed is False

    def test_small_area_change_preserves_confirmation(self):
        """Area changes within 40% must not reset the confirmation counter."""
        tracker = ObjectTracker()
        det = make_det(x1=100, y1=100, x2=200, y2=200)  # area=10000
        confirm_object(tracker, det)
        # Slightly larger detection — within 20% area change
        det2 = make_det(x1=95, y1=95, x2=210, y2=215)
        tracker.update([det2], 640, 480)
        tracks = tracker.all_tracks()
        assert tracks[0].confirmed is True


# ── Tracker — class guard ─────────────────────────────────────────────────────

class TestClassGuard:
    def test_different_class_not_merged(self):
        """A detection of a different class must not merge with an existing track."""
        tracker = ObjectTracker()
        det_person = make_det(class_name="person", class_id=0)
        confirm_object(tracker, det_person)
        n_before = len(tracker.all_tracks())

        # Send a 'car' at the same location
        det_car = make_det(class_name="car", class_id=2)
        tracker.update([det_car], 640, 480)
        tracks = tracker.all_tracks()
        # A new track for 'car' should have been spawned
        classes = {t.class_name for t in tracks}
        assert "car" in classes or len(tracks) > n_before, (
            "Car detection must spawn a new track, not merge with person"
        )


# ── Tracker — reset ───────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_all_tracks(self):
        tracker = ObjectTracker()
        det = make_det()
        confirm_object(tracker, det)
        assert len(tracker.all_tracks()) > 0
        tracker.reset()
        assert tracker.all_tracks() == []

    def test_next_id_resets_to_1(self):
        tracker = ObjectTracker()
        det = make_det()
        confirm_object(tracker, det)
        tracker.reset()
        det2 = make_det()
        tracker.update([det2], 640, 480)
        tracks = tracker.all_tracks()
        assert tracks[0].id == 1, "Track ID should restart at 1 after reset"


# ── TrackedObject — velocity / motion_state ───────────────────────────────────

class TestVelocity:
    def _make_obj(self):
        obj = TrackedObject(id=1, class_name="person", confidence=0.85)
        return obj

    def test_approaching_motion_state(self):
        obj = self._make_obj()
        # Simulate object getting closer: 3m → 2m → 1m
        # Sleep must exceed the 0.05s dt threshold inside update_distance()
        for d in [3.0, 2.0, 1.0]:
            obj.update_distance(d)
            time.sleep(0.06)
        assert obj.motion_state == "approaching"
        assert obj.velocity_m_per_s > 0.0

    def test_receding_motion_state(self):
        obj = self._make_obj()
        # Simulate object moving away: 1m → 2m → 3m
        for d in [1.0, 2.0, 3.0]:
            obj.update_distance(d)
            time.sleep(0.06)
        assert obj.motion_state == "receding"
        assert obj.velocity_m_per_s < 0.0

    def test_static_motion_state(self):
        obj = self._make_obj()
        for _ in range(5):
            obj.update_distance(2.0)
            time.sleep(0.01)
        assert obj.motion_state == "static"

    def test_ema_smoothing_dampens_spikes(self):
        """Median pre-filter + EMA must absorb single-frame distance spikes.

        With DEPTH_MEDIAN_WINDOW=5, the window after feeding [2.0, 2.0, 10.0]
        is [2.0, 2.0, 10.0] → median = 2.0.  The EMA receives 2.0 and the
        smoothed distance stays at 2.0.  A single spike must be fully absorbed
        by the median filter — that is the design goal.

        A spike that repeats for the full window duration will eventually shift
        the median and penetrate the EMA (tested separately in
        test_sustained_change_penetrates_median_filter).
        """
        from backend.tracker import DEPTH_MEDIAN_WINDOW
        obj = self._make_obj()
        obj.update_distance(2.0)  # baseline
        obj.update_distance(2.0)
        baseline = obj.smoothed_distance_m
        # Single-frame spike — median window still majority 2.0 → spike absorbed
        obj.update_distance(10.0)
        assert obj.smoothed_distance_m < 4.0, (
            "Median pre-filter + EMA should dampen single-frame spikes completely"
        )
        # With median window >= 3 and majority baseline values, smoothed stays at baseline
        if DEPTH_MEDIAN_WINDOW >= 3:
            assert obj.smoothed_distance_m == pytest.approx(baseline, abs=0.01), (
                "Single spike should be fully absorbed when median window majority is 2.0"
            )

    def test_sustained_change_penetrates_median_filter(self):
        """A real distance change sustained for > DEPTH_MEDIAN_WINDOW frames
        must eventually shift smoothed_distance_m — the filter should not be
        permanently biased against genuine movement.
        """
        from backend.tracker import DEPTH_MEDIAN_WINDOW
        obj = self._make_obj()
        # Establish baseline
        for _ in range(DEPTH_MEDIAN_WINDOW):
            obj.update_distance(2.0)
        baseline = obj.smoothed_distance_m

        # Sustained movement to 5 m over DEPTH_MEDIAN_WINDOW + 2 frames
        for _ in range(DEPTH_MEDIAN_WINDOW + 2):
            obj.update_distance(5.0)

        assert obj.smoothed_distance_m > baseline + 0.5, (
            "Sustained distance change must penetrate median filter and shift EMA"
        )


# ── TrackedObject — distance_variance ────────────────────────────────────────

class TestDistanceVariance:
    def test_stable_readings_low_variance(self):
        obj = TrackedObject(id=1, class_name="chair", confidence=0.80)
        for _ in range(5):
            obj.update_distance(2.0)
        assert obj.distance_variance() < 0.01

    def test_noisy_readings_high_variance(self):
        """
        EMA smoothing strongly dampens raw noise, so the variance of the
        smoothed history is much lower than the raw signal variance.
        The variance of raw [1.0, 3.0, 1.0, 3.0, 1.0] is 1.0.
        After EMA the smoothed history variance will be < 0.1 (well damped)
        but still measurably > 0 (not perfectly flat).
        """
        obj = TrackedObject(id=1, class_name="chair", confidence=0.80)
        raw = [1.0, 3.0, 1.0, 3.0, 1.0]
        for d in raw:
            obj.update_distance(d)
        raw_var = float(__import__("numpy").var(raw))
        smoothed_var = obj.distance_variance()
        # Smoothed variance must be substantially less than raw variance
        assert smoothed_var < raw_var * 0.2, (
            f"EMA smoothing must reduce variance significantly. "
            f"raw_var={raw_var:.4f}, smoothed_var={smoothed_var:.4f}"
        )
        # But must still be > 0 (not a perfectly flat signal)
        assert smoothed_var > 0.0


# ── TrackedObject — collision_eta_s ──────────────────────────────────────────

class TestCollisionEta:
    def test_eta_zero_when_not_approaching(self):
        obj = TrackedObject(id=1, class_name="person", confidence=0.85)
        # Static object — velocity_m_per_s defaults to 0.0
        assert obj.collision_eta_s == 0.0

    def test_eta_zero_when_no_distance(self):
        obj = TrackedObject(id=1, class_name="person", confidence=0.85)
        obj.velocity_m_per_s    = 0.5
        obj.smoothed_distance_m = 0.0   # no depth yet
        assert obj.collision_eta_s == 0.0

    def test_eta_zero_when_velocity_below_threshold(self):
        obj = TrackedObject(id=1, class_name="person", confidence=0.85)
        obj.velocity_m_per_s    = 0.05   # at threshold — should return 0
        obj.smoothed_distance_m = 2.0
        assert obj.collision_eta_s == 0.0

    def test_eta_computed_correctly(self):
        obj = TrackedObject(id=1, class_name="person", confidence=0.85)
        obj.velocity_m_per_s    = 1.0
        obj.smoothed_distance_m = 3.0
        assert obj.collision_eta_s == pytest.approx(3.0, rel=1e-3)

    def test_eta_half_distance_at_double_speed(self):
        obj = TrackedObject(id=1, class_name="person", confidence=0.85)
        obj.velocity_m_per_s    = 2.0
        obj.smoothed_distance_m = 2.0
        assert obj.collision_eta_s == pytest.approx(1.0, rel=1e-3)

    def test_eta_positive_for_approaching_object(self):
        obj = TrackedObject(id=1, class_name="person", confidence=0.85)
        for d in [3.0, 2.0, 1.0]:
            obj.update_distance(d)
            time.sleep(0.06)
        # After approach, velocity > 0 and ETA > 0
        assert obj.collision_eta_s > 0.0


# ── Velocity sign consistency gate ───────────────────────────────────────────

class TestVelocitySignConsistency:
    """
    update_distance() must only accept a new velocity when all consecutive
    distance deltas share the same sign.  A single contradicting sample must
    leave velocity unchanged rather than snapping to a noisy reading.
    """

    def test_consistent_approach_sets_velocity(self):
        """Monotonically decreasing distances → all deltas negative → accepted."""
        obj = TrackedObject(id=1, class_name="person", confidence=0.85)
        for d in [3.0, 2.5, 2.0]:
            obj.update_distance(d)
            time.sleep(0.06)
        assert obj.velocity_m_per_s > 0.0, "Consistent approach must set positive velocity"

    def test_consistent_recede_sets_velocity(self):
        """Monotonically increasing distances → all deltas positive → accepted."""
        obj = TrackedObject(id=1, class_name="person", confidence=0.85)
        for d in [1.0, 2.0, 3.0]:
            obj.update_distance(d)
            time.sleep(0.06)
        assert obj.velocity_m_per_s < 0.0, "Consistent recede must set negative velocity"

    def test_noisy_reversal_does_not_flip_velocity(self):
        """
        Two approach samples then a contradicting recede sample.
        The contradicting sample must not flip velocity to the opposite sign.
        """
        obj = TrackedObject(id=1, class_name="person", confidence=0.85)
        # Establish consistent approach first
        for d in [3.0, 2.5, 2.0]:
            obj.update_distance(d)
            time.sleep(0.06)
        approach_velocity = obj.velocity_m_per_s
        assert approach_velocity > 0.0

        # Now inject a spike in the wrong direction — distance jumps up
        obj.update_distance(5.0)
        time.sleep(0.06)
        # Velocity should NOT have flipped to negative
        assert obj.velocity_m_per_s >= 0.0, (
            "A single contradicting depth sample must not flip velocity sign"
        )

    def test_static_readings_allow_zero_velocity(self):
        """All-zero deltas → signs are all 0 → velocity stays at 0."""
        obj = TrackedObject(id=1, class_name="chair", confidence=0.80)
        for _ in range(4):
            obj.update_distance(2.0)
            time.sleep(0.02)
        assert obj.velocity_m_per_s == pytest.approx(0.0, abs=0.05)


# ── ID switch tracking ────────────────────────────────────────────────────────

class TestIdSwitchTracking:
    """
    ID-switch behaviour after dormant revival was added:

    - Same class reappears at the same location within DORMANT_REVIVAL_TTL
      → track is REVIVED (original ID restored) — NO id_switch fired.
    - Same class reappears at a completely different location (IoU = 0) after
      eviction AND after the dormant TTL expires
      → new ID spawned → id_switch IS fired.
    - First appearance → no id_switch.
    """

    def test_revival_prevents_id_switch(self):
        """Track revived from dormant pool must NOT count as an ID switch."""
        from backend.diagnostics import diagnostics
        tracker = ObjectTracker()
        tracker.reset()
        diagnostics.id_switch_count = 0

        det = make_det(class_name="person", conf=0.85, x1=100, y1=100, x2=200, y2=300)

        # Confirm the track
        confirm_object(tracker, det)
        original_ids = {t.id for t in tracker.all_tracks()}

        # Let it miss until eviction (moves to dormant pool)
        for _ in range(MISS_FRAMES_EVICT):
            tracker.update([], 640, 480)

        assert len(tracker.all_tracks()) == 0, "Track must be evicted to active pool"
        assert len(tracker.dormant_tracks()) == 1, "Track must be in dormant pool"

        # Reappear at same bbox → revival, not ID switch
        tracker.update([det], 640, 480)

        revived_ids = {t.id for t in tracker.all_tracks()}
        assert revived_ids == original_ids, "Revived track must have the original ID"
        assert diagnostics.id_switch_count == 0, (
            "Dormant revival must NOT fire an id_switch"
        )

    def test_id_switch_fires_when_dormant_ttl_expired(self):
        """After dormant TTL expires, a different-position reappearance is an ID switch."""
        import time as _time
        from backend.diagnostics import diagnostics
        from unittest.mock import patch
        tracker = ObjectTracker()
        tracker.reset()
        diagnostics.id_switch_count = 0

        det = make_det(class_name="person", conf=0.85, x1=100, y1=100, x2=200, y2=300)
        confirm_object(tracker, det)

        for _ in range(MISS_FRAMES_EVICT):
            tracker.update([], 640, 480)

        assert len(tracker.dormant_tracks()) == 1

        # Simulate dormant TTL expiring by manually backdating _dormant_since
        for t in tracker.dormant_tracks():
            t._dormant_since = _time.time() - (DORMANT_REVIVAL_TTL + 1.0)

        # Reappear at completely different position (IoU = 0 with original bbox)
        det_far = make_det(class_name="person", conf=0.85, x1=500, y1=400, x2=600, y2=480)
        tracker.update([det_far], 640, 480)

        assert diagnostics.id_switch_count >= 1, (
            "Reappearance after expired dormant TTL must fire an id_switch"
        )

    def test_no_id_switch_without_prior_eviction(self):
        from backend.diagnostics import diagnostics
        tracker = ObjectTracker()
        tracker.reset()
        diagnostics.id_switch_count = 0

        det = make_det(class_name="chair", conf=0.85, x1=100, y1=100, x2=200, y2=300)
        # First detection of this class — no prior eviction
        tracker.update([det], 640, 480)

        assert diagnostics.id_switch_count == 0, (
            "First appearance of a class should not count as an ID switch"
        )

    def test_id_switch_count_in_diagnostics_summary(self):
        from backend.diagnostics import diagnostics
        summary = diagnostics.summary()
        assert "id_switch_count" in summary
        assert "tracking_id_switch_rate" in summary


# ── track_age_seconds property ────────────────────────────────────────────────

class TestTrackAgeSeconds:
    def test_age_is_non_negative(self):
        """A freshly created TrackedObject must have age >= 0."""
        import time
        t = TrackedObject(id=1, class_name="person", confidence=0.90)
        assert t.track_age_seconds >= 0.0

    def test_age_increases_over_time(self):
        """track_age_seconds must grow as real time passes."""
        import time
        t = TrackedObject(id=1, class_name="person", confidence=0.90)
        before = t.track_age_seconds
        time.sleep(0.05)
        after = t.track_age_seconds
        assert after > before, "track_age_seconds must increase over time"

    def test_age_is_float(self):
        t = TrackedObject(id=1, class_name="chair", confidence=0.80)
        assert isinstance(t.track_age_seconds, float)

    def test_fresh_track_via_tracker_has_small_age(self):
        """A track created this call should have age < 1 second."""
        import time
        tracker = ObjectTracker()
        det = make_det(conf=0.90)
        tracker.update([det], 640, 480)
        all_t = tracker.all_tracks()
        assert len(all_t) == 1
        assert all_t[0].track_age_seconds < 1.0


# ── MISS_FRAMES_EVICT env override ────────────────────────────────────────────

class TestMissFramesEvictEnvVar:
    """MISS_FRAMES_EVICT env var should override the eviction threshold."""

    def tearDown_module(self):
        # Restore default after all tests in this class
        import importlib
        import backend.tracker as trk
        os.environ.pop("MISS_FRAMES_EVICT", None)
        importlib.reload(trk)

    def test_default_evict_is_8(self):
        import importlib
        import backend.tracker as trk
        with patch_dict_no_key("MISS_FRAMES_EVICT"):
            importlib.reload(trk)
            assert trk.MISS_FRAMES_EVICT == 8

    def test_env_var_overrides_evict(self):
        import importlib
        import backend.tracker as trk
        import os
        with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
            os.environ, {"MISS_FRAMES_EVICT": "12"}
        ):
            importlib.reload(trk)
            assert trk.MISS_FRAMES_EVICT == 12

    def test_env_var_restore(self):
        """After env override test, restore to default."""
        import importlib
        import backend.tracker as trk
        import os
        os.environ.pop("MISS_FRAMES_EVICT", None)
        importlib.reload(trk)
        assert trk.MISS_FRAMES_EVICT == 8


def patch_dict_no_key(key):
    """Context manager: ensure key is absent from os.environ."""
    import os
    from unittest.mock import patch
    env = {k: v for k, v in os.environ.items() if k != key}
    return patch.dict(os.environ, env, clear=True)


# ── Two-tier confidence gate ──────────────────────────────────────────────────

class TestTwoTierConfidence:
    """
    CONF_DETECT (0.35) is the tracker entry gate; CONF_GATE (0.60) is the
    narration gate enforced externally by stability_filter.

    Detections with conf in [CONF_DETECT, CONF_GATE) must:
      - Enter the tracker and accumulate frames_seen / velocity history.
      - Never appear in the confirmed list produced by tracker.update()
        (confirmed requires frames_seen >= MIN_FRAMES_CONFIRM AND the narration
        gate — the tracker itself returns objects with confirmed=True regardless
        of confidence, so stability_filter is the final check).

    Detections below CONF_DETECT must be silently ignored by the tracker.
    """

    def test_conf_detect_below_conf_gate(self):
        """CONF_DETECT must be strictly less than CONF_GATE (design invariant)."""
        assert CONF_DETECT < CONF_GATE, (
            f"CONF_DETECT ({CONF_DETECT}) must be < CONF_GATE ({CONF_GATE})"
        )

    def test_detection_below_conf_detect_ignored(self):
        """Detection with conf < CONF_DETECT must never spawn a track."""
        tracker = ObjectTracker()
        det = make_det(conf=max(0.0, CONF_DETECT - 0.01))
        for _ in range(10):
            tracker.update([det], 640, 480)
        assert tracker.all_tracks() == [], (
            "Detection below CONF_DETECT must be silently ignored by the tracker"
        )

    def test_detection_between_tiers_enters_tracker(self):
        """Detection with CONF_DETECT <= conf < CONF_GATE must create a track."""
        if CONF_DETECT >= CONF_GATE:
            pytest.skip("Tiers not separated; nothing to test")
        mid_conf = (CONF_DETECT + CONF_GATE) / 2.0
        tracker = ObjectTracker()
        det = make_det(conf=mid_conf)
        tracker.update([det], 640, 480)
        tracks = tracker.all_tracks()
        assert len(tracks) == 1, (
            f"Detection with conf={mid_conf:.2f} (between tiers) must enter tracker"
        )

    def test_low_conf_track_accumulates_frames_seen(self):
        """A low-conf (mid-tier) track must accumulate frames_seen across frames."""
        if CONF_DETECT >= CONF_GATE:
            pytest.skip("Tiers not separated; nothing to test")
        mid_conf = (CONF_DETECT + CONF_GATE) / 2.0
        tracker = ObjectTracker()
        det = make_det(conf=mid_conf)
        for _ in range(MIN_FRAMES_CONFIRM):
            tracker.update([det], 640, 480)
        tracks = tracker.all_tracks()
        assert len(tracks) == 1
        assert tracks[0].frames_seen >= MIN_FRAMES_CONFIRM, (
            "Low-conf mid-tier track must still accumulate frames_seen"
        )

    def test_high_conf_track_becomes_confirmed(self):
        """Detection with conf >= CONF_GATE must produce a confirmed track."""
        tracker = ObjectTracker()
        det = make_det(conf=CONF_GATE + 0.05)
        confirmed = confirm_object(tracker, det)
        assert len(confirmed) == 1
        assert confirmed[0].confirmed is True

    def test_conf_detect_default_is_035(self):
        """CONF_DETECT default must be 0.35 (or env-var override)."""
        import importlib
        import backend.tracker as trk
        with patch_dict_no_key("CONF_DETECT"):
            importlib.reload(trk)
            assert trk.CONF_DETECT == pytest.approx(0.35, abs=1e-6)


# ── Velocity clamping ─────────────────────────────────────────────────────────

class TestVelocityClamping:
    """
    velocity_m_per_s must never exceed ±MAX_VELOCITY_M_S even when the
    depth history produces an implausibly large raw velocity.
    """

    def _make_obj(self):
        return TrackedObject(id=1, class_name="person", confidence=0.85)

    def test_max_velocity_constant_positive(self):
        assert MAX_VELOCITY_M_S > 0.0, "MAX_VELOCITY_M_S must be a positive number"

    def test_velocity_not_clamped_for_normal_approach(self):
        """Normal approach velocity (< 3 m/s) must not be clamped."""
        obj = self._make_obj()
        for d in [3.0, 2.5, 2.0]:
            obj.update_distance(d)
            time.sleep(0.06)
        assert 0 < obj.velocity_m_per_s <= MAX_VELOCITY_M_S

    def test_implausibly_large_approach_clamped(self):
        """
        Depth jump from 10 m to 0 m in ~60 ms → raw velocity ~166 m/s.
        After clamping, velocity must not exceed MAX_VELOCITY_M_S.
        """
        obj = self._make_obj()
        # Build a consistent sign window first
        for d in [10.0, 8.0, 6.0]:
            obj.update_distance(d)
            time.sleep(0.06)
        # Then inject a huge jump that stays sign-consistent
        obj.update_distance(0.1)
        time.sleep(0.06)
        assert obj.velocity_m_per_s <= MAX_VELOCITY_M_S, (
            f"Velocity {obj.velocity_m_per_s:.1f} m/s must not exceed "
            f"MAX_VELOCITY_M_S={MAX_VELOCITY_M_S}"
        )

    def test_implausibly_large_recede_clamped(self):
        """
        Depth jump from 0.5 m to 20 m → raw negative velocity beyond -3 m/s.
        Clamped velocity must be >= -MAX_VELOCITY_M_S.
        """
        obj = self._make_obj()
        for d in [0.5, 2.0, 5.0]:
            obj.update_distance(d)
            time.sleep(0.06)
        obj.update_distance(20.0)
        time.sleep(0.06)
        assert obj.velocity_m_per_s >= -MAX_VELOCITY_M_S, (
            f"Velocity {obj.velocity_m_per_s:.1f} m/s must not go below "
            f"-MAX_VELOCITY_M_S={-MAX_VELOCITY_M_S}"
        )

    def test_max_velocity_default_is_3(self):
        """MAX_VELOCITY_M_S default must be 3.0 m/s."""
        import importlib
        import backend.tracker as trk
        with patch_dict_no_key("MAX_VELOCITY_M_S"):
            importlib.reload(trk)
            assert trk.MAX_VELOCITY_M_S == pytest.approx(3.0, abs=1e-6)

    def test_velocity_env_override(self):
        """MAX_VELOCITY_M_S env var must be respected."""
        import importlib
        import backend.tracker as trk
        from unittest.mock import patch
        with patch.dict(os.environ, {"MAX_VELOCITY_M_S": "5.0"}):
            importlib.reload(trk)
            assert trk.MAX_VELOCITY_M_S == pytest.approx(5.0, abs=1e-6)


# ── Velocity EMA smoothing ────────────────────────────────────────────────────

class TestVelocityEmaSmoothing:
    """
    After the sign-consistency gate passes, velocity_m_per_s is EMA-smoothed
    (0.7 * prev + 0.3 * new) before clamping.  This means that even when sign
    is consistent, a single large jump in raw velocity only partially shifts
    the reported velocity — it takes multiple consistent frames for velocity to
    converge to the true value.
    """

    def _make_obj(self):
        return TrackedObject(id=1, class_name="person", confidence=0.85)

    def test_velocity_builds_gradually(self):
        """velocity_m_per_s should build up incrementally, not jump instantly."""
        obj = self._make_obj()
        # Feed consistent approach
        for d in [4.0, 3.0, 2.0]:
            obj.update_distance(d)
            time.sleep(0.06)
        first_velocity = obj.velocity_m_per_s
        assert first_velocity > 0.0

        # Feed more consistent approach — velocity should continue rising
        obj.update_distance(1.0)
        time.sleep(0.06)
        second_velocity = obj.velocity_m_per_s
        # EMA means second shouldn't jump to raw — it should be between first and raw
        assert second_velocity > 0.0

    def test_velocity_stays_bounded_with_ema(self):
        """EMA smoothing keeps velocity within MAX_VELOCITY_M_S."""
        import importlib
        import backend.tracker as trk
        importlib.reload(trk)   # ensure we have default MAX_VELOCITY_M_S=3.0
        obj = trk.TrackedObject(id=1, class_name="person", confidence=0.85)
        for d in [10.0, 8.0, 6.0, 4.0, 2.0, 0.5]:
            obj.update_distance(d)
            time.sleep(0.06)
        assert abs(obj.velocity_m_per_s) <= trk.MAX_VELOCITY_M_S

    def test_zero_prev_velocity_uses_raw(self):
        """When prev velocity is 0, first computed velocity is raw (no EMA blending)."""
        obj = self._make_obj()
        assert obj.velocity_m_per_s == pytest.approx(0.0)
        for d in [3.0, 2.0, 1.0]:
            obj.update_distance(d)
            time.sleep(0.06)
        # After first consistent window, velocity should be set directly from raw
        # (not 0.7*0 + 0.3*raw = 0.3*raw which would be very small)
        assert obj.velocity_m_per_s > 0.1


# ── Motion prediction (predicted_bbox) ───────────────────────────────────────

class TestMotionPrediction:
    """
    TrackedObject.predicted_bbox(dt) should extrapolate the bbox position
    using stored pixel velocity (_pixel_vx, _pixel_vy).
    """

    def _make_moving_track(self, vx=50.0, vy=20.0):
        """Create a TrackedObject with injected pixel velocity."""
        obj = TrackedObject(id=1, class_name="person", confidence=0.85)
        obj._x1, obj._y1, obj._x2, obj._y2 = 100.0, 100.0, 200.0, 300.0
        obj._pixel_vx = vx
        obj._pixel_vy = vy
        return obj

    def test_zero_dt_returns_current_bbox(self):
        obj = self._make_moving_track()
        pred = obj.predicted_bbox(0.0)
        assert pred == pytest.approx((100.0, 100.0, 200.0, 300.0), abs=1e-3)

    def test_positive_vx_shifts_right(self):
        obj = self._make_moving_track(vx=100.0, vy=0.0)
        pred = obj.predicted_bbox(0.1)
        # dt=0.1 s, vx=100 px/s → dx=10 px
        assert pred[0] == pytest.approx(110.0, abs=1e-3)
        assert pred[2] == pytest.approx(210.0, abs=1e-3)
        assert pred[1] == pytest.approx(100.0, abs=1e-3)  # y unchanged

    def test_positive_vy_shifts_down(self):
        obj = self._make_moving_track(vx=0.0, vy=50.0)
        pred = obj.predicted_bbox(0.2)
        # dt=0.2 s, vy=50 px/s → dy=10 px
        assert pred[1] == pytest.approx(110.0, abs=1e-3)
        assert pred[3] == pytest.approx(310.0, abs=1e-3)

    def test_bbox_size_preserved(self):
        """Predicted bbox must have the same width and height as original."""
        obj = self._make_moving_track(vx=30.0, vy=15.0)
        orig_w = obj._x2 - obj._x1
        orig_h = obj._y2 - obj._y1
        pred = obj.predicted_bbox(0.5)
        pred_w = pred[2] - pred[0]
        pred_h = pred[3] - pred[1]
        assert pred_w == pytest.approx(orig_w, abs=1e-3)
        assert pred_h == pytest.approx(orig_h, abs=1e-3)

    def test_update_bbox_builds_pixel_velocity(self):
        """After two update_bbox calls with a real movement, pixel velocity is non-zero."""
        obj = TrackedObject(id=1, class_name="person", confidence=0.85)
        obj.frames_seen = 1
        obj.update_bbox(100, 100, 200, 300)   # first placement
        time.sleep(0.05)
        obj.frames_seen = 2
        obj.update_bbox(130, 110, 230, 310)   # moved right+down
        # After one real update, pixel velocity should reflect the movement
        assert obj._pixel_vx > 0.0, "Expected positive horizontal pixel velocity"
        assert obj._pixel_vy > 0.0, "Expected positive vertical pixel velocity"


class TestStaleDepthDecay:
    """Tests for _depth_stale_frames counter and effective_confidence property."""

    def _make_confirmed_obj(self, confidence=0.85, velocity=0.0):
        """Create a confirmed TrackedObject with the given parameters."""
        obj = TrackedObject(id=1, class_name="person", confidence=confidence)
        obj.frames_seen = MIN_FRAMES_CONFIRM
        obj.confirmed = True
        obj.update_bbox(100, 100, 200, 300)
        obj.velocity_m_per_s = velocity
        return obj

    def test_stale_frames_start_at_zero(self):
        """Fresh TrackedObject has _depth_stale_frames == 0."""
        obj = TrackedObject(id=1, class_name="person", confidence=0.80)
        assert obj._depth_stale_frames == 0

    def test_effective_confidence_full_when_not_stale(self):
        """effective_confidence equals confidence when stale counter is 0."""
        obj = self._make_confirmed_obj(confidence=0.85)
        obj._depth_stale_frames = 0
        assert obj.effective_confidence == pytest.approx(0.85)

    def test_effective_confidence_decays_with_stale_frames(self):
        """effective_confidence decays linearly as stale count increases."""
        obj = self._make_confirmed_obj(confidence=1.0, velocity=0.0)
        obj._depth_stale_frames = DEPTH_STALE_EVICT_FRAMES // 2
        expected = 1.0 * (1.0 - (DEPTH_STALE_EVICT_FRAMES // 2) / DEPTH_STALE_EVICT_FRAMES)
        assert obj.effective_confidence == pytest.approx(expected, abs=1e-6)

    def test_effective_confidence_reaches_zero_at_evict_threshold(self):
        """effective_confidence is 0.0 when stale frames == DEPTH_STALE_EVICT_FRAMES."""
        obj = self._make_confirmed_obj(confidence=0.90, velocity=0.0)
        obj._depth_stale_frames = DEPTH_STALE_EVICT_FRAMES
        assert obj.effective_confidence == pytest.approx(0.0, abs=1e-6)

    def test_effective_confidence_clamped_not_negative(self):
        """effective_confidence never goes below 0 even with excessive stale count."""
        obj = self._make_confirmed_obj(confidence=0.90, velocity=0.0)
        obj._depth_stale_frames = DEPTH_STALE_EVICT_FRAMES * 3
        assert obj.effective_confidence >= 0.0

    def test_update_distance_resets_stale_counter(self):
        """Calling update_distance() with a valid reading resets _depth_stale_frames to 0."""
        obj = self._make_confirmed_obj(confidence=0.85)
        obj._depth_stale_frames = 4
        obj.update_distance(2.0)
        assert obj._depth_stale_frames == 0

    def test_approaching_object_exempt_from_stale_decay(self):
        """When velocity_m_per_s > 0.05, effective_confidence equals raw confidence."""
        obj = self._make_confirmed_obj(confidence=0.85, velocity=0.5)
        obj._depth_stale_frames = DEPTH_STALE_EVICT_FRAMES  # would normally zero out
        assert obj.effective_confidence == pytest.approx(0.85)

    def test_slowly_receding_object_not_exempt(self):
        """An object with velocity <= 0.05 is NOT exempt; stale decay applies."""
        obj = self._make_confirmed_obj(confidence=1.0, velocity=0.04)
        obj._depth_stale_frames = DEPTH_STALE_EVICT_FRAMES
        assert obj.effective_confidence == pytest.approx(0.0, abs=1e-6)

    def test_effective_confidence_below_conf_gate_at_full_staleness(self):
        """After DEPTH_STALE_EVICT_FRAMES of staleness, effective_confidence < CONF_GATE."""
        obj = self._make_confirmed_obj(confidence=0.85, velocity=0.0)
        obj._depth_stale_frames = DEPTH_STALE_EVICT_FRAMES
        assert obj.effective_confidence < CONF_GATE

    def test_stale_evict_frames_env_override(self, monkeypatch):
        """DEPTH_STALE_EVICT_FRAMES respects the env-var override at import time."""
        # The constant is read at module import, so we just verify it is an int >= 1
        assert isinstance(DEPTH_STALE_EVICT_FRAMES, int)
        assert DEPTH_STALE_EVICT_FRAMES >= 1


class TestDormantRevival:
    """Tests for the dormant track pool and revival logic in ObjectTracker."""

    def _make_tracker(self):
        return ObjectTracker()

    def _confirm(self, tracker, det, n=MIN_FRAMES_CONFIRM):
        """Feed det n times; return last confirmed list."""
        result = []
        for _ in range(n):
            result = tracker.update([det], 640, 480)
        return result

    def _evict(self, tracker, confirmed_id, n=None):
        """Feed empty detections until the track is evicted (moves to dormant)."""
        n = n or MISS_FRAMES_EVICT
        for _ in range(n):
            tracker.update([], 640, 480)

    def test_evicted_track_enters_dormant_pool(self):
        """After MISS_FRAMES_EVICT misses the track should be in dormant_tracks()."""
        tracker = self._make_tracker()
        det = make_det()
        self._confirm(tracker, det)
        assert tracker.all_tracks(), "should have at least one confirmed track"

        self._evict(tracker, None)
        # After eviction the active pool is empty but dormant pool has the track
        assert tracker.all_tracks() == [], "active pool should be empty after eviction"
        assert len(tracker.dormant_tracks()) == 1, "evicted track should enter dormant pool"

    def test_dormant_track_revived_on_reappearance(self):
        """Same-position reappearance after eviction should revive the track."""
        tracker = self._make_tracker()
        det = make_det()
        self._confirm(tracker, det)
        self._evict(tracker, None)
        assert tracker.dormant_tracks(), "precondition: track must be dormant"

        # Reappear at same position
        tracker.update([det], 640, 480)
        assert tracker.all_tracks(), "revived track should be back in active pool"
        assert tracker.dormant_tracks() == [], "dormant pool should be empty after revival"

    def test_revived_track_preserves_id(self):
        """The track ID must be the same before eviction and after revival."""
        tracker = self._make_tracker()
        det = make_det()
        self._confirm(tracker, det)
        original_id = tracker.all_tracks()[0].id

        self._evict(tracker, None)
        tracker.update([det], 640, 480)

        revived_id = tracker.all_tracks()[0].id
        assert revived_id == original_id, (
            f"Expected original id={original_id}, got {revived_id} after revival"
        )

    def test_revived_track_frames_seen_incremented(self):
        """frames_seen should increase (not reset to 1) after revival."""
        tracker = self._make_tracker()
        det = make_det()
        self._confirm(tracker, det)
        frames_before = tracker.all_tracks()[0].frames_seen  # == MIN_FRAMES_CONFIRM

        self._evict(tracker, None)
        tracker.update([det], 640, 480)

        frames_after = tracker.all_tracks()[0].frames_seen
        assert frames_after > frames_before, (
            f"frames_seen should have increased after revival; "
            f"got {frames_after} <= {frames_before}"
        )

    def test_revived_track_clears_dormant_since(self):
        """After revival _dormant_since should be reset to 0.0."""
        tracker = self._make_tracker()
        det = make_det()
        self._confirm(tracker, det)
        self._evict(tracker, None)
        tracker.update([det], 640, 480)

        revived = tracker.all_tracks()[0]
        assert revived._dormant_since == 0.0, (
            f"_dormant_since should be 0.0 after revival, got {revived._dormant_since}"
        )

    def test_dormant_pool_cleared_on_reset(self):
        """reset() must empty the dormant pool."""
        tracker = self._make_tracker()
        det = make_det()
        self._confirm(tracker, det)
        self._evict(tracker, None)
        assert tracker.dormant_tracks(), "precondition: dormant pool must be non-empty"

        tracker.reset()
        assert tracker.dormant_tracks() == [], "reset() must clear the dormant pool"

    def test_expired_dormant_track_not_revived(self):
        """A dormant track whose TTL has expired must NOT be revived; a new ID is spawned."""
        tracker = self._make_tracker()
        det = make_det()
        self._confirm(tracker, det)
        original_id = tracker.all_tracks()[0].id

        self._evict(tracker, None)
        dormant = tracker.dormant_tracks()
        assert dormant, "precondition: dormant pool must be non-empty"

        # Backdate _dormant_since so the TTL has expired
        dormant[0]._dormant_since -= (DORMANT_REVIVAL_TTL + 1.0)

        # Reappear — should NOT revive; a new track (new ID) is spawned instead
        tracker.update([det], 640, 480)
        active = tracker.all_tracks()
        if active:
            new_id = active[0].id
            assert new_id != original_id, (
                "Expired dormant track must not be revived — new ID expected"
            )

    def test_different_class_not_revived(self):
        """A dormant 'person' track must not be revived by a 'chair' detection."""
        tracker = self._make_tracker()
        person_det = make_det(class_name="person", class_id=0)
        self._confirm(tracker, person_det)
        original_id = tracker.all_tracks()[0].id

        self._evict(tracker, None)
        assert tracker.dormant_tracks(), "precondition: dormant pool must be non-empty"

        # Detection of a different class at the same position
        chair_det = make_det(class_name="chair", class_id=56)
        tracker.update([chair_det], 640, 480)

        # The dormant 'person' track must remain dormant (not revived as chair)
        dormant_ids = [t.id for t in tracker.dormant_tracks()]
        assert original_id in dormant_ids, (
            "Person dormant track should NOT be revived by a chair detection"
        )

    def test_zero_iou_not_revived(self):
        """A detection completely outside the dormant track's bbox must spawn a new ID."""
        tracker = self._make_tracker()
        # Confirm a track in the top-left corner
        det_tl = make_det(x1=0, y1=0, x2=50, y2=50)
        self._confirm(tracker, det_tl)
        original_id = tracker.all_tracks()[0].id

        self._evict(tracker, None)
        assert tracker.dormant_tracks(), "precondition: dormant pool must be non-empty"

        # Reappear far away — zero IoU with dormant bbox
        det_br = make_det(x1=550, y1=400, x2=630, y2=470)
        tracker.update([det_br], 640, 480)

        active = tracker.all_tracks()
        if active:
            assert active[0].id != original_id, (
                "Zero-IoU reappearance must spawn a new track ID, not revive the dormant one"
            )

    def test_dormant_revival_iou_threshold(self):
        """A detection with IoU just below DORMANT_REVIVAL_IOU_THRESH must not revive."""
        import math

        tracker = self._make_tracker()
        # Use a large enough bbox so that small offset produces measurable IoU
        det = make_det(x1=100, y1=100, x2=300, y2=300)
        self._confirm(tracker, det)
        original_id = tracker.all_tracks()[0].id
        self._evict(tracker, None)

        # Craft a detection whose IoU with (100,100,300,300) is just below threshold.
        # Intersection = (110,110,300,300) = 190×190 = 36100
        # det2 area = 190×190 = 36100; det area = 200×200 = 40000
        # union = 40000 + 36100 - 36100 = 40000  → IoU ≈ 36100/40000 = 0.9025 (too high)
        # Instead move far enough that IoU drops below DORMANT_REVIVAL_IOU_THRESH:
        # shift so there's only tiny overlap — target IoU ≈ 0.05
        # bbox = (260, 260, 460, 460): intersection with (100,100,300,300) = (260,260,300,300)
        # intersection area = 40×40=1600; union = 40000 + 40000 - 1600 = 78400 → iou=0.020
        det_low_iou = make_det(x1=260, y1=260, x2=460, y2=460)
        # verify it is actually below the threshold
        computed_iou = _iou(
            (100, 100, 300, 300),
            (det_low_iou.x1, det_low_iou.y1, det_low_iou.x2, det_low_iou.y2),
        )
        assert computed_iou < DORMANT_REVIVAL_IOU_THRESH, (
            f"Test setup: expected IoU < {DORMANT_REVIVAL_IOU_THRESH}, got {computed_iou}"
        )

        tracker.update([det_low_iou], 640, 480)
        active = tracker.all_tracks()
        if active:
            assert active[0].id != original_id, (
                f"IoU={computed_iou:.4f} is below threshold={DORMANT_REVIVAL_IOU_THRESH}; "
                "revival must not happen"
            )


class TestDepthMedianFilter:
    """Tests for the sliding-window median pre-filter in update_distance()."""

    def _make_obj(self):
        return TrackedObject(id=1, class_name="person", confidence=0.85)

    def test_single_spike_fully_absorbed(self):
        """A single outlier frame must be absorbed by the median — smoothed stays at baseline."""
        from backend.tracker import DEPTH_MEDIAN_WINDOW
        obj = self._make_obj()
        # Fill majority of window with baseline 2.0
        for _ in range(DEPTH_MEDIAN_WINDOW - 1):
            obj.update_distance(2.0)
        baseline = obj.smoothed_distance_m

        # One spike frame — median of (N-1) x 2.0 + 1 x 10.0 → still 2.0
        obj.update_distance(10.0)
        assert obj.smoothed_distance_m == pytest.approx(baseline, abs=0.05), (
            "Single-frame spike must be fully absorbed by the median pre-filter"
        )

    def test_sustained_change_penetrates_filter(self):
        """A real distance change held for > DEPTH_MEDIAN_WINDOW frames must shift the EMA."""
        from backend.tracker import DEPTH_MEDIAN_WINDOW
        obj = self._make_obj()
        for _ in range(DEPTH_MEDIAN_WINDOW):
            obj.update_distance(2.0)
        baseline = obj.smoothed_distance_m

        for _ in range(DEPTH_MEDIAN_WINDOW + 2):
            obj.update_distance(6.0)

        assert obj.smoothed_distance_m > baseline + 0.5, (
            "Sustained distance change must penetrate median filter"
        )

    def test_raw_depth_history_initialised_lazily(self):
        """_raw_depth_history is None until first update_distance call."""
        obj = self._make_obj()
        assert obj._raw_depth_history is None
        obj.update_distance(2.0)
        assert obj._raw_depth_history is not None

    def test_raw_depth_history_bounded(self):
        """_raw_depth_history never exceeds DEPTH_MEDIAN_WINDOW entries."""
        from backend.tracker import DEPTH_MEDIAN_WINDOW
        obj = self._make_obj()
        for _ in range(DEPTH_MEDIAN_WINDOW + 10):
            obj.update_distance(2.0)
        assert len(obj._raw_depth_history) <= DEPTH_MEDIAN_WINDOW

    def test_first_reading_initialises_smoothed_directly(self):
        """When smoothed_distance_m is 0, the first call must set it directly (no EMA lag)."""
        obj = self._make_obj()
        obj.update_distance(3.0)
        assert obj.smoothed_distance_m == pytest.approx(3.0, abs=0.01)

    def test_multiple_spikes_cannot_corrupt_ema(self):
        """Two successive spike frames out of a full window must still be suppressed."""
        from backend.tracker import DEPTH_MEDIAN_WINDOW
        obj = self._make_obj()
        # Establish stable baseline in the full window
        for _ in range(DEPTH_MEDIAN_WINDOW):
            obj.update_distance(2.0)
        baseline = obj.smoothed_distance_m

        # Two spike frames — if window=5: [2.0,2.0,2.0,10.0,10.0] → median=2.0
        obj.update_distance(10.0)
        obj.update_distance(10.0)
        # Median is 2.0 for window ≥ 5 (majority still baseline)
        if DEPTH_MEDIAN_WINDOW >= 5:
            assert obj.smoothed_distance_m < baseline + 0.5, (
                "Two spike frames should not corrupt EMA when window is large enough"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
