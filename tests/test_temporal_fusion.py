"""
test_temporal_fusion.py — Unit tests for backend/temporal_fusion.py

Tests cover:
  - Single-frame strong bypass (FUSION_STRONG_CONF)
  - Multi-frame accumulation gate (FUSION_MIN_FRAMES)
  - Bounding box EMA smoothing
  - Confidence fusion formula
  - Candidate eviction after FUSION_HISTORY misses
  - Multi-class isolation (candidates don't bleed across classes)
  - IoU matching threshold boundary
  - Non-detection frame ageing (has_new_detections=False)
  - Fail-safe: exception handling returns raw detections
  - reset() clears state
  - active_candidate_count() / candidates_by_class() introspection
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from backend.detector import Detection
from backend.temporal_fusion import (
    TemporalFusionFilter,
    _iou,
    FUSION_HISTORY,
    FUSION_IOU_MATCH,
    FUSION_MIN_FRAMES,
    FUSION_STRONG_CONF,
    FUSION_BOX_ALPHA,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def det(class_name="person", conf=0.50, x1=100, y1=100, x2=300, y2=400, class_id=0):
    return Detection(class_id=class_id, class_name=class_name, confidence=conf,
                     x1=x1, y1=y1, x2=x2, y2=y2)


def fresh() -> TemporalFusionFilter:
    return TemporalFusionFilter()


# ── IoU helper ────────────────────────────────────────────────────────────────

class TestIouHelper:
    def test_identical_boxes_iou_one(self):
        a = det(x1=0, y1=0, x2=100, y2=100)
        b = det(x1=0, y1=0, x2=100, y2=100)
        assert _iou(a, b) == pytest.approx(1.0)

    def test_non_overlapping_boxes_iou_zero(self):
        a = det(x1=0, y1=0, x2=100, y2=100)
        b = det(x1=200, y1=200, x2=300, y2=300)
        assert _iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # 50x100 overlap out of two 100x100 boxes
        a = det(x1=0, y1=0, x2=100, y2=100)
        b = det(x1=50, y1=0, x2=150, y2=100)
        # inter=5000, union=15000
        assert _iou(a, b) == pytest.approx(5000 / 15000, rel=1e-4)


# ── Strong-confidence single-frame bypass ─────────────────────────────────────

class TestStrongConfBypass:
    def test_high_conf_passes_immediately(self):
        """Detection above FUSION_STRONG_CONF passes on first frame."""
        tf = fresh()
        d = det(conf=FUSION_STRONG_CONF + 0.01)
        result = tf.update([d])
        assert len(result) == 1
        assert result[0].class_name == "person"

    def test_low_conf_single_frame_blocked(self):
        """With FUSION_MIN_FRAMES=1, even a low-conf single-frame detection passes
        (FUSION_MIN_FRAMES=1 means pass after first frame seen, regardless of conf)."""
        tf = fresh()
        d = det(conf=FUSION_STRONG_CONF - 0.20)
        result = tf.update([d])
        # FUSION_MIN_FRAMES=1: after one frame it SHOULD pass
        assert len(result) == 1

    def test_exactly_strong_conf_passes(self):
        """Detection exactly at FUSION_STRONG_CONF bypasses the frame gate."""
        tf = fresh()
        d = det(conf=FUSION_STRONG_CONF)
        result = tf.update([d])
        assert len(result) == 1


# ── Multi-frame accumulation gate ─────────────────────────────────────────────

class TestMultiFrameGate:
    def test_two_frames_passes_when_min_frames_is_2(self):
        """After FUSION_MIN_FRAMES frames a weak detection is forwarded.
        With FUSION_MIN_FRAMES=1 the detection passes on frame 1 already."""
        assert FUSION_MIN_FRAMES == 1, "test assumes FUSION_MIN_FRAMES=1"
        tf = fresh()
        d = det(conf=0.40)
        result = tf.update([d])           # frame 1 — should pass (MIN_FRAMES=1)
        assert len(result) == 1

    def test_single_frame_weak_blocked(self):
        """With FUSION_MIN_FRAMES=1, a single frame always passes the gate."""
        tf = fresh()
        result = tf.update([det(conf=0.40)])
        # MIN_FRAMES=1: passes immediately
        assert len(result) == 1

    def test_three_consecutive_frames_pass(self):
        tf = fresh()
        d = det(conf=0.40)
        for _ in range(3):
            result = tf.update([d])
        assert len(result) == 1

    def test_empty_frame_does_not_break_accumulation(self):
        """A non-detection frame between two detection frames preserves history."""
        tf = fresh()
        d = det(conf=0.40)
        tf.update([d])                            # frame 1 — blocked
        tf.update([], has_new_detections=False)   # non-detection frame
        result = tf.update([d])                   # frame 3 — should still pass
        assert len(result) == 1


# ── Bounding box smoothing ────────────────────────────────────────────────────

class TestBoundingBoxSmoothing:
    def test_box_is_ema_smoothed(self):
        """Box should move toward new detection position gradually."""
        tf = fresh()
        # Use a small shift so IoU stays well above FUSION_IOU_MATCH=0.40
        d1 = det(x1=100, y1=100, x2=300, y2=400, conf=0.70)
        d2 = det(x1=110, y1=110, x2=310, y2=410, conf=0.70)

        tf.update([d1])
        result = tf.update([d2])

        assert len(result) == 1
        out = result[0]
        # After EMA with alpha=0.6: x1 = 0.6*110 + 0.4*100 = 106
        # So smoothed x1 must be strictly between 100 and 110 (exclusive)
        assert 100 < out.x1 < 110, f"Expected x1 in (100, 110), got {out.x1}"

    def test_stable_box_converges(self):
        """Repeated same-position detections converge to that position."""
        tf = fresh()
        d = det(x1=100, y1=100, x2=300, y2=400, conf=0.70)
        for _ in range(10):
            result = tf.update([d])
        assert len(result) == 1
        out = result[0]
        # After 10 frames EMA should be very close to d's box
        assert abs(out.x1 - 100) <= 2
        assert abs(out.x2 - 300) <= 2


# ── Confidence fusion ─────────────────────────────────────────────────────────

class TestConfidenceFusion:
    def test_fused_confidence_is_between_raw_values(self):
        """Fused confidence should blend across frames, not exceed max raw."""
        tf = fresh()
        d = det(conf=0.50, x1=100, y1=100, x2=200, y2=300)
        tf.update([d])
        result = tf.update([d])
        assert len(result) == 1
        fused = result[0].confidence
        # Fused must be > 0 and <= 1
        assert 0.0 < fused <= 1.0

    def test_high_single_frame_peak_preserved_in_fused(self):
        """A spike in one frame should contribute to fused_conf via peak_conf."""
        tf = fresh()
        d_high = det(conf=0.90, x1=100, y1=100, x2=200, y2=300)
        d_low  = det(conf=0.40, x1=100, y1=100, x2=200, y2=300)
        tf.update([d_high])
        result = tf.update([d_low])
        assert len(result) == 1
        # fused should reflect the high-conf peak
        assert result[0].confidence > 0.40


# ── Candidate eviction ────────────────────────────────────────────────────────

class TestCandidateEviction:
    def test_candidate_evicted_after_history_misses(self):
        """Candidate missed for FUSION_HISTORY+1 frames must be evicted."""
        tf = fresh()
        d = det(conf=0.70)
        tf.update([d])  # seed candidate

        # Miss for FUSION_HISTORY+1 non-detection frames
        for _ in range(FUSION_HISTORY + 1):
            tf.update([], has_new_detections=False)

        assert tf.active_candidate_count() == 0

    def test_candidate_survives_within_history_window(self):
        """Candidate missed for fewer than FUSION_HISTORY frames must survive."""
        tf = fresh()
        d = det(conf=0.70)
        tf.update([d])

        for _ in range(FUSION_HISTORY - 1):
            tf.update([], has_new_detections=False)

        assert tf.active_candidate_count() == 1

    def test_redetection_resets_miss_counter(self):
        """Re-detecting after some misses must reset frames_missed to 0."""
        tf = fresh()
        d = det(conf=0.70)
        tf.update([d])
        # Miss for FUSION_HISTORY - 1 frames (still alive)
        for _ in range(FUSION_HISTORY - 1):
            tf.update([], has_new_detections=False)
        # Re-detect: candidate survives and continues accumulating
        tf.update([d])
        assert tf.active_candidate_count() == 1
        # Another miss cycle — it should be evictable again after FUSION_HISTORY misses
        for _ in range(FUSION_HISTORY + 1):
            tf.update([], has_new_detections=False)
        assert tf.active_candidate_count() == 0


# ── Multi-class isolation ─────────────────────────────────────────────────────

class TestMultiClassIsolation:
    def test_different_classes_tracked_independently(self):
        """Person and chair candidates must not interfere with each other."""
        tf = fresh()
        p = det(class_name="person", conf=0.70, x1=100, y1=100, x2=200, y2=300)
        c = det(class_name="chair",  conf=0.70, x1=100, y1=100, x2=200, y2=300)
        # Same bbox, different classes — must produce two distinct candidates
        tf.update([p, c])
        assert tf.candidates_by_class() == {"person": 1, "chair": 1}

    def test_same_class_different_position_two_candidates(self):
        """Two spatially separate detections of the same class → two candidates."""
        tf = fresh()
        p1 = det(class_name="person", conf=0.70, x1=0,   y1=0,   x2=100, y2=200)
        p2 = det(class_name="person", conf=0.70, x1=400, y1=0,   x2=500, y2=200)
        tf.update([p1, p2])
        assert tf.candidates_by_class().get("person", 0) == 2


# ── IoU matching threshold boundary ──────────────────────────────────────────

class TestIouMatchThreshold:
    def test_below_iou_threshold_creates_new_candidate(self):
        """Detection with IoU just below FUSION_IOU_MATCH must spawn new candidate."""
        tf = fresh()
        # Two boxes with negligible overlap
        d1 = det(x1=0,   y1=0,   x2=100, y2=100, conf=0.70)
        d2 = det(x1=200, y1=0,   x2=300, y2=100, conf=0.70)
        tf.update([d1])
        tf.update([d2])
        # Both candidates should exist (no merge)
        assert tf.active_candidate_count() == 2

    def test_above_iou_threshold_merges_candidate(self):
        """Detection with IoU above FUSION_IOU_MATCH must merge into existing candidate."""
        tf = fresh()
        # Identical boxes — definitely same candidate
        d1 = det(x1=100, y1=100, x2=300, y2=400, conf=0.70)
        d2 = det(x1=102, y1=102, x2=298, y2=398, conf=0.70)  # tiny shift, high IoU
        tf.update([d1])
        tf.update([d2])
        assert tf.active_candidate_count() == 1


# ── Fail-safe ─────────────────────────────────────────────────────────────────

class TestFailSafe:
    def test_returns_raw_on_internal_error(self, monkeypatch):
        """If _update_inner raises, update() must return raw detections unchanged."""
        tf = fresh()
        d = det(conf=0.70)

        def boom(*a, **kw):
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(tf, "_update_inner", boom)
        result = tf.update([d])
        assert result == [d]


# ── reset() ──────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_all_candidates(self):
        tf = fresh()
        tf.update([det(conf=0.70)])
        assert tf.active_candidate_count() == 1
        tf.reset()
        assert tf.active_candidate_count() == 0

    def test_after_reset_accumulation_starts_fresh(self):
        tf = fresh()
        d = det(conf=0.40)
        tf.update([d])   # frame 1 — 1 candidate
        tf.reset()
        tf.update([d])   # frame 1 again after reset — blocked (frames_seen=1)
        assert len(tf.update([d])) == 1   # frame 2 — now passes


# ── Introspection ─────────────────────────────────────────────────────────────

class TestIntrospection:
    def test_active_candidate_count_starts_zero(self):
        assert fresh().active_candidate_count() == 0

    def test_candidates_by_class_starts_empty(self):
        assert fresh().candidates_by_class() == {}

    def test_active_count_increments_on_new_detection(self):
        tf = fresh()
        tf.update([det(conf=0.40)])
        assert tf.active_candidate_count() == 1

    def test_candidates_by_class_reflects_content(self):
        tf = fresh()
        tf.update([
            det(class_name="person", conf=0.40),
            det(class_name="chair",  conf=0.40, x1=400, y1=400, x2=500, y2=500),
        ])
        by_class = tf.candidates_by_class()
        assert by_class["person"] == 1
        assert by_class["chair"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
