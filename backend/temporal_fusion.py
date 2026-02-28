"""
temporal_fusion.py — Multi-frame detection confidence fusion for VisionTalk.

Problem this solves
-------------------
YOLO is high-precision but temporally noisy: a real object may appear in
frames 1, 3, 5 but be missed in frames 2 and 4.  The tracker's confirmation
gate (MIN_FRAMES_CONFIRM=3) already handles this at the track level, but
flickering raw detections cause:

  - Unstable bounding boxes (jitter) passed to the tracker each frame.
  - High-confidence single-frame hallucinations that seed false tracks.
  - Transient missed detections that reset tracker history for real objects.

Solution
--------
TemporalFusionFilter maintains a rolling window (FUSION_HISTORY frames) of
raw detections.  For each new detection batch it:

  1. Matches new detections to candidates in the history via IoU.
  2. Accumulates per-candidate frame count and a recency-weighted confidence.
  3. Passes a detection to the tracker only when EITHER:
       a. The candidate has been seen for >= FUSION_MIN_FRAMES consecutive
          (or near-consecutive) YOLO frames, OR
       b. Its raw confidence exceeds FUSION_STRONG_CONF (strong single-frame
          detections are passed immediately to avoid latency on fast hazards).
  4. Averages the bounding boxes of matched history entries to produce a
     temporally stable box.

Integration
-----------
In pipeline.py, replace:

    confirmed = object_tracker.update(detections, w, h)

with:

    fused = temporal_fusion.update(detections, has_new_frame=True)
    confirmed = object_tracker.update(fused, w, h)

Pass has_new_frame=False on non-detection frames so the history window does
not age out candidates unnecessarily.

Parameters (tuned for 10–15 FPS YOLO cadence)
----------------------------------------------
FUSION_HISTORY      = 5     # frames of raw detection history kept
FUSION_IOU_MATCH    = 0.40  # IoU threshold to consider same object
FUSION_MIN_FRAMES   = 1     # minimum seen-frames to pass to tracker
FUSION_STRONG_CONF  = 0.45  # single-frame bypass threshold (fast hazards)
FUSION_BOX_ALPHA    = 0.6   # EMA weight for box smoothing (current frame)

Design constraints
------------------
- Non-blocking; no threading required (called from main pipeline thread).
- Stateless public interface: update(detections) → List[Detection].
- Never raises; on any error returns raw detections unchanged (fail-safe).
- No neural networks or external model dependencies.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from backend.detector import Detection

logger = logging.getLogger(__name__)

# ── Parameters ────────────────────────────────────────────────────────────────

FUSION_HISTORY:     int   = int(os.getenv("FUSION_HISTORY",    "5"))
FUSION_IOU_MATCH:   float = float(os.getenv("FUSION_IOU_MATCH", "0.40"))
FUSION_MIN_FRAMES:  int   = int(os.getenv("FUSION_MIN_FRAMES", "1"))
FUSION_STRONG_CONF: float = float(os.getenv("FUSION_STRONG_CONF", "0.45"))
FUSION_BOX_ALPHA:   float = float(os.getenv("FUSION_BOX_ALPHA",  "0.6"))


# ── Internal data structures ──────────────────────────────────────────────────

@dataclass
class _Candidate:
    """A detection candidate being tracked across frames."""
    class_name:  str
    class_id:    int

    # Smoothed bounding box (EMA of matched raw boxes)
    x1: float
    y1: float
    x2: float
    y2: float

    # Accumulation state
    frames_seen:  int   = 1      # how many YOLO frames it has appeared in
    peak_conf:    float = 0.0    # highest single-frame confidence seen
    fused_conf:   float = 0.0    # recency-weighted confidence

    # Frames since last match (incremented on non-detection frames and misses)
    frames_missed: int  = 0

    def update(self, det: Detection, alpha: float = FUSION_BOX_ALPHA) -> None:
        """Merge a new raw detection into this candidate."""
        # EMA box smoothing
        self.x1 = alpha * det.x1 + (1 - alpha) * self.x1
        self.y1 = alpha * det.y1 + (1 - alpha) * self.y1
        self.x2 = alpha * det.x2 + (1 - alpha) * self.x2
        self.y2 = alpha * det.y2 + (1 - alpha) * self.y2

        self.frames_seen += 1
        self.frames_missed = 0
        self.peak_conf = max(self.peak_conf, det.confidence)

        # Recency-weighted confidence: weight current frame more heavily
        # fused = 0.5 * current + 0.3 * prev_fused + 0.2 * peak
        self.fused_conf = (
            0.5 * det.confidence
            + 0.3 * self.fused_conf
            + 0.2 * self.peak_conf
        )

    def to_detection(self) -> Detection:
        """Emit a Detection with smoothed box and fused confidence."""
        return Detection(
            class_id=self.class_id,
            class_name=self.class_name,
            confidence=self.fused_conf,
            x1=int(round(self.x1)),
            y1=int(round(self.y1)),
            x2=int(round(self.x2)),
            y2=int(round(self.y2)),
        )

    @property
    def passes_gate(self) -> bool:
        """True if this candidate should be forwarded to the tracker."""
        if self.peak_conf >= FUSION_STRONG_CONF:
            return True                        # strong single-frame bypass
        return self.frames_seen >= FUSION_MIN_FRAMES


# ── IoU helper ────────────────────────────────────────────────────────────────

def _iou(a: "_Candidate | Detection", b: Detection) -> float:
    """Intersection-over-union between two boxes."""
    ax1 = a.x1 if hasattr(a, "x1") else a.x1
    ax2 = a.x2 if hasattr(a, "x2") else a.x2
    ay1 = a.y1 if hasattr(a, "y1") else a.y1
    ay2 = a.y2 if hasattr(a, "y2") else a.y2

    ix1 = max(ax1, b.x1)
    iy1 = max(ay1, b.y1)
    ix2 = min(ax2, b.x2)
    iy2 = min(ay2, b.y2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union  = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


# ── Main class ────────────────────────────────────────────────────────────────

class TemporalFusionFilter:
    """
    Stateful multi-frame detection fusion filter.

    One instance is intended to be used per pipeline session (singleton).
    Call update() once per frame with the raw YOLO detections for that frame.
    """

    def __init__(self) -> None:
        # Active candidates: class_name → list of _Candidate
        # Keyed by class to limit the matching search space.
        self._candidates: dict[str, list[_Candidate]] = {}

    def reset(self) -> None:
        """Clear all candidate history. Call between pipeline sessions."""
        self._candidates.clear()

    def update(
        self,
        detections: List[Detection],
        has_new_detections: bool = True,
    ) -> List[Detection]:
        """
        Ingest one frame of raw YOLO detections; return fused detections.

        Args:
            detections:         Raw detector output for this frame.
            has_new_detections: False when the adaptive cadence skipped YOLO
                                this frame (detections=[]).  Candidates are
                                aged but not evicted so history is preserved.

        Returns:
            List of Detection objects with smoothed boxes and fused confidence,
            ready to pass directly to ObjectTracker.update().
        """
        try:
            return self._update_inner(detections, has_new_detections)
        except Exception:
            logger.exception("[TemporalFusion] Unexpected error — returning raw detections")
            return detections

    def _update_inner(
        self,
        detections: List[Detection],
        has_new_detections: bool,
    ) -> List[Detection]:

        matched_candidate_ids: set[int] = set()   # id(candidate) for matched ones

        if has_new_detections:
            for det in detections:
                best_cand: Optional[_Candidate] = None
                best_iou:  float = FUSION_IOU_MATCH - 1e-9  # must beat threshold

                for cand in self._candidates.get(det.class_name, []):
                    iou_val = _iou(cand, det)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_cand = cand

                if best_cand is not None:
                    best_cand.update(det)
                    matched_candidate_ids.add(id(best_cand))
                else:
                    # New candidate — initialise with first detection
                    cand = _Candidate(
                        class_name=det.class_name,
                        class_id=det.class_id,
                        x1=float(det.x1),
                        y1=float(det.y1),
                        x2=float(det.x2),
                        y2=float(det.y2),
                        frames_seen=1,
                        peak_conf=det.confidence,
                        fused_conf=det.confidence,
                        frames_missed=0,
                    )
                    self._candidates.setdefault(det.class_name, []).append(cand)

            # Age unmatched candidates
            for class_name, cands in self._candidates.items():
                for cand in cands:
                    if id(cand) not in matched_candidate_ids:
                        cand.frames_missed += 1
        else:
            # Non-detection frame — gently age all candidates
            for cands in self._candidates.values():
                for cand in cands:
                    cand.frames_missed += 1

        # Evict stale candidates (missed more than FUSION_HISTORY frames)
        for class_name in list(self._candidates.keys()):
            self._candidates[class_name] = [
                c for c in self._candidates[class_name]
                if c.frames_missed <= FUSION_HISTORY
            ]
            if not self._candidates[class_name]:
                del self._candidates[class_name]

        # Collect passing candidates
        output: List[Detection] = []
        for cands in self._candidates.values():
            for cand in cands:
                if cand.passes_gate:
                    output.append(cand.to_detection())

        return output

    # ── Debug / introspection ─────────────────────────────────────────────────

    def active_candidate_count(self) -> int:
        """Total number of candidates currently tracked."""
        return sum(len(v) for v in self._candidates.values())

    def candidates_by_class(self) -> dict[str, int]:
        """Per-class candidate count (for diagnostics)."""
        return {k: len(v) for k, v in self._candidates.items()}


# Module-level singleton — mirrors the tracker/diagnostics pattern.
temporal_fusion = TemporalFusionFilter()
