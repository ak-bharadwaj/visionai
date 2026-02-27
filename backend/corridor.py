"""
corridor.py — Walking path corridor model for VisionTalk.

Defines a walking corridor as the center 40% of the frame horizontally
(from 30% to 70% of frame width), full height.

For each tracked object, computes:
  - path_overlap_ratio: intersection(box, corridor) / box_area
  - is_blocking: overlap_ratio > BLOCK_THRESHOLD
  - free_space_pct: percentage of corridor height that is estimated clear

All methods are pure functions — no state.
"""

import logging
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.tracker import TrackedObject

import numpy as np

logger = logging.getLogger(__name__)

# Corridor: center 40% of frame width
CORRIDOR_LEFT_RATIO  = 0.30
CORRIDOR_RIGHT_RATIO = 0.70

# An object is "blocking" the corridor if this fraction of its box overlaps
BLOCK_THRESHOLD = 0.30


class CorridorModel:
    """
    Computes path overlap for tracked objects and estimates free space ahead.
    Stateless — call compute() each frame with current tracked objects.
    """

    def compute(
        self,
        tracks: "List[TrackedObject]",
        frame_w: int,
        frame_h: int,
        depth_map: "np.ndarray | None" = None,
    ) -> dict:
        """
        Computes corridor state for this frame.

        Returns:
            {
                corridor_x1: int,
                corridor_x2: int,
                blocking_tracks: List[TrackedObject],   # overlap > BLOCK_THRESHOLD
                free_space_pct: float,                  # 0.0–1.0
                path_clear: bool,                       # True if no blockers
            }
        """
        cx1 = int(frame_w * CORRIDOR_LEFT_RATIO)
        cx2 = int(frame_w * CORRIDOR_RIGHT_RATIO)
        corridor = (cx1, 0, cx2, frame_h)

        blocking = []
        for t in tracks:
            overlap = _box_overlap_ratio(
                (t.x1, t.y1, t.x2, t.y2), corridor
            )
            t.path_overlap_ratio = overlap
            if overlap >= BLOCK_THRESHOLD:
                blocking.append(t)

        # Free space estimate: fraction of corridor vertical slices
        # not significantly occupied by any tracked bounding box.
        free_pct = _estimate_free_space(tracks, cx1, cx2, frame_h)

        return {
            "corridor_x1":    cx1,
            "corridor_x2":    cx2,
            "blocking_tracks": blocking,
            "free_space_pct":  free_pct,
            "path_clear":      len(blocking) == 0,
        }


def _box_overlap_ratio(box: tuple, corridor: tuple) -> float:
    """
    Compute intersection area / box area.

    box, corridor: (x1, y1, x2, y2)
    Returns 0.0 if no overlap or box has zero area.
    """
    bx1, by1, bx2, by2 = box
    cx1, cy1, cx2, cy2 = corridor

    ix1 = max(bx1, cx1)
    iy1 = max(by1, cy1)
    ix2 = min(bx2, cx2)
    iy2 = min(by2, cy2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter_area = inter_w * inter_h

    box_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    if box_area == 0:
        return 0.0
    return inter_area / box_area


def _estimate_free_space(
    tracks: "List[TrackedObject]",
    cx1: int,
    cx2: int,
    frame_h: int,
) -> float:
    """
    Estimate percentage of the corridor that is free (not covered by a bounding box).

    Strategy: scan corridor in vertical slices of 5% height.
    A slice is blocked if any confirmed track's bounding box covers it
    AND the track's y-range intersects the slice.

    Returns float in [0.0, 1.0]. 1.0 = fully clear, 0.0 = fully blocked.
    """
    if frame_h <= 0 or cx2 <= cx1:
        return 1.0

    n_slices = 20  # 20 × 5% = 100% coverage
    slice_h  = frame_h / n_slices
    free     = 0

    for i in range(n_slices):
        sy1 = i * slice_h
        sy2 = sy1 + slice_h
        blocked = False
        for t in tracks:
            # Check if this track overlaps the corridor horizontally
            if t.x2 < cx1 or t.x1 > cx2:
                continue
            # Check if this track overlaps this vertical slice
            if t.y2 < sy1 or t.y1 > sy2:
                continue
            blocked = True
            break
        if not blocked:
            free += 1

    return free / n_slices


# Module-level singleton
corridor_model = CorridorModel()
