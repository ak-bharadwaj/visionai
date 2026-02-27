"""
path_model.py — Walking corridor modelling and free-space estimation.

Responsibilities:
  1. Corridor overlap ratio — what fraction of an object's horizontal span
     falls inside the centre 40% of the frame.  (Already computed per-track
     in tracker.py via _corridor_overlap(); this module re-exposes it as a
     standalone function for use in tests and diagnostics.)

  2. Free-space percentage — what fraction of the bottom-centre floor zone
     is "open" (low depth score = far from camera on the floor plane).
     Used to decide whether to say "Path ahead appears clear."

  3. Floor zone estimation — returns a bool that indicates whether a usable
     floor region exists.

Design constraints (safety-critical):
  - Returns conservative estimates: prefer reporting less free space
    (safer) over reporting more.
  - Never raises; always returns a valid result.
  - No side effects; pure functions.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# ── Corridor definition ─────────────────────────────────────────────────────
# Centre 40% of frame width (30%–70%) is the walking corridor.
CORRIDOR_LEFT_FRAC  = 0.30
CORRIDOR_RIGHT_FRAC = 0.70

# ── Free-space zone definition ──────────────────────────────────────────────
# Bottom 35% of frame, centre 50% horizontally.
FLOOR_ZONE_TOP_FRAC    = 0.65   # top of floor zone as fraction of frame height
FLOOR_ZONE_LEFT_FRAC   = 0.25
FLOOR_ZONE_RIGHT_FRAC  = 0.75

# Depth score below which a pixel is considered "clear" (far = open floor).
# MiDaS: 0 = far, 1 = close. Clear floor = low score.
FREE_SPACE_DEPTH_THRESH = 0.35

# Minimum fraction of the floor zone that must be clear to report "path clear".
FREE_SPACE_CLEAR_THRESHOLD = 0.60


def corridor_overlap(x1: int, x2: int, frame_w: int) -> float:
    """
    Fraction of an object's horizontal span (x1..x2) that lies inside the
    centre walking corridor (CORRIDOR_LEFT_FRAC .. CORRIDOR_RIGHT_FRAC of frame).

    Returns 0.0–1.0.  Returns 0.0 if frame_w <= 0 or object width is zero.
    """
    if frame_w <= 0 or x2 <= x1:
        return 0.0
    c_left  = int(frame_w * CORRIDOR_LEFT_FRAC)
    c_right = int(frame_w * CORRIDOR_RIGHT_FRAC)
    ol      = max(x1, c_left)
    or_     = min(x2, c_right)
    overlap_w = max(0, or_ - ol)
    obj_w     = max(1, x2 - x1)
    return overlap_w / obj_w


def free_space_fraction(depth_map: "np.ndarray | None", frame_h: int, frame_w: int) -> float:
    """
    Estimate the fraction of the floor zone that is clear (open path).

    Args:
        depth_map: float32 array [H, W] with values 0=far, 1=close.
                   May be None if MiDaS not loaded.
        frame_h: frame height in pixels.
        frame_w: frame width in pixels.

    Returns:
        float 0.0–1.0.  Returns 0.0 (conservative: no depth = unknown = not clear)
        if depth_map is None or too small.
    """
    if depth_map is None or frame_h <= 0 or frame_w <= 0:
        return 0.0
    try:
        y0 = int(frame_h * FLOOR_ZONE_TOP_FRAC)
        x0 = int(frame_w * FLOOR_ZONE_LEFT_FRAC)
        x1 = int(frame_w * FLOOR_ZONE_RIGHT_FRAC)
        zone = depth_map[y0:, x0:x1]
        if zone.size == 0:
            return 0.0
        clear_pixels = float(np.sum(zone < FREE_SPACE_DEPTH_THRESH))
        return clear_pixels / zone.size
    except Exception as exc:
        logger.error("[PathModel] free_space_fraction error: %s", exc)
        return 0.0


def is_path_clear(depth_map: "np.ndarray | None", frame_h: int, frame_w: int,
                  confirmed_objects: list) -> bool:
    """
    Return True only when BOTH conditions hold:
      1. No confirmed HIGH or MEDIUM risk object is in the corridor.
      2. free_space_fraction >= FREE_SPACE_CLEAR_THRESHOLD.

    This is the gate for emitting "Path ahead appears clear."

    Args:
        depth_map       : MiDaS depth output or None.
        frame_h, frame_w: frame dimensions.
        confirmed_objects: list of TrackedObjects with risk_level and
                           path_overlap_ratio already populated.
    """
    # Check for corridor-blocking objects
    for obj in confirmed_objects:
        if obj.risk_level in ("HIGH", "MEDIUM") and obj.path_overlap_ratio > 0.3:
            return False

    # Check floor free space
    frac = free_space_fraction(depth_map, frame_h, frame_w)
    return frac >= FREE_SPACE_CLEAR_THRESHOLD


def floor_zone_exists(depth_map: "np.ndarray | None", frame_h: int, frame_w: int) -> bool:
    """
    True if the floor zone contains enough pixels to be meaningful.
    Used to skip stair-drop detection on tiny or malformed frames.
    """
    if depth_map is None or frame_h <= 0 or frame_w <= 0:
        return False
    y0 = int(frame_h * FLOOR_ZONE_TOP_FRAC)
    x0 = int(frame_w * FLOOR_ZONE_LEFT_FRAC)
    x1 = int(frame_w * FLOOR_ZONE_RIGHT_FRAC)
    zone = depth_map[y0:, x0:x1]
    return zone.size >= 64
