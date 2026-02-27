"""
risk_engine.py — Deterministic risk scoring for VisionTalk NAVIGATE mode.

Risk formula (all weights sum to 1.0):
    risk_score = w1*distance_factor
               + w2*path_overlap_factor
               + w3*velocity_factor
               + w4*object_type_weight

distance_factor   : <1 m=1.0 | 1–2 m=0.7 | 2–3 m=0.4 | >3 m=0.1
                    Overridden to 1.0 if bbox fills >SIZE_LARGE_THRESH of frame.
                    For approaching objects, PROJECTED distance is used instead of
                    current distance: proj_dist = max(0.5, dist - velocity * TRAJ_HORIZON_S).
                    This catches fast-approaching hazards before they are immediately close.
path_overlap_factor: corridor overlap ratio (0.0–1.0)
velocity_factor   : approaching=1.0 | static=0.5 | receding=0.2
object_type_weight: person/vehicle=1.0 | table/door/stairs=0.7 | chair=0.6 | small=0.5 | wall/pole=0.4

Age decay (applied after weighting, before thresholding):
    If an object has been static (not approaching) for > AGE_DECAY_ONSET_S seconds,
    its risk_score is multiplied by a decay factor that reaches AGE_DECAY_MIN_FACTOR
    at AGE_DECAY_FULL_S seconds.  Approaching objects are exempt from age decay —
    movement always resets urgency.

    This prevents the narration gate from permanently firing on a chair that
    has sat at 2 m for 30 seconds without moving.

Normalised score → risk level:
    HIGH   >= 0.75
    MEDIUM  0.60–0.74   (narrate; 4 s cooldown)
    LOW    < 0.60       (suppress — do not narrate)
    NONE    no depth / unconfirmed (score = 0.0)

Design constraints (safety-critical):
  - Pure function — no side effects.
  - Never raises; any error returns (0.0, "NONE").
  - Deterministic: same inputs always produce the same output.
  - No LLM, no probabilistic reasoning.
"""

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.tracker import TrackedObject

logger = logging.getLogger(__name__)

# ── Weights ─────────────────────────────────────────────────────────────────
W_DISTANCE  = 0.35
W_OVERLAP   = 0.30
W_VELOCITY  = 0.20
W_CLASS     = 0.15

assert abs(W_DISTANCE + W_OVERLAP + W_VELOCITY + W_CLASS - 1.0) < 1e-9, \
    "Risk weights must sum to 1.0"

# ── Risk thresholds ──────────────────────────────────────────────────────────
THRESH_HIGH   = 0.75
THRESH_MEDIUM = 0.60

# ── Size factor threshold ────────────────────────────────────────────────────
# If an object's bbox occupies more than this fraction of the frame area,
# treat it as if it were within 1 m regardless of depth reading.
SIZE_LARGE_THRESH = 0.20   # 20 % of frame

# ── Trajectory projection ────────────────────────────────────────────────────
# For approaching objects, project position TRAJ_HORIZON_S seconds ahead before
# computing the distance factor.  This catches fast-approaching hazards earlier.
# Projected distance is floored at TRAJ_MIN_DIST_M to avoid negative/zero values.
# Only applied when velocity_m_per_s > 0 (approaching); static/receding objects
# are always scored at their current distance.
TRAJ_HORIZON_S   = 2.0   # look-ahead seconds for approaching objects
TRAJ_MIN_DIST_M  = 0.5   # minimum projected distance (safety floor)

# ── Lateral motion boost ─────────────────────────────────────────────────────
# If an object is moving quickly sideways (lateral_speed_px_per_s > threshold)
# AND is within LATERAL_DIST_THRESH metres, we boost the raw risk score by a
# fixed additive amount.  Lateral crossings at close range are a real collision
# hazard (cyclist cutting across path, person stepping out) that the forward-
# velocity component alone cannot capture.
#
# Threshold chosen empirically:
#   At 30 FPS, a person walking at ~1 m/s at 1 m distance moves ≈ 150 px/s on a
#   640-wide sensor.  100 px/s is a conservative lower bound — covers fast walks
#   and cycling.  Receding lateral motion is also included (object already
#   crossing, still partly in path).
LATERAL_PX_THRESH  = float(os.getenv("LATERAL_PX_THRESH",  "100.0"))  # px/s
LATERAL_DIST_THRESH = float(os.getenv("LATERAL_DIST_THRESH", "2.5"))   # metres
LATERAL_RISK_BOOST  = float(os.getenv("LATERAL_RISK_BOOST",  "0.10"))  # additive

# ── Age decay constants ──────────────────────────────────────────────────────
# After AGE_DECAY_ONSET_S seconds of being static, risk score starts decaying.
# At AGE_DECAY_FULL_S seconds the multiplier reaches AGE_DECAY_MIN_FACTOR.
# Only applies to non-approaching objects — movement always resets urgency.
AGE_DECAY_ONSET_S    = 15.0   # seconds before decay begins
AGE_DECAY_FULL_S     = 60.0   # seconds at which minimum factor is reached
AGE_DECAY_MIN_FACTOR = 0.50   # minimum multiplier (never below 50% of raw score)

# ── Object type weights ──────────────────────────────────────────────────────
# person / vehicles are highest hazard — always weight 1.0
_HIGH_HAZARD = {
    "person", "car", "motorcycle", "bicycle", "bus", "truck", "train",
    "boat", "horse", "cow", "sheep", "dog", "bear",
}
# Moderate obstruction — significant but not as dangerous as vehicles/people
_MODERATE = {
    "table", "door", "stairs",
}
# Static environment features — low intrinsic hazard (spec: wall=0.4, pole=0.4)
_STATIC_FEATURE = {
    "wall", "pole",
}
# NOTE: chair is explicitly 0.6 per spec (between furniture 0.8 and small 0.5)


def _age_decay_factor(track_age_s: float, motion_state: str) -> float:
    """
    Compute a [AGE_DECAY_MIN_FACTOR, 1.0] multiplier based on how long an
    object has been present and whether it is moving.

    Approaching objects always return 1.0 — movement resets urgency.
    Static/receding objects decay linearly from 1.0 (at AGE_DECAY_ONSET_S)
    to AGE_DECAY_MIN_FACTOR (at AGE_DECAY_FULL_S), then remain flat.

    Args:
        track_age_s  : seconds since the track was first created.
        motion_state : "approaching" | "static" | "receding"

    Returns:
        Multiplier in [AGE_DECAY_MIN_FACTOR, 1.0].
    """
    if motion_state == "approaching":
        return 1.0   # moving objects are always full urgency
    if track_age_s <= AGE_DECAY_ONSET_S:
        return 1.0   # too young to decay
    decay_range = AGE_DECAY_FULL_S - AGE_DECAY_ONSET_S
    if decay_range <= 0:
        return AGE_DECAY_MIN_FACTOR
    progress = min(1.0, (track_age_s - AGE_DECAY_ONSET_S) / decay_range)
    return 1.0 - progress * (1.0 - AGE_DECAY_MIN_FACTOR)


def _projected_distance(dist_m: float, velocity_m_per_s: float) -> float:
    """
    Return the distance an approaching object will be at TRAJ_HORIZON_S seconds.

    Only meaningful when velocity_m_per_s > 0 (approaching).  For static/receding
    objects the caller should use the current distance directly.

    The result is floored at TRAJ_MIN_DIST_M so the distance factor never
    receives a zero or negative input.

    Args:
        dist_m           : current smoothed distance in metres.
        velocity_m_per_s : positive = approaching (m/s).

    Returns:
        Projected distance in metres, >= TRAJ_MIN_DIST_M.
    """
    projected = dist_m - velocity_m_per_s * TRAJ_HORIZON_S
    return max(TRAJ_MIN_DIST_M, projected)


def _distance_factor(dist_m: float) -> float:
    """Map smoothed distance in metres to a 0–1 hazard factor."""
    if dist_m <= 0.0:
        return 0.0          # unknown distance — cannot score
    if dist_m < 1.0:
        return 1.0
    if dist_m < 2.0:
        return 0.7
    if dist_m < 3.0:
        return 0.4
    return 0.1


def _velocity_factor(motion_state: str) -> float:
    """Map motion state string to velocity hazard factor."""
    if motion_state == "approaching":
        return 1.0
    if motion_state == "static":
        return 0.5
    return 0.2   # receding


def _class_weight(class_name: str) -> float:
    """Return object-type hazard weight (spec-exact values)."""
    if class_name in _HIGH_HAZARD:
        return 1.0
    if class_name == "chair":
        return 0.6   # spec: chair=0.6 (between moderate and small)
    if class_name in _MODERATE:
        return 0.7   # table, door, stairs
    if class_name in _STATIC_FEATURE:
        return 0.4   # wall, pole — static environment features
    return 0.5       # small / unknown objects


def _size_boosted_distance_factor(dist_m: float,
                                   x1: int, y1: int, x2: int, y2: int,
                                   frame_w: int, frame_h: int,
                                   velocity_m_per_s: float = 0.0) -> float:
    """
    Compute distance factor with a size-based override and trajectory projection.

    If the object's bounding box occupies >= SIZE_LARGE_THRESH of the frame
    area, the distance factor is clamped to 1.0 (treating the object as if
    it were < 1 m away), regardless of the depth estimate.

    For approaching objects (velocity_m_per_s > 0), the projected distance at
    TRAJ_HORIZON_S seconds ahead is used instead of the current distance.  This
    scores fast-approaching hazards more urgently before they arrive.

    Args:
        dist_m           : EMA-smoothed distance in metres.
        x1, y1, x2, y2  : Bounding box coordinates.
        frame_w, frame_h : Frame dimensions in pixels.
        velocity_m_per_s : Current velocity (positive = approaching).

    Returns:
        Float 0.0–1.0 distance factor.
    """
    frame_area = max(1, frame_w * frame_h)
    bbox_area  = max(0, (x2 - x1)) * max(0, (y2 - y1))
    size_ratio = bbox_area / frame_area
    if size_ratio >= SIZE_LARGE_THRESH:
        return 1.0   # object fills frame — treat as very close

    # Use projected distance for approaching objects so we score risk based on
    # where the object will be, not where it is right now.
    if velocity_m_per_s > 0.0:
        effective_dist = _projected_distance(dist_m, velocity_m_per_s)
    else:
        effective_dist = dist_m

    return _distance_factor(effective_dist)


def score(obj: "TrackedObject",
          frame_w: int = 0, frame_h: int = 0) -> tuple[float, str]:
    """
    Compute (risk_score, risk_level) for a confirmed TrackedObject.

    Returns (0.0, "NONE") if the object has no usable distance data.

    Args:
        obj: A TrackedObject with smoothed_distance_m, path_overlap_ratio,
             motion_state (property), velocity_m_per_s, and class_name populated.
        frame_w, frame_h: Frame dimensions used for bbox size factor.
                          If omitted (0), size boosting is skipped.

    Returns:
        (score: float 0–1, level: str "HIGH"|"MEDIUM"|"LOW"|"NONE")
    """
    try:
        dist_m   = obj.smoothed_distance_m
        if dist_m <= 0.0:
            return 0.0, "NONE"

        vel_m_s  = float(getattr(obj, "velocity_m_per_s", 0.0))

        # Distance factor — with size boost and trajectory projection.
        # Both are handled inside _size_boosted_distance_factor; when frame
        # dimensions are not provided (0), we fall through to a direct distance
        # factor with trajectory projection applied manually.
        if frame_w > 0 and frame_h > 0:
            df = _size_boosted_distance_factor(
                dist_m, obj.x1, obj.y1, obj.x2, obj.y2,
                frame_w, frame_h, velocity_m_per_s=vel_m_s,
            )
        else:
            # No bbox size data — still apply trajectory projection for consistency.
            if vel_m_s > 0.0:
                eff_dist = _projected_distance(dist_m, vel_m_s)
            else:
                eff_dist = dist_m
            df = _distance_factor(eff_dist)

        of = float(obj.path_overlap_ratio)          # already 0–1
        vf = _velocity_factor(obj.motion_state)
        cw = _class_weight(obj.class_name)

        raw = (
            W_DISTANCE * df
            + W_OVERLAP * of
            + W_VELOCITY * vf
            + W_CLASS * cw
        )

        # Lateral motion boost — an object moving quickly sideways at close
        # range is a crossing hazard even when it isn't approaching directly.
        # Boost is additive and applied before age decay so that a fresh lateral
        # crossing always gets full urgency.
        lat_px_s = float(getattr(obj, "lateral_speed_px_per_s", 0.0))
        if lat_px_s >= LATERAL_PX_THRESH and dist_m <= LATERAL_DIST_THRESH:
            raw = min(1.0, raw + LATERAL_RISK_BOOST)
            logger.debug(
                "[RiskEngine] lateral boost +%.2f applied to %s id=%d "
                "(lat=%.0fpx/s dist=%.2fm)",
                LATERAL_RISK_BOOST, obj.class_name, obj.id, lat_px_s, dist_m,
            )

        # Age decay — reduce priority of objects that have been static a long time.
        # Approaching objects are always exempt.
        age_s = float(getattr(obj, "track_age_seconds", 0.0))
        decay = _age_decay_factor(age_s, obj.motion_state)
        raw   = raw * decay

        # Clamp to [0, 1] (weights already sum to 1 so raw is bounded,
        # but floating-point drift is possible)
        risk_score = min(1.0, max(0.0, raw))

        if risk_score >= THRESH_HIGH:
            level = "HIGH"
        elif risk_score >= THRESH_MEDIUM:
            level = "MEDIUM"
        else:
            level = "LOW"

        logger.debug(
            "[RiskEngine] %s id=%d dist=%.2fm vel=%.2fm/s overlap=%.2f motion=%s "
            "age=%.1fs decay=%.2f → score=%.3f level=%s",
            obj.class_name, obj.id, dist_m, vel_m_s, of, obj.motion_state,
            age_s, decay, risk_score, level,
        )
        return risk_score, level

    except Exception as exc:
        logger.error("[RiskEngine] score() error for %s: %s", getattr(obj, "class_name", "?"), exc)
        return 0.0, "NONE"


def score_all(objects: list,
              frame_w: int = 0, frame_h: int = 0) -> list:
    """
    Apply score() to every object in the list and attach the results
    back onto each TrackedObject (.risk_score, .risk_level).

    Returns the same list sorted by risk_score descending (highest first).
    """
    for obj in objects:
        s, lvl = score(obj, frame_w=frame_w, frame_h=frame_h)
        obj.risk_score = s
        obj.risk_level = lvl
    return sorted(objects, key=lambda o: o.risk_score, reverse=True)
