"""
risk_engine.py — Deterministic risk scoring for VisionTalk NAVIGATE mode.

Risk formula (all weights sum to 1.0):
    risk_score = w1*distance_factor
               + w2*path_overlap_factor
               + w3*velocity_factor
               + w4*object_type_weight

distance_factor   : <1 m=1.0 | 1–2 m=0.7 | 2–3 m=0.4 | >3 m=0.1
path_overlap_factor: corridor overlap ratio (0.0–1.0)
velocity_factor   : approaching=1.0 | static=0.5 | receding=0.2
object_type_weight: person/vehicle=1.0 | furniture=0.8 | small=0.5

Normalised score → risk level:
    HIGH   >= 0.75
    MEDIUM  0.50–0.74
    LOW    < 0.50
    NONE    no depth / unconfirmed (score = 0.0)

Design constraints (safety-critical):
  - Pure function — no side effects.
  - Never raises; any error returns (0.0, "NONE").
  - Deterministic: same inputs always produce the same output.
  - No LLM, no probabilistic reasoning.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.tracker import TrackedObject

logger = logging.getLogger(__name__)

# ── Weights ─────────────────────────────────────────────────────────────────
W_DISTANCE  = 0.40
W_OVERLAP   = 0.25
W_VELOCITY  = 0.20
W_CLASS     = 0.15

assert abs(W_DISTANCE + W_OVERLAP + W_VELOCITY + W_CLASS - 1.0) < 1e-9, \
    "Risk weights must sum to 1.0"

# ── Risk thresholds ──────────────────────────────────────────────────────────
THRESH_HIGH   = 0.75
THRESH_MEDIUM = 0.50

# ── Object type weights ──────────────────────────────────────────────────────
# person / vehicles are highest hazard — always weight 1.0
_HIGH_HAZARD = {
    "person", "car", "motorcycle", "bicycle", "bus", "truck", "train",
    "boat", "horse", "cow", "sheep", "dog", "bear",
}
# Furniture — significant obstruction but rarely fast-moving
_FURNITURE = {
    "chair", "couch", "sofa", "bed", "dining table", "table", "bench",
    "toilet", "refrigerator", "oven", "microwave", "door",
    "suitcase", "backpack",
}


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
    """Return object-type hazard weight."""
    if class_name in _HIGH_HAZARD:
        return 1.0
    if class_name in _FURNITURE:
        return 0.8
    return 0.5   # small / unknown objects


def score(obj: "TrackedObject") -> tuple[float, str]:
    """
    Compute (risk_score, risk_level) for a confirmed TrackedObject.

    Returns (0.0, "NONE") if the object has no usable distance data.

    Args:
        obj: A TrackedObject with smoothed_distance_m, path_overlap_ratio,
             motion_state (property), and class_name populated.

    Returns:
        (score: float 0–1, level: str "HIGH"|"MEDIUM"|"LOW"|"NONE")
    """
    try:
        dist_m = obj.smoothed_distance_m
        if dist_m <= 0.0:
            return 0.0, "NONE"

        df = _distance_factor(dist_m)
        of = float(obj.path_overlap_ratio)          # already 0–1
        vf = _velocity_factor(obj.motion_state)
        cw = _class_weight(obj.class_name)

        raw = (
            W_DISTANCE * df
            + W_OVERLAP * of
            + W_VELOCITY * vf
            + W_CLASS * cw
        )
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
            "[RiskEngine] %s id=%d dist=%.2fm overlap=%.2f motion=%s → "
            "score=%.3f level=%s",
            obj.class_name, obj.id, dist_m, of, obj.motion_state,
            risk_score, level,
        )
        return risk_score, level

    except Exception as exc:
        logger.error("[RiskEngine] score() error for %s: %s", getattr(obj, "class_name", "?"), exc)
        return 0.0, "NONE"


def score_all(objects: list) -> list:
    """
    Apply score() to every object in the list and attach the results
    back onto each TrackedObject (.risk_score, .risk_level).

    Returns the same list sorted by risk_score descending (highest first).
    """
    for obj in objects:
        s, lvl = score(obj)
        obj.risk_score = s
        obj.risk_level = lvl
    return sorted(objects, key=lambda o: o.risk_score, reverse=True)
