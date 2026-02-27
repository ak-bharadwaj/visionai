"""
narrator.py — Deterministic narration builder for VisionTalk NAVIGATE mode.

Rules (NON-NEGOTIABLE, safety-critical):
  - Templates are fixed strings only — no LLM, no adjectives, no speculation.
  - Only THREE output templates are permitted:
      "[Object] approaching from [clock direction], [distance] metres."
      "[Object] blocking center path, [distance] metres ahead."
      "Path ahead appears clear."
  - Distance rounded to 0.1 m.
  - Direction is the TrackedObject.direction clock string (e.g. "12 o'clock").
  - Narration gate (ALL required before calling build()):
      • obj.confirmed == True
      • obj.confidence >= 0.60
      • obj.smoothed_distance_m > 0.0
      • obj.distance_variance() <= DIST_VARIANCE_GATE
      • obj.risk_level in ("HIGH", "MEDIUM")
    — The stability_filter module enforces this gate; narrator trusts the caller.
  - Never returns an empty string for a valid HIGH/MEDIUM object (fallback exists).
  - No state — pure functions only.

This module does NOT contain cooldown logic (that lives in stability_filter.py).
"""

import logging

logger = logging.getLogger(__name__)

# Minimum path_overlap_ratio to use "blocking center path" template
BLOCKING_OVERLAP_THRESH = 0.50

# "Path ahead appears clear." literal — exported so pipeline can import it
PATH_CLEAR_MESSAGE = "Path ahead appears clear."


def build(obj) -> str:
    """
    Build a deterministic narration string for a confirmed TrackedObject.

    Template selection:
      - If path_overlap_ratio >= BLOCKING_OVERLAP_THRESH:
            "[Class] blocking center path, [dist] metres ahead."
      - Elif motion_state == "approaching":
            "[Class] approaching from [direction], [dist] metres."
      - Else (static/receding in path):
            "[Class] ahead, [dist] metres."   (minimal alert, still deterministic)

    Args:
        obj: TrackedObject with class_name, smoothed_distance_m, direction,
             path_overlap_ratio, and motion_state populated.

    Returns:
        Narration string.  Never empty for a valid object.
    """
    try:
        cls_name = obj.class_name.lower()
        dist_m   = round(obj.smoothed_distance_m, 1)
        dist_str = f"{dist_m} metres"
        direction = obj.direction           # e.g. "12 o'clock"
        overlap   = float(obj.path_overlap_ratio)
        motion    = obj.motion_state        # "approaching" | "static" | "receding"

        if overlap >= BLOCKING_OVERLAP_THRESH:
            text = f"{cls_name.capitalize()} blocking center path, {dist_str} ahead."
        elif motion == "approaching":
            text = f"{cls_name.capitalize()} approaching from {direction}, {dist_str}."
        else:
            # Static or receding but still qualifies as MEDIUM/HIGH risk
            text = f"{cls_name.capitalize()} at {direction}, {dist_str}."

        logger.debug(
            "[Narrator] build id=%d class=%s → %r",
            getattr(obj, "id", -1), cls_name, text,
        )
        return text

    except Exception as exc:
        logger.error("[Narrator] build() error: %s", exc)
        # Absolute fallback — still deterministic, never empty
        cls_name = getattr(obj, "class_name", "object")
        return f"{cls_name.capitalize()} detected nearby."


def path_clear() -> str:
    """Return the canonical 'path clear' message."""
    return PATH_CLEAR_MESSAGE


def select_highest_risk(confirmed_objects: list):
    """
    From a list of confirmed TrackedObjects (already risk-scored),
    return the single highest-risk object to narrate this cycle.

    Priority:
      1. Highest risk_score (already computed by risk_engine.score_all()).
      2. Tie-break: closest distance (lowest smoothed_distance_m).

    Returns None if the list is empty.
    """
    eligible = [o for o in confirmed_objects if o.risk_level in ("HIGH", "MEDIUM")]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda o: (-o.risk_score, o.smoothed_distance_m),
    )
