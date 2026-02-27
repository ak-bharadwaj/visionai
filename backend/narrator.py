"""
narrator.py — Deterministic narration builder for VisionTalk NAVIGATE mode.

Rules (NON-NEGOTIABLE, safety-critical):
  - Templates are fixed strings only — no LLM, no adjectives, no speculation.
  - Only FOUR output templates are permitted:
      "STOP. [Object] approaching, [N] seconds."                 (ETA < 3s)
      "[Object] ahead. [dist]m."                                 (blocking centre)
      "[Object] approaching, [clock direction]. [dist]m."
      "[Object] [clock direction]. [dist]m."                     (static/receding)
  - Distance rounded to 0.1 m.  ETA rounded to nearest whole second (min 1).
  - Direction is the TrackedObject.direction clock string (e.g. "12 o'clock").
  - Confidence qualification: if obj.confidence < CONF_HEDGE_THRESH, non-urgent
    templates are prefixed with "Possible". STOP alerts are NEVER hedged —
    erring toward safety is always correct for imminent collisions.
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
import os

logger = logging.getLogger(__name__)

# Minimum path_overlap_ratio to use "blocking center path" template
BLOCKING_OVERLAP_THRESH = 0.50

# ETA threshold below which the urgent contact template is used (seconds)
ETA_URGENT_THRESH = 3.0

# Confidence below this threshold prefixes non-urgent narrations with "Possible"
# STOP alerts are never hedged — erring toward safety is always correct.

# Maximum number of distinct narrations that may be spoken in a single pipeline
# frame.  A hard cap prevents TTS overload in crowded scenes where many objects
# are simultaneously HIGH/MEDIUM risk.
#
# Priority order within the cap (descending urgency):
#   1. STOP / collision ETA < ETA_URGENT_THRESH  (imminent contact)
#   2. Highest risk_score among remaining HIGH objects
#   3. Highest risk_score among MEDIUM objects
#
# In practice the pipeline currently emits exactly one speak() per navigate
# frame, so this constant acts as a documented safety ceiling and is enforced
# by select_top_n().  Set to 1 for strict single-narration mode; increase only
# if the TTS engine supports queueing without lag.
MAX_NARRATIONS_PER_FRAME = int(os.getenv("MAX_NARRATIONS_PER_FRAME", "2"))
CONF_HEDGE_THRESH = 0.75

# "Path ahead appears clear." literal — exported so pipeline can import it
PATH_CLEAR_MESSAGE = "Path ahead appears clear."


def build(obj) -> str:
    """
    Build a deterministic narration string for a confirmed TrackedObject.

    Template selection (in priority order):
      1. ETA < ETA_URGENT_THRESH and approaching:
             "STOP. [Class] approaching, [N] seconds."
      2. path_overlap_ratio >= BLOCKING_OVERLAP_THRESH:
             "[Class] ahead. [dist]m."
      3. motion_state == "approaching":
             "[Class] approaching, [direction]. [dist]m."
      4. Else (static/receding in path):
             "[Class] [direction]. [dist]m."

    Args:
        obj: TrackedObject with class_name, smoothed_distance_m, direction,
             path_overlap_ratio, motion_state, and collision_eta_s populated.

    Returns:
        Narration string.  Never empty for a valid object.
    """
    try:
        cls_name   = obj.class_name.lower()
        dist_m     = round(obj.smoothed_distance_m, 1)
        dist_str   = f"{dist_m}m"
        direction  = obj.direction           # e.g. "12 o'clock"
        overlap    = float(obj.path_overlap_ratio)
        motion     = obj.motion_state        # "approaching" | "static" | "receding"
        eta        = getattr(obj, "collision_eta_s", 0.0)
        confidence = float(getattr(obj, "confidence", 1.0))

        # Whether to prefix non-urgent narrations with "Possible"
        # STOP alerts are never hedged — erring toward safety is always correct.
        hedge = confidence < CONF_HEDGE_THRESH

        # Template 1 — imminent collision (STOP prefix; never hedged)
        if motion == "approaching" and 0.0 < eta < ETA_URGENT_THRESH:
            eta_s = max(1, round(eta))
            text = f"STOP. {cls_name.capitalize()} approaching, {eta_s} seconds."

        # Template 2 — blocking centre corridor
        elif overlap >= BLOCKING_OVERLAP_THRESH:
            if hedge:
                text = f"Possible {cls_name} ahead. {dist_str}."
            else:
                text = f"{cls_name.capitalize()} ahead. {dist_str}."

        # Template 3 — general approach
        elif motion == "approaching":
            if hedge:
                text = f"Possible {cls_name} approaching, {direction}. {dist_str}."
            else:
                text = f"{cls_name.capitalize()} approaching, {direction}. {dist_str}."

        # Template 4 — static or receding but still qualifies as MEDIUM/HIGH risk
        else:
            if hedge:
                text = f"Possible {cls_name} {direction}. {dist_str}."
            else:
                text = f"{cls_name.capitalize()} {direction}. {dist_str}."

        logger.debug(
            "[Narrator] build id=%d class=%s eta=%.1fs → %r",
            getattr(obj, "id", -1), cls_name, eta, text,
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


def build_multi(objects: list, max_objects: int = 2) -> str | None:
    """
    Build a compound narration for multiple medium-or-higher risk objects
    in different directions.

    Rules:
      - Only combines objects that are MEDIUM or HIGH risk.
      - Only combines when objects are in distinct directions (different clock positions).
      - Returns None if fewer than 2 eligible objects exist (caller falls back to build()).
      - Maximum max_objects objects combined.
      - Format: "[Class] [dir] [dist]m, [class] [dir] [dist]m."

    Args:
        objects : risk-sorted list of confirmed TrackedObjects.
        max_objects : maximum number of objects to combine (default 2).

    Returns:
        Compound narration string, or None.
    """
    try:
        eligible = [
            o for o in objects
            if o.risk_level in ("HIGH", "MEDIUM")
        ]
        if len(eligible) < 2:
            return None

        # Pick objects in distinct directions (clock position string)
        seen_dirs: set = set()
        chosen = []
        for o in eligible:
            if o.direction not in seen_dirs:
                seen_dirs.add(o.direction)
                chosen.append(o)
            if len(chosen) >= max_objects:
                break

        if len(chosen) < 2:
            return None  # all objects in same direction — single narration is better

        parts = []
        for o in chosen:
            dist_m  = round(o.smoothed_distance_m, 1)
            parts.append(f"{o.class_name.lower()} {o.direction} {dist_m}m")

        text = ", ".join(parts) + "."
        text = text[0].upper() + text[1:]   # capitalise first letter
        logger.debug("[Narrator] build_multi %d objects → %r", len(chosen), text)
        return text

    except Exception as exc:
        logger.error("[Narrator] build_multi() error: %s", exc)
        return None


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


def select_top_n(confirmed_objects: list, n: int = MAX_NARRATIONS_PER_FRAME) -> list:
    """
    Return up to n objects eligible for narration this frame, sorted by
    descending priority.

    Priority ordering (same as select_highest_risk, now extended to N items):
      1. Imminent collision (collision_eta_s > 0 and < ETA_URGENT_THRESH) — STOP alerts
      2. HIGH risk objects by descending risk_score, tie-break closest distance
      3. MEDIUM risk objects by descending risk_score, tie-break closest distance

    This function enforces MAX_NARRATIONS_PER_FRAME so the caller never needs
    to worry about TTS overload.  It is a pure function — no side effects.

    Args:
        confirmed_objects : risk-scored confirmed TrackedObject list.
        n                 : max items to return (default MAX_NARRATIONS_PER_FRAME).

    Returns:
        List of TrackedObjects, length <= n, highest priority first.
        Empty list if no HIGH/MEDIUM objects.
    """
    try:
        eligible = [o for o in confirmed_objects if o.risk_level in ("HIGH", "MEDIUM")]
        if not eligible:
            return []

        def _priority_key(o):
            # Lower key = higher priority (for sorting ascending then slicing).
            eta = getattr(o, "collision_eta_s", 0.0)
            imminent = eta > 0.0 and eta < ETA_URGENT_THRESH
            tier = 0 if imminent else (1 if o.risk_level == "HIGH" else 2)
            return (tier, -o.risk_score, o.smoothed_distance_m)

        sorted_eligible = sorted(eligible, key=_priority_key)
        return sorted_eligible[:n]

    except Exception as exc:
        logger.error("[Narrator] select_top_n() error: %s", exc)
        return []


def build_routing(confirmed_objects: list, frame_w: int) -> str | None:
    """
    Build a proactive routing instruction for a blind user based on which
    frame zones are blocked by HIGH/MEDIUM risk objects.

    Frame is divided into three horizontal zones:
      Left   — center_x in [0,      frame_w/3)
      Center — center_x in [frame_w/3, 2*frame_w/3)
      Right  — center_x in [2*frame_w/3, frame_w)

    Logic:
      - Mark zones blocked by any HIGH/MEDIUM risk confirmed object.
      - If center is blocked, recommend the clearest side (left or right).
      - If all zones are blocked, issue a stop/caution message.
      - If center is clear, return None (no routing supplement needed; hazard
        narration already described the off-center hazard).

    Returns:
        A short routing instruction string, or None if center is clear
        (caller can omit the routing supplement entirely).

    Examples:
      "Turn right — left side is clear."
      "Turn left — right side is clear."
      "Stop — obstacles on all sides. Move slowly."
      None   (center clear, no routing needed)
    """
    if not confirmed_objects or frame_w <= 0:
        return None

    try:
        thirds = frame_w / 3.0
        blocked_left   = False
        blocked_center = False
        blocked_right  = False

        for obj in confirmed_objects:
            if obj.risk_level not in ("HIGH", "MEDIUM"):
                continue
            cx = (obj.x1 + obj.x2) / 2.0
            if cx < thirds:
                blocked_left = True
            elif cx < 2 * thirds:
                blocked_center = True
            else:
                blocked_right = True

        if not blocked_center:
            # Centre is free — if there are HIGH-risk objects on the sides, warn the
            # user proactively so they don't walk into an off-axis hazard.
            high_left  = any(
                obj.risk_level == "HIGH" and (obj.x1 + obj.x2) / 2.0 < thirds
                for obj in confirmed_objects if obj.risk_level in ("HIGH", "MEDIUM")
            )
            high_right = any(
                obj.risk_level == "HIGH" and (obj.x1 + obj.x2) / 2.0 >= 2 * thirds
                for obj in confirmed_objects if obj.risk_level in ("HIGH", "MEDIUM")
            )
            if high_left and not high_right:
                return "Obstacle to your left — path ahead clear, continue forward."
            if high_right and not high_left:
                return "Obstacle to your right — path ahead clear, continue forward."
            if high_left and high_right:
                return "Obstacles on both sides — path straight ahead is clear."
            return None   # No high-risk off-center hazards — no routing needed

        if not blocked_left and not blocked_right:
            # Centre blocked, both sides clear — pick right (convention)
            return "Turn right or left — path to either side is clear."

        if not blocked_right:
            return "Turn right — path is clear to your right."

        if not blocked_left:
            return "Turn left — path is clear to your left."

        # All three zones blocked
        return "Stop — obstacles on all sides. Move slowly."

    except Exception as exc:
        logger.error("[Narrator] build_routing() error: %s", exc)
        return None
