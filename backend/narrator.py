from backend.spatial import SpatialResult
from typing import List

RISK_SCORE: dict[str, int] = {
    "person": 10, "car": 10, "motorcycle": 10, "bicycle": 10,
    "bus": 10, "truck": 10, "train": 10,
    "chair": 7, "couch": 7, "dining table": 6, "bench": 6,
    "door": 5, "suitcase": 5, "bed": 5,
    "toilet": 4, "sink": 4, "refrigerator": 4,
}
DEFAULT_RISK = 3

# FIXED: "{dir}" is always used with preposition now
# "ahead" → "Chair ahead, very close"
# "left"  → "Chair nearby, to your left"
NAVIGATE_TEMPLATES: dict[int, str] = {
    1: "Stop! {Cls} directly {dir}",
    2: "{Cls} nearby, to your {dir}",
    3: "{Cls} to your {dir}, a few steps away",
    4: "",       # suppress — too far, not useful
}
# Special cases where "ahead" sounds better without "to your"
AHEAD_TEMPLATES: dict[int, str] = {
    1: "Stop! {Cls} directly ahead",
    2: "{Cls} nearby, directly ahead",
    3: "{Cls} ahead, a few steps away",
    4: "",
}

APPROACH_TEMPLATES: dict[str, str] = {
    "person":   "Warning! Person approaching from {dir}",
    "car":      "Warning! Car approaching from {dir}",
    "bicycle":  "Warning! Bicycle approaching from {dir}",
    "default":  "Warning! {Cls} approaching from {dir}",
}


DANGER_ALERT_AT_2 = {"person", "car", "motorcycle", "bicycle", "bus", "truck"}


class Narrator:
    def prioritize(self, alerts: List[SpatialResult]) -> List[SpatialResult]:
        def should_announce(r: SpatialResult) -> bool:
            if r.distance_level == 1:
                return True
            if r.distance_level == 2 and r.class_name in DANGER_ALERT_AT_2:
                return True
            return False
        filtered = [r for r in alerts if should_announce(r)]
        return sorted(
            filtered,
            key=lambda r: (-RISK_SCORE.get(r.class_name, DEFAULT_RISK), r.distance_level)
        )

    def narrate(self, result: SpatialResult) -> str:
        """
        FIXED: use AHEAD_TEMPLATES when direction == 'ahead' (sounds natural).
        Use NAVIGATE_TEMPLATES with 'to your {dir}' for left/right.
        Appends zone qualifier when object is aerial (overhead hazard).
        Appends approximate distance in metres when depth data is available.
        """
        if result.direction == "ahead":
            tmpl = AHEAD_TEMPLATES.get(result.distance_level, "")
        else:
            tmpl = NAVIGATE_TEMPLATES.get(result.distance_level, "")

        if not tmpl:
            return ""
        msg = tmpl.format(
            cls=result.class_name,
            Cls=result.class_name.title(),
            dir=result.direction,
        )
        # Append zone qualifier for overhead objects (ground-level is implicit)
        if result.zone == "aerial" and msg:
            msg += ", overhead"
        # Append metric distance when depth data is meaningful (> 0.1 m)
        # SpatialResult only has distance_ft — convert to metres here.
        dist_ft = getattr(result, "distance_ft", 0.0) or 0.0
        dist_m  = dist_ft * 0.3048
        if msg and dist_m > 0.1:
            msg += f", approximately {dist_m:.1f} metres away"
        return msg

    def narrate_approaching(self, result: SpatialResult) -> str:
        """NEW: Generate approaching alert string."""
        tmpl = APPROACH_TEMPLATES.get(result.class_name,
                                       APPROACH_TEMPLATES["default"])
        return tmpl.format(
            cls=result.class_name,
            Cls=result.class_name.title(),
            dir=result.direction,
        )

    def narrate_persons(self, results: List[SpatialResult]) -> str | None:
        """
        Social awareness — only report persons at level 1 (very close).
        Returns None if no persons are very close.
        """
        persons = [r for r in results if r.class_name == "person" and r.distance_level == 1]
        if not persons:
            return None

        closest = min(persons, key=lambda r: r.distance_level)
        count = len(persons)

        if count == 1:
            p = closest
            if p.direction == "ahead":
                return "Stop! Person directly ahead, very close"
            return f"Stop! Person to your {p.direction}, very close"

        # Multiple very-close persons
        return f"Stop! {count} people very close"


narrator = Narrator()
