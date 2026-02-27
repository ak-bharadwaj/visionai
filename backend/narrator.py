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

MIN_ALERT_LEVEL: dict[str, int] = {
    "person": 4, "car": 4, "motorcycle": 4, "bicycle": 4,
    "bus": 4, "truck": 4, "train": 4,
    "chair": 3, "couch": 3, "dining table": 3, "bench": 3, "door": 3,
}
DEFAULT_MIN_LEVEL = 2

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


class Narrator:
    def prioritize(self, alerts: List[SpatialResult]) -> List[SpatialResult]:
        def should_announce(r: SpatialResult) -> bool:
            max_level = MIN_ALERT_LEVEL.get(r.class_name, DEFAULT_MIN_LEVEL)
            return r.distance_level <= max_level
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
        if msg and getattr(result, "distance_m", 0.0) > 0.1:
            msg += f", approximately {result.distance_m:.1f} metres away"
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
        NEW Feature 2: Social awareness — count persons and generate a context-aware
        message. Called every NAVIGATE cycle when persons are present.
        Returns None if no persons detected.
        """
        persons = [r for r in results if r.class_name == "person"]
        if not persons:
            return None

        count = len(persons)
        closest = min(persons, key=lambda r: r.distance_level)

        if count == 1:
            p = closest
            if p.distance_level == 1:
                return f"Stop! Person directly {p.direction}, very close"
            elif p.distance_level == 2:
                if p.direction == "ahead":
                    return "Person nearby, directly ahead — someone may be addressing you"
                return f"Person nearby, to your {p.direction}"
            else:
                return None  # too far — suppress

        # Multiple persons
        close_count = sum(1 for p in persons if p.distance_level <= 2)
        if close_count >= 2:
            return (
                f"{count} people nearby — {close_count} very close, "
                "someone may be addressing you"
            )
        elif close_count == 1:
            return f"{count} people in scene, one nearby {closest.direction}"
        else:
            return f"{count} people detected, all at a distance"

    def path_clear(self, results: List[SpatialResult]) -> str | None:
        blocking = [r for r in results
                    if r.direction == "ahead" and r.distance_level <= 2]
        return "Path clear ahead" if not blocking else None


narrator = Narrator()
