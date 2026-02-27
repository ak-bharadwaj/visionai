import numpy as np
from dataclasses import dataclass
from backend.detector import Detection
from backend.depth import depth_estimator


@dataclass
class SpatialResult:
    class_name:     str
    confidence:     float
    direction:      str    # "far left"|"left"|"ahead"|"right"|"far right"
    distance:       str    # "very close"|"nearby"|"ahead"|"far"
    distance_level: int    # 1=very close, 2=nearby, 3=ahead, 4=far
    zone:           str    # "ground"|"mid"|"aerial"
    depth_score:    float
    distance_ft:    float  # approximate distance in feet (0.0 = unknown)
    x1: int; y1: int; x2: int; y2: int

    @property
    def key(self) -> str:
        return f"{self.class_name}_{self.direction}"

    @property
    def bbox_area(self) -> int:
        # ADDED: used by color_sense and any other module needing area
        return (self.x2 - self.x1) * (self.y2 - self.y1)


class SpatialAnalyzer:
    def _depth_to_feet(self, depth_score: float) -> float:
        """
        Piecewise-linear MiDaS depth_score → real-world feet.
        Calibrated for MiDaS "Small" model outputs in indoor/outdoor environments.

        depth_score convention: 0.0 = far/no depth, 1.0 = very close.

        Breakpoints (metres, then × 3.28084):
          score < 0.3 : Far zone    — 5.0 + (1.0 - score) * 5
          0.3 – 0.6   : Mid zone    — 2.0 + (0.6 - score) * 3 / 0.3
          0.6 – 0.8   : Nearby zone — 1.0 + (0.8 - score) * 1 / 0.2
          > 0.8       : Very close  — 0.5 * (1.1 - score) / 0.3
        """
        if depth_score <= 0.0:
            return 0.0  # no depth data available
        if depth_score < 0.3:
            metres = 5.0 + (1.0 - depth_score) * 5.0
        elif depth_score < 0.6:
            metres = 2.0 + (0.6 - depth_score) * 3.0 / 0.3
        elif depth_score < 0.8:
            metres = 1.0 + (0.8 - depth_score) * 1.0 / 0.2
        else:
            metres = max(0.1, 0.5 * (1.1 - depth_score) / 0.3)
        return round(metres * 3.28084, 1)

    def analyze(self, det: Detection, fw: int, fh: int,
                depth_map) -> SpatialResult:
        cx_ratio = (det.x1 + det.x2) / 2 / fw
        if cx_ratio < 0.20:    direction = "far left"
        elif cx_ratio < 0.40:  direction = "left"
        elif cx_ratio < 0.60:  direction = "ahead"
        elif cx_ratio < 0.75:  direction = "right"
        else:                  direction = "far right"

        area_ratio   = ((det.x2-det.x1)*(det.y2-det.y1)) / (fw * fh)
        bottom_ratio = det.y2 / fh

        if area_ratio > 0.15 and bottom_ratio > 0.70:
            level = 1; dist_str = "very close"
        elif area_ratio > 0.06:
            level = 2; dist_str = "nearby"
        elif area_ratio > 0.02:
            level = 3; dist_str = "ahead"
        else:
            level = 4; dist_str = "far"

        d_score = depth_estimator.get_region_depth(
            depth_map, det.x1, det.y1, det.x2, det.y2)
        if d_score > 0.80:
            level = min(level, 1); dist_str = "very close"
        elif d_score > 0.60:
            level = min(level, 2); dist_str = "nearby"

        distance_ft = self._depth_to_feet(d_score)

        top_ratio = det.y1 / fh
        if top_ratio < 0.30:   zone = "aerial"
        elif top_ratio < 0.65: zone = "mid"
        else:                  zone = "ground"

        return SpatialResult(
            class_name=det.class_name, confidence=det.confidence,
            direction=direction, distance=dist_str, distance_level=level,
            zone=zone, depth_score=d_score, distance_ft=distance_ft,
            x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2,
        )


spatial_analyzer = SpatialAnalyzer()
