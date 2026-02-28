"""
google_level_postprocessing.py — Google-level post-processing.

Advanced post-processing techniques for maximum accuracy:
- Confidence calibration
- Size-based filtering
- Aspect ratio validation
- Spatial consistency
"""

import numpy as np
import logging
from typing import List
from backend.detector import Detection

logger = logging.getLogger(__name__)


class GoogleLevelPostProcessor:
    """Google-level post-processor for maximum accuracy."""
    
    def __init__(self):
        self._min_area = 100  # Minimum bbox area in pixels
        self._max_area_ratio = 0.8  # Maximum bbox area as ratio of frame
        self._min_aspect_ratio = 0.1  # Minimum width/height ratio
        self._max_aspect_ratio = 10.0  # Maximum width/height ratio
    
    def filter_detections(self, detections: List[Detection], 
                         frame_w: int, frame_h: int) -> List[Detection]:
        """
        Filter detections using Google-level criteria.
        
        Filters:
        1. Minimum area (too small = noise)
        2. Maximum area (too large = likely error)
        3. Aspect ratio (unrealistic shapes = likely error)
        4. Boundary checks (must be within frame)
        """
        if not detections:
            return detections
        
        filtered = []
        frame_area = frame_w * frame_h
        min_area_pixels = self._min_area
        max_area_pixels = int(frame_area * self._max_area_ratio)
        
        for det in detections:
            # Check area
            area = det.bbox_area
            if area < min_area_pixels or area > max_area_pixels:
                continue
            
            # Check aspect ratio
            width = det.x2 - det.x1
            height = det.y2 - det.y1
            if height == 0:
                continue
            aspect_ratio = width / height
            if aspect_ratio < self._min_aspect_ratio or aspect_ratio > self._max_aspect_ratio:
                continue
            
            # Check boundaries
            if det.x1 < 0 or det.y1 < 0 or det.x2 > frame_w or det.y2 > frame_h:
                # Clip to boundaries
                det.x1 = max(0, min(det.x1, frame_w - 1))
                det.y1 = max(0, min(det.y1, frame_h - 1))
                det.x2 = max(det.x1 + 1, min(det.x2, frame_w))
                det.y2 = max(det.y1 + 1, min(det.y2, frame_h))
            
            filtered.append(det)
        
        return filtered
    
    def calibrate_confidence(self, detections: List[Detection]) -> List[Detection]:
        """
        Calibrate confidence scores using Google-level techniques.
        
        Adjustments:
        - Boost confidence for large objects
        - Boost confidence for objects near center
        - Reduce confidence for edge objects
        """
        if not detections:
            return detections
        
        calibrated = []
        for det in detections:
            # Size-based boost (larger objects are more reliable)
            area_ratio = det.bbox_area / (640 * 480)  # Normalized to standard frame
            size_boost = min(1.2, 1.0 + area_ratio * 0.2)
            
            # Center-based boost (center objects are more reliable)
            center_x_ratio = det.center_x / 640  # Normalized
            center_y_ratio = det.center_y / 480
            center_dist = np.sqrt((center_x_ratio - 0.5)**2 + (center_y_ratio - 0.5)**2)
            center_boost = 1.0 + (1.0 - center_dist) * 0.1  # Up to 10% boost at center
            
            # Apply boosts
            new_conf = det.confidence * size_boost * center_boost
            new_conf = min(1.0, new_conf)  # Cap at 1.0
            
            # Create new detection with calibrated confidence
            calibrated_det = Detection(
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=new_conf,
                x1=det.x1,
                y1=det.y1,
                x2=det.x2,
                y2=det.y2,
            )
            calibrated.append(calibrated_det)
        
        return calibrated


# Global instance
google_level_postprocessor = GoogleLevelPostProcessor()
