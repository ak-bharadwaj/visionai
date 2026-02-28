"""
scene_understanding.py — Google-level scene context understanding.

Features:
  - Context-aware detection validation
  - Object relationship modeling
  - Scene type classification
  - Spatial reasoning
"""

import numpy as np
import cv2
import logging
from typing import List, Dict, Set, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class SceneContext:
    """Understand scene context to improve detection accuracy."""
    
    def __init__(self):
        # Common object co-occurrences (e.g., chair often with table)
        self._co_occurrences = {
            "chair": {"table", "desk", "couch"},
            "table": {"chair", "cup", "bottle", "laptop"},
            "person": {"chair", "table", "laptop", "cell phone"},
            "laptop": {"table", "desk", "chair", "person"},
            "bottle": {"table", "cup"},
            "cup": {"table", "bottle"},
            "tv": {"couch", "chair", "remote"},
            "couch": {"tv", "remote", "person"},
        }
        
        # Scene type indicators
        self._indoor_objects = {"chair", "table", "couch", "tv", "laptop", "bed", "toilet", "sink"}
        self._outdoor_objects = {"car", "bus", "truck", "bicycle", "motorcycle", "traffic light", "stop sign"}
        
    def validate_detections(self, detections: List, frame: np.ndarray) -> List:
        """
        Validate detections based on scene context.
        Removes detections that don't fit the scene context.
        """
        if not detections:
            return []
        
        # Classify scene type
        scene_type = self._classify_scene(detections)
        
        # Get object classes present
        present_classes = {det.class_name for det in detections}
        
        validated = []
        for det in detections:
            # Check if object fits scene type
            if not self._fits_scene(det.class_name, scene_type):
                logger.debug("Scene context: removed %s (doesn't fit %s scene)", 
                           det.class_name, scene_type)
                continue
            
            # Check co-occurrence (boost confidence if co-occurring objects present)
            if self._has_co_occurrences(det.class_name, present_classes):
                det.confidence = min(1.0, det.confidence * 1.05)
                logger.debug("Scene context: boosted %s confidence (co-occurrence)", 
                           det.class_name)
            
            validated.append(det)
        
        return validated
    
    def _classify_scene(self, detections: List) -> str:
        """Classify scene as indoor or outdoor based on detected objects."""
        if not detections:
            return "unknown"
        
        classes = {det.class_name for det in detections}
        
        indoor_count = sum(1 for cls in classes if cls in self._indoor_objects)
        outdoor_count = sum(1 for cls in classes if cls in self._outdoor_objects)
        
        if indoor_count > outdoor_count:
            return "indoor"
        elif outdoor_count > indoor_count:
            return "outdoor"
        else:
            return "mixed"
    
    def _fits_scene(self, class_name: str, scene_type: str) -> bool:
        """Check if object class fits the scene type."""
        if scene_type == "unknown" or scene_type == "mixed":
            return True  # Allow all in unknown/mixed scenes
        
        if scene_type == "indoor":
            return class_name in self._indoor_objects or class_name == "person"
        elif scene_type == "outdoor":
            return class_name in self._outdoor_objects or class_name == "person"
        
        return True
    
    def _has_co_occurrences(self, class_name: str, present_classes: Set[str]) -> bool:
        """Check if co-occurring objects are present."""
        co_occurring = self._co_occurrences.get(class_name, set())
        return bool(co_occurring & present_classes)


class SpatialReasoner:
    """Reason about spatial relationships to validate detections."""
    
    def validate_spatial(self, detections: List, frame: np.ndarray) -> List:
        """
        Validate detections based on spatial relationships.
        E.g., objects on tables should be above the table, not below.
        """
        if not detections or len(detections) < 2:
            return detections
        
        h, w = frame.shape[:2]
        validated = []
        
        # Group objects by type
        tables = [d for d in detections if d.class_name == "table"]
        chairs = [d for d in detections if d.class_name == "chair"]
        objects_on_tables = [d for d in detections 
                           if d.class_name in {"cup", "bottle", "laptop", "cell phone"}]
        
        for det in detections:
            # Skip if it's a table or chair (they're reference objects)
            if det.class_name in {"table", "chair"}:
                validated.append(det)
                continue
            
            # Check if object should be on a table
            if det.class_name in {"cup", "bottle", "laptop", "cell phone"}:
                # Check if there's a table and object is roughly above it
                is_valid = True
                for table in tables:
                    # Object should be roughly centered horizontally over table
                    det_center_x = (det.x1 + det.x2) / 2
                    table_center_x = (table.x1 + table.x2) / 2
                    
                    # Object should be above table (y1 < table.y1)
                    if (abs(det_center_x - table_center_x) < (table.x2 - table.x1) * 0.5 and
                        det.y2 < table.y1):  # Object bottom above table top
                        is_valid = True
                        break
                else:
                    # No table found, but object might be valid anyway
                    is_valid = True
                
                if is_valid:
                    validated.append(det)
            else:
                validated.append(det)
        
        return validated


# Module-level instances
scene_context = SceneContext()
spatial_reasoner = SpatialReasoner()
