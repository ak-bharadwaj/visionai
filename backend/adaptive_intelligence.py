"""
adaptive_intelligence.py — Google-level adaptive learning and optimization.

Features:
  - Adaptive confidence thresholds
  - Scene-specific optimization
  - Learning from feedback
  - Dynamic parameter adjustment
  - Context-aware detection
"""

import logging
import threading
from collections import defaultdict, deque
from typing import Dict, List, Optional
import time

logger = logging.getLogger(__name__)


class AdaptiveThresholds:
    """Adaptively adjust confidence thresholds based on scene and performance."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._base_thresholds: Dict[str, float] = {
            "person": 0.35,
            "chair": 0.30,
            "table": 0.30,
            "car": 0.40,
            "bottle": 0.25,
            "cup": 0.25,
            "laptop": 0.35,
        }
        self._adaptive_thresholds: Dict[str, float] = self._base_thresholds.copy()
        self._performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
    
    def get_threshold(self, class_name: str, scene_type: str = "unknown") -> float:
        """Get adaptive threshold for a class."""
        with self._lock:
            base = self._adaptive_thresholds.get(class_name, 0.35)
            
            # Adjust based on scene type
            if scene_type == "indoor":
                # Indoor objects can use lower thresholds
                if class_name in {"chair", "table", "laptop", "bottle", "cup"}:
                    return base * 0.9
            elif scene_type == "outdoor":
                # Outdoor objects need higher thresholds
                if class_name in {"car", "bus", "truck", "bicycle"}:
                    return base * 1.1
            
            return base
    
    def update_from_performance(self, class_name: str, 
                               false_positive_rate: float,
                               false_negative_rate: float):
        """Update threshold based on performance metrics."""
        with self._lock:
            current = self._adaptive_thresholds.get(class_name, 0.35)
            
            # If too many false positives, increase threshold
            if false_positive_rate > 0.15:
                new_threshold = min(0.6, current * 1.1)
                self._adaptive_thresholds[class_name] = new_threshold
                logger.debug(
                    "Adaptive threshold: %s increased to %.2f (FPR: %.2f)",
                    class_name, new_threshold, false_positive_rate
                )
            # If too many false negatives, decrease threshold
            elif false_negative_rate > 0.20:
                new_threshold = max(0.2, current * 0.9)
                self._adaptive_thresholds[class_name] = new_threshold
                logger.debug(
                    "Adaptive threshold: %s decreased to %.2f (FNR: %.2f)",
                    class_name, new_threshold, false_negative_rate
                )


class SceneAdaptiveOptimizer:
    """Optimize detection parameters based on scene characteristics."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._scene_profiles: Dict[str, Dict] = {
            "indoor": {
                "detection_cadence": 2,
                "ocr_frequency": 5,
                "depth_frequency": 3,
                "preferred_classes": {"chair", "table", "person", "laptop", "bottle"},
            },
            "outdoor": {
                "detection_cadence": 3,
                "ocr_frequency": 10,
                "depth_frequency": 5,
                "preferred_classes": {"car", "person", "bicycle", "bus"},
            },
            "mixed": {
                "detection_cadence": 3,
                "ocr_frequency": 7,
                "depth_frequency": 4,
                "preferred_classes": set(),
            },
        }
        self._current_scene = "unknown"
    
    def get_optimal_cadence(self, scene_type: str) -> int:
        """Get optimal detection cadence for scene."""
        with self._lock:
            profile = self._scene_profiles.get(scene_type, self._scene_profiles["mixed"])
            return profile["detection_cadence"]
    
    def get_optimal_ocr_frequency(self, scene_type: str) -> int:
        """Get optimal OCR frequency for scene."""
        with self._lock:
            profile = self._scene_profiles.get(scene_type, self._scene_profiles["mixed"])
            return profile["ocr_frequency"]
    
    def update_scene(self, scene_type: str):
        """Update current scene type."""
        with self._lock:
            if scene_type != self._current_scene:
                logger.info("Scene adaptive: switched to %s", scene_type)
                self._current_scene = scene_type


class FeedbackLearner:
    """Learn from user feedback to improve accuracy."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._feedback_history: deque = deque(maxlen=1000)
        self._corrections: Dict[str, List] = defaultdict(list)
    
    def record_feedback(self, detection_id: str, class_name: str,
                       was_correct: bool, user_correction: Optional[str] = None):
        """Record user feedback on a detection."""
        with self._lock:
            self._feedback_history.append({
                "detection_id": detection_id,
                "class_name": class_name,
                "was_correct": was_correct,
                "user_correction": user_correction,
                "timestamp": time.time(),
            })
            
            if not was_correct and user_correction:
                self._corrections[class_name].append(user_correction)
                logger.debug(
                    "Feedback learner: %s corrected to %s",
                    class_name, user_correction
                )
    
    def get_correction_suggestions(self, class_name: str) -> List[str]:
        """Get common corrections for a class."""
        with self._lock:
            return self._corrections.get(class_name, [])


class ContextAwareDetector:
    """Context-aware detection adjustments."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._context_history: deque = deque(maxlen=30)
    
    def adjust_confidence(self, detection, context: Dict) -> float:
        """Adjust detection confidence based on context."""
        confidence = detection.confidence
        
        # Boost confidence if object appears in expected context
        scene_objects = context.get("scene_objects", set())
        if detection.class_name in self._get_expected_objects(scene_objects):
            confidence = min(1.0, confidence * 1.1)
        
        # Reduce confidence if object is unexpected
        if detection.class_name in self._get_unexpected_objects(scene_objects):
            confidence = confidence * 0.9
        
        return confidence
    
    def _get_expected_objects(self, scene_objects: set) -> set:
        """Get objects expected in current scene."""
        expected = set()
        if "table" in scene_objects:
            expected.update({"cup", "bottle", "laptop", "cell phone"})
        if "chair" in scene_objects:
            expected.update({"person", "laptop"})
        if "tv" in scene_objects:
            expected.update({"couch", "chair", "remote"})
        return expected
    
    def _get_unexpected_objects(self, scene_objects: set) -> set:
        """Get objects unexpected in current scene."""
        unexpected = set()
        if "car" in scene_objects or "bus" in scene_objects:
            # Outdoor scene
            unexpected.update({"chair", "table", "laptop", "tv"})
        else:
            # Indoor scene
            unexpected.update({"car", "bus", "truck", "traffic light"})
        return unexpected


# Module-level instances
adaptive_thresholds = AdaptiveThresholds()
scene_optimizer = SceneAdaptiveOptimizer()
feedback_learner = FeedbackLearner()
context_aware = ContextAwareDetector()
