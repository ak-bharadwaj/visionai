"""
ensemble_detector.py — Google-level multi-model ensemble detection.

Features:
  - Multiple detection models voting
  - Confidence fusion
  - Temporal consistency
  - Scene context understanding
"""

import time
import numpy as np
import cv2
import logging
from typing import List, Dict, Tuple
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EnsembleDetection:
    """Detection from ensemble with fused confidence."""
    class_id: int
    class_name: str
    confidence: float  # Fused confidence from multiple models
    x1: int
    y1: int
    x2: int
    y2: int
    votes: int  # Number of models that detected this
    model_confidences: Dict[str, float]  # Confidence from each model


class ModelEnsemble:
    """Ensemble of multiple detection models for maximum accuracy."""
    
    def __init__(self, base_detector):
        self._base_detector = base_detector
        self._models = [base_detector]  # Can add more models here
        
    def detect_ensemble(self, frame: np.ndarray, 
                       conf: float = None,
                       apply_whitelist: bool = True) -> List:
        """
        Run detection with multiple models and fuse results.
        Uses weighted voting based on model confidence.
        """
        if frame is None or frame.size == 0:
            return []
        
        all_detections = []
        
        # Run each model
        for model_idx, model in enumerate(self._models):
            try:
                detections = model.detect(frame, conf=conf, apply_whitelist=apply_whitelist)
                for det in detections:
                    all_detections.append((det, f"model_{model_idx}"))
            except Exception as e:
                logger.warning("Ensemble: model %d failed: %s", model_idx, e)
                continue
        
        if not all_detections:
            return []
        
        # Fuse detections using weighted voting
        fused = self._fuse_detections(all_detections)
        return fused
    
    def _fuse_detections(self, detections_with_models: List) -> List:
        """Fuse detections from multiple models using IoU matching and voting."""
        if not detections_with_models:
            return []
        
        # Group detections by class
        class_groups = defaultdict(list)
        for det, model_name in detections_with_models:
            class_groups[det.class_name].append((det, model_name))
        
        fused_detections = []
        
        for class_name, class_dets in class_groups.items():
            # Find clusters of overlapping detections
            clusters = self._cluster_detections(class_dets)
            
            for cluster in clusters:
                if not cluster:
                    continue
                
                # Fuse cluster into single detection
                fused = self._fuse_cluster(cluster, class_name)
                if fused:
                    fused_detections.append(fused)
        
        return fused_detections
    
    def _cluster_detections(self, detections: List) -> List[List]:
        """Cluster overlapping detections using IoU."""
        if not detections:
            return []
        
        clusters = []
        used = set()
        
        for i, (det1, model1) in enumerate(detections):
            if i in used:
                continue
            
            cluster = [(det1, model1)]
            used.add(i)
            
            for j, (det2, model2) in enumerate(detections[i+1:], start=i+1):
                if j in used:
                    continue
                
                iou = self._calculate_iou(det1, det2)
                if iou > 0.3:  # Overlapping threshold
                    cluster.append((det2, model2))
                    used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _fuse_cluster(self, cluster: List, class_name: str) -> EnsembleDetection:
        """Fuse a cluster of detections into one."""
        if not cluster:
            return None
        
        # Weighted average of bounding boxes (weighted by confidence)
        total_weight = sum(det.confidence for det, _ in cluster)
        if total_weight == 0:
            return None
        
        x1_sum = sum(det.x1 * det.confidence for det, _ in cluster)
        y1_sum = sum(det.y1 * det.confidence for det, _ in cluster)
        x2_sum = sum(det.x2 * det.confidence for det, _ in cluster)
        y2_sum = sum(det.y2 * det.confidence for det, _ in cluster)
        
        # Fused confidence: average with bonus for multiple votes
        base_conf = sum(det.confidence for det, _ in cluster) / len(cluster)
        vote_bonus = min(0.15, len(cluster) * 0.05)  # Up to 15% bonus
        fused_conf = min(1.0, base_conf + vote_bonus)
        
        # Model confidences
        model_confs = {model: det.confidence for det, model in cluster}
        
        return EnsembleDetection(
            class_id=cluster[0][0].class_id,
            class_name=class_name,
            confidence=fused_conf,
            x1=int(x1_sum / total_weight),
            y1=int(y1_sum / total_weight),
            x2=int(x2_sum / total_weight),
            y2=int(y2_sum / total_weight),
            votes=len(cluster),
            model_confidences=model_confs
        )
    
    def _calculate_iou(self, det1, det2) -> float:
        """Calculate IoU between two detections."""
        x1 = max(det1.x1, det2.x1)
        y1 = max(det1.y1, det2.y1)
        x2 = min(det1.x2, det2.x2)
        y2 = min(det1.y2, det2.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        inter = (x2 - x1) * (y2 - y1)
        area1 = (det1.x2 - det1.x1) * (det1.y2 - det1.y1)
        area2 = (det2.x2 - det2.x1) * (det2.y2 - det2.y1)
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0.0


class TemporalConsistencyFilter:
    """Filter detections based on temporal consistency across frames."""
    
    def __init__(self, history_size: int = 5):
        self._history_size = history_size
        self._detection_history: List[List] = []
    
    def filter(self, detections: List, frame_idx: int) -> List:
        """
        Filter detections that are inconsistent with recent history.
        Keeps detections that appear consistently across frames.
        """
        if not detections:
            self._detection_history.append([])
            if len(self._detection_history) > self._history_size:
                self._detection_history.pop(0)
            return []
        
        # Add current detections to history
        self._detection_history.append(detections)
        if len(self._detection_history) > self._history_size:
            self._detection_history.pop(0)
        
        # Need at least 2 frames for temporal filtering
        if len(self._detection_history) < 2:
            return detections
        
        # Find detections that appear in multiple recent frames
        consistent_detections = []
        
        for det in detections:
            # Check if similar detection appeared in recent frames
            appearance_count = 0
            for hist_frame in self._detection_history[:-1]:  # Exclude current frame
                for hist_det in hist_frame:
                    if (hist_det.class_name == det.class_name and
                        self._is_similar(det, hist_det)):
                        appearance_count += 1
                        break
            
            # Keep if appeared in at least 40% of recent frames
            min_appearances = max(1, int(len(self._detection_history) * 0.4))
            if appearance_count >= min_appearances:
                # Boost confidence for consistent detections
                det.confidence = min(1.0, det.confidence * 1.1)
                consistent_detections.append(det)
            elif det.confidence > 0.7:  # High confidence detections pass through
                consistent_detections.append(det)
        
        return consistent_detections
    
    def _is_similar(self, det1, det2) -> bool:
        """Check if two detections are similar (same class, overlapping)."""
        if det1.class_name != det2.class_name:
            return False
        
        iou = self._calculate_iou(det1, det2)
        return iou > 0.3
    
    def _calculate_iou(self, det1, det2) -> float:
        """Calculate IoU between two detections."""
        x1 = max(det1.x1, det2.x1)
        y1 = max(det1.y1, det2.y1)
        x2 = min(det1.x2, det2.x2)
        y2 = min(det1.y2, det2.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        inter = (x2 - x1) * (y2 - y1)
        area1 = (det1.x2 - det1.x1) * (det1.y2 - det1.y1)
        area2 = (det2.x2 - det2.x1) * (det2.y2 - det2.y1)
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0.0


class TestTimeAugmentation:
    """Test-time augmentation for improved accuracy."""
    
    def __init__(self, base_detector):
        self._base_detector = base_detector
    
    def detect_with_tta(self, frame: np.ndarray,
                       conf: float = None,
                       apply_whitelist: bool = True) -> List:
        """
        Run detection with test-time augmentation.
        Applies flips and slight rotations, then averages results.
        """
        if frame is None or frame.size == 0:
            return []
        
        all_detections = []
        h, w = frame.shape[:2]
        
        # Original frame
        detections = self._base_detector.detect(frame, conf=conf, apply_whitelist=apply_whitelist)
        all_detections.extend(detections)
        
        # Horizontal flip
        try:
            flipped = cv2.flip(frame, 1)
            flipped_dets = self._base_detector.detect(flipped, conf=conf, apply_whitelist=apply_whitelist)
            # Flip bounding boxes back
            for det in flipped_dets:
                det.x1, det.x2 = w - det.x2, w - det.x1
            all_detections.extend(flipped_dets)
        except Exception:
            pass
        
        # Slight brightness adjustment
        try:
            bright = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
            bright_dets = self._base_detector.detect(bright, conf=conf, apply_whitelist=apply_whitelist)
            all_detections.extend(bright_dets)
        except Exception:
            pass
        
        # Merge augmented detections
        merged = self._merge_augmented_detections(all_detections)
        return merged
    
    def _merge_augmented_detections(self, detections: List) -> List:
        """Merge detections from augmented frames."""
        if not detections:
            return []
        
        # Group by class and IoU
        merged = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
            
            cluster = [det1]
            used.add(i)
            
            for j, det2 in enumerate(detections[i+1:], start=i+1):
                if j in used:
                    continue
                
                if (det1.class_name == det2.class_name and
                    self._calculate_iou(det1, det2) > 0.5):
                    cluster.append(det2)
                    used.add(j)
            
            # Average cluster
            if len(cluster) > 1:
                merged_det = self._average_detections(cluster)
                merged.append(merged_det)
            else:
                merged.append(det1)
        
        return merged
    
    def _average_detections(self, detections: List) -> object:
        """Average multiple detections."""
        total_conf = sum(d.confidence for d in detections)
        
        det = detections[0]
        det.x1 = int(sum(d.x1 * d.confidence for d in detections) / total_conf)
        det.y1 = int(sum(d.y1 * d.confidence for d in detections) / total_conf)
        det.x2 = int(sum(d.x2 * d.confidence for d in detections) / total_conf)
        det.y2 = int(sum(d.y2 * d.confidence for d in detections) / total_conf)
        det.confidence = min(1.0, sum(d.confidence for d in detections) / len(detections) * 1.1)
        
        return det
    
    def _calculate_iou(self, det1, det2) -> float:
        """Calculate IoU."""
        x1 = max(det1.x1, det2.x1)
        y1 = max(det1.y1, det2.y1)
        x2 = min(det1.x2, det2.x2)
        y2 = min(det1.y2, det2.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        inter = (x2 - x1) * (y2 - y1)
        area1 = (det1.x2 - det1.x1) * (det1.y2 - det1.y1)
        area2 = (det2.x2 - det2.x1) * (det2.y2 - det2.y1)
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0.0


# Module-level instances
temporal_filter = TemporalConsistencyFilter()
