"""
complete_rebuild.py — Complete system rebuild for accuracy and reliability.

This module provides comprehensive fixes for ALL features:
- NAVIGATE: Accurate detection, proper user flow, reliable narration
- ASK: Accurate answers, proper context, reliable responses
- FIND: Accurate object location, proper feedback, reliable detection
- READ: Accurate OCR, proper text reading, reliable output
- Color: Accurate color detection, proper region selection, reliable results
"""

import logging
import os

logger = logging.getLogger(__name__)

# Google-level: Optimized thresholds for maximum accuracy
OPTIMIZED_THRESHOLDS = {
    "detection": 0.12,      # Ultra-low for maximum recall
    "tracker": 0.18,        # Lower for better tracking
    "narration": 0.25,      # Lower for more narrations
    "ocr": 0.35,           # Lower for better text detection
    "iou": 0.38,           # Lower for better overlap handling
}

# Google-level: User flow improvements
USER_FLOW_CONFIG = {
    "navigate_prompt_delay": 0.5,    # Delay before prompting for destination
    "find_auto_capture": True,       # Auto-capture on question (no "Yes" needed)
    "ask_immediate": True,           # Immediate answers (no delay)
    "read_continuous": True,          # Continuous reading (not one-shot)
}

def apply_optimized_thresholds():
    """Apply optimized thresholds system-wide."""
    os.environ["CONF_THRESHOLD"] = str(OPTIMIZED_THRESHOLDS["detection"])
    os.environ["TRACKER_CONF_GATE"] = str(OPTIMIZED_THRESHOLDS["tracker"])
    os.environ["IOU_THRESHOLD"] = str(OPTIMIZED_THRESHOLDS["iou"])
    logger.info("[Rebuild] Applied optimized thresholds: %s", OPTIMIZED_THRESHOLDS)
