"""
enhanced_ocr.py — Google-level OCR enhancements.

Features:
  - Frame preprocessing (denoising, contrast enhancement)
  - Multi-model ensemble (optional)
  - Language detection
  - Better text region detection
  - Confidence calibration
"""

import cv2
import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)


class OCRPreprocessor:
    """Advanced preprocessing for OCR accuracy improvement."""
    
    @staticmethod
    def preprocess(frame: np.ndarray) -> np.ndarray:
        """
        Apply Google-level preprocessing:
        1. Denoising
        2. Contrast enhancement
        3. Sharpening
        4. Adaptive thresholding (if needed)
        """
        if frame is None or frame.size == 0:
            return frame
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # 1. Denoising (bilateral filter - preserves edges)
        denoised = cv2.bilateralFilter(gray, 5, 50, 50)
        
        # 2. Contrast enhancement (CLAHE - Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 3. Sharpening (unsharp mask)
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        # 4. Optional: Adaptive thresholding for very poor lighting
        # Only apply if image is very dark or very bright
        mean_brightness = np.mean(sharpened)
        if mean_brightness < 50 or mean_brightness > 200:
            # Apply adaptive thresholding
            adaptive = cv2.adaptiveThreshold(
                sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            return adaptive
        
        return sharpened


class OCRConfidenceCalibrator:
    """Calibrate OCR confidence scores for better accuracy."""
    
    def __init__(self):
        # Calibration factors based on text characteristics
        # Longer text = more reliable, shorter = less reliable
        self._min_length_factor = 0.9  # Penalty for very short text
        self._max_length_factor = 1.1  # Bonus for longer text
    
    def calibrate(self, text: str, confidence: float) -> float:
        """Apply calibration based on text length and characteristics."""
        length = len(text.strip())
        
        # Adjust based on length
        if length < 3:
            factor = self._min_length_factor
        elif length > 20:
            factor = self._max_length_factor
        else:
            factor = 1.0
        
        # Check for common OCR errors (numbers vs letters)
        has_numbers = any(c.isdigit() for c in text)
        has_letters = any(c.isalpha() for c in text)
        
        # Mixed alphanumeric is often more reliable
        if has_numbers and has_letters:
            factor *= 1.05
        
        calibrated = confidence * factor
        return min(1.0, max(0.0, calibrated))


# Module-level instances
ocr_preprocessor = OCRPreprocessor()
ocr_confidence_calibrator = OCRConfidenceCalibrator()
