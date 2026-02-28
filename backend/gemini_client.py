"""
gemini_client.py — Google Gemini API integration with 4-key rotation.

Uses GEMINI_API_KEY_1, GEMINI_API_KEY_2, GEMINI_API_KEY_3, GEMINI_API_KEY_4
from environment variables. Rotates keys for reliability and rate limit handling.

Provides:
- Visual QA (image + question → answer) - replaces LLaVA when available
- Color verification - improves color detection accuracy
- Scene understanding - better context for all modes
"""

import base64
import logging
import os
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Load up to 4 Gemini API keys from env
# Supports: GEMINI_API_KEY_1, GEMINI_API_KEY_2, GEMINI_API_KEY_3, GEMINI_API_KEY_4
GEMINI_KEYS: list[str] = []
for i in range(1, 5):
    key = os.getenv(f"GEMINI_API_KEY_{i}", "").strip()
    if key:
        GEMINI_KEYS.append(key)

# Also support single GEMINI_API_KEY for backward compatibility
_single = os.getenv("GEMINI_API_KEY", "").strip()
if _single and _single not in GEMINI_KEYS:
    GEMINI_KEYS.insert(0, _single)

GEMINI_ENABLED = len(GEMINI_KEYS) > 0
_current_key_index = 0
_key_lock = threading.Lock()

# Model to use (gemini-1.5-flash is fast and good for vision)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_TIMEOUT = float(os.getenv("GEMINI_TIMEOUT", "10.0"))


def _get_next_key() -> Optional[str]:
    """Rotate to next API key (round-robin)."""
    global _current_key_index
    if not GEMINI_KEYS:
        return None
    with _key_lock:
        key = GEMINI_KEYS[_current_key_index]
        _current_key_index = (_current_key_index + 1) % len(GEMINI_KEYS)
        return key


def _encode_frame_b64(frame: np.ndarray, max_width: int = 512) -> Optional[str]:
    """Encode frame to JPEG base64 for Gemini API."""
    try:
        import cv2
        h, w = frame.shape[:2]
        if w > max_width:
            scale = max_width / w
            new_w = max_width
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", frame)
        return base64.b64encode(buf).decode("utf-8")
    except Exception as e:
        logger.warning("[Gemini] Frame encode error: %s", e)
        return None


def ask_vision(question: str, frame: np.ndarray, context: str = "") -> Optional[str]:
    """
    Send image + question to Gemini for visual QA.
    
    Args:
        question: User's question
        frame: BGR image (numpy array)
        context: Optional context (e.g. detected objects)
    
    Returns:
        Answer string or None if failed
    """
    if not GEMINI_ENABLED or frame is None or frame.size == 0:
        return None
    
    img_b64 = _encode_frame_b64(frame, max_width=640)
    if not img_b64:
        return None
    
    key = _get_next_key()
    if not key:
        return None
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        prompt_parts = []
        if context:
            prompt_parts.append(f"Context: {context}\n\n")
        prompt_parts.append(f"Question: {question}\n\n")
        prompt_parts.append("Answer in 1-2 clear sentences. Be accurate and concise.")
        
        # Gemini API: use Image or inline_data format
        prompt_text = "\n".join(prompt_parts)
        try:
            # Try Part.from_bytes (newer SDK)
            image_part = genai.types.Part.from_bytes(
                data=base64.b64decode(img_b64),
                mime_type="image/jpeg"
            )
            content = [image_part, prompt_text]
        except (AttributeError, TypeError):
            # Fallback: dict format for inline_data
            content = [
                {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
                prompt_text
            ]
        
        response = model.generate_content(
            content,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=150,
                temperature=0.2,
            ),
            request_options={"timeout": GEMINI_TIMEOUT}
        )
        
        if response and response.text:
            return response.text.strip()
    except ImportError:
        logger.warning("[Gemini] google-generativeai not installed. Run: pip install google-generativeai")
    except Exception as e:
        logger.warning("[Gemini] Vision API error (key rotation will try next): %s", e)
    
    return None


def ask_vision_with_fallback(question: str, frame: np.ndarray, 
                             context: str = "",
                             fallback_fn=None) -> str:
    """
    Try Gemini first, fall back to local LLM if Gemini fails.
    """
    result = ask_vision(question, frame, context)
    if result:
        logger.info("[Gemini] Vision answer used (AI-powered)")
        return result
    if fallback_fn:
        return fallback_fn()
    return "I couldn't analyze the image. Please try again."


def verify_color(frame: np.ndarray, region_hint: str = "center") -> Optional[str]:
    """
    Use Gemini to verify/identify clothing color - NOT skin tone.
    Returns color description or None.
    """
    if not GEMINI_ENABLED or frame is None:
        return None
    
    question = (
        "What color is the SHIRT or CLOTHING the person is wearing? "
        "NOT their skin tone or face. ONLY the clothing. "
        "Answer with one color phrase like 'dark gray', 'black', 'light blue', 'white'. "
        "Be precise - ignore skin color completely."
    )
    return ask_vision(question, frame)


def ask_text_only(prompt: str, context: str = "") -> Optional[str]:
    """
    Text-only query to Gemini (no image). For structured reasoning.
    """
    if not GEMINI_ENABLED:
        return None
    
    key = _get_next_key()
    if not key:
        return None
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=200,
                temperature=0.1,
            ),
            request_options={"timeout": GEMINI_TIMEOUT}
        )
        
        if response and response.text:
            return response.text.strip()
    except Exception as e:
        logger.warning("[Gemini] Text API error: %s", e)
    
    return None


def is_available() -> bool:
    """Check if Gemini API is configured and available."""
    if not GEMINI_ENABLED:
        return False
    try:
        import google.generativeai as genai
        return True
    except ImportError:
        return False
