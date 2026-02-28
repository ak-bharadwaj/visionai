"""
tts_wrapper.py — Google-level TTS wrapper for reliable voice output.

Wraps all TTS calls with reliability checks and error handling.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def safe_speak(tts_engine, text: str, priority: bool = False):
    """
    Google-level: Safe TTS speak wrapper with reliability checks.
    Use this instead of direct tts_engine.speak() calls.
    """
    if not text or not text.strip():
        return
    
    try:
        from backend.voice_fix import voice_reliability
        voice_reliability.ensure_speak(tts_engine, text, priority=priority)
    except Exception as e:
        logger.debug("Voice reliability wrapper failed, using direct call: %s", e)
        try:
            tts_engine.speak(text, priority=priority)
        except Exception as e2:
            logger.error("TTS speak failed completely: %s", e2)
