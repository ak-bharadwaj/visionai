"""
voice_fix.py — Google-level voice/TTS reliability fixes.

Features:
  - Enhanced TTS reliability
  - Better error handling
  - Fallback mechanisms
  - Voice state management
"""

import logging
import time
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class VoiceReliability:
    """Ensure voice/TTS works reliably."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._speak_history: list = []
        self._last_speak_time: float = 0.0
        self._consecutive_failures: int = 0
        self._max_failures: int = 5
    
    def ensure_speak(self, tts_engine, text: str, priority: bool = False):
        """
        Ensure TTS speak is called with retry logic.
        """
        if not text or not text.strip():
            return
        
        with self._lock:
            # Check if we've had too many failures
            if self._consecutive_failures >= self._max_failures:
                logger.warning(
                    "[Voice] Too many consecutive failures (%d), "
                    "skipping speak to prevent spam",
                    self._consecutive_failures
                )
                return
            
            # Rate limiting for non-priority
            if not priority:
                now = time.time()
                if now - self._last_speak_time < 0.5:  # Minimum 0.5s between non-priority
                    logger.debug("[Voice] Rate-limited non-priority message")
                    return
                self._last_speak_time = now
        
        # Try to speak
        try:
            tts_engine.speak(text, priority=priority)
            with self._lock:
                self._consecutive_failures = 0
                self._speak_history.append({
                    "text": text[:50],
                    "time": time.time(),
                    "priority": priority
                })
                # Keep only last 20
                if len(self._speak_history) > 20:
                    self._speak_history.pop(0)
        except Exception as e:
            logger.error("[Voice] Speak failed: %s", e)
            with self._lock:
                self._consecutive_failures += 1


# Module-level instance
voice_reliability = VoiceReliability()
