"""
main_tts_fix.py — Fix all TTS calls in main.py with safe wrapper.
"""

# This module provides a helper to fix TTS in main.py
# Import this and use safe_speak instead of direct tts_engine.speak

from backend.tts_wrapper import safe_speak

__all__ = ['safe_speak']
