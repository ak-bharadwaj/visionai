"""
server_tts.py — Server-side TTS as backup for browser TTS.

Uses pyttsx3 for reliable offline TTS when browser TTS fails.
"""

import logging
import threading
import queue
import time

logger = logging.getLogger(__name__)

class ServerTTS:
    """Server-side TTS using pyttsx3 as backup."""
    
    def __init__(self):
        self._engine = None
        self._lock = threading.Lock()
        self._initialized = False
        self._q = queue.Queue(maxsize=10)
        self._worker_running = False
    
    def _init_engine(self):
        """Initialize pyttsx3 engine."""
        if self._initialized:
            return
        
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            # Configure voice settings
            self._engine.setProperty('rate', 150)  # Normal speed
            self._engine.setProperty('volume', 1.0)  # Full volume
            # Try to set a better voice
            voices = self._engine.getProperty('voices')
            if voices:
                # Prefer female voice if available
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self._engine.setProperty('voice', voice.id)
                        break
            self._initialized = True
            logger.info("Server TTS (pyttsx3) initialized")
        except Exception as e:
            logger.warning("Server TTS failed to initialize: %s", e)
            self._initialized = False
    
    def speak(self, text: str):
        """Speak text using server-side TTS."""
        if not text or not text.strip():
            return
        
        self._init_engine()
        if not self._initialized or not self._engine:
            return
        
        try:
            with self._lock:
                self._engine.say(text[:500])  # Limit length
                self._engine.runAndWait()
        except Exception as e:
            logger.error("Server TTS speak error: %s", e)
    
    def speak_async(self, text: str):
        """Speak text asynchronously (non-blocking)."""
        if not text or not text.strip():
            return
        
        try:
            self._q.put_nowait(text[:500])
            if not self._worker_running:
                self._start_worker()
        except queue.Full:
            logger.warning("Server TTS queue full, dropping: %s", text[:50])
    
    def _start_worker(self):
        """Start async worker thread."""
        if self._worker_running:
            return
        
        self._worker_running = True
        threading.Thread(target=self._worker, daemon=True, name="ServerTTSWorker").start()
    
    def _worker(self):
        """Worker thread for async TTS."""
        while True:
            try:
                text = self._q.get(timeout=30.0)
                self.speak(text)
                self._q.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Server TTS worker error: %s", e)


# Global instance
server_tts = ServerTTS()
