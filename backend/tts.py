import queue, threading, time, logging, asyncio
from typing import Callable

logger = logging.getLogger(__name__)


class TTSEngine:
    DEDUP_WINDOW    = 8.0   # seconds — how long to suppress identical phrases
    NAV_RATE_LIMIT  = 2.0   # seconds — min gap between non-priority messages

    def __init__(self):
        # Increased queue size for better reliability
        self._q:              queue.Queue      = queue.Queue(maxsize=20)
        self._last:           dict[str, float] = {}
        self._last_nav_time:  float            = 0.0
        self._broadcast:      Callable | None  = None
        self._loop:           asyncio.AbstractEventLoop | None = None
        self._speak_count:    int              = 0  # Track total speaks for cleanup

    def start(self, broadcast_fn: Callable = None, event_loop: asyncio.AbstractEventLoop = None):
        """Store broadcast callback and start worker thread."""
        self._broadcast = broadcast_fn
        self._loop      = event_loop
        threading.Thread(target=self._worker, daemon=True, name="TTSThread").start()
        logger.info("TTS relay engine ready (browser speechSynthesis).")

    def speak(self, text: str, priority: bool = False):
        """
        Queue text to be relayed to the browser via WebSocket {"type":"speak","text":"..."}.
        priority=True: clear queue first (for urgent alerts, level-1 distance).
        Dedup: drops if same text was spoken within DEDUP_WINDOW seconds.
        Rate limit: non-priority messages are dropped if last nav message was < NAV_RATE_LIMIT ago.
        
        Google-level: Enhanced reliability with better error handling.
        """
        if not text or not text.strip():
            return
        
        # Google-level: Clean and validate text
        text = text.strip()
        if len(text) == 0:
            return
        
        # Google-level: Ensure text is not too long (browser TTS limit)
        if len(text) > 500:
            text = text[:497] + "..."
            logger.debug("[TTS] Truncated long text to 500 chars")
        
        now = time.time()

        # Dedup check — identical phrase spoken too recently.
        # Priority=True (brain answers, urgent alerts) always bypasses dedup so
        # the user never silently misses a repeated answer within the window.
        if not priority and text in self._last and now - self._last[text] < self.DEDUP_WINDOW:
            logger.debug("TTS dedup: dropped %r (%.1fs ago)", text[:60], now - self._last[text])
            return

        # Rate-limit non-priority (navigation) messages
        if not priority and (now - self._last_nav_time) < self.NAV_RATE_LIMIT:
            logger.debug("TTS nav rate-limit: dropped %r (%.1fs since last nav)", text[:60], now - self._last_nav_time)
            return

        if priority:
            # Clear pending low-priority items
            while not self._q.empty():
                try: self._q.get_nowait()
                except queue.Empty: break
        else:
            self._last_nav_time = now

        try:
            self._q.put_nowait(text)
            logger.debug("TTS queued (%s): %r", "priority" if priority else "normal", text[:80])
        except queue.Full:
            logger.warning("TTS queue full — dropped: %r", text[:60])  # drop if queue full — never block the pipeline

        self._last[text] = now

    def _worker(self):
        speak_count = 0
        consecutive_failures = 0
        while True:
            try:
                text = self._q.get(timeout=30.0)
            except queue.Empty:
                # No message in 30s — keep worker alive, check queue again
                continue
            if self._broadcast and self._loop:
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        self._broadcast({"type": "speak", "text": text}),
                        self._loop
                    )
                    try:
                        future.result(timeout=2.0)
                        logger.debug("TTS spoken: %r", text[:60])
                        consecutive_failures = 0  # Reset on success
                    except Exception as e:
                        consecutive_failures += 1
                        logger.error("TTS broadcast failed: %s (failures: %d)", e, consecutive_failures)
                        # Google-level: Fallback to server-side TTS after 3 failures
                        if consecutive_failures >= 3:
                            try:
                                from backend.server_tts import server_tts
                                server_tts.speak_async(text)
                                logger.info("TTS: Using server-side fallback for: %r", text[:60])
                            except Exception as fallback_error:
                                logger.error("TTS fallback also failed: %s", fallback_error)
                except Exception as e:
                    consecutive_failures += 1
                    logger.error("TTS relay scheduling error: %s (failures: %d)", e, consecutive_failures)
                    # Fallback to server TTS
                    if consecutive_failures >= 3:
                        try:
                            from backend.server_tts import server_tts
                            server_tts.speak_async(text)
                        except Exception:
                            pass
            else:
                # No broadcast available - use server TTS directly
                try:
                    from backend.server_tts import server_tts
                    server_tts.speak_async(text)
                except Exception:
                    pass
            
            self._speak_count += 1
            # Purge dedup dict every 100 speaks to prevent unbounded memory growth
            # More frequent cleanup for better memory management
            if self._speak_count % 100 == 0:
                cutoff = time.time() - 60.0
                self._last = {k: v for k, v in self._last.items() if v > cutoff}
                logger.debug("TTS: cleaned dedup dict (kept %d entries)", len(self._last))
            self._q.task_done()


tts_engine = TTSEngine()
