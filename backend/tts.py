import queue, threading, time, logging, asyncio
from typing import Callable

logger = logging.getLogger(__name__)


class TTSEngine:
    DEDUP_WINDOW    = 8.0   # seconds — how long to suppress identical phrases
    NAV_RATE_LIMIT  = 5.0   # seconds — min gap between non-priority messages

    def __init__(self):
        self._q:              queue.Queue      = queue.Queue(maxsize=5)
        self._last:           dict[str, float] = {}
        self._last_nav_time:  float            = 0.0
        self._broadcast:      Callable | None  = None
        self._loop:           asyncio.AbstractEventLoop | None = None

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
        """
        if not text:
            return
        now = time.time()

        # Dedup check — identical phrase spoken too recently
        if text in self._last and now - self._last[text] < self.DEDUP_WINDOW:
            return

        # Rate-limit non-priority (navigation) messages
        if not priority and (now - self._last_nav_time) < self.NAV_RATE_LIMIT:
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
        except queue.Full:
            pass  # drop if queue full — never block the pipeline

        self._last[text] = now

    def _worker(self):
        speak_count = 0
        while True:
            text = self._q.get()
            if self._broadcast and self._loop:
                try:
                    asyncio.run_coroutine_threadsafe(
                        self._broadcast({"type": "speak", "text": text}),
                        self._loop
                    )
                except Exception as e:
                    logger.error(f"TTS relay error: {e}")
            speak_count += 1
            # Purge dedup dict every 200 speaks to prevent unbounded memory growth
            if speak_count % 200 == 0:
                cutoff = time.time() - 60.0
                self._last = {k: v for k, v in self._last.items() if v > cutoff}
            self._q.task_done()


tts_engine = TTSEngine()
