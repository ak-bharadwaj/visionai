import queue, threading, time, logging, os

logger = logging.getLogger(__name__)


class TTSEngine:
    DEDUP_WINDOW    = 3.0   # seconds — how long to suppress identical phrases
    NAV_RATE_LIMIT  = 2.5   # seconds — min gap between non-priority messages

    def __init__(self):
        self._q:              queue.Queue      = queue.Queue(maxsize=5)
        self._last:           dict[str, float] = {}
        self._last_nav_time:  float            = 0.0
        self._engine                           = None

    def start(self):
        """Initialize pyttsx3 and start worker thread."""
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate",   int(os.getenv("TTS_RATE",   "175")))
            self._engine.setProperty("volume", float(os.getenv("TTS_VOLUME","1.0")))
            logger.info("TTS engine ready.")
        except Exception as e:
            logger.warning(f"TTS init failed: {e}. Voice output disabled.")
        threading.Thread(target=self._worker, daemon=True, name="TTSThread").start()

    def speak(self, text: str, priority: bool = False):
        """
        Queue text for speech.
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
        while True:
            text = self._q.get()
            if self._engine:
                try:
                    self._engine.say(text)
                    self._engine.runAndWait()
                except Exception as e:
                    logger.error(f"TTS speak error: {e}")
            self._q.task_done()


tts_engine = TTSEngine()
