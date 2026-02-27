import threading

VALID_MODES = {"NAVIGATE", "ASK", "READ"}

class ModeManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._mode = "NAVIGATE"
        self._show_overlay = False

    def set_mode(self, mode: str):
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode: {mode!r}. Must be one of {VALID_MODES}")
        with self._lock:
            self._mode = mode

    def toggle_overlay(self):
        with self._lock:
            self._show_overlay = not self._show_overlay

    def is_navigate(self) -> bool:
        with self._lock: return self._mode == "NAVIGATE"

    def is_ask(self) -> bool:
        with self._lock: return self._mode == "ASK"

    def is_read(self) -> bool:
        with self._lock: return self._mode == "READ"

    def snapshot(self) -> dict:
        with self._lock:
            return {"current_mode": self._mode, "show_overlay": self._show_overlay}

# Module-level singleton — import this everywhere
mode_manager = ModeManager()
