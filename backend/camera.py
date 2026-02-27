import cv2, threading, time, os, logging
import numpy as np

logger = logging.getLogger(__name__)

# How many consecutive read failures before we attempt a full reconnect
_FAIL_THRESHOLD = 5


class CameraStream:
    def __init__(self, source):
        self._source = int(source) if str(source).isdigit() else source
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._cap: cv2.VideoCapture | None = None

    def _open_capture(self) -> cv2.VideoCapture:
        """Open the capture source and apply zero-lag buffer setting."""
        cap = cv2.VideoCapture(self._source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def start(self) -> "CameraStream":
        self._cap = self._open_capture()
        if not self._cap.isOpened():
            logger.warning(
                f"👁 [Camera] Cannot open '{self._source}'. Falling back to webcam 0."
            )
            self._source = 0
            self._cap = self._open_capture()
            if not self._cap.isOpened():
                raise RuntimeError("No camera available. Set CAMERA_SOURCE in .env")
        self._running = True
        threading.Thread(target=self._loop, daemon=True, name="CameraThread").start()
        logger.info(f"👁 [Camera] Started: {self._source}")
        return self

    def _reconnect(self):
        """Release the current capture and re-open from scratch."""
        logger.warning("👁 [Camera] Reconnecting after repeated read failures…")
        try:
            self._cap.release()
        except Exception:
            pass
        time.sleep(0.2)
        self._cap = self._open_capture()
        if self._cap.isOpened():
            logger.info("👁 [Camera] Reconnected successfully.")
        else:
            logger.error("👁 [Camera] Reconnect failed — will retry next cycle.")

    def _loop(self):
        fail_streak = 0
        while self._running:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                fail_streak = 0
                with self._lock:
                    self._frame = frame
            else:
                fail_streak += 1
                time.sleep(0.1)
                if fail_streak >= _FAIL_THRESHOLD:
                    self._reconnect()
                    fail_streak = 0

    def read(self) -> np.ndarray | None:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def set_source(self, new_source: str):
        """Hot-swap camera source at runtime without restarting the thread."""
        new_src = int(new_source) if str(new_source).isdigit() else new_source
        logger.info(f"👁 [Camera] Switching source to: {new_src}")
        self._source = new_src
        self._reconnect()

    def stop(self):
        self._running = False
        time.sleep(0.15)
        if self._cap:
            self._cap.release()
        logger.info("👁 [Camera] Stopped.")


def get_camera() -> CameraStream:
    source = os.getenv("CAMERA_SOURCE", "0")
    return CameraStream(source).start()
