"""
tracker.py — Multi-object tracker with persistent IDs.
Implements IoU-based greedy matching (ByteTrack-inspired, no external deps).

Design constraints (safety-critical):
  - Hard confidence gate: detections < CONF_GATE are silently dropped.
  - Confirmation gate: objects need frames_seen >= MIN_FRAMES_CONFIRM before narration.
  - Eviction: tracks missing for >= MISS_FRAMES_EVICT consecutive frames are removed.
  - Bbox stability: area change > AREA_CHANGE_LIMIT resets the confirmation counter.
  - EMA bbox smoothing eliminates single-frame jitter.
  - EMA distance smoothing: 0.8 * prev + 0.2 * current.
  - Velocity computed over a rolling VELOCITY_WINDOW sample window.

TrackedObject schema (consumed by risk_engine, narrator, pipeline):
  id                  : int        — unique monotonic track id
  class_name          : str
  confidence          : float      — latest detection confidence
  frames_seen         : int        — confirmed frames
  frames_missed       : int        — consecutive frames without a match
  confirmed           : bool       — frames_seen >= MIN_FRAMES_CONFIRM
  x1, y1, x2, y2     : int        — EMA-smoothed bounding box
  smoothed_distance_m : float      — EMA distance in metres
  velocity_m_per_s    : float      — +ve = approaching, -ve = receding
  direction           : str        — clock direction string
  path_overlap_ratio  : float      — fraction of bbox inside walking corridor
  risk_level          : str        — "HIGH" | "MEDIUM" | "LOW" | "NONE"
  risk_score          : float      — 0.0–1.0 normalised
"""

import time
import threading
import logging
from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np

logger = logging.getLogger(__name__)

# ── Tuning constants ────────────────────────────────────────────────
MIN_FRAMES_CONFIRM = 3      # frames_seen before object reaches narration
MISS_FRAMES_EVICT  = 2      # consecutive misses before eviction
IOU_MATCH_THRESH   = 0.35   # minimum IoU to associate detection → track
BBOX_ALPHA         = 0.55   # EMA weight for bbox smoothing (higher = faster)
CONF_GATE          = 0.60   # hard minimum confidence; below this = ignored
AREA_CHANGE_LIMIT  = 0.40   # max fractional bbox area change per frame
VELOCITY_WINDOW    = 3      # distance samples for velocity estimation


@dataclass
class TrackedObject:
    id:            int
    class_name:    str
    confidence:    float
    frames_seen:   int   = 0
    frames_missed: int   = 0
    confirmed:     bool  = False

    # Internal float bbox for EMA smoothing; exposed as int properties
    _x1: float = 0.0
    _y1: float = 0.0
    _x2: float = 0.0
    _y2: float = 0.0

    smoothed_distance_m: float = 0.0
    velocity_m_per_s:    float = 0.0
    direction:           str   = "12 o'clock"
    path_overlap_ratio:  float = 0.0
    risk_level:          str   = "NONE"
    risk_score:          float = 0.0

    # Distance history ring buffer
    _dist_history:    List[float] = field(default_factory=list)
    _dist_timestamps: List[float] = field(default_factory=list)

    first_seen_t: float = field(default_factory=time.time)
    last_seen_t:  float = field(default_factory=time.time)
    _prev_area:   float = 0.0

    # ── Bbox properties ────────────────────────────────────────────
    @property
    def x1(self) -> int: return int(self._x1)
    @property
    def y1(self) -> int: return int(self._y1)
    @property
    def x2(self) -> int: return int(self._x2)
    @property
    def y2(self) -> int: return int(self._y2)

    @property
    def center_x(self) -> float:
        return (self._x1 + self._x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self._y1 + self._y2) / 2.0

    @property
    def area(self) -> float:
        return max(0.0, (self._x2 - self._x1) * (self._y2 - self._y1))

    # ── Stability check ────────────────────────────────────────────
    def bbox_area_stable(self, nx1: int, ny1: int, nx2: int, ny2: int) -> bool:
        """True if new bbox area is within AREA_CHANGE_LIMIT of current."""
        if self._prev_area <= 0:
            return True
        new_area = max(0.0, (nx2 - nx1) * (ny2 - ny1))
        change = abs(new_area - self._prev_area) / max(self._prev_area, 1.0)
        return change <= AREA_CHANGE_LIMIT

    # ── Update helpers ─────────────────────────────────────────────
    def update_bbox(self, x1: int, y1: int, x2: int, y2: int):
        """EMA-smooth the bounding box coordinates."""
        if self.frames_seen <= 1:
            self._x1, self._y1 = float(x1), float(y1)
            self._x2, self._y2 = float(x2), float(y2)
        else:
            a = BBOX_ALPHA
            self._x1 = a * x1 + (1 - a) * self._x1
            self._y1 = a * y1 + (1 - a) * self._y1
            self._x2 = a * x2 + (1 - a) * self._x2
            self._y2 = a * y2 + (1 - a) * self._y2
        self._prev_area = self.area

    def update_distance(self, dist_m: float):
        """
        Apply 0.8/0.2 exponential smoothing to distance measurement.
        Compute velocity when VELOCITY_WINDOW samples are available.
        Positive velocity_m_per_s = approaching.
        """
        now = time.time()
        if self.smoothed_distance_m <= 0.0:
            self.smoothed_distance_m = dist_m
        else:
            self.smoothed_distance_m = (
                0.8 * self.smoothed_distance_m + 0.2 * dist_m
            )
        self._dist_history.append(self.smoothed_distance_m)
        self._dist_timestamps.append(now)
        # Trim to window size + 1 buffer
        if len(self._dist_history) > VELOCITY_WINDOW + 2:
            self._dist_history.pop(0)
            self._dist_timestamps.pop(0)
        # Velocity estimation
        if len(self._dist_history) >= VELOCITY_WINDOW:
            dt = self._dist_timestamps[-1] - self._dist_timestamps[0]
            if dt > 0.05:
                dd = self._dist_history[-1] - self._dist_history[0]
                # Negative Δd = getting closer = positive "approaching" velocity
                self.velocity_m_per_s = -dd / dt

    def distance_variance(self) -> float:
        """Variance of recent distance samples. High = unstable depth."""
        if len(self._dist_history) < 2:
            return 0.0
        return float(np.var(self._dist_history))

    @property
    def distance_level(self) -> int:
        """1=very close (<1 m), 2=nearby (1-2 m), 3=ahead (2-3 m), 4=far (>3 m)."""
        d = self.smoothed_distance_m
        if d < 1.0: return 1
        if d < 2.0: return 2
        if d < 3.0: return 3
        return 4

    @property
    def motion_state(self) -> str:
        v = self.velocity_m_per_s
        if v > 0.05:  return "approaching"
        if v < -0.05: return "receding"
        return "static"


# ── IoU helper ──────────────────────────────────────────────────────

def _iou(a: tuple, b: tuple) -> float:
    """Intersection-over-Union for two (x1,y1,x2,y2) boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _clock_direction(center_x: float, frame_w: int) -> str:
    """Map horizontal center position to clock direction string."""
    ratio = center_x / max(frame_w, 1)
    if ratio < 0.15: return "9 o'clock"
    if ratio < 0.30: return "10 o'clock"
    if ratio < 0.45: return "11 o'clock"
    if ratio < 0.55: return "12 o'clock"
    if ratio < 0.70: return "1 o'clock"
    if ratio < 0.85: return "2 o'clock"
    return "3 o'clock"


# ── Tracker ─────────────────────────────────────────────────────────

class ObjectTracker:
    """
    Thread-safe IoU-based greedy multi-object tracker.

    Usage per frame:
        confirmed = tracker.update(detections, frame_w, frame_h)
        # confirmed is a list of TrackedObject with .confirmed == True
    """

    def __init__(self):
        self._tracks:  Dict[int, TrackedObject] = {}
        self._next_id: int = 1
        self._lock:    threading.Lock = threading.Lock()

    def reset(self):
        """Clear all tracks. Call on mode change."""
        with self._lock:
            self._tracks.clear()
            self._next_id = 1

    def update(self, detections: list,
               frame_w: int, frame_h: int) -> List[TrackedObject]:
        """
        Match detections to tracks, age unmatched tracks, spawn new tracks.

        Args:
            detections: list of Detection objects (class_name, confidence, x1,y1,x2,y2)
            frame_w, frame_h: frame dimensions

        Returns:
            List of confirmed TrackedObjects (frames_seen >= MIN_FRAMES_CONFIRM).
        """
        with self._lock:
            # 1. Hard confidence gate
            valid = [d for d in detections if d.confidence >= CONF_GATE]

            matched_track_ids: set = set()
            matched_det_idxs:  set = set()

            # 2. Greedy IoU matching
            if self._tracks and valid:
                tracks = list(self._tracks.values())
                mat = np.zeros((len(tracks), len(valid)), dtype=np.float32)
                for ti, t in enumerate(tracks):
                    tb = (t._x1, t._y1, t._x2, t._y2)
                    for di, d in enumerate(valid):
                        mat[ti, di] = _iou(tb, (d.x1, d.y1, d.x2, d.y2))

                while mat.size > 0:
                    idx = np.argmax(mat)
                    ti, di = np.unravel_index(idx, mat.shape)
                    if mat[ti, di] < IOU_MATCH_THRESH:
                        break
                    t = tracks[ti]
                    d = valid[di]

                    # Class guard — never merge different classes
                    if t.class_name != d.class_name:
                        mat[ti, di] = 0.0
                        continue

                    # Bbox area stability gate
                    if not t.bbox_area_stable(d.x1, d.y1, d.x2, d.y2):
                        logger.debug(
                            "[Tracker] Track %d (%s) bbox area jump > %d%% — "
                            "resetting confirmation.",
                            t.id, t.class_name, int(AREA_CHANGE_LIMIT * 100)
                        )
                        t.frames_seen = 1
                        t.confirmed   = False
                        mat[ti, di]   = 0.0
                        continue

                    # Accept match
                    t.frames_seen  += 1
                    t.frames_missed = 0
                    t.confidence    = d.confidence
                    t.last_seen_t   = time.time()
                    t.update_bbox(d.x1, d.y1, d.x2, d.y2)
                    t.confirmed     = (t.frames_seen >= MIN_FRAMES_CONFIRM)
                    t.direction     = _clock_direction(t.center_x, frame_w)
                    # Path overlap ratio (centre 40% corridor)
                    t.path_overlap_ratio = _corridor_overlap(t.x1, t.x2, frame_w)

                    matched_track_ids.add(t.id)
                    matched_det_idxs.add(di)
                    mat[ti, :] = 0.0
                    mat[:, di] = 0.0

            # 3. Age unmatched tracks; evict expired
            for t in list(self._tracks.values()):
                if t.id not in matched_track_ids:
                    t.frames_missed += 1
                    if t.frames_missed >= MISS_FRAMES_EVICT:
                        logger.debug(
                            "[Tracker] Evicting track %d (%s).",
                            t.id, t.class_name
                        )
                        del self._tracks[t.id]

            # 4. Spawn new tentative tracks for unmatched detections
            for di, d in enumerate(valid):
                if di in matched_det_idxs:
                    continue
                t = TrackedObject(
                    id=self._next_id,
                    class_name=d.class_name,
                    confidence=d.confidence,
                    frames_seen=1,
                )
                t.update_bbox(d.x1, d.y1, d.x2, d.y2)
                t.direction          = _clock_direction(t.center_x, frame_w)
                t.path_overlap_ratio = _corridor_overlap(t.x1, t.x2, frame_w)
                self._tracks[self._next_id] = t
                self._next_id += 1

            return [t for t in self._tracks.values() if t.confirmed]

    def all_tracks(self) -> List[TrackedObject]:
        """Return all tracks including unconfirmed. For debug/diagnostics."""
        with self._lock:
            return list(self._tracks.values())


# ── Shared helpers ───────────────────────────────────────────────────

def _corridor_overlap(x1: int, x2: int, frame_w: int) -> float:
    """
    Fraction of object's horizontal span inside the centre 40% walking corridor.
    Corridor spans 30%–70% of frame width.
    """
    c_left  = int(frame_w * 0.30)
    c_right = int(frame_w * 0.70)
    ol = max(x1, c_left)
    or_ = min(x2, c_right)
    overlap_w = max(0, or_ - ol)
    obj_w = max(1, x2 - x1)
    return overlap_w / obj_w


# Module-level singleton
object_tracker = ObjectTracker()
