"""
tracker.py — Multi-object tracker with persistent IDs.
Implements IoU-based greedy matching (ByteTrack-inspired, no external deps).

Design constraints (safety-critical):
  - Two-tier confidence gate:
      CONF_DETECT (default 0.35) — minimum to enter the tracker and accumulate
        frames_seen.  Low-conf detections can build track history without
        triggering narration.
      CONF_GATE (default 0.60) — narration / output gate, enforced by
        stability_filter.object_passes_gate().  Nothing below this is ever
        spoken.  This is NOT re-applied inside the tracker; stability_filter owns it.
  - Confirmation gate: objects need frames_seen >= MIN_FRAMES_CONFIRM before narration.
  - Eviction: tracks missing for >= MISS_FRAMES_EVICT consecutive frames are removed.
  - Bbox stability: area change > AREA_CHANGE_LIMIT resets the confirmation counter.
  - EMA bbox smoothing eliminates single-frame jitter.
  - EMA distance smoothing: 0.8 * prev + 0.2 * current.
  - Velocity computed over a rolling VELOCITY_WINDOW sample window.
  - Velocity clamped to ±MAX_VELOCITY_M_S (default 3.0 m/s) to suppress
    depth-spike artefacts that pass the sign-consistency gate.

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
import os
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np

from backend.diagnostics import diagnostics as _diagnostics

logger = logging.getLogger(__name__)

# ── Tuning constants ────────────────────────────────────────────────
MIN_FRAMES_CONFIRM = 3      # frames_seen before object reaches narration
# Consecutive missed frames before a track is evicted.
# Overridable via env var for tuning (e.g. MISS_FRAMES_EVICT=8 for slower cameras).
# Default 6: more forgiving than original 2 — brief occlusion / frame skip won't
# kill a confirmed track.  Keep low enough that truly-gone objects evict quickly.
MISS_FRAMES_EVICT  = int(os.getenv("MISS_FRAMES_EVICT", "8"))
IOU_MATCH_THRESH   = 0.35   # minimum IoU to associate detection → track
BBOX_ALPHA         = 0.55   # EMA weight for bbox smoothing (higher = faster)

# Two-tier confidence gate:
#   CONF_DETECT — minimum confidence for a detection to enter the tracker.
#     Low-conf detections can accumulate frames_seen and build velocity history
#     without ever reaching narration (which requires CONF_GATE, enforced by
#     stability_filter.object_passes_gate()).
#   CONF_GATE — minimum confidence for narration output.  This is NOT enforced
#     here; stability_filter owns the output gate.  Exposed as a constant so
#     tests and external callers can import it.
# Both are overridable via env vars for debugging / threshold tuning.
CONF_DETECT = float(os.getenv("CONF_DETECT",        "0.35"))
CONF_GATE   = float(os.getenv("TRACKER_CONF_GATE",  "0.60"))

AREA_CHANGE_LIMIT  = 0.40   # max fractional bbox area change per frame
VELOCITY_WINDOW    = 3      # distance samples for velocity estimation
# Hard cap on absolute velocity to suppress depth-spike artefacts.
MAX_VELOCITY_M_S   = float(os.getenv("MAX_VELOCITY_M_S", "3.0"))

# Stale depth confidence decay.
# After DEPTH_STALE_EVICT_FRAMES consecutive frames without a depth update,
# a confirmed object's effective_confidence is decayed so it falls below
# CONF_GATE and the narration gate suppresses it.  Prevents outdated distance
# readings from being narrated as if they were current measurements.
# Reset to 0 every time update_distance() is called with a fresh measurement.
DEPTH_STALE_EVICT_FRAMES = int(os.getenv("DEPTH_STALE_EVICT_FRAMES", "6"))

# Dormant track revival.
# When a track is evicted (frames_missed >= MISS_FRAMES_EVICT), it moves to a
# dormant pool instead of being deleted immediately.  If a new detection of the
# same class appears within DORMANT_REVIVAL_TTL seconds and has IoU >=
# DORMANT_REVIVAL_IOU_THRESH with the evicted bbox, the original track is
# revived — preserving its ID, velocity history, and distance history.
# This greatly reduces ID fragmentation on brief occlusions (hands in front of
# camera, person stepping behind a pillar, etc.).
DORMANT_REVIVAL_TTL       = float(os.getenv("DORMANT_REVIVAL_TTL",       "2.0"))
DORMANT_REVIVAL_IOU_THRESH = float(os.getenv("DORMANT_REVIVAL_IOU_THRESH", "0.25"))

# Depth history median filter.
# The last DEPTH_MEDIAN_WINDOW raw depth readings (before EMA) are stored per
# track.  The median of this window is computed first, then fed into the EMA
# smoother.  This eliminates single-frame MiDaS spikes that would otherwise
# enter the EMA and corrupt velocity estimation.
# Window of 5 frames at ~10 FPS ≈ 0.5 s of depth history.
DEPTH_MEDIAN_WINDOW = int(os.getenv("DEPTH_MEDIAN_WINDOW", "5"))


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

    # Distance history ring buffer (deque for O(1) append + auto-trim)
    _dist_history:    object = field(default=None, repr=False)
    _dist_timestamps: object = field(default=None, repr=False)

    first_seen_t: float = field(default_factory=time.time)
    last_seen_t:  float = field(default_factory=time.time)
    _prev_area:   float = 0.0

    # Pixel-space velocity for bbox motion prediction.
    # Updated every matched frame; used to extrapolate a predicted bbox for
    # the next frame's IoU matching — reduces ID switches for fast-moving objects.
    _pixel_vx: float = 0.0   # horizontal pixel velocity (pixels/second)
    _pixel_vy: float = 0.0   # vertical pixel velocity   (pixels/second)
    _last_update_t: float = field(default_factory=time.time)

    # Stale depth counter — incremented each frame that a confirmed object is
    # matched but update_distance() is NOT called (depth rejected or unavailable).
    # Reset to 0 on every accepted depth measurement.
    # effective_confidence decays toward zero once this exceeds a threshold,
    # causing the narration gate to suppress stale readings.
    _depth_stale_frames: int = 0

    # Timestamp when this track entered the dormant pool (set by ObjectTracker
    # when evicting; 0.0 means the track is active, not dormant).
    _dormant_since: float = 0.0

    # Sliding window of raw (pre-EMA) depth readings for median pre-filtering.
    # Before the EMA smoother sees a new distance value, the median of this
    # deque is computed first.  This removes single-frame MiDaS spikes that
    # pass depth_jump_reject but are still outliers within a short window.
    # Initialised lazily (None → deque on first update_distance call).
    _raw_depth_history: object = field(default=None, repr=False)

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

    @property
    def effective_confidence(self) -> float:
        """
        Confidence after stale-depth penalty.

        When depth readings stop being accepted (depth map unavailable, or all
        measurements are being rejected by depth_jump_reject), this decays from
        the raw confidence toward 0 linearly over DEPTH_STALE_EVICT_FRAMES
        frames.  Once it drops below CONF_GATE (0.60), the narration gate will
        suppress this object — preventing outdated distances from being spoken.

        Approaching objects (velocity_m_per_s > 0.05) are exempt from stale
        decay: if something is definitely moving toward you, the last valid
        distance reading is still safety-critical information.

        Returns float in [0.0, confidence].
        """
        # Approaching objects: exempt from stale depth decay (last reading
        # is still directionally correct for imminent collision warning)
        if self.velocity_m_per_s > 0.05:
            return self.confidence
        stale = self._depth_stale_frames
        if stale <= 0:
            return self.confidence
        # Linear decay: 0 stale = full confidence, N stale = 0 confidence
        fraction = max(0.0, 1.0 - stale / max(1, DEPTH_STALE_EVICT_FRAMES))
        return self.confidence * fraction

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
        """EMA-smooth the bounding box coordinates and update pixel velocity."""
        now = time.time()
        if self.frames_seen <= 1:
            self._x1, self._y1 = float(x1), float(y1)
            self._x2, self._y2 = float(x2), float(y2)
            self._pixel_vx = 0.0
            self._pixel_vy = 0.0
        else:
            dt = now - self._last_update_t
            if dt > 0.001:
                # EMA pixel velocity: how fast the bbox center is moving
                new_cx = (x1 + x2) / 2.0
                new_cy = (y1 + y2) / 2.0
                old_cx = (self._x1 + self._x2) / 2.0
                old_cy = (self._y1 + self._y2) / 2.0
                raw_vx = (new_cx - old_cx) / dt
                raw_vy = (new_cy - old_cy) / dt
                # EMA smooth pixel velocity (α=0.4 — responsive but not twitchy)
                self._pixel_vx = 0.6 * self._pixel_vx + 0.4 * raw_vx
                self._pixel_vy = 0.6 * self._pixel_vy + 0.4 * raw_vy
            a = BBOX_ALPHA
            self._x1 = a * x1 + (1 - a) * self._x1
            self._y1 = a * y1 + (1 - a) * self._y1
            self._x2 = a * x2 + (1 - a) * self._x2
            self._y2 = a * y2 + (1 - a) * self._y2
        self._prev_area   = self.area
        self._last_update_t = now

    def predicted_bbox(self, dt: float) -> tuple:
        """
        Return (x1, y1, x2, y2) extrapolated dt seconds into the future using
        the last measured pixel velocity.  Used for IoU matching to reduce ID
        switches for fast-moving objects.

        Args:
            dt: elapsed seconds since last update (typically 0.03–0.20 s).

        Returns:
            Predicted bounding box as float tuple (x1, y1, x2, y2).
        """
        dx = self._pixel_vx * dt
        dy = self._pixel_vy * dt
        return (self._x1 + dx, self._y1 + dy, self._x2 + dx, self._y2 + dy)

    def update_distance(self, dist_m: float):
        """
        Apply median pre-filter then 0.8/0.2 exponential smoothing to a new
        distance measurement.  Compute velocity when VELOCITY_WINDOW samples
        are available.  Positive velocity_m_per_s = approaching.

        Depth pipeline (in order):
          0. Median pre-filter — last DEPTH_MEDIAN_WINDOW raw readings stored in
             a sliding deque.  The median of the window is passed to the EMA
             smoother instead of the raw value.  This eliminates single-frame
             MiDaS spikes that pass depth_jump_reject but are outliers within a
             short window (e.g. one bad frame in five).
          1. EMA smoothing — 0.8 * prev + 0.2 * median_filtered_reading.
          2. Velocity sign-consistency gate — all consecutive distance deltas
             must share the same sign; contradicting samples leave velocity
             unchanged.
          3. Velocity EMA — 0.7/0.3 dampens spikes that pass the sign gate.
          4. Hard clamp — result never exceeds ±MAX_VELOCITY_M_S.

        Also resets _depth_stale_frames to 0 — an accepted measurement means
        depth is fresh and effective_confidence returns to full value.
        """
        # Fresh depth measurement accepted — reset stale counter.
        self._depth_stale_frames = 0
        now = time.time()

        # ── Lazy-init distance history deques ────────────────────────────────
        if self._dist_history is None:
            self._dist_history    = deque(maxlen=VELOCITY_WINDOW + 2)
            self._dist_timestamps = deque(maxlen=VELOCITY_WINDOW + 2)

        # ── Step 0: Median pre-filter ─────────────────────────────────────────
        if self._raw_depth_history is None:
            self._raw_depth_history = deque(maxlen=DEPTH_MEDIAN_WINDOW)
        self._raw_depth_history.append(dist_m)
        # Use median of window as the value fed into EMA (robust to spikes).
        filtered_dist = float(np.median(self._raw_depth_history))

        # ── Step 1: EMA smoothing ─────────────────────────────────────────────
        if self.smoothed_distance_m <= 0.0:
            self.smoothed_distance_m = filtered_dist
        else:
            self.smoothed_distance_m = (
                0.8 * self.smoothed_distance_m + 0.2 * filtered_dist
            )
        self._dist_history.append(self.smoothed_distance_m)
        self._dist_timestamps.append(now)
        # Deque auto-trims to maxlen=VELOCITY_WINDOW + 2; no manual pop() needed.
        # Velocity estimation
        if len(self._dist_history) >= VELOCITY_WINDOW:
            dt = self._dist_timestamps[-1] - self._dist_timestamps[0]
            if dt > 0.05:
                dd = self._dist_history[-1] - self._dist_history[0]
                # Negative Δd = getting closer = positive "approaching" velocity.
                # Require that all consecutive deltas share the same sign before
                # accepting — a single noisy depth spike cannot flip direction.
                deltas = [
                    self._dist_history[i] - self._dist_history[i - 1]
                    for i in range(1, len(self._dist_history))
                ]
                signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in deltas]
                nonzero = [s for s in signs if s != 0]
                sign_consistent = len(nonzero) > 0 and len(set(nonzero)) == 1
                if sign_consistent:
                    raw_vel = -dd / dt
                    # EMA smoothing: reduces single-frame velocity spikes that
                    # pass the sign-consistency gate but reflect noise rather
                    # than true motion.  0.7/0.3 weights keep the estimate
                    # responsive while significantly dampening outliers.
                    smoothed_vel = (
                        0.7 * self.velocity_m_per_s + 0.3 * raw_vel
                        if self.velocity_m_per_s != 0.0
                        else raw_vel
                    )
                    # Hard clamp — no velocity artefact can exceed physical limit.
                    self.velocity_m_per_s = max(
                        -MAX_VELOCITY_M_S, min(MAX_VELOCITY_M_S, smoothed_vel)
                    )
                # If sign is inconsistent leave velocity unchanged (use last
                # stable estimate) rather than snapping to a noisy reading.

    def distance_variance(self) -> float:
        """Variance of recent distance samples. High = unstable depth."""
        if len(self._dist_history) < 2:
            return 0.0
        return float(np.var(self._dist_history))

    @property
    def track_age_seconds(self) -> float:
        """
        Elapsed seconds since this track was first seen.

        Used by the risk engine to decay the priority of objects that have
        been static in the scene for a long time — a chair that has been
        sitting at 2 m for 30 seconds is much less urgent than one that
        just appeared.
        """
        return time.time() - self.first_seen_t

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

    @property
    def collision_eta_s(self) -> float:
        """
        Estimated seconds until collision if the object continues at current velocity.

        Returns:
            Positive float (seconds) when approaching and velocity > 0.1 m/s.
            0.0 if not approaching, stationary, or distance unknown.
        """
        if self.velocity_m_per_s < 0.1:
            return 0.0
        if self.smoothed_distance_m <= 0.0:
            return 0.0
        return self.smoothed_distance_m / self.velocity_m_per_s

    @property
    def lateral_speed_px_per_s(self) -> float:
        """
        Absolute horizontal pixel velocity of this track (pixels/second).

        A high value with a short distance indicates the object is crossing the
        path laterally — e.g. a person walking left-to-right at 1 m distance.
        This property is consumed by the risk engine's lateral boost.

        Returns:
            Non-negative float.  0.0 before the first bbox update.
        """
        return abs(self._pixel_vx)


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
        self._tracks:   Dict[int, TrackedObject] = {}
        self._next_id:  int = 1
        self._lock:     threading.Lock = threading.Lock()
        # Short-lived record of recently evicted tracks for ID-switch detection.
        # Maps class_name → (evicted_id, eviction_time).
        self._recent_evictions: Dict[str, tuple] = {}
        # Dormant track pool — evicted tracks held here briefly so they can be
        # revived if the same object reappears within DORMANT_REVIVAL_TTL seconds.
        # Maps track_id → TrackedObject (with _dormant_since set).
        self._dormant: Dict[int, TrackedObject] = {}

    def reset(self):
        """Clear all tracks. Call on mode change."""
        with self._lock:
            self._tracks.clear()
            self._dormant.clear()
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
            # 1. Two-tier confidence gate — CONF_DETECT (0.35) allows low-conf
            # detections into the tracker so they can accumulate frames_seen and
            # build velocity history.  The narration gate (CONF_GATE = 0.60) is
            # enforced later by stability_filter.object_passes_gate(), not here.
            valid = [d for d in detections if d.confidence >= CONF_DETECT]

            matched_track_ids: set = set()
            matched_det_idxs:  set = set()

            # 2. Greedy IoU matching using motion-predicted bboxes.
            # For each existing track, extrapolate its position by dt seconds
            # using its pixel-space velocity before computing IoU.  This
            # maintains correct associations for objects that move significantly
            # between detection frames (DETECT_EVERY_N=4 ≈ 130 ms at 30 FPS).
            if self._tracks and valid:
                now_match = time.time()
                tracks = list(self._tracks.values())
                mat = np.zeros((len(tracks), len(valid)), dtype=np.float32)
                for ti, t in enumerate(tracks):
                    dt_pred = now_match - t._last_update_t
                    # Clamp prediction horizon — don't extrapolate more than
                    # ~300 ms to avoid over-shooting when a track has been lost.
                    dt_pred = min(dt_pred, 0.30)
                    pb = t.predicted_bbox(dt_pred)
                    for di, d in enumerate(valid):
                        mat[ti, di] = _iou(pb, (d.x1, d.y1, d.x2, d.y2))

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

            # 3. Age unmatched tracks; evict expired → move to dormant pool
            now_t = time.time()
            # Prune expired dormant tracks first (keep pool small).
            expired_dormant = [
                tid for tid, dt in self._dormant.items()
                if now_t - dt._dormant_since > DORMANT_REVIVAL_TTL
            ]
            for tid in expired_dormant:
                del self._dormant[tid]

            for t in list(self._tracks.values()):
                if t.id not in matched_track_ids:
                    t.frames_missed += 1
                    if t.frames_missed >= MISS_FRAMES_EVICT:
                        logger.debug(
                            "[Tracker] Evicting track %d (%s) → dormant pool.",
                            t.id, t.class_name
                        )
                        # Record eviction so new tracks of same class can be
                        # identified as likely ID switches (within 3 seconds).
                        self._recent_evictions[t.class_name] = (t.id, now_t)
                        # Move to dormant pool instead of deleting.
                        t._dormant_since = now_t
                        self._dormant[t.id] = t
                        del self._tracks[t.id]

            # 4. Spawn new tracks — but first try to revive dormant ones
            _ID_SWITCH_WINDOW_S = 3.0   # seconds; evicted track within this window = likely ID switch
            for di, d in enumerate(valid):
                if di in matched_det_idxs:
                    continue

                # ── Dormant revival attempt ───────────────────────────────────
                # Search the dormant pool for a same-class track whose last bbox
                # overlaps the new detection well enough to be the same object.
                best_dormant: "TrackedObject | None" = None
                best_dormant_iou: float = DORMANT_REVIVAL_IOU_THRESH - 1e-9
                det_bbox = (d.x1, d.y1, d.x2, d.y2)
                for dormant_t in self._dormant.values():
                    if dormant_t.class_name != d.class_name:
                        continue
                    dormant_bbox = (dormant_t._x1, dormant_t._y1,
                                    dormant_t._x2, dormant_t._y2)
                    iou_val = _iou(dormant_bbox, det_bbox)
                    if iou_val > best_dormant_iou:
                        best_dormant_iou = iou_val
                        best_dormant     = dormant_t

                if best_dormant is not None:
                    # Revival: restore track into active pool with original ID.
                    best_dormant.frames_missed = 0
                    best_dormant.frames_seen  += 1
                    best_dormant.confidence    = d.confidence
                    best_dormant.last_seen_t   = now_t
                    best_dormant._dormant_since = 0.0
                    best_dormant.update_bbox(d.x1, d.y1, d.x2, d.y2)
                    best_dormant.confirmed = (best_dormant.frames_seen >= MIN_FRAMES_CONFIRM)
                    best_dormant.direction = _clock_direction(best_dormant.center_x, frame_w)
                    best_dormant.path_overlap_ratio = _corridor_overlap(
                        best_dormant.x1, best_dormant.x2, frame_w
                    )
                    self._tracks[best_dormant.id] = best_dormant
                    del self._dormant[best_dormant.id]
                    logger.debug(
                        "[Tracker] Revived dormant track %d (%s) iou=%.2f.",
                        best_dormant.id, best_dormant.class_name, best_dormant_iou,
                    )
                    _diagnostics.track_revived(best_dormant.id, best_dormant.class_name)
                    continue

                # No dormant revival possible — spawn a fresh tentative track.
                new_id = self._next_id
                t = TrackedObject(
                    id=new_id,
                    class_name=d.class_name,
                    confidence=d.confidence,
                    frames_seen=1,
                )
                t.update_bbox(d.x1, d.y1, d.x2, d.y2)
                t.direction          = _clock_direction(t.center_x, frame_w)
                t.path_overlap_ratio = _corridor_overlap(t.x1, t.x2, frame_w)
                self._tracks[new_id] = t
                self._next_id += 1
                _diagnostics.track_created(new_id, d.class_name)

                # Check if this new track is a likely ID switch
                eviction = self._recent_evictions.get(d.class_name)
                if eviction is not None:
                    old_id, evicted_at = eviction
                    if (time.time() - evicted_at) <= _ID_SWITCH_WINDOW_S:
                        _diagnostics.id_switch(old_id, new_id, d.class_name)
                        del self._recent_evictions[d.class_name]

            return [t for t in self._tracks.values() if t.confirmed]

    def all_tracks(self) -> List[TrackedObject]:
        """Return all tracks including unconfirmed. For debug/diagnostics."""
        with self._lock:
            return list(self._tracks.values())

    def dormant_tracks(self) -> List[TrackedObject]:
        """Return all currently dormant (recently evicted) tracks."""
        with self._lock:
            return list(self._dormant.values())


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
