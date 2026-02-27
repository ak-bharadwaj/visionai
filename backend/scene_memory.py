import time, threading, logging
from collections import deque
from dataclasses import dataclass, field
from typing import List
from backend.spatial import SpatialResult

logger = logging.getLogger(__name__)


@dataclass
class SceneEntry:
    result:        SpatialResult
    first_seen:    float
    last_seen:     float
    announced:     bool
    last_level:    int
    # NEW: track last 3 distance_levels to detect approaching movement
    level_history: deque = field(default_factory=lambda: deque(maxlen=3))
    approach_warned: bool = False   # don't repeat approach alert


class SceneMemory:
    TTL = 8.0

    def __init__(self):
        self._entries: dict[str, SceneEntry] = {}
        self._lock = threading.Lock()
        self._announced: dict[str, float] = {}
        self._ttl: float = 8.0   # seconds between repeated announcements

    def update(self, results: List[SpatialResult]):
        now = time.time()
        with self._lock:
            for r in results:
                k = r.key
                if k in self._entries:
                    e = self._entries[k]
                    e.last_seen = now
                    e.result    = r
                    e.level_history.append(r.distance_level)
                    if abs(r.distance_level - e.last_level) >= 1:
                        e.announced      = False
                        e.approach_warned = False   # reset on level change
                        e.last_level     = r.distance_level
                else:
                    entry = SceneEntry(
                        result=r, first_seen=now, last_seen=now,
                        announced=False, last_level=r.distance_level,
                    )
                    entry.level_history.append(r.distance_level)
                    self._entries[k] = entry

            # SAFE expire: build list first, THEN delete
            expired = [k for k, e in self._entries.items()
                       if now - e.last_seen > self.TTL]
            for k in expired:
                del self._entries[k]

    def get_new_alerts(self) -> List[SpatialResult]:
        with self._lock:
            return [e.result for e in self._entries.values() if not e.announced]

    def get_approaching(self) -> List[SpatialResult]:
        """
        NEW FEATURE — Approaching Alert.
        Returns objects where distance_level decreased 2 steps in last 3 frames.
        Pattern: [4,3,2] or [3,2,1] = object is approaching.
        Only fires if object is now at level <= 2 (nearby or very close).
        """
        approaching = []
        with self._lock:
            for e in self._entries.values():
                h = list(e.level_history)
                if len(h) >= 3:
                    # strictly decreasing AND now close
                    if h[-1] < h[-2] < h[-3] and h[-1] <= 2 and not e.approach_warned:
                        approaching.append(e.result)
        return approaching

    def mark_approach_warned(self, key: str):
        with self._lock:
            if key in self._entries:
                self._entries[key].approach_warned = True

    def mark_announced(self, key: str):
        with self._lock:
            if key in self._entries:
                self._entries[key].announced = True

    def get_snapshot(self) -> dict:
        """Returns a frozen copy of current scene for comparison."""
        with self._lock:
            return {k: e.result.class_name for k, e in self._entries.items()}

    def clear(self):
        with self._lock:
            self._entries.clear()

    def is_new_by_key(self, key: str) -> bool:
        """Returns True if this key hasn't been announced recently (TTL dedup)."""
        now = time.time()
        if key in self._announced and now - self._announced[key] < self._ttl:
            return False
        self._announced[key] = now
        # FIX #11: prune stale entries to prevent unbounded dict growth
        prune_before = now - self._ttl * 2
        stale = [k for k, t in self._announced.items() if t < prune_before]
        for k in stale:
            del self._announced[k]
        return True


scene_memory = SceneMemory()
