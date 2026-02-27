"""
scene_memory.py — Scene state tracking for VisionTalk non-navigate modes.

In NAVIGATE mode, object state is managed by tracker.py (per-track ID with
full temporal stability). This module is retained for:
  - ASK / READ / FIND modes: snapshot and scene-diff queries.
  - WS handler: take_snapshot() / get_scene_diff() surface API.
  - YOLOWorld dedup in NAVIGATE (is_new_by_key TTL gate).

Approach detection has been removed — velocity is now computed by tracker.py.
"""

import time
import threading
import logging
from dataclasses import dataclass
from typing import List

from backend.spatial import SpatialResult

logger = logging.getLogger(__name__)


@dataclass
class SceneEntry:
    result:     SpatialResult
    first_seen: float
    last_seen:  float


class SceneMemory:
    TTL = 12.0   # seconds before a missing object is evicted from memory

    def __init__(self):
        self._entries:  dict[str, SceneEntry] = {}
        self._lock:     threading.Lock = threading.Lock()
        self._announced: dict[str, float] = {}
        self._ttl: float = 12.0  # dedup TTL for is_new_by_key

    # ── Called by pipeline for non-navigate modes ─────────────────
    def update(self, results: List[SpatialResult]):
        """Update scene entries from a list of SpatialResult (non-navigate)."""
        now = time.time()
        with self._lock:
            for r in results:
                k = r.key
                if k in self._entries:
                    self._entries[k].last_seen = now
                    self._entries[k].result    = r
                else:
                    self._entries[k] = SceneEntry(
                        result=r, first_seen=now, last_seen=now
                    )
            # Evict stale entries
            stale = [k for k, e in self._entries.items()
                     if now - e.last_seen > self.TTL]
            for k in stale:
                del self._entries[k]

    def get_snapshot(self) -> dict:
        """Return a frozen {key: class_name} dict for scene-diff queries."""
        with self._lock:
            return {k: e.result.class_name for k, e in self._entries.items()}

    def clear(self):
        with self._lock:
            self._entries.clear()

    # ── TTL dedup gate (used by YOLOWorld extra detections) ───────
    def is_new_by_key(self, key: str) -> bool:
        """
        Returns True if *key* has not been announced within the dedup TTL.
        Side-effect: records the announcement timestamp if returning True.
        """
        now = time.time()
        with self._lock:
            if key in self._announced and now - self._announced[key] < self._ttl:
                return False
            self._announced[key] = now
            # Prune stale entries (prevent unbounded growth)
            prune_before = now - self._ttl * 2
            stale = [k for k, t in self._announced.items() if t < prune_before]
            for k in stale:
                del self._announced[k]
        return True


scene_memory = SceneMemory()
