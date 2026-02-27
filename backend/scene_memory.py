"""
scene_memory.py — Scene state tracking for VisionTalk non-navigate modes.

In NAVIGATE mode, object state is managed by tracker.py (per-track ID with
full temporal stability). This module is retained for:
  - ASK / READ / FIND modes: snapshot and scene-diff queries.
  - WS handler: take_snapshot() / get_scene_diff() surface API.
  - YOLOWorld dedup in NAVIGATE (is_new_by_key TTL gate).
  - Scene graph: spatial relations between tracked objects.

Approach detection has been removed — velocity is now computed by tracker.py.

Scene Graph
-----------
build_scene_graph(tracked_objects) computes pairwise spatial relations between
confirmed tracked objects and returns a structured dict consumable by the
frontend and brain.answer() for richer context.

Relations computed:
  - "near"      : centres within NEAR_THRESHOLD metres of each other
  - "left_of"   : object A centre_x < object B centre_x (in frame coordinates)
  - "right_of"  : object A centre_x > object B centre_x
  - "in_front"  : object A closer to camera than object B (lower distance_m)
  - "behind"    : object A further from camera than object B

Output schema:
  {
    "objects": [
      {"id": 3, "type": "chair", "distance_m": 1.3, "direction": "12 o'clock",
       "velocity": "approaching"},
      ...
    ],
    "relations": [
      {"subject": "person #12", "relation": "near", "object": "chair #3"},
      {"subject": "chair #3",   "relation": "left_of", "object": "door #7"},
      ...
    ],
    "hazards": [
      {"type": "collision", "object": "person #12", "eta_s": 1.5}
    ]
  }
"""

import time
import threading
import logging
from dataclasses import dataclass
from typing import List

from backend.spatial import SpatialResult

logger = logging.getLogger(__name__)

# ── Scene graph tuning ────────────────────────────────────────────────────────
# Two objects are considered "near" when their centres are within this distance.
NEAR_THRESHOLD_M = 1.5   # metres


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


def build_scene_graph(tracked_objects: list) -> dict:
    """
    Build a structured scene graph from a list of TrackedObjects.

    Returns a dict with:
      - objects : list of object dicts (id, type, distance_m, direction, velocity)
      - relations: list of pairwise spatial relation dicts
      - hazards  : list of imminent collision hazard dicts

    Only confirmed objects (obj.confirmed == True) are included.
    Safe to call with an empty list — returns empty graph.
    """
    try:
        confirmed = [o for o in tracked_objects if getattr(o, "confirmed", False)]

        # ── Object list ───────────────────────────────────────────────────────
        objects = []
        for o in confirmed:
            objects.append({
                "id":         o.id,
                "type":       o.class_name,
                "distance_m": round(o.smoothed_distance_m, 2),
                "direction":  o.direction,
                "velocity":   o.motion_state,
                "risk_level": o.risk_level,
            })

        # ── Relations (pairwise) ──────────────────────────────────────────────
        relations = []
        for i, a in enumerate(confirmed):
            for b in confirmed[i + 1:]:
                # Estimate lateral distance from frame centre positions
                # centre_x is in pixels; we use it as a proxy for left/right
                cx_a = getattr(a, "center_x", (a.x1 + a.x2) / 2)
                cx_b = getattr(b, "center_x", (b.x1 + b.x2) / 2)

                dist_a = a.smoothed_distance_m
                dist_b = b.smoothed_distance_m

                label_a = f"{a.class_name} #{a.id}"
                label_b = f"{b.class_name} #{b.id}"

                # Near — use depth distance difference as a proxy
                if dist_a > 0 and dist_b > 0:
                    depth_diff = abs(dist_a - dist_b)
                    if depth_diff <= NEAR_THRESHOLD_M:
                        relations.append({
                            "subject":  label_a,
                            "relation": "near",
                            "object":   label_b,
                        })

                # Left / right (frame-relative, not world-relative)
                if cx_a < cx_b - 20:   # 20 px dead-band to avoid noise
                    relations.append({
                        "subject":  label_a,
                        "relation": "left_of",
                        "object":   label_b,
                    })
                elif cx_b < cx_a - 20:
                    relations.append({
                        "subject":  label_a,
                        "relation": "right_of",
                        "object":   label_b,
                    })

                # In front / behind (closer = in front)
                if dist_a > 0 and dist_b > 0:
                    if dist_a < dist_b - 0.5:
                        relations.append({
                            "subject":  label_a,
                            "relation": "in_front_of",
                            "object":   label_b,
                        })
                    elif dist_b < dist_a - 0.5:
                        relations.append({
                            "subject":  label_a,
                            "relation": "behind",
                            "object":   label_b,
                        })

        # ── Hazards (imminent collisions) ─────────────────────────────────────
        hazards = []
        for o in confirmed:
            eta = getattr(o, "collision_eta_s", 0.0)
            if eta > 0.0 and o.velocity_m_per_s >= 0.1:
                hazards.append({
                    "type":   "collision",
                    "object": f"{o.class_name} #{o.id}",
                    "eta_s":  round(eta, 1),
                })
        # Sort hazards by ETA ascending (most urgent first)
        hazards.sort(key=lambda h: h["eta_s"])

        return {
            "objects":   objects,
            "relations": relations,
            "hazards":   hazards,
        }

    except Exception as exc:
        logger.error("[SceneMemory] build_scene_graph error: %s", exc)
        return {"objects": [], "relations": [], "hazards": []}


scene_memory = SceneMemory()
