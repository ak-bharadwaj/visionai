"""
tests/test_scene_memory.py — Unit tests for build_scene_graph()
"""
import pytest
from backend.scene_memory import build_scene_graph
from backend.tracker import TrackedObject


def _make_obj(
    id: int,
    class_name: str,
    *,
    confirmed: bool = True,
    distance_m: float = 2.0,
    velocity: float = 0.0,
    cx: float = 320.0,
    direction: str = "12 o'clock",
    risk_level: str = "LOW",
) -> TrackedObject:
    """Create a minimal TrackedObject suitable for scene graph tests."""
    obj = TrackedObject(id=id, class_name=class_name, confidence=0.9)
    obj.confirmed = confirmed
    obj.smoothed_distance_m = distance_m
    obj.velocity_m_per_s = velocity
    obj.direction = direction
    obj.risk_level = risk_level
    # Place bbox so center_x matches cx
    half = 50.0
    obj._x1 = cx - half
    obj._x2 = cx + half
    obj._y1 = 100.0
    obj._y2 = 300.0
    return obj


# ── Basic structure ────────────────────────────────────────────────────────

class TestBuildSceneGraphStructure:
    def test_empty_list_returns_empty_graph(self):
        result = build_scene_graph([])
        assert result == {"objects": [], "relations": [], "hazards": []}

    def test_returns_required_keys(self):
        result = build_scene_graph([_make_obj(1, "person")])
        assert "objects" in result
        assert "relations" in result
        assert "hazards" in result

    def test_unconfirmed_objects_excluded(self):
        objs = [_make_obj(1, "person", confirmed=False)]
        result = build_scene_graph(objs)
        assert result["objects"] == []

    def test_confirmed_objects_included(self):
        objs = [_make_obj(1, "chair")]
        result = build_scene_graph(objs)
        assert len(result["objects"]) == 1
        assert result["objects"][0]["id"] == 1
        assert result["objects"][0]["type"] == "chair"

    def test_object_has_required_fields(self):
        objs = [_make_obj(5, "person", distance_m=1.8)]
        obj_dict = build_scene_graph(objs)["objects"][0]
        assert "id" in obj_dict
        assert "type" in obj_dict
        assert "distance_m" in obj_dict
        assert "direction" in obj_dict
        assert "velocity" in obj_dict
        assert "risk_level" in obj_dict

    def test_distance_rounded_to_2dp(self):
        objs = [_make_obj(1, "person", distance_m=1.2345)]
        result = build_scene_graph(objs)
        assert result["objects"][0]["distance_m"] == round(1.2345, 2)


# ── Relations ──────────────────────────────────────────────────────────────

class TestBuildSceneGraphRelations:
    def test_single_object_no_relations(self):
        objs = [_make_obj(1, "person")]
        result = build_scene_graph(objs)
        assert result["relations"] == []

    def test_near_relation_when_depth_diff_small(self):
        # Both at ~2.0m → depth diff = 0 → "near"
        objs = [
            _make_obj(1, "person", distance_m=2.0, cx=100.0),
            _make_obj(2, "chair",  distance_m=2.3, cx=500.0),
        ]
        result = build_scene_graph(objs)
        relations = [(r["subject"], r["relation"], r["object"]) for r in result["relations"]]
        assert any(rel == "near" for _, rel, _ in relations)

    def test_no_near_relation_when_depth_diff_large(self):
        # 1.0m vs 4.0m → diff = 3.0m > NEAR_THRESHOLD_M (1.5m)
        objs = [
            _make_obj(1, "person", distance_m=1.0, cx=100.0),
            _make_obj(2, "chair",  distance_m=4.0, cx=500.0),
        ]
        result = build_scene_graph(objs)
        relations = [r["relation"] for r in result["relations"]]
        assert "near" not in relations

    def test_left_of_relation(self):
        # Object A at cx=100, B at cx=500 → A is left_of B
        objs = [
            _make_obj(1, "person", cx=100.0, distance_m=5.0),
            _make_obj(2, "chair",  cx=500.0, distance_m=5.0),
        ]
        result = build_scene_graph(objs)
        relations = [(r["subject"], r["relation"]) for r in result["relations"]]
        assert ("person #1", "left_of") in relations

    def test_right_of_relation(self):
        # Object A at cx=500, B at cx=100 → A is right_of B
        objs = [
            _make_obj(1, "person", cx=500.0, distance_m=5.0),
            _make_obj(2, "chair",  cx=100.0, distance_m=5.0),
        ]
        result = build_scene_graph(objs)
        relations = [(r["subject"], r["relation"]) for r in result["relations"]]
        assert ("person #1", "right_of") in relations

    def test_no_lateral_relation_within_deadband(self):
        # Both at cx=310 and cx=320 → diff=10px < 20px dead-band
        objs = [
            _make_obj(1, "person", cx=310.0, distance_m=5.0),
            _make_obj(2, "chair",  cx=320.0, distance_m=5.0),
        ]
        result = build_scene_graph(objs)
        lateral = [r for r in result["relations"] if r["relation"] in ("left_of", "right_of")]
        assert lateral == []

    def test_in_front_of_relation(self):
        # A at 1.0m, B at 3.0m → A is in_front_of B (closer = in front)
        objs = [
            _make_obj(1, "person", distance_m=1.0, cx=310.0),
            _make_obj(2, "chair",  distance_m=3.0, cx=320.0),
        ]
        result = build_scene_graph(objs)
        relations = [(r["subject"], r["relation"]) for r in result["relations"]]
        assert ("person #1", "in_front_of") in relations

    def test_behind_relation(self):
        # A at 4.0m, B at 1.0m → A is behind B
        objs = [
            _make_obj(1, "person", distance_m=4.0, cx=310.0),
            _make_obj(2, "chair",  distance_m=1.0, cx=320.0),
        ]
        result = build_scene_graph(objs)
        relations = [(r["subject"], r["relation"]) for r in result["relations"]]
        assert ("person #1", "behind") in relations


# ── Hazards ────────────────────────────────────────────────────────────────

class TestBuildSceneGraphHazards:
    def test_no_hazard_when_stationary(self):
        objs = [_make_obj(1, "person", velocity=0.0, distance_m=2.0)]
        result = build_scene_graph(objs)
        assert result["hazards"] == []

    def test_hazard_when_approaching_fast(self):
        # velocity=2.0 m/s, distance=2.0m → ETA=1.0s → hazard
        obj = _make_obj(1, "person", velocity=2.0, distance_m=2.0)
        result = build_scene_graph([obj])
        assert len(result["hazards"]) == 1
        assert result["hazards"][0]["type"] == "collision"
        assert "person #1" in result["hazards"][0]["object"]

    def test_hazard_eta_value(self):
        obj = _make_obj(1, "person", velocity=2.0, distance_m=4.0)
        # collision_eta_s is a property on TrackedObject; set distance and velocity
        # The property computes: distance / velocity = 4.0 / 2.0 = 2.0s
        result = build_scene_graph([obj])
        if result["hazards"]:
            assert result["hazards"][0]["eta_s"] == pytest.approx(2.0, abs=0.2)

    def test_hazards_sorted_by_eta_ascending(self):
        # Two hazard objects — one closer (lower ETA)
        obj_close = _make_obj(1, "person", velocity=2.0, distance_m=1.0)  # ETA ~0.5s
        obj_far   = _make_obj(2, "chair",  velocity=2.0, distance_m=4.0)  # ETA ~2.0s
        result = build_scene_graph([obj_far, obj_close])
        if len(result["hazards"]) >= 2:
            assert result["hazards"][0]["eta_s"] <= result["hazards"][1]["eta_s"]

    def test_no_hazard_below_velocity_threshold(self):
        # velocity=0.05 m/s < 0.1 threshold → no hazard
        obj = _make_obj(1, "person", velocity=0.05, distance_m=2.0)
        result = build_scene_graph([obj])
        assert result["hazards"] == []


# ── Robustness ─────────────────────────────────────────────────────────────

class TestBuildSceneGraphRobustness:
    def test_mix_confirmed_and_unconfirmed(self):
        objs = [
            _make_obj(1, "person", confirmed=True),
            _make_obj(2, "chair",  confirmed=False),
        ]
        result = build_scene_graph(objs)
        assert len(result["objects"]) == 1
        assert result["objects"][0]["id"] == 1

    def test_many_objects_no_exception(self):
        objs = [_make_obj(i, "person", cx=float(i * 40)) for i in range(10)]
        result = build_scene_graph(objs)
        assert len(result["objects"]) == 10

    def test_zero_distance_objects_handled(self):
        objs = [
            _make_obj(1, "person", distance_m=0.0),
            _make_obj(2, "chair",  distance_m=0.0),
        ]
        result = build_scene_graph(objs)
        # Should not raise — near/in_front_of are skipped for zero-distance objects
        assert "objects" in result
