"""
test_brain_routing.py — Unit tests for brain.py routing fixes.

Tests cover:
  1. MEDICINE_KEYWORDS no longer contains 'bottle' or 'packet'.
     "find the bottle" must NOT route to medicine handler.
  2. Person count uses smoothed_distance_m (not a fallback 'nearby' string).
  3. needs_llm() returns True for medicine questions (holding message plays).
  4. STTEngine.cmd_q is bounded (maxsize=20) and drops with put_nowait.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import queue
import pytest
from unittest.mock import MagicMock, patch
from backend.brain import Brain, MEDICINE_KEYWORDS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_person(smoothed_distance_m=2.5, direction="12 o'clock"):
    """Create a minimal TrackedObject-like mock for a person."""
    p = MagicMock()
    p.class_name = "person"
    p.smoothed_distance_m = smoothed_distance_m
    p.direction = direction
    return p


def _make_det(class_name="chair", distance_level=3, direction="12 o'clock"):
    d = MagicMock()
    d.class_name = class_name
    d.distance_level = distance_level
    d.direction = direction
    d.motion_state = "static"
    return d


# ── MEDICINE_KEYWORDS membership ─────────────────────────────────────────────

class TestMedicineKeywords:
    def test_bottle_not_in_medicine_keywords(self):
        """'bottle' was incorrectly in MEDICINE_KEYWORDS — must be removed."""
        assert "bottle" not in MEDICINE_KEYWORDS, (
            "'bottle' is in MEDICINE_KEYWORDS — 'find the bottle' would route to "
            "medicine handler instead of LLaVA visual search."
        )

    def test_packet_not_in_medicine_keywords(self):
        """'packet' was incorrectly in MEDICINE_KEYWORDS — must be removed."""
        assert "packet" not in MEDICINE_KEYWORDS

    def test_safe_not_in_medicine_keywords(self):
        """'safe' was incorrectly in MEDICINE_KEYWORDS (also in SAFETY_KEYWORDS) — must be removed."""
        assert "safe" not in MEDICINE_KEYWORDS

    def test_take_not_in_medicine_keywords(self):
        """'take' was incorrectly in MEDICINE_KEYWORDS — generic word, causes false routes."""
        assert "take" not in MEDICINE_KEYWORDS

    def test_label_not_in_medicine_keywords(self):
        """'label' was incorrectly in MEDICINE_KEYWORDS — too broad."""
        assert "label" not in MEDICINE_KEYWORDS

    def test_actual_medicine_keywords_present(self):
        """Genuine medicine-only terms must still be present."""
        for word in ("pill", "tablet", "capsule", "paracetamol", "ibuprofen", "aspirin",
                     "medicine", "medication", "prescription"):
            assert word in MEDICINE_KEYWORDS, f"'{word}' should be in MEDICINE_KEYWORDS"


# ── 'bottle' query routes to LLaVA, not medicine ─────────────────────────────

class TestBottleRouting:
    """
    'Find the bottle' / 'Can you see the bottle?' should NOT go to the medicine
    handler (which requires OCR label text).  It should route to LLaVA (visual)
    or the structured LLM — not to _answer_medicine_question.
    """

    def test_find_bottle_does_not_call_medicine_handler(self):
        brain = Brain()
        with patch.object(brain, "_answer_medicine_question") as med_mock, \
             patch.object(brain, "_run_llava", return_value="I see a bottle on the shelf.") as llava_mock, \
             patch.object(brain, "_run_structured_llm", return_value="A bottle is visible."):
            import numpy as np
            dummy_frame = np.zeros((100, 100, 3), dtype="uint8")
            # Provide a detection that gives LLaVA a grounding bbox
            det = MagicMock()
            det.class_name = "bottle"
            det.x1, det.y1, det.x2, det.y2 = 10, 10, 50, 50

            brain.answer("Can you find the bottle?", dummy_frame, [det], [])

        med_mock.assert_not_called(), (
            "_answer_medicine_question was called for 'bottle' query — wrong route!"
        )

    def test_pill_question_still_routes_to_medicine(self):
        brain = Brain()
        with patch.object(brain, "_answer_medicine_question",
                          return_value="The label says Paracetamol 500mg.") as med_mock:
            brain.answer("What pill is this?", None, [], ["Paracetamol 500mg"])

        med_mock.assert_called_once()


# ── Person count uses smoothed_distance_m ────────────────────────────────────

class TestPersonCountDistance:
    """
    Person count route must use smoothed_distance_m (float), not a non-existent
    'distance' attribute (which would always return the fallback string 'nearby').
    """

    def test_single_person_uses_real_distance(self):
        brain = Brain()
        person = _make_person(smoothed_distance_m=3.2, direction="12 o'clock")
        result = brain.answer("How many people are there?", None, [person], [])
        # Must contain the actual distance, not the fallback string
        assert "3.2m" in result, (
            f"Expected '3.2m' in result but got: {result!r} — "
            "person count is using the wrong distance attribute."
        )
        assert "nearby" not in result, (
            f"Result contains 'nearby' fallback — wrong attribute used: {result!r}"
        )

    def test_multiple_people_uses_real_distance(self):
        brain = Brain()
        p1 = _make_person(smoothed_distance_m=1.5, direction="12 o'clock")
        p2 = _make_person(smoothed_distance_m=4.0, direction="3 o'clock")
        result = brain.answer("Are there any people nearby?", None, [p1, p2], [])
        assert "1.5m" in result, f"Expected nearest person distance '1.5m' in: {result!r}"

    def test_person_with_zero_distance_uses_nearby_fallback(self):
        """When smoothed_distance_m == 0 (no depth), 'nearby' is the correct fallback."""
        brain = Brain()
        person = _make_person(smoothed_distance_m=0.0, direction="9 o'clock")
        result = brain.answer("Is anyone here?", None, [person], [])
        # With distance=0 the code should produce "nearby" — that's the correct path
        assert "nearby" in result or "person" in result.lower(), (
            f"Unexpected result for zero-distance person: {result!r}"
        )

    def test_no_people_returns_clear_message(self):
        brain = Brain()
        result = brain.answer("How many people can you see?", None, [], [])
        assert "don't see any people" in result.lower() or "no people" in result.lower() or \
               "people" in result.lower(), f"Unexpected empty-scene result: {result!r}"


# ── needs_llm() medicine gate ─────────────────────────────────────────────────

class TestNeedsLLMMedicine:
    """
    needs_llm() must return True for medicine questions so the pipeline plays
    the 'Let me check that' holding message before the slow phi3:mini call.
    """

    def test_pill_question_needs_llm(self):
        brain = Brain()
        assert brain.needs_llm("What pill is this?") is True

    def test_tablet_question_needs_llm(self):
        brain = Brain()
        # "Is this tablet safe to take?" hits SAFETY_KEYWORDS first ('safe') → fast route.
        # Use a pure medicine query (no safety overlap) to verify the medicine gate.
        assert brain.needs_llm("What tablet is this?") is True

    def test_paracetamol_question_needs_llm(self):
        brain = Brain()
        assert brain.needs_llm("How much paracetamol should I take?") is True

    def test_medicine_question_needs_llm(self):
        brain = Brain()
        assert brain.needs_llm("What medicine is this?") is True

    def test_color_question_does_not_need_llm(self):
        brain = Brain()
        assert brain.needs_llm("What color is this?") is False

    def test_person_count_does_not_need_llm(self):
        brain = Brain()
        assert brain.needs_llm("How many people are there?") is False

    def test_safety_does_not_need_llm(self):
        brain = Brain()
        assert brain.needs_llm("Is it safe to walk?") is False

    def test_direction_does_not_need_llm(self):
        brain = Brain()
        assert brain.needs_llm("Which way should I go?") is False

    def test_general_visual_needs_llm(self):
        brain = Brain()
        assert brain.needs_llm("What is in front of me?") is True


# ── STTEngine cmd_q is bounded ────────────────────────────────────────────────

class TestCmdQBounded:
    """
    stt_engine.cmd_q must have maxsize=20 so it cannot grow without bound
    when the asyncio dispatcher is blocked during slow LLM inference.
    Commands that overflow the queue must be dropped (put_nowait), not block.
    """

    def test_cmd_q_has_maxsize(self):
        from backend.stt import STTEngine
        engine = STTEngine()
        assert engine.cmd_q.maxsize == 20, (
            f"cmd_q.maxsize={engine.cmd_q.maxsize} — expected 20. "
            "An unbounded queue can grow without limit during slow LLM inference."
        )

    def test_cmd_q_module_singleton_has_maxsize(self):
        from backend.stt import stt_engine
        assert stt_engine.cmd_q.maxsize == 20

    def test_cmd_q_does_not_block_when_full(self):
        """
        When cmd_q is full, put_nowait raises queue.Full (not blocking).
        The _transcribe_thread catches this and logs a warning.
        Verify that filling the queue and then calling put_nowait raises Full.
        """
        from backend.stt import STTEngine
        engine = STTEngine()
        # Fill to capacity
        for i in range(engine.cmd_q.maxsize):
            engine.cmd_q.put_nowait({"action": "set_mode", "mode": "NAVIGATE"})
        # Next put_nowait must raise Full, not block
        with pytest.raises(queue.Full):
            engine.cmd_q.put_nowait({"action": "set_mode", "mode": "READ"})
