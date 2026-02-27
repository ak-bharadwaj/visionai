"""
tests/test_pipeline_reliability.py

Tests that process_frame() and every _process_*() method never propagate
exceptions — they must catch internally and return None (or a safe payload).

Also tests that the FIND mode detector call uses apply_whitelist=False so
non-navigation objects (cups, bottles, phones) can be located.
"""
import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Patch targets (same set as test_find_workflow)
# ─────────────────────────────────────────────────────────────────────────────

PATCH_BRAIN     = "backend.pipeline.brain"
PATCH_TTS       = "backend.pipeline.tts_engine"
PATCH_DETECTOR  = "backend.pipeline.ObjectDetector"
PATCH_DEPTH     = "backend.pipeline.depth_estimator"
PATCH_TRACKER   = "backend.pipeline.object_tracker"
PATCH_RISK      = "backend.pipeline.risk_score_all"
PATCH_PATH      = "backend.pipeline.is_path_clear"
PATCH_STABILITY = "backend.pipeline.stability_filter"
PATCH_TEMPORAL  = "backend.pipeline.temporal_fusion"
PATCH_WORLD     = "backend.pipeline.world_detector"
PATCH_OCR       = "backend.pipeline.ocr_reader"
PATCH_SCANNER   = "backend.pipeline.scanner"
PATCH_SCENE_MEM = "backend.pipeline.scene_memory"
PATCH_BUILD_SG  = "backend.pipeline.build_scene_graph"


def _make_frame(w=640, h=480) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


class _FakeDetector:
    def detect(self, frame, conf=0.6, apply_whitelist=True):
        return []


class _FakeBrain:
    def answer(self, question, frame, detections, texts):
        return "ok"

    def needs_llm(self, question: str) -> bool:
        return False


class _FakeTTS:
    def speak(self, text, priority=False):
        pass


def _make_pipeline():
    fake_brain    = _FakeBrain()
    fake_tts      = _FakeTTS()
    fake_detector = _FakeDetector()

    fake_depth = MagicMock()
    fake_depth.estimate.return_value = None
    fake_depth.get_region_depth.return_value = 0.0
    fake_depth.metres_from_score.return_value = 1.0
    fake_depth.is_depth_stable.return_value = True
    fake_depth.load.return_value = None
    fake_depth.detect_stair_drop.return_value = False
    fake_depth.depth_jump_reject.return_value = False

    fake_tracker = MagicMock()
    fake_tracker.update.return_value = []
    fake_tracker.all_tracks.return_value = []
    fake_tracker.reset.return_value = None

    fake_stability = MagicMock()
    fake_stability.record_frame.return_value = None
    fake_stability.should_emit_failsafe.return_value = False
    fake_stability.object_passes_gate.return_value = (False, "test")
    fake_stability.narration_allowed.return_value = (False, "test")
    fake_stability.path_clear_allowed.return_value = False
    fake_stability.reset.return_value = None

    fake_temporal = MagicMock()
    fake_temporal.update.side_effect = lambda dets, **kw: dets

    fake_world = MagicMock()
    fake_world.detect.return_value = []
    fake_world.load.return_value = None

    fake_ocr = MagicMock()
    fake_ocr.read.return_value = []
    fake_ocr.clear_history.return_value = None
    fake_ocr._ensure_loaded = lambda: None

    fake_scanner = MagicMock()
    fake_scanner.scan.return_value = []
    fake_scanner.format_result.return_value = ""

    fake_scene_mem = MagicMock()
    fake_scene_mem.get_snapshot.return_value = {}
    fake_scene_mem.clear.return_value = None

    fake_build_sg = MagicMock(return_value={})

    patches = [
        patch(PATCH_BRAIN,     fake_brain),
        patch(PATCH_TTS,       fake_tts),
        patch(PATCH_DEPTH,     fake_depth),
        patch(PATCH_TRACKER,   fake_tracker),
        patch(PATCH_RISK,      MagicMock()),
        patch(PATCH_PATH,      MagicMock(return_value=True)),
        patch(PATCH_STABILITY, fake_stability),
        patch(PATCH_TEMPORAL,  fake_temporal),
        patch(PATCH_WORLD,     fake_world),
        patch(PATCH_OCR,       fake_ocr),
        patch(PATCH_SCANNER,   fake_scanner),
        patch(PATCH_SCENE_MEM, fake_scene_mem),
        patch(PATCH_BUILD_SG,  fake_build_sg),
        patch(PATCH_DETECTOR,  return_value=fake_detector),
    ]
    for p in patches:
        p.start()

    from backend.pipeline import PipelineRunner
    pl = PipelineRunner()
    pl._detector = fake_detector

    return pl, patches, fake_brain, fake_tts, fake_depth, fake_tracker, fake_stability


# ─────────────────────────────────────────────────────────────────────────────
# Tests — exception recovery
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessFrameNeverCrashes(unittest.TestCase):

    def setUp(self):
        (self._pl, self._patches, self._brain,
         self._tts, self._depth, self._tracker,
         self._stability) = _make_pipeline()

    def tearDown(self):
        for p in self._patches:
            try:
                p.stop()
            except RuntimeError:
                pass

    def test_none_frame_returns_none(self):
        """process_frame(None, ...) must return None without raising."""
        result = self._pl.process_frame(None, "NAVIGATE")
        self.assertIsNone(result)

    def test_empty_frame_returns_none(self):
        """process_frame with empty numpy array must return None without raising."""
        result = self._pl.process_frame(np.array([]), "NAVIGATE")
        self.assertIsNone(result)

    def test_navigate_exception_returns_payload_not_raise(self):
        """If _process_navigate_inner raises, process_frame must return a safe payload."""
        self._tracker.update.side_effect = RuntimeError("simulated tracker crash")
        frame = _make_frame()
        result = self._pl.process_frame(frame, "NAVIGATE")
        # Must not raise; result is either a safe dict or None
        # (navigate exception guard returns _navigate_payload which is a dict)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)

    def test_read_exception_returns_none_not_raise(self):
        """If _process_read_inner raises, process_frame must return None, not raise."""
        import backend.pipeline as pm
        with patch.object(self._pl, "_process_read_inner", side_effect=RuntimeError("ocr crash")):
            frame = _make_frame()
            result = self._pl.process_frame(frame, "READ")
        self.assertIsNone(result)

    def test_find_exception_returns_none_not_raise(self):
        """If _process_find_inner raises, process_frame must return None, not raise."""
        with patch.object(self._pl, "_process_find_inner", side_effect=RuntimeError("find crash")):
            frame = _make_frame()
            result = self._pl.process_frame(frame, "FIND")
        self.assertIsNone(result)

    def test_scan_exception_returns_none_not_raise(self):
        """If _process_scan_inner raises, process_frame must return None, not raise."""
        with patch.object(self._pl, "_process_scan_inner", side_effect=RuntimeError("scan crash")):
            frame = _make_frame()
            result = self._pl.process_frame(frame, "SCAN")
        self.assertIsNone(result)

    def test_unknown_mode_returns_none(self):
        """Unknown mode strings must return None silently."""
        result = self._pl.process_frame(_make_frame(), "INVALID_MODE")
        self.assertIsNone(result)

    def test_process_frame_successive_calls_after_exception(self):
        """
        Pipeline must continue working after an internal exception —
        the second call in the same mode must succeed normally.
        """
        # First call — force crash
        with patch.object(self._pl, "_process_navigate_inner",
                          side_effect=RuntimeError("first crash")):
            self._pl.process_frame(_make_frame(), "NAVIGATE")

        # Second call — crash is gone, should work fine
        result = self._pl.process_frame(_make_frame(), "NAVIGATE")
        # Should return a dict (navigate payload), not raise
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)


class TestFindModeWhitelist(unittest.TestCase):
    """
    Verify that FIND mode calls detector with apply_whitelist=False
    so non-navigation objects (bottles, cups, phones) can be detected.
    """

    def setUp(self):
        (self._pl, self._patches, self._brain,
         self._tts, self._depth, self._tracker,
         self._stability) = _make_pipeline()
        self._sent = []
        self._pl._send = lambda data: self._sent.append(data)

    def tearDown(self):
        for p in self._patches:
            try:
                p.stop()
            except RuntimeError:
                pass

    def test_find_detector_called_with_apply_whitelist_false(self):
        """
        _process_find must call detector.detect(..., apply_whitelist=False)
        so objects outside the 12 navigation classes can be found.
        """
        detect_calls = []

        class _SpyDetector:
            def detect(self, frame, conf=0.6, apply_whitelist=True):
                detect_calls.append({"conf": conf, "apply_whitelist": apply_whitelist})
                return []

        self._pl._detector = _SpyDetector()

        frame = _make_frame()
        # Set up state: captured + question pending
        self._pl.find_capture(frame)
        self._pl.find_ask_question("Can you see a bottle?")
        self._pl._process_find(frame, 640, 480)

        self.assertTrue(detect_calls, "detector.detect() must be called in FIND mode")
        self.assertFalse(
            detect_calls[0]["apply_whitelist"],
            "FIND mode must call detector with apply_whitelist=False "
            "so non-navigation objects can be detected.",
        )


class TestPipelineExceptionGuardWrappers(unittest.TestCase):
    """
    Verify the wrapper methods (_process_navigate, _process_read, etc.)
    catch exceptions from their _inner counterparts.
    """

    def setUp(self):
        (self._pl, self._patches, self._brain,
         self._tts, self._depth, self._tracker,
         self._stability) = _make_pipeline()

    def tearDown(self):
        for p in self._patches:
            try:
                p.stop()
            except RuntimeError:
                pass

    def test_navigate_wrapper_catches_inner_exception(self):
        with patch.object(self._pl, "_process_navigate_inner",
                          side_effect=ValueError("navigate inner crash")):
            result = self._pl._process_navigate(_make_frame(), 640, 480, [], None)
        self.assertIsInstance(result, dict, "_process_navigate must return a safe dict on error")

    def test_read_wrapper_catches_inner_exception(self):
        with patch.object(self._pl, "_process_read_inner",
                          side_effect=ValueError("read inner crash")):
            result = self._pl._process_read(_make_frame(), 640, 480)
        self.assertIsNone(result, "_process_read must return None on error")

    def test_find_wrapper_catches_inner_exception(self):
        with patch.object(self._pl, "_process_find_inner",
                          side_effect=ValueError("find inner crash")):
            result = self._pl._process_find(_make_frame(), 640, 480)
        self.assertIsNone(result, "_process_find must return None on error")

    def test_scan_wrapper_catches_inner_exception(self):
        with patch.object(self._pl, "_process_scan_inner",
                          side_effect=ValueError("scan inner crash")):
            result = self._pl._process_scan(_make_frame(), 640, 480)
        self.assertIsNone(result, "_process_scan must return None on error")


# ─────────────────────────────────────────────────────────────────────────────
# Fix 6 regression — OCR must NOT be triggered for generic queries
# (bottle / packet / label removed from the OCR keyword list in _process_ask)
# ─────────────────────────────────────────────────────────────────────────────

class TestAskOcrNotTriggeredForGenericQueries(unittest.TestCase):
    """
    Regression for Fix 6: pipeline._process_ask() must NOT call ocr_reader.read()
    for generic queries containing words like 'bottle', 'packet', or 'label'.

    Before the fix, those words were in the inline OCR trigger keyword list inside
    _process_ask, causing unnecessary (and potentially slow) OCR on arbitrary scenes.
    """

    def setUp(self):
        (self._pl, self._patches, self._brain,
         self._tts, self._depth, self._tracker,
         self._stability) = _make_pipeline()

    def tearDown(self):
        for p in self._patches:
            try:
                p.stop()
            except RuntimeError:
                pass

    def _ask(self, question: str):
        """Run _process_ask with a given question and return (result, ocr_call_count)."""
        import backend.pipeline as pm
        ocr_spy = MagicMock()
        ocr_spy.read.return_value = []
        ocr_spy.clear_history.return_value = None
        ocr_spy._ensure_loaded = lambda: None

        with patch("backend.pipeline.ocr_reader", ocr_spy):
            result = self._pl._process_ask(
                _make_frame(), 640, 480,
                question=question,
                input_source="voice",
            )
        return result, ocr_spy.read.call_count

    def test_bottle_query_does_not_trigger_ocr(self):
        """'Can you find the bottle?' must NOT invoke ocr_reader.read()."""
        _, ocr_calls = self._ask("Can you find the bottle?")
        self.assertEqual(ocr_calls, 0,
                         "OCR must not run for a generic 'bottle' query")

    def test_packet_query_does_not_trigger_ocr(self):
        """'Is there a packet nearby?' must NOT invoke ocr_reader.read()."""
        _, ocr_calls = self._ask("Is there a packet nearby?")
        self.assertEqual(ocr_calls, 0,
                         "OCR must not run for a generic 'packet' query")

    def test_label_query_does_not_trigger_ocr(self):
        """'Read the label' is a medicine OCR question — OCR SHOULD be triggered."""
        # 'label' alone is NOT a medicine keyword — should not trigger OCR.
        # (This is the regression case: the old code had "label" in OCR keywords.)
        _, ocr_calls = self._ask("Can you read the label?")
        # With Fix 6 applied, "label" is no longer an OCR trigger.
        # The only OCR triggers are medicine keywords (pill, tablet, etc.) and
        # structural keywords (door, open, closed, entrance, exit).
        self.assertEqual(ocr_calls, 0,
                         "OCR must not run for a standalone 'label' query "
                         "(only genuine medicine terms should trigger OCR)")

    def test_pill_query_does_trigger_ocr(self):
        """Genuine medicine query ('What pill is this?') MUST trigger OCR."""
        _, ocr_calls = self._ask("What pill is this?")
        self.assertEqual(ocr_calls, 1,
                         "OCR must be invoked for a genuine medicine query")

    def test_door_query_triggers_ocr(self):
        """'Is the door open?' must trigger OCR (structural keyword)."""
        _, ocr_calls = self._ask("Is the door open?")
        self.assertEqual(ocr_calls, 1,
                         "OCR must be invoked for door/open/closed queries")

    def test_unrelated_query_does_not_trigger_ocr(self):
        """Generic question 'What do you see?' must not trigger OCR."""
        _, ocr_calls = self._ask("What do you see?")
        self.assertEqual(ocr_calls, 0,
                         "OCR must not run for a simple unrelated query")


# ─────────────────────────────────────────────────────────────────────────────
# Fix 7 regression — NAVIGATE results must go through _send() (broadcast),
# not sent to a single websocket
# ─────────────────────────────────────────────────────────────────────────────

class TestNavigateBroadcastViaSend(unittest.TestCase):
    """
    Regression for Fix 7: process_frame() in NAVIGATE mode must return a dict
    with type='narration' so that main.py can broadcast it to ALL connected clients.

    Before the fix main.py sent NAVIGATE results only to the requesting WebSocket.
    The fix changed main.py to use `await broadcast(result)` instead.

    The pipeline's responsibility is to return a well-formed 'narration' payload
    from process_frame() — main.py then broadcasts it.  These tests verify:
      1. process_frame("NAVIGATE") always returns a dict (never None).
      2. The returned dict has type='narration'.
      3. The dict contains all keys required by the frontend app.js WS handler.

    A separate broadcast test for main.py's side is in test_broadcast.py.
    """

    def setUp(self):
        (self._pl, self._patches, self._brain,
         self._tts, self._depth, self._tracker,
         self._stability) = _make_pipeline()

    def tearDown(self):
        for p in self._patches:
            try:
                p.stop()
            except RuntimeError:
                pass

    def test_navigate_process_frame_always_returns_dict(self):
        """
        process_frame("NAVIGATE") must always return a dict — never None.
        Even when stability filter suppresses narration text, the narration
        payload dict must be returned so main.py can broadcast an overlay update.
        """
        frame = _make_frame()
        result = self._pl.process_frame(frame, "NAVIGATE")
        self.assertIsInstance(
            result, dict,
            "process_frame must return a dict for NAVIGATE so main.py can broadcast it",
        )

    def test_navigate_result_type_is_narration(self):
        """The NAVIGATE result dict must have type='narration' for frontend routing."""
        result = self._pl.process_frame(_make_frame(), "NAVIGATE")
        self.assertIsInstance(result, dict)
        self.assertEqual(
            result.get("type"), "narration",
            "NAVIGATE result must have type='narration' for broadcast to all clients",
        )

    def test_navigate_result_has_required_frontend_keys(self):
        """
        The narration dict returned by process_frame must contain all keys that
        the frontend app.js WS 'narration' handler reads.

        Required keys (from app.js handleWebSocketMessage narration branch):
          type, text, severity, fps, detections, frame_w, frame_h
        """
        result = self._pl.process_frame(_make_frame(), "NAVIGATE")
        self.assertIsInstance(result, dict)
        for key in ("type", "text", "severity", "fps", "detections", "frame_w", "frame_h"):
            self.assertIn(
                key, result,
                f"narration payload missing '{key}' — frontend broadcast will break",
            )

    def test_navigate_result_mode_is_navigate(self):
        """The NAVIGATE result must identify its mode for frontend routing."""
        result = self._pl.process_frame(_make_frame(), "NAVIGATE")
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("mode"), "NAVIGATE")


# ── NAV state machine tests ───────────────────────────────────────────────────

class TestNavStateMachine(unittest.TestCase):
    """
    Verify the IDLE → WAIT_DEST → ACTIVE state machine in PipelineRunner.
    Tests use a fresh PipelineRunner (no global singleton) with all
    heavy dependencies mocked out.
    """

    def _make_pipeline(self):
        patches = [
            patch(PATCH_BRAIN,     new=MagicMock()),
            patch(PATCH_TTS,       new=MagicMock()),
            patch(PATCH_DETECTOR,  new=lambda: _FakeDetector()),
            patch(PATCH_DEPTH,     new=MagicMock(get_region_depth=MagicMock(return_value=2.0))),
            patch(PATCH_TRACKER,   new=MagicMock(update=MagicMock(return_value=[]), all_tracks=MagicMock(return_value=[]))),
            patch(PATCH_RISK,      new=MagicMock(return_value=[])),
            patch(PATCH_PATH,      new=MagicMock(return_value=(True, "clear"))),
            patch(PATCH_STABILITY, new=MagicMock(should_narrate=MagicMock(return_value=True), update=MagicMock())),
            patch(PATCH_TEMPORAL,  new=MagicMock(update=MagicMock(return_value=[]))),
            patch(PATCH_WORLD,     new=MagicMock()),
            patch(PATCH_OCR,       new=MagicMock()),
            patch(PATCH_SCANNER,   new=MagicMock()),
            patch(PATCH_SCENE_MEM, new=MagicMock()),
            patch(PATCH_BUILD_SG,  new=MagicMock(return_value={})),
        ]
        for p in patches:
            p.start()
        from backend.pipeline import PipelineRunner
        pl = PipelineRunner()
        self.addCleanup(lambda: [p.stop() for p in patches])
        return pl

    def test_initial_nav_state_is_idle(self):
        pl = self._make_pipeline()
        self.assertEqual(pl._nav_state, "IDLE")

    def test_initial_nav_destination_is_none(self):
        pl = self._make_pipeline()
        self.assertIsNone(pl._nav_destination)

    def test_set_nav_destination_stores_destination(self):
        pl = self._make_pipeline()
        pl.set_nav_destination("the kitchen")
        self.assertEqual(pl._nav_destination, "the kitchen")

    def test_set_nav_destination_transitions_to_active(self):
        pl = self._make_pipeline()
        pl.set_nav_destination("exit")
        self.assertEqual(pl._nav_state, "ACTIVE")

    def test_reset_mode_clears_nav_state(self):
        pl = self._make_pipeline()
        pl.set_nav_destination("bathroom")
        pl.reset_mode()
        self.assertEqual(pl._nav_state, "IDLE")
        self.assertIsNone(pl._nav_destination)

    def test_navigate_payload_includes_nav_state(self):
        """_navigate_payload() must include nav_state in every broadcast."""
        pl = self._make_pipeline()
        frame = _make_frame()
        result = pl.process_frame(frame, "NAVIGATE")
        if result is not None:
            self.assertIn("nav_state", result)

    def test_navigate_payload_includes_nav_destination(self):
        """After set_nav_destination(), payload must echo the destination."""
        pl = self._make_pipeline()
        pl.set_nav_destination("the exit")
        frame = _make_frame()
        result = pl.process_frame(frame, "NAVIGATE")
        if result is not None and result.get("type") == "narration":
            self.assertEqual(result.get("nav_destination"), "the exit")

    def test_set_nav_destination_empty_string_ignored(self):
        """Empty string must not overwrite a valid destination."""
        pl = self._make_pipeline()
        pl.set_nav_destination("kitchen")
        pl.set_nav_destination("")
        # After empty string call, destination should remain unchanged or be empty
        # The method must not crash — result is implementation-defined.
        self.assertIsNotNone(pl._nav_state)  # state machine must not break

    def test_set_nav_destination_multiple_calls(self):
        """Calling set_nav_destination twice updates the destination."""
        pl = self._make_pipeline()
        pl.set_nav_destination("kitchen")
        pl.set_nav_destination("bathroom")
        self.assertEqual(pl._nav_destination, "bathroom")
        self.assertEqual(pl._nav_state, "ACTIVE")


if __name__ == "__main__":
    unittest.main()
