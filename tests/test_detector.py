"""
test_detector.py — Unit tests for ObjectDetector.

Strategy: ObjectDetector.__init__ calls YOLO(), which requires the model file.
We patch `ultralytics.YOLO` at import time so no model is loaded.
Individual tests then inject a fake `_model.predict()` return value.
"""

import logging
import os
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


# ── Helpers to build fake ultralytics result objects ─────────────────────────

def _make_box(cls_id: int, conf: float, xyxy=(10, 10, 50, 50)):
    """Return a mock box object matching the ultralytics API."""
    box = MagicMock()
    box.cls  = [cls_id]
    box.conf = [conf]
    box.xyxy = [xyxy]
    return box


def _make_result(boxes):
    """Return a mock result object with a .boxes list and .names dict."""
    result = MagicMock()
    result.boxes = boxes
    result.names = {}   # fallback names dict (not used when COCO_CLASSES covers it)
    return result


# ── Fixture: patch YOLO before importing detector ────────────────────────────

class TestObjectDetectorDebugMode(unittest.TestCase):
    """Tests for DEBUG_DETECTIONS raw-logging behaviour."""

    def setUp(self):
        # Patch ultralytics.YOLO so ObjectDetector.__init__ doesn't need a file
        self.yolo_patcher = patch("ultralytics.YOLO", autospec=False)
        mock_yolo_cls = self.yolo_patcher.start()
        self.mock_model = MagicMock()
        self.mock_model.overrides = {}
        mock_yolo_cls.return_value = self.mock_model

        # Force-reload detector so env-var changes take effect
        import importlib
        import backend.detector as det_mod
        importlib.reload(det_mod)
        self.det_mod = det_mod

        from backend.detector import ObjectDetector
        self.detector = ObjectDetector()
        self.detector._model = self.mock_model

        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def tearDown(self):
        self.yolo_patcher.stop()

    # ── Basic functionality ───────────────────────────────────────────────────

    def test_returns_empty_on_none_frame(self):
        result = self.detector.detect(None)
        self.assertEqual(result, [])

    def test_model_predict_error_returns_empty(self):
        self.mock_model.predict.side_effect = RuntimeError("GPU OOM")
        result = self.detector.detect(self.frame)
        self.assertEqual(result, [])

    def test_no_boxes_returns_empty(self):
        self.mock_model.predict.return_value = [_make_result(None)]
        result = self.detector.detect(self.frame)
        self.assertEqual(result, [])

    def test_empty_boxes_returns_empty(self):
        self.mock_model.predict.return_value = [_make_result([])]
        result = self.detector.detect(self.frame)
        self.assertEqual(result, [])

    # ── Confidence gate ───────────────────────────────────────────────────────

    def test_detection_above_gate_passes(self):
        # person = class_id 0, conf 0.90 (well above default 0.60 gate)
        self.mock_model.predict.return_value = [
            _make_result([_make_box(0, 0.90)])
        ]
        result = self.detector.detect(self.frame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].class_name, "person")
        self.assertAlmostEqual(result[0].confidence, 0.90, places=2)

    def test_detection_below_gate_dropped(self):
        # person conf=0.40 < default gate 0.60
        self.mock_model.predict.return_value = [
            _make_result([_make_box(0, 0.40)])
        ]
        result = self.detector.detect(self.frame)
        self.assertEqual(result, [])

    def test_custom_conf_override_lowers_gate(self):
        # With explicit conf=0.35, conf=0.40 person should now pass
        self.mock_model.predict.return_value = [
            _make_result([_make_box(0, 0.40)])
        ]
        result = self.detector.detect(self.frame, conf=0.35)
        self.assertEqual(len(result), 1)

    # ── Whitelist filtering ───────────────────────────────────────────────────

    def test_non_whitelist_class_dropped_by_default(self):
        # class_id 14 = bird (not in ALLOWED_CLASSES)
        self.mock_model.predict.return_value = [
            _make_result([_make_box(14, 0.90)])
        ]
        result = self.detector.detect(self.frame)
        self.assertEqual(result, [])

    def test_non_whitelist_class_passes_with_apply_whitelist_false(self):
        # class_id 14 = bird, but apply_whitelist=False (ASK/FIND mode)
        self.mock_model.predict.return_value = [
            _make_result([_make_box(14, 0.90)])
        ]
        result = self.detector.detect(self.frame, apply_whitelist=False)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].class_name, "bird")

    def test_whitelist_class_passes(self):
        # class_id 56 = chair (in ALLOWED_CLASSES)
        self.mock_model.predict.return_value = [
            _make_result([_make_box(56, 0.90)])
        ]
        result = self.detector.detect(self.frame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].class_name, "chair")

    # ── Multiple detections ───────────────────────────────────────────────────

    def test_mixed_detections_only_valid_pass(self):
        # person@0.90 (passes), bird@0.90 (whitelist fail), chair@0.30 (conf fail)
        self.mock_model.predict.return_value = [
            _make_result([
                _make_box(0,  0.90),   # person — should pass
                _make_box(14, 0.90),   # bird   — whitelist block
                _make_box(56, 0.30),   # chair  — conf block
            ])
        ]
        result = self.detector.detect(self.frame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].class_name, "person")

    # ── Debug logging ─────────────────────────────────────────────────────────

    def test_debug_mode_logs_raw_detections(self):
        """When DEBUG_DETECTIONS=True, a DEBUG log line is emitted per call."""
        import backend.detector as det_mod
        original = det_mod.DEBUG_DETECTIONS
        det_mod.DEBUG_DETECTIONS = True
        self.detector  # already created
        # Patch the module-level flag on the detector module
        with self.assertLogs("backend.detector", level="DEBUG") as cm:
            self.mock_model.predict.return_value = [
                _make_result([_make_box(0, 0.90)])   # person passes
            ]
            # Temporarily enable debug on logger
            det_log = logging.getLogger("backend.detector")
            det_log.setLevel(logging.DEBUG)
            # We need to set the flag on the module and re-use detect() logic
            # via a fresh call; the flag is read inside detect() each call.
            det_mod.DEBUG_DETECTIONS = True
            self.detector.detect(self.frame)
            det_mod.DEBUG_DETECTIONS = original

        # At least one log record mentioning "Raw YOLO"
        combined = " ".join(cm.output)
        self.assertIn("Raw YOLO", combined)

    def test_debug_mode_labels_passed_detection(self):
        """Passed detections are labelled [passed] in debug output."""
        import backend.detector as det_mod
        det_mod.DEBUG_DETECTIONS = True
        with self.assertLogs("backend.detector", level="DEBUG") as cm:
            self.mock_model.predict.return_value = [
                _make_result([_make_box(0, 0.90)])
            ]
            det_mod.DEBUG_DETECTIONS = True
            self.detector.detect(self.frame)
            det_mod.DEBUG_DETECTIONS = False

        combined = " ".join(cm.output)
        self.assertIn("[passed]", combined)

    def test_debug_mode_labels_conf_drop(self):
        """Detections dropped by confidence gate are labelled [below_conf_gate]."""
        import backend.detector as det_mod
        det_mod.DEBUG_DETECTIONS = True
        det_mod.DEBUG_RAW_CONF_MIN = 0.10   # ensure low-conf detection is shown
        with self.assertLogs("backend.detector", level="DEBUG") as cm:
            self.mock_model.predict.return_value = [
                _make_result([_make_box(0, 0.20)])   # below gate=0.60
            ]
            det_mod.DEBUG_DETECTIONS = True
            self.detector.detect(self.frame)
            det_mod.DEBUG_DETECTIONS = False

        combined = " ".join(cm.output)
        self.assertIn("below_conf_gate", combined)

    def test_debug_mode_labels_whitelist_drop(self):
        """Detections dropped by whitelist are labelled [not_in_whitelist]."""
        import backend.detector as det_mod
        det_mod.DEBUG_DETECTIONS = True
        det_mod.DEBUG_RAW_CONF_MIN = 0.10
        with self.assertLogs("backend.detector", level="DEBUG") as cm:
            # bird = class_id 14, not in ALLOWED_CLASSES, conf high enough to pass gate
            self.mock_model.predict.return_value = [
                _make_result([_make_box(14, 0.85)])
            ]
            det_mod.DEBUG_DETECTIONS = True
            self.detector.detect(self.frame)
            det_mod.DEBUG_DETECTIONS = False

        combined = " ".join(cm.output)
        self.assertIn("not_in_whitelist", combined)

    def test_debug_off_does_not_emit_raw_log(self):
        """When DEBUG_DETECTIONS=False, no 'Raw YOLO' debug lines are logged."""
        import backend.detector as det_mod
        det_mod.DEBUG_DETECTIONS = False
        det_log = logging.getLogger("backend.detector")
        det_log.setLevel(logging.DEBUG)
        self.mock_model.predict.return_value = [
            _make_result([_make_box(0, 0.90)])
        ]
        with self.assertLogs("backend.detector", level="DEBUG") as cm:
            self.detector.detect(self.frame)

        combined = " ".join(cm.output)
        self.assertNotIn("Raw YOLO", combined)


# ── TRACKER_CONF_GATE env override ────────────────────────────────────────────

class TestTrackerConfGateEnvVar(unittest.TestCase):
    """TRACKER_CONF_GATE env var should override the tracker's hard gate."""

    def tearDown(self):
        # Always restore tracker module to default state after each test so
        # other test modules that import CONF_GATE are not affected.
        import importlib
        import backend.tracker as trk
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TRACKER_CONF_GATE", None)
            importlib.reload(trk)

    def test_default_gate_is_0_60(self):
        # Without env var, default should be 0.60
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TRACKER_CONF_GATE", None)
            import importlib
            import backend.tracker as trk
            importlib.reload(trk)
            self.assertAlmostEqual(trk.CONF_GATE, 0.60, places=5)

    def test_env_var_lowers_gate(self):
        with patch.dict(os.environ, {"TRACKER_CONF_GATE": "0.35"}):
            import importlib
            import backend.tracker as trk
            importlib.reload(trk)
            self.assertAlmostEqual(trk.CONF_GATE, 0.35, places=5)

    def test_env_var_raises_gate(self):
        with patch.dict(os.environ, {"TRACKER_CONF_GATE": "0.75"}):
            import importlib
            import backend.tracker as trk
            importlib.reload(trk)
            self.assertAlmostEqual(trk.CONF_GATE, 0.75, places=5)


if __name__ == "__main__":
    unittest.main()
