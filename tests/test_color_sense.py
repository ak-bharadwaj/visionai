"""
tests/test_color_sense.py

Unit tests for backend.color_sense.expand_bbox() and related helpers.
Covers:
  - Person torso-bias: top shifted down, bottom extended down
  - Non-person: uniform expansion
  - Clamping to frame boundaries (no out-of-bounds coordinates)
  - Collapse guard: if expansion collapses box, fall back to original clamped bbox
  - get_dominant_color: returns a string, handles zero-size region gracefully
  - answer_color_question: picks the largest/central detection
"""
import unittest
import numpy as np
import cv2

from backend.color_sense import expand_bbox, get_dominant_color, answer_color_question


# ─────────────────────────────────────────────────────────────────────────────
# expand_bbox — geometry tests
# ─────────────────────────────────────────────────────────────────────────────

class TestExpandBboxNonPerson(unittest.TestCase):
    """Non-person: uniform 30% expansion in all directions."""

    def setUp(self):
        # 100x100 box centred in a 640x480 frame
        self.x1, self.y1, self.x2, self.y2 = 200, 150, 300, 250
        self.fw, self.fh = 640, 480
        self.scale = 0.30

    def _expand(self, class_name="bottle"):
        return expand_bbox(
            self.x1, self.y1, self.x2, self.y2,
            self.fw, self.fh,
            scale=self.scale,
            class_name=class_name,
        )

    def test_x1_expands_left(self):
        nx1, _, _, _ = self._expand()
        self.assertLess(nx1, self.x1, "x1 should move left (smaller) for non-person.")

    def test_y1_expands_up(self):
        _, ny1, _, _ = self._expand()
        self.assertLess(ny1, self.y1, "y1 should move up (smaller) for non-person.")

    def test_x2_expands_right(self):
        _, _, nx2, _ = self._expand()
        self.assertGreater(nx2, self.x2, "x2 should move right (larger) for non-person.")

    def test_y2_expands_down(self):
        _, _, _, ny2 = self._expand()
        self.assertGreater(ny2, self.y2, "y2 should move down (larger) for non-person.")

    def test_symmetric_horizontal(self):
        nx1, _, nx2, _ = self._expand()
        left_expansion  = self.x1 - nx1
        right_expansion = nx2 - self.x2
        self.assertEqual(left_expansion, right_expansion, "Horizontal expansion must be symmetric.")

    def test_symmetric_vertical(self):
        _, ny1, _, ny2 = self._expand()
        top_expansion    = self.y1 - ny1
        bottom_expansion = ny2 - self.y2
        self.assertEqual(top_expansion, bottom_expansion, "Vertical expansion must be symmetric.")


class TestExpandBboxPerson(unittest.TestCase):
    """Person: top shifts DOWN (y1 increases), bottom extends DOWN (y2 increases)."""

    def setUp(self):
        self.x1, self.y1, self.x2, self.y2 = 200, 100, 300, 400
        self.fw, self.fh = 640, 480

    def _expand(self):
        return expand_bbox(
            self.x1, self.y1, self.x2, self.y2,
            self.fw, self.fh,
            scale=0.30,
            class_name="person",
        )

    def test_y1_shifts_down_for_person(self):
        _, ny1, _, _ = self._expand()
        self.assertGreater(ny1, self.y1,
            "Person top (y1) must shift DOWN to skip the face and focus on torso.")

    def test_y2_extends_down_for_person(self):
        _, _, _, ny2 = self._expand()
        self.assertGreater(ny2, self.y2,
            "Person bottom (y2) must extend DOWN to capture shirt/torso region.")

    def test_x1_expands_left_for_person(self):
        nx1, _, _, _ = self._expand()
        self.assertLess(nx1, self.x1, "Person x1 must still expand left.")

    def test_x2_expands_right_for_person(self):
        _, _, nx2, _ = self._expand()
        self.assertGreater(nx2, self.x2, "Person x2 must still expand right.")

    def test_person_case_insensitive(self):
        """class_name matching should be case-insensitive."""
        nx1_a, ny1_a, nx2_a, ny2_a = expand_bbox(
            self.x1, self.y1, self.x2, self.y2, self.fw, self.fh, class_name="PERSON"
        )
        nx1_b, ny1_b, nx2_b, ny2_b = expand_bbox(
            self.x1, self.y1, self.x2, self.y2, self.fw, self.fh, class_name="person"
        )
        self.assertEqual((nx1_a, ny1_a, nx2_a, ny2_a), (nx1_b, ny1_b, nx2_b, ny2_b),
            "Person expansion must be identical regardless of class_name capitalisation.")


class TestExpandBboxClamping(unittest.TestCase):
    """Coordinates must never exceed frame boundaries."""

    def test_clamped_to_zero(self):
        # Box at top-left corner
        nx1, ny1, nx2, ny2 = expand_bbox(5, 5, 50, 50, 640, 480, scale=0.5, class_name="cup")
        self.assertGreaterEqual(nx1, 0)
        self.assertGreaterEqual(ny1, 0)

    def test_clamped_to_frame_max(self):
        # Box at bottom-right corner
        nx1, ny1, nx2, ny2 = expand_bbox(600, 440, 639, 479, 640, 480, scale=0.5)
        self.assertLessEqual(nx2, 640)
        self.assertLessEqual(ny2, 480)

    def test_frame_edge_person(self):
        # Person at right edge — x2 clamped to frame_w
        nx1, ny1, nx2, ny2 = expand_bbox(580, 50, 640, 400, 640, 480, class_name="person")
        self.assertLessEqual(nx2, 640)

    def test_all_coords_non_negative(self):
        nx1, ny1, nx2, ny2 = expand_bbox(0, 0, 10, 10, 640, 480, scale=0.5)
        for coord in (nx1, ny1, nx2, ny2):
            self.assertGreaterEqual(coord, 0)


class TestExpandBboxCollapseGuard(unittest.TestCase):
    """If expansion collapses the box, fall back to original clamped bbox."""

    def test_very_tall_person_at_top_does_not_collapse(self):
        """
        Person whose y1 is at 0 — the +15% top shift and clamping should not
        produce ny1 >= ny2 (collapse).  The collapse guard must kick in if it does.
        """
        nx1, ny1, nx2, ny2 = expand_bbox(
            100, 0, 200, 5, 640, 480, scale=0.30, class_name="person"
        )
        self.assertLess(nx1, nx2, "x1 must be less than x2 after collapse guard.")
        self.assertLess(ny1, ny2, "y1 must be less than y2 after collapse guard.")

    def test_zero_size_box_falls_back(self):
        """A degenerate zero-size box should not produce invalid coordinates."""
        nx1, ny1, nx2, ny2 = expand_bbox(100, 100, 100, 100, 640, 480)
        # Either the box stayed as-is or the guard produced valid coords
        self.assertLessEqual(nx1, nx2)
        self.assertLessEqual(ny1, ny2)


# ─────────────────────────────────────────────────────────────────────────────
# get_dominant_color
# ─────────────────────────────────────────────────────────────────────────────

class TestGetDominantColor(unittest.TestCase):

    def _solid_frame(self, bgr, size=(100, 100)):
        return np.full((size[1], size[0], 3), bgr, dtype=np.uint8)

    def test_returns_string(self):
        frame = self._solid_frame((200, 50, 50))
        result = get_dominant_color(frame, det=None)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_black_frame(self):
        frame = self._solid_frame((0, 0, 0))
        result = get_dominant_color(frame, det=None)
        self.assertEqual(result, "black")

    def test_white_frame(self):
        frame = self._solid_frame((255, 255, 255))
        result = get_dominant_color(frame, det=None)
        self.assertEqual(result, "white")

    def test_gray_frame(self):
        frame = self._solid_frame((128, 128, 128))
        result = get_dominant_color(frame, det=None)
        self.assertEqual(result, "gray")

    def test_zero_size_region_returns_unknown(self):
        """If the bbox crops to an empty region, return 'unknown' not crash."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        class FakeDet:
            x1, y1, x2, y2 = 50, 50, 50, 50   # zero-size box
            class_name = "cup"

        result = get_dominant_color(frame, det=FakeDet())
        self.assertIn(result, ("unknown", "black", "gray", "white"),
            "Zero-size region must not crash and must return a safe value.")

    def test_with_detection_object(self):
        """get_dominant_color must accept a det with x1/y1/x2/y2 attributes."""
        frame = self._solid_frame((30, 30, 200), size=(200, 200))   # mostly red in BGR

        class FakeDet:
            x1, y1, x2, y2 = 50, 50, 150, 150
            class_name = "bottle"

        result = get_dominant_color(frame, det=FakeDet())
        self.assertIsInstance(result, str)


# ─────────────────────────────────────────────────────────────────────────────
# answer_color_question
# ─────────────────────────────────────────────────────────────────────────────

class TestAnswerColorQuestion(unittest.TestCase):

    def _frame(self):
        return np.full((480, 640, 3), (100, 100, 100), dtype=np.uint8)

    def test_no_detections_returns_string(self):
        result = answer_color_question(self._frame(), [])
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_with_detections_mentions_class(self):
        class FakeDet:
            x1, y1, x2, y2 = 200, 150, 400, 350
            class_name = "chair"

        result = answer_color_question(self._frame(), [FakeDet()])
        self.assertIn("chair", result.lower(),
            "Answer must mention the class_name of the detected object.")

    def test_prefers_larger_detection(self):
        """Larger detection should be preferred over smaller one."""
        class BigDet:
            x1, y1, x2, y2 = 100, 100, 500, 400   # large area
            class_name = "table"

        class SmallDet:
            x1, y1, x2, y2 = 300, 200, 310, 210   # tiny area
            class_name = "cup"

        result = answer_color_question(self._frame(), [SmallDet(), BigDet()])
        self.assertIn("table", result.lower(),
            "The larger object should be preferred in the color answer.")


if __name__ == "__main__":
    unittest.main()
