import base64
import threading
import urllib.request, urllib.error, json, logging, os
from typing import List
from collections import deque

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Keyword sets ──────────────────────────────────────────────────────────────
COLOR_KEYWORDS    = {"color", "colour", "shade", "hue", "what color"}
MEDICINE_KEYWORDS = {"medicine", "medication", "drug", "pill", "tablet", "capsule",
                     "dose", "dosage", "prescription", "paracetamol", "ibuprofen", "aspirin"}
PERSON_COUNT_KEYWORDS = {"people", "person", "anyone", "anybody",
                         "someone", "somebody", "how many", "crowd"}
SAFETY_KEYWORDS   = {"safe", "walk", "move", "proceed", "step"}
DIRECTION_KEYWORDS = {"which way", "which direction", "turn left", "turn right",
                      "go left", "go right", "left or right",
                      "where should i go", "where do i go"}
# FIND mode: question prefixes that indicate object-location queries
FIND_PREFIXES = {"where is", "where's", "find", "locate", "can you find",
                 "can you see", "do you see", "show me", "look for",
                 "is there a", "is there an"}


def _clock_from_center_x(center_x: float, frame_w: int) -> str:
    """
    Compute a clock-position direction string from an object's horizontal
    centre pixel and the frame width.  Mirrors tracker._clock_direction().
    Used for raw Detection objects which have no .direction attribute.
    """
    ratio = center_x / max(frame_w, 1)
    if ratio < 0.15: return "9 o'clock"
    if ratio < 0.30: return "10 o'clock"
    if ratio < 0.45: return "11 o'clock"
    if ratio < 0.55: return "12 o'clock"
    if ratio < 0.70: return "1 o'clock"
    if ratio < 0.85: return "2 o'clock"
    return "3 o'clock"


def _clock_to_user_guidance(clock: str) -> tuple:
    """
    Convert a clock-position string to (side_label, action_instruction).

    Returns a tuple:
      side_label  — e.g. "to your left", "straight ahead", "to your right"
      action      — e.g. "Turn left to face it.", "It is directly ahead of you."
    """
    if "9" in clock or "10" in clock:
        return "to your left", "Turn left to face it."
    if "11" in clock:
        return "slightly to your left", "Turn slightly left to face it."
    if "12" in clock:
        return "straight ahead", "It is directly in front of you."
    if "1" in clock:
        return "slightly to your right", "Turn slightly right to face it."
    if "2" in clock or "3" in clock:
        return "to your right", "Turn right to face it."
    return "ahead", "It is in front of you."

# Questions about visual appearance that require actual image analysis (LLaVA).
# phi3:mini has no image input — these questions would get garbage answers from
# structured JSON alone.  Route them to LLaVA with grounded crop.
VISUAL_APPEARANCE_KEYWORDS = {
    "wearing", "shirt", "tshirt", "t-shirt", "jacket", "coat", "dress",
    "clothes", "clothing", "outfit", "uniform", "hat", "cap", "glasses",
    "id", "badge", "card", "lanyard", "bag", "backpack", "holding",
    "face", "hair", "beard", "beard", "tall", "short", "sitting", "standing",
    "reading", "doing", "look like", "looks like", "appearance",
    "open", "closed", "sign", "writing", "text on", "label on",
}

# Max conversation turns kept in memory (each turn = question + answer)
MAX_HISTORY = 4


class Brain:
    """
    On-device AI reasoning for ASK / FIND / READ modes.

    Routing priority (in order):
      1. Color        → OpenCV HSV (fast, no LLM needed)
      2. Medicine     → OCR text + phi3:mini (label-reading, no vision needed)
      3. Person count → YOLO detections (instant)
      4. Safety       → YOLO detections (instant)
      5. Direction    → YOLO detections (instant)
      6. Everything else (scene description, location, door, general visual
                          questions) → LLaVA with actual camera frame

    LLaVA is a multimodal model that can actually see the image.
    phi3:mini is used only for medicine label reading (OCR text → LLM).
    """

    def __init__(self):
        self._text_model   = os.getenv("OLLAMA_MODEL",        "phi3:mini")
        self._vision_model = os.getenv("OLLAMA_VISION_MODEL", "llava")
        # Conversation history: deque of (question, answer) tuples
        self._history: deque = deque(maxlen=MAX_HISTORY)
        # Generation counter — bumped on every clear_history() call.
        # answer() captures this before any slow LLM I/O and only appends
        # to history if the counter hasn't changed (i.e. no clear fired
        # while the LLM was running).  Prevents stale answers from
        # re-populating history after a mode switch clears it.
        self._history_gen: int = 0
        self._history_lock: threading.Lock = threading.Lock()

    def clear_history(self):
        """Reset conversation memory. Call when leaving ASK mode or on demand."""
        with self._history_lock:
            self._history.clear()
            self._history_gen += 1

    def get_history(self) -> list:
        """Returns conversation history as list of dicts for frontend."""
        return [{"question": q, "answer": a} for q, a in self._history]

    def _history_append(self, question: str, result: str, gen: int) -> None:
        """
        Thread-safe conditional history append.

        Only appends if the history generation counter still matches `gen`,
        i.e. clear_history() has NOT been called since `gen` was captured.
        This prevents a stale LLM answer from re-populating history after a
        mode switch has already cleared it.
        """
        with self._history_lock:
            if self._history_gen == gen:
                self._history.append((question, result))
            else:
                logger.debug(
                    "[Brain] Skipping history append — history was cleared "
                    "while LLM was running (gen %d → %d).",
                    gen, self._history_gen,
                )

    def needs_llm(self, question: str) -> bool:
        """
        Returns True if this question needs a slow LLM/vision call.
        Used by pipeline.py to decide whether to play the 'Let me check that'
        holding message — we don't want it on instant answers.
        """
        q_lower = question.lower()
        q_words = set(q_lower.split())
        # Fast routes — no holding message needed
        if any(k in q_lower for k in COLOR_KEYWORDS):         return False
        if any(k in q_lower for k in PERSON_COUNT_KEYWORDS):  return False
        if any(k in q_words  for k in SAFETY_KEYWORDS):       return False
        if any(k in q_lower  for k in DIRECTION_KEYWORDS):    return False
        # Scene description / visual questions → LLaVA (slow, needs holding message)
        # These require actual image analysis and cannot be answered from structured
        # YOLO data alone — they MUST go to the vision model.
        VISUAL_DESCRIBE = {"describe", "what do you see", "what's around",
                           "what is around", "scene", "around me",
                           "what's in front", "overview", "what can you see",
                           "what is in front", "what is around", "tell me what",
                           "look around", "show me", "what's there"}
        if any(k in q_lower for k in VISUAL_DESCRIBE):        return True
        # Medicine questions use phi3:mini (slow) — play holding message
        if any(k in q_words  for k in MEDICINE_KEYWORDS):     return True
        # Everything else goes to LLaVA or phi3 — play the holding message
        return True

    def _needs_visual_model(self, question: str) -> bool:
        """
        Returns True if the question requires actual image analysis (LLaVA),
        not just structured JSON from YOLO (phi3:mini).

        Questions about clothing, appearance, ID cards, signs, whether a door
        is open, scene descriptions, and general "what do you see" queries
        cannot be answered from bounding-box data alone.
        """
        q_lower = question.lower()
        # Scene description / overview questions — always need vision model
        SCENE_DESCRIBE_KEYWORDS = {
            "describe", "what do you see", "what's around", "what is around",
            "scene", "around me", "what's in front", "overview",
            "what can you see", "what is in front", "tell me what",
            "look around", "what's there", "what is there", "what's happening",
            "what is happening", "give me an overview",
        }
        if any(k in q_lower for k in SCENE_DESCRIBE_KEYWORDS):
            return True
        return any(k in q_lower for k in VISUAL_APPEARANCE_KEYWORDS)

    def answer(self, question: str, frame,
               detections: list, texts: List[str]) -> str:
        """
        Main entry point for answering a user question.

        Args:
            question:   The user's question string.
            frame:      Current camera frame (numpy BGR) — may be None.
            detections: List of Detection or TrackedObject instances.
            texts:      OCR text strings from the current frame.
        """
        q_lower = question.lower()
        q_words = set(q_lower.split())

        # Snapshot the history generation counter before any slow I/O.
        # If clear_history() fires while an LLM call is in flight we will
        # detect the generation bump and skip the stale append.
        with self._history_lock:
            _gen = self._history_gen

        # ── 0. FIND — locate a named object instantly from YOLO bbox ────────────
        # Triggered when the question contains a "find/where is/locate" prefix
        # AND detections are present.  Works with raw Detection objects (no
        # .direction attribute) by computing direction from center_x / frame_w.
        # This is the primary fast path for FIND mode questions.
        if any(k in q_lower for k in FIND_PREFIXES):
            logger.info("[Brain] route=find_object q=%r", question[:80])
            frame_w = frame.shape[1] if frame is not None else 640
            # Extract target name from question for LLaVA grounding
            _stop = {"where", "is", "the", "a", "an", "can", "you", "see", "find",
                     "locate", "look", "for", "show", "me", "there", "do",
                     "please", "i", "want", "to", "need", "are"}
            _target_words = [w.strip("?.,!") for w in q_lower.split()
                             if w.strip("?.,!") and w.strip("?.,!") not in _stop]
            _find_target = " ".join(_target_words[:3]) if _target_words else ""
            if detections:
                result = self._answer_find_object(question, detections, frame_w)
                if result is not None:
                    self._history_append(question, result, _gen)
                    return result
                logger.info("[Brain] find_object: no YOLO match — routing to LLaVA for visual search")
            # No matching YOLO detection (or no detections at all) — use LLaVA
            # to visually search the full image.  Pass find_target so the prompt
            # tells LLaVA exactly what to look for (full frame, no cropping).
            if frame is not None:
                result = self._run_llava(question, frame, detections, texts,
                                         find_target=_find_target)
                self._history_append(question, result, _gen)
                return result

        # ── 1. Color — Gemini AI or OpenCV (accuracy priority) ─────────────────────────────
        if any(k in q_lower for k in COLOR_KEYWORDS):
            logger.info("[Brain] route=color q=%r", question[:80])
            if frame is not None:
                try:
                    # Determine the target object from detections or question
                    target_obj = "person"  # default
                    if detections:
                        # Pick the most central/largest detection as the subject
                        frame_cx = frame.shape[1] / 2
                        frame_cy = frame.shape[0] / 2
                        def _score(d):
                            area = (d.x2 - d.x1) * (d.y2 - d.y1) if hasattr(d, 'x2') else 0
                            dx = ((d.x1 + d.x2) / 2 - frame_cx) if hasattr(d, 'x1') else 999
                            dy = ((d.y1 + d.y2) / 2 - frame_cy) if hasattr(d, 'y1') else 999
                            dist = (dx**2 + dy**2) ** 0.5
                            return area / (1.0 + dist * 0.02)
                        best = max(detections, key=_score)
                        target_obj = getattr(best, 'class_name', 'person')
                    # Also check if the question mentions a specific object
                    for word in q_lower.split():
                        if word not in COLOR_KEYWORDS and len(word) > 3:
                            # crude noun check — if a detected class name appears in question, use it
                            for d in detections:
                                if word in getattr(d, 'class_name', ''):
                                    target_obj = d.class_name
                                    break

                    # Try Gemini first for AI-powered accuracy
                    from backend.gemini_client import verify_color, is_available
                    if is_available():
                        gemini_color = verify_color(frame, target_object=target_obj)
                        if gemini_color:
                            color_clean = gemini_color.strip().rstrip('.').lower()
                            if len(color_clean) > 60:
                                color_clean = color_clean.split('.')[0].split(',')[0].strip()
                            if target_obj == "person":
                                result = f"The person's clothing appears to be {color_clean}."
                            else:
                                result = f"The {target_obj} appears to be {color_clean}."
                            logger.info("[Brain] color (Gemini AI): %r", result)
                            self._history_append(question, result, _gen)
                            return result
                    # Fallback: OpenCV color detection
                    from backend.color_sense import answer_color_question
                    result = answer_color_question(frame, detections)
                    logger.info("[Brain] color result: %r", result)
                except Exception as e:
                    logger.error("[Brain] color detection error: %s", e)
                    result = "I can't determine the color right now. Please try again."
            else:
                result = "I can't determine the color without a camera frame."
            self._history_append(question, result, _gen)
            return result

        # ── 2. Medicine — OCR + phi3:mini (label text → LLM) ─────────────────
        if any(k in q_words for k in MEDICINE_KEYWORDS):
            logger.info("[Brain] route=medicine q=%r", question[:80])
            result = self._answer_medicine_question(question, detections, texts)
            self._history_append(question, result, _gen)
            return result

        # ── 3. Person count — instant YOLO ───────────────────────────────────
        if any(k in q_lower for k in PERSON_COUNT_KEYWORDS):
            logger.info("[Brain] route=person_count q=%r", question[:80])
            people = [d for d in detections
                      if hasattr(d, "class_name") and d.class_name == "person"]
            if not people:
                result = "I don't see any people in the current view."
            elif len(people) == 1:
                p = people[0]
                dist_m = getattr(p, "smoothed_distance_m", 0.0)
                dist   = f"{dist_m:.1f}m away" if dist_m > 0 else "nearby"
                dirn   = getattr(p, "direction", "ahead")
                result = f"I can see one person, {dist}, to your {dirn}."
            else:
                p = people[0]
                dist_m = getattr(p, "smoothed_distance_m", 0.0)
                dist   = f"{dist_m:.1f}m away" if dist_m > 0 else "nearby"
                dirn   = getattr(p, "direction", "ahead")
                result = (f"I can see {len(people)} people. "
                          f"The nearest is {dist}, to your {dirn}.")
            self._history_append(question, result, _gen)
            return result

        # ── 4. Safety check — instant YOLO ───────────────────────────────────
        if any(k in q_words for k in SAFETY_KEYWORDS):
            logger.info("[Brain] route=safety q=%r", question[:80])
            danger = [d for d in detections
                      if hasattr(d, "distance_level") and d.distance_level <= 2]
            if not danger:
                result = "Path looks clear. You can move forward cautiously."
            else:
                names  = ", ".join(set(d.class_name for d in danger[:3]))
                result = (f"Caution — {names} detected nearby. "
                          "Move slowly and check ahead.")
            self._history_append(question, result, _gen)
            return result

        # ── 5. Direction — instant YOLO ───────────────────────────────────────
        if any(k in q_lower for k in DIRECTION_KEYWORDS):
            logger.info("[Brain] route=direction q=%r", question[:80])
            fw = frame.shape[1] if frame is not None else 640

            def _get_clock(d):
                """Get clock direction string from TrackedObject or raw Detection."""
                clk = getattr(d, "direction", None)
                if clk:
                    return clk
                cx = getattr(d, "center_x", None)
                if cx is None:
                    cx = (getattr(d, "x1", 0) + getattr(d, "x2", fw)) // 2
                return _clock_from_center_x(cx, fw)

            def _is_close(d):
                """True if object is within ~2 m (distance_level <= 2 or bbox large)."""
                dl = getattr(d, "distance_level", None)
                if dl is not None:
                    return dl <= 2
                # For raw Detection: use bbox bottom proximity as proxy for closeness
                return getattr(d, "y2", 0) > (frame.shape[0] * 0.6 if frame is not None else 300)

            blocking = [d for d in detections if "12" in _get_clock(d) and _is_close(d)]
            left_blocked = any("9" in _get_clock(d) or "10" in _get_clock(d)
                               for d in detections if _is_close(d))
            right_blocked = any("2" in _get_clock(d) or "3" in _get_clock(d)
                                for d in detections if _is_close(d))
            if not blocking:
                result = "Path ahead looks clear. You can go straight."
            elif not left_blocked and not right_blocked:
                result = (f"There is a {blocking[0].class_name} blocking ahead. "
                          "Both sides appear open — turn left or right.")
            elif not left_blocked:
                result = (f"There is a {blocking[0].class_name} ahead. "
                          "Turn left — that side appears clearer.")
            elif not right_blocked:
                result = (f"There is a {blocking[0].class_name} ahead. "
                          "Turn right — that side appears clearer.")
            else:
                result = ("Objects detected on all sides. "
                          "Stop and move slowly — I'll guide you as you turn.")
            self._history_append(question, result, _gen)
            return result

        # ── 6. Visual appearance → LLaVA with grounded crop ──────────────────
        # Questions about clothing, ID cards, signs, door states etc. require
        # actual image analysis.  phi3:mini has no image input and would hallucinate
        # from YOLO labels alone.  Route to LLaVA with the best object crop.
        if frame is not None and self._needs_visual_model(question):
            logger.info("[Brain] route=llava q=%r", question[:80])
            result = self._run_llava(question, frame, detections, texts)
            self._history_append(question, result, _gen)
            return result

        # ── 7. Everything else → structured JSON context + phi3:mini ─────────
        # Spec: ASK mode must NOT send raw image to LLM.
        # Build structured JSON from perception data; LLM converts to explanation.
        logger.info("[Brain] route=structured_llm q=%r", question[:80])
        fw = frame.shape[1] if frame is not None else 640
        result = self._run_structured_llm(question, detections, texts, frame_w=fw)
        self._history_append(question, result, _gen)
        return result

    # ── FIND object locator (fast, no LLM) ───────────────────────────────────

    # Common synonyms: user word → possible YOLO class substrings
    _FIND_SYNONYMS: dict = {
        "phone":     ["cell phone", "phone", "mobile"],
        "mobile":    ["cell phone", "phone", "mobile"],
        "cell":      ["cell phone"],
        "tv":        ["tv", "television", "monitor", "screen"],
        "television":["tv", "television"],
        "remote":    ["remote", "remote control"],
        "couch":     ["couch", "sofa"],
        "sofa":      ["sofa", "couch"],
        "laptop":    ["laptop", "computer"],
        "computer":  ["laptop", "computer"],
        "glasses":   ["glasses", "sunglasses"],
        "sunglasses":["glasses", "sunglasses"],
        "bag":       ["bag", "backpack", "handbag", "suitcase"],
        "backpack":  ["backpack", "bag"],
        "cup":       ["cup", "mug"],
        "mug":       ["mug", "cup"],
        "bottle":    ["bottle", "water bottle"],
        "keys":      ["keys", "key", "keychain"],
        "key":       ["key", "keys", "keychain"],
        "wallet":    ["wallet", "purse"],
        "purse":     ["purse", "wallet", "handbag"],
        "bike":      ["bicycle", "bike"],
        "bicycle":   ["bicycle", "bike"],
        "motorbike": ["motorbike", "motorcycle"],
        "motorcycle":["motorcycle", "motorbike"],
        "car":       ["car", "vehicle", "automobile"],
        "truck":     ["truck", "lorry"],
        "person":    ["person", "people", "man", "woman", "human"],
        "man":       ["person", "people"],
        "woman":     ["person", "people"],
        "dog":       ["dog", "cat"],  # common pets
        "cat":       ["cat", "dog"],
        "chair":     ["chair", "seat"],
        "table":     ["table", "desk", "dining table"],
        "desk":      ["desk", "table", "laptop"],
        "book":      ["book", "notebook"],
    }

    def _answer_find_object(self, question: str,
                            detections: list, frame_w: int) -> "str | None":
        """
        Locate a named object in detections and return a user-centric directional
        answer without using an LLM.

        Strategy:
          1. Extract candidate object name words from the question (strip stop words).
          2. Find the best-matching Detection by class_name substring match.
          3. Compute clock direction from center_x / frame_w (works for raw
             Detection objects which have no .direction attribute).
          4. Return a plain-English answer: "The fan is to your left — turn left
             to face it.  It is approximately 2.1 m away."

        Returns None if no matching detection found (caller should fall through
        to LLM routes).
        """
        q_lower = question.lower()

        # Strip common stop words and find candidate target words
        stop_words = {
            "where", "is", "the", "a", "an", "can", "you", "see", "find",
            "locate", "look", "for", "show", "me", "there", "do",
            "please", "i", "want", "to", "need", "are",
        }
        words = [w.strip("?.,!") for w in q_lower.split()]
        target_words = [w for w in words if w and w not in stop_words]

        if not target_words:
            return None

        # Build expanded synonym set for each target word
        expanded_targets: set[str] = set()
        for tw in target_words:
            expanded_targets.add(tw)
            for syn in self._FIND_SYNONYMS.get(tw, []):
                expanded_targets.add(syn)

        # Find best detection by class_name match (substring, case-insensitive)
        # Now also checks synonym expansions so "phone" matches "cell phone" etc.
        best_det = None
        best_score = 0
        for d in detections:
            name = getattr(d, "class_name", "").lower().replace("_", " ")
            if not name:
                continue
            # Primary: direct substring match with expanded targets
            score = sum(1 for tw in expanded_targets if tw in name or name in tw)
            # Fallback: individual words of class name appear in question
            if score == 0:
                name_parts = name.split()
                score = sum(1 for np_ in name_parts if np_ in q_lower)
            if score > best_score:
                best_score = score
                best_det = d

        if best_det is None or best_score == 0:
            # No named match — return None so caller can fall through to LLaVA
            return None

        # ── Compute direction ────────────────────────────────────────────────
        # Prefer the .direction attribute (set for TrackedObject), fall back to
        # computing it from center_x for raw Detection objects.
        clock = getattr(best_det, "direction", None)
        if not clock:
            cx = getattr(best_det, "center_x", None)
            if cx is None:
                cx = (getattr(best_det, "x1", 0) + getattr(best_det, "x2", frame_w)) // 2
            clock = _clock_from_center_x(cx, frame_w)

        side, action = _clock_to_user_guidance(clock)

        # ── Distance ─────────────────────────────────────────────────────────
        try:
            dist_m = float(getattr(best_det, "smoothed_distance_m", 0.0) or 0.0)
        except (TypeError, ValueError):
            dist_m = 0.0
        if dist_m > 0.1:
            dist_ft = dist_m * 3.28084
            dist_str = f"  It is approximately {dist_m:.1f} metres ({dist_ft:.0f} feet) away."
        else:
            dist_str = ""

        target_name = best_det.class_name.replace("_", " ")
        logger.info(
            "[Brain] find_object: target=%r clock=%r side=%r dist_m=%.1f",
            target_name, clock, side, dist_m,
        )
        return (
            f"I found the {target_name} — it is {side} of you ({clock}).  "
            f"{action}{dist_str}"
        )

    # ── Structured context builder (ASK mode — spec: no raw image to LLM) ─────

    def _build_structured_context(self, detections: list,
                                   frame_w: int = 640) -> dict:
        """
        Convert YOLO detections + depth into a structured JSON dict.
        This is the ONLY input the LLM receives in ASK mode.
        Spec: LLM must not receive raw images or infer beyond structured data.

        Works with both TrackedObject (has .direction) and raw Detection
        objects (direction is computed from center_x / frame_w).
        """
        objects = []
        for d in detections[:8]:
            name  = getattr(d, "class_name", "object")
            # Direction: prefer .direction (TrackedObject), fall back to center_x calc
            direction = getattr(d, "direction", None)
            if not direction:
                cx = getattr(d, "center_x", None)
                if cx is None:
                    cx = (getattr(d, "x1", 0) + getattr(d, "x2", frame_w)) // 2
                direction = _clock_from_center_x(cx, frame_w)
            dist_m    = getattr(d, "smoothed_distance_m", None)
            distance  = getattr(d, "distance", "unknown")
            entry = {"name": name, "direction": direction}
            if dist_m is not None:
                entry["distance_m"] = round(float(dist_m), 2)
            else:
                entry["distance"] = distance
            objects.append(entry)

        # Determine path status: anything at 12 o'clock within ~2 m = blocked
        def _clk(d):
            clk = getattr(d, "direction", None)
            if clk:
                return clk
            cx = getattr(d, "center_x", None)
            if cx is None:
                cx = (getattr(d, "x1", 0) + getattr(d, "x2", frame_w)) // 2
            return _clock_from_center_x(cx, frame_w)

        blocking = [
            d for d in detections
            if "12" in _clk(d)
            and getattr(d, "distance_level", 99) <= 2
        ]
        path_status = "blocked" if blocking else "clear"

        approaching = [
            getattr(d, "class_name", "object")
            for d in detections
            if getattr(d, "motion_state", "") == "approaching"
        ]

        return {
            "objects":            objects,
            "path_status":        path_status,
            "approaching_objects": approaching,
        }

    def _run_structured_llm(self, question: str,
                             detections: list, texts: List[str],
                             frame_w: int = 640) -> str:
        """
        ASK mode: sends only structured JSON (no raw image) to phi3:mini.
        Temperature=0.0 — deterministic, no hallucination.
        LLM role: convert structured scene truth → formatted explanation only.
        """
        context = self._build_structured_context(detections, frame_w=frame_w)
        text_desc = ", ".join(texts[:5]) or "none"
        history_str = _build_history_str(self._history)

        # Try Gemini for structured reasoning (when keys configured)
        try:
            from backend.gemini_client import ask_text_only, is_available
            if is_available():
                gemini_prompt = (
                    f"Scene data (from sensors): {json.dumps(context)}\n"
                    f"Visible text: {text_desc}\n"
                    f"{history_str}"
                    f"Question: {question}\n\n"
                    "Answer clearly and accurately using ONLY the scene data above. "
                    "Keep your answer to 1-2 sentences. "
                    "Do not add objects, distances, or context not listed in the scene data. "
                    "If the data is insufficient, say so honestly."
                )
                gemini_answer = ask_text_only(
                    f"You are helping a visually impaired person. Answer clearly in 1-2 sentences.\n\nQuestion: {question}",
                    context=f"Scene data: {json.dumps(context)}\nVisible text: {text_desc}"
                )
                if gemini_answer:
                    logger.info("[Brain] Using Gemini AI for structured answer")
                    return gemini_answer
        except Exception as e:
            logger.debug("[Brain] Gemini text fallback: %s", e)

        prompt = (
            "You are an AI assistant helping a visually impaired person navigate safely.\n"
            f"Scene data (from sensors): {json.dumps(context)}\n"
            f"Visible text: {text_desc}\n"
            f"{history_str}"
            f"Question: {question}\n\n"
            "Answer clearly and accurately using ONLY the scene data above. "
            "Keep your answer to 1-2 sentences. "
            "Do not add objects, distances, or context not listed in the scene data. "
            "If the data is insufficient, say so honestly."
        )
        try:
            payload = json.dumps({
                "model":  self._text_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 150,
                    "temperature": 0.0,
                    "top_p":       1.0,
                    "stop":        ["\n\n", "###"],
                },
            }).encode("utf-8")

            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                data   = json.loads(resp.read().decode("utf-8"))
                answer = data.get("response", "").strip()
                if answer:
                    logger.debug("[Brain] Structured LLM answered: %s", answer[:80])
                    return answer
                logger.warning("[Brain] Structured LLM empty response — using fallback.")
        except urllib.error.URLError:
            logger.warning("[Brain] Ollama not running at localhost:11434.")
        except TimeoutError:
            logger.warning("[Brain] Structured LLM timed out.")
        except Exception as exc:
            logger.error("[Brain] Structured LLM error: %s", exc)

        return self._fallback(self._describe_objects(detections), texts)

    # ── LLaVA visual answering (visual appearance questions + FIND mode) ─────

    def _run_llava(self, question: str, frame,
                   detections: list, texts: List[str],
                   find_target: str = "") -> str:
        """
        Send the camera frame (or a grounded object crop) to LLaVA for a
        genuine visual answer.

        Grounding strategy:
          - FIND mode (find_target set): use the full frame so LLaVA can scan
            the entire scene for the requested object.  Cropping to a detected
            object would be wrong here — the target may be anywhere in the frame.
          - Visual appearance questions (no find_target): find the most salient
            detected object and send an expanded crop so LLaVA focuses on it.
          - Falls back to full frame if no detections or crop fails.
        """
        if frame is None:
            return self._run_phi3_text(question, detections, texts)

        # ── Select image region for LLaVA ─────────────────────────────────────
        grounded_class = ""
        crop_frame = frame   # default: full frame

        # Only crop to a specific object when NOT in FIND mode.
        # In FIND mode we want LLaVA to see the whole scene.
        if not find_target and detections:
            frame_h, frame_w = frame.shape[:2]
            frame_cx = frame_w / 2
            frame_cy = frame_h / 2

            def _score(d):
                if not hasattr(d, "x1"):
                    return 0
                area = (d.x2 - d.x1) * (d.y2 - d.y1)
                dx   = ((d.x1 + d.x2) / 2 - frame_cx)
                dy   = ((d.y1 + d.y2) / 2 - frame_cy)
                dist = (dx**2 + dy**2) ** 0.5
                return area / (1 + dist * 0.01)

            best_det = max(detections, key=_score)
            if hasattr(best_det, "x1"):
                from backend.color_sense import expand_bbox
                frame_h2, frame_w2 = frame.shape[:2]
                class_name = getattr(best_det, "class_name", "")
                ex1, ey1, ex2, ey2 = expand_bbox(
                    best_det.x1, best_det.y1, best_det.x2, best_det.y2,
                    frame_w2, frame_h2,
                    scale=0.30,
                    class_name=class_name,
                )
                cropped = frame[ey1:ey2, ex1:ex2]
                if cropped.size > 0:
                    crop_frame = cropped
                    grounded_class = class_name

        # ── Try Gemini first (AI-powered accuracy when keys configured) ────────
        try:
            from backend.gemini_client import ask_vision, is_available
            if is_available():
                context_parts = []
                if detections:
                    obj_desc = self._describe_objects(detections)
                    context_parts.append(f"Detected objects: {obj_desc}")
                if texts:
                    context_parts.append(f"Visible text: {', '.join(texts[:3])}")
                context = " ".join(context_parts) if context_parts else ""
                gemini_answer = ask_vision(question, crop_frame, context)
                if gemini_answer:
                    logger.info("[Brain] Using Gemini AI for visual answer")
                    return gemini_answer
        except Exception as e:
            logger.debug("[Brain] Gemini vision fallback: %s", e)

        # ── Encode crop (or full frame) to JPEG base64 ────────────────────────
        img_b64 = _encode_frame_b64(crop_frame, max_width=512)
        if img_b64 is None:
            return self._run_phi3_text(question, detections, texts)

        # ── Build grounded prompt (LLaVA fallback) ─────────────────────────────
        history_str = _build_history_str(self._history)

        if find_target:
            grounding_prefix = (
                f"You are helping a visually impaired person locate a '{find_target}' "
                "in their environment. Scan the entire image carefully.\n"
            )
        elif grounded_class:
            grounding_prefix = (
                f"You are analysing a detected {grounded_class.upper()} in the scene. "
                f"The image shown is a crop focused on that {grounded_class}. "
                "Focus your answer only on what you can see in this cropped region.\n"
            )
        else:
            grounding_prefix = (
                "You are analysing the current scene visible through the camera.\n"
            )

        prompt = (
            "You are an AI assistant helping a visually impaired person.\n"
            f"{grounding_prefix}"
            f"{history_str}"
            f"Question: {question}\n\n"
            "Look carefully at the image and answer accurately. "
            "Be specific about what you actually see — include locations (left, right, center), "
            "distances if apparent, and relevant details. "
            "Keep your answer to 2-3 sentences maximum. "
            "Do not guess — if you cannot see something clearly, say so."
        )

        try:
            payload = json.dumps({
                "model":  self._vision_model,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
                "options": {
                    "num_predict": 200,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "stop": ["\n\n", "###"],
                },
            }).encode("utf-8")

            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data   = json.loads(resp.read().decode("utf-8"))
                answer = data.get("response", "").strip()
                if answer:
                    logger.debug("[Brain] LLaVA answered: %s", answer[:80])
                    return answer
                logger.warning("[Brain] LLaVA returned empty response — falling back.")
        except urllib.error.URLError:
            logger.warning("[Brain] LLaVA not reachable — falling back to structured LLM.")
        except TimeoutError:
            logger.warning("[Brain] LLaVA timed out — falling back to structured LLM.")
        except Exception as exc:
            logger.error("[Brain] LLaVA error: %s — falling back to structured LLM.", exc)

        # Fallback: structured LLM (no image)
        return self._run_structured_llm(question, detections, texts)

    # ── phi3:mini text-only answering (medicine + LLaVA fallback) ────────────

    def _run_phi3_text(self, question: str,
                       detections: list, texts: List[str]) -> str:
        """phi3:mini with YOLO object list + OCR text as context (no image)."""
        obj_desc  = self._describe_objects(detections)
        text_desc = ", ".join(texts[:5]) or "no visible text"
        history_str = _build_history_str(self._history)

        prompt = (
            "You are an AI assistant helping a visually impaired person.\n"
            f"Scene (from camera): {obj_desc}\n"
            f"Visible text: {text_desc}\n"
            f"{history_str}"
            f"Question: {question}\n\n"
            "Answer accurately in 1-2 sentences using ONLY what is described above. "
            "Do not guess. If unsure, say what you can see."
        )
        return self._run_ollama(prompt, obj_desc, texts)

    def _answer_medicine_question(self, question: str,
                                  detections: list, texts: List[str]) -> str:
        """
        Medicine/safety: prioritise OCR text for label reading, then pass to
        phi3:mini with a safety-focused prompt.
        """
        if not texts:
            return (
                "I can't read any label text right now. "
                "Please point the camera directly at the label and hold it steady. "
                "Make sure there is good lighting."
            )
        label_text = ", ".join(texts[:8])
        obj_desc   = self._describe_objects(detections)

        prompt = (
            "You are an AI assistant helping a visually impaired person identify medicine safely.\n"
            f"Objects in scene: {obj_desc}\n"
            f"Text visible on label or packaging: {label_text}\n"
            f"Question: {question}\n\n"
            "Instructions:\n"
            "1. Read out the medicine name and dosage EXACTLY as it appears in the label text.\n"
            "2. If you cannot confirm the medicine name from the label, say so clearly.\n"
            "3. Do NOT invent dosage information. Use ONLY what is in the label text above.\n"
            "4. Keep your answer to TWO sentences maximum.\n"
            "Example: 'The label reads Paracetamol 500mg. The dosage shown is two tablets every four hours.'"
        )
        return self._run_ollama(prompt, obj_desc, texts)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _describe_objects(self, detections: list) -> str:
        """Google-level: Enhanced object description with more details."""
        if not detections:
            return "nothing specific detected"
        parts = []
        for d in detections[:10]:  # Increased from 8 to 10 for better context
            name      = d.class_name if hasattr(d, "class_name") else str(d)
            direction = getattr(d, "direction", "")
            distance  = getattr(d, "distance",  "")
            confidence = getattr(d, "confidence", 0.0)
            # Include confidence for better context
            if direction and distance:
                parts.append(f"{name} ({direction}, {distance}, {int(confidence*100)}% confidence)")
            elif direction:
                parts.append(f"{name} ({direction})")
            else:
                parts.append(name)
        return "; ".join(parts)

    def _run_ollama(self, prompt: str, obj_desc: str, texts: List[str]) -> str:
        """phi3:mini text-only via Ollama REST API."""
        try:
            payload = json.dumps({
                "model":  self._text_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 150,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "stop": ["\n\n", "###"],
                },
            }).encode("utf-8")

            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                data   = json.loads(resp.read().decode("utf-8"))
                answer = data.get("response", "").strip()
                if answer:
                    return answer
                logger.warning("[Brain] phi3 returned empty response — using fallback.")
        except urllib.error.URLError:
            logger.warning("[Brain] Ollama not running at localhost:11434.")
        except TimeoutError:
            logger.warning("[Brain] phi3 timed out.")
        except Exception as exc:
            logger.error("[Brain] phi3 error: %s", exc)

        return self._fallback(obj_desc, texts)

    def _fallback(self, obj_desc: str, texts: List[str]) -> str:
        parts = []
        if obj_desc and obj_desc != "nothing specific detected":
            parts.append(f"I can see: {obj_desc}")
        if texts:
            parts.append(f"Text in view: {', '.join(texts[:3])}")
        return ". ".join(parts) if parts else \
               "I can see the scene. Try asking about a specific object."


# ── Module-level helpers ──────────────────────────────────────────────────────

def _encode_frame_b64(frame: np.ndarray, max_width: int = 512) -> str | None:
    """Resize frame to max_width and return base64-encoded JPEG string."""
    try:
        h, w = frame.shape[:2]
        if w > max_width:
            scale  = max_width / w
            frame  = cv2.resize(frame, (max_width, int(h * scale)),
                                 interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("utf-8")
    except Exception as exc:
        logger.error("[Brain] Frame encode error: %s", exc)
        return None


def _build_history_str(history: deque) -> str:
    if not history:
        return ""
    lines = []
    for q, a in history:
        lines.append(f"User: {q}")
        lines.append(f"Assistant: {a}")
    return "\nPrevious conversation:\n" + "\n".join(lines) + "\n"


brain = Brain()
