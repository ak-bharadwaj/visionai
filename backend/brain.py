import urllib.request, urllib.error, json, logging, os
from typing import List
from collections import deque

logger = logging.getLogger(__name__)

COLOR_KEYWORDS    = {"color", "colour", "shade", "hue", "what color"}
DOOR_KEYWORDS     = {"door", "open", "closed", "shut", "entrance", "exit"}
MEDICINE_KEYWORDS = {"medicine", "medication", "drug", "pill", "tablet", "capsule",
                     "dose", "dosage", "safe", "take", "prescription", "label",
                     "bottle", "packet", "paracetamol", "ibuprofen", "aspirin"}
SCENE_KEYWORDS        = {"front", "ahead", "see", "there", "around",
                          "scene", "room", "nearby", "surroundings"}
PERSON_COUNT_KEYWORDS = {"people", "person", "anyone", "anybody",
                          "someone", "somebody", "how many", "crowd"}
SAFETY_KEYWORDS       = {"safe", "walk", "move", "proceed", "go", "step"}

# Max conversation turns kept in memory (each turn = question + answer)
MAX_HISTORY = 4


class Brain:
    """
    On-device AI reasoning for ASK mode.
    Uses Ollama (run `ollama serve` + `ollama pull phi3:mini` before demo).
    100% offline — no internet required.
    Supports multi-turn conversation memory (up to MAX_HISTORY turns).
    """

    def __init__(self):
        self._model = os.getenv("OLLAMA_MODEL", "phi3:mini")
        # Conversation history: deque of (question, answer) tuples
        self._history: deque = deque(maxlen=MAX_HISTORY)

    def clear_history(self):
        """Reset conversation memory. Call when leaving ASK mode or on demand."""
        self._history.clear()

    def get_history(self) -> list:
        """Returns conversation history as list of dicts for frontend."""
        return [{"question": q, "answer": a} for q, a in self._history]

    def needs_llm(self, question: str) -> bool:
        """
        Returns True if this question will fall through to the Ollama LLM.
        Returns False if it will be answered instantly by a rule-based route.
        Used by pipeline.py to decide whether to play the 'Let me check that'
        holding message — we don't want it on instant answers.
        """
        q_lower = question.lower()
        q_words = set(q_lower.split())
        if any(k in q_lower for k in COLOR_KEYWORDS):    return False
        if any(k in q_lower for k in DOOR_KEYWORDS):     return False
        # NOTE: medicine questions are NOT listed here — _answer_medicine_question()
        # calls _run_ollama(), so needs_llm() must return True for them so the
        # "Let me check that" holding message plays during the LLM call.
        if any(k in q_words for k in SCENE_KEYWORDS) and len(q_words) < 8:
            return False
        if any(k in q_lower for k in PERSON_COUNT_KEYWORDS): return False
        if any(k in q_words for k in SAFETY_KEYWORDS):       return False
        LOCATION_KEYWORDS = {"where", "is there", "can you see", "find", "locate", "spot"}
        if any(k in q_lower for k in LOCATION_KEYWORDS):   return False
        return True  # nothing matched — will hit the LLM

    def answer(self, question: str, frame,
               detections: list, texts: List[str]) -> str:
        """
        Main entry point for answering a user question.
        Routes special question types before falling through to LLM.
        Falls back gracefully if Ollama fails.
        Maintains conversation memory across multiple turns.
        """
        q_lower = question.lower()
        q_words = set(q_lower.split())

        # 1. Color questions → instant OpenCV answer (no LLM)
        if any(k in q_lower for k in COLOR_KEYWORDS):
            from backend.color_sense import answer_color_question
            result = answer_color_question(frame, detections)
            if not detections and "no object" in result.lower():
                result = (
                    "I can't detect any object to check the color. "
                    "Please point the camera at a specific object and try again."
                )
            self._history.append((question, result))
            return result

        # 2. Door open/closed → spatial + OCR heuristic
        if any(k in q_lower for k in DOOR_KEYWORDS):
            result = self._answer_door_question(q_lower, detections, texts)
            self._history.append((question, result))
            return result

        # 3. Medicine / safety → OCR label + LLM with safety-focused prompt
        if any(k in q_words for k in MEDICINE_KEYWORDS):
            result = self._answer_medicine_question(question, detections, texts)
            self._history.append((question, result))
            return result

        # 4a. "What is in front of me?" / "What do you see?" — instant from YOLO
        if any(k in q_words for k in SCENE_KEYWORDS) and len(q_words) < 8:
            if detections:
                from collections import Counter
                counts = Counter(
                    d.class_name for d in detections
                    if hasattr(d, "class_name")
                )
                parts = [f"{v} {k}{'s' if v > 1 else ''}"
                         for k, v in counts.most_common(5)]
                result = "I can see: " + ", ".join(parts) + "."
            else:
                result = "I don't see any specific objects right now."
            self._history.append((question, result))
            return result

        # 4b. "How many people?" / "Anyone here?" — instant person count
        if any(k in q_lower for k in PERSON_COUNT_KEYWORDS):
            people = [d for d in detections
                      if hasattr(d, "class_name") and d.class_name == "person"]
            if not people:
                result = "I don't see any people in the current view."
            elif len(people) == 1:
                p = people[0]
                result = (f"I can see one person, {p.distance}, "
                          f"to your {p.direction}.")
            else:
                result = (f"I can see {len(people)} people. "
                          f"The nearest is {people[0].distance}, "
                          f"to your {people[0].direction}.")
            self._history.append((question, result))
            return result

        # 4c. "Is it safe?" / "Can I walk?" — instant obstacle check
        if any(k in q_words for k in SAFETY_KEYWORDS):
            danger = [d for d in detections
                      if hasattr(d, "distance_level") and d.distance_level <= 2]
            if not danger:
                result = "Path looks clear. You can move forward cautiously."
            else:
                names  = ", ".join(set(d.class_name for d in danger[:3]))
                result = (f"Caution — {names} detected nearby. "
                          "Move slowly and check ahead.")
            self._history.append((question, result))
            return result

        # 4d. Object-specific not-found guidance
        LOCATION_KEYWORDS = {"where", "is there", "can you see", "find", "locate", "spot"}
        if any(k in q_lower for k in LOCATION_KEYWORDS):
            q_words_list = q_lower.split()
            STOPWORDS = {"where", "is", "there", "a", "the", "can", "you", "see",
                         "find", "locate", "spot", "any", "my"}
            target = " ".join(w for w in q_words_list if w not in STOPWORDS).strip()
            found = any(
                target in d.class_name.lower() or d.class_name.lower() in target
                for d in detections if hasattr(d, "class_name")
            ) if target else True
            if not found and target:
                import random
                DIRECTION_HINTS = [
                    "Try slowly turning to your left.",
                    "Try looking to your right.",
                    "Try pointing the camera a bit higher.",
                    "Move forward slowly and scan around.",
                ]
                hint = random.choice(DIRECTION_HINTS)
                result = (
                    f"I don't see a {target} in the current view. {hint} "
                    "I'll let you know as soon as I spot it."
                )
                self._history.append((question, result))
                return result

        # 5. General LLM path — includes conversation history context
        obj_desc   = self._describe_objects(detections)
        text_desc  = ", ".join(texts[:5]) or "no visible text"

        # Build conversation history string for context
        history_str = ""
        if self._history:
            history_lines = []
            for q, a in self._history:
                history_lines.append(f"User: {q}")
                history_lines.append(f"Assistant: {a}")
            history_str = "\nPrevious conversation:\n" + "\n".join(history_lines) + "\n"

        prompt = (
            "You are an AI assistant helping a visually impaired person.\n"
            f"Scene (from camera): {obj_desc}\n"
            f"Visible text: {text_desc}\n"
            f"{history_str}"
            f"Question: {question}\n\n"
            "Instructions: Answer in ONE short sentence. Use ONLY what is described "
            "above and the conversation history. Do not guess. If unsure, say what "
            "you can see clearly. If this is a follow-up question, refer to the "
            "previous answers appropriately."
        )

        result = self._run_ollama(prompt, obj_desc, texts)
        self._history.append((question, result))
        return result

    # ── Specialised answer handlers ────────────────────────────────────

    def _answer_door_question(self, q_lower: str,
                              detections: list, texts: List[str]) -> str:
        """
        Heuristic door open/closed detection.
        YOLOv8n on COCO-80 has NO 'door' class — searching detections for it
        always returns None and we'd always fall through. Instead, go directly
        to the two reliable signals we actually have:
          1. OCR text on or near the door (push/pull/open/closed/exit/etc.)
          2. Positional depth fallback with any visible text context.
        """
        DOOR_TEXT_CLOSED = {"closed", "close", "shut", "no entry", "exit only",
                            "do not enter", "staff only", "private"}
        DOOR_TEXT_OPEN   = {"push", "pull", "open", "welcome", "enter", "entrance"}
        texts_lower = [t.lower() for t in texts]
        combined    = " ".join(texts_lower)

        for word in DOOR_TEXT_CLOSED:
            if word in combined:
                return (
                    f"I see text that says '{word.upper()}' — "
                    "the door appears to be closed or restricted."
                )
        for word in DOOR_TEXT_OPEN:
            if word in combined:
                return (
                    f"I see text that says '{word.upper()}' — "
                    "there may be an open door or entrance here."
                )

        # No OCR clues — use depth of largest detection as a proxy for open/closed
        # (large nearby object blocking the centre = likely closed door)
        centre_blockers = [
            d for d in detections
            if hasattr(d, "direction") and d.direction in ("ahead", "left", "right")
            and hasattr(d, "distance_level") and d.distance_level <= 2
        ]
        if centre_blockers:
            blocker = centre_blockers[0]
            return (
                f"I can see a {blocker.class_name} {blocker.distance} ahead — "
                "something may be blocking or the door could be closed."
            )

        nearby_text = ", ".join(texts[:3]) if texts else "no text visible"
        return (
            "I don't see a door directly, but there may be one nearby. "
            f"Visible text: {nearby_text}."
        )

    def _answer_medicine_question(self, question: str,
                                  detections: list, texts: List[str]) -> str:
        """
        Medicine/safety: prioritise OCR text for label reading, then pass to LLM
        with a safety-focused prompt that instructs it to repeat name + dosage.
        """
        if not texts:
            return (
                "I can't read any label text right now. "
                "Please point the camera directly at the label and hold it steady. "
                "Make sure there is good lighting."
            )
        label_text = ", ".join(texts[:8]) if texts else "no text visible on label"
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

    # ── Internal helpers ───────────────────────────────────────────────

    def _describe_objects(self, detections: list) -> str:
        if not detections:
            return "nothing specific detected"
        parts = []
        for d in detections[:8]:  # cap at 8 to keep prompt short
            name      = d.class_name if hasattr(d, "class_name") else str(d)
            direction = d.direction  if hasattr(d, "direction")  else ""
            distance  = d.distance   if hasattr(d, "distance")   else ""
            parts.append(f"{name} ({direction}, {distance})")
        return "; ".join(parts)

    def _run_ollama(self, prompt: str, obj_desc: str, texts: List[str]) -> str:
        """
        Use Ollama REST API instead of subprocess — 2-3x faster.
        POST to http://localhost:11434/api/generate
        No process-spawn overhead; reuses Ollama's already-running server.
        """
        try:
            payload = json.dumps({
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 80,     # max 80 tokens = 1-2 sentences
                    "temperature": 0.3,    # low temp = less hallucination
                    "top_p": 0.9,
                    "stop": ["\n\n", "###"]
                }
            }).encode("utf-8")

            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=4) as resp:
                data   = json.loads(resp.read().decode("utf-8"))
                answer = data.get("response", "").strip()
                if answer:
                    return answer
                logger.warning("👁 [Brain] Ollama API returned empty response. Using fallback.")
        except urllib.error.URLError:
            logger.warning("👁 [Brain] Ollama not running at localhost:11434. Using fallback.")
        except TimeoutError:
            logger.warning("👁 [Brain] Ollama API timed out. Using fallback.")
        except Exception as e:
            logger.error(f"👁 [Brain] Ollama API error: {e}. Using fallback.")

        return self._fallback(obj_desc, texts)

    def _fallback(self, obj_desc: str, texts: List[str]) -> str:
        parts = []
        if obj_desc and obj_desc != "nothing specific detected":
            parts.append(f"I can see: {obj_desc}")
        if texts:
            parts.append(f"Text in view: {', '.join(texts[:3])}")
        return ". ".join(parts) if parts else \
               "I can see the scene. Try asking about a specific object."


brain = Brain()
