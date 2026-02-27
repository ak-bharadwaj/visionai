"""
stt.py — Continuous Whisper STT listener for VisionTalk.

Architecture:
  - faster-whisper (tiny, int8, CPU) transcribes 3-second mic chunks.
  - Deterministic command classifier maps transcript → action dict.
  - Results are put on a thread-safe queue consumed by main.py.
  - Graceful fallback if mic or faster-whisper are unavailable.

Supported voice commands (case/punctuation-insensitive):
  Natural language navigation aliases:
    "Navigate / guide me / walk mode / start navigation / help me walk"
                          → action: set_mode,  mode: NAVIGATE
  Natural language find aliases:
    "Find X / locate X / where is X / look for X"
                          → action: find_object, target: X
  Natural language read aliases:
    "Read / read this / scan text / read the sign"
                          → action: set_mode,  mode: READ
  Natural language ask aliases:
    "What's ahead / what do you see / describe this / what is this"
                          → action: ask, question: ...
  "Is the path clear"   → action: ask,       question: "Is the path clear?"
  "Stop narration"      → action: set_mode,  mode: NAVIGATE  (re-enter clears queue)
  "Repeat"              → action: speak,     text: <last spoken>
  "What changed"        → action: scene_diff
  "Remember this"       → action: snapshot
  "Scan"                → action: set_mode,  mode: SCAN

Repetition suppression:
  Identical commands within DEDUP_COMMAND_WINDOW seconds are silently dropped.
  This prevents double-fires from mic echo or natural repetition.

Design rules:
  - Never calls an LLM.  All routing is string matching.
  - Never raises outside its own thread — errors are logged and skipped.
  - Does not import main.py (avoids circular import).
"""

import queue
import threading
import logging
import os
import re
import time
import numpy as np

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 16000          # Whisper expects 16 kHz
CHUNK_SECONDS  = 3              # record N seconds then transcribe
SILENCE_THRESH = 0.005          # RMS below this → skip transcription (silence)
MODEL_SIZE     = "tiny"         # tiny is fast enough on CPU (<150 ms per chunk)
LANGUAGE       = "en"           # lock to English — faster, less ambiguity

# Repetition suppression: identical command text within this window is dropped.
# Prevents double-firing from mic echo or users naturally repeating themselves
# before the system has had time to act.
DEDUP_COMMAND_WINDOW = float(os.getenv("STT_DEDUP_COMMAND_WINDOW", "2.0"))  # seconds

# Minimum number of characters a Whisper transcript must contain to be
# considered real speech.  Single-character outputs (e.g. ".", "I", "A") are
# almost always hallucinations from background noise that slipped past VAD and
# the lang_prob gate.  Two characters is the practical minimum for any valid
# English command word.
MIN_TRANSCRIPT_CHARS = int(os.getenv("STT_MIN_TRANSCRIPT_CHARS", "2"))

# ASR confidence gates.
# lang_prob  : faster-whisper's estimate that audio is the target language.
#              Below STT_MIN_LANG_PROB the chunk is likely silence/noise/non-English.
# no_speech  : per-segment probability that the segment contains no real speech.
#              Above STT_MAX_NO_SPEECH_PROB the segment is dropped before joining.
# avg_logprob: average log-probability of all tokens in the segment.
#              Below STT_MIN_AVG_LOGPROB the segment is very uncertain — dropped.
STT_MIN_LANG_PROB    = float(os.getenv("STT_MIN_LANG_PROB",   "0.70"))
STT_MAX_NO_SPEECH    = float(os.getenv("STT_MAX_NO_SPEECH",   "0.60"))
STT_MIN_AVG_LOGPROB  = float(os.getenv("STT_MIN_AVG_LOGPROB", "-1.0"))

# ── Command patterns — order matters (more specific first) ─────────────────────
# Each entry: (regex_pattern, handler_fn)
# handler receives the regex match object and returns an action dict or None.
_PATTERNS: list[tuple[str, callable]] = []


def _register(pattern: str):
    """Decorator to register a command handler."""
    def decorator(fn):
        _PATTERNS.append((re.compile(pattern, re.IGNORECASE), fn))
        return fn
    return decorator


@_register(r"\bfind\b\s+(.+)")
def _cmd_find(m):
    target = m.group(1).strip().rstrip("?.!")
    if target:
        return {"action": "find_object", "target": target}


@_register(r"\b(locate|where is|look for)\b\s+(.+)")
def _cmd_locate(m):
    target = m.group(2).strip().rstrip("?.!")
    if target:
        return {"action": "find_object", "target": target}


@_register(r"\b(what(?:'?s| is) ahead|what(?:'?s| is) in front)\b")
def _cmd_ahead(m):
    return {"action": "ask", "question": "What is ahead of me?", "input_source": "voice"}


@_register(r"\bwhat (can you|do you) see\b")
def _cmd_see(m):
    return {"action": "ask", "question": "What do you see?", "input_source": "voice"}


@_register(r"\b(describe this|what is this|what('?s| is) (around|here|nearby))\b")
def _cmd_describe(m):
    return {"action": "ask", "question": "What do you see?", "input_source": "voice"}


@_register(r"\bis (the )?path clear\b")
def _cmd_path(m):
    return {"action": "ask", "question": "Is the path clear?", "input_source": "voice"}


@_register(r"\bread( text| label| sign| this)?\b")
def _cmd_read(m):
    return {"action": "set_mode", "mode": "READ"}


@_register(r"\bscan (text|label|sign)\b")
def _cmd_scan_text(m):
    return {"action": "set_mode", "mode": "READ"}


@_register(r"\b(?:navigate|go|head|walk|take me)\s+to\s+(.+)")
def _cmd_nav_destination(m):
    dest = m.group(1).strip().rstrip("?.!")
    if dest:
        return {"action": "nav_destination", "destination": dest}


@_register(r"\bdestination\s+(.+)")
def _cmd_nav_dest_keyword(m):
    dest = m.group(1).strip().rstrip("?.!")
    if dest:
        return {"action": "nav_destination", "destination": dest}


@_register(r"\bi\s+want\s+to\s+go\s+to\s+(.+)")
def _cmd_nav_want(m):
    dest = m.group(1).strip().rstrip("?.!")
    if dest:
        return {"action": "nav_destination", "destination": dest}


@_register(r"\b(navigate|navigation|go to navigate|guide me|walk mode|start navigation|help me walk)\b")
def _cmd_navigate(m):
    return {"action": "set_mode", "mode": "NAVIGATE"}


@_register(r"\bscan\b")
def _cmd_scan(m):
    return {"action": "set_mode", "mode": "SCAN"}


@_register(r"\b(stop narration|stop talking|be quiet|silence)\b")
def _cmd_stop(m):
    return {"action": "set_mode", "mode": "NAVIGATE"}


@_register(r"\brepeat( that)?\b")
def _cmd_repeat(m):
    return {"action": "repeat"}


@_register(r"\bremember this\b")
def _cmd_remember(m):
    return {"action": "snapshot"}


@_register(r"\bwhat( has)? changed\b")
def _cmd_diff(m):
    return {"action": "scene_diff", "input_source": "voice"}


@_register(r"\bask\b\s+(.+)")
def _cmd_ask(m):
    question = m.group(1).strip().rstrip(".")
    if question:
        return {"action": "ask", "question": question, "input_source": "voice"}


def _action_key(action: dict) -> str:
    """
    Produce a dedup key for a classified action dict.
    Two actions are considered identical if they have the same action type
    and the same primary payload (mode / target / question).
    """
    a = action.get("action", "")
    if a == "set_mode":
        return f"set_mode:{action.get('mode', '')}"
    if a == "find_object":
        return f"find_object:{action.get('target', '').lower()}"
    if a == "nav_destination":
        return f"nav_destination:{action.get('destination', '').lower()}"
    if a == "ask":
        return f"ask:{action.get('question', '').lower()}"
    return a  # repeat, snapshot, scene_diff, etc. — key is action type only


# Module-level repetition suppression state (thread-safe via GIL for simple reads/writes)
_last_cmd_key:  str   = ""
_last_cmd_time: float = 0.0


def classify(transcript: str) -> dict | None:
    """
    Map a Whisper transcript to an action dict, or None if not recognised.

    Deterministic — no LLM.
    Includes repetition suppression: identical commands within
    DEDUP_COMMAND_WINDOW seconds return None to prevent double-firing.
    """
    global _last_cmd_key, _last_cmd_time

    t = transcript.strip()
    if not t:
        return None
    for pattern, handler in _PATTERNS:
        m = pattern.search(t)
        if m:
            action = handler(m)
            if action is None:
                continue
            # Repetition suppression gate
            key = _action_key(action)
            now = time.monotonic()
            if key == _last_cmd_key and (now - _last_cmd_time) < DEDUP_COMMAND_WINDOW:
                logger.debug(
                    "[STT] Duplicate command suppressed within %.1fs window: %r",
                    DEDUP_COMMAND_WINDOW, key,
                )
                return None
            _last_cmd_key  = key
            _last_cmd_time = now
            return action
    return None


# ── STT engine ────────────────────────────────────────────────────────────────

class STTEngine:
    """
    Continuous microphone listener.  Runs two daemon threads:
      - _record_thread: fills audio chunks into _audio_q
      - _transcribe_thread: pops chunks, runs Whisper, classifies, puts in cmd_q

    cmd_q is a public queue.Queue that main.py drains via asyncio.
    """

    def __init__(self):
        self._model         = None
        self._loaded        = False
        self._running       = False
        self._audio_q: queue.Queue  = queue.Queue(maxsize=4)
        self.cmd_q:    queue.Queue  = queue.Queue(maxsize=20)  # bounded: drop if dispatcher falls behind
        # Unrecognised transcriptions — speech heard but no command matched.
        # main.py drains this to give audible "didn't catch that" feedback.
        self.unrecognised_q: queue.Queue = queue.Queue(maxsize=8)
        # Push-to-talk / software mute.  When True the record thread drops all
        # audio chunks without queuing them.  main.py toggles this via
        # action="stt_mute" / action="stt_unmute" WebSocket messages, allowing
        # the frontend mic button to gate recording without touching frontend JS.
        self.muted: bool = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        """Load model and start listener threads. Non-blocking."""
        threading.Thread(target=self._load_and_run, daemon=True, name="STTLoader").start()

    def stop(self):
        self._running = False

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_and_run(self):
        """Load Whisper model, then start record + transcribe threads."""
        try:
            from faster_whisper import WhisperModel
            logger.info("[STT] Loading faster-whisper/%s (int8, CPU)…", MODEL_SIZE)
            self._model  = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
            self._loaded = True
            logger.info("[STT] Whisper ready.")
        except Exception as exc:
            logger.warning("[STT] Could not load Whisper: %s — voice commands disabled.", exc)
            return

        try:
            import sounddevice as sd  # noqa: F401
        except Exception as exc:
            logger.warning("[STT] sounddevice unavailable: %s — voice commands disabled.", exc)
            return

        self._running = True
        threading.Thread(target=self._record_thread,    daemon=True, name="STTRecord").start()
        threading.Thread(target=self._transcribe_thread, daemon=True, name="STTTranscribe").start()

    def _record_thread(self):
        """Continuously record CHUNK_SECONDS of audio and enqueue."""
        import sounddevice as sd
        logger.info("[STT] Microphone listener started (%.1fs chunks, %d Hz).",
                    CHUNK_SECONDS, SAMPLE_RATE)
        while self._running:
            try:
                audio = sd.rec(
                    int(CHUNK_SECONDS * SAMPLE_RATE),
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                )
                sd.wait()
                # Push-to-talk / software mute gate — checked after sd.wait()
                # so the recording completes normally (no partial chunk artifacts)
                # but the result is simply discarded when muted.
                if self.muted:
                    continue
                chunk = audio.flatten()
                # Silence gate — skip if too quiet
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                if rms < SILENCE_THRESH:
                    continue
                try:
                    self._audio_q.put_nowait(chunk)
                except queue.Full:
                    pass   # drop oldest — transcription is behind, that's ok
            except Exception as exc:
                logger.error("[STT] Record error: %s", exc)
                time.sleep(1.0)

    def _transcribe_thread(self):
        """Pop audio chunks, transcribe with Whisper, classify, enqueue commands."""
        logger.info("[STT] Transcription thread started.")
        while self._running:
            try:
                chunk = self._audio_q.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                segments, info = self._model.transcribe(
                    chunk,
                    language=LANGUAGE,
                    beam_size=1,         # fastest
                    vad_filter=True,     # skip silence automatically
                )

                # ── ASR confidence gate ───────────────────────────────────
                # Drop the chunk if Whisper is not confident it heard English
                # speech.  This prevents garbled noise from being misclassified
                # as voice commands.
                lang_prob = getattr(info, "language_probability", 1.0)
                if lang_prob < STT_MIN_LANG_PROB:
                    logger.debug(
                        "[STT] Dropped: lang_prob=%.2f < %.2f (likely non-speech)",
                        lang_prob, STT_MIN_LANG_PROB,
                    )
                    continue

                # Filter individual segments by no_speech_prob and avg_logprob
                kept_texts = []
                for seg in segments:
                    no_speech  = getattr(seg, "no_speech_prob", 0.0)
                    avg_logprob = getattr(seg, "avg_logprob", 0.0)
                    if no_speech > STT_MAX_NO_SPEECH:
                        logger.debug(
                            "[STT] Segment dropped: no_speech_prob=%.2f > %.2f",
                            no_speech, STT_MAX_NO_SPEECH,
                        )
                        continue
                    if avg_logprob < STT_MIN_AVG_LOGPROB:
                        logger.debug(
                            "[STT] Segment dropped: avg_logprob=%.2f < %.2f",
                            avg_logprob, STT_MIN_AVG_LOGPROB,
                        )
                        continue
                    kept_texts.append(seg.text)

                text = " ".join(kept_texts).strip()
                if not text:
                    continue
                # Minimum length gate — single-character outputs from Whisper
                # (e.g. ".", "I", "A") are near-certain hallucinations from
                # background noise that slipped past VAD and lang_prob checks.
                if len(text) < MIN_TRANSCRIPT_CHARS:
                    logger.debug(
                        "[STT] Dropped: transcript too short (%d chars): %r",
                        len(text), text,
                    )
                    continue
                logger.info("[STT] Heard (lang_prob=%.2f): %r", lang_prob, text)
                action = classify(text)
                if action:
                    logger.info("[STT] Command: %s", action)
                    try:
                        self.cmd_q.put_nowait(action)
                    except queue.Full:
                        logger.warning("[STT] cmd_q full (maxsize=20) — dropping command: %s", action)
                else:
                    logger.debug("[STT] Unrecognised: %r", text)
                    try:
                        self.unrecognised_q.put_nowait(text)
                    except queue.Full:
                        pass  # drop oldest — feedback is best-effort
            except Exception as exc:
                logger.error("[STT] Transcribe error: %s", exc)


stt_engine = STTEngine()
