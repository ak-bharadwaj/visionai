"""
test_stt.py — Unit tests for backend/stt.py

Tests cover:
  - classify(): all registered command patterns (original + new NL aliases)
  - classify(): unrecognised input returns None
  - classify(): repetition suppression within DEDUP_COMMAND_WINDOW
  - Minimum transcript length gate (MIN_TRANSCRIPT_CHARS)
  - STTEngine.muted push-to-talk flag
  - Help hint rotation constants
  - ASR confidence gate constants are properly typed and in range
  - _transcribe_thread logic via STTEngine internal helper (mocked Whisper)
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock, patch
import backend.stt as stt_module
from backend.stt import (
    classify,
    STTEngine,
    STT_MIN_LANG_PROB,
    STT_MAX_NO_SPEECH,
    STT_MIN_AVG_LOGPROB,
    DEDUP_COMMAND_WINDOW,
    MIN_TRANSCRIPT_CHARS,
    _action_key,
)


def _reset_dedup():
    """Reset the module-level repetition suppression state between tests."""
    stt_module._last_cmd_key  = ""
    stt_module._last_cmd_time = 0.0


# ── classify() — command routing ──────────────────────────────────────────────

class TestClassify:
    def setup_method(self):
        _reset_dedup()   # each test starts with a clean dedup slate

    def test_find_object(self):
        r = classify("find the chair")
        assert r is not None
        assert r["action"] == "find_object"
        assert r["target"] == "the chair"

    def test_find_strips_punctuation(self):
        _reset_dedup()
        r = classify("find door!")
        assert r["target"] == "door"

    # Natural language FIND aliases
    def test_locate_alias(self):
        r = classify("locate the bottle")
        assert r is not None
        assert r["action"] == "find_object"
        assert "bottle" in r["target"]

    def test_where_is_alias(self):
        r = classify("where is my keys")
        assert r is not None
        assert r["action"] == "find_object"
        assert "keys" in r["target"]

    def test_look_for_alias(self):
        r = classify("look for the exit")
        assert r is not None
        assert r["action"] == "find_object"
        assert "exit" in r["target"]

    def test_whats_ahead(self):
        r = classify("what's ahead")
        assert r["action"] == "ask"
        assert "ahead" in r["question"].lower()

    def test_what_is_ahead(self):
        r = classify("what is ahead")
        assert r["action"] == "ask"

    def test_what_is_in_front(self):
        r = classify("what is in front")
        assert r["action"] == "ask"

    def test_what_do_you_see(self):
        r = classify("what do you see")
        assert r["action"] == "ask"

    def test_what_can_you_see(self):
        r = classify("what can you see")
        assert r["action"] == "ask"

    # Natural language ASK aliases
    def test_describe_this_alias(self):
        r = classify("describe this")
        assert r is not None
        assert r["action"] == "ask"

    def test_what_is_this_alias(self):
        r = classify("what is this")
        assert r is not None
        assert r["action"] == "ask"

    def test_is_path_clear(self):
        r = classify("is the path clear")
        assert r["action"] == "ask"

    def test_read_text(self):
        r = classify("read text")
        assert r["action"] == "set_mode"
        assert r["mode"] == "READ"

    def test_read_sign(self):
        r = classify("read sign")
        assert r["action"] == "set_mode"
        assert r["mode"] == "READ"

    def test_read_this_alias(self):
        r = classify("read this")
        assert r is not None
        assert r["action"] == "set_mode"
        assert r["mode"] == "READ"

    def test_scan_text_alias(self):
        r = classify("scan text")
        assert r is not None
        assert r["action"] == "set_mode"
        assert r["mode"] == "READ"

    def test_navigate(self):
        r = classify("navigate")
        assert r["action"] == "set_mode"
        assert r["mode"] == "NAVIGATE"

    # Natural language NAVIGATE aliases
    def test_guide_me_alias(self):
        r = classify("guide me")
        assert r is not None
        assert r["action"] == "set_mode"
        assert r["mode"] == "NAVIGATE"

    def test_walk_mode_alias(self):
        r = classify("walk mode")
        assert r is not None
        assert r["action"] == "set_mode"
        assert r["mode"] == "NAVIGATE"

    def test_start_navigation_alias(self):
        r = classify("start navigation")
        assert r is not None
        assert r["action"] == "set_mode"
        assert r["mode"] == "NAVIGATE"

    def test_help_me_walk_alias(self):
        r = classify("help me walk")
        assert r is not None
        assert r["action"] == "set_mode"
        assert r["mode"] == "NAVIGATE"

    def test_scan(self):
        r = classify("scan")
        assert r["action"] == "set_mode"
        assert r["mode"] == "SCAN"

    def test_stop_narration(self):
        r = classify("stop narration")
        assert r["action"] == "set_mode"
        assert r["mode"] == "NAVIGATE"

    def test_be_quiet(self):
        r = classify("be quiet")
        assert r["action"] == "set_mode"
        assert r["mode"] == "NAVIGATE"

    def test_repeat(self):
        r = classify("repeat")
        assert r["action"] == "repeat"

    def test_repeat_that(self):
        _reset_dedup()
        r = classify("repeat that")
        assert r["action"] == "repeat"

    def test_remember_this(self):
        r = classify("remember this")
        assert r["action"] == "snapshot"

    def test_what_changed(self):
        r = classify("what changed")
        assert r["action"] == "scene_diff"

    def test_what_has_changed(self):
        _reset_dedup()
        r = classify("what has changed")
        assert r["action"] == "scene_diff"

    def test_ask_prefix(self):
        r = classify("ask is there a step ahead")
        assert r is not None
        assert r["action"] == "ask"
        assert "step" in r["question"]

    def test_unrecognised_returns_none(self):
        assert classify("hello there general kenobi") is None

    def test_empty_string_returns_none(self):
        assert classify("") is None

    def test_whitespace_only_returns_none(self):
        assert classify("   ") is None

    def test_case_insensitive(self):
        r = classify("NAVIGATE")
        assert r is not None
        assert r["mode"] == "NAVIGATE"

    def test_find_ignores_empty_target(self):
        """'find ' with no target should return None (empty target stripped)."""
        r = classify("find")
        assert r is None


# ── ASR confidence gate constants ─────────────────────────────────────────────

class TestASTConfidenceGateConstants:
    def test_min_lang_prob_in_range(self):
        assert 0.0 <= STT_MIN_LANG_PROB <= 1.0

    def test_max_no_speech_in_range(self):
        assert 0.0 <= STT_MAX_NO_SPEECH <= 1.0

    def test_min_avg_logprob_is_negative(self):
        """avg_logprob is a log-probability; it must be <= 0."""
        assert STT_MIN_AVG_LOGPROB <= 0.0

    def test_defaults_are_sensible(self):
        """Defaults must not be so strict they block all real speech."""
        assert STT_MIN_LANG_PROB <= 0.85
        assert STT_MAX_NO_SPEECH >= 0.40
        assert STT_MIN_AVG_LOGPROB >= -3.0


# ── _transcribe_thread confidence gate (mocked Whisper) ──────────────────────

def _make_segment(text, no_speech_prob=0.05, avg_logprob=-0.3):
    seg = MagicMock()
    seg.text = text
    seg.no_speech_prob = no_speech_prob
    seg.avg_logprob = avg_logprob
    return seg


def _make_info(language_probability=0.95):
    info = MagicMock()
    info.language_probability = language_probability
    return info


class TestASRConfidenceGate:
    """
    Test the segment filtering logic directly without running actual threads.
    We replicate the logic from _transcribe_thread here against mocked segments.
    """

    def _filter(self, segments, info):
        """Mirror the gate logic from _transcribe_thread."""
        lang_prob = getattr(info, "language_probability", 1.0)
        if lang_prob < STT_MIN_LANG_PROB:
            return None   # whole chunk rejected

        kept = []
        for seg in segments:
            no_speech = getattr(seg, "no_speech_prob", 0.0)
            avg_lp    = getattr(seg, "avg_logprob", 0.0)
            if no_speech > STT_MAX_NO_SPEECH:
                continue
            if avg_lp < STT_MIN_AVG_LOGPROB:
                continue
            kept.append(seg.text)
        return " ".join(kept).strip() or None

    def test_good_segment_passes(self):
        seg  = _make_segment("navigate", no_speech_prob=0.05, avg_logprob=-0.2)
        info = _make_info(language_probability=0.95)
        result = self._filter([seg], info)
        assert result == "navigate"

    def test_low_lang_prob_rejects_whole_chunk(self):
        seg  = _make_segment("navigate", no_speech_prob=0.05, avg_logprob=-0.2)
        info = _make_info(language_probability=STT_MIN_LANG_PROB - 0.05)
        result = self._filter([seg], info)
        assert result is None

    def test_high_no_speech_prob_drops_segment(self):
        seg  = _make_segment("navigate", no_speech_prob=STT_MAX_NO_SPEECH + 0.05, avg_logprob=-0.2)
        info = _make_info(language_probability=0.95)
        result = self._filter([seg], info)
        assert result is None

    def test_low_avg_logprob_drops_segment(self):
        seg  = _make_segment("navigate", no_speech_prob=0.05, avg_logprob=STT_MIN_AVG_LOGPROB - 0.5)
        info = _make_info(language_probability=0.95)
        result = self._filter([seg], info)
        assert result is None

    def test_mixed_segments_keeps_good_drops_bad(self):
        good = _make_segment("read text",  no_speech_prob=0.05, avg_logprob=-0.2)
        bad  = _make_segment("hmm noise",  no_speech_prob=STT_MAX_NO_SPEECH + 0.1, avg_logprob=-0.2)
        info = _make_info(language_probability=0.95)
        result = self._filter([good, bad], info)
        assert result == "read text"

    def test_all_bad_segments_returns_none(self):
        bad1 = _make_segment("noise",  no_speech_prob=STT_MAX_NO_SPEECH + 0.1, avg_logprob=-0.2)
        bad2 = _make_segment("static", no_speech_prob=0.05, avg_logprob=STT_MIN_AVG_LOGPROB - 1.0)
        info = _make_info(language_probability=0.95)
        result = self._filter([bad1, bad2], info)
        assert result is None

    def test_missing_attributes_default_to_passing(self):
        """Segments without no_speech_prob / avg_logprob attrs must still pass."""
        seg = MagicMock(spec=[])   # no attributes at all
        seg.text = "navigate"
        info = _make_info(language_probability=0.95)
        result = self._filter([seg], info)
        assert result == "navigate"

    def test_boundary_lang_prob_exactly_at_threshold_passes(self):
        """lang_prob exactly equal to STT_MIN_LANG_PROB must pass."""
        seg  = _make_segment("navigate")
        info = _make_info(language_probability=STT_MIN_LANG_PROB)
        result = self._filter([seg], info)
        assert result == "navigate"


# ── _action_key() ─────────────────────────────────────────────────────────────

class TestActionKey:
    def test_set_mode_key(self):
        assert _action_key({"action": "set_mode", "mode": "NAVIGATE"}) == "set_mode:NAVIGATE"

    def test_find_object_key_lowercased(self):
        assert _action_key({"action": "find_object", "target": "Bottle"}) == "find_object:bottle"

    def test_ask_key_lowercased(self):
        k = _action_key({"action": "ask", "question": "What is ahead?"})
        assert k == "ask:what is ahead?"

    def test_repeat_key(self):
        assert _action_key({"action": "repeat"}) == "repeat"

    def test_snapshot_key(self):
        assert _action_key({"action": "snapshot"}) == "snapshot"

    def test_scene_diff_key(self):
        assert _action_key({"action": "scene_diff"}) == "scene_diff"

    def test_unknown_action_key(self):
        assert _action_key({"action": "unknown_thing"}) == "unknown_thing"


# ── Repetition suppression ────────────────────────────────────────────────────

class TestRepetitionSuppression:
    """
    classify() must suppress identical commands within DEDUP_COMMAND_WINDOW
    and allow them again once the window expires.
    """

    def setup_method(self):
        _reset_dedup()

    def test_first_call_passes(self):
        r = classify("navigate")
        assert r is not None
        assert r["action"] == "set_mode"

    def test_immediate_repeat_suppressed(self):
        classify("navigate")   # first — passes
        r = classify("navigate")  # immediate repeat — suppressed
        assert r is None

    def test_different_command_not_suppressed(self):
        classify("navigate")
        r = classify("scan")   # different command — must pass
        assert r is not None
        assert r["mode"] == "SCAN"

    def test_repeat_after_window_passes(self):
        """After the dedup window expires the same command must be accepted."""
        classify("navigate")
        # Artificially expire the window
        stt_module._last_cmd_time = time.monotonic() - DEDUP_COMMAND_WINDOW - 0.1
        r = classify("navigate")
        assert r is not None

    def test_find_object_dedup_by_target(self):
        """Two 'find bottle' commands close together → second is suppressed."""
        classify("find bottle")
        r = classify("find bottle")
        assert r is None

    def test_find_different_targets_both_pass(self):
        """'find bottle' then 'find chair' are different keys — both pass."""
        classify("find bottle")
        r = classify("find chair")
        assert r is not None
        assert "chair" in r["target"]

    def test_dedup_window_constant_is_positive(self):
        assert DEDUP_COMMAND_WINDOW > 0.0


# ── STTEngine.unrecognised_q ──────────────────────────────────────────────────

class TestUnrecognisedQueue:
    def test_engine_has_unrecognised_q(self):
        engine = STTEngine()
        assert hasattr(engine, "unrecognised_q")
        import queue
        assert isinstance(engine.unrecognised_q, queue.Queue)

    def test_unrecognised_q_initially_empty(self):
        engine = STTEngine()
        assert engine.unrecognised_q.empty()


# ── Interaction diagnostics ───────────────────────────────────────────────────

class TestInteractionDiagnostics:
    """
    Verify the new voice / interaction counters in Diagnostics.
    """

    def setup_method(self):
        from backend.diagnostics import Diagnostics
        self.diag = Diagnostics()  # fresh instance per test

    def test_initial_voice_counters_zero(self):
        assert self.diag.voice_commands_received == 0
        assert self.diag.voice_commands_rejected == 0
        assert self.diag.mode_changes == 0
        assert self.diag.tts_interruptions == 0

    def test_voice_command_received_increments(self):
        self.diag.voice_command_received(action="set_mode", detail="NAVIGATE")
        assert self.diag.voice_commands_received == 1

    def test_voice_command_rejected_increments(self):
        self.diag.voice_command_rejected(reason="unrecognised")
        assert self.diag.voice_commands_rejected == 1

    def test_mode_changed_increments(self):
        self.diag.mode_changed(old_mode="NAVIGATE", new_mode="READ", source="voice")
        assert self.diag.mode_changes == 1

    def test_tts_interrupted_increments(self):
        self.diag.tts_interrupted()
        assert self.diag.tts_interruptions == 1

    def test_command_recognition_rate_no_data(self):
        assert self.diag.command_recognition_rate() == 1.0

    def test_command_recognition_rate_all_received(self):
        self.diag.voice_command_received("set_mode", "NAVIGATE")
        self.diag.voice_command_received("find_object", "bottle")
        assert self.diag.command_recognition_rate() == 1.0

    def test_command_recognition_rate_mixed(self):
        self.diag.voice_command_received("set_mode", "NAVIGATE")
        self.diag.voice_command_rejected("unrecognised")
        rate = self.diag.command_recognition_rate()
        assert abs(rate - 0.5) < 1e-9

    def test_command_recognition_rate_all_rejected(self):
        self.diag.voice_command_rejected()
        self.diag.voice_command_rejected()
        assert self.diag.command_recognition_rate() == 0.0

    def test_summary_includes_interaction_keys(self):
        s = self.diag.summary()
        for key in (
            "voice_commands_received",
            "voice_commands_rejected",
            "mode_changes",
            "tts_interruptions",
            "command_recognition_rate",
        ):
            assert key in s, f"Missing key in summary: {key}"

    def test_multiple_threads_safe(self):
        """Concurrent increments must not corrupt counters."""
        import threading
        def _recv():
            for _ in range(100):
                self.diag.voice_command_received("set_mode", "NAVIGATE")
        def _rej():
            for _ in range(100):
                self.diag.voice_command_rejected()
        threads = [threading.Thread(target=_recv), threading.Thread(target=_rej)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert self.diag.voice_commands_received == 100
        assert self.diag.voice_commands_rejected == 100


# ── VoiceState enum ──────────────────────────────────────────────────────────

class TestVoiceStateEnum:
    def test_voice_state_importable(self):
        from backend.main import VoiceState
        assert VoiceState.IDLE is not None
        assert VoiceState.LISTENING is not None
        assert VoiceState.PROCESSING is not None
        assert VoiceState.SPEAKING is not None

    def test_voice_state_initial_idle(self):
        from backend.main import _voice_state, VoiceState
        # The module-level default must be IDLE at import time.
        assert _voice_state == VoiceState.IDLE

    def test_voice_state_values_distinct(self):
        from backend.main import VoiceState
        states = [VoiceState.IDLE, VoiceState.LISTENING, VoiceState.PROCESSING, VoiceState.SPEAKING]
        assert len(set(states)) == 4


# ── Minimum transcript length gate ───────────────────────────────────────────

class TestMinTranscriptChars:
    """
    MIN_TRANSCRIPT_CHARS filters out single-character Whisper hallucinations
    (e.g. ".", "I", "A") before they reach classify() or unrecognised_q.
    The gate lives in _transcribe_thread; we test the constant and the logic
    mirror used in the existing confidence gate test helper.
    """

    def test_constant_is_positive(self):
        assert MIN_TRANSCRIPT_CHARS > 0

    def test_constant_is_at_least_two(self):
        """Single characters are always hallucinations; minimum must be >= 2."""
        assert MIN_TRANSCRIPT_CHARS >= 2

    def test_constant_is_not_too_large(self):
        """The shortest valid command ('scan') is 4 chars; gate must be <= 4."""
        assert MIN_TRANSCRIPT_CHARS <= 4

    def test_single_char_below_gate(self):
        assert len(".") < MIN_TRANSCRIPT_CHARS

    def test_single_char_I_below_gate(self):
        assert len("I") < MIN_TRANSCRIPT_CHARS

    def test_short_word_navigate_above_gate(self):
        assert len("navigate") >= MIN_TRANSCRIPT_CHARS

    def test_short_word_scan_above_gate(self):
        assert len("scan") >= MIN_TRANSCRIPT_CHARS

    def _filter_with_length_gate(self, text: str) -> bool:
        """Return True if text would pass the length gate."""
        return len(text.strip()) >= MIN_TRANSCRIPT_CHARS

    def test_gate_passes_valid_transcript(self):
        assert self._filter_with_length_gate("navigate") is True

    def test_gate_rejects_single_char(self):
        assert self._filter_with_length_gate("I") is False

    def test_gate_rejects_period(self):
        assert self._filter_with_length_gate(".") is False

    def test_gate_rejects_whitespace_only(self):
        # strip() reduces to empty string → length 0
        assert self._filter_with_length_gate("  ") is False


# ── STTEngine.muted push-to-talk flag ────────────────────────────────────────

class TestSTTMutedFlag:
    def test_engine_muted_attribute_exists(self):
        engine = STTEngine()
        assert hasattr(engine, "muted")

    def test_engine_muted_default_false(self):
        """STT must start unmuted so voice commands work immediately."""
        engine = STTEngine()
        assert engine.muted is False

    def test_muted_can_be_set_true(self):
        engine = STTEngine()
        engine.muted = True
        assert engine.muted is True

    def test_muted_can_be_toggled(self):
        engine = STTEngine()
        engine.muted = True
        engine.muted = False
        assert engine.muted is False


# ── Help hint constants ───────────────────────────────────────────────────────

class TestHelpHintConstants:
    """
    The dispatcher uses _HELP_HINTS and _HELP_HINT_EVERY (defined as locals).
    We verify the expected structural properties via the module-level dispatcher
    source rather than importing private locals — instead we check behaviour
    through the stt module constants that drive it.
    """

    def test_help_hints_list_non_empty(self):
        """There must be at least 2 distinct help hints to rotate through."""
        # We import main to check; the list is defined as a local in the
        # coroutine so we verify via inspection of the source string.
        import inspect
        import backend.main as main_module
        src = inspect.getsource(main_module._stt_dispatcher)
        assert "_HELP_HINTS" in src
        assert "You can say" in src or "Try saying" in src

    def test_hint_every_constant_present(self):
        import inspect
        import backend.main as main_module
        src = inspect.getsource(main_module._stt_dispatcher)
        assert "_HELP_HINT_EVERY" in src

    def test_stt_mute_action_handled(self):
        """WebSocket handler must contain stt_mute / stt_unmute action branches."""
        import inspect
        import backend.main as main_module
        src = inspect.getsource(main_module.ws_endpoint)
        assert "stt_mute" in src
        assert "stt_unmute" in src


# ── nav_destination patterns ──────────────────────────────────────────────────

class TestNavDestinationPatterns:
    """
    Verify that 'navigate to X', 'go to X', 'take me to X', etc. are
    classified as nav_destination actions (not as set_mode: NAVIGATE).
    """

    def setup_method(self):
        _reset_dedup()

    def test_navigate_to_kitchen(self):
        r = classify("navigate to the kitchen")
        assert r is not None
        assert r["action"] == "nav_destination"
        assert r["destination"] == "the kitchen"

    def test_go_to_exit(self):
        _reset_dedup()
        r = classify("go to the exit")
        assert r is not None
        assert r["action"] == "nav_destination"
        assert "exit" in r["destination"]

    def test_head_to_bedroom(self):
        _reset_dedup()
        r = classify("head to the bedroom")
        assert r is not None
        assert r["action"] == "nav_destination"
        assert "bedroom" in r["destination"]

    def test_walk_to_door(self):
        _reset_dedup()
        r = classify("walk to the door")
        assert r is not None
        assert r["action"] == "nav_destination"
        assert "door" in r["destination"]

    def test_take_me_to_bathroom(self):
        _reset_dedup()
        r = classify("take me to the bathroom")
        assert r is not None
        assert r["action"] == "nav_destination"
        assert "bathroom" in r["destination"]

    def test_destination_keyword(self):
        _reset_dedup()
        r = classify("destination kitchen")
        assert r is not None
        assert r["action"] == "nav_destination"
        assert "kitchen" in r["destination"]

    def test_i_want_to_go_to(self):
        _reset_dedup()
        r = classify("i want to go to the elevator")
        assert r is not None
        assert r["action"] == "nav_destination"
        assert "elevator" in r["destination"]

    def test_bare_navigate_still_set_mode(self):
        """'navigate' alone (no destination) → set_mode: NAVIGATE, not nav_destination."""
        _reset_dedup()
        r = classify("navigate")
        assert r is not None
        assert r["action"] == "set_mode"
        assert r["mode"] == "NAVIGATE"

    def test_guide_me_still_set_mode(self):
        """'guide me' → set_mode: NAVIGATE (no destination)."""
        _reset_dedup()
        r = classify("guide me")
        assert r is not None
        assert r["action"] == "set_mode"
        assert r["mode"] == "NAVIGATE"

    def test_nav_destination_strips_punctuation(self):
        _reset_dedup()
        r = classify("go to the kitchen!")
        assert r is not None
        assert r["action"] == "nav_destination"
        assert not r["destination"].endswith("!")

    def test_nav_destination_dedup(self):
        """Same destination twice quickly → second is suppressed."""
        classify("go to the kitchen")
        _reset_dedup()   # manually reset to test the key logic
        # Re-seed with the same key
        import backend.stt as stt_mod
        stt_mod._last_cmd_key  = "nav_destination:the kitchen"
        stt_mod._last_cmd_time = time.monotonic()
        r = classify("go to the kitchen")
        assert r is None

    def test_action_key_nav_destination(self):
        """_action_key must produce a lowercased key for nav_destination."""
        key = _action_key({"action": "nav_destination", "destination": "Kitchen"})
        assert key == "nav_destination:kitchen"

    def test_different_destinations_not_deduped(self):
        """'go to kitchen' then 'go to exit' are different keys — both pass."""
        classify("go to the kitchen")
        _reset_dedup()
        stt_module._last_cmd_key  = "nav_destination:the kitchen"
        stt_module._last_cmd_time = time.monotonic()
        r = classify("go to the exit")
        assert r is not None
        assert r["action"] == "nav_destination"
        assert "exit" in r["destination"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
