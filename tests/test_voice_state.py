"""
tests/test_voice_state.py

Unit tests for the VoiceState machine and watchdog in backend.main.

Tests are designed to run without a running FastAPI server by exercising
the module-level helpers directly via importlib and monkeypatching the
broadcast function.

Covers:
  - _set_voice_state updates module-level state and calls broadcast
  - _voice_state_since is updated on every state transition
  - Watchdog: resets PROCESSING → IDLE after > 10 s
  - Watchdog: resets SPEAKING → IDLE after > 10 s
  - Watchdog: does NOT reset IDLE (correct steady state)
  - Watchdog: does NOT reset if stuck < 10 s
"""
import asyncio
import time
import unittest
from unittest.mock import AsyncMock, patch

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _import_main():
    """
    Import backend.main with all heavy startup imports patched out so we can
    test the state-machine helpers in isolation.
    """
    import sys
    import types

    # Minimal stubs for heavy backend modules that main.py imports at the
    # module level.  We only need the thin layer around VoiceState /
    # _set_voice_state / _voice_state_watchdog.
    stubs = {
        "backend.pipeline":    types.SimpleNamespace(
            pipeline=types.SimpleNamespace(
                start=lambda *a, **kw: None,
                stop=lambda: None,
                get_last_frame=lambda: None,
                fps=0.0,
                reset_mode=lambda: None,
            )
        ),
        "backend.modes":       types.SimpleNamespace(
            mode_manager=types.SimpleNamespace(
                snapshot=lambda: {"current_mode": "NAVIGATE", "show_overlay": True},
                set_mode=lambda m: None,
                toggle_overlay=lambda: None,
            )
        ),
        "backend.brain":       types.SimpleNamespace(
            brain=types.SimpleNamespace(
                clear_history=lambda: None,
                needs_llm=lambda q: False,
                answer=lambda *a, **kw: "stub",
            )
        ),
        "backend.ocr":         types.SimpleNamespace(
            ocr_reader=types.SimpleNamespace(
                clear_history=lambda: None,
                read=lambda *a, **kw: [],
            )
        ),
        "backend.tts":         types.SimpleNamespace(
            tts_engine=types.SimpleNamespace(
                start=lambda *a, **kw: None,
                speak=lambda *a, **kw: None,
            )
        ),
        "backend.stt":         types.SimpleNamespace(
            stt_engine=types.SimpleNamespace(
                start=lambda: None,
                stop=lambda: None,
                muted=False,
                cmd_q=__import__("queue").Queue(),
                unrecognised_q=__import__("queue").Queue(),
            )
        ),
        "backend.diagnostics": types.SimpleNamespace(
            diagnostics=types.SimpleNamespace(
                voice_command_received=lambda **kw: None,
                voice_command_rejected=lambda **kw: None,
                mode_changed=lambda **kw: None,
            )
        ),
        # Third-party stubs
        "qrcode":              types.SimpleNamespace(make=lambda url: types.SimpleNamespace(
            save=lambda buf, format: None)),
    }

    # Register stubs only for modules not already imported
    for mod_name, stub in stubs.items():
        if mod_name not in sys.modules:
            sys.modules[mod_name] = stub   # type: ignore

    # Force a fresh import of backend.main
    if "backend.main" in sys.modules:
        del sys.modules["backend.main"]

    import backend.main as main_mod
    return main_mod


# ─────────────────────────────────────────────────────────────────────────────
# _set_voice_state tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSetVoiceState(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.main = _import_main()
        # Replace broadcast with a no-op async mock
        self.broadcast_calls = []

        async def fake_broadcast(data):
            self.broadcast_calls.append(data)

        self.main.broadcast = fake_broadcast

    async def test_state_is_updated(self):
        VoiceState = self.main.VoiceState
        await self.main._set_voice_state(VoiceState.PROCESSING)
        self.assertIs(self.main._voice_state, VoiceState.PROCESSING)

    async def test_broadcast_called_with_system_message(self):
        VoiceState = self.main.VoiceState
        await self.main._set_voice_state(VoiceState.SPEAKING)
        self.assertTrue(
            any(c.get("type") == "system" and c.get("voice_state") == "SPEAKING"
                for c in self.broadcast_calls),
            "broadcast must be called with {type:system, voice_state:'SPEAKING'}",
        )

    async def test_voice_state_since_updated(self):
        VoiceState = self.main.VoiceState
        before = time.monotonic()
        await self.main._set_voice_state(VoiceState.LISTENING)
        after = time.monotonic()
        self.assertGreaterEqual(self.main._voice_state_since, before)
        self.assertLessEqual(self.main._voice_state_since, after)

    async def test_idle_to_idle_transition(self):
        """Setting IDLE when already IDLE should not raise."""
        VoiceState = self.main.VoiceState
        await self.main._set_voice_state(VoiceState.IDLE)
        self.assertIs(self.main._voice_state, VoiceState.IDLE)


# ─────────────────────────────────────────────────────────────────────────────
# _voice_state_watchdog tests
# ─────────────────────────────────────────────────────────────────────────────

class TestVoiceStateWatchdog(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.main = _import_main()
        self.broadcast_calls = []

        async def fake_broadcast(data):
            self.broadcast_calls.append(data)

        self.main.broadcast = fake_broadcast

    async def _run_watchdog_once(self, sleep_patch_val=0.0):
        """
        Run one cycle of the watchdog with asyncio.sleep patched to return
        immediately, then cancel the task.
        """
        original_sleep = asyncio.sleep

        call_count = [0]

        async def fast_sleep(delay):
            call_count[0] += 1
            if call_count[0] > 2:
                raise asyncio.CancelledError()
            await original_sleep(0)   # yield control without blocking

        with patch("asyncio.sleep", side_effect=fast_sleep):
            task = asyncio.create_task(self.main._voice_state_watchdog())
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def test_watchdog_resets_processing_after_timeout(self):
        VoiceState = self.main.VoiceState
        self.main._voice_state = VoiceState.PROCESSING
        # Backdate the state entry time so it appears stuck for 15 s
        self.main._voice_state_since = time.monotonic() - 15.0

        await self._run_watchdog_once()

        self.assertIs(
            self.main._voice_state, VoiceState.IDLE,
            "Watchdog must reset PROCESSING → IDLE after >10 s.",
        )

    async def test_watchdog_resets_speaking_after_timeout(self):
        VoiceState = self.main.VoiceState
        self.main._voice_state = VoiceState.SPEAKING
        self.main._voice_state_since = time.monotonic() - 12.0

        await self._run_watchdog_once()

        self.assertIs(
            self.main._voice_state, VoiceState.IDLE,
            "Watchdog must reset SPEAKING → IDLE after >10 s.",
        )

    async def test_watchdog_does_not_reset_idle(self):
        VoiceState = self.main.VoiceState
        self.main._voice_state = VoiceState.IDLE
        self.main._voice_state_since = time.monotonic() - 30.0   # very old

        await self._run_watchdog_once()

        self.assertIs(
            self.main._voice_state, VoiceState.IDLE,
            "Watchdog must not interfere with IDLE state.",
        )

    async def test_watchdog_does_not_reset_before_timeout(self):
        VoiceState = self.main.VoiceState
        self.main._voice_state = VoiceState.PROCESSING
        # Only 3 seconds elapsed — within the 10 s window
        self.main._voice_state_since = time.monotonic() - 3.0

        await self._run_watchdog_once()

        self.assertIs(
            self.main._voice_state, VoiceState.PROCESSING,
            "Watchdog must NOT reset state that has been stuck < 10 s.",
        )

    async def test_watchdog_broadcasts_idle_on_reset(self):
        VoiceState = self.main.VoiceState
        self.main._voice_state = VoiceState.PROCESSING
        self.main._voice_state_since = time.monotonic() - 20.0

        await self._run_watchdog_once()

        self.assertTrue(
            any(c.get("type") == "system" and c.get("voice_state") == "IDLE"
                for c in self.broadcast_calls),
            "Watchdog reset must broadcast {type:system, voice_state:'IDLE'}.",
        )

    async def test_watchdog_logs_warning_on_reset(self):
        VoiceState = self.main.VoiceState
        self.main._voice_state = VoiceState.SPEAKING
        self.main._voice_state_since = time.monotonic() - 11.0

        with self.assertLogs("backend.main", level="WARNING") as cm:
            await self._run_watchdog_once()

        self.assertTrue(
            any("watchdog" in line.lower() or "stuck" in line.lower() for line in cm.output),
            "Watchdog must emit a WARNING log when it resets state.",
        )


if __name__ == "__main__":
    unittest.main()
