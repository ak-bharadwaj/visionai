"""
tests/test_broadcast.py

Tests for pipeline._send() and tts._worker() broadcast failure handling.

Covers:
  - pipeline._send(): broadcast future failure is logged, not re-raised
  - pipeline._send(): scheduling error (run_coroutine_threadsafe raises) is caught
  - pipeline._send(): timeout on future.result is logged, not re-raised
  - tts._worker(): broadcast future failure is logged, worker continues
  - tts._worker(): broadcast scheduling error is logged, worker continues
  - tts._worker(): timeout on future.result is logged, worker continues
"""
import asyncio
import concurrent.futures
import queue
import threading
import time
import unittest
from unittest.mock import MagicMock, patch, call

from backend.tts import TTSEngine


# ─────────────────────────────────────────────────────────────────────────────
# pipeline._send() tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineSendBroadcastFailure(unittest.TestCase):
    """
    _send() must never raise — all broadcast failures must be caught and logged.
    """

    def _make_runner(self):
        """Return a PipelineRunner with mocked broadcast and loop."""
        # We import here so patches applied in tests are effective
        from backend.pipeline import PipelineRunner
        pl = PipelineRunner()
        pl._loop = MagicMock(spec=asyncio.AbstractEventLoop)
        return pl

    def test_send_catches_future_exception(self):
        """If future.result() raises, _send must not propagate it."""
        pl = self._make_runner()

        bad_future = concurrent.futures.Future()
        bad_future.set_exception(RuntimeError("connection lost"))

        pl._bcast = MagicMock(return_value=None)

        with patch("asyncio.run_coroutine_threadsafe", return_value=bad_future):
            # Must not raise
            pl._send({"type": "speak", "text": "hello"})

    def test_send_catches_scheduling_error(self):
        """If run_coroutine_threadsafe itself raises, _send must not propagate it."""
        pl = self._make_runner()
        pl._bcast = MagicMock()

        with patch("asyncio.run_coroutine_threadsafe",
                   side_effect=RuntimeError("loop closed")):
            # Must not raise
            pl._send({"type": "narration", "text": "test"})

    def test_send_catches_timeout(self):
        """If future.result() times out, _send must not propagate it."""
        pl = self._make_runner()
        pl._bcast = MagicMock()

        # Create a future that never completes (so .result(timeout=...) times out)
        pending_future = concurrent.futures.Future()
        # We'll patch future.result directly to raise TimeoutError
        pending_future_mock = MagicMock()
        pending_future_mock.result.side_effect = concurrent.futures.TimeoutError()

        with patch("asyncio.run_coroutine_threadsafe", return_value=pending_future_mock):
            # Must not raise
            pl._send({"type": "answer", "text": "test"})

    def test_send_no_op_when_loop_is_none(self):
        """_send must silently skip if _loop is None."""
        pl = self._make_runner()
        pl._loop = None
        pl._bcast = MagicMock()
        # Must not raise
        pl._send({"type": "speak", "text": "hello"})
        pl._bcast.assert_not_called()

    def test_send_no_op_when_bcast_is_none(self):
        """_send must silently skip if _bcast is None."""
        pl = self._make_runner()
        pl._bcast = None
        with patch("asyncio.run_coroutine_threadsafe") as mock_rct:
            pl._send({"type": "speak", "text": "hello"})
            mock_rct.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# tts._worker() tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTTSWorkerBroadcastFailure(unittest.TestCase):
    """
    TTS worker must not stall or crash on broadcast failures.
    The worker thread must continue processing subsequent messages.
    """

    def _make_engine_with_mock_broadcast(self):
        e = TTSEngine()
        e._last_nav_time = 0.0
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        e._loop = loop
        e._broadcast = MagicMock()
        return e

    def _run_worker_drain(self, engine, texts, future_factory):
        """
        Enqueue `texts` into the engine queue, patch asyncio.run_coroutine_threadsafe
        to return future_factory(), run the worker until all items are processed,
        then return.
        """
        results = []

        def fake_rct(coro, loop):
            fut = future_factory()
            results.append(fut)
            return fut

        drained = threading.Event()
        original_task_done = engine._q.task_done

        processed = []

        def counting_task_done():
            original_task_done()
            processed.append(1)
            if len(processed) >= len(texts):
                drained.set()

        engine._q.task_done = counting_task_done

        with patch("asyncio.run_coroutine_threadsafe", side_effect=fake_rct):
            t = threading.Thread(target=engine._worker, daemon=True)
            t.start()

            for text in texts:
                engine._q.put(text)

            drained.wait(timeout=3.0)

        return processed

    def test_worker_continues_after_future_exception(self):
        """Worker must process the second message after the first future fails."""
        engine = self._make_engine_with_mock_broadcast()

        call_count = [0]

        def make_failing_future():
            call_count[0] += 1
            f = concurrent.futures.Future()
            if call_count[0] == 1:
                f.set_exception(RuntimeError("broadcast failed"))
            else:
                f.set_result(None)
            return f

        processed = self._run_worker_drain(engine, ["first", "second"], make_failing_future)
        self.assertEqual(len(processed), 2,
                         "Worker must process both messages even if first broadcast fails")

    def test_worker_continues_after_scheduling_error(self):
        """Worker must process the second message after run_coroutine_threadsafe raises."""
        engine = self._make_engine_with_mock_broadcast()

        call_count = [0]

        def fake_rct_raises(coro, loop):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("loop not running")
            f = concurrent.futures.Future()
            f.set_result(None)
            return f

        processed_count = [0]
        drained = threading.Event()

        original_task_done = engine._q.task_done
        def counting_task_done():
            original_task_done()
            processed_count[0] += 1
            if processed_count[0] >= 2:
                drained.set()

        engine._q.task_done = counting_task_done

        with patch("asyncio.run_coroutine_threadsafe", side_effect=fake_rct_raises):
            t = threading.Thread(target=engine._worker, daemon=True)
            t.start()
            engine._q.put("first")
            engine._q.put("second")
            drained.wait(timeout=3.0)

        self.assertEqual(processed_count[0], 2,
                         "Worker must process both messages even if first scheduling raises")

    def test_worker_continues_after_timeout(self):
        """Worker must process the second message after first future.result() times out."""
        engine = self._make_engine_with_mock_broadcast()

        call_count = [0]

        def make_timeout_then_ok_future():
            call_count[0] += 1
            m = MagicMock()
            if call_count[0] == 1:
                m.result.side_effect = concurrent.futures.TimeoutError()
            else:
                m.result.return_value = None
            return m

        processed = self._run_worker_drain(
            engine, ["first", "second"], make_timeout_then_ok_future
        )
        self.assertEqual(len(processed), 2,
                         "Worker must process both messages even if first future times out")


# ─────────────────────────────────────────────────────────────────────────────
# Voice-state watchdog initial value test
# ─────────────────────────────────────────────────────────────────────────────

class TestVoiceStateWatchdogInitialValue(unittest.TestCase):
    """
    _voice_state_since must not be 0.0 at startup — a zero epoch would make
    the watchdog think the system has been stuck for billions of seconds and
    immediately force-reset to IDLE on first tick.
    """

    def test_voice_state_since_is_not_zero(self):
        import backend.main as main_mod
        self.assertGreater(
            main_mod._voice_state_since, 0.0,
            "_voice_state_since must be initialized to time.monotonic(), not 0.0",
        )

    def test_voice_state_since_is_recent(self):
        """
        _voice_state_since should be within 60 seconds of now
        (i.e. it was set at import time, not at the Unix epoch).
        """
        import backend.main as main_mod
        age = time.monotonic() - main_mod._voice_state_since
        self.assertLess(
            age, 60.0,
            f"_voice_state_since is {age:.1f}s in the past — expected < 60s",
        )


if __name__ == "__main__":
    unittest.main()
