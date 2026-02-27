"""
tests/test_tts.py

Unit tests for backend.tts.TTSEngine.
Covers:
  - Priority=True bypasses dedup
  - Non-priority dedup suppression within window
  - Non-priority rate-limit (NAV_RATE_LIMIT)
  - Priority clears the queue
  - Drop logging (dedup and rate-limit paths)
  - Queue-full: drop and log
  - Worker broadcasts speak message
"""
import asyncio
import queue
import time
import unittest
from unittest.mock import MagicMock, patch, call

from backend.tts import TTSEngine


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _engine() -> TTSEngine:
    """Return a fresh TTSEngine with _last_nav_time far in the past."""
    e = TTSEngine()
    e._last_nav_time = 0.0
    return e


# ─────────────────────────────────────────────────────────────────────────────
# Dedup behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestTTSDedupNonPriority(unittest.TestCase):
    """Non-priority calls must be deduplicated within DEDUP_WINDOW."""

    def test_first_call_always_queued(self):
        e = _engine()
        e.speak("hello", priority=False)
        self.assertFalse(e._q.empty(), "First speak should be queued.")

    def test_duplicate_within_window_dropped(self):
        e = _engine()
        e.speak("hello", priority=False)
        e._q.get_nowait()   # drain
        e.speak("hello", priority=False)   # within 0 s — well inside window
        self.assertTrue(e._q.empty(), "Identical non-priority text within window should be dropped.")

    def test_duplicate_after_window_queued(self):
        e = _engine()
        e.speak("hello", priority=False)
        e._q.get_nowait()
        # Backdate both the dedup timestamp AND the nav rate-limit timestamp
        # so that neither gate blocks the second call.
        e._last["hello"]    = time.time() - TTSEngine.DEDUP_WINDOW  - 1.0
        e._last_nav_time    = time.time() - TTSEngine.NAV_RATE_LIMIT - 1.0
        e.speak("hello", priority=False)
        self.assertFalse(e._q.empty(), "Identical text after window expiry should be re-queued.")

    def test_dedup_logs_debug_on_drop(self):
        e = _engine()
        e.speak("hello", priority=False)
        e._q.get_nowait()
        with self.assertLogs("backend.tts", level="DEBUG") as cm:
            e.speak("hello", priority=False)
        self.assertTrue(
            any("dedup" in line for line in cm.output),
            "A DEBUG log mentioning 'dedup' must be emitted on drop.",
        )


class TestTTSDedupPriority(unittest.TestCase):
    """Priority=True MUST bypass dedup — identical answers must always be spoken."""

    def test_priority_bypasses_dedup(self):
        e = _engine()
        e.speak("answer text", priority=True)
        e._q.get_nowait()   # drain
        # Second identical call — should NOT be dropped even though it's within the window
        e.speak("answer text", priority=True)
        self.assertFalse(
            e._q.empty(),
            "Priority speak of identical text must NOT be deduplicated.",
        )

    def test_priority_bypasses_dedup_cross_call(self):
        """Non-priority first, then priority: priority must still go through."""
        e = _engine()
        e.speak("same text", priority=False)
        e._q.get_nowait()
        e.speak("same text", priority=True)
        self.assertFalse(
            e._q.empty(),
            "Priority speak must bypass dedup even when non-priority set the dedup timestamp.",
        )

    def test_priority_updates_dedup_timestamp(self):
        """After a priority speak, a non-priority repeat SHOULD be deduped."""
        e = _engine()
        e.speak("msg", priority=True)
        e._q.get_nowait()
        e.speak("msg", priority=False)   # same text, non-priority, within window
        self.assertTrue(
            e._q.empty(),
            "Non-priority duplicate after priority speak should still be deduped.",
        )


# ─────────────────────────────────────────────────────────────────────────────
# NAV_RATE_LIMIT behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestTTSNavRateLimit(unittest.TestCase):
    """Non-priority messages must be rate-limited to one per NAV_RATE_LIMIT seconds."""

    def test_rate_limit_drops_second_nav_message(self):
        e = _engine()
        e.speak("path clear", priority=False)
        e._q.get_nowait()
        # Different text — not dedup'd, but too soon
        e.speak("obstacle ahead", priority=False)
        self.assertTrue(
            e._q.empty(),
            "Second non-priority message within NAV_RATE_LIMIT should be dropped.",
        )

    def test_rate_limit_logs_debug_on_drop(self):
        e = _engine()
        e.speak("path clear", priority=False)
        e._q.get_nowait()
        with self.assertLogs("backend.tts", level="DEBUG") as cm:
            e.speak("obstacle ahead", priority=False)
        self.assertTrue(
            any("rate-limit" in line or "rate_limit" in line or "nav" in line.lower()
                for line in cm.output),
            "A DEBUG log mentioning rate-limit must be emitted.",
        )

    def test_rate_limit_allows_after_window(self):
        e = _engine()
        e.speak("path clear", priority=False)
        e._q.get_nowait()
        # Backdate nav time so the window has elapsed
        e._last_nav_time = time.time() - TTSEngine.NAV_RATE_LIMIT - 1.0
        # Also clear the dedup entry
        e._last.pop("obstacle ahead", None)
        e.speak("obstacle ahead", priority=False)
        self.assertFalse(
            e._q.empty(),
            "Non-priority message should be allowed after NAV_RATE_LIMIT has elapsed.",
        )

    def test_priority_ignores_nav_rate_limit(self):
        """Priority messages must not be blocked by NAV_RATE_LIMIT."""
        e = _engine()
        # Simulate a recent nav message
        e._last_nav_time = time.time()
        e.speak("urgent alert", priority=True)
        self.assertFalse(
            e._q.empty(),
            "Priority message must not be blocked by NAV_RATE_LIMIT.",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Queue management
# ─────────────────────────────────────────────────────────────────────────────

class TestTTSQueueManagement(unittest.TestCase):

    def test_priority_clears_queue_before_enqueue(self):
        e = _engine()
        # Fill queue with non-priority items
        for i in range(4):
            e._q.put_nowait(f"nav msg {i}")
        e.speak("urgent!", priority=True)
        # Queue should contain only the urgent message
        items = []
        while not e._q.empty():
            items.append(e._q.get_nowait())
        self.assertEqual(items, ["urgent!"], "Priority speak must clear old queue items.")

    def test_queue_full_logs_warning(self):
        e = _engine()
        # Fill the queue to capacity (maxsize=5) by bypassing speak() gating
        for i in range(5):
            e._q.put_nowait(f"filler {i}")
            e._last[f"filler {i}"] = 0.0   # old timestamp — won't trigger dedup

        # Use priority=True so it tries to clear then re-enqueue; but first we
        # patch put_nowait to raise Full after clearing, simulating a race.
        # Simpler approach: call speak() with a text whose dedup is expired, but
        # the queue is full because a concurrent call filled it.
        # The real test: enqueue directly, then force a Full on the next put.
        import queue as q_mod
        original_put = e._q.put_nowait

        def raising_put(item):
            raise q_mod.Full()

        e._q.put_nowait = raising_put

        # Use priority=True so it skips dedup/rate-limit and reaches put_nowait
        with self.assertLogs("backend.tts", level="WARNING") as cm:
            e.speak("overflow text", priority=True)

        self.assertTrue(
            any("full" in line.lower() or "dropped" in line.lower() for line in cm.output),
            "A WARNING must be logged when the queue is full and a message is dropped.",
        )

    def test_empty_text_is_ignored(self):
        e = _engine()
        e.speak("", priority=True)
        e.speak("   " * 0, priority=False)
        self.assertTrue(e._q.empty(), "Empty text must never be queued.")


# ─────────────────────────────────────────────────────────────────────────────
# Worker — broadcast integration
# ─────────────────────────────────────────────────────────────────────────────

class TestTTSWorkerBroadcast(unittest.TestCase):

    def test_worker_calls_broadcast_with_speak_type(self):
        """The worker must send {"type": "speak", "text": ...} via the broadcast fn."""
        e = TTSEngine()
        broadcast_calls = []

        async def fake_broadcast(data):
            broadcast_calls.append(data)

        # Create a dedicated event loop for this test
        loop = asyncio.new_event_loop()

        e._broadcast = fake_broadcast
        e._loop = loop

        # Put an item directly into the queue (bypass speak() gating)
        e._q.put_nowait("test message")

        import threading

        # Run the worker in a thread; it will schedule the coroutine on our loop
        t = threading.Thread(target=e._worker, daemon=True)
        t.start()

        # Give the worker thread time to dequeue and schedule the coroutine
        time.sleep(0.1)

        # Drain all pending coroutines that were scheduled via run_coroutine_threadsafe
        async def drain():
            await asyncio.sleep(0)   # one event-loop tick

        loop.run_until_complete(drain())
        loop.close()

        self.assertTrue(
            any(c == {"type": "speak", "text": "test message"} for c in broadcast_calls),
            f"Broadcast must be called with speak payload; got: {broadcast_calls}",
        )


if __name__ == "__main__":
    unittest.main()
