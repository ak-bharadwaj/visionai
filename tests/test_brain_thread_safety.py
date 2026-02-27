"""
test_brain_thread_safety.py

Regression tests for the Brain history generation-counter fix.

Bug: clear_history() called from the asyncio thread while answer() is running
an LLM call in an executor thread could cause the stale LLM answer to be
appended AFTER the history was cleared, polluting the next ASK session.

Fix: _history_gen counter is bumped on clear_history(); answer() captures the
gen before the slow call and skips the append if the gen has changed.
"""

import threading
import time
import pytest

# We do NOT mock Ollama — all slow routes will fall through to fallback or
# return immediately.  We only test the history guard logic itself.
from backend.brain import Brain


class TestBrainHistoryGenCounter:
    """Brain._history_gen increments on each clear_history() call."""

    def test_initial_gen_is_zero(self):
        b = Brain()
        assert b._history_gen == 0

    def test_clear_history_increments_gen(self):
        b = Brain()
        b.clear_history()
        assert b._history_gen == 1

    def test_multiple_clears_increment_gen_each_time(self):
        b = Brain()
        for i in range(5):
            b.clear_history()
        assert b._history_gen == 5

    def test_history_is_empty_after_clear(self):
        b = Brain()
        # Seed history manually (bypassing answer())
        with b._history_lock:
            b._history.append(("q1", "a1"))
        assert len(b._history) == 1
        b.clear_history()
        assert len(b._history) == 0


class TestBrainHistoryAppendHelper:
    """Brain._history_append only writes when gen matches."""

    def test_append_succeeds_when_gen_matches(self):
        b = Brain()
        with b._history_lock:
            gen = b._history_gen
        b._history_append("q", "a", gen)
        assert len(b._history) == 1
        assert b._history[0] == ("q", "a")

    def test_append_skipped_when_gen_mismatch(self):
        b = Brain()
        captured_gen = b._history_gen   # gen = 0
        b.clear_history()               # gen becomes 1
        b._history_append("q", "a", captured_gen)   # captured=0, current=1 → skip
        assert len(b._history) == 0

    def test_append_skipped_after_multiple_clears(self):
        b = Brain()
        captured_gen = b._history_gen
        b.clear_history()
        b.clear_history()
        b.clear_history()
        b._history_append("q", "a", captured_gen)
        assert len(b._history) == 0

    def test_append_succeeds_on_fresh_gen_after_clear(self):
        b = Brain()
        b.clear_history()
        # Capture gen AFTER the clear
        with b._history_lock:
            gen = b._history_gen
        b._history_append("q2", "a2", gen)
        assert len(b._history) == 1
        assert b._history[0] == ("q2", "a2")


class TestBrainClearRaceSimulation:
    """
    Simulate the real race: executor thread holds gen=0, then the asyncio
    thread calls clear_history(), then executor tries to append with gen=0.
    The append should be silently skipped.
    """

    def test_race_clear_fires_before_append(self):
        b = Brain()
        # Step 1: executor captures gen before slow call
        with b._history_lock:
            gen_before_llm = b._history_gen   # 0

        # Step 2 (simulated slow LLM I/O — asyncio fires clear_history mid-flight)
        b.clear_history()   # gen → 1

        # Step 3: executor returns and tries to append with stale gen
        b._history_append("stale question", "stale answer", gen_before_llm)

        # History must remain empty — the stale answer must NOT be stored
        assert len(b._history) == 0

    def test_race_clear_does_not_fire(self):
        """Control: if clear never fires, the append should succeed normally."""
        b = Brain()
        with b._history_lock:
            gen = b._history_gen
        # No clear_history() call — normal path
        b._history_append("q", "a", gen)
        assert len(b._history) == 1

    def test_concurrent_clear_and_append_thread_safety(self):
        """
        Hammer test: 100 iterations of concurrent clear + append, verifying
        history never exceeds the maxlen and the lock is never deadlocked.
        """
        b = Brain()
        errors = []

        def worker():
            for _ in range(100):
                with b._history_lock:
                    gen = b._history_gen
                b._history_append("q", "a", gen)
                b.clear_history()

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # All threads should have finished (not deadlocked)
        alive = [t for t in threads if t.is_alive()]
        assert not alive, f"{len(alive)} thread(s) still alive — possible deadlock"
        # History must have consistent state (not crashed)
        assert len(b._history) <= 4   # maxlen=MAX_HISTORY=4


class TestBrainGetHistoryThreadSafe:
    """get_history() should still work correctly after the lock is added."""

    def test_get_history_returns_list_of_dicts(self):
        b = Brain()
        with b._history_lock:
            b._history.append(("What color is it?", "It is red."))
            b._history.append(("Any people?", "One person nearby."))
        result = b.get_history()
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == {"question": "What color is it?", "answer": "It is red."}
        assert result[1] == {"question": "Any people?", "answer": "One person nearby."}

    def test_get_history_empty_after_clear(self):
        b = Brain()
        with b._history_lock:
            b._history.append(("q", "a"))
        b.clear_history()
        assert b.get_history() == []
