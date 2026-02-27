"""
tests/test_main_ws_broadcast.py

Regression tests for main.py WebSocket handler broadcast correctness.

Bugs fixed (Phase 5 audit):
  Bug 8: WS 'ask' result was sent only to requesting socket (websocket.send_json)
          — should be broadcast() so all clients get the answer.
  Bug 9: 'scene_diff', 'find_object' result, 'find_start_capture', 'find_capture'
          used websocket.send_json — should be broadcast() so demo-mode
          phone + laptop both see the same state.

These tests verify the main.py action handlers call broadcast() for payloads
that all connected clients must receive.  We do this by inspecting the source
code directly (AST check) and by behavioural mock tests where feasible.

AST-based tests are the most reliable here because ws_endpoint is an async
generator (WebSocket receive loop) that is difficult to unit-test without
running a full ASGI stack.  The AST approach verifies the call site rather
than the runtime path, but it is deterministic and immune to test isolation
issues.
"""
import ast
import inspect
import textwrap
import unittest
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Helpers — parse main.py once for all tests
# ─────────────────────────────────────────────────────────────────────────────

_MAIN_PY = Path(__file__).parent.parent / "backend" / "main.py"


def _load_main_source() -> str:
    return _MAIN_PY.read_text(encoding="utf-8")


def _find_send_json_lines(source: str) -> list[int]:
    """
    Return line numbers of all 'websocket.send_json(...)' calls in main.py.
    These are candidate single-socket sends that may need to be broadcast().
    """
    tree = ast.parse(source)
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if (isinstance(func, ast.Attribute)
                    and func.attr == "send_json"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "websocket"):
                lines.append(node.lineno)
    return lines


def _find_broadcast_calls(source: str) -> list[int]:
    """Return line numbers of all 'await broadcast(...)' calls in main.py."""
    tree = ast.parse(source)
    lines = []
    for node in ast.walk(tree):
        # Match: await broadcast(...)
        if (isinstance(node, ast.Await)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "broadcast"):
            lines.append(node.lineno)
    return lines


def _get_action_block_source(source: str, action_name: str) -> str:
    """
    Extract the elif/if block for a specific action string from the WS handler.
    This is a best-effort text search — used only to locate relevant context.
    """
    lines = source.splitlines()
    result = []
    in_block = False
    for line in lines:
        stripped = line.strip()
        if f'action == "{action_name}"' in stripped or f"action == '{action_name}'" in stripped:
            in_block = True
        if in_block:
            result.append(line)
            # Stop at the next elif/else at the same or higher indent, or a blank line
            if result and len(result) > 3 and (
                stripped.startswith("elif ") or stripped.startswith("else:")
            ) and stripped != result[0].strip():
                result.pop()
                break
    return "\n".join(result)


# ─────────────────────────────────────────────────────────────────────────────
# AST-based broadcast correctness tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMainBroadcastCorrectness(unittest.TestCase):
    """
    Verify that the WebSocket action handlers in main.py use broadcast()
    for payloads that all connected clients must receive, not websocket.send_json().
    """

    @classmethod
    def setUpClass(cls):
        cls.source = _load_main_source()

    def _assert_action_uses_broadcast_not_send_json(self, action_name: str):
        """
        For a given WS action name, assert that the handler uses broadcast()
        and NOT websocket.send_json() for its primary result payload.

        This is done by finding the action block, then checking that within it
        broadcast() is called for the non-trivial payload.
        """
        source = self.source
        # Find all broadcast() calls and websocket.send_json() calls
        broadcast_lines  = _find_broadcast_calls(source)
        send_json_lines  = _find_send_json_lines(source)

        # Locate the line range of this action's handler block
        lines = source.splitlines()
        action_start = None
        for i, line in enumerate(lines, start=1):
            if f'action == "{action_name}"' in line or f"action == '{action_name}'" in line:
                action_start = i
                break

        self.assertIsNotNone(
            action_start,
            f"Could not find 'action == \"{action_name}\"' in main.py — handler may have been removed",
        )

        # Find the next action block or end of function
        action_end = len(lines) + 1
        for i in range(action_start + 1, len(lines) + 1):
            line = lines[i - 1].strip()
            if line.startswith("elif action") or line.startswith("except WebSocketDisconnect"):
                action_end = i
                break

        # Check: broadcast must be called in this block
        block_broadcasts = [l for l in broadcast_lines if action_start <= l <= action_end]
        # Check: websocket.send_json must NOT be called in this block for result payloads
        # (system/ui feedback to requesting socket only is acceptable via send_json
        #  — but answer/find_prompt/reading results must use broadcast)
        block_send_json = [l for l in send_json_lines if action_start <= l <= action_end]

        # Build context for failure messages
        block_text = "\n".join(lines[action_start - 1 : min(action_end, action_start + 30)])

        self.assertGreater(
            len(block_broadcasts), 0,
            f"Action '{action_name}' handler must call broadcast() for result payload,\n"
            f"but no broadcast() found in lines {action_start}–{action_end}.\n"
            f"Handler block:\n{block_text}",
        )
        self.assertEqual(
            len(block_send_json), 0,
            f"Action '{action_name}' handler must NOT call websocket.send_json() for result,\n"
            f"but found send_json at lines: {block_send_json}.\n"
            f"Fix: change to await broadcast(...) so all clients receive the update.\n"
            f"Handler block:\n{block_text}",
        )

    def test_ask_action_uses_broadcast(self):
        """
        Regression for Bug 8: 'ask' WS action must broadcast() the answer dict,
        not websocket.send_json() — so all connected clients (phone + laptop) receive it.
        """
        self._assert_action_uses_broadcast_not_send_json("ask")

    def test_scene_diff_uses_broadcast(self):
        """
        Regression for Bug 9a: 'scene_diff' action must broadcast() the answer,
        not websocket.send_json() — scene change info is globally relevant.
        """
        self._assert_action_uses_broadcast_not_send_json("scene_diff")

    def test_find_start_capture_uses_broadcast(self):
        """
        Regression for Bug 9b: 'find_start_capture' must broadcast() the
        find_prompt state so both clients show the capture confirmation UI.
        """
        self._assert_action_uses_broadcast_not_send_json("find_start_capture")

    def test_find_capture_uses_broadcast(self):
        """
        Regression for Bug 9c: 'find_capture' must broadcast() the find_prompt
        'captured' state so both clients advance their UI.
        """
        self._assert_action_uses_broadcast_not_send_json("find_capture")


class TestMainBroadcastCount(unittest.TestCase):
    """
    High-level: count websocket.send_json() calls in main.py.
    After Bugs 8/9 fixes, only a small set of legitimate single-socket sends
    should remain (stt_mute, stt_unmute, set_camera, find_cancel, camera-not-ready
    fallbacks, find_object camera-not-ready, and the init message).
    """

    @classmethod
    def setUpClass(cls):
        cls.source = _load_main_source()
        cls.send_json_lines = _find_send_json_lines(cls.source)

    def test_single_socket_sends_are_few(self):
        """
        After broadcast fixes, only a small set of legitimate single-socket sends
        should remain in main.py:
          - init message (L353) — only the connecting client needs it
          - camera-not-ready fallbacks (find_object, find_question paths) — UI feedback
          - set_camera confirmation — only the requesting client changed camera
          - find_cancel confirmation — only the requesting client cancelled
          - stt_mute / stt_unmute responses — mic state for requesting client only

        Total: 7 legitimate single-socket sends.  If this count grows, review
        each new call to ensure it's intentionally single-socket.
        """
        n = len(self.send_json_lines)
        self.assertLessEqual(
            n, 7,
            f"Found {n} websocket.send_json() calls at lines {self.send_json_lines}.\n"
            f"Expected ≤7 legitimate single-socket sends (init, 2× camera-not-ready, "
            f"set_camera, find_cancel, stt_mute, stt_unmute).\n"
            f"Review each and change to broadcast() if all clients should receive it.",
        )

    def test_no_answer_payload_via_single_socket(self):
        """
        No 'answer' type dict should be sent via websocket.send_json() —
        answers must always go to all clients via broadcast().
        """
        source = self.source
        lines = source.splitlines()
        for lineno in self.send_json_lines:
            # Check lines near the send_json call for type="answer"
            context = "\n".join(lines[max(0, lineno - 5): lineno + 3])
            self.assertNotIn(
                '"answer"',
                context,
                f"Found 'answer' payload in a websocket.send_json() call near line {lineno}.\n"
                f"All 'answer' payloads must use await broadcast() so all clients receive them.\n"
                f"Context:\n{context}",
            )


if __name__ == "__main__":
    unittest.main()
