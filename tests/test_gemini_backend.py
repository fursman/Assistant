"""The Gemini backend (voice_assistant.py: GeminiSession, _gemini_run) against
a fake `agy` that speaks the Antigravity CLI's stream-json protocol: text
deltas and tool steps per turn, a result per turn, resume by conversation id.

    .venv/bin/python -m pytest tests/test_gemini_backend.py
"""
import json
import logging
import os
import stat
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import voice_assistant as va  # noqa: E402

va.VOICE_UNLOCK = False

FAKE_AGY = r'''#!/usr/bin/env python3
"""A stand-in for `agy --input-format stream-json --output-format stream-json`.
One line in per turn; init once, then step_updates and a result per turn.
`--conversation gone` fails before init, the way a lost conversation does."""
import json, os, sys, time
args = sys.argv[1:]
assert args[-1] == "-p=", args          # the prompt flag goes last, empty
conv = args[args.index("--conversation") + 1] if "--conversation" in args else None
if conv == "gone":
    sys.stderr.write("Error: conversation not found\n")
    sys.exit(1)
cid = conv or "conv-%d" % os.getpid()
def emit(o):
    sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
def step(idx, **k):
    emit({"event": "step_update", "step_update": dict(conversation_id=cid, step_index=idx, **k)})
emit({"event": "init", "conversation_id": cid,
      "init": {"cwd": os.getcwd(), "tools": ["run_command"], "permission_mode": "always-proceed",
               "model": next((args[i + 1] for i, a in enumerate(args) if a == "--model"), None)}})
turns = idx = 0
for line in sys.stdin:
    text = json.loads(line)["message"]["content"]
    turns += 1
    step(idx, state="DONE", step_type="user_input"); idx += 1
    if "slow" in text:
        time.sleep(10)
    if "run" in text:
        info = {"name": "run_command", "parameters": {"CommandLine": "date +%Y"}}
        step(idx, state="ACTIVE", step_type="tool", tool_name="run_command", tool_info=info)
        step(idx, state="DONE", step_type="tool", tool_name="run_command",
             tool_info=dict(info, output="2026\n")); idx += 1
    if "fail" in text:
        emit({"event": "result", "result": {"conversation_id": cid, "status": "ERROR",
                                             "error": "quota exhausted", "num_turns": turns}})
        continue
    reply = "Echo: " + text
    step(idx, state="ACTIVE", step_type="agent_response", text_delta=reply)
    step(idx, state="DONE", step_type="agent_response", text_delta="\n"); idx += 1
    emit({"event": "result", "result": {"conversation_id": cid, "status": "SUCCESS",
                                         "response": reply + "\n", "num_turns": turns,
                                         "usage": {"total_tokens": 100 * turns}}})
'''


class Ledger:
    def __init__(self):
        self.remembered = []

    def recap_for(self, backend):
        return ""


class GeminiHost(va.VoiceAssistant):
    """Just enough of the assistant for a Gemini turn: no audio, no speech."""
    def __init__(self, state_dir):
        self.logger = logging.getLogger("test-gemini")
        self.state_dir = state_dir
        self.is_active = True
        self.backend = "gemini"
        self._claude = self._claude_process = self._dsh = None   # _abort_inflight looks
        self._claude_turn_active = False
        self._local_stream = self._tool_process = self._sentence_queue = None
        self.player = SimpleNamespace(abort=lambda: None)
        self._gemini = None
        self._gemini_available = True
        self._gemini_turn_active = False
        self._gemini_restart_pending = False
        self._gemini_error = ""
        self._gemini_session_id = None
        self._gemini_seen_steps = set()
        self._assistant_text = ""
        self.tools = []
        self.flushed = 0
        self._ledger_obj = Ledger()

    def _ledger(self):
        return self._ledger_obj

    def _flush_sentences(self, final=False):
        self.flushed += 1

    def _flush_pending_tool_notice(self):
        pass

    # The rest of what _abort_inflight touches: nothing here to stop.
    def _kill_tool_process(self, *a, **k): pass
    def _play_chime_async(self, *a, **k): pass
    def _set_status(self, *a, **k): pass
    def _notify(self, *a, **k): pass

    def _notify_tool_use(self, name, input_json_str):
        self.tools.append((name, json.loads(input_json_str)))


@pytest.fixture
def agy(tmp_path, monkeypatch):
    """A fake agy first on PATH, and the backend's files under tmp_path."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "agy"
    fake.write_text(FAKE_AGY)
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setattr(va, "GEMINI_HOME", tmp_path / "gemini")
    monkeypatch.setattr(va, "GEMINI_SESSION_FILE", tmp_path / "gemini_session_id")
    monkeypatch.setattr(va, "GEMINI_MODEL", "")
    monkeypatch.setattr(va, "GEMINI_EFFORT", "")
    monkeypatch.setattr(va, "_refresh_session_env", lambda: None)
    yield fake


def test_a_turn_streams_text_and_tools_and_keeps_the_conversation(tmp_path, agy):
    host = GeminiHost(tmp_path)
    abort = threading.Event()
    try:
        assert host._stream_gemini("please run date", abort) == "ok"
        assert host._assistant_text == "Echo: please run date\n" and host.flushed == 2
        # The tool step is shown once, under Claude's names, so the popup and
        # transcript lines read the same whoever answers.
        assert host.tools == [("Bash", {"command": "date +%Y"})]
        cid = host._gemini_session_id
        assert cid and cid.startswith("conv-") and (tmp_path / "gemini_session_id").read_text() == cid
        # The persona is where the CLI reads it: AGENTS.md in its working directory.
        assert (tmp_path / "gemini" / "AGENTS.md").read_text() == va.GEMINI_VOICE_PROMPT
        assert "You are Gemini" in va.GEMINI_VOICE_PROMPT

        # A second turn rides the same process.
        pid = host._gemini.proc.pid
        host._assistant_text, host._gemini_seen_steps = "", set()
        assert host._stream_gemini("and again", abort) == "ok"
        assert host._assistant_text == "Echo: and again\n" and host._gemini.proc.pid == pid
        assert host._gemini_session_id == cid

        # A failed result is an error with its reason, and the process lives on.
        host._assistant_text = ""
        assert host._stream_gemini("fail this", abort) == "error"
        assert host._gemini_error == "quota exhausted" and host._gemini.alive
    finally:
        if host._gemini is not None:
            host._gemini.stop()


def test_a_conversation_that_cannot_be_resumed_starts_fresh_once(tmp_path, agy):
    host = GeminiHost(tmp_path)
    host._gemini_session_id = "gone"
    (tmp_path / "gemini_session_id").write_text("gone")
    try:
        assert host._stream_gemini("hello", threading.Event()) == "ok"
        assert host._assistant_text == "Echo: hello\n"
        assert host._gemini_session_id != "gone" and host._gemini.conversation_id == host._gemini_session_id
        assert (tmp_path / "gemini_session_id").read_text() == host._gemini_session_id
    finally:
        host._gemini.stop()


def test_abort_kills_the_process_and_the_next_turn_resumes_by_id(tmp_path, agy):
    host = GeminiHost(tmp_path)
    abort = threading.Event()
    try:
        assert host._stream_gemini("hello", abort) == "ok"
        cid = host._gemini_session_id
        done = {}

        def turn():
            done["status"] = host._stream_gemini("slow one", abort)
        t = threading.Thread(target=turn)
        t.start()
        # The abort is what the toggle key does: the event, then the kill.
        import time
        time.sleep(0.5)
        abort.set()
        host._abort_inflight()
        t.join(timeout=5)
        assert done["status"] == "aborted" and not host._gemini.alive
        abort.clear()
        host._assistant_text = ""
        assert host._stream_gemini("back", abort) == "ok"
        assert host._gemini_session_id == cid, "resumed, not restarted"
    finally:
        host._gemini.stop()


def test_new_conversation_mid_turn_is_deferred_to_its_end(tmp_path, agy):
    host = GeminiHost(tmp_path)
    abort = threading.Event()
    try:
        assert host._stream_gemini("hello", abort) == "ok"
        first = host._gemini_session_id
        # What _clear_session does when a turn is in flight.
        host._gemini_turn_active = True
        host._gemini_restart_pending = True
        host._gemini_turn_active = False
        host._assistant_text = ""
        assert host._stream_gemini("next", abort) == "ok"      # applies the restart after
        assert host._gemini.conversation_id is None or host._gemini.conversation_id != first
        host._assistant_text = ""
        assert host._stream_gemini("fresh", abort) == "ok"
        assert host._gemini_session_id != first
    finally:
        host._gemini.stop()


@pytest.mark.parametrize("name, params, want", [
    ("run_command", {"CommandLine": "ls -la"}, ("Bash", {"command": "ls -la"})),
    ("view_file", {"AbsolutePath": "/etc/hosts"}, ("Read", {"file_path": "/etc/hosts"})),
    ("search_web", {"query": "weather"}, ("WebSearch", {"query": "weather"})),
    ("read_url_content", {"Url": "https://x.y"}, ("WebFetch", {"url": "https://x.y"})),
    ("generate_image", {"prompt": "a cat"}, ("generate_image", {"prompt": "a cat"})),
    ("run_command", None, ("Bash", {"command": ""})),
])
def test_antigravity_tools_are_shown_under_claudes_names(name, params, want):
    assert va._gemini_tool(name, params) == want
