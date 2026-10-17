"""Spoken commands the assistant handles itself (voice_assistant.py:
_parse_backend_request) and the shell tool's refusal to restart the service
it runs in (_run_shell_tool).

    .venv/bin/python -m pytest tests
"""
import asyncio
import json
import logging
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import voice_assistant as va  # noqa: E402


@pytest.mark.parametrize("text, want", [
    ("Can you switch to clog please?", "claude"),
    ("switch to claude", "claude"),
    ("Switch to Claude Code.", "claude"),
    ("please swap over to cloud", "claude"),
    ("go back to the local model", "local"),
    ("switch to qwen", "local"),
    ("change to the local one", "local"),
    ("use the harness", "dsh"),
    ("switch to deep seek", "dsh"),
    ("switch to auto", "auto"),
    ("use claude for this", "claude"),
    # not commands
    ("use the local time zone", None),
    ("claude is good at this sort of thing", None),
    ("the cloud looks heavy today", None),
    ("switch to fable", None),                      # a Claude model, not a backend
    ("set the effort to high", None),
    ("switch to claude and then explain how the encoder works in detail", None),  # discussion length
    ("what time is it", None),
])
def test_parse_backend_request(text, want):
    assert va._parse_backend_request(text) == want


def test_model_and_backend_commands_do_not_overlap():
    assert va._parse_claude_request("switch to fable") == ("fable", None)
    assert va._parse_backend_request("switch to fable") is None
    assert va._parse_backend_request("switch to claude") == "claude"
    assert va._parse_claude_request("switch to claude") == (None, None)


class Host(va.VoiceAssistant):
    def __init__(self):
        self.logger = logging.getLogger("test-voice-commands")
        self._tool_process = None

    def _sanitize_tool_output(self, out):
        return out


def test_shell_tool_defers_a_command_that_restarts_the_assistant(monkeypatch):
    launched = []

    def fake_popen(args, **kw):
        launched.append(args)
        return object()
    monkeypatch.setattr(va.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(va, "SELF_RESTART_DELAY", 6)
    host = Host()
    for cmd in ("which voice-llm && voice-llm claude", "systemctl --user restart voice-assistant.service",
                "systemctl --user stop voice-assistant"):
        out = host._run_shell_tool(cmd)
        assert out.startswith("(deferred:") and "switch to Claude" in out
    assert len(launched) == 3
    assert launched[0][:5] == ["systemd-run", "--user", "--scope", "--collect", "--quiet"]
    assert launched[0][-1] == "sleep 6; which voice-llm && voice-llm claude"


@pytest.mark.parametrize("cmd, restarts", [
    ("systemctl --user restart voice-assistant", True),
    ("systemctl restart --user voice-assistant.service", True),
    ("systemctl --user --no-block restart voice-assistant", True),
    ("systemctl --user disable --now voice-assistant", True),
    ("pkill -f voice-assistant", True),
    ("pkill -f voice_assistant.py", True),
    ("kill $(pgrep -f voice_assistant)", True),
    ("voice-llm network", True),
    # looking is not restarting
    ("systemctl --user status voice-assistant", False),
    ("cat ~/.config/voice-assistant/env", False),
    ("pgrep -f voice_assistant", False),
    ("systemctl --user list-units | grep voice-assistant", False),
    ("systemctl --user restart voice-assistant-jabra", False),
    ("kill -USR1 $(cat ~/.local/state/voice-assistant/voice-assistant.pid)", False),
    ("systemctl --user status voice-assistant; systemctl --user restart qwen38", False),
])
def test_self_restart_pattern(cmd, restarts):
    assert bool(va._SELF_RESTART_RE.search(cmd)) == restarts


def test_shell_tool_runs_ordinary_commands():
    host = Host()
    assert host._run_shell_tool("echo hello").strip() == "hello"
    assert "[exit code 3]" in host._run_shell_tool("exit 3")


# --- the status file the panels read ---------------------------------------------

class StatusHost(va.VoiceAssistant):
    def __init__(self, state_dir, backend):
        self.logger = logging.getLogger("test-status")
        self.state_dir = state_dir
        self.backend = backend
        self._status_state = "off"


def test_status_payload_names_the_backend_and_its_model(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(va, "CLAUDE_MODEL", "fable")
    monkeypatch.setattr(va, "CLAUDE_EFFORT", "xhigh")
    monkeypatch.setattr(va, "LOCAL_LLM_MODEL", "qwen3.8-27b")
    for backend in ("dsh", "local", "claude"):
        host = StatusHost(tmp_path, backend)
        host._set_status("ready")
        d = json.loads((tmp_path / "status").read_text())
        assert d["class"] == ["ready", backend] and d["backend"] == backend
        assert d["model"] == "fable" and d["effort"] == "xhigh", "Claude's settings, for the Claude menu"
        assert d["local_model"] == "Qwen3.8 27B"
    host._set_status("thinking", backend="claude")
    d = json.loads((tmp_path / "status").read_text())
    assert d["backend"] == "claude" and d["class"] == ["thinking", "claude"]
    assert va._local_model_label("qwen3.8-27b-opt") == "Qwen3.8 27B Opt"
    assert va._local_model_label("") == ""


# --- a spoken model/effort change --------------------------------------------------

class SpokenHost(va.VoiceAssistant):
    """Just enough of the assistant for _process_audio to take a spoken command."""
    def __init__(self, state_dir, heard):
        self.logger = logging.getLogger("test-spoken")
        self.state_dir = state_dir
        self.backend = "claude"
        self._status_state = "thinking"
        self.is_active = True
        self.is_processing = True        # the audio loop sets it before the turn starts
        self._abort_event = threading.Event()
        self._claude = None
        self._claude_pending_model = self._claude_pending_effort = None
        self.capture = self.vad = SimpleNamespace(
            mute=lambda *a: None, flush=lambda: None, reset=lambda: None)
        self.heard, self.said = heard, []

    def _transcribe(self, audio): return self.heard
    def _save_turn(self, audio, transcript): pass
    def _say(self, text): self.said.append(text)
    def _append_transcript(self, role, text): pass
    def _notify(self, *a, **k): pass
    def _router_finish(self): pass
    def _play_chime_async(self, name): pass


def test_a_spoken_effort_change_applies_at_once(tmp_path, monkeypatch):
    monkeypatch.setattr(va, "CLAUDE_MODEL", "opus")
    monkeypatch.setattr(va, "CLAUDE_EFFORT", "max")
    monkeypatch.setattr(va, "TTS_TAIL_GATE", 0)
    written = []
    monkeypatch.setattr(va, "_write_env_setting", lambda key, value: written.append((key, value)))
    host = SpokenHost(tmp_path, "Set the effort to high.")
    asyncio.run(host._process_audio([]))
    # It used to be deferred to the end of the next Claude reply -- the turn's
    # own is_processing looked like a reply in flight -- while the user heard
    # "Already on opus at high effort." and the panel still said max.
    assert host.said == ["Switched to high effort."]
    assert va.CLAUDE_EFFORT == "high" and written == [("VOICE_ASSISTANT_EFFORT", "high")]
    assert host._claude_pending_effort is None
    assert json.loads((tmp_path / "status").read_text())["effort"] == "high", "what the panel draws"
    # The menus and the CLI can land mid-reply; those still wait for its end.
    host.is_processing = True
    res = host.set_claude_setting("fable")
    assert res["deferred"] and va.CLAUDE_MODEL == "opus" and host._claude_pending_model == "fable"
