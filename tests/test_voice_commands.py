"""Spoken commands the assistant handles itself (voice_assistant.py:
_parse_backend_request) and the shell tool's refusal to restart the service
it runs in (_run_shell_tool).

    .venv/bin/python -m pytest tests
"""
import logging
import subprocess
import sys
from pathlib import Path

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


def test_shell_tool_runs_ordinary_commands():
    host = Host()
    assert host._run_shell_tool("echo hello").strip() == "hello"
    assert "[exit code 3]" in host._run_shell_tool("exit 3")
