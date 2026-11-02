"""Where speech runs, switched at run time (voice_assistant.py: set_speech_mode,
_use_remote_speech, _parse_speech_request; `assistant --speech`): "network"
hands recognition and synthesis to the LAN servers for their tiers, "local"
keeps both on this machine. No real servers: the boards and this machine's
engines are stand-ins.

    .venv/bin/python -m pytest tests
"""
import asyncio
import contextlib
import importlib.machinery
import importlib.util
import io
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import remote_speech  # noqa: E402
import voice_assistant as va  # noqa: E402

va.VOICE_UNLOCK = False

STT = "192.168.1.200:8772,192.168.1.201:8772"
TTS = "http://192.168.1.200:8771,http://192.168.1.201:8771"


class SpeechHost(va.VoiceAssistant):
    """The speech choice alone: no audio, no models, no servers."""
    def __init__(self, state_dir, mode="network"):
        self.logger = logging.getLogger("test-speech-mode")
        self.state_dir = state_dir
        self.backend, self.llm_tier, self.backend_pref = "claude", "local", "auto"
        self._status_state = "ready"
        self.is_active = False
        self.is_processing = False
        self._dsh = None
        self._dsh_available = False
        self._claude_available = True
        self.voice_only, self._voiceprint, self._session_id = False, None, "s"
        self.speech_mode = mode
        self.notes, self.transcript = [], []

    def _append_transcript(self, role, text): self.transcript.append((role, text))
    def _notify(self, text, **k): self.notes.append((text, k))


@pytest.fixture
def boards(monkeypatch):
    """Two boards configured for the network and Claude tiers; what is written."""
    written = []
    monkeypatch.setattr(va, "REMOTE_STT", STT)
    monkeypatch.setattr(va, "REMOTE_TTS", TTS)
    monkeypatch.setattr(va, "REMOTE_SPEECH_TIERS", {"network", "claude"})
    monkeypatch.setattr(va, "_write_env_setting", lambda key, value: written.append((key, value)))
    monkeypatch.setattr(va, "_local_llm_health", lambda timeout=1.0: "down")
    monkeypatch.setattr(va, "NETWORK_LLM_URL", "")
    return written


@pytest.fixture
def no_boards(boards, monkeypatch):
    monkeypatch.setattr(va, "REMOTE_STT", "")
    monkeypatch.setattr(va, "REMOTE_TTS", "")
    return boards


# --- the switch ---------------------------------------------------------------

def test_toggle_flips_and_persists(tmp_path, boards):
    host = SpeechHost(tmp_path)
    res = host.set_speech_mode("toggle")
    assert res["ok"] and res["speech"] == "local" and res["changed"]
    assert res["label"] == "Speech: local"
    assert host.set_speech_mode("toggle")["speech"] == "network"
    assert boards == [("VOICE_ASSISTANT_SPEECH", "local"), ("VOICE_ASSISTANT_SPEECH", "network")]


def test_local_and_network_are_set_outright(tmp_path, boards):
    host = SpeechHost(tmp_path)
    res = host.set_speech_mode("network")
    assert res["speech"] == "network" and not res["changed"]
    assert host.set_speech_mode("local")["speech"] == "local"
    assert host.set_speech_mode("LOCAL")["speech"] == "local"
    assert not host.set_speech_mode("maybe")["ok"]
    assert host.speech_mode == "local"


def test_a_switch_notifies_logs_and_shows(tmp_path, boards, caplog):
    host = SpeechHost(tmp_path)
    with caplog.at_level(logging.INFO):
        host.set_speech_mode("local")
    text, k = host.notes[-1]
    assert text == "Speech: local"
    assert k["slot"] == "speech" and k["transient"], "its own slot, gone in a few seconds"
    assert host.transcript == [("system", "Speech: local")]
    assert "Speech: local (was network)" in caplog.text
    assert json.loads((tmp_path / "status").read_text())["speech"] == "local"
    host.set_speech_mode("network")
    assert host.notes[-1][0] == "Speech: network"
    assert json.loads((tmp_path / "status").read_text())["speech"] == "network"


def test_without_boards_network_is_not_possible(tmp_path, no_boards):
    host = SpeechHost(tmp_path, mode="local")
    for value in ("network", "toggle"):
        res = host.set_speech_mode(value)
        assert res["ok"] and res["speech"] == "local" and not res["changed"]
        assert res["label"] == "Speech: local (no network speech configured)"
        assert host.notes[-1][0] == "Speech: local (no network speech configured)"
    assert host.speech_mode == "local"
    assert no_boards == [], "nothing written, so boards configured later are used"


def test_the_env_setting_is_rewritten_in_place(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    env = tmp_path / ".config/voice-assistant/env"
    env.parent.mkdir(parents=True)
    env.write_text("VOICE_ASSISTANT_REMOTE_STT=a:1\nVOICE_ASSISTANT_SPEECH=network\n")
    monkeypatch.setattr(va, "REMOTE_STT", "a:1")
    monkeypatch.setattr(va, "REMOTE_TTS", "")
    host = SpeechHost(tmp_path)
    host.set_speech_mode("toggle")
    assert env.read_text() == "VOICE_ASSISTANT_REMOTE_STT=a:1\nVOICE_ASSISTANT_SPEECH=local\n"


def test_the_control_socket_switches_and_reports(tmp_path, boards):
    host = SpeechHost(tmp_path)
    res = asyncio.run(host._control_command({"cmd": "speech", "value": "toggle"}))
    assert res["ok"] and res["speech"] == "local" and res["label"] == "Speech: local"
    r = asyncio.run(host._control_command({"cmd": "status"}))
    assert r["speech"] == "local"
    host.set_speech_mode("network")
    r = asyncio.run(host._control_command({"cmd": "status"}))
    assert r["speech"] == "network"
    assert r["speech_servers"] == {"stt": STT.split(","), "tts": TTS.split(",")}
    assert r["speech_tiers"] == ["claude", "network"]


# --- the wrappers follow it at once -------------------------------------------

class Up:
    """A health check that always answers, with no thread behind it."""
    def __init__(self, name, probe): self.name = name
    def up(self): return True
    def mark_down(self, why): pass


class Board:
    """A recogniser and a synthesiser on a board, recording what it did."""
    log = []

    def __init__(self, spec, *a, **k):
        self.host = self.name = spec

    def begin(self): self.log.append(("begin", self.host))
    def feed(self, chunk, rate): pass
    def partial(self): return ""
    def finish(self): return "board heard"
    def batch(self, audio): return "board batch"
    def close(self): pass
    def probe(self): return True

    def stream(self, text, voice=None, speed=None):
        self.log.append(("say", text))
        for _ in range(3):
            yield np.ones(10, np.float32), 24000


class Here:
    """This machine's recogniser and synthesiser."""
    def __init__(self): self.calls = []
    def begin_utterance(self): self.calls.append("begin")
    def feed(self, chunk, rate): pass
    def partial(self): return ""
    def finish(self): return "here heard"

    def transcribe(self, audio, **k):
        self.calls.append("batch")
        return [remote_speech._Segment("here batch")], None

    def stream(self, text, voice=None, speed=None):
        self.calls.append(("say", text))
        yield np.zeros(10, np.float32), 24000


@pytest.fixture
def wrapped(tmp_path, boards, monkeypatch):
    monkeypatch.setattr(remote_speech, "Health", Up)
    monkeypatch.setattr(remote_speech, "RemoteStt", Board)
    monkeypatch.setattr(remote_speech, "RemoteTts", Board)
    Board.log = []
    host = SpeechHost(tmp_path)
    host.stt, host.kokoro, host.tts_available = Here(), Here(), True
    local_stt, local_tts = host.stt, host.kokoro
    host._setup_remote_speech()
    return host, local_stt, local_tts


def _utterance(stt):
    stt.begin_utterance()
    return stt.finish()


def test_recognition_follows_the_mode_from_the_next_utterance(wrapped):
    host, here, _ = wrapped
    assert _utterance(host.stt) == "board heard"
    host.set_speech_mode("local")
    assert _utterance(host.stt) == "here heard" and here.calls == ["begin"]
    assert host.stt.transcribe(np.zeros(10, np.float32))[0][0].text == "here batch"
    host.set_speech_mode("network")
    assert _utterance(host.stt) == "board heard"
    assert host.stt.transcribe(np.zeros(10, np.float32))[0][0].text == "board batch"


def test_synthesis_follows_the_mode_from_the_next_sentence(wrapped):
    host, _, here = wrapped
    list(host.kokoro.stream("One."))
    assert host.kokoro.speaking_remote and ("say", "One.") in Board.log
    host.set_speech_mode("toggle")
    list(host.kokoro.stream("Two."))
    assert not host.kokoro.speaking_remote and here.calls == [("say", "Two.")]
    assert ("say", "Two.") not in Board.log


def test_a_sentence_on_a_board_finishes_there(wrapped):
    host, _, here = wrapped
    gen = host.kokoro.stream("Halfway through.")
    next(gen)
    host.set_speech_mode("local")
    assert len(list(gen)) == 2, "the rest of it, from the board"
    assert here.calls == []
    list(host.kokoro.stream("Next."))
    assert here.calls == [("say", "Next.")]


def test_network_still_keeps_the_local_tier_here(wrapped):
    host, here, tts_here = wrapped
    host.backend, host.llm_tier = "local", "local"
    assert _utterance(host.stt) == "here heard"
    list(host.kokoro.stream("Hi."))
    assert tts_here.calls == [("say", "Hi.")] and Board.log == []


def test_it_starts_where_the_env_file_left_it(monkeypatch):
    monkeypatch.setattr(va, "REMOTE_STT", STT)
    monkeypatch.setattr(va, "REMOTE_TTS", "")
    for mode in ("local", "network"):
        monkeypatch.setattr(va, "SPEECH_MODE", mode)
        assert va._initial_speech_mode() == mode
    monkeypatch.setattr(va, "REMOTE_STT", "")
    assert va._initial_speech_mode() == "local", "nothing to use on the network"


@pytest.mark.parametrize("value, want", [
    ("local", "local"), (" Local ", "local"), ("network", "network"),
    ("", "network"), ("boards", "network"), (None, "network"),
])
def test_the_env_variable(value, want):
    """Read once, at import, in a process of its own."""
    env = {k: v for k, v in os.environ.items() if k != "VOICE_ASSISTANT_SPEECH"}
    if value is not None:
        env["VOICE_ASSISTANT_SPEECH"] = value
    out = subprocess.run(
        [sys.executable, "-c", "import voice_assistant as va; print(va.SPEECH_MODE)"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert out.stdout.strip().splitlines()[-1] == want, out.stderr[-2000:]


# --- by voice -----------------------------------------------------------------

@pytest.mark.parametrize("text, want", [
    ("Switch speech to local.", "local"),
    ("switch the speech to network", "network"),
    ("change speech back to local please", "local"),
    ("use network speech", "network"),
    ("Use local speech.", "local"),
    ("go back to network speech", "network"),
    ("Speech: local", "local"),
    ("speech network", "network"),
    # not commands
    ("switch to local", None),                      # the model
    ("use the local model", None),
    ("tell me about speech recognition on the network", None),
    ("what is local speech", None),
])
def test_parse_speech_request(text, want):
    assert va._parse_speech_request(text) == want


def test_switch_speech_is_not_heard_as_a_backend_switch():
    """The backend parser alone would take it for the local model; the turn
    asks the speech parser first."""
    assert va._parse_backend_request("switch speech to local") == "local"
    assert va._parse_speech_request("switch speech to local") == "local"


class SpokenHost(SpeechHost):
    def __init__(self, state_dir, heard, mode="network"):
        super().__init__(state_dir, mode)
        import threading
        from types import SimpleNamespace
        self.is_active = True
        self.is_processing = True
        self._abort_event = threading.Event()
        self.capture = self.vad = SimpleNamespace(
            mute=lambda *a: None, flush=lambda: None, reset=lambda: None)
        self.heard, self.said, self.asked = heard, [], []
        self.voice_unlock = self._embedder = None

    def _transcribe(self, audio): return self.heard
    def _save_turn(self, audio, transcript): pass
    def _say(self, text): self.said.append(text)
    def _router_finish(self): pass
    def _play_chime_async(self, name): pass
    def _query_and_speak(self, text, abort, *a): self.asked.append(text)
    def _switch_backend(self, *a): raise AssertionError("not a backend switch")


def test_a_spoken_switch_never_reaches_a_model(tmp_path, boards, monkeypatch):
    monkeypatch.setattr(va, "TTS_TAIL_GATE", 0)
    host = SpokenHost(tmp_path, "Switch speech to local.")
    asyncio.run(host._process_audio([]))
    assert host.speech_mode == "local" and host.said == ["Okay, speech is local."]
    assert host.asked == [] and boards == [("VOICE_ASSISTANT_SPEECH", "local")]
    host = SpokenHost(tmp_path, "Use local speech.", mode="local")
    asyncio.run(host._process_audio([]))
    assert host.said == ["Speech is already local."]


def test_a_spoken_switch_without_boards_says_so(tmp_path, no_boards, monkeypatch):
    monkeypatch.setattr(va, "TTS_TAIL_GATE", 0)
    host = SpokenHost(tmp_path, "Use network speech.", mode="local")
    asyncio.run(host._process_audio([]))
    assert host.speech_mode == "local"
    assert host.said == ["There's no network speech configured, so speech stays on this machine."]


# --- `assistant --speech`, `assistant --status` -------------------------------

def _cli():
    loader = importlib.machinery.SourceFileLoader("assistant_cli", str(ROOT / "assistant"))
    spec = importlib.util.spec_from_loader("assistant_cli", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def _run(cli, monkeypatch, argv, reply):
    sent = []
    monkeypatch.setattr(cli, "request", lambda payload: sent.append(payload) or reply)
    monkeypatch.setattr(sys, "argv", ["assistant"] + argv)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert cli.main() == 0
    return sent, out.getvalue()


def test_cli_speech_sends_the_command_and_prints_the_state(monkeypatch):
    cli = _cli()
    for value in ("toggle", "local", "network"):
        sent, out = _run(cli, monkeypatch, ["--speech", value],
                         {"ok": True, "speech": "local", "label": "Speech: local"})
        assert sent == [{"cmd": "speech", "value": value}] and out == "Speech: local\n"
    _, out = _run(cli, monkeypatch, ["--speech", "network"],
                  {"ok": True, "speech": "local",
                   "label": "Speech: local (no network speech configured)"})
    assert out == "Speech: local (no network speech configured)\n"
    monkeypatch.setattr(sys, "argv", ["assistant", "--speech", "sometimes"])
    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
        cli.main()


STATUS = {"ok": True, "label": "Claude", "preference": "auto", "local_llm": "down",
          "network_llm": None, "dsh": "installed", "voice_active": True, "busy": False,
          "voice_only": False, "session": "s"}


def test_cli_status_shows_where_speech_runs(monkeypatch):
    cli = _cli()
    _, out = _run(cli, monkeypatch, ["--status"], dict(
        STATUS, speech="network", speech_tiers=["claude", "network"],
        speech_servers={"stt": ["a:8772", "b:8772"], "tts": ["http://a:8771"]}))
    assert "speech     network  (stt a:8772, b:8772; tts http://a:8771; claude/network tiers)\n" in out
    _, out = _run(cli, monkeypatch, ["--status"], dict(
        STATUS, speech="local", speech_servers={"stt": ["a:8772"], "tts": []}))
    assert "speech     local\n" in out
    _, out = _run(cli, monkeypatch, ["--status"], dict(
        STATUS, speech="local", speech_servers={"stt": [], "tts": []}))
    assert "speech     local  (no network speech configured)\n" in out
    _, out = _run(cli, monkeypatch, ["--status"], STATUS)
    assert "speech" not in out, "an older service says nothing about it"
