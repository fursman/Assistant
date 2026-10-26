"""Spoken commands the assistant handles itself (voice_assistant.py:
_parse_backend_request, _parse_voice_only_request), voice-only's gate on a
turn, and the shell tool's refusal to restart the service it runs in
(_run_shell_tool).

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

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import voice_assistant as va  # noqa: E402

# The machine's own settings must not decide a test: with voice unlock on, a
# turn would ask whether this desktop's screen is locked right now.
va.VOICE_UNLOCK = False


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
    ("switch to the network model", "network"),
    ("go over to the cluster", "network"),
    ("use the cluster", "network"),
    ("switch to hosted", "claude"),
    ("use the hosted model", "claude"),
    ("switch back to local", "local"),
    # not commands
    ("use the local time zone", None),
    ("use the network printer", None),
    ("use the hosted docs", None),
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
        self.voice_only, self.voice_unlock = False, None
        self._voiceprint = self._embedder = None

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


# --- tiers: local, network, hosted -------------------------------------------------

class TierHost(va.VoiceAssistant):
    """The backend choice alone: no audio, no CLI, no servers."""
    def __init__(self, state_dir, local="absent", claude=True):
        self.logger = logging.getLogger("test-tiers")
        self.state_dir = state_dir
        self.backend, self.llm_tier, self.backend_pref = "claude", "local", "auto"
        self._status_state = "ready"
        self.is_active = False
        self._dsh = None
        self._dsh_available = False
        self._claude_available = claude
        self._router_verdict = None
        self._local_state = local
        self.notes = []

    def _local_tier_state(self):
        return self._local_state, f"local is {self._local_state}"

    def _append_transcript(self, role, text): pass
    def _notify(self, text, **k): self.notes.append(text)


@pytest.fixture
def tiers(monkeypatch):
    """A network model configured, nothing reachable, nothing written."""
    monkeypatch.setattr(va, "NETWORK_LLM_URL", "http://llm.lan:8080/v1")
    monkeypatch.setattr(va, "NETWORK_LLM_MODEL", "qwen3.8-flash-next")
    # Whatever this machine's env configured, and whatever a scan found: the
    # current URL is then the one network server, as these tests mean it.
    monkeypatch.setattr(va, "NETWORK_LLM_CONFIGURED", ("", ""))
    monkeypatch.setattr(va, "_DISCOVERED_SERVERS", [])
    monkeypatch.setattr(va, "NETWORK_HARNESS", "native")
    monkeypatch.setattr(va, "LOCAL_HARNESS", "native")
    monkeypatch.setattr(va, "LOCAL_LLM_MODEL", "qwen3.8-27b")
    monkeypatch.setattr(va.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(va, "_write_env_setting", lambda key, value: None)
    monkeypatch.setattr(va, "_llm_health", lambda url, timeout=1.0: "ready")
    monkeypatch.setattr(va, "_start_user_unit", lambda unit: True)


def test_only_local_settings_make_the_local_tier_as_before(monkeypatch):
    monkeypatch.setattr(va, "NETWORK_LLM_URL", "")
    llm = va._llm_tier("local")
    assert (llm.url, llm.model, llm.harness) == (va.LOCAL_LLM_URL, va.LOCAL_LLM_MODEL, va.LOCAL_HARNESS)
    assert llm.base == va.LOCAL_LLM_URL.rstrip("/").removesuffix("/v1")
    assert va._backend_switch_note().count("network") == 0


def test_toggle_without_a_gpu_flips_between_network_and_claude(tmp_path, tiers):
    host = TierHost(tmp_path, local="absent")
    seen = [host._switch_backend("toggle")["tier"] for _ in range(4)]
    assert seen == ["network", "claude", "network", "claude"]


def test_toggle_cycles_all_three_tiers_and_skips_a_missing_one(tmp_path, tiers, monkeypatch):
    host = TierHost(tmp_path, local="ready")
    seen = [host._switch_backend("toggle")["tier"] for _ in range(4)]
    assert seen == ["local", "network", "claude", "local"]
    monkeypatch.setattr(va, "NETWORK_LLM_URL", "")
    seen = [host._switch_backend("toggle")["tier"] for _ in range(3)]
    assert seen == ["claude", "local", "claude"]


def test_auto_prefers_local_then_network_then_claude(tmp_path, tiers, monkeypatch):
    assert TierHost(tmp_path, local="starting")._resolve_backend()[:2] == ("local", "local")
    assert TierHost(tmp_path, local="stopped")._resolve_backend()[:2] == ("local", "network")
    monkeypatch.setattr(va, "NETWORK_LLM_URL", "")
    assert TierHost(tmp_path, local="absent")._resolve_backend()[:2] == ("claude", "claude")


def test_dsh_is_the_local_tier_under_the_harness(tmp_path, tiers, monkeypatch):
    host = TierHost(tmp_path, local="absent")
    host.backend_pref = "dsh"
    monkeypatch.setattr(va.importlib.util, "find_spec", lambda name: object())
    assert host._resolve_backend()[:2] == ("dsh", "local")
    # The network tier takes the harness from its own setting.
    monkeypatch.setattr(va, "NETWORK_HARNESS", "dsh")
    host.backend_pref = "network"
    assert host._resolve_backend()[:2] == ("dsh", "network")


def test_network_tier_is_what_the_model_and_the_panels_hear(tmp_path, tiers):
    host = TierHost(tmp_path, local="absent")
    res = host._switch_backend("network")
    assert res["ok"] and (host.backend, host.tier) == ("local", "network")
    assert res["label"] == "Network · qwen3.8-flash-next" and host.notes[-1].endswith(res["label"])
    assert host._backend_label(spoken=True) == "the network model"
    llm = host._llm()
    assert (llm.url, llm.model, llm.base) == ("http://llm.lan:8080/v1", "qwen3.8-flash-next",
                                              "http://llm.lan:8080")
    prompt = va._tier_prompt(va.LOCAL_VOICE_PROMPT, llm)
    assert "qwen3.8-flash-next, served by llama.cpp at llm.lan:8080" in prompt
    assert "switch to the network model" in prompt
    d = json.loads((tmp_path / "status").read_text())
    assert d["class"] == ["off", "network"] and d["tier"] == "network"
    assert d["label"] == "Network · qwen3.8-flash-next"
    assert d["local_model"] == "Qwen3.8 Flash Next", "the active tier's model"
    assert host._switch_backend("hosted")["label"] == "Claude" and host.tier == "claude"


def test_network_tier_needs_a_url(tmp_path, tiers, monkeypatch):
    monkeypatch.setattr(va, "NETWORK_LLM_URL", "")
    host = TierHost(tmp_path)
    res = host._switch_backend("network")
    assert not res["ok"] and host.tier == "claude"


def test_network_model_down_falls_back_to_claude(tmp_path, tiers, monkeypatch):
    asked = []
    monkeypatch.setattr(va, "_llm_health", lambda url, timeout=1.0: asked.append(url) or "down")
    monkeypatch.setattr(va, "LLM_FALLBACK", True)
    host = TierHost(tmp_path)
    host.backend, host.llm_tier = "local", "network"
    host._play_chime_async = lambda name: None
    assert host._pick_backend_for_query() == "claude"
    assert asked == ["http://llm.lan:8080/v1"], "the network server, not this machine's"
    assert "Network model not running" in host.notes[-1]
    monkeypatch.setattr(va, "LLM_FALLBACK", False)
    assert host._pick_backend_for_query() == "local"


def test_a_spoken_switch_names_the_tier_briefly(tmp_path, tiers, monkeypatch):
    monkeypatch.setattr(va, "TTS_TAIL_GATE", 0)
    host = SpokenHost(tmp_path, "Switch to the network model.")
    host.llm_tier, host.backend_pref = "local", "claude"
    host._dsh, host._dsh_available, host._claude_available = None, False, True
    host._local_tier_state = lambda: ("absent", "no GPU")
    asyncio.run(host._process_audio([]))
    assert host.said == ["Switched to the network model."]
    assert host.tier == "network"


# --- voice-only ----------------------------------------------------------------------

@pytest.mark.parametrize("text, want", [
    ("Only listen to me.", "on"),
    ("ok, only listen to me from now on", "on"),
    ("listen only to me please", "on"),
    ("Ignore everyone else.", "on"),
    ("turn on voice-only mode", "on"),
    ("voice only", "on"),
    ("Listen to everyone.", "off"),
    ("listen to everybody again", "off"),
    ("can you turn off voice only", "off"),
    ("stop ignoring other people", "off"),
    ("voice only off", "off"),
    # not commands
    ("I want you to listen to everyone's opinion first", None),
    ("only listen to me when I say your name", None),
    ("does voice only mode work on the laptop", None),
    ("listen to this", None),
])
def test_parse_voice_only_request(text, want):
    assert va._parse_voice_only_request(text) == want


def _unit(*head):
    v = np.zeros(256, np.float32)
    v[:len(head)] = head
    return v / np.linalg.norm(v)


USER_VOICE, OTHER_VOICE = _unit(1.0, 0.1), _unit(0.0, 0.0, 1.0)


class GateHost(SpokenHost):
    """A turn all the way to the model, with a voiceprint that knows the user."""
    def __init__(self, state_dir, heard, voice, samples=8):
        import voiceprint
        super().__init__(state_dir, heard)
        self.voice_only = True
        self.voice_unlock = None
        self._voiceprint = voiceprint.Voiceprint(state_dir / "vp.npz")
        for _ in range(samples):
            self._voiceprint.consider(USER_VOICE)
        self._embedder = SimpleNamespace(embed=lambda audio: voice)
        self._last_tts_text = ""
        self.asked, self.chimes = [], []

    def _router_judge_spoken(self, text): return None
    def _query_and_speak(self, text, abort, *a): self.asked.append(text)
    def _play_chime_async(self, name): self.chimes.append(name)


@pytest.fixture
def wakes(monkeypatch):
    """Screen wakes, recorded instead of tapping Shift on the real desktop."""
    calls = []
    monkeypatch.setattr(va, "_wake_screen", lambda: calls.append(True))
    return calls


@pytest.fixture
def gate(monkeypatch, wakes):
    """Nothing written to the real env file; the chime would play if anything spoke."""
    written = []
    monkeypatch.setattr(va, "_write_env_setting", lambda key, value: written.append((key, value)))
    monkeypatch.setattr(va, "TTS_TAIL_GATE", 0)
    monkeypatch.setattr(va, "LISTEN_RESUME_CHIME", True)
    return written


def _turn(host, seconds=3.0):
    asyncio.run(host._process_audio(np.zeros(int(seconds * va.SAMPLE_RATE), np.float32)))


def _wait_for(cond, timeout=5.0):
    """The voiceprint learns on its own thread."""
    import time
    deadline = time.monotonic() + timeout
    while not cond() and time.monotonic() < deadline:
        time.sleep(0.01)
    return cond()


def test_voice_only_drops_another_voice_silently(tmp_path, gate, wakes, caplog):
    host = GateHost(tmp_path, "What's the weather tomorrow?", OTHER_VOICE)
    with caplog.at_level(logging.INFO):
        _turn(host)
    assert host.asked == [] and host.said == [] and host.chimes == []
    assert host._turns_screened() and wakes == [], "nor does it wake the screen"
    assert "Voice-only: voice" in caplog.text and "IGNORED" in caplog.text
    assert host._voiceprint.count == 8, "an ignored turn is not learned from"


def test_voice_only_passes_the_users_voice(tmp_path, gate, wakes, caplog):
    host = GateHost(tmp_path, "What's the weather tomorrow?", USER_VOICE)
    with caplog.at_level(logging.INFO):
        _turn(host)
    assert host.asked == ["What's the weather tomorrow?"] and wakes == [True]
    assert "Voice-only: voice 1.00 vs 0.45 ok" in caplog.text
    assert _wait_for(lambda: host._voiceprint.count == 9), "and learns from it"


def test_voice_only_lets_everyone_through_while_learning(tmp_path, gate):
    host = GateHost(tmp_path, "What's the weather tomorrow?", USER_VOICE, samples=2)
    _turn(host)
    assert host.asked == ["What's the weather tomorrow?"]
    assert _wait_for(lambda: host._voiceprint.count == 3)
    host = GateHost(tmp_path / "b", "Breaking news tonight.", OTHER_VOICE, samples=2)
    _turn(host)
    assert host.asked == ["Breaking news tonight."], "not ready: nobody is shut out"


def test_only_the_user_can_turn_voice_only_off(tmp_path, gate):
    host = GateHost(tmp_path, "Listen to everyone.", OTHER_VOICE)
    _turn(host)
    assert host.voice_only and host.said == [] and gate == []
    host = GateHost(tmp_path / "b", "Listen to everyone.", USER_VOICE)
    _turn(host)
    assert not host.voice_only and host.said == ["Okay, I'll listen to everyone."]
    assert gate == [("VOICE_ASSISTANT_VOICE_ONLY", "0")] and host.asked == []
    assert json.loads((tmp_path / "b" / "status").read_text())["voice_only"] is False


def test_turning_voice_only_on_before_the_voice_is_learned_says_so(tmp_path, gate):
    host = GateHost(tmp_path, "Only listen to me.", USER_VOICE, samples=3)
    host.voice_only = False
    _turn(host)
    assert host.voice_only and gate == [("VOICE_ASSISTANT_VOICE_ONLY", "1")]
    assert host.said == ["I'm still learning your voice, so I'll listen to everyone "
                         "until I know it."]


def test_voice_only_toggle_persists_and_shows(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    env = tmp_path / ".config/voice-assistant/env"
    env.parent.mkdir(parents=True)
    env.write_text("VOICE_ASSISTANT_LLM_BACKEND=auto\nVOICE_ASSISTANT_VOICE_ONLY=0\n")
    host = GateHost(tmp_path, "", USER_VOICE)
    host.voice_only = False
    host._status_state = "ready"
    res = asyncio.run(host._control_command({"cmd": "voice_only", "value": "toggle"}))
    assert res["ok"] and res["voice_only"] and res["ready"] and res["label"] == "Only your voice"
    assert env.read_text() == "VOICE_ASSISTANT_LLM_BACKEND=auto\nVOICE_ASSISTANT_VOICE_ONLY=1\n"
    d = json.loads((tmp_path / "status").read_text())
    assert d["voice_only"] is True and d["class"] == ["ready", "claude", "voice-only"]
    res = asyncio.run(host._control_command({"cmd": "voice_only", "value": "off"}))
    assert not res["voice_only"] and "VOICE_ASSISTANT_VOICE_ONLY=0" in env.read_text()
    assert json.loads((tmp_path / "status").read_text())["class"] == ["ready", "claude"]
    assert not asyncio.run(host._control_command({"cmd": "voice_only", "value": "maybe"}))["ok"]


def test_a_locked_screen_stays_shut_without_the_voiceprint(tmp_path, gate, wakes, monkeypatch):
    """Voice unlock on but its model would not load: a locked screen still
    gives nobody a model, and does not wake for them."""
    import voiceprint
    monkeypatch.setattr(va, "VOICE_UNLOCK", True)
    monkeypatch.setattr(voiceprint, "hyprlock_pid", lambda: 4242)
    host = GateHost(tmp_path, "What's the weather tomorrow?", USER_VOICE)
    host.voice_only, host._voiceprint, host._embedder = False, None, None
    _turn(host)
    assert host.asked == [] and host.said == [] and wakes == []
    monkeypatch.setattr(voiceprint, "hyprlock_pid", lambda: None)
    _turn(host)
    assert host.asked == ["What's the weather tomorrow?"], "unlocked, it answers"
