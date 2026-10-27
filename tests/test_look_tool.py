"""The `look` tool (look_tools.py, and voice_assistant.py: _server_vision,
_stream_local_llm's tool loop, _shrink_for_history, _run_turn, _control_ask)
and `assistant --image`.

No server, no screen and no camera: /props is a fake urlopen, captures are a
fake subprocess.run, and the OpenAI client is a fake that records each
request and streams a scripted reply (a `look` call, then words). ffmpeg is
run for real in one test, on a generated image, when it is installed.

    .venv/bin/python -m pytest tests
"""
import asyncio
import importlib.machinery
import importlib.util
import io
import json
import logging
import queue
import shutil
import subprocess
import sys
import threading
import types
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import look_tools as lt  # noqa: E402
import voice_assistant as va  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
URI = "data:image/jpeg;base64,/9j/AAAA"


@pytest.fixture
def settings(monkeypatch):
    """This machine's VOICE_ASSISTANT_* settings decide nothing here."""
    monkeypatch.setattr(va, "LOCAL_LLM_URL", "http://llm.test:8081/v1")
    monkeypatch.setattr(va, "LOCAL_LLM_MODEL", "test-model")
    monkeypatch.setattr(va, "LOCAL_LLM_THINK", False)
    monkeypatch.setattr(va, "LOCAL_HARNESS", "native")
    monkeypatch.setattr(va, "NETWORK_LLM_URL", "http://vl.test:8080/v1")
    monkeypatch.setattr(va, "NETWORK_LLM_MODEL", "vl-model")
    monkeypatch.setattr(va, "LOCAL_TOOLS_ENABLED", True)
    monkeypatch.setattr(va, "LOCAL_MAX_TOOL_ITERS", 5)
    monkeypatch.setattr(va, "LOCAL_TOOL_HISTORY_OUTPUT", 4000)
    monkeypatch.setattr(va, "LOOK_ENABLED", True)
    monkeypatch.setattr(va, "LOOK_ATTACH", "tool")
    monkeypatch.setattr(va, "LOOK_FORCE", True)
    monkeypatch.setattr(va, "ROUTER_TOOLS", True)
    monkeypatch.setattr(va, "ROUTER_THINK", True)
    monkeypatch.setattr(va, "HARD_URL", "")
    monkeypatch.setattr(va, "CLAUDE_PERSISTENT", True)
    monkeypatch.setattr(va, "LLM_FALLBACK", True)
    monkeypatch.setattr(va, "_VISION", {})
    monkeypatch.setattr(lt, "LOOK_MAX_SIDE", 1280)
    monkeypatch.setattr(lt, "LOOK_SCREEN", "focused")
    monkeypatch.setattr(lt, "LOOK_CAMERA", "")
    monkeypatch.setattr(lt, "LOOK_MAX_FILE_MB", 40.0)


def _sees(monkeypatch, vision):
    monkeypatch.setattr(va, "_server_vision", lambda url, timeout=1.0: vision)


# --- does the model see? -------------------------------------------------------

class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _props(monkeypatch, answers):
    """urlopen for /props: each call takes the next answer (a dict to return
    as JSON, or an exception to raise). Returns the URLs asked."""
    asked = []
    answers = list(answers)

    def urlopen(url, timeout=None):
        asked.append(url)
        a = answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return _Resp(json.dumps(a).encode())
    monkeypatch.setattr(va.urllib.request, "urlopen", urlopen)
    return asked


def test_vision_is_read_from_props_and_cached_per_server(settings, monkeypatch):
    asked = _props(monkeypatch, [
        {"modalities": {"vision": True, "audio": False}},
        {"modalities": {"vision": False}},
    ])
    assert va._server_vision("http://vl.test:8080/v1") is True
    assert va._server_vision("http://vl.test:8080") is True, "same server, with or without /v1"
    assert va._server_vision("http://qwen.test:8080/v1") is False
    assert asked == ["http://vl.test:8080/props", "http://qwen.test:8080/props"]


@pytest.mark.parametrize("answer", [
    {},                                         # an older server: no modalities at all
    {"modalities": None},
    {"modalities": {"vision": "true"}},         # anything but a literal true
    {"modalities": {"vision": 1}},
    urllib.error.HTTPError("u", 404, "Not Found", {}, None),   # not llama-server
    urllib.error.URLError("connection refused"),
    TimeoutError("timed out"),
])
def test_vision_fails_closed(settings, monkeypatch, answer):
    _props(monkeypatch, [answer])
    assert va._server_vision("http://x.test:8080/v1") is False


def test_no_answer_is_asked_again_but_a_definite_one_is_not(settings, monkeypatch):
    asked = _props(monkeypatch, [urllib.error.URLError("down"), {"modalities": {"vision": True}}])
    assert va._server_vision("http://x.test:8080/v1") is False
    assert va._server_vision("http://x.test:8080/v1") is False, "not asked again at once"
    monkeypatch.setattr(va, "VISION_RETRY_SECONDS", 0.0)
    assert va._server_vision("http://x.test:8080/v1") is True
    assert va._server_vision("http://x.test:8080/v1") is True
    assert len(asked) == 2
    assert va._server_vision("") is False


def test_a_swap_asks_again(settings, monkeypatch):
    """A server that restarted on another model is not trusted from before."""
    va._VISION["http://vl.test:8080"] = (True, 0.0, True)
    va._VISION["http://llm.test:8081"] = (False, 0.0, True)
    monkeypatch.setattr(va, "_write_env_setting", lambda k, v: None)
    host = va.VoiceAssistant.__new__(va.VoiceAssistant)
    host.llm_tier = "local"
    host._set_llm_tier("network")
    assert "http://vl.test:8080" not in va._VISION
    assert "http://llm.test:8081" in va._VISION, "the tier left keeps its answer"
    va._VISION["http://other.test:8080"] = (True, 0.0, True)
    va._use_network_server(va.NetworkServer("http://other.test:8080/v1", "m", ""))
    assert "http://other.test:8080" not in va._VISION


# --- fakes for the local turn -------------------------------------------------------

class _Fn:
    def __init__(self, name=None, arguments=None):
        self.name, self.arguments = name, arguments


class _TC:
    def __init__(self, index, id=None, name=None, arguments=None):
        self.index, self.id, self.function = index, id, _Fn(name, arguments)


class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content, self.reasoning_content, self.tool_calls = content, None, tool_calls


class _Chunk:
    def __init__(self, delta, finish=None):
        self.choices = [types.SimpleNamespace(delta=delta, finish_reason=finish)]


class _Stream:
    def __init__(self, chunks):
        self._it = iter(chunks)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)

    def close(self):
        pass


class FakeCompletions:
    """Answers each request with the next scripted step: ("look", args) for
    a tool call, ("look", args, "words") for words and then the call, or a
    string to say."""

    def __init__(self, steps):
        self.steps, self.calls = list(steps), []

    def create(self, **kw):
        # A copy of the messages as they were sent: the loop appends to the list.
        self.calls.append(dict(kw, messages=[dict(m) for m in kw["messages"]]))
        step = self.steps.pop(0) if self.steps else "Done."
        if isinstance(step, tuple):
            name, args, *words = step
            return _Stream([_Chunk(_Delta(content=w)) for w in words]
                           + [_Chunk(_Delta(tool_calls=[_TC(0, "call_1", name, json.dumps(args))])),
                              _Chunk(_Delta(), finish="tool_calls")])
        return _Stream([_Chunk(_Delta(content=step)), _Chunk(_Delta(), finish="stop")])


class FakeLedger:
    def __init__(self):
        self.remembered = []

    def recap_for(self, backend):
        return ""

    def remember(self, user, reply, backend=None):
        self.remembered.append((user, reply, backend))


class Host(va.VoiceAssistant):
    """What a local turn touches, with the real history handling."""

    def __init__(self, steps, verdict=None):
        self.logger = logging.getLogger("test-look")
        self.client = types.SimpleNamespace(chat=types.SimpleNamespace(
            completions=FakeCompletions(steps)))
        self.ledger = FakeLedger()
        self.llm_tier = "network"
        self._router = None
        self._router_verdict = verdict
        self._local_history = []
        self._assistant_text = ""
        self._thinking_text = ""
        self._sentence_queue = queue.Queue()
        self._local_stream = None
        self._hard_client = None
        self._hard_health = (0.0, False)
        self._turn_tool_calls = 0
        self.looked = []
        self.notified = []
        self.fillers = []

    @property
    def calls(self):
        return self.client.chat.completions.calls

    def _local_llm_client(self):
        return self.client

    def _ledger(self):
        return self.ledger

    def _flush_sentences(self, final=False):
        pass

    def _maybe_notify_thinking(self):
        pass

    def _notify_tool_use(self, name, args):
        self.notified.append((name, json.loads(args)))

    def _save_local_history(self):
        pass

    def _router_used(self, backend):
        pass

    def _say_filler(self, text):
        self.fillers.append(text)

    def _look_tool(self, source, path=""):
        self.looked.append((source, path))
        return "[image of the screen, 20:15]", URI


def _tool_names(kw):
    return [t["function"]["name"] for t in kw.get("tools") or []]


# --- offered only to a model that sees ---------------------------------------------

def test_look_is_offered_when_the_model_sees(settings, monkeypatch):
    _sees(monkeypatch, True)
    host = Host(["Hello."])
    assert host._stream_local_llm("hello", threading.Event()) is True
    kw = host.calls[0]
    assert _tool_names(kw) == ["run_shell", "web_search", "fetch_page", "weather", "look"]
    assert va.LOOK_VOICE_NOTE in kw["messages"][0]["content"]


@pytest.mark.parametrize("vision, enabled, tools", [
    (False, True, True),        # the server has no projector, or did not answer
    (True, False, True),        # VOICE_ASSISTANT_LOOK=0
    (True, True, False),        # VOICE_ASSISTANT_LOCAL_TOOLS=0
])
def test_look_is_not_offered_otherwise(settings, monkeypatch, vision, enabled, tools):
    _sees(monkeypatch, vision)
    monkeypatch.setattr(va, "LOOK_ENABLED", enabled)
    monkeypatch.setattr(va, "LOCAL_TOOLS_ENABLED", tools)
    host = Host(["Hello."])
    host._stream_local_llm("what's on my screen", threading.Event())
    kw = host.calls[0]
    assert "look" not in _tool_names(kw)
    assert va.LOOK_VOICE_NOTE not in kw["messages"][0]["content"]
    if tools:
        assert kw["messages"][0]["content"] == va._tier_prompt(va.LOCAL_VOICE_PROMPT, host._llm()), \
            "a model that cannot see gets the prompt exactly as before"


def test_the_prompt_note_goes_after_the_tools_paragraph(settings):
    p = va._local_voice_prompt(va._llm_tier("local"), look=True)
    assert p.index("rather than explaining how.") < p.index("You can also see, but only through look") \
        < p.index("Never search with run_shell")
    assert va._LOOK_ANCHOR in va.LOCAL_VOICE_PROMPT


# --- the image in the request, and out of the history ------------------------------------

def test_the_image_rides_in_the_tool_result(settings, monkeypatch):
    _sees(monkeypatch, True)
    host = Host([("look", {"source": "screen"}), "A terminal with a build log."])
    assert host._stream_local_llm("what's on my screen", threading.Event()) is True
    assert host.looked == [("screen", "")]
    assert host.notified == [("Look", {"source": "screen", "path": ""})]
    sent = host.calls[1]["messages"]
    assert sent[-2]["role"] == "assistant" and sent[-2]["tool_calls"][0]["function"]["name"] == "look"
    assert sent[-1] == {"role": "tool", "tool_call_id": "call_1", "content": [
        {"type": "text", "text": "[image of the screen, 20:15]"},
        {"type": "image_url", "image_url": {"url": URI}}]}


def test_the_history_keeps_a_note_not_the_image(settings, monkeypatch):
    _sees(monkeypatch, True)
    host = Host([("look", {"source": "camera"}), "You are holding a mug."])
    host._stream_local_llm("what am I holding", threading.Event())
    history = host._local_history
    assert [m["role"] for m in history] == ["user", "assistant", "tool", "assistant"]
    assert history[2]["content"] == "[image of the screen, 20:15]"
    assert all(isinstance(m.get("content"), (str, type(None))) for m in history)
    assert "base64" not in json.dumps(history)
    # The next turn sends the note, not the picture.
    host.client.chat.completions.steps = ["Still a mug."]
    host._stream_local_llm("and now", threading.Event())
    assert "base64" not in json.dumps(host.calls[-1]["messages"])


def test_attach_as_a_user_message_when_asked(settings, monkeypatch):
    _sees(monkeypatch, True)
    monkeypatch.setattr(va, "LOOK_ATTACH", "user")
    host = Host([("look", {"source": "screen"}), "A browser."])
    host._stream_local_llm("look at my screen", threading.Event())
    sent = host.calls[1]["messages"]
    assert sent[-2] == {"role": "tool", "tool_call_id": "call_1",
                        "content": "[image of the screen, 20:15]"}
    assert sent[-1]["role"] == "user"
    assert sent[-1]["content"][1] == {"type": "image_url", "image_url": {"url": URI}}
    assert [m["role"] for m in host._local_history] == ["user", "assistant", "tool", "assistant"], \
        "the carrier message is not remembered"
    assert "base64" not in json.dumps(host._local_history)


def test_a_failed_capture_is_a_text_result(settings, monkeypatch):
    _sees(monkeypatch, True)

    def no_camera(source, path="", desktop="", logger=None):
        raise lt.LookError("there is no camera on this machine")
    monkeypatch.setattr(va.look_tools, "look", no_camera)
    monkeypatch.setattr(va, "_refresh_session_env", lambda: None)

    class RealLook(Host):
        _look_tool = va.VoiceAssistant._look_tool
    host = RealLook([("look", {"source": "camera"}), "There is no camera here."])
    host._stream_local_llm("what am I holding", threading.Event())
    assert host.calls[1]["messages"][-1]["content"] == "(no image: there is no camera on this machine)"


def test_look_is_refused_when_it_was_not_offered(settings, monkeypatch):
    _sees(monkeypatch, False)
    host = Host([("look", {"source": "screen"}), "I cannot see."])
    host._stream_local_llm("what's on my screen", threading.Event())
    assert host.looked == []
    assert host.calls[1]["messages"][-1]["content"] == "(unknown tool look)"


# --- the router ------------------------------------------------------------------------------

class Verdict:
    def __init__(self, tools=(), hard=False):
        self.tools, self.hard, self.backend = list(tools), hard, None


def test_a_turn_that_asks_to_look_must_call_a_tool_first(settings, monkeypatch):
    """Nemotron, asked "what's on my screen" under the voice prompt, described
    an invented screen without calling look. The first step now has to call."""
    _sees(monkeypatch, True)
    host = Host([("look", {"source": "screen"}), "A terminal."], verdict=Verdict(tools=[]))
    host._stream_local_llm("what's on my screen", threading.Event())
    assert host.calls[0]["tool_choice"] == "required"
    assert "look" in _tool_names(host.calls[0])
    assert "tool_choice" not in host.calls[1], "then auto, though the router wanted no tool"


def test_words_before_the_forced_call_are_not_spoken(settings, monkeypatch):
    """What Nemotron did when made to call: a made-up description, then the call."""
    _sees(monkeypatch, True)
    spoken = []
    host = Host([("look", {"source": "screen"}, "I see a terminal and a chat window."),
                 "Two dot-matrix hands on black."], verdict=Verdict(tools=[]))
    host._flush_sentences = lambda final=False: spoken.append(host._assistant_text)
    host._stream_local_llm("what's on my screen", threading.Event())
    assert host._assistant_text == "Two dot-matrix hands on black."
    assert all("terminal" not in t for t in spoken)
    assert host.calls[1]["messages"][-2]["content"] is None, "nor shown to the model as its own"
    assert "terminal" not in json.dumps(host._local_history)


def test_a_forced_step_that_answers_anyway_is_spoken(settings, monkeypatch):
    _sees(monkeypatch, True)
    host = Host(["I cannot take a screenshot right now."])
    host._stream_local_llm("what's on my screen", threading.Event())
    assert host.calls[0]["tool_choice"] == "required"
    assert host._assistant_text == "I cannot take a screenshot right now."


def test_forcing_can_be_switched_off(settings, monkeypatch):
    _sees(monkeypatch, True)
    monkeypatch.setattr(va, "LOOK_FORCE", False)
    host = Host(["Hi."], verdict=Verdict(tools=[]))
    host._stream_local_llm("what's on my screen", threading.Event())
    assert "tool_choice" not in host.calls[0], "auto, though the router wanted no tool"
    host = Host(["Hi."])
    host._stream_local_llm("what is two plus two", threading.Event())
    assert "tool_choice" not in host.calls[0], "and nothing is forced on other turns"


def test_other_toolless_turns_are_gated_as_before(settings, monkeypatch):
    _sees(monkeypatch, True)
    host = Host(["Four."], verdict=Verdict(tools=[]))
    host._stream_local_llm("what is two plus two", threading.Event())
    assert host.calls[0]["tool_choice"] == "none"
    _sees(monkeypatch, False)
    host = Host(["I cannot see."], verdict=Verdict(tools=[]))
    host._stream_local_llm("what's on my screen", threading.Event())
    assert host.calls[0]["tool_choice"] == "none", "no look on a model that cannot see"


def test_a_look_turn_skips_the_hard_lane(settings, monkeypatch):
    _sees(monkeypatch, True)
    host = Host(["A diagram."], verdict=Verdict(tools=[], hard=True))
    host._stream_local_llm("describe the picture in Downloads", threading.Event())
    kw = host.calls[0]
    assert host.fillers == [] and kw["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    host = Host(["Because."], verdict=Verdict(tools=[], hard=True))
    host._stream_local_llm("why does entropy always increase", threading.Event())
    assert host.fillers, "a hard turn that is not about looking still thinks"


@pytest.mark.parametrize("text, hint", [
    ("what's on my screen", True),
    ("what am I holding", True),
    ("look at this", True),
    ("describe the image in Downloads/x.png", True),
    ("can you see the error", True),
    ("I'm holding up a book", True),
    ("what is the weather tomorrow", False),
    ("restart the network", False),
    ("you see, the thing is I forgot", False),
    ("let's look at the options tomorrow", False),
    ("I'm holding off on that", False),
])
def test_look_hint(text, hint):
    assert bool(va._LOOK_HINT.search(text)) == hint


# --- capture: which command on which desktop -----------------------------------------------

class _Proc:
    def __init__(self, returncode=0, stdout=b"", stderr=b""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def _subprocess(monkeypatch, handler):
    ran = []

    def run(cmd, **kw):
        ran.append(cmd)
        return handler(cmd)
    monkeypatch.setattr(lt.subprocess, "run", run)
    return ran


def test_hyprland_screenshots_the_focused_monitor(settings, monkeypatch):
    monitors = json.dumps([{"name": "DP-1", "focused": False}, {"name": "HDMI-A-1", "focused": True}])
    ran = _subprocess(monkeypatch, lambda cmd: _Proc(stdout=monitors.encode()))
    assert lt.screen_command("Ubuntu, Hyprland on Wayland", "/t/s.png") == \
        ["grim", "-t", "png", "-o", "HDMI-A-1", "/t/s.png"]
    assert ran == [["hyprctl", "-j", "monitors"]]


def test_hyprland_all_outputs_or_when_hyprctl_fails(settings, monkeypatch):
    monkeypatch.setattr(lt, "LOOK_SCREEN", "all")
    ran = _subprocess(monkeypatch, lambda cmd: _Proc(stdout=b"[]"))
    assert lt.screen_command("Hyprland", "/t/s.png") == ["grim", "-t", "png", "/t/s.png"]
    assert ran == [], "no need to ask which monitor"
    monkeypatch.setattr(lt, "LOOK_SCREEN", "focused")
    _subprocess(monkeypatch, lambda cmd: _Proc(returncode=1, stdout=b"not json"))
    assert lt.screen_command("Hyprland", "/t/s.png") == ["grim", "-t", "png", "/t/s.png"]


def test_sway_asks_swaymsg(settings, monkeypatch):
    ran = _subprocess(monkeypatch, lambda cmd: _Proc(stdout=b'[{"name": "eDP-1", "focused": true}]'))
    assert lt.screen_command("Arch, Sway on Wayland", "/t/s.png")[-3:] == ["-o", "eDP-1", "/t/s.png"]
    assert ran[0][0] == "swaymsg"


def test_gnome_uses_the_portal(settings, monkeypatch):
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    cmd = lt.screen_command("Ubuntu, GNOME on Wayland", "/t/s.png")
    assert cmd[:2] == ["/usr/bin/python3", "-c"] and "org.freedesktop.portal.Screenshot" in cmd[2]


def test_unknown_desktop_without_wayland_uses_the_portal(settings, monkeypatch):
    monkeypatch.delenv("WAYLAND_DISPLAY")
    assert lt.screen_command("KDE on X11", "/t/s.png")[0] == "/usr/bin/python3"


def test_portal_screenshot_is_moved_out_of_the_users_files(settings, monkeypatch, tmp_path):
    shot = tmp_path / "Pictures" / "Screenshot from now.png"
    shot.parent.mkdir()
    shot.write_bytes(PNG)
    _subprocess(monkeypatch, lambda cmd: _Proc(stdout=f"file://{shot}".replace(" ", "%20").encode()))
    out = tmp_path / "out.png"
    assert lt.capture_screen("GNOME", str(out)) == "the screen"
    assert out.read_bytes() == PNG and not shot.exists()


def test_gnome_failure_says_what_to_do(settings, monkeypatch):
    _subprocess(monkeypatch, lambda cmd: _Proc(returncode=1, stderr=b"response 2"))
    with pytest.raises(lt.LookError, match="permission"):
        lt.capture_screen("Ubuntu, GNOME on Wayland", "/t/s.png")


def test_grim_failure_and_missing_grim(settings, monkeypatch):
    monkeypatch.setattr(lt, "LOOK_SCREEN", "all")
    _subprocess(monkeypatch, lambda cmd: _Proc(returncode=1, stderr=b"compositor doesn't support wlr-screencopy"))
    with pytest.raises(lt.LookError, match="screencopy"):
        lt.capture_screen("Hyprland", "/t/s.png")

    def missing(cmd, **kw):
        raise FileNotFoundError(cmd[0])
    monkeypatch.setattr(lt.subprocess, "run", missing)
    with pytest.raises(lt.LookError, match="grim is not installed"):
        lt.capture_screen("Hyprland", "/t/s.png")


# --- capture: the camera ------------------------------------------------------------------

def test_no_camera_is_said_plainly(settings, monkeypatch):
    monkeypatch.setattr(lt, "camera_device", lambda: "")
    with pytest.raises(lt.LookError, match="no camera"):
        lt.capture_camera("/t/c.jpg")


def test_camera_by_ffmpeg_then_its_defaults(settings, monkeypatch, tmp_path):
    monkeypatch.setattr(lt, "camera_device", lambda: "/dev/video4")
    out = tmp_path / "c.jpg"

    def run(cmd):
        if "-input_format" in cmd:
            return _Proc(returncode=1, stderr=b"Cannot set format")
        out.write_bytes(b"\xff\xd8\xff")
        return _Proc()
    ran = _subprocess(monkeypatch, run)
    assert lt.capture_camera(str(out)) == "the camera"
    assert ran[0][:7] == ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "v4l2"]
    assert ran[0][ran[0].index("-i") + 1] == "/dev/video4" and "mjpeg" in ran[0]
    assert "-input_format" not in ran[1]


def test_busy_camera(settings, monkeypatch):
    monkeypatch.setattr(lt, "camera_device", lambda: "/dev/video0")
    _subprocess(monkeypatch, lambda cmd: _Proc(returncode=1, stderr=b"/dev/video0: Device or resource busy"))
    with pytest.raises(lt.LookError, match="busy"):
        lt.capture_camera("/t/c.jpg")


def test_camera_command_from_the_settings(settings, monkeypatch):
    monkeypatch.setattr(lt, "LOOK_CAMERA", "cam look {file} 1280x720")
    assert lt.camera_command("/tmp/a b.jpg") == ["bash", "-c", "cam look '/tmp/a b.jpg' 1280x720"]
    monkeypatch.setattr(lt, "LOOK_CAMERA", "/dev/null")
    assert lt.camera_device() == "/dev/null"


# --- files ------------------------------------------------------------------------------

def test_image_files_are_found_and_checked(settings, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "Downloads").mkdir()
    (tmp_path / "Downloads" / "x.png").write_bytes(PNG)
    want = tmp_path / "Downloads" / "x.png"
    assert lt.check_image_file("~/Downloads/x.png") == want
    assert lt.check_image_file("Downloads/x.png") == want, "relative to home, as said aloud"
    assert lt.check_image_file(f'"{want}"') == want


@pytest.mark.parametrize("name, data, match", [
    ("notes.png", b"just some text, named like a picture", "not an image"),
    ("doc.pdf", b"%PDF-1.7", "not an image"),
    ("missing.png", None, "no file"),
    ("", None, "no file path"),
])
def test_non_images_are_refused(settings, tmp_path, name, data, match):
    if data is not None:
        (tmp_path / name).write_bytes(data)
    with pytest.raises(lt.LookError, match=match):
        lt.check_image_file(str(tmp_path / name) if name else "")


def test_folders_and_huge_files_are_refused(settings, monkeypatch, tmp_path):
    with pytest.raises(lt.LookError, match="folder"):
        lt.check_image_file(str(tmp_path))
    big = tmp_path / "big.png"
    big.write_bytes(PNG + b"\x00" * 2_000_000)
    monkeypatch.setattr(lt, "LOOK_MAX_FILE_MB", 1.0)
    with pytest.raises(lt.LookError, match="limit"):
        lt.check_image_file(str(big))


@pytest.mark.parametrize("head, fmt", [
    (b"\xff\xd8\xff\xe0", "JPEG"), (b"GIF89a", "GIF"), (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "WebP"),
    (b"BM\x00\x00", "BMP"), (b"II*\x00", "TIFF"), (b"RIFF\x00\x00\x00\x00WAVE", ""),
])
def test_image_format(head, fmt):
    assert lt.image_format(head) == fmt


# --- downscale -----------------------------------------------------------------------------

def test_downscale_asks_ffmpeg_for_a_bounded_jpeg(settings, monkeypatch):
    monkeypatch.setattr(lt.shutil, "which", lambda name: "/usr/bin/" + name)
    ran = _subprocess(monkeypatch, lambda cmd: _Proc(stdout=b"\xff\xd8jpeg"))
    monkeypatch.setattr(lt, "LOOK_MAX_SIDE", 768)
    assert lt.downscale("/t/in.png") == b"\xff\xd8jpeg"
    cmd = ran[0]
    assert cmd[0] == "ffmpeg" and cmd[cmd.index("-i") + 1] == "/t/in.png"
    vf = cmd[cmd.index("-vf") + 1]
    assert "min(768,iw)" in vf and "min(768,ih)" in vf and "force_original_aspect_ratio=decrease" in vf
    assert cmd[-5:] == ["-f", "image2pipe", "-c:v", "mjpeg", "-"], "JPEG on stdout, no file"


def test_downscale_falls_back_to_imagemagick(settings, monkeypatch):
    monkeypatch.setattr(lt.shutil, "which", lambda name: "/usr/bin/convert" if name == "convert" else None)
    ran = _subprocess(monkeypatch, lambda cmd: _Proc(stdout=b"\xff\xd8"))
    lt.downscale("/t/in.gif")
    assert ran[0][:2] == ["/usr/bin/convert", "/t/in.gif[0]"] and "1280x1280>" in ran[0]
    monkeypatch.setattr(lt.shutil, "which", lambda name: None)
    with pytest.raises(lt.LookError, match="install ffmpeg"):
        lt.downscale("/t/in.png")


def test_downscale_for_real(settings, tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is not installed")
    src = tmp_path / "wide.png"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc=size=2000x1000", "-frames:v", "1", "-y", str(src)], check=True)
    label, uri = lt.look("file", str(src))
    assert label.startswith(f"image file {src}, ") and uri.startswith("data:image/jpeg;base64,")
    out = tmp_path / "out.jpg"
    import base64
    out.write_bytes(base64.b64decode(uri.split(",", 1)[1]))
    size = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height",
                           "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout.strip()
    assert size == "1280,640"
    small = tmp_path / "small.png"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc=size=320x200", "-frames:v", "1", "-y", str(small)], check=True)
    out.write_bytes(lt.downscale(str(small)))
    size = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height",
                           "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout.strip()
    assert size == "320,200", "never enlarged"


def test_look_leaves_no_capture_behind(settings, monkeypatch, tmp_path):
    made = []

    def capture(desktop, out):
        Path(out).write_bytes(PNG)
        made.append(Path(out))
        return "the screen"
    monkeypatch.setattr(lt, "capture_screen", capture)
    monkeypatch.setattr(lt, "downscale", lambda src, max_side=None: b"\xff\xd8")
    label, uri = lt.look("screen", desktop="Hyprland")
    assert label.startswith("image of the screen, ") and uri == "data:image/jpeg;base64,/9g="
    assert made and not made[0].exists() and not made[0].parent.exists()
    with pytest.raises(lt.LookError, match="unknown source"):
        lt.look("window")


# --- assistant --image, end to end in the service -----------------------------------------

def test_attached_image_goes_to_a_model_that_sees(settings, monkeypatch, tmp_path):
    _sees(monkeypatch, True)
    monkeypatch.setattr(va.look_tools, "look",
                        lambda source, path="", desktop="", logger=None:
                        (f"image file {path}, 20:15", URI))
    host = Host(["A fern with brown tips."], verdict=Verdict(tools=[], hard=True))
    host._turn_image = "/home/u/plant.jpg"
    host._stream_local_llm("what is wrong with this leaf", threading.Event())
    user = host.calls[0]["messages"][-1]
    assert user["content"] == [
        {"type": "text", "text": "[image file /home/u/plant.jpg, 20:15]\nwhat is wrong with this leaf"},
        {"type": "image_url", "image_url": {"url": URI}}]
    assert host.fillers == [], "not sent to the hard lane, which may not see"
    assert host._local_history[0]["content"] == \
        "[image file /home/u/plant.jpg, 20:15]\nwhat is wrong with this leaf"
    assert host.ledger.remembered[0][0].startswith("[image attached: /home/u/plant.jpg]")


class TurnHost(va.VoiceAssistant):
    """_run_turn with each backend recording what it was sent."""

    def __init__(self, backend, sees=False, local_ok=True):
        self.logger = logging.getLogger("test-look-turn")
        self.backend_to_pick = backend
        self.sees, self.local_ok = sees, local_ok
        self.sent = []
        self._assistant_text = ""
        self._claude_available = True
        self.ledger = FakeLedger()

    def _pick_backend_for_query(self):
        return self.backend_to_pick

    def _router_used(self, backend):
        pass

    def _play_chime_async(self, name):
        pass

    def _cue_fallback(self, why):
        pass

    def _llm_sees(self):
        return self.sees

    def _ledger(self):
        return self.ledger

    def _stream_local_llm(self, text, abort, allow_hard=True):
        self.sent.append(("local", text))
        return self.local_ok

    def _stream_dsh(self, text, abort, **kw):
        self.sent.append(("dsh", text))
        return True

    def _stream_claude_persistent(self, text, abort):
        self.sent.append(("claude", text))
        return "ok"

    def _finish_turn(self, sentence_q, success, tts_thread):
        return "reply"


@pytest.mark.parametrize("backend, sees, want", [
    ("claude", False, "[the user attached an image: /p/x.png -- read the file to see it]\nq"),
    ("local", False, "[the user attached an image, /p/x.png, but you cannot see images; "
                     "say so if the question needs it]\nq"),
    ("dsh", True, "[the user attached an image, /p/x.png, but you cannot see images; "
                  "say so if the question needs it]\nq"),
    ("local", True, "q"),                   # the image itself goes in (_stream_local_llm)
])
def test_attached_image_reaches_each_backend(settings, backend, sees, want):
    host = TurnHost(backend, sees)
    host._turn_image = "/p/x.png"
    host._run_turn("q", threading.Event(), None, None)
    assert host.sent == [(backend, want)]


def test_attached_image_follows_a_fallback_to_claude(settings):
    host = TurnHost("local", sees=True, local_ok=False)
    host._turn_image = "/p/x.png"
    host._run_turn("q", threading.Event(), None, None)
    assert host.sent == [("local", "q"),
                         ("claude", "[the user attached an image: /p/x.png -- read the file to see it]\nq")]


def test_no_image_no_note(settings):
    host = TurnHost("claude")
    host._run_turn("q", threading.Event(), None, None)
    assert host.sent == [("claude", "q")]


class _Capture:
    def mute(self, m):
        pass

    def flush(self):
        pass


def _ask_host():
    host = va.VoiceAssistant.__new__(va.VoiceAssistant)
    host.logger = logging.getLogger("test-look-ask")
    host.is_active = False
    host.is_processing = False
    host._recording = False
    host.backend = "local"
    host.capture = _Capture()
    host._set_status = lambda s, **kw: None
    host._router_judge = lambda text, typed: None
    host._router_finish = lambda: None
    return host


def test_control_ask_carries_the_image_for_the_turn_only(settings, tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(PNG)
    host = _ask_host()
    seen = []
    host._query_and_speak = lambda text, abort, speak: seen.append((text, host._turn_image)) or "ok"
    res = asyncio.run(host._control_command({"cmd": "ask", "text": "what is this", "image": str(img)}))
    assert res["ok"] and seen == [("what is this", str(img))]
    assert host._turn_image is None, "cleared when the turn is over"
    asyncio.run(host._control_command({"cmd": "ask", "text": "and this"}))
    assert seen[-1] == ("and this", None)


def test_control_ask_refuses_a_bad_image_before_the_turn(settings, tmp_path):
    (tmp_path / "notes.png").write_text("not a picture")
    host = _ask_host()
    host._query_and_speak = lambda *a: pytest.fail("no turn for a bad image")
    res = asyncio.run(host._control_ask("what is this", False, 1.0, str(tmp_path / "notes.png")))
    assert not res["ok"] and "not an image" in res["error"]


# --- the assistant CLI ----------------------------------------------------------------------

def _cli():
    loader = importlib.machinery.SourceFileLoader("assistant_cli", str(ROOT / "assistant"))
    spec = importlib.util.spec_from_loader("assistant_cli", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def _run_cli(monkeypatch, argv, tty=False):
    cli = _cli()
    sent = []
    monkeypatch.setattr(cli, "request", lambda payload: sent.append(payload) or {"ok": True, "reply": "ok"})
    monkeypatch.setattr(sys, "argv", ["assistant"] + argv)
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: tty, read=lambda: ""))
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    code = cli.main()
    return code, sent


def test_cli_image_sends_the_whole_path(settings, monkeypatch, tmp_path):
    (tmp_path / "x.png").write_bytes(PNG)
    monkeypatch.chdir(tmp_path)
    code, sent = _run_cli(monkeypatch, ["--image", "x.png", "what", "is", "this"])
    assert code == 0
    assert sent == [{"cmd": "ask", "text": "what is this", "speak": False,
                     "image": str((tmp_path / "x.png").resolve())}]


def test_cli_image_alone_asks_what_it_is(settings, monkeypatch, tmp_path):
    (tmp_path / "x.png").write_bytes(PNG)
    code, sent = _run_cli(monkeypatch, ["--image", str(tmp_path / "x.png")], tty=True)
    assert sent[0]["text"] == "What is in this image?" and sent[0]["image"].endswith("x.png")


def test_cli_missing_image(settings, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    with pytest.raises(SystemExit):
        _run_cli(monkeypatch, ["--image", str(tmp_path / "nope.png"), "what is this"])


def test_cli_without_image_is_unchanged(settings, monkeypatch):
    code, sent = _run_cli(monkeypatch, ["hello", "there"])
    assert sent == [{"cmd": "ask", "text": "hello there", "speak": False}]
