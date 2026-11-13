"""One turn at a time, and a harness turn that overflows its context keeps
what it did (voice_assistant.py: _control_ask / _recording,
_speak_unsolicited, _stream_dsh's overflow retry, _dsh_retry_note).

    .venv/bin/python -m pytest tests
"""
import asyncio
import logging
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import voice_assistant as va  # noqa: E402


# --- a typed question waits for a recording in progress -----------------------

class Capture:
    def __init__(self):
        self.muted = []

    def mute(self, m):
        self.muted.append(m)

    def flush(self):
        pass


class VAD:
    def reset(self):
        pass


def _bare_host():
    host = va.VoiceAssistant.__new__(va.VoiceAssistant)
    host.logger = logging.getLogger("test-turn-safety")
    host.is_active = True
    host.is_processing = False
    host._recording = False
    host.capture = Capture()
    host.vad = VAD()
    host.statuses = []
    host._set_status = lambda s, **kw: host.statuses.append(s)
    return host


def test_typed_question_waits_for_the_recording_then_runs():
    host = _bare_host()
    ran = []
    host._router_judge = lambda text, typed: None
    host._router_finish = lambda: None
    host._query_and_speak = lambda text, abort, speak: ran.append(text) or "ok"
    host.backend = "local"

    async def scenario():
        host._recording = True
        task = asyncio.create_task(host._control_ask("hello", False, 5.0))
        await asyncio.sleep(0.2)
        assert ran == [], "no typed turn while a voice turn is being recorded"
        assert host.capture.muted == [], "and the microphone was left alone"
        host._recording = False
        return await task

    res = asyncio.run(scenario())
    assert res["ok"] and ran == ["hello"]
    assert host.is_processing is False


def test_typed_question_gives_up_if_the_recording_never_ends():
    host = _bare_host()
    host._recording = True
    res = asyncio.run(host._control_ask("hello", False, 0.1))
    assert not res["ok"] and "busy" in res["error"]


def test_unsolicited_reply_is_a_turn_of_its_own(monkeypatch):
    monkeypatch.setattr(va, "TTS_TAIL_GATE", 0.0)
    host = _bare_host()
    said = []

    def say(text):
        assert host.is_processing, "nothing else may start while it plays"
        assert host.capture.muted == [True], "and the microphone is shut"
        said.append(text)
    host._say = say
    asyncio.run(host._speak_unsolicited("your build finished"))
    assert said == ["your build finished"]
    assert host.capture.muted == [True, False] and host.is_processing is False


def test_unsolicited_reply_is_not_spoken_over_a_turn():
    host = _bare_host()
    host._say = lambda text: pytest.fail("spoke over a turn")
    host._recording = True
    asyncio.run(host._speak_unsolicited("x"))
    host._recording, host.is_processing = False, True
    asyncio.run(host._speak_unsolicited("x"))


# --- harness context overflow keeps the tool work --------------------------------

def test_tool_summary_and_retry_note():
    assert va._dsh_tool_summary("Bash", {"command": "ls   -la\n/tmp"}) == "Bash ls -la /tmp"
    assert va._dsh_tool_summary("Search", {"query": "x" * 200}).endswith("...")
    assert va._dsh_tool_summary("Tool", {}) == "Tool"
    note = va._dsh_retry_note(["Bash ls", "Bash ls", "Fetch https://a"])
    assert "3 tool calls" in note and "Bash ls; Fetch https://a" in note
    assert "did use tools" in note


class FakeSession:
    def __init__(self):
        self.prompts = []
        self.remembered = []
        self.rotated = 0
        self.alive = True
        self.prompt_tokens = 0

    def turn(self, prompt, on_event):
        self.prompts.append(prompt)
        if len(self.prompts) == 1:
            for cmd in ("grim /tmp/shot.png", "grim /tmp/shot.png", "identify /tmp/shot.png"):
                on_event({"type": "tool/call", "data": {"name": "bash",
                                                        "arguments": f'{{"command": "{cmd}"}}'}})
            on_event({"type": "turn/end", "data": {"reason": {
                "kind": "error", "error": {"message": "request exceeds the available context size"}}}})
            return
        on_event({"type": "assistant/chunk", "data": {"chunk": {"type": "text-delta",
                                                                "text": "Your screen shows a terminal."}}})
        on_event({"type": "turn/end", "data": {"reason": {"kind": "done"}}})

    def rotate(self):
        self.rotated += 1

    def remember(self, user, assistant, backend="dsh"):
        self.remembered.append((user, assistant))

    def needs_rotation(self):
        return False

    def stop(self):
        pass

    def reset(self):
        pass


def test_overflow_retry_is_told_what_ran_and_the_record_keeps_it(monkeypatch):
    monkeypatch.setattr(va, "DSH_MAX_REPEATS", 10)
    host = va.VoiceAssistant.__new__(va.VoiceAssistant)
    host.logger = logging.getLogger("test-turn-safety")
    host._dsh = FakeSession()
    host._ensure_dsh_session = lambda: True
    host._dsh_turn_active = False
    host._dsh_reset_pending = False
    host._assistant_text = ""
    host._thinking_text = ""
    host._sentence_queue = None
    host._notify_tool_use = lambda label, detail: None
    host._flush_sentences = lambda final=False: None
    host._dsh_error = ""
    assert host._stream_dsh("what is on my screen", threading.Event())
    s = host._dsh.prompts
    assert s[0] == "what is on my screen"
    assert s[1].startswith("what is on my screen\n\n[Note: your first attempt")
    assert "3 tool calls" in s[1] and "grim /tmp/shot.png; " in s[1]
    assert host._dsh.rotated == 1
    user, record = host._dsh.remembered[-1]
    assert user == "what is on my screen", "the ledger keeps what the user said"
    assert record.startswith("[Ran: ") and record.endswith("Your screen shows a terminal.")


# --- speech that another program transcribed is screened like the microphone ----

class _Verdict:
    def __init__(self, junk, typed):
        self.junk, self.typed, self.outcome = junk, typed, {}
        # The real rule (turn_router.Verdict.drop): spoken input only.
        self.drop = (not typed) and junk >= 0.8

    def log_line(self):
        return f"Router: junk={self.junk}"


def _screened_host(junk):
    host = _bare_host()
    host.backend = "claude"
    host.judged, host.ran, host.shown, host.recorded = [], [], [], []

    def judge(text, typed):
        host.judged.append((text, typed))
        return None if junk is None else _Verdict(junk, typed)
    host._router_judge = judge
    host._router_finish = lambda: None
    host._router_screening = lambda: True
    host._router_record = host.recorded.append
    host._append_transcript = lambda role, text: host.shown.append((role, text))
    host._query_and_speak = lambda text, abort, speak: host.ran.append(text) or "ok"
    return host


WATCH = "[context: asked from the Galaxy Watch called watch two; the answer is shown there]\n"


def test_speech_from_another_program_is_dropped_when_the_judge_calls_it_junk():
    host = _screened_host(0.91)
    res = asyncio.run(host._control_ask(WATCH + "and then she put it on the shelf", False, 1.0,
                                        spoken=True))
    assert res["ok"] and res["dropped"] and res["reply"] == "" and res["junk"] == 0.91
    assert host.ran == [], "a dropped turn never reaches a model"
    assert host.judged == [("and then she put it on the shelf", False)], \
        "judged as speech, and without the caller's context line"
    assert host.recorded and host.recorded[0].outcome == {"dropped": True, "screen": True}
    assert host.shown and host.shown[0][0] == "system" and "Not for me" in host.shown[0][1]
    assert host.is_processing is False and host.statuses[-1] == "ready"


def test_speech_from_another_program_that_passes_is_answered_with_its_context():
    host = _screened_host(0.04)
    res = asyncio.run(host._control_ask(WATCH + "what is the tank level", False, 1.0, spoken=True))
    assert res["ok"] and not res.get("dropped") and res["reply"] == "ok"
    assert host.ran == [WATCH.strip() + "\nwhat is the tank level"], "the model still gets the context line"


def test_typed_input_is_never_dropped_whatever_the_judge_says():
    host = _screened_host(0.99)
    res = asyncio.run(host._control_ask("asdf qwer", False, 1.0))
    assert res["ok"] and not res.get("dropped") and host.ran == ["asdf qwer"]
    assert host.judged == [("asdf qwer", True)]


def test_no_judge_means_the_spoken_turn_just_goes_ahead():
    host = _screened_host(None)          # not configured, down, or too slow: no opinion
    res = asyncio.run(host._control_ask(WATCH + "hello there", False, 1.0, spoken=True))
    assert res["ok"] and not res.get("dropped") and res["reply"] == "ok"
