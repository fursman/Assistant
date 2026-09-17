"""The turn router's overlap with the end-of-turn wait (voice_assistant.py:
_router_prefetch_start / _router_take_prefetch / _router_judge_spoken, and
the checkpoint in _record_until_silence that fires the prefetch).

No audio hardware and no network: the router is a fake that records what it
was asked and sleeps, the STT adapter is a fake with a settable partial(),
and the recording loop is driven with silent chunks and a scripted VAD.

    .venv/bin/python -m pytest tests
"""
import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import voice_assistant as va  # noqa: E402


# --- fakes -------------------------------------------------------------------

class FakeVerdict:
    latency_ms = 700.0


class FakeRouter:
    """Records every judge() call; answers after `delay` seconds."""

    def __init__(self, delay=0.05, fail=False):
        self.calls = []
        self.delay = delay
        self.fail = fail
        self.verdict = FakeVerdict()

    def judge(self, text, typed=False, previous=None, timeout=None):
        self.calls.append((text, typed))
        self.timeouts = getattr(self, "timeouts", []) + [timeout]
        if self.fail:
            raise RuntimeError("judge exploded")
        time.sleep(self.delay)
        return self.verdict


class FakeCapture:
    def __init__(self):
        self.reads = 0
        self.overflows = 0
        self.dropped = 0

    def read(self, duration, timeout=None):
        self.reads += 1
        return np.zeros(int(va.SAMPLE_RATE * duration), dtype=np.float32)

    def describe(self):
        return "fake capture"


class FakeVAD:
    """active() follows a script of booleans, then stays silent."""

    def __init__(self, pattern):
        self.pattern = list(pattern)
        self.i = 0

    def active(self, chunk):
        v = self.pattern[self.i] if self.i < len(self.pattern) else False
        self.i += 1
        return v


class FakeSTT:
    is_streaming = True

    def __init__(self, partial_text, host=None, events=None):
        self.partial_text = partial_text
        self.host = host
        self.events = events if events is not None else []
        self.partial_calls = []

    def begin_utterance(self):
        pass

    def feed(self, chunk, sample_rate):
        pass

    def partial(self):
        # When in the recording loop the host's capture count says which
        # chunk the call came on.
        self.partial_calls.append(self.host.capture.reads if self.host else None)
        self.events.append("partial")
        if isinstance(self.partial_text, Exception):
            raise self.partial_text
        if isinstance(self.partial_text, list):
            # a transcript that grows: one entry per call, the last one repeating
            i = min(len(self.partial_calls) - 1, len(self.partial_text) - 1)
            return self.partial_text[i]
        return self.partial_text


class FakeSmartTurn:
    def __init__(self, probs, events=None):
        self.probs = list(probs)
        self.calls = 0
        self.events = events if events is not None else []

    def predict(self, audio):
        p = self.probs[min(self.calls, len(self.probs) - 1)]
        self.calls += 1
        self.events.append("predict")
        return p, 0.001


class Host(va.VoiceAssistant):
    """A VoiceAssistant with only the state the prefetch path touches."""

    def __init__(self, router, vocab=None):
        self.logger = logging.getLogger("test-router-prefetch")
        self._router = router
        self._router_prefetch = None
        self._vocab = vocab or (lambda t: t)
        # for _record_until_silence
        self.is_active = True
        self._abort_event = threading.Event()
        self.capture = FakeCapture()
        self.events = []

    def _router_previous(self):
        return None


@pytest.fixture
def loop_env(monkeypatch):
    """Deterministic recording-loop constants: 0.2 s chunks, checkpoints at
    0.35 s and 0.7 s of silence, a 2.5 s hard timeout."""
    monkeypatch.setattr(va, "ROUTER_PREFETCH", True)
    monkeypatch.setattr(va, "ROUTER_PREFETCH_EARLY", True)
    monkeypatch.setattr(va, "RECORD_CHUNK_DURATION", 0.2)
    monkeypatch.setattr(va, "SMART_TURN_CHECKPOINTS", [(0.35, 0.9), (0.7, 0.75)])
    monkeypatch.setattr(va, "SILENCE_TIMEOUT", 2.5)
    monkeypatch.setattr(va, "MAX_RECORD_DURATION", 10)
    monkeypatch.setattr(va, "ROUTER_TIMEOUT", 1.5)


def _record(host, vad_pattern, partial_text, probs):
    host.vad = FakeVAD(vad_pattern)
    host.stt = FakeSTT(partial_text, host=host, events=host.events)
    host.smart_turn = FakeSmartTurn(probs, events=host.events)
    pre = np.zeros(int(va.SAMPLE_RATE * 0.2), dtype=np.float32)
    audio = host._record_until_silence(pre)
    assert isinstance(audio, np.ndarray)
    return host


def _wait_prefetch(host, timeout=2.0):
    pf = host._router_prefetch
    if pf is not None:
        assert pf.done.wait(timeout)
    return pf


# --- (a) the prefetch fires once, at the first checkpoint -------------------

def test_prefetch_fires_once_at_first_checkpoint(loop_env):
    """Two speech chunks (plus the pre-roll), then silence. With 0.2 s chunks
    the first checkpoint (0.35 s) is crossed on the second silent chunk, the
    fourth read overall. smart-turn calls it unfinished there and finished
    at 0.7 s; the prefetch must fire at the first checkpoint, before
    smart-turn is consulted. At the end of the turn the transcript is looked
    at again, and since nothing new was heard the first call stands alone."""
    router = FakeRouter(delay=0.02)
    host = Host(router)
    _record(host, [True, True, False, False, False, False], "what time is it", [0.1, 0.99])
    pf = _wait_prefetch(host)

    assert host.stt.partial_calls == [4, 6], "partial() at 0.35 s, and again when the turn ended"
    assert host.events == ["partial", "predict", "predict", "partial"]
    assert host.smart_turn.calls == 2, "the later checkpoint was reached"
    assert pf is not None and pf.raw == "what time is it"
    assert router.calls == [("what time is it", False)], "one router call, spoken"


def test_default_is_one_prefetch_at_the_end_of_the_turn(loop_env, monkeypatch):
    """With the early checkpoint call off (the default), the only judgment is
    the one made when the turn completes, on the transcript as it stands; it
    carries the longer prefetch timeout."""
    monkeypatch.setattr(va, "ROUTER_PREFETCH_EARLY", False)
    monkeypatch.setattr(va, "ROUTER_PREFETCH_TIMEOUT", 3.0)
    router = FakeRouter(delay=0.02)
    host = Host(router)
    _record(host, [True, True, False, False, False, False], "what time is it", [0.1, 0.99])
    pf = _wait_prefetch(host)
    assert host.stt.partial_calls == [6], "partial() only when the turn ended"
    assert host.events == ["predict", "predict", "partial"]
    assert pf is not None and pf.raw == "what time is it"
    assert router.calls == [("what time is it", False)]
    assert router.timeouts == [3.0]
    assert host._router_judge_spoken("What time is it?") is router.verdict
    assert len(router.calls) == 1


def test_prefetch_refreshed_at_end_of_turn_when_words_arrived(loop_env):
    """The decoder lagged: at 0.35 s it had two words, by the end of the turn
    it had five. The end-of-turn look replaces the prefetch, and the final
    transcript then reuses the second call."""
    router = FakeRouter(delay=0.02)
    host = Host(router)
    _record(host, [True, True, False, False, False, False],
            ["what time", "what time is it in tokyo"], [0.1, 0.99])
    pf = _wait_prefetch(host)
    assert pf is not None and pf.raw == "what time is it in tokyo"
    time.sleep(0.1)
    assert router.calls == [("what time", False), ("what time is it in tokyo", False)]
    verdict = host._router_judge_spoken("What time is it in Tokyo?")
    assert verdict is router.verdict
    assert len(router.calls) == 2, "the refreshed prefetch was reused"


def test_prefetch_kept_at_end_of_turn_when_it_covers_the_words(loop_env):
    """One more word by the end of the turn is within the prefix rule, so the
    first call stands and no second one is made."""
    router = FakeRouter(delay=0.02)
    host = Host(router)
    _record(host, [True, True, False, False, False, False],
            ["what time is it", "what time is it now"], [0.1, 0.99])
    _wait_prefetch(host)
    time.sleep(0.05)
    assert router.calls == [("what time is it", False)]


def test_prefetch_refreshed_on_silence_timeout(loop_env):
    router = FakeRouter(delay=0.02)
    host = Host(router)
    # smart-turn never satisfied: the 2.5 s timeout ends the turn
    _record(host, [True, True] + [False] * 20, ["what", "what is the tallest mountain"], [0.1, 0.1, 0.1])
    _wait_prefetch(host)
    time.sleep(0.05)
    assert router.calls == [("what", False), ("what is the tallest mountain", False)]


def test_prefix_rule():
    ok = va.VoiceAssistant._router_prefix_ok
    assert ok("what time is it", "what time is it")
    assert ok("what time is", "what time is it")                 # 12/15 chars
    assert ok("what time is", "what time is it now")             # 63% but only 2 words short
    assert not ok("what time is", "what time is it now please")  # 3 words short
    assert not ok("what time", "what is the time")               # not a prefix
    assert not ok("", "what time is it")
    assert ok("stop", "stop the music")


def test_router_bypassed_when_claude_is_the_backend(monkeypatch):
    monkeypatch.setattr(va, "ROUTER_PREFETCH", True)
    monkeypatch.setattr(va, "ROUTER_BACKENDS", frozenset({"local", "dsh"}))
    router = FakeRouter(delay=0.01)
    host = Host(router)
    host.backend = "claude"
    assert host._router_active() is False
    assert host._router_judge("what time is it", False) is None
    assert host._router_judge("what time is it", True) is None
    assert host._router_prefetch_start("what time is it") is False
    assert host._router_judge_spoken("what time is it") is None
    assert router.calls == [], "Claude's turns never reach the judge"
    host.backend = "local"
    assert host._router_active() is True
    assert host._router_judge("what time is it", True) is router.verdict
    monkeypatch.setattr(va, "ROUTER_BACKENDS", frozenset({"local", "dsh", "claude"}))
    host.backend = "claude"
    assert host._router_active() is True, "opt back in with VOICE_ASSISTANT_ROUTER_BACKENDS"


def test_prefetch_kept_when_speech_resumes(loop_env):
    """Speech after the checkpoint does not start a second prefetch: the
    first stays and is the one on the host."""
    router = FakeRouter(delay=0.02)
    host = Host(router)
    _record(host, [True, True, False, False, True, True, False, False, False, False],
            "what time is it", [0.1, 0.1, 0.99])
    pf = _wait_prefetch(host)
    assert pf is not None and pf.raw == "what time is it"
    assert len(router.calls) == 1


def test_no_prefetch_on_empty_partial(loop_env):
    router = FakeRouter()
    host = Host(router)
    _record(host, [True, True, False, False], "", [0.99])
    assert host._router_prefetch is None
    assert router.calls == []


def test_prefetch_start_refuses_a_second(monkeypatch):
    monkeypatch.setattr(va, "ROUTER_PREFETCH", True)
    router = FakeRouter(delay=0.02)
    host = Host(router)
    assert host._router_prefetch_start("first") is True
    assert host._router_prefetch_start("second") is False
    assert host._router_prefetch_start("") is False
    _wait_prefetch(host)
    assert router.calls == [("first", False)]


# --- (b) a matching final transcript reuses the verdict -----------------------

def test_matching_final_reuses_prefetched_verdict(monkeypatch):
    monkeypatch.setattr(va, "ROUTER_PREFETCH", True)
    monkeypatch.setattr(va, "ROUTER_TIMEOUT", 1.5)
    router = FakeRouter(delay=0.4)
    host = Host(router)
    assert host._router_prefetch_start("what time is it")
    time.sleep(0.5)                      # the end-of-turn wait, in effect
    t0 = time.monotonic()
    verdict = host._router_judge_spoken("What time is it?")
    took = time.monotonic() - t0
    assert verdict is router.verdict
    assert router.calls == [("what time is it", False)], "no second router call"
    assert took < 0.2, f"reused the verdict without waiting for a fresh call ({took:.2f}s)"
    assert host._router_prefetch is None, "the prefetch is consumed"


def test_match_waits_for_a_prefetch_still_running(monkeypatch):
    """Final transcript arrives while the prefetch is mid-flight: wait for it
    rather than ask again."""
    monkeypatch.setattr(va, "ROUTER_PREFETCH", True)
    monkeypatch.setattr(va, "ROUTER_TIMEOUT", 1.5)
    router = FakeRouter(delay=0.3)
    host = Host(router)
    assert host._router_prefetch_start("turn the lights off")
    verdict = host._router_judge_spoken("turn the lights off")
    assert verdict is router.verdict
    assert len(router.calls) == 1


def test_match_goes_through_the_same_vocabulary_repair(monkeypatch):
    """_transcribe repairs the final transcript with VocabularyCorrector; the
    prefetch text gets the same repair, so the two still compare equal."""
    monkeypatch.setattr(va, "ROUTER_PREFETCH", True)
    monkeypatch.setattr(va, "ROUTER_TIMEOUT", 1.5)
    router = FakeRouter(delay=0.02)
    repair = lambda t: t.replace("fersman", "Fursman")  # noqa: E731
    host = Host(router, vocab=repair)
    assert host._router_prefetch_start("call andrew fersman")
    verdict = host._router_judge_spoken(repair("call andrew fersman."))
    assert verdict is router.verdict
    assert router.calls == [("call andrew Fursman", False)], "the router saw the repaired text"


def test_router_norm():
    n = va.VoiceAssistant._router_norm
    assert n("What  time is it?") == n("what time is it")
    assert n("Stop.") == n("stop")
    assert n("what time is") != n("what time is it")
    assert n(None) == "" and n("") == ""


# --- (c) a mismatching final transcript judges again, once ------------------

def test_prefix_prefetch_is_reused_when_it_covers_most_of_the_final(monkeypatch):
    # The streaming decoder lags by about a word: a prefetch that is a prefix covering at
    # least ROUTER_PREFETCH_MIN_MATCH of the final transcript was judged on the same request.
    monkeypatch.setattr(va, "ROUTER_PREFETCH", True)
    monkeypatch.setattr(va, "ROUTER_TIMEOUT", 1.5)
    monkeypatch.setattr(va, "ROUTER_PREFETCH_MIN_MATCH", 0.8)
    router = FakeRouter(delay=0.05)
    host = Host(router)
    assert host._router_prefetch_start("what time is")      # the last word not decoded yet
    verdict = host._router_judge_spoken("What time is it?")  # "what time is" = 12 of 15 chars
    assert verdict is router.verdict
    time.sleep(0.1)
    assert router.calls == [("what time is", False)]
    assert host._router_prefetch is None


def test_mismatching_final_judges_synchronously_once(monkeypatch):
    monkeypatch.setattr(va, "ROUTER_PREFETCH", True)
    monkeypatch.setattr(va, "ROUTER_TIMEOUT", 1.5)
    monkeypatch.setattr(va, "ROUTER_PREFETCH_MIN_MATCH", 0.8)
    router = FakeRouter(delay=0.05)
    host = Host(router)
    assert host._router_prefetch_start("what time is")
    verdict = host._router_judge_spoken("Delete everything in my downloads folder.")  # different words
    assert verdict is router.verdict
    # The abandoned prefetch is waited out before the second call, so both are counted.
    time.sleep(0.1)
    assert router.calls == [("what time is", False), ("Delete everything in my downloads folder.", False)]
    assert host._router_prefetch is None


def test_no_prefetch_judges_synchronously_once(monkeypatch):
    monkeypatch.setattr(va, "ROUTER_PREFETCH", True)
    router = FakeRouter(delay=0.01)
    host = Host(router)
    verdict = host._router_judge_spoken("hello there")
    assert verdict is router.verdict
    assert router.calls == [("hello there", False)]


# --- (d) prefetch off ----------------------------------------------------------

def test_prefetch_off_never_prefetches(loop_env, monkeypatch):
    monkeypatch.setattr(va, "ROUTER_PREFETCH", False)
    router = FakeRouter(delay=0.01)
    host = Host(router)
    _record(host, [True, True, False, False, False, False], "what time is it", [0.1, 0.99])
    assert host.stt.partial_calls == [], "partial() never consulted"
    assert host._router_prefetch is None
    assert router.calls == []
    assert host._router_prefetch_start("what time is it") is False
    verdict = host._router_judge_spoken("what time is it")
    assert verdict is router.verdict
    assert router.calls == [("what time is it", False)], "exactly one synchronous call"


# --- fail-open ------------------------------------------------------------------

def test_prefetch_fails_open_when_the_router_raises(monkeypatch):
    monkeypatch.setattr(va, "ROUTER_PREFETCH", True)
    monkeypatch.setattr(va, "ROUTER_TIMEOUT", 1.5)
    router = FakeRouter(fail=True)
    host = Host(router)
    assert host._router_prefetch_start("what time is it")
    verdict = host._router_judge_spoken("what time is it")
    assert verdict is None, "no opinion, as a failing synchronous call gives"
    assert len(router.calls) == 1, "the router had its one chance"


def test_partial_failure_is_logged_and_the_turn_proceeds(loop_env, caplog):
    router = FakeRouter()
    host = Host(router)
    with caplog.at_level(logging.WARNING, logger="test-router-prefetch"):
        _record(host, [True, True, False, False], RuntimeError("no stream"), [0.99])
    assert host._router_prefetch is None
    assert router.calls == []
    assert any("Router prefetch skipped" in r.message for r in caplog.records)
    # ...and the final transcript is judged exactly as before.
    assert host._router_judge_spoken("what time is it") is router.verdict
    assert router.calls == [("what time is it", False)]


def test_vocab_failure_in_prefetch_falls_back_to_raw_text(monkeypatch):
    monkeypatch.setattr(va, "ROUTER_PREFETCH", True)
    monkeypatch.setattr(va, "ROUTER_TIMEOUT", 1.5)
    router = FakeRouter(delay=0.02)

    def broken(_t):
        raise ValueError("bad vocabulary")
    host = Host(router, vocab=broken)
    assert host._router_prefetch_start("what time is it")
    verdict = host._router_judge_spoken("what time is it")
    assert verdict is router.verdict
    assert router.calls == [("what time is it", False)]
