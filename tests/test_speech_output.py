"""Speech out (voice_assistant.py: _next_boundary, ToolMarkupGate, the Pocket
adapter's stream(), _tts_worker, PcmPlayer's dry count and _log_playback).

No model and no audio device: Pocket is a fake module, the player a fake
that records what was written and when.

    .venv/bin/python -m pytest tests
"""
import logging
import queue
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import voice_assistant as va  # noqa: E402

nb = va.VoiceAssistant._next_boundary


# --- where a unit of speech ends ----------------------------------------------------

def test_first_unit_takes_a_later_clause_break_when_the_first_is_too_early():
    text = "Ah, well the short answer to that, is that it depends on the day"
    end = nb(None, text, True)
    assert end is not None and text[:end] == "Ah, well the short answer to that,"


def test_first_unit_still_prefers_a_full_stop():
    text = "Yes. It is, as far as I can tell, running"
    assert text[:nb(None, text, True)] == "Yes."


def test_first_unit_clause_break_needs_four_words():
    assert nb(None, "Yes, sure, ok", True) is None
    assert nb(None, "Right, so here it is, the plan", True) == len("Right, so here it is,")


def test_later_units_wait_for_a_sentence():
    assert nb(None, "and then, after that, we", False) is None
    text = "It ran fine. Next"
    assert text[:nb(None, text, False)] == "It ran fine."


def test_long_opening_without_breaks_splits_where_a_speaker_breathes():
    text = " ".join(["word"] * 20)
    end = nb(None, text, True)
    assert end is not None and 5 <= len(text[:end].split()) <= 13


# --- the first sentence of a reply, when long, in two units -----------------------------

LONG_FIRST = ("The evening is settling in over the island tonight, and rain should "
              "reach us from the southwest by Thursday. More later.")


def test_long_first_sentence_is_cut_once_at_its_first_pause(monkeypatch):
    monkeypatch.setattr(va, "TTS_SPLIT_FIRST", True)
    end = nb(None, LONG_FIRST, True)
    assert LONG_FIRST[:end] == "The evening is settling in over the island tonight,"
    rest = LONG_FIRST[end:]
    # the rest of that sentence is one unit; nothing else is split
    assert rest[:nb(None, rest, False)].strip() == \
        "and rain should reach us from the southwest by Thursday."


def test_split_first_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(va, "TTS_SPLIT_FIRST", False)
    end = nb(None, LONG_FIRST, True)
    assert LONG_FIRST[:end].endswith("by Thursday.")


def test_short_first_sentence_stays_whole(monkeypatch):
    monkeypatch.setattr(va, "TTS_SPLIT_FIRST", True)
    text = "Sure, I can help with that right now. Next"
    assert text[:nb(None, text, True)] == "Sure, I can help with that right now."


def test_later_sentences_are_never_split(monkeypatch):
    monkeypatch.setattr(va, "TTS_SPLIT_FIRST", True)
    assert LONG_FIRST[:nb(None, LONG_FIRST, False)].endswith("by Thursday.")


def test_split_skips_a_pause_with_too_few_words_before_it(monkeypatch):
    monkeypatch.setattr(va, "TTS_SPLIT_FIRST", True)
    text = "Well, the ferry from Fulford leaves at four thirty, and it is usually busy then."
    assert text[:nb(None, text, True)] == "Well, the ferry from Fulford leaves at four thirty,"


def test_split_needs_a_rest_that_can_be_ready_in_time(monkeypatch):
    monkeypatch.setattr(va, "TTS_SPLIT_FIRST", True)
    # "It is late," leaves a rest five times its length: too long to make
    # while it plays, so the sentence stays whole (no later pause either).
    text = ("It is late tonight, so the rain that is moving in from the southwest "
            "should reach the island by about midnight. Then")
    assert text[:nb(None, text, True)].endswith("midnight.")
    # a later pause that balances it is taken instead
    text = ("It is late tonight, the rain is moving in from the southwest, and it "
            "should reach the island by about midnight. Then")
    assert text[:nb(None, text, True)] == \
        "It is late tonight, the rain is moving in from the southwest,"


def test_split_never_leaves_a_stub(monkeypatch):
    monkeypatch.setattr(va, "TTS_SPLIT_FIRST", True)
    text = "The forecast for the whole island this evening says rain later, I think. Then"
    assert text[:nb(None, text, True)].endswith("I think.")


# --- tool markup never reaches speech --------------------------------------------------

def test_markup_gate_passes_prose_and_stops_at_an_opener_split_across_deltas():
    g = va.ToolMarkupGate()
    out = g.feed("Let me check that. ") + g.feed("<tool") + g.feed("_call>{\"name\": \"x\"}")
    assert out == "Let me check that. "
    assert g.tripped and g.feed("more") == "" and g.close() == ""


def test_markup_gate_releases_a_harmless_angle_bracket():
    g = va.ToolMarkupGate()
    out = g.feed("if a <") + g.feed(" b then")
    out += g.close()
    assert out == "if a < b then"
    g = va.ToolMarkupGate()
    assert g.feed("x <b") + g.close() == "x <b"


# --- Pocket: stream() under one lock ----------------------------------------------------

class FakeTTSModel:
    sample_rate = 24000
    active = 0
    overlap = False

    @classmethod
    def load_model(cls):
        return cls()

    def get_state_for_audio_prompt(self, voice):
        return {}

    def _enter(self):
        FakeTTSModel.active += 1
        if FakeTTSModel.active > 1:
            FakeTTSModel.overlap = True

    def generate_audio_stream(self, state, text, **kw):
        import torch
        self._enter()
        try:
            for _ in range(5):
                time.sleep(0.02)
                yield torch.full((1920,), 0.1)
        finally:
            FakeTTSModel.active -= 1

    def generate_audio(self, state, text, **kw):
        import torch
        return torch.cat(list(self.generate_audio_stream(state, text)))


@pytest.fixture
def pocket(monkeypatch):
    monkeypatch.setitem(sys.modules, "pocket_tts", types.SimpleNamespace(TTSModel=FakeTTSModel))
    FakeTTSModel.active = 0
    FakeTTSModel.overlap = False
    host = va.VoiceAssistant.__new__(va.VoiceAssistant)
    host.logger = logging.getLogger("test-speech-output")
    assert va.VoiceAssistant._load_pocket(host)
    return host.kokoro


def test_pocket_stream_yields_as_it_decodes(pocket):
    t0 = time.time()
    it = pocket.stream("hello")
    first, sr = next(it)
    assert sr == 24000 and first.shape == (1920,) and first.dtype == np.float32
    assert time.time() - t0 < 0.08, "the first piece comes before the rest are made"
    rest = list(it)
    assert len(rest) == 4


def test_abandoned_stream_is_drained_before_the_next_synthesis(pocket):
    it = pocket.stream("one")
    next(it)
    it.close()                          # barge-in: nobody wants the rest
    samples, _ = pocket.create("two")   # must wait for the drain, not run beside it
    assert samples.shape == (1920 * 5,)
    assert not FakeTTSModel.overlap
    assert pocket._lock.acquire(timeout=1.0), "the lock came back"
    pocket._lock.release()


def test_pocket_logger_is_quiet(pocket):
    assert logging.getLogger("pocket_tts").level == logging.WARNING


# --- the TTS worker plays pieces as they arrive -------------------------------------

class FakePlayer:
    rate = 24000

    def __init__(self):
        self.writes = []            # (time, samples)
        self.dry = 0
        self.underruns = 0

    def write(self, samples):
        self.writes.append((time.time(), len(samples)))


class StreamingTTS:
    def __init__(self, pieces=6, gap=0.05):
        self.pieces = pieces
        self.gap = gap
        self.closed = 0
        self.made = 0

    def stream(self, text, voice=None, speed=None):
        try:
            for _ in range(self.pieces):
                time.sleep(self.gap)
                self.made += 1
                yield np.full(2400, 0.1, np.float32), 24000      # 0.1 s each
        finally:
            self.closed += 1

    def create(self, text, voice=None, speed=None):
        raise AssertionError("streaming engine: create() should not be used")


def _worker_host(tts):
    host = va.VoiceAssistant.__new__(va.VoiceAssistant)
    host.logger = logging.getLogger("test-speech-output")
    host.kokoro = tts
    host.player = FakePlayer()
    host.is_active = False
    host._first_audio_at = None
    host._turn_started_at = time.time()
    return host


def test_worker_starts_playing_before_the_unit_is_finished(monkeypatch):
    monkeypatch.setattr(va, "TTS_PREBUFFER_SECONDS", 0.2)
    monkeypatch.setattr(va, "TTS_STREAM", True)
    tts = StreamingTTS(pieces=6, gap=0.05)
    host = _worker_host(tts)
    q, abort = queue.Queue(), threading.Event()
    t = threading.Thread(target=host._tts_worker, args=(q, abort))
    t0 = time.time()
    t.start()
    q.put("A sentence of about six tenths of a second.")
    q.put(None)
    t.join(5)
    assert sum(n for _, n in host.player.writes) == 6 * 2400
    first_write = host.player.writes[0][0] - t0
    assert first_write < 0.2, f"cushion of 0.2 s = two pieces, not the whole unit ({first_write:.2f}s)"
    assert host._first_audio_at is not None
    assert tts.closed == 1


def test_worker_abort_mid_unit_stops_and_closes_the_stream(monkeypatch):
    monkeypatch.setattr(va, "TTS_PREBUFFER_SECONDS", 0.0)
    monkeypatch.setattr(va, "TTS_STREAM", True)
    tts = StreamingTTS(pieces=50, gap=0.02)
    host = _worker_host(tts)
    q, abort = queue.Queue(), threading.Event()
    t = threading.Thread(target=host._tts_worker, args=(q, abort))
    t.start()
    q.put("A long sentence.")
    time.sleep(0.15)
    abort.set()
    t.join(2)
    assert not t.is_alive()
    assert tts.made < 50 and tts.closed == 1


def test_worker_uses_create_when_streaming_is_off(monkeypatch):
    monkeypatch.setattr(va, "TTS_STREAM", False)
    monkeypatch.setattr(va, "TTS_PREBUFFER_SECONDS", 0.0)

    class Whole:
        def create(self, text, voice=None, speed=None):
            return np.zeros(4800, np.float32), 24000

        def stream(self, *a, **k):
            raise AssertionError("streaming is off")
    host = _worker_host(Whole())
    q, abort = queue.Queue(), threading.Event()
    q.put("one")
    q.put(None)
    host._tts_worker(q, abort)
    assert [n for _, n in host.player.writes] == [4800]


def test_worker_survives_a_synthesis_error(monkeypatch):
    monkeypatch.setattr(va, "TTS_STREAM", True)
    monkeypatch.setattr(va, "TTS_PREBUFFER_SECONDS", 0.0)

    class Flaky:
        calls = 0

        def stream(self, text, voice=None, speed=None):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise RuntimeError("bad text")
            yield np.zeros(2400, np.float32), 24000
    host = _worker_host(Flaky())
    q, abort = queue.Queue(), threading.Event()
    for x in ("bad", "good", None):
        q.put(x)
    host._tts_worker(q, abort)
    assert [n for _, n in host.player.writes] == [2400]


# --- a pool of boards: the next units synthesise while one plays ---------------------

class PoolTTS:
    """FallbackTts-like: a pool of `boards`, whole-sentence synthesis taking
    `per_char` seconds per character, one chunk per unit."""

    def __init__(self, boards=4, per_char=0.004, up=True):
        self.pool = types.SimpleNamespace(health=[types.SimpleNamespace(up=lambda: up)] * boards)
        self.remote_ok = True
        self.per_char = per_char
        self.lock = threading.Lock()
        self.active = self.peak = 0
        self.started, self.closed = [], 0
        self.speaking_remote = True

    def use_remote(self):
        return self.remote_ok

    def stream(self, text, voice=None, speed=None):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.started.append((time.time(), text))
        try:
            time.sleep(self.per_char * len(text))
            n = int(len(text) * 240)                    # 10 ms of audio per character
            yield np.full(n, float(len(text)), np.float32), 24000
        finally:
            with self.lock:
                self.active -= 1
                self.closed += 1


def _run_worker(host, units, abort=None):
    q, abort = queue.Queue(), abort or threading.Event()
    for u in units:
        q.put(u)
    q.put(None)
    t = threading.Thread(target=host._tts_worker, args=(q, abort))
    t.start()
    t.join(10)
    assert not t.is_alive()


def test_pool_synthesises_ahead_and_plays_in_order(monkeypatch):
    monkeypatch.setattr(va, "TTS_STREAM", True)
    monkeypatch.setattr(va, "TTS_PREBUFFER_SECONDS", 0.0)
    monkeypatch.setattr(va, "TTS_REMOTE_PREBUFFER_SECONDS", 0.0)
    monkeypatch.setattr(va, "REMOTE_TTS_AHEAD", 4)
    tts = PoolTTS(per_char=0.004)
    host = _worker_host(tts)
    written = []
    host.player.write = lambda x: written.append(float(x[0]))
    units = ["Short one,", "and then a much longer rest of the first sentence follows.", "Two.", "Three!"]
    _run_worker(host, units)
    assert written == [float(len(u)) for u in units]        # in order, each whole
    assert tts.peak >= 3                                      # synthesised at the same time
    starts = [t for t, _ in tts.started]
    assert max(starts) - min(starts) < 0.05                   # all asked for at once


def test_pool_ahead_is_bounded_by_boards_answering(monkeypatch):
    monkeypatch.setattr(va, "TTS_STREAM", True)
    monkeypatch.setattr(va, "TTS_PREBUFFER_SECONDS", 0.0)
    monkeypatch.setattr(va, "REMOTE_TTS_AHEAD", 4)
    tts = PoolTTS(boards=2, per_char=0.002)
    host = _worker_host(tts)
    _run_worker(host, ["one unit here."] * 5)
    assert tts.peak == 2 and tts.closed == 5


def test_one_at_a_time_without_the_pool_or_when_off(monkeypatch):
    monkeypatch.setattr(va, "TTS_STREAM", True)
    monkeypatch.setattr(va, "TTS_PREBUFFER_SECONDS", 0.0)
    monkeypatch.setattr(va, "REMOTE_TTS_AHEAD", 4)
    tts = PoolTTS(per_char=0.002)
    tts.remote_ok = False                        # this tier speaks on this machine
    host = _worker_host(tts)
    _run_worker(host, ["a unit.", "another unit.", "a third."])
    assert tts.peak == 1
    monkeypatch.setattr(va, "REMOTE_TTS_AHEAD", 1)
    tts = PoolTTS(per_char=0.002)
    host = _worker_host(tts)
    _run_worker(host, ["a unit.", "another unit.", "a third."])
    assert tts.peak == 1


def test_pool_abort_stops_everything(monkeypatch):
    monkeypatch.setattr(va, "TTS_STREAM", True)
    monkeypatch.setattr(va, "TTS_PREBUFFER_SECONDS", 0.0)
    monkeypatch.setattr(va, "REMOTE_TTS_AHEAD", 4)
    tts = PoolTTS(per_char=0.01)
    host = _worker_host(tts)
    q, abort = queue.Queue(), threading.Event()
    for u in ["x" * 30] * 4:
        q.put(u)
    t = threading.Thread(target=host._tts_worker, args=(q, abort))
    t.start()
    time.sleep(0.1)
    abort.set()
    t.join(2)
    assert not t.is_alive()
    assert host.player.writes == []


def test_pool_worker_survives_a_failed_unit(monkeypatch):
    monkeypatch.setattr(va, "TTS_STREAM", True)
    monkeypatch.setattr(va, "TTS_PREBUFFER_SECONDS", 0.0)
    monkeypatch.setattr(va, "REMOTE_TTS_AHEAD", 4)
    tts = PoolTTS(per_char=0.001)
    real = tts.stream

    def stream(text, voice=None, speed=None):
        if text == "bad":
            raise RuntimeError("bad text")
        yield from real(text, voice, speed)
    tts.stream = stream
    host = _worker_host(tts)
    written = []
    host.player.write = lambda x: written.append(float(x[0]))
    _run_worker(host, ["good one.", "bad", "good two."])
    assert written == [9.0, 9.0]


# --- playback gaps -------------------------------------------------------------------

def test_player_counts_running_dry_only_while_playing():
    p = va.PcmPlayer(pa=None, rate=24000, block=480)
    p._stream = object()                # write() only queues on an open stream
    p._callback(None, 480, None, 0)
    assert p.dry == 0, "an idle queue is not a gap"
    p.write(np.ones(700, np.float32))
    p._callback(None, 480, None, 0)
    assert p.dry == 0
    p._callback(None, 480, None, 0)     # 220 left: runs out mid-block
    assert p.dry == 1
    p._callback(None, 480, None, 0)
    assert p.dry == 1


def test_log_playback_reports_gaps_but_not_the_end(caplog):
    host = va.VoiceAssistant.__new__(va.VoiceAssistant)
    host.logger = logging.getLogger("test-speech-output")
    host.player = FakePlayer()
    host._first_audio_at = time.time()
    host._playback_mark = (10, 3)
    host.player.dry, host.player.underruns = 11, 3
    with caplog.at_level(logging.INFO, logger="test-speech-output"):
        host._log_playback()
    assert "Playback: 0 gaps mid-reply, 0 device underruns" in caplog.text
    host.player.dry, host.player.underruns = 14, 4
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="test-speech-output"):
        host._log_playback()
    assert "Playback: 3 gaps mid-reply, 1 device underruns" in caplog.text
    assert caplog.records[-1].levelno == logging.WARNING


def test_key_combos_are_said_as_key_names():
    say = va._prepare_for_speech
    assert say("Press Super+Ctrl+V now.") == "Press Super Control V now."
    assert say("Cmd + Shift + V pastes.") == "Command Shift V pastes."
    assert "C++" in say("C++ is fine.")


def test_a_long_reply_is_heard_to_the_end_before_listening_resumes():
    """Synthesis runs ahead, so a reply can be queued whole: the wait for it to
    be heard must cover what is queued, not stop at a fixed two minutes."""
    class Player:
        rate = 24000

        def __init__(self):
            self.timeouts = []

        def queued_seconds(self):
            return 150.0              # a long answer, made faster than it is spoken

        def drain(self, timeout=None):
            self.timeouts.append(timeout)
            return True

    host = types.SimpleNamespace(player=Player(), _sentence_queue=object(), _log_playback=lambda: None)
    q = queue.Queue()
    va.VoiceAssistant._end_tts(host, q, None)
    assert q.get_nowait() is None, "the worker is told the reply is over"
    assert host.player.timeouts[0] >= 150, "waits for all 150 s still queued"
