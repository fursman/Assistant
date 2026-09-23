"""End of turn (voice_assistant.py: _record_until_silence, SpeechDetector.scan,
_parse_checkpoints, _tail, _save_turn).

Silence is timed in samples from the end of the last speech window, so a
checkpoint fires within one read of the time it names, and no sum of 0.2 s
floats can push the 1.6 s checkpoint to 1.8 s or the timeout past itself.
Each smart-turn call starts SMART_TURN_LEAD before its checkpoint.

No audio hardware: the capture hands out silent chunks, the VAD follows a
script of speech intervals in seconds, and smart-turn is a fake.

    .venv/bin/python -m pytest tests
"""
import json
import logging
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import voice_assistant as va  # noqa: E402

SR = va.SAMPLE_RATE


class Capture:
    def __init__(self):
        self.reads = 0
        self.overflows = 0
        self.dropped = 0

    def read(self, duration, timeout=None):
        self.reads += 1
        return np.zeros(int(round(SR * duration)), dtype=np.float32)

    def describe(self):
        return "fake capture"


class ScriptedVAD:
    """Speech during the given [start, end) intervals, in seconds from the
    start of the recording (after the pre-roll), judged per 512-sample window
    like the real detector."""

    def __init__(self, speech):
        self.speech = speech
        self.pos = 0

    def _is_speech(self, sample):
        t = sample / SR
        return any(a <= t < b for a, b in self.speech)

    def scan(self, chunk):
        n = len(chunk) // 512 * 512
        end = None
        for w in range(0, n, 512):
            if self._is_speech(self.pos + w):
                end = w + 512
        self.pos += len(chunk)
        return n, end


class SmartTurn:
    """Answers from a function of the recording position (seconds) at the time
    of the call; records when it was called."""

    def __init__(self, host, prob):
        self.host = host
        self.prob = prob
        self.called_at = []

    def predict(self, audio):
        # From the audio it was handed (runs on the smart-turn worker, so the
        # recorder may have read on since): everything after the pre-roll.
        t = (len(audio) - PRE_ROLL) / SR
        self.called_at.append(t)
        p = self.prob(t) if callable(self.prob) else self.prob
        return p, 0.001


class STT:
    pass


class Host(va.VoiceAssistant):
    def __init__(self):
        self.logger = logging.getLogger("test-end-of-turn")
        self._router = None
        self._router_prefetch = None
        self.is_active = True
        self._abort_event = threading.Event()
        self.capture = Capture()
        self.stt = STT()


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(va, "RECORD_CHUNK_DURATION", 0.064)
    monkeypatch.setattr(va, "SMART_TURN_CHECKPOINTS", [(0.30, 0.9), (0.70, 0.7), (1.10, 0.4), (1.60, 0.2)])
    monkeypatch.setattr(va, "SILENCE_TIMEOUT", 2.0)
    monkeypatch.setattr(va, "SMART_TURN_LEAD", 0.0)
    monkeypatch.setattr(va, "MAX_RECORD_DURATION", 20)
    monkeypatch.setattr(va, "ROUTER_PREFETCH", False)


def _run(speech, prob):
    host = Host()
    host.vad = ScriptedVAD(speech)
    host.smart_turn = SmartTurn(host, prob)
    host._record_until_silence(np.zeros(PRE_ROLL, dtype=np.float32))
    return host


CHUNK = 0.064
PRE_ROLL = int(SR * 0.4)


def test_default_checkpoints_and_parsing():
    assert va._parse_checkpoints("0.7:0.75, 0.3:0.9,junk,1.0:") == [(0.3, 0.9), (0.7, 0.75), (1.0, 0.5)]
    assert va._parse_checkpoints("") == list(va._DEFAULT_CHECKPOINTS)
    assert va._DEFAULT_CHECKPOINTS[0] == (0.45, 0.90)


def test_finished_turn_ends_at_the_first_checkpoint_measured_from_the_last_word(env):
    """Speech stops 1.01 s in, part-way through a read. The first checkpoint
    fires within one read of 0.30 s after that, not at a chunk edge after it."""
    host = _run([(0.0, 1.01)], 0.99)
    eot = host._last_eot
    assert eot["ended_by"] == "smart-turn"
    assert 0.30 <= eot["silence_s"] < 0.30 + CHUNK
    assert len(host.smart_turn.called_at) == 1
    # position of the call: last speech window end + silence
    speech_end = np.ceil(1.01 * SR / 512) * 512 / SR
    assert host.smart_turn.called_at[0] - speech_end == pytest.approx(eot["silence_s"], abs=1e-6)


def test_every_checkpoint_and_the_timeout_fire_on_time(env):
    """smart-turn never satisfied: four checks, each within one read of its
    checkpoint, then the 2.0 s timeout -- not 2.1, not 2.6."""
    host = _run([(0.0, 0.5)], 0.01)
    eot = host._last_eot
    assert eot["ended_by"] == "timeout"
    assert 2.0 <= eot["silence_s"] < 2.0 + CHUNK
    times = [c[0] for c in eot["checks"]]
    assert len(times) == 4
    for got, (want, _) in zip(times, va.SMART_TURN_CHECKPOINTS):
        assert want <= got < want + CHUNK, (got, want)


def test_later_checkpoint_accepts_a_lower_score(env):
    host = _run([(0.0, 0.5)], 0.5)
    eot = host._last_eot
    assert eot["ended_by"] == "smart-turn"
    assert 1.10 <= eot["silence_s"] < 1.10 + CHUNK
    assert [c[2] for c in eot["checks"]] == [0.9, 0.7, 0.4]


def test_the_call_starts_lead_seconds_before_its_checkpoint(env, monkeypatch):
    monkeypatch.setattr(va, "SMART_TURN_LEAD", 0.12)
    host = _run([(0.0, 1.0)], 0.99)
    eot = host._last_eot
    assert eot["ended_by"] == "smart-turn"
    assert 0.30 <= eot["silence_s"] < 0.30 + CHUNK, "the answer is still read at the checkpoint"
    called_silence = host.smart_turn.called_at[0] - np.ceil(1.0 * SR / 512) * 512 / SR
    assert 0.18 <= called_silence < 0.18 + CHUNK, "but asked for 0.12 s earlier"


def test_an_early_call_is_discarded_when_speech_resumes(env, monkeypatch):
    """The early call saw a pause that sounded finished (p=0.99); the speaker
    went on before the checkpoint. That answer must not end the turn: the
    next pause is judged afresh (p=0.01 from then on) and runs to the
    timeout."""
    monkeypatch.setattr(va, "SMART_TURN_LEAD", 0.12)
    host = _run([(0.0, 1.0), (1.25, 2.0)], lambda t: 0.99 if t < 1.25 else 0.01)
    eot = host._last_eot
    assert eot["ended_by"] == "timeout"
    assert host.smart_turn.called_at[0] < 1.25, "the early call was made in the first pause"
    assert all(c[1] == 0.01 for c in eot["checks"]), "and its answer was never used"


def test_resumed_speech_resets_the_checkpoints(env):
    host = _run([(0.0, 0.5), (0.9, 1.5)], lambda t: 0.8 if t < 0.9 else 0.95)
    eot = host._last_eot
    assert eot["ended_by"] == "smart-turn"
    # first pause: 0.8 < 0.9 at 0.30; speech resumes at 0.9 before the 0.70 check
    assert [c[1] for c in eot["checks"]] == [0.8, 0.95]
    assert 0.30 <= eot["checks"][-1][0] < 0.30 + CHUNK


def test_no_smart_turn_means_the_timeout_alone(env):
    host = Host()
    host.vad = ScriptedVAD([(0.0, 0.3)])
    host.smart_turn = None
    host._record_until_silence(np.zeros(SR // 4, dtype=np.float32))
    assert host._last_eot["ended_by"] == "timeout"
    assert host._last_eot["checks"] == []


def test_smart_turn_failure_falls_back_to_the_timeout(env):
    host = Host()
    host.vad = ScriptedVAD([(0.0, 0.3)])

    class Broken:
        def predict(self, audio):
            raise RuntimeError("onnx fell over")
    host.smart_turn = Broken()
    host._record_until_silence(np.zeros(SR // 4, dtype=np.float32))
    assert host.smart_turn is None
    assert host._last_eot["ended_by"] == "timeout"


# --- the VAD ---------------------------------------------------------------------

class CountingModel:
    """Silero stand-in: 'speech' when the window's mean is above 0.5."""

    def __init__(self):
        self.windows = 0

    def __call__(self, chunk, sr):
        self.windows += 1
        assert chunk.shape[0] == 512
        return _Item(1.0 if float(chunk.mean()) > 0.5 else 0.0)

    def reset_states(self):
        pass


class _Item:
    def __init__(self, v):
        self.v = v

    def item(self):
        return self.v


def test_vad_windows_are_contiguous_across_reads():
    """3200-sample reads used to lose their last 128 samples each; now the
    remainder is carried, so two reads are 12 windows and the 13th starts
    exactly where the 12th ended."""
    m = CountingModel()
    det = va.SpeechDetector(m)
    det.probabilities(np.zeros(3200, np.float32))
    det.probabilities(np.zeros(3200, np.float32))
    assert m.windows == 12
    det.probabilities(np.zeros(3200, np.float32))
    assert m.windows == 18, "9600 samples = 18.75 windows"
    det.reset()
    det.probabilities(np.zeros(100, np.float32))
    assert m.windows == 18, "reset drops the carried remainder"


def test_scan_reports_where_speech_ended():
    det = va.SpeechDetector(CountingModel())
    x = np.zeros(2048, np.float32)
    x[:1024] = 1.0                      # windows 0 and 1 are speech
    assert det.scan(x) == (2048, 1024)
    assert det.scan(np.zeros(1024, np.float32)) == (1024, None)


def test_tail_takes_the_end_without_joining_everything():
    frames = [np.full(100, i, np.float32) for i in range(10)]
    t = va._tail(frames, 250)
    assert t.shape == (250,)
    assert t[0] == 7 and t[-1] == 9
    assert va._tail(frames, 5000).shape == (1000,)
    assert va._tail([], 10).shape == (0,)


# --- saved turns -------------------------------------------------------------------

def test_saved_turns_are_opt_in(monkeypatch, tmp_path):
    monkeypatch.setattr(va, "SAVE_TURNS", False)
    monkeypatch.setattr(va, "SAVE_TURNS_DIR", tmp_path)
    Host()._save_turn(np.zeros(SR, np.float32), "hello")
    time.sleep(0.1)
    assert list(tmp_path.iterdir()) == []


def test_saved_turn_is_a_wav_and_a_json_line_and_old_ones_go(monkeypatch, tmp_path):
    monkeypatch.setattr(va, "SAVE_TURNS", True)
    monkeypatch.setattr(va, "SAVE_TURNS_DIR", tmp_path)
    monkeypatch.setattr(va, "SAVE_TURNS_KEEP", 2)
    host = Host()
    host._last_eot = {"ended_by": "timeout", "checks": [(0.3, 0.01, 0.9)], "silence_s": 2.0}
    for i in range(3):
        host._save_turn(np.full(SR // 2, 0.1, np.float32), f"turn {i}")
        deadline = time.time() + 3
        while time.time() < deadline and len((tmp_path / "turns.jsonl").read_text().splitlines()
                                                 if (tmp_path / "turns.jsonl").exists() else []) < i + 1:
            time.sleep(0.01)
        time.sleep(1.05 if i < 2 else 0.05)     # distinct second-resolution stems, oldest first
    lines = [json.loads(l) for l in (tmp_path / "turns.jsonl").read_text().splitlines()]
    assert [l["transcript"] for l in lines] == ["turn 0", "turn 1", "turn 2"]
    assert lines[0]["ended_by"] == "timeout" and lines[0]["seconds"] == 0.5
    wavs = sorted(tmp_path.glob("*.wav"))
    assert [w.name for w in wavs] == [lines[1]["wav"], lines[2]["wav"]]
    with wave.open(str(wavs[-1])) as f:
        assert f.getframerate() == SR and f.getnframes() == SR // 2
