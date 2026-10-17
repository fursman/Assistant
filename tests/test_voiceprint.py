"""Voice unlock (voiceprint.py): the phrase check, learning the voiceprint,
and what one attempt at the lock screen does.

    .venv/bin/python -m pytest tests
"""
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import voiceprint as vpm  # noqa: E402

LOG = logging.getLogger("test")


def _vec(*head):
    v = np.zeros(256, np.float32)
    v[:len(head)] = head
    return v / np.linalg.norm(v)


USER = _vec(1.0, 0.1)
STRANGER = _vec(0.0, 0.0, 1.0)


class FakeEmbedder:
    def __init__(self, emb):
        self.emb = emb

    def embed(self, audio):
        return self.emb


def test_phrase_matches_in_order_with_slips():
    phrase = "maple river seven candle"
    assert vpm.phrase_matches("Unlock, maple river seven candle.", phrase)
    assert vpm.phrase_matches("maple rivers seven candles", phrase)
    assert not vpm.phrase_matches("river maple seven candle", phrase)
    assert not vpm.phrase_matches("maple river candle", phrase)
    assert not vpm.phrase_matches("unlock", phrase)
    assert not vpm.phrase_matches("anything", "")


def test_numbers_count_as_words():
    assert vpm.phrase_matches("tiger 7 harbor", "tiger seven harbor")


def test_unlock_attempt_is_the_word_or_the_phrase():
    assert not vpm.is_unlock_attempt("unlock the screen", "")
    assert not vpm.is_unlock_attempt("unlock", "maple river seven candle")
    assert vpm.is_unlock_attempt("maple river seven candle", "maple river seven candle")
    assert not vpm.is_unlock_attempt("what's the weather", "maple river seven candle")


def test_voiceprint_learns_and_leaves_out_strangers(tmp_path, monkeypatch):
    monkeypatch.setattr(vpm, "ENROLL_MIN", 3)
    monkeypatch.setattr(vpm, "ENROLL_SAMPLES", 5)
    vp = vpm.Voiceprint(tmp_path / "vp.npz")
    for _ in range(3):
        assert vp.consider(USER)[0]
    assert vp.ready
    added, score = vp.consider(STRANGER)
    assert not added and score < vpm.ENROLL_OUTLIER
    vp.consider(USER)
    vp.consider(USER)
    assert vp.full and not vp.consider(USER)[0]
    assert vpm.Voiceprint(tmp_path / "vp.npz").count == 5     # saved


def _locked(tmp_path, monkeypatch, voice, phrase="maple river seven candle"):
    monkeypatch.setattr(vpm, "PHRASE_PATH", tmp_path / "unlock-phrase")
    monkeypatch.setattr(vpm, "ENROLL_MIN", 2)
    (tmp_path / "unlock-phrase").write_text(f"4242 {phrase}\n")
    vp = vpm.Voiceprint(tmp_path / "vp.npz")
    vp.consider(USER)
    vp.consider(USER)
    signals = []
    monkeypatch.setattr(vpm.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    return vpm.VoiceUnlock(FakeEmbedder(voice), vp, LOG), signals


def test_right_words_right_voice_unlocks(tmp_path, monkeypatch):
    vu, signals = _locked(tmp_path, monkeypatch, USER)
    assert vu.attempt(None, "maple river seven candle", 4242) == "Welcome back."
    assert signals == [(4242, 10)]
    assert not (tmp_path / "unlock-phrase").exists()     # retired


def test_wrong_voice_stays_locked_and_counts(tmp_path, monkeypatch):
    vu, signals = _locked(tmp_path, monkeypatch, STRANGER)
    assert "don't recognize" in vu.attempt(None, "maple river seven candle", 4242)
    assert signals == [] and vu._failures[4242] == 1
    assert not (tmp_path / "unlock-phrase").exists()


def test_wrong_words_stay_locked_without_counting(tmp_path, monkeypatch):
    vu, signals = _locked(tmp_path, monkeypatch, USER)
    assert "words on the screen" in vu.attempt(None, "unlock", 4242)
    assert signals == [] and 4242 not in vu._failures


def test_phrase_of_another_lock_is_ignored(tmp_path, monkeypatch):
    vu, signals = _locked(tmp_path, monkeypatch, USER)
    assert "can't see" in vu.attempt(None, "maple river seven candle", 9999)
    assert signals == []


def test_too_many_strangers_turn_it_off(tmp_path, monkeypatch):
    vu, signals = _locked(tmp_path, monkeypatch, STRANGER)
    for _ in range(vpm.MAX_ATTEMPTS):
        (tmp_path / "unlock-phrase").write_text("4242 maple river seven candle\n")
        vu.attempt(None, "maple river seven candle", 4242)
    vu.embedder = FakeEmbedder(USER)
    (tmp_path / "unlock-phrase").write_text("4242 maple river seven candle\n")
    assert "too many tries" in vu.attempt(None, "maple river seven candle", 4242)
    assert signals == []


def test_not_learned_yet(tmp_path, monkeypatch):
    monkeypatch.setattr(vpm, "PHRASE_PATH", tmp_path / "unlock-phrase")
    vu = vpm.VoiceUnlock(FakeEmbedder(USER), vpm.Voiceprint(tmp_path / "none.npz"), LOG)
    assert "haven't learned" in vu.attempt(None, "unlock", 4242)


# --- voice-only --------------------------------------------------------------------

def test_voice_only_verdict_holds_short_turns_to_the_outlier_bar(tmp_path, monkeypatch):
    vp = vpm.Voiceprint(tmp_path / "vp.npz")
    for _ in range(3):
        vp.consider(USER)
    near = _vec(1.0, 0.1, 2.4)       # about 0.4: a short "yes" in the user's voice
    assert vpm.ENROLL_OUTLIER < vp.score(near) < vpm.VOICE_ONLY_THRESHOLD
    long_s, short_s = vpm.ENROLL_MIN_SECONDS + 1, vpm.ENROLL_MIN_SECONDS - 0.5
    assert vpm.voice_only_verdict(vp, USER, long_s)[0]
    ok, score, bar = vpm.voice_only_verdict(vp, STRANGER, long_s)
    assert not ok and bar == vpm.VOICE_ONLY_THRESHOLD
    assert not vpm.voice_only_verdict(vp, near, long_s)[0]
    ok, _, bar = vpm.voice_only_verdict(vp, near, short_s)
    assert ok and bar == vpm.ENROLL_OUTLIER
    assert not vpm.voice_only_verdict(vp, STRANGER, short_s)[0], "short, but clearly someone else"


def test_more_than_the_phrase_is_not_an_unlock(tmp_path, monkeypatch):
    """The phrase in a stranger's voice after a recording of the user's: the
    words are there, the voice would score, and it still stays locked."""
    vu, signals = _locked(tmp_path, monkeypatch, USER)
    said = "so anyway I was telling you about the weekend maple river seven candle"
    assert "Just the words" in vu.attempt(None, said, 4242)
    assert signals == [] and not (tmp_path / "unlock-phrase").exists()
    (tmp_path / "unlock-phrase").write_text("4242 maple river seven candle\n")
    long_turn = np.zeros(int((vpm.UNLOCK_MAX_SECONDS + 5) * vpm.SAMPLE_RATE), np.float32)
    assert "Just the words" in vu.attempt(long_turn, "maple river seven candle", 4242)
    (tmp_path / "unlock-phrase").write_text("4242 maple river seven candle\n")
    assert vu.attempt(None, "okay maple river seven candle", 4242) == "Welcome back."
