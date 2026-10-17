"""Voiceprint: who is speaking, and the two things that depend on it: the
spoken unlock, and voice-only (turns in anyone else's voice are ignored).

A speaker-embedding model (WeSpeaker ResNet34, trained on VoxCeleb, 26 MB
ONNX) turns a turn's audio into a 256-number vector. Turns spoken by the same
person point the same way, so the cosine between a new turn and the average of
the user's own turns says how much it sounds like them.

The voiceprint learns from ordinary turns: while the screen is unlocked, each
turn long enough to judge is added until ENROLL_SAMPLES are stored. Once a few
are in, a turn that looks nothing like the rest (the TV, a visitor) is left
out, so one stray voice cannot drag the average.

The unlock has two parts, both required:
  - the words: the lock screen shows a random phrase (contrib/voice-unlock/
    voice-unlock-phrase writes it, hyprlock displays it); the transcript has
    to contain it. A new phrase after every attempt, so a recording of an
    earlier unlock is useless.
  - the voice: the turn has to score at least THRESHOLD against the voiceprint.

This is a convenience for a machine at home, not real security: a good clone
of the user's voice reading the phrase off the screen would pass.

Voice-only needs no phrase: once the voiceprint is ready, a turn scoring under
VOICE_ONLY_THRESHOLD is dropped before it reaches a model (voice_only_verdict).

    .venv/bin/python voiceprint.py score some.wav ...   # score WAVs (16 kHz mono)
    .venv/bin/python voiceprint.py enroll some.wav ...  # add WAVs to the voiceprint
    .venv/bin/python voiceprint.py status
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000

# Where the assistant keeps its status and pid file, which does not follow
# XDG_STATE_HOME: the service and hyprlock's label script must agree on it.
STATE_DIR = Path.home() / ".local/state/voice-assistant"
MODEL_PATH = Path(os.getenv(
    "VOICE_ASSISTANT_VOICEPRINT_MODEL",
    str(Path.home() / ".cache/voice-assistant/wespeaker_en_voxceleb_resnet34.onnx")))
# k2-fsa's copy carries the sample rate in its metadata; "recongition" is
# their spelling of the release tag.
MODEL_URL = os.getenv(
    "VOICE_ASSISTANT_VOICEPRINT_URL",
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx")
STORE_PATH = STATE_DIR / "voiceprint.npz"
PHRASE_PATH = STATE_DIR / "unlock-phrase"      # "<hyprlock pid> word word word"
STATUS_PATH = STATE_DIR / "voice-unlock"       # one line for the lock screen

ENROLL_SAMPLES = int(os.getenv("VOICE_ASSISTANT_VOICEPRINT_SAMPLES", "20"))
# Ready once this many are stored (the rest keep coming in).
ENROLL_MIN = int(os.getenv("VOICE_ASSISTANT_VOICEPRINT_MIN", "8"))
ENROLL_MIN_SECONDS = float(os.getenv("VOICE_ASSISTANT_VOICEPRINT_MIN_SECONDS", "1.5"))
# A candidate scoring below this against the voiceprint so far is someone else.
ENROLL_OUTLIER = float(os.getenv("VOICE_ASSISTANT_VOICEPRINT_OUTLIER", "0.35"))
# Same-speaker turns score around 0.7-0.85 with this model, other people
# mostly under 0.35 (two similar Kokoro voices reached 0.6 on a short phrase). The log line of every attempt shows the score, for tuning.
THRESHOLD = float(os.getenv("VOICE_ASSISTANT_VOICE_UNLOCK_THRESHOLD", "0.6"))
MAX_ATTEMPTS = int(os.getenv("VOICE_ASSISTANT_VOICE_UNLOCK_ATTEMPTS", "5"))
# An attempt is the phrase and little else. A long turn could pair the
# phrase in a stranger's voice with a recording of the user's, which would
# carry the voice score; neither fits in this.
UNLOCK_EXTRA_WORDS = 3
UNLOCK_MAX_SECONDS = 10.0
# Voice-only's bar, lower than the unlock's: a wrongly ignored turn means
# saying it again, not an open screen. The user's own long turns have scored
# as low as 0.54; other people 0.1-0.2, TTS voices from a speaker about 0.
VOICE_ONLY_THRESHOLD = float(os.getenv("VOICE_ASSISTANT_VOICE_ONLY_THRESHOLD", "0.45"))


class SpeakerEmbedder:
    """WeSpeaker ONNX: 80-bin Kaldi fbank, mean-normalised, in; 256-d out."""

    def __init__(self, model_path=MODEL_PATH, url=MODEL_URL, threads=2):
        self.model_path = Path(model_path)
        self.url = url
        self.threads = threads
        self.session = None
        self._lock = threading.Lock()

    def download(self, logger=None):
        if self.model_path.exists() and self.model_path.stat().st_size > 1_000_000:
            return
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.model_path.with_suffix(".part")
        if logger:
            logger.info(f"Downloading the voiceprint model from {self.url}")
        # A timeout, unlike urlretrieve: a stalled download would otherwise
        # hold the loader (and a spoken "only listen to me") forever.
        try:
            with urllib.request.urlopen(self.url, timeout=30) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            tmp.replace(self.model_path)
        finally:
            tmp.unlink(missing_ok=True)

    def load(self, logger=None):
        import onnxruntime as ort
        self.download(logger)
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = self.threads
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=so, providers=["CPUExecutionProvider"])
        self._in = self.session.get_inputs()[0].name
        self.embed(np.zeros(SAMPLE_RATE, np.float32))    # warm up
        return self

    @staticmethod
    def features(audio16k):
        import torch
        import torchaudio.compliance.kaldi as kaldi
        x = torch.from_numpy(np.asarray(audio16k, dtype=np.float32).ravel() * 32768.0)
        feats = kaldi.fbank(x.unsqueeze(0), num_mel_bins=80, frame_length=25,
                            frame_shift=10, dither=0.0, sample_frequency=SAMPLE_RATE,
                            window_type="hamming", energy_floor=0.0)
        feats = feats - feats.mean(dim=0, keepdim=True)
        return feats.numpy()[None].astype(np.float32)

    def embed(self, audio16k) -> np.ndarray:
        """Unit-length embedding of one stretch of 16 kHz float audio."""
        feats = self.features(audio16k)
        with self._lock:
            emb = self.session.run(None, {self._in: feats})[0][0]
        return emb / (np.linalg.norm(emb) + 1e-9)


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


class Voiceprint:
    """The stored embeddings of the user's turns, and scores against them."""

    def __init__(self, path=STORE_PATH):
        self.path = Path(path)
        self.embeddings = np.zeros((0, 256), np.float32)
        self._lock = threading.Lock()
        try:
            with np.load(self.path) as data:
                self.embeddings = np.asarray(data["embeddings"], np.float32)
        except (OSError, KeyError, ValueError):
            pass

    @property
    def count(self) -> int:
        return len(self.embeddings)

    @property
    def ready(self) -> bool:
        return self.count >= ENROLL_MIN

    @property
    def full(self) -> bool:
        return self.count >= ENROLL_SAMPLES

    def score(self, emb) -> float:
        """Cosine against the average of the stored turns; -1 when empty."""
        if not self.count:
            return -1.0
        return float(np.dot(_unit(self.embeddings.mean(axis=0)), emb))

    def consider(self, emb) -> tuple:
        """Add one turn's embedding unless full or an outlier.
        Returns (added, score)."""
        with self._lock:
            s = self.score(emb)
            if self.full:
                return False, s
            # The first few are trusted: nothing to compare them with yet.
            if self.count >= 3 and s < ENROLL_OUTLIER:
                return False, s
            self.embeddings = np.vstack([self.embeddings, emb[None].astype(np.float32)])
            self.save()
            return True, s

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp.npz")
        np.savez(tmp, embeddings=self.embeddings)
        os.chmod(tmp, 0o600)
        tmp.replace(self.path)

    def status_line(self) -> str:
        if self.ready:
            return "ready"
        return f"learning {self.count}/{ENROLL_MIN}"


def voice_only_verdict(voiceprint, emb, seconds) -> tuple:
    """Voice-only's decision on one turn: (passes, score, bar).

    A turn shorter than the voiceprint learns from ("yes", "stop") is held to
    the outlier bar instead: a second of audio scores lower even in the
    user's own voice, while other voices stay near zero at any length.
    Dropping those would mean repeating every short answer."""
    score = voiceprint.score(emb)
    bar = VOICE_ONLY_THRESHOLD if seconds >= ENROLL_MIN_SECONDS else ENROLL_OUTLIER
    return score >= bar, score, bar


# --- the spoken unlock -------------------------------------------------------

_NUMBER_WORDS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
                 "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
                 "10": "ten"}


def _words(text: str) -> list:
    text = re.sub(r"[^a-z0-9' ]+", " ", (text or "").lower().replace("-", " "))
    return [_NUMBER_WORDS.get(w, w) for w in text.split()]


def phrase_matches(transcript: str, phrase: str, min_ratio=0.8) -> bool:
    """Every word of the phrase, in order, each close enough to a word of the
    transcript (STT spelling slips: "colour"/"color", "gray"/"grey")."""
    import difflib
    want, said = _words(phrase), _words(transcript)
    if not want:
        return False
    i = 0
    for w in said:
        if i < len(want) and (w == want[i] or difflib.SequenceMatcher(
                None, w, want[i]).ratio() >= min_ratio):
            i += 1
    return i == len(want)


def hyprlock_pid():
    """The running hyprlock, or None when the screen is not locked."""
    try:
        out = subprocess.run(["pidof", "-s", "hyprlock"], capture_output=True,
                             text=True, timeout=2).stdout.strip()
        return int(out) if out else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def current_phrase(pid) -> str:
    """The phrase the lock screen shows for this hyprlock, or ""."""
    try:
        owner, _, phrase = PHRASE_PATH.read_text().strip().partition(" ")
        return phrase if owner == str(pid) else ""
    except OSError:
        return ""


def retire_phrase():
    """Make the lock screen draw a new phrase (it does within a second)."""
    try:
        PHRASE_PATH.unlink()
    except OSError:
        pass


def write_status(line):
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        if line is None:
            STATUS_PATH.unlink(missing_ok=True)
        else:
            STATUS_PATH.write_text(line + "\n")
    except OSError:
        pass


def is_unlock_attempt(transcript: str, phrase: str) -> bool:
    """Something close to the phrase on screen. Saying "unlock" is not an
    attempt: only reading the phrase is, so a locked screen stays silent for
    anyone who doesn't."""
    return bool(phrase) and phrase_matches(transcript, phrase, min_ratio=0.7)


class VoiceUnlock:
    """Decides one locked turn. The assistant calls attempt(); it says what to
    speak and has already unlocked when that is the outcome."""

    def __init__(self, embedder, voiceprint, logger):
        self.embedder = embedder
        self.voiceprint = voiceprint
        self.logger = logger
        self._failures = {}   # hyprlock pid -> failed attempts

    def attempt(self, audio, transcript, pid) -> str:
        phrase = current_phrase(pid)
        fails = self._failures.get(pid, 0)
        if fails >= MAX_ATTEMPTS:
            self.logger.info(f"Voice unlock: refused, {fails} failed attempts on this lock")
            return "Voice unlock is off after too many tries. Please type your password."
        if not self.voiceprint.ready:
            return ("I haven't learned your voice yet. "
                    f"I have {self.voiceprint.count} of {ENROLL_MIN} samples.")
        if not phrase:
            return "I can't see an unlock phrase on the screen yet. Try again in a moment."
        if (len(_words(transcript)) > len(_words(phrase)) + UNLOCK_EXTRA_WORDS
                or (audio is not None and len(audio) > UNLOCK_MAX_SECONDS * SAMPLE_RATE)):
            self.logger.info("Voice unlock: refused, more than the phrase was said")
            retire_phrase()
            return "Just the words on the screen, please. Here's a new phrase."
        score = self.voiceprint.score(self.embedder.embed(audio))
        words_ok = phrase_matches(transcript, phrase)
        voice_ok = score >= THRESHOLD
        self.logger.info(
            f"Voice unlock: words {'ok' if words_ok else 'WRONG'} (wanted \"{phrase}\"), "
            f"voice {score:.2f} vs {THRESHOLD:.2f} {'ok' if voice_ok else 'REJECTED'}")
        # A fresh phrase after every attempt: nothing said here works twice.
        retire_phrase()
        if words_ok and voice_ok:
            self._failures.pop(pid, None)
            try:
                os.kill(pid, 10)     # SIGUSR1: hyprlock unlocks
            except OSError as e:
                self.logger.warning(f"Voice unlock: could not signal hyprlock ({e})")
                return "That matched, but I couldn't unlock the screen."
            return "Welcome back."
        if not voice_ok:
            # Only a voice that isn't the user's counts against the limit:
            # the user misreading a word is not an attack. Only this lock's
            # count is kept; an earlier hyprlock's is history.
            self._failures = {pid: fails + 1}
            return "Sorry, I don't recognize your voice."
        return "Say the words on the screen. Here's a new phrase."


def _read_wav(path):
    import wave
    with wave.open(str(path), "rb") as f:
        if f.getframerate() != SAMPLE_RATE or f.getnchannels() != 1 or f.getsampwidth() != 2:
            raise SystemExit(f"{path}: need 16 kHz mono 16-bit")
        return np.frombuffer(f.readframes(f.getnframes()), "<i2").astype(np.float32) / 32768.0


def main(argv):
    if not argv or argv[0] not in ("score", "enroll", "status"):
        raise SystemExit(__doc__)
    vp = Voiceprint()
    if argv[0] == "status":
        print(f"{vp.count} samples, {vp.status_line()}, threshold {THRESHOLD} "
              f"(voice-only {VOICE_ONLY_THRESHOLD})")
        return
    emb = SpeakerEmbedder().load()
    for p in argv[1:]:
        e = emb.embed(_read_wav(p))
        if argv[0] == "enroll":
            added, s = vp.consider(e)
            print(f"{p}: {s:.3f} {'added' if added else 'skipped'}")
        else:
            print(f"{p}: {vp.score(e):.3f}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
