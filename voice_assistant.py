#!/usr/bin/env python3
"""
Hyprland Voice Assistant

Pipeline: VAD → Whisper STT → Claude Code LLM (streaming subprocess) → Kokoro TTS (streaming) → PipeWire playback

Thinking events stream as 🧠 notifications; response sentences stream to TTS as they arrive.
"""

import asyncio
import json
import logging
import logging.handlers
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import wave  # used by chime generation
from pathlib import Path
from typing import Optional

import numpy as np
import pyaudio
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, get_speech_timestamps


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
AUDIO_FORMAT = pyaudio.paInt16

VAD_CHUNK_DURATION = 0.5
SILENCE_TIMEOUT = 2.5
MAX_RECORD_DURATION = 30

WHISPER_MODEL = "small"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE = "float16"

TTS_VOICE = "af_heart"
TTS_SPEED = 1.0

CLAUDE_MODEL = os.getenv("VOICE_ASSISTANT_MODEL", "sonnet")
CLAUDE_VOICE_PROMPT = (
    "You are a voice assistant integrated into a Linux desktop (Hyprland on Wayland). "
    "The user speaks to you and hears your responses via text-to-speech. "
    "Keep responses concise and conversational — avoid code blocks, markdown formatting, "
    "and long lists unless specifically asked. Prefer natural spoken language. "
    "You have full access to the system and can run commands, read/write files, search the web, and more. "
    "Be helpful, proactive, and efficient. When the user asks you to do something on their system, just do it. "
    "Give brief confirmations rather than lengthy explanations.\n\n"
    "IMPORTANT — You are running INSIDE the voice assistant service. "
    "To stop listening / turn off voice mode / deactivate, run: "
    "kill -USR1 $(cat ~/.local/state/voice-assistant/voice-assistant.pid) "
    "— this toggles voice mode off gracefully. "
    "To start a new conversation, run: "
    "kill -USR2 $(cat ~/.local/state/voice-assistant/voice-assistant.pid) "
    "— NEVER stop or restart the voice-assistant systemd service, "
    "as that kills the process you are running inside of."
)

SESSION_FILE = Path.home() / ".local/state/voice-assistant/session_id"

# Voice commands that trigger a new session (checked after lowercasing + stripping trailing punctuation)
_NEW_SESSION_PHRASES = {
    "new conversation", "start a new conversation",
    "fresh start", "start fresh", "new session",
    "reset conversation", "clear conversation",
    "begin a new conversation",
}

CHIME_SAMPLE_RATE = 44100
CHIME_NOTE_DURATION = 0.2

# Regex to strip markdown formatting before TTS
_MD_STRIP = re.compile(
    r"\*\*(.+?)\*\*"   # **bold**
    r"|\*(.+?)\*"      # *italic*
    r"|__(.+?)__"      # __bold__
    r"|_(.+?)_"        # _italic_
    r"|`([^`]+)`"      # `code`
    r"|```[\s\S]*?```" # code blocks
    r"|\[([^\]]+)\]\([^)]+\)"  # [text](url)
    r"|^#{1,6}\s+"     # headings
    r"|^[-*]\s+"       # list bullets
    r"|^>\s+"          # blockquotes
    , re.MULTILINE
)

# Number words for speech normalization
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
         "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]

def _num_to_words(n: int) -> str:
    """Convert integer to English words (handles 0 to 999 billion)."""
    if n < 0:
        return "negative " + _num_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " and " + _num_to_words(rest))
    if n < 1000000:
        rest = n % 1000
        return _num_to_words(n // 1000) + " thousand" + ("" if rest == 0 else " " + _num_to_words(rest))
    if n < 1000000000:
        rest = n % 1000000
        return _num_to_words(n // 1000000) + " million" + ("" if rest == 0 else " " + _num_to_words(rest))
    if n < 1000000000000:
        rest = n % 1000000000
        return _num_to_words(n // 1000000000) + " billion" + ("" if rest == 0 else " " + _num_to_words(rest))
    return str(n)

# Pre-compiled abbreviation expansions for TTS
_ABBREVS = [
    (re.compile(r"\bvs\."), "versus"),
    (re.compile(r"\bvs\b"), "versus"),
    (re.compile(r"\betc\."), "et cetera"),
    (re.compile(r"\be\.g\.\s?"), "for example, "),
    (re.compile(r"\bi\.e\.\s?"), "that is, "),
    (re.compile(r"\bw/o\b"), "without"),
    (re.compile(r"\bw/\b"), "with"),
    (re.compile(r"\bGPUs\b"), "G P Us"),
    (re.compile(r"\bGPU\b"), "G P U"),
    (re.compile(r"\bCPUs\b"), "C P Us"),
    (re.compile(r"\bCPU\b"), "C P U"),
    (re.compile(r"\bVRAM\b"), "V ram"),
    (re.compile(r"\bAPIs\b"), "A P Is"),
    (re.compile(r"\bAPI\b"), "A P I"),
    (re.compile(r"\bLLMs\b"), "L L Ms"),
    (re.compile(r"\bLLM\b"), "L L M"),
    (re.compile(r"\bTTS\b"), "T T S"),
    (re.compile(r"\bSTT\b"), "S T T"),
    (re.compile(r"\bUI\b"), "U I"),
    (re.compile(r"\bURLs\b"), "U R Ls"),
    (re.compile(r"\bURL\b"), "U R L"),
    (re.compile(r"\bGB\b"), "gigabytes"),
    (re.compile(r"\bMB\b"), "megabytes"),
    (re.compile(r"\bTB\b"), "terabytes"),
    (re.compile(r"\bSSH\b"), "S S H"),
    (re.compile(r"\bNVLink\b"), "N V Link"),
    (re.compile(r"\bRTX\b"), "R T X"),
    (re.compile(r"\bRAM\b"), "ram"),
    (re.compile(r"\bEDA\b"), "E D A"),
    (re.compile(r"\bIMO\b"), "in my opinion"),
    (re.compile(r"\bSOTA\b"), "state of the art"),
    (re.compile(r"\bINT4\b"), "int four"),
    (re.compile(r"\bINT8\b"), "int eight"),
    (re.compile(r"\bFP16\b"), "F P sixteen"),
    (re.compile(r"\bFP32\b"), "F P thirty-two"),
]

def _prepare_for_speech(text: str) -> str:
    """Strip markdown and normalize text for natural TTS pronunciation."""
    # Step 1: Replace code blocks with spoken placeholder before stripping
    text = re.sub(r"```[\s\S]*?```", "code block", text)
    # Step 1b: Strip remaining markdown
    def _pick(m):
        for g in m.groups():
            if g is not None:
                return g
        return ""
    text = _MD_STRIP.sub(_pick, text).strip()

    # Step 2: Remove bare URLs
    text = re.sub(r"https?://\S+", "", text)

    # Step 3: Remove emoji
    text = re.sub(r"[\U0001f300-\U0001f9ff\U00002600-\U000027bf\U0000fe00-\U0000feff]", "", text)

    # Step 4: Currency — $1,234.56 → "one thousand two hundred thirty four dollars and fifty six cents"
    def _currency(m):
        sign = m.group(1) or ""
        whole = m.group(2).replace(",", "")
        cents = m.group(4) or ""
        prefix = "negative " if sign == "-" else ""
        w = int(whole) if whole else 0
        result = prefix + _num_to_words(w) + (" dollar" if w == 1 else " dollars")
        if cents:
            c = int(cents)
            if c > 0:
                result += " and " + _num_to_words(c) + (" cent" if c == 1 else " cents")
        return result
    text = re.sub(r"(-?)\$([0-9,]+)(\.(\d{1,2}))?", _currency, text)

    # Step 5: Percentages — 80.2% → "eighty point two percent"
    def _percent(m):
        whole = m.group(1)
        dec = m.group(3)
        result = _num_to_words(int(whole))
        if dec:
            result += " point " + " ".join(_ONES[int(d)] for d in dec)
        return result + " percent"
    text = re.sub(r"(\d+)(\.(\d+))?%", _percent, text)

    # Step 6: Multipliers — 3.5x → "three point five x"
    def _multiplier(m):
        whole = m.group(1)
        dec = m.group(3)
        result = _num_to_words(int(whole))
        if dec:
            result += " point " + " ".join(_ONES[int(d)] for d in dec)
        return result + " x"
    text = re.sub(r"(\d+)(\.(\d+))x\b", _multiplier, text)

    # Step 7: Decimal numbers — 3.14 → "three point one four"
    def _decimal(m):
        whole = m.group(1).replace(",", "")
        dec = m.group(2)
        result = _num_to_words(int(whole))
        result += " point " + " ".join(_ONES[int(d)] for d in dec)
        return result
    text = re.sub(r"(\d[\d,]*)\.([\d]+)", _decimal, text)

    # Step 8: Large numbers with commas — 1,000,000 → "one million"
    def _big_num(m):
        n = int(m.group(0).replace(",", ""))
        return _num_to_words(n)
    text = re.sub(r"\d{1,3}(?:,\d{3})+", _big_num, text)

    # Step 9: Split number+unit combos — "16GB" → "16 GB", "3080Ti" → "3080 Ti"
    text = re.sub(r"(\d)([A-Z]{2,})\b", r"\1 \2", text)

    # Step 10: Remaining standalone numbers (up to 6 digits)
    def _plain_num(m):
        n = int(m.group(0))
        if n <= 999999:
            return _num_to_words(n)
        return m.group(0)
    text = re.sub(r"\b\d{1,6}\b", _plain_num, text)

    # Step 11: Abbreviations
    for pattern, replacement in _ABBREVS:
        text = pattern.sub(replacement, text)

    # Step 12: Slash alternatives — "input/output" → "input or output"
    text = re.sub(r"(\w)/(\w)", r"\1 or \2", text)

    # Step 13: Clean up extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _strip_markdown(text: str) -> str:
    """Remove common markdown so TTS reads cleanly (used for notifications)."""
    text = re.sub(r"```[\s\S]*?```", "[code block]", text)
    def _pick(m):
        # Return first non-None captured group (the inner text)
        for g in m.groups():
            if g is not None:
                return g
        return ""
    return _MD_STRIP.sub(_pick, text).strip()


# Common Whisper hallucination patterns (on silence/noise)
_HALLUCINATION_PATTERNS = {
    "thank you for watching", "thanks for watching", "subscribe",
    "like and subscribe", "please subscribe", "see you next time",
    "bye bye", "thank you", "thanks for listening",
    "you", "the end", "so",
}


def _text_similarity(a: str, b: str) -> float:
    """Simple word-overlap similarity ratio (0.0 to 1.0)."""
    if not a or not b:
        return 0.0
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    overlap = len(words_a & words_b)
    return overlap / max(len(words_a), len(words_b))


class VoiceAssistant:
    def __init__(self):
        self.is_active = False
        self.is_processing = False
        self._flush_mic_buffer = False
        self._post_tts_gate = False
        self._post_tts_silence_count = 0
        self._abort_event = threading.Event()
        self._tts_process: Optional[subprocess.Popen] = None

        # Claude subprocess state
        self._claude_process: Optional[subprocess.Popen] = None
        self._session_id: Optional[str] = None

        # Per-run state for streaming
        self._sentence_queue: Optional[queue.Queue] = None
        self._thinking_text = ""
        self._thinking_shown_len = 0
        self._last_thinking_notify = 0.0
        self._last_tool_notify = 0.0
        self._assistant_text = ""        # cumulative text from deltas
        self._assistant_spoken_pos = 0   # cursor: how much we've already queued for TTS
        self._last_tts_text = ""  # last text spoken by TTS, for echo detection

        # Stream parser state for tool tracking
        self._current_tool_idx: Optional[int] = None
        self._current_tool_name: str = ""
        self._current_tool_input: str = ""

        # Paths
        self.state_dir = Path.home() / ".local/state/voice-assistant"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.state_dir / "voice-assistant.log"
        self.pid_file = self.state_dir / "voice-assistant.pid"
        self.chimes_dir = self.state_dir / "chimes"
        self.chimes_dir.mkdir(exist_ok=True)
        self.tts_dir = self.state_dir / "tts"
        self.tts_dir.mkdir(exist_ok=True)

        # Initialize components
        self._setup_logging()
        self._preflight_checks()
        self._session_id = self._load_session_id()
        self._setup_audio()
        self._setup_models()
        self._ensure_chimes()

        signal.signal(signal.SIGUSR1, self._toggle_handler)
        signal.signal(signal.SIGUSR2, self._new_session_handler)
        self.pid_file.write_text(str(os.getpid()))

        self._set_waybar_status("off")

    # ------------------------------------------------------------------
    # Preflight checks
    # ------------------------------------------------------------------

    def _preflight_checks(self):
        """Verify critical runtime requirements at startup."""
        ok = True

        # Claude Code must be installed
        claude_path = subprocess.run(
            ["which", "claude"], capture_output=True, text=True
        ).stdout.strip()
        if not claude_path:
            self.logger.error("PREFLIGHT FAIL: 'claude' CLI not found in PATH")
            ok = False
        else:
            self.logger.info(f"Claude CLI: {claude_path}")

        # --dangerously-skip-permissions must be accepted in settings
        settings_file = Path.home() / ".claude/settings.json"
        if settings_file.exists():
            try:
                settings = json.loads(settings_file.read_text())
                if settings.get("skipDangerousModePermissionPrompt"):
                    self.logger.info("Claude skipDangerousModePermissionPrompt: enabled")
                else:
                    self.logger.warning(
                        "PREFLIGHT WARN: skipDangerousModePermissionPrompt not set in "
                        "~/.claude/settings.json — voice assistant needs "
                        "--dangerously-skip-permissions to work non-interactively"
                    )
            except (json.JSONDecodeError, Exception):
                pass
        else:
            self.logger.warning(
                "PREFLIGHT WARN: ~/.claude/settings.json not found — "
                "set skipDangerousModePermissionPrompt: true"
            )

        # Passwordless sudo is required for system commands
        sudo_check = subprocess.run(
            ["sudo", "-n", "true"], capture_output=True, timeout=5
        )
        if sudo_check.returncode == 0:
            self.logger.info("Passwordless sudo: available")
        else:
            self.logger.warning(
                "PREFLIGHT WARN: passwordless sudo not available — "
                "Claude may fail on system commands. "
                "Fix: echo '%s ALL=(ALL) NOPASSWD: ALL' | sudo tee /etc/sudoers.d/%s"
                % (os.getenv("USER", "user"), os.getenv("USER", "user"))
            )
            ok = False

        if ok:
            self.logger.info("Preflight checks passed")

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------

    def _load_session_id(self) -> Optional[str]:
        """Load persisted session ID from state file."""
        try:
            if SESSION_FILE.exists():
                sid = SESSION_FILE.read_text().strip()
                if sid:
                    self.logger.info(f"Loaded persisted session: {sid}")
                    return sid
        except Exception:
            pass
        return None

    def _save_session_id(self, sid: str):
        """Persist session ID to state file."""
        try:
            SESSION_FILE.write_text(sid)
            self.logger.info(f"Persisted session: {sid}")
        except Exception as e:
            self.logger.error(f"Failed to persist session ID: {e}")

    def _clear_session(self):
        """Clear session ID from memory and disk, so next query starts fresh."""
        self._session_id = None
        try:
            SESSION_FILE.unlink(missing_ok=True)
        except Exception:
            pass
        self.logger.info("Session cleared — next query starts a new conversation")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_logging(self):
        handler = logging.handlers.RotatingFileHandler(
            self.log_file, maxBytes=2 * 1024 * 1024, backupCount=2,
        )
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[handler, logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Voice Assistant starting...")

    def _setup_audio(self):
        self.audio = pyaudio.PyAudio()
        default_info = self.audio.get_default_input_device_info()
        self.input_device = default_info["index"]
        self.logger.info(
            f"Audio input: {self.input_device} ({default_info['name']}, "
            f"native rate: {int(default_info['defaultSampleRate'])})"
        )

    def _setup_models(self):
        try:
            self.vad_model = load_silero_vad(onnx=False)
            self.logger.info("Silero VAD loaded")

            self.whisper_model = WhisperModel(
                WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE
            )
            self.logger.info("Faster Whisper loaded")

            self._setup_tts()
            self._warmup()
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            sys.exit(1)

    def _setup_tts(self):
        self.tts_available = False
        self.kokoro = None
        try:
            from kokoro_onnx import Kokoro
            self.kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
            self.tts_available = True
            self.logger.info("Kokoro TTS loaded")
        except ImportError:
            self.logger.warning("kokoro-onnx not installed, using espeak fallback")
        except Exception as e:
            self.logger.warning(f"Kokoro TTS setup failed: {e}")
            try:
                from kokoro_onnx import Kokoro
                self.kokoro = Kokoro.from_pretrained()
                self.tts_available = True
                self.logger.info("Kokoro TTS loaded from pretrained")
            except Exception as e2:
                self.logger.warning(f"Kokoro fallback also failed: {e2}, using espeak")

    def _warmup(self):
        self.logger.info("Warming up models...")
        dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)
        segments, _ = self.whisper_model.transcribe(dummy)
        list(segments)
        self.logger.info("Whisper warmup complete")

        if self.tts_available and self.kokoro:
            self.logger.info("Warming up Kokoro TTS...")
            self.kokoro.create("Hello.", voice=TTS_VOICE, speed=TTS_SPEED)
            self.logger.info("Kokoro TTS warmup complete")

    # ------------------------------------------------------------------
    # Chimes
    # ------------------------------------------------------------------

    def _ensure_chimes(self):
        chimes = {
            "listening": ([440, 523.25, 659.25], False),
            "processing": ([440], True),
            "deactivate": ([659.25, 523.25, 440], False),
        }
        if all((self.chimes_dir / f"{n}.wav").exists() for n in chimes):
            return
        for name, (freqs, fade) in chimes.items():
            data = self._create_chime(freqs, fade)
            path = self.chimes_dir / f"{name}.wav"
            with wave.open(str(path), "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(CHIME_SAMPLE_RATE)
                f.writeframes(data.tobytes())
        self.logger.info("Chimes generated")

    def _create_chime(self, frequencies, fade=False):
        spn = int(CHIME_SAMPLE_RATE * CHIME_NOTE_DURATION)
        audio = np.zeros(spn * len(frequencies))
        for i, freq in enumerate(frequencies):
            t = np.linspace(0, CHIME_NOTE_DURATION, spn)
            note = 0.3 * np.sin(2 * np.pi * freq * t)
            env = np.exp(-2 * t)
            if fade and i == len(frequencies) - 1:
                env *= np.linspace(1, 0, spn)
            audio[i * spn : (i + 1) * spn] = note * env
        return (audio * 32767).astype(np.int16)

    def _play_chime(self, name):
        path = self.chimes_dir / f"{name}.wav"
        if path.exists():
            subprocess.run(["pw-play", str(path)], capture_output=True, check=False)

    def _play_chime_async(self, name):
        threading.Thread(target=self._play_chime, args=(name,), daemon=True).start()

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def _notify(self, message, title="Voice Assistant", timeout_ms=None, silent=False):
        cmd = ["notify-send", title, message]
        if silent:
            cmd.extend(["-h", "string:suppress-popup:true", "-u", "low"])
        # Always set an explicit expire time so voice assistant notifications
        # remain persistent regardless of swaync's global timeout setting.
        # -1 = never expire (freedesktop spec), overrides any server default.
        expire = str(timeout_ms) if timeout_ms is not None else "-1"
        cmd.extend(["--expire-time=" + expire])
        subprocess.run(cmd, capture_output=True, check=False)

    # ------------------------------------------------------------------
    # Waybar
    # ------------------------------------------------------------------

    def _set_waybar_status(self, state):
        status_file = self.state_dir / "waybar-status"
        symbols = {"off": "◯", "ready": "●", "listening": "◉", "thinking": "◈", "speaking": "◆"}
        status_file.write_text(json.dumps({"text": symbols.get(state, ""), "class": state}))

    # ------------------------------------------------------------------
    # Claude Code — subprocess per query with streaming JSON
    # ------------------------------------------------------------------

    def _build_claude_cmd(self, message: str) -> list:
        """Build the claude CLI command for a query."""
        cmd = [
            "claude", "-p", message,
            "--output-format", "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--dangerously-skip-permissions",
            "--model", CLAUDE_MODEL,
            "--append-system-prompt", CLAUDE_VOICE_PROMPT,
        ]
        if self._session_id:
            cmd.extend(["--resume", self._session_id])
        else:
            cmd.extend(["--continue"])
        return cmd

    def _parse_claude_stream(self, process):
        """Parse claude stream-json stdout line by line, driving notifications and TTS."""
        for raw_line in iter(process.stdout.readline, b''):
            if self._abort_event.is_set():
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type")

            if etype == "system":
                sid = event.get("session_id")
                if sid:
                    self._session_id = sid
                    self._save_session_id(sid)
                    self.logger.info(f"Claude session: {sid}")

            elif etype == "stream_event":
                self._handle_stream_event(event.get("event", {}))

            elif etype == "result":
                if event.get("is_error"):
                    self.logger.warning(f"Claude error: {event.get('result', '')}")
                else:
                    self.logger.info(
                        f"Claude done (cost=${event.get('total_cost_usd', 0):.4f}, "
                        f"turns={event.get('num_turns', 1)})"
                    )
                return True

        return False  # process ended without a result event

    def _handle_stream_event(self, se):
        """Handle a single stream_event from claude's raw API stream."""
        se_type = se.get("type")

        if se_type == "content_block_start":
            cb = se.get("content_block", {})
            idx = se.get("index")
            if cb.get("type") == "tool_use":
                self._current_tool_idx = idx
                self._current_tool_name = cb.get("name", "")
                self._current_tool_input = ""

        elif se_type == "content_block_delta":
            delta = se.get("delta", {})
            dt = delta.get("type")

            if dt == "thinking_delta":
                self._thinking_text += delta.get("thinking", "")
                self._maybe_notify_thinking()

            elif dt == "text_delta":
                self._assistant_text += delta.get("text", "")
                self._flush_sentences(final=False)

            elif dt == "input_json_delta":
                if se.get("index") == self._current_tool_idx:
                    self._current_tool_input += delta.get("partial_json", "")

        elif se_type == "content_block_stop":
            idx = se.get("index")
            if idx == self._current_tool_idx and self._current_tool_name:
                self._notify_tool_use(self._current_tool_name, self._current_tool_input)
                self._current_tool_idx = None
                self._current_tool_name = ""
                self._current_tool_input = ""

    def _maybe_notify_thinking(self):
        """Rate-limited thinking notifications (every 2 seconds, sentence-aligned)."""
        text = self._thinking_text
        prev_len = self._thinking_shown_len
        if len(text) <= prev_len:
            return
        now = time.time()
        if now - self._last_thinking_notify < 2.0:
            return
        new_text = text[prev_len:]
        last_boundary = -1
        for m in re.finditer(r"(?<!\d)[.!?](?:\s|$)", new_text):
            last_boundary = m.end()
        if last_boundary > 0:
            to_send = new_text[:last_boundary].strip()
            if len(to_send) >= 20:
                self._last_thinking_notify = now
                self._thinking_shown_len = prev_len + last_boundary
                self._notify(f"🧠 {to_send}", title="Thinking...", silent=True)

    def _notify_tool_use(self, tool_name, input_json_str):
        """Show a tool-use notification with friendly label and details."""
        labels = {
            "Bash": "Running command",
            "Read": "Reading file",
            "Edit": "Editing file",
            "Write": "Writing file",
            "Glob": "Searching files",
            "Grep": "Searching code",
            "WebSearch": "Searching web",
            "WebFetch": "Fetching page",
            "Task": "Running agent",
            "NotebookEdit": "Editing notebook",
        }
        label = labels.get(tool_name, f"Using {tool_name}")
        detail = ""
        try:
            args = json.loads(input_json_str) if input_json_str else {}
        except json.JSONDecodeError:
            args = {}

        if tool_name == "Bash":
            cmd = args.get("command", "")
            if cmd:
                first_line = cmd.split("\n")[0].strip()
                detail = f"\n{first_line[:120]}"
        elif tool_name in ("Read", "Edit", "Write"):
            path = args.get("file_path", "")
            if path:
                detail = f"\n{path}"
        elif tool_name == "WebSearch":
            query = args.get("query", "")
            if query:
                detail = f"\n{query}"
        elif tool_name == "WebFetch":
            url = args.get("url", "")
            if url:
                detail = f"\n{url[:120]}"

        now = time.time()
        if now - self._last_tool_notify >= 2.0:
            self._last_tool_notify = now
            self._notify(f"🔧 {label}{detail}", title="Working...", silent=True)

    def _flush_sentences(self, final=False):
        """Extract complete sentences from _assistant_text beyond _assistant_spoken_pos."""
        remaining = self._assistant_text[self._assistant_spoken_pos:]
        if not remaining:
            return

        while True:
            match = re.search(r"(?<!\d)[.!?](?:\s|$)", remaining)
            if not match:
                break
            end = match.end()
            sentence = remaining[:end].strip()
            remaining = remaining[end:]
            self._assistant_spoken_pos += end
            if sentence and self._sentence_queue:
                self._sentence_queue.put(_prepare_for_speech(sentence))
                self.logger.info(f"→ TTS: {sentence[:80]}...")

        if final and remaining.strip():
            sentence = remaining.strip()
            self._assistant_spoken_pos += len(remaining)
            if sentence and self._sentence_queue:
                self._sentence_queue.put(_prepare_for_speech(sentence))
                self.logger.info(f"→ TTS (final): {sentence[:80]}...")

    # ------------------------------------------------------------------
    # Query via Claude Code subprocess
    # ------------------------------------------------------------------

    def _query_and_speak(self, text):
        """Send query to Claude Code subprocess, stream thinking/tools/response to TTS."""
        # Reset per-run state
        self._thinking_text = ""
        self._thinking_shown_len = 0
        self._last_thinking_notify = 0.0
        self._last_tool_notify = 0.0
        self._assistant_text = ""
        self._assistant_spoken_pos = 0
        self._current_tool_idx = None
        self._current_tool_name = ""
        self._current_tool_input = ""

        # Clean up any leftover TTS files from a previous run
        for f in self.tts_dir.glob("tts_*.wav"):
            f.unlink(missing_ok=True)

        # Set up streaming TTS pipeline: sentences → synthesis → playback
        sentence_q: queue.Queue[Optional[str]] = queue.Queue()
        audio_q: queue.Queue[Optional[Path]] = queue.Queue()
        self._sentence_queue = sentence_q

        def tts_worker():
            idx = 0
            while True:
                sentence = sentence_q.get()
                if sentence is None or self._abort_event.is_set():
                    audio_q.put(None)
                    return
                try:
                    samples, sr = self.kokoro.create(sentence, voice=TTS_VOICE, speed=TTS_SPEED)
                    path = self.tts_dir / f"tts_{idx}.wav"
                    sf.write(str(path), samples, sr)
                    audio_q.put(path)
                    idx += 1
                except Exception as e:
                    self.logger.error(f"TTS error on sentence: {e}")
                    continue

        def playback_worker():
            first = True
            prev_path = None
            while True:
                if self._abort_event.is_set():
                    return
                path = audio_q.get()
                if path is None:
                    if self._tts_process and self._tts_process.poll() is None:
                        self._tts_process.wait()
                    self._tts_process = None
                    if prev_path and prev_path.exists():
                        prev_path.unlink(missing_ok=True)
                    return
                if first:
                    self._set_waybar_status("speaking")
                    first = False
                if self._tts_process and self._tts_process.poll() is None:
                    self._tts_process.wait()
                if self._abort_event.is_set():
                    return
                if prev_path and prev_path.exists():
                    prev_path.unlink(missing_ok=True)
                self._tts_process = subprocess.Popen(
                    ["pw-play", str(path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                prev_path = path

        tts_thread = play_thread = None
        if self.tts_available and self.kokoro:
            tts_thread = threading.Thread(target=tts_worker, daemon=True)
            play_thread = threading.Thread(target=playback_worker, daemon=True)
            tts_thread.start()
            play_thread.start()

        # Build and launch Claude subprocess
        cmd = self._build_claude_cmd(text)
        self.logger.info(f"Query: {text[:80]}...")

        # Play processing chime while waiting (non-blocking)
        self._play_chime_async("processing")

        env = os.environ.copy()
        env.pop("CLAUDECODE", None)  # prevent nesting check
        try:
            self._claude_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=env,
                cwd=str(Path(__file__).parent),
            )
        except FileNotFoundError:
            self.logger.error("'claude' CLI not found in PATH")
            self._sentence_queue = None
            sentence_q.put(None)
            return "Sorry, Claude Code is not installed."
        except Exception as e:
            self.logger.error(f"Failed to spawn claude: {e}")
            self._sentence_queue = None
            sentence_q.put(None)
            return "Sorry, I couldn't start Claude."

        # Parse streaming output (blocks until done or abort)
        success = self._parse_claude_stream(self._claude_process)

        # Reap the process
        try:
            self._claude_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._claude_process.kill()
            self._claude_process.wait(timeout=2)
        except Exception:
            pass

        self._claude_process = None
        self._sentence_queue = None

        if self._abort_event.is_set():
            sentence_q.put(None)
            return None

        # Flush any remaining thinking as notification
        if self._thinking_text:
            remaining = self._thinking_text[self._thinking_shown_len:].strip()
            if remaining:
                self._notify(f"🧠 {remaining}", title="Thinking...", silent=True)

        full_response = self._assistant_text.strip()

        # If Claude used tools but produced no text response, say "Done"
        if not full_response and success:
            full_response = "Done."
            self._assistant_text = full_response

        _is_silent = not full_response

        # Clear thinking/tool notifications before showing response
        subprocess.run(["swaync-client", "--close-all"], capture_output=True, check=False)

        # Show final response as one notification
        if not _is_silent:
            preview = _strip_markdown(full_response)
            if preview:
                self._notify(f"🧙 {preview}", title="Assistant")

        # Flush ALL remaining text to TTS
        if not _is_silent:
            self._flush_sentences(final=True)
        else:
            self.logger.info("Empty reply — skipping TTS")

        # Signal end of TTS pipeline
        sentence_q.put(None)

        if not full_response:
            full_response = "Sorry, I got an empty response."

        self.logger.info(f"Response: {full_response[:200]}...")
        self._last_tts_text = full_response

        # Wait for TTS pipeline to drain
        if tts_thread:
            tts_thread.join(timeout=60)
        if play_thread:
            play_thread.join(timeout=60)

        # Dismiss all notifications after playback finishes
        subprocess.run(["swaync-client", "--close-all"], capture_output=True, check=False)

        return full_response

    # ------------------------------------------------------------------
    # Abort / Toggle
    # ------------------------------------------------------------------

    def _abort_inflight(self):
        self._abort_event.set()

        # Kill Claude subprocess
        if self._claude_process and self._claude_process.poll() is None:
            self._claude_process.kill()
            self.logger.info("Killed Claude process")

        # Kill TTS playback
        if self._tts_process and self._tts_process.poll() is None:
            self._tts_process.kill()
            self.logger.info("Killed TTS playback")

        subprocess.run(["pkill", "-f", "pw-play.*tts_"], capture_output=True, check=False)

        # Signal end to TTS pipeline
        if self._sentence_queue:
            self._sentence_queue.put(None)

    def _delayed_dismiss(self):
        time.sleep(2)
        subprocess.run(["swaync-client", "--close-all"], capture_output=True, check=False)

    def _toggle_handler(self, signum, frame):
        self.is_active = not self.is_active
        if self.is_active:
            self.logger.info("Activated")
            self._abort_event.clear()
            self._notify("🎤 Voice Mode ON")
            self._set_waybar_status("ready")
            self._play_chime_async("listening")
        else:
            self.logger.info("Deactivated")
            self._notify("🎤 Voice Mode OFF")
            self._set_waybar_status("off")
            self._abort_inflight()
            self._play_chime_async("deactivate")
            # Dismiss all notifications after a short delay
            threading.Thread(target=self._delayed_dismiss, daemon=True).start()

    def _new_session_handler(self, signum, frame):
        """SIGUSR2 handler: clear session so next query starts fresh."""
        self._clear_session()
        self._notify("🔄 New conversation ready", title="Voice Assistant")
        self._play_chime_async("deactivate")

    # ------------------------------------------------------------------
    # Audio capture
    # ------------------------------------------------------------------

    def _read_chunk(self, stream, duration=0.5):
        num_frames = int(SAMPLE_RATE * duration)
        frames = []
        for _ in range(num_frames // CHUNK_SIZE):
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            frames.append(data)
        if not frames:
            return np.zeros(int(SAMPLE_RATE * duration), dtype=np.float32)
        raw = np.frombuffer(b"".join(frames), dtype=np.int16)
        return raw.astype(np.float32) / 32768.0

    def _detect_speech(self, audio_data):
        tensor = torch.from_numpy(audio_data)
        timestamps = get_speech_timestamps(tensor, self.vad_model, sampling_rate=SAMPLE_RATE)
        return len(timestamps) > 0

    def _record_until_silence(self, stream, pre_audio=None):
        frames = []
        silence_limit = int(SILENCE_TIMEOUT / VAD_CHUNK_DURATION)
        silence_chunks = 0
        had_speech = pre_audio is not None
        if pre_audio is not None:
            frames.append(pre_audio)
        for _ in range(int(MAX_RECORD_DURATION / VAD_CHUNK_DURATION)):
            if not self.is_active:
                break
            chunk = self._read_chunk(stream, VAD_CHUNK_DURATION)
            frames.append(chunk)
            if self._detect_speech(chunk):
                had_speech = True
                silence_chunks = 0
            elif had_speech:
                silence_chunks += 1
                if silence_chunks >= silence_limit:
                    self.logger.info("Silence detected")
                    break
        if not frames:
            return np.zeros(SAMPLE_RATE, dtype=np.float32)
        return np.concatenate(frames)

    # ------------------------------------------------------------------
    # Whisper STT
    # ------------------------------------------------------------------

    def _transcribe(self, audio_data):
        try:
            duration = len(audio_data) / SAMPLE_RATE
            self.logger.info(f"Processing audio with duration {int(duration // 60):02d}:{duration % 60:06.3f}")
            # faster-whisper accepts numpy arrays directly — skip disk I/O
            segments, _ = self.whisper_model.transcribe(audio_data)
            text = " ".join(seg.text for seg in segments).strip()
            return text
        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
            return ""

    # ------------------------------------------------------------------
    # Main listen loop
    # ------------------------------------------------------------------

    async def _listen_loop(self):
        loop = asyncio.get_event_loop()
        stream = None
        prev_chunk = None

        while True:
            if not self.is_active:
                if stream is not None:
                    try:
                        stream.stop_stream()
                        stream.close()
                    except OSError:
                        pass
                    stream = None
                await asyncio.sleep(0.1)
                continue

            if stream is None:
                try:
                    stream = self.audio.open(
                        format=AUDIO_FORMAT, channels=CHANNELS, rate=SAMPLE_RATE,
                        input=True, input_device_index=self.input_device,
                        frames_per_buffer=CHUNK_SIZE,
                    )
                    self.logger.info("Audio stream opened")
                except OSError as e:
                    self.logger.error(f"Failed to open audio stream: {e}")
                    await asyncio.sleep(1)
                    continue

            # Drain mic buffer while processing to prevent overflow
            if self.is_processing:
                try:
                    await loop.run_in_executor(None, self._read_chunk, stream, 0.1)
                except OSError:
                    pass
                await asyncio.sleep(0.05)
                continue

            # Flush stale mic buffer after TTS playback ends
            # Close and reopen the stream to guarantee a clean buffer
            if self._flush_mic_buffer:
                self._flush_mic_buffer = False
                try:
                    stream.stop_stream()
                    stream.close()
                except OSError:
                    pass
                stream = None
                prev_chunk = None
                self._post_tts_gate = True
                self._post_tts_silence_count = 0
                self.logger.info("Flushed mic buffer (stream reset)")
                continue

            # After flush, wait for room to go quiet before listening
            if self._post_tts_gate and stream is not None:
                try:
                    chunk = await loop.run_in_executor(
                        None, self._read_chunk, stream, VAD_CHUNK_DURATION
                    )
                except OSError:
                    continue
                if self._detect_speech(chunk):
                    self._post_tts_silence_count = 0
                else:
                    self._post_tts_silence_count += 1
                    if self._post_tts_silence_count >= 3:
                        self._post_tts_gate = False
                        self._post_tts_silence_count = 0
                        self.logger.info("Room quiet — resuming listening")
                continue

            try:
                audio_chunk = await loop.run_in_executor(
                    None, self._read_chunk, stream, VAD_CHUNK_DURATION
                )
            except OSError as e:
                self.logger.error(f"Audio read error: {e}")
                try:
                    stream.close()
                except OSError:
                    pass
                stream = None
                continue

            if not self.is_active:
                prev_chunk = None
                continue

            if self._detect_speech(audio_chunk):
                self._set_waybar_status("listening")
                self.logger.info("Speech detected, recording...")

                # Include the previous chunk to capture speech onset
                if prev_chunk is not None:
                    pre_audio = np.concatenate([prev_chunk, audio_chunk])
                else:
                    pre_audio = audio_chunk

                full_audio = await loop.run_in_executor(
                    None, self._record_until_silence, stream, pre_audio
                )
                prev_chunk = None
                self.is_processing = True
                self._set_waybar_status("thinking")
                asyncio.create_task(self._process_audio(full_audio))
            else:
                prev_chunk = audio_chunk

            await asyncio.sleep(0.05)

    async def _process_audio(self, audio_data):
        loop = asyncio.get_event_loop()
        did_tts = False
        try:
            transcription = await loop.run_in_executor(None, self._transcribe, audio_data)
            if not transcription:
                self.is_processing = False
                return

            # Check for new-session voice commands (before length filter)
            words = transcription.split()
            lower = transcription.lower().strip().rstrip(".")
            if lower in _NEW_SESSION_PHRASES:
                self.logger.info(f"Voice command: new session ({transcription})")
                self._clear_session()
                self._notify("🔄 New conversation started", title="Voice Assistant")
                if self.tts_available and self.kokoro:
                    self._set_waybar_status("speaking")
                    try:
                        samples, sr = self.kokoro.create(
                            "Starting a new conversation.", voice=TTS_VOICE, speed=TTS_SPEED
                        )
                        path = self.tts_dir / "tts_new_session.wav"
                        sf.write(str(path), samples, sr)
                        subprocess.run(["pw-play", str(path)], capture_output=True, check=False)
                        path.unlink(missing_ok=True)
                    except Exception as e:
                        self.logger.error(f"TTS confirmation error: {e}")
                self.is_processing = False
                self._flush_mic_buffer = True
                if self.is_active:
                    self._set_waybar_status("ready")
                return

            # Filter hallucinations: too short, known patterns, low-content
            if len(words) < 3:
                self.logger.info(f"Rejected (too short): {transcription}")
                self.is_processing = False
                return
            if lower in _HALLUCINATION_PATTERNS:
                self.logger.info(f"Rejected (hallucination): {transcription}")
                self.is_processing = False
                return

            # Echo detection: reject if too similar to last TTS output
            if self._last_tts_text and _text_similarity(transcription, self._last_tts_text) > 0.6:
                self.logger.info(f"Rejected (echo of TTS): {transcription}")
                self.is_processing = False
                return

            did_tts = True

            self.logger.info(f"Transcription: {transcription}")
            self._notify(f"🎤 {transcription}", title="You Said")

            response = await loop.run_in_executor(None, self._query_and_speak, transcription)

            if response is None:
                return

            # espeak fallback when Kokoro isn't available
            if not self.tts_available and self.is_active:
                self._set_waybar_status("speaking")
                await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["espeak", "-s", "150", "-v", "en+f3", response],
                        capture_output=True, check=False,
                    ),
                )
        except Exception as e:
            self.logger.error(f"Processing error: {e}")
        finally:
            self._flush_mic_buffer = True
            self.is_processing = False
            if self.is_active:
                self._set_waybar_status("ready")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _cleanup(self):
        # Clean up TTS temp files
        for f in self.tts_dir.glob("tts_*.wav"):
            f.unlink(missing_ok=True)
        if hasattr(self, "audio"):
            self.audio.terminate()
        self.pid_file.unlink(missing_ok=True)
        self.logger.info("Voice Assistant stopped")

    def run(self):
        try:
            self.logger.info("Voice Assistant ready — send SIGUSR1 to toggle")
            asyncio.run(self._listen_loop())
        except KeyboardInterrupt:
            pass
        finally:
            self._cleanup()


def main():
    assistant = VoiceAssistant()
    assistant.run()


if __name__ == "__main__":
    main()
