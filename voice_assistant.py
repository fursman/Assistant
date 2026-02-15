#!/usr/bin/env python3
"""
Hyprland Voice Assistant

Pipeline: VAD → Whisper STT → OpenClaw LLM (streaming via WS) → Kokoro TTS (streaming) → PipeWire playback

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
import uuid
import wave  # used by chime generation
from pathlib import Path
from typing import Optional

import numpy as np
import pyaudio
import soundfile as sf
import torch
import websocket
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
SILENCE_TIMEOUT = 1.5
MAX_RECORD_DURATION = 30

WHISPER_MODEL = "small"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE = "float16"

TTS_VOICE = "af_heart"
TTS_SPEED = 1.0

GATEWAY_URL = "ws://127.0.0.1:18789"
GATEWAY_SESSION_KEY = "agent:main:openai-user:voice-assistant"


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

def _strip_markdown(text: str) -> str:
    """Remove common markdown so TTS reads cleanly."""
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
        self._abort_event = threading.Event()
        self._tts_process: Optional[subprocess.Popen] = None

        # WebSocket state
        self._ws: Optional[websocket.WebSocket] = None
        self._ws_lock = threading.Lock()
        self._ws_connected = threading.Event()

        # Per-run state for streaming
        self._active_req_id: Optional[str] = None
        self._sentence_queue: Optional[queue.Queue] = None
        self._thinking_text = ""
        self._thinking_shown_len = 0
        self._last_thinking_notify = 0.0
        self._last_tool_notify = 0.0
        self._assistant_text = ""        # cumulative text from data.text
        self._assistant_spoken_pos = 0   # cursor: how much we've already queued for TTS
        self._run_done_event = threading.Event()
        self._last_tts_text = ""  # last text spoken by TTS, for echo detection

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
        self._setup_audio()
        self._setup_models()
        self._ensure_chimes()

        signal.signal(signal.SIGUSR1, self._toggle_handler)
        self.pid_file.write_text(str(os.getpid()))

        # Start WebSocket connection
        self._start_ws()
        self._set_waybar_status("off")

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
        if timeout_ms is not None:
            cmd.extend(["-t", str(timeout_ms)])
        subprocess.run(cmd, capture_output=True, check=False)

    # ------------------------------------------------------------------
    # Waybar
    # ------------------------------------------------------------------

    def _set_waybar_status(self, state):
        status_file = self.state_dir / "waybar-status"
        symbols = {"off": "◯", "ready": "●", "listening": "●", "thinking": "●", "speaking": "●"}
        status_file.write_text(json.dumps({"text": symbols.get(state, ""), "class": state}))

    # ------------------------------------------------------------------
    # WebSocket — single persistent connection
    # ------------------------------------------------------------------

    def _start_ws(self):
        threading.Thread(target=self._ws_loop, daemon=True).start()
        self._ws_connected.wait(timeout=10)

    def _ws_loop(self):
        """Persistent WS connection with auto-reconnect (exponential backoff)."""
        token = os.getenv("OPENCLAW_GATEWAY_TOKEN", "")
        backoff = 1.0
        while True:
            try:
                self._ws_connect(token)
                backoff = 1.0  # reset on successful connection
            except Exception as e:
                self.logger.warning(f"WS error: {e}")
            self._ws_connected.clear()
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    def _ws_connect(self, token):
        ws = websocket.WebSocket()
        ws.connect(GATEWAY_URL)
        self.logger.info("WS connected")

        challenge = json.loads(ws.recv())
        if challenge.get("event") != "connect.challenge":
            self.logger.warning(f"Expected connect.challenge, got: {challenge.get('event')}")
            ws.close()
            return

        ws.send(json.dumps({
            "type": "req",
            "id": "connect",
            "method": "connect",
            "params": {
                "minProtocol": 3,
                "maxProtocol": 3,
                "client": {
                    "id": "gateway-client",
                    "version": "2.0",
                    "platform": "linux",
                    "mode": "backend",
                },
                "caps": ["tool-events"],
                "auth": {"token": token},
                "scopes": ["agent", "operator.admin"],
            },
        }))

        resp = json.loads(ws.recv())
        if not (resp.get("type") == "res" and resp.get("ok")):
            self.logger.warning(f"WS handshake failed: {resp}")
            ws.close()
            return
        self.logger.info("WS handshake OK")

        self._ws = ws
        self._ws_connected.set()

        # Configure voice-assistant session: thinking + reasoning stream
        # (gateway patched to allow text streaming alongside reasoning stream)
        ws.send(json.dumps({
            "type": "req",
            "id": "patch-session",
            "method": "sessions.patch",
            "params": {
                "key": GATEWAY_SESSION_KEY,
                "thinkingLevel": "high",
                "reasoningLevel": "stream",
            },
        }))

        # Listen loop
        ws.settimeout(60)
        while True:
            try:
                raw = ws.recv()
                if not raw:
                    break
                self._handle_ws_message(json.loads(raw))
            except websocket.WebSocketTimeoutException:
                try:
                    ws.ping()
                except Exception:
                    break
            except Exception as e:
                self.logger.warning(f"WS recv error: {e}")
                break

        ws.close()
        self._ws = None
        # Unblock any waiting run
        if self._sentence_queue:
            self._sentence_queue.put(None)
        self._run_done_event.set()
        self.logger.info("WS disconnected")

    def _handle_ws_message(self, msg):
        msg_type = msg.get("type")

        if msg_type == "res":
            req_id = msg.get("id")
            if req_id == "patch-session":
                if msg.get("ok"):
                    self.logger.info("Session configured (thinking=high, reasoning=stream)")
                else:
                    self.logger.warning(f"Session patch failed: {msg.get('error')}")
            return

        if msg_type == "event" and msg.get("event") == "agent":
            payload = msg.get("payload", {})
            if payload.get("sessionKey") != GATEWAY_SESSION_KEY:
                return
            self._handle_agent_event(payload)

    def _handle_agent_event(self, payload):
        stream = payload.get("stream")
        data = payload.get("data", {})

        if not self._active_req_id:
            return

        if stream == "thinking":
            # text = full accumulated thinking, use it as source of truth
            full_text = data.get("text", "")
            if full_text:
                self._thinking_text = full_text
                # Strip "Reasoning:\n" prefix and italic underscores from formatting
                clean = full_text
                if clean.startswith("Reasoning:\n"):
                    clean = clean[len("Reasoning:\n"):]
                clean = re.sub(r"(?m)^_(.*)_$", r"\1", clean).strip()

                # Find sendable text: everything up to last real sentence end
                # Real sentence end = .!? followed by space, but NOT after a number (e.g. "1.")
                prev_len = self._thinking_shown_len
                if len(clean) > prev_len:
                    now = time.time()
                    last = self._last_thinking_notify
                    if now - last >= 2.0:
                        new_text = clean[prev_len:]
                        # Match sentence ends: .!? followed by whitespace,
                        # but not digits before . (avoids "1." "2." etc)
                        last_boundary = -1
                        for m in re.finditer(r"(?<!\d)[.!?](?:\s|$)", new_text):
                            last_boundary = m.end()
                        if last_boundary > 0:
                            to_send = new_text[:last_boundary].strip()
                            if len(to_send) >= 20:
                                self._last_thinking_notify = now
                                self._thinking_shown_len = prev_len + last_boundary
                                self._notify(
                                    f"🧠 {to_send}",
                                    title="Thinking...",
                                    silent=True,
                                )

        elif stream == "tool":
            phase = data.get("phase", "")
            tool_name = data.get("name", "tool")
            if phase == "start":
                labels = {
                    "exec": "Running command",
                    "web_search": "Searching web",
                    "web_fetch": "Fetching page",
                    "Read": "Reading file",
                    "Edit": "Editing file",
                    "Write": "Writing file",
                    "browser": "Using browser",
                    "image": "Analyzing image",
                    "memory_search": "Searching memory",
                }
                label = labels.get(tool_name, f"Using {tool_name}")
                args = data.get("args", {})
                detail = ""
                if tool_name == "exec":
                    cmd = args.get("command", "")
                    if cmd:
                        # Show first line, truncated
                        first_line = cmd.split("\n")[0].strip()
                        detail = f"\n{first_line[:120]}"
                elif tool_name in ("Read", "Edit", "Write"):
                    path = args.get("file_path", "") or args.get("path", "")
                    if path:
                        detail = f"\n{path}"
                elif tool_name == "web_search":
                    query = args.get("query", "")
                    if query:
                        detail = f"\n{query}"
                elif tool_name == "web_fetch":
                    url = args.get("url", "")
                    if url:
                        detail = f"\n{url[:120]}"
                now = time.time()
                last_tool = self._last_tool_notify
                if now - last_tool >= 2.0:
                    self._last_tool_notify = now
                    self._notify(
                        f"🔧 {label}{detail}",
                        title="Working...",
                        silent=True,
                    )

        elif stream == "assistant":
            text = data.get("text", "")
            if text:
                self._assistant_text = text
                # Stream sentences to TTS as they arrive
                self._flush_sentences(final=False)

        elif stream == "lifecycle":
            phase = data.get("phase")
            if phase == "start":
                self.logger.info("Agent run started")
            elif phase in ("end", "error"):
                if phase == "error":
                    self.logger.warning(f"Agent run error: {data.get('error', 'unknown')}")

                # Flush any unsent thinking
                if self._thinking_text:
                    clean = self._thinking_text
                    if clean.startswith("Reasoning:\n"):
                        clean = clean[len("Reasoning:\n"):]
                    clean = re.sub(r"(?m)^_(.*)_$", r"\1", clean).strip()
                    prev_len = self._thinking_shown_len
                    remaining = clean[prev_len:].strip()
                    if remaining:
                        self._notify(
                            f"🧠 {remaining}",
                            title="Thinking...",
                            silent=True,
                            
                        )

                # Skip silent/empty replies (NO_REPLY token or fragments)
                _silent = self._assistant_text.strip().upper().replace("_", "").replace(" ", "")
                _is_silent = not self._assistant_text.strip() or _silent in ("NOREPLY", "NO", "HEARTBEATOK")

                # Clear thinking/tool notifications before showing response
                subprocess.run(["swaync-client", "--close-all"], capture_output=True, check=False)

                # Show final response as one notification
                if not _is_silent:
                    preview = _strip_markdown(self._assistant_text)
                    if preview:
                        self._notify(
                            f"🧙 {preview}",
                            title="Clawbook",
                        )

                # Now flush ALL text to TTS at once
                if not _is_silent:
                    self._flush_sentences(final=True)
                else:
                    self.logger.info("Silent reply — skipping TTS")

                if self._sentence_queue:
                    self._sentence_queue.put(None)
                self._run_done_event.set()

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
                self._sentence_queue.put(_strip_markdown(sentence))
                self.logger.info(f"→ TTS: {sentence[:80]}...")

        if final and remaining.strip():
            sentence = remaining.strip()
            self._assistant_spoken_pos += len(remaining)
            if sentence and self._sentence_queue:
                self._sentence_queue.put(_strip_markdown(sentence))
                self.logger.info(f"→ TTS (final): {sentence[:80]}...")

    # ------------------------------------------------------------------
    # Query via WebSocket
    # ------------------------------------------------------------------

    def _query_and_speak(self, text):
        """Send query via WS, stream thinking as notifications, stream response to TTS."""
        req_id = f"voice-{uuid.uuid4().hex[:12]}"

        # Reset per-run state
        self._active_req_id = req_id
        self._thinking_text = ""
        self._thinking_shown_len = 0
        self._last_thinking_notify = 0.0
        self._last_tool_notify = 0.0
        self._assistant_text = ""
        self._assistant_spoken_pos = 0
        self._run_done_event.clear()

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
                    # Skip this sentence, keep going
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

        # Send query
        if not self._ws_connected.is_set() or not self._ws:
            self.logger.error("WS not connected")
            self._sentence_queue = None
            sentence_q.put(None)
            return "Sorry, I'm not connected to the gateway."

        try:
            with self._ws_lock:
                self._ws.send(json.dumps({
                    "type": "req",
                    "id": req_id,
                    "method": "chat.send",
                    "params": {
                        "sessionKey": GATEWAY_SESSION_KEY,
                        "message": text,
                        "deliver": False,
                        "idempotencyKey": req_id,
                    },
                }))
            self.logger.info(f"Query: {text[:80]}...")
        except Exception as e:
            self.logger.error(f"Failed to send query: {e}")
            self._sentence_queue = None
            sentence_q.put(None)
            return "Sorry, I couldn't send that query."

        # Play processing chime while waiting (non-blocking)
        self._play_chime_async("processing")

        # Wait for run to complete (or abort)
        while not self._run_done_event.wait(timeout=0.5):
            if self._abort_event.is_set():
                self._sentence_queue = None
                sentence_q.put(None)
                self._active_req_id = None
                return None

        self._sentence_queue = None
        self._active_req_id = None

        if self._abort_event.is_set():
            return None

        full_response = self._assistant_text.strip()
        if not full_response:
            full_response = "Sorry, I got an empty response."

        self.logger.info(f"Response: {full_response[:200]}...")
        self._last_tts_text = full_response

        # Wait for TTS pipeline to drain
        if tts_thread:
            tts_thread.join(timeout=60)
        if play_thread:
            play_thread.join(timeout=60)

        return full_response

    # ------------------------------------------------------------------
    # Abort / Toggle
    # ------------------------------------------------------------------

    def _abort_inflight(self):
        self._abort_event.set()

        if self._tts_process and self._tts_process.poll() is None:
            self._tts_process.kill()
            self.logger.info("Killed TTS playback")

        subprocess.run(["pkill", "-f", "pw-play.*tts_"], capture_output=True, check=False)

        def send_stop():
            try:
                if self._ws and self._ws_connected.is_set():
                    stop_id = f"stop-{uuid.uuid4().hex[:8]}"
                    with self._ws_lock:
                        self._ws.send(json.dumps({
                            "type": "req",
                            "id": stop_id,
                            "method": "chat.send",
                            "params": {
                                "sessionKey": GATEWAY_SESSION_KEY,
                                "message": "/stop",
                                "deliver": False,
                                "idempotencyKey": stop_id,
                            },
                        }))
                    self.logger.info("Sent /stop")
            except Exception as e:
                self.logger.warning(f"Failed to send /stop: {e}")

        threading.Thread(target=send_stop, daemon=True).start()

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
            if self._flush_mic_buffer:
                self._flush_mic_buffer = False
                try:
                    avail = stream.get_read_available()
                    while avail > 0:
                        stream.read(min(avail, CHUNK_SIZE), exception_on_overflow=False)
                        avail = stream.get_read_available()
                except OSError:
                    pass
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

            # Filter hallucinations: too short, known patterns, low-content
            words = transcription.split()
            lower = transcription.lower().strip().rstrip(".")
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
            if did_tts:
                # Keep draining mic for 1.5s after playback ends to clear any echo
                await asyncio.sleep(1.5)
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
