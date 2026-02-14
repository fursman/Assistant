#!/usr/bin/env python3
"""
Hyprland Voice Assistant
A voice-activated assistant for Hyprland with VAD, STT, LLM, and TTS.

Pipeline: VAD → Whisper STT → OpenClaw LLM (streaming) → Kokoro TTS (streaming) → PipeWire playback
TTS begins as soon as the first sentence finishes streaming from the LLM.
"""

import asyncio
import json
import logging
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import pyaudio
import soundfile as sf
import torch
import websocket
from faster_whisper import WhisperModel
from openai import OpenAI
from silero_vad import load_silero_vad, get_speech_timestamps


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Audio
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
AUDIO_FORMAT = pyaudio.paInt16

# VAD
VAD_CHUNK_DURATION = 0.5        # seconds per VAD check
SILENCE_TIMEOUT = 1.5           # seconds of silence to stop recording
MAX_RECORD_DURATION = 30        # max seconds to record

# Whisper
WHISPER_MODEL = "small"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE = "float16"

# TTS
TTS_VOICE = "af_heart"
TTS_SPEED = 1.0
TTS_MIN_CHUNK_LEN = 40          # minimum chars before splitting a TTS chunk

# LLM
LLM_MODEL = "openclaw:main"
LLM_MAX_TOKENS = 1024

# Notifications — swaync uses -r <id> for replacement
NOTIFY_ID_LISTENING = 59001     # "Listening..." / "You Said" (replaced in-place)
NOTIFY_ID_THINKING = 59002      # thinking chunks (replaced in-place)

# Chime frequencies (Hz)
CHIME_SAMPLE_RATE = 44100
CHIME_NOTE_DURATION = 0.2


class VoiceAssistant:
    def __init__(self):
        self.is_active = False
        self.is_processing = False
        self._abort_event = threading.Event()
        self._tts_process: Optional[subprocess.Popen] = None

        # Paths
        self.state_dir = Path.home() / ".local/state/voice-assistant"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.state_dir / "voice-assistant.log"
        self.pid_file = self.state_dir / "voice-assistant.pid"
        self.chimes_dir = self.state_dir / "chimes"
        self.chimes_dir.mkdir(exist_ok=True)

        # Initialize components
        self._setup_logging()
        self._setup_audio()
        self._setup_models()
        self._ensure_chimes()

        # Signal handler
        signal.signal(signal.SIGUSR1, self._toggle_handler)

        # Write PID file
        self.pid_file.write_text(str(os.getpid()))

        # Start WebSocket thinking listener
        self._start_thinking_listener()

        # Note: reasoning mode is set via WS in _thinking_ws_connect()

        # Initialize waybar status
        self._set_waybar_status("off")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(),
            ],
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
            # Silero VAD
            self.vad_model = load_silero_vad(onnx=False)
            self.logger.info("Silero VAD loaded")

            # Faster Whisper
            self.whisper_model = WhisperModel(
                WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE
            )
            self.logger.info("Faster Whisper loaded")

            # OpenClaw gateway
            self.openai = OpenAI(
                base_url="http://127.0.0.1:18789/v1",
                api_key=os.getenv("OPENCLAW_GATEWAY_TOKEN", ""),
            )
            self.logger.info("OpenClaw gateway client initialized")

            # Kokoro TTS
            self._setup_tts()

            # Warmup
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
        """Run dummy inferences to pre-compile CUDA/ONNX kernels."""
        self.logger.info("Warming up models...")

        # Whisper warmup
        temp_path = self.state_dir / "warmup.wav"
        dummy = np.zeros(SAMPLE_RATE, dtype=np.int16)
        with wave.open(str(temp_path), "wb") as f:
            f.setnchannels(CHANNELS)
            f.setsampwidth(2)
            f.setframerate(SAMPLE_RATE)
            f.writeframes(dummy.tobytes())
        segments, _ = self.whisper_model.transcribe(str(temp_path))
        list(segments)
        temp_path.unlink(missing_ok=True)
        self.logger.info("Whisper warmup complete")

        # Kokoro warmup
        if self.tts_available and self.kokoro:
            self.logger.info("Warming up Kokoro TTS...")
            self.kokoro.create("Hello.", voice=TTS_VOICE, speed=TTS_SPEED)
            self.logger.info("Kokoro TTS warmup complete")

    # ------------------------------------------------------------------
    # Chimes (generated once, cached on disk)
    # ------------------------------------------------------------------

    def _ensure_chimes(self):
        """Generate chime WAV files if they don't already exist."""
        chimes = {
            "listening": ([440, 523.25, 659.25], False),
            "processing": ([440], True),
            "deactivate": ([659.25, 523.25, 440], False),
        }
        any_missing = any(
            not (self.chimes_dir / f"{name}.wav").exists() for name in chimes
        )
        if not any_missing:
            self.logger.info("Chimes already cached")
            return

        for name, (freqs, fade) in chimes.items():
            audio_data = self._create_chime(freqs, fade)
            path = self.chimes_dir / f"{name}.wav"
            with wave.open(str(path), "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(CHIME_SAMPLE_RATE)
                f.writeframes(audio_data.tobytes())
        self.logger.info("Chimes generated")

    def _create_chime(self, frequencies, fade=False):
        samples_per_note = int(CHIME_SAMPLE_RATE * CHIME_NOTE_DURATION)
        total = samples_per_note * len(frequencies)
        audio = np.zeros(total)

        for i, freq in enumerate(frequencies):
            start = i * samples_per_note
            t = np.linspace(0, CHIME_NOTE_DURATION, samples_per_note)
            note = 0.3 * np.sin(2 * np.pi * freq * t)
            envelope = np.exp(-2 * t)
            if fade and i == len(frequencies) - 1:
                envelope *= np.linspace(1, 0, samples_per_note)
            audio[start : start + samples_per_note] = note * envelope

        return (audio * 32767).astype(np.int16)

    def _play_chime(self, name):
        path = self.chimes_dir / f"{name}.wav"
        if path.exists():
            subprocess.run(["pw-play", str(path)], capture_output=True, check=False)

    # ------------------------------------------------------------------
    # Notifications (swaync-compatible)
    # ------------------------------------------------------------------

    def _notify(self, message, title="Voice Assistant", replace_id=None, timeout_ms=None):
        """Send a desktop notification. replace_id is a numeric ID for swaync replacement."""
        cmd = ["notify-send", title, message]
        if replace_id is not None:
            cmd.extend(["-r", str(replace_id)])
        if timeout_ms is not None:
            cmd.extend(["-t", str(timeout_ms)])
        subprocess.run(cmd, capture_output=True, check=False)

    def _dismiss_notification(self, replace_id):
        """Dismiss a notification by sending a zero-length replacement that expires instantly."""
        subprocess.run(
            ["notify-send", "", "", "-r", str(replace_id), "-t", "1"],
            capture_output=True, check=False,
        )

    # ------------------------------------------------------------------
    # WebSocket thinking listener
    # ------------------------------------------------------------------

    def _start_thinking_listener(self):
        """Connect to OpenClaw gateway WebSocket to receive thinking events in real-time."""
        self._thinking_ws = None
        self._thinking_text = ""  # accumulated thinking for current run
        self._thinking_run_id = None
        thread = threading.Thread(target=self._thinking_listener_loop, daemon=True)
        thread.start()
        self.logger.info("Thinking listener thread started")

    def _thinking_listener_loop(self):
        """Persistent WebSocket connection with auto-reconnect."""
        token = os.getenv("OPENCLAW_GATEWAY_TOKEN", "")
        while True:
            try:
                self._thinking_ws_connect(token)
            except Exception as e:
                self.logger.warning(f"Thinking WS error: {e}")
            time.sleep(3)  # reconnect backoff

    def _thinking_ws_connect(self, token):
        """Connect and listen for thinking events."""
        ws = websocket.WebSocket()
        ws.connect("ws://127.0.0.1:18789")
        self.logger.info("Thinking WS connected")

        # Wait for connect.challenge
        raw = ws.recv()
        challenge = json.loads(raw)
        if challenge.get("event") != "connect.challenge":
            self.logger.warning(f"Expected connect.challenge, got: {challenge}")
            ws.close()
            return

        # Send handshake (after challenge)
        handshake = {
            "type": "req",
            "id": "1",
            "method": "connect",
            "params": {
                "minProtocol": 3,
                "maxProtocol": 3,
                "client": {
                    "id": "gateway-client",
                    "version": "voice-assistant",
                    "platform": "linux",
                    "mode": "backend",
                },
                "caps": [],
                "auth": {"token": token},
                "scopes": ["agent", "operator.admin"],
            },
        }
        ws.send(json.dumps(handshake))

        # Read handshake response
        try:
            raw_resp = ws.recv()
        except websocket.WebSocketConnectionClosedException as e:
            self.logger.warning(f"Thinking WS closed during handshake: {e}")
            return
        if not raw_resp:
            self.logger.warning("Thinking WS: empty handshake response")
            return
        resp = json.loads(raw_resp)
        if resp.get("type") == "res" and resp.get("ok"):
            self.logger.info("Thinking WS handshake OK")
        else:
            self.logger.warning(f"Thinking WS handshake failed: {resp}")
            ws.close()
            return

        # Set reasoning level to "stream" on the voice assistant session
        ws.send(json.dumps({
            "type": "req",
            "id": "set-reasoning",
            "method": "sessions.patch",
            "params": {
                "key": "agent:main:openai-user:voice-assistant",
                "reasoningLevel": "stream",
            },
        }))
        # Read messages until we get our response (skip interleaved events)
        for _ in range(20):
            msg = json.loads(ws.recv())
            if msg.get("type") == "res" and msg.get("id") == "set-reasoning":
                if msg.get("ok"):
                    self.logger.info("Reasoning level set to 'stream' via WS")
                else:
                    self.logger.warning(f"Failed to set reasoning: {msg.get('error')}")
                break

        # Listen for events
        self._thinking_ws = ws
        ws.settimeout(60)  # heartbeat interval
        while True:
            try:
                raw = ws.recv()
                if not raw:
                    break
                msg = json.loads(raw)
                self._handle_ws_message(msg)
            except websocket.WebSocketTimeoutException:
                # Send keepalive ping
                try:
                    ws.ping()
                except Exception:
                    break
            except Exception as e:
                self.logger.warning(f"Thinking WS recv error: {e}")
                break

        ws.close()
        self._thinking_ws = None
        self.logger.info("Thinking WS disconnected")

    def _handle_ws_message(self, msg):
        """Process incoming WebSocket messages, looking for thinking events."""
        if msg.get("type") != "event" or msg.get("event") != "agent":
            return

        payload = msg.get("payload", {})
        stream_type = payload.get("stream")
        data = payload.get("data", {})
        run_id = payload.get("runId")

        if stream_type == "thinking":
            # New run resets accumulated thinking
            if run_id != self._thinking_run_id:
                self._thinking_run_id = run_id
                self._thinking_text = ""

            delta = data.get("delta", "")
            if delta:
                self._thinking_text += delta

                # Fire notification when a sentence completes
                if re.search(r"[.!?]\s*$", self._thinking_text):
                    sentence = self._thinking_text.strip()
                    self._thinking_text = ""
                    self._notify(
                        f"🧠 {sentence}\n",
                        title="Thinking...",
                        timeout_ms=8000,
                    )
                    self.logger.info(f"Thinking sentence: {sentence[:60]}...")

        elif stream_type == "lifecycle":
            phase = data.get("phase")
            if phase in ("end", "error"):
                # Flush any remaining thinking text
                if self._thinking_text.strip():
                    self._notify(
                        f"🧠 {self._thinking_text.strip()}\n",
                        title="Thinking...",
                        timeout_ms=8000,
                    )
                self._thinking_text = ""
                self._thinking_run_id = None

    # ------------------------------------------------------------------
    # Waybar
    # ------------------------------------------------------------------

    def _set_waybar_status(self, state):
        """Update waybar status. States: off, ready, listening, thinking, speaking."""
        status_file = self.state_dir / "waybar-status"
        symbols = {
            "off": "◯", "ready": "●", "listening": "●",
            "thinking": "●", "speaking": "●",
        }
        status_file.write_text(json.dumps({
            "text": symbols.get(state, ""),
            "class": state,
        }))

    # ------------------------------------------------------------------
    # Abort / Toggle
    # ------------------------------------------------------------------

    def _abort_inflight(self):
        """Abort any in-flight request and kill TTS playback."""
        self._abort_event.set()

        if self._tts_process and self._tts_process.poll() is None:
            self._tts_process.kill()
            self.logger.info("Killed TTS playback")

        # Kill any lingering pw-play TTS chunks
        subprocess.run(
            ["pkill", "-f", "pw-play.*tts_chunk"], capture_output=True, check=False
        )

        # Send /stop to the session in background
        def send_stop():
            try:
                self.openai.chat.completions.create(
                    model=LLM_MODEL,
                    max_tokens=32,
                    messages=[{"role": "user", "content": "/stop"}],
                    extra_headers={"x-openclaw-agent-id": "main"},
                    user="voice-assistant",
                )
                self.logger.info("Sent /stop to abort server-side processing")
            except Exception as e:
                self.logger.warning(f"Failed to send /stop: {e}")

        threading.Thread(target=send_stop, daemon=True).start()
        self.logger.info("Signalled abort")

    def _toggle_handler(self, signum, frame):
        self.is_active = not self.is_active

        if self.is_active:
            self.logger.info("Voice Assistant activated")
            self._abort_event.clear()
            self._notify("🎤 Voice Mode ON")
            self._set_waybar_status("ready")
            self._play_chime_async("listening")
        else:
            self.logger.info("Voice Assistant deactivated")
            self._notify("🎤 Voice Mode OFF")
            self._set_waybar_status("off")
            self._abort_inflight()
            self._play_chime_async("deactivate")

    def _play_chime_async(self, name):
        threading.Thread(
            target=self._play_chime, args=(name,), daemon=True
        ).start()

    # ------------------------------------------------------------------
    # Audio capture
    # ------------------------------------------------------------------

    def _read_chunk(self, stream, duration=0.5):
        """Read a chunk of audio from an open PyAudio stream."""
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
        timestamps = get_speech_timestamps(
            tensor, self.vad_model, sampling_rate=SAMPLE_RATE
        )
        return len(timestamps) > 0

    def _record_until_silence(self, stream, pre_audio=None):
        """Record from an open stream until silence is detected after speech."""
        frames = []
        chunk_duration = VAD_CHUNK_DURATION
        silence_limit = int(SILENCE_TIMEOUT / chunk_duration)
        silence_chunks = 0
        had_speech = pre_audio is not None

        if pre_audio is not None:
            frames.append(pre_audio)

        max_chunks = int(MAX_RECORD_DURATION / chunk_duration)
        for _ in range(max_chunks):
            if not self.is_active:
                break
            chunk = self._read_chunk(stream, chunk_duration)
            frames.append(chunk)

            if self._detect_speech(chunk):
                had_speech = True
                silence_chunks = 0
            elif had_speech:
                silence_chunks += 1
                if silence_chunks >= silence_limit:
                    self.logger.info("Silence detected, stopping recording")
                    break

        if not frames:
            return np.zeros(SAMPLE_RATE, dtype=np.float32)
        return np.concatenate(frames)

    # ------------------------------------------------------------------
    # Whisper STT
    # ------------------------------------------------------------------

    def _transcribe(self, audio_data):
        try:
            temp_path = self.state_dir / "temp_audio.wav"
            audio_int16 = (audio_data * 32767).astype(np.int16)
            with wave.open(str(temp_path), "wb") as f:
                f.setnchannels(CHANNELS)
                f.setsampwidth(2)
                f.setframerate(SAMPLE_RATE)
                f.writeframes(audio_int16.tobytes())

            segments, _ = self.whisper_model.transcribe(str(temp_path))
            text = " ".join(seg.text for seg in segments).strip()
            temp_path.unlink(missing_ok=True)
            return text
        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
            return ""

    # ------------------------------------------------------------------
    # Sentence splitting (shared by streaming notifications and TTS)
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text):
        """Split text into sentence-sized chunks, merging short fragments."""
        parts = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        buf = ""
        for part in parts:
            buf = f"{buf} {part}".strip() if buf else part
            if len(buf) >= TTS_MIN_CHUNK_LEN:
                chunks.append(buf)
                buf = ""
        if buf:
            if chunks:
                chunks[-1] += " " + buf
            else:
                chunks.append(buf)
        return chunks

    # ------------------------------------------------------------------
    # Streaming LLM → TTS pipeline
    # ------------------------------------------------------------------

    def _stream_and_speak(self, text):
        """Stream LLM response, sending completed sentences to TTS as they arrive.

        Three concurrent stages connected by queues:
          1. LLM streaming → sentence_queue  (this thread)
          2. TTS generation ← sentence_queue → audio_queue  (worker thread)
          3. Audio playback ← audio_queue  (worker thread)

        Returns (full_response_text, tts_thread, play_thread) or (None, None, None) if aborted.
        The caller should post notifications and then join the threads.
        """
        sentence_queue: queue.Queue[Optional[str]] = queue.Queue()
        audio_queue: queue.Queue[Optional[Path]] = queue.Queue()
        full_response_parts: list[str] = []

        # --- TTS generation worker ---
        def tts_worker():
            idx = 0
            while True:
                sentence = sentence_queue.get()
                if sentence is None or self._abort_event.is_set():
                    audio_queue.put(None)  # signal playback to stop
                    return
                try:
                    samples, sr = self.kokoro.create(
                        sentence, voice=TTS_VOICE, speed=TTS_SPEED
                    )
                    path = self.chimes_dir / f"tts_chunk_{idx}.wav"
                    sf.write(str(path), samples, sr)
                    audio_queue.put(path)
                    idx += 1
                except Exception as e:
                    self.logger.error(f"TTS generation error: {e}")
                    audio_queue.put(None)
                    return

        # --- Playback worker ---
        def playback_worker():
            first = True
            prev_path = None
            while True:
                if self._abort_event.is_set():
                    return
                path = audio_queue.get()
                if path is None:
                    # Wait for last chunk to finish
                    if self._tts_process and self._tts_process.poll() is None:
                        self._tts_process.wait()
                    self._tts_process = None
                    if prev_path and prev_path.exists():
                        prev_path.unlink(missing_ok=True)
                    return

                if first:
                    self._set_waybar_status("speaking")
                    first = False

                # Wait for previous playback to finish
                if self._tts_process and self._tts_process.poll() is None:
                    self._tts_process.wait()

                if self._abort_event.is_set():
                    return

                # Clean up previous file
                if prev_path and prev_path.exists():
                    prev_path.unlink(missing_ok=True)

                self.logger.info(f"Playing TTS chunk: {path.name}")
                self._tts_process = subprocess.Popen(
                    ["pw-play", str(path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                prev_path = path

        # Start workers
        tts_thread = threading.Thread(target=tts_worker, daemon=True)
        play_thread = threading.Thread(target=playback_worker, daemon=True)

        if self.tts_available and self.kokoro:
            tts_thread.start()
            play_thread.start()

        # --- LLM streaming (runs in this thread) ---
        try:
            stream = self.openai.chat.completions.create(
                model=LLM_MODEL,
                max_tokens=LLM_MAX_TOKENS,
                stream=True,
                messages=[{"role": "user", "content": text}],
                extra_headers={"x-openclaw-agent-id": "main"},
                user="voice-assistant",
            )

            pending = ""
            for chunk in stream:
                if self._abort_event.is_set():
                    self.logger.info("Streaming aborted")
                    sentence_queue.put(None)
                    return None, None, None

                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    pending += delta.content

                    # Split completed sentences from pending text
                    while True:
                        match = re.search(r"[.!?](?:\s|$)", pending)
                        if not match:
                            break
                        # Split at end of sentence
                        split_pos = match.end()
                        sentence = pending[:split_pos].strip()
                        pending = pending[split_pos:]
                        if sentence:
                            full_response_parts.append(sentence)
                            self.logger.info(f"Sentence complete: {sentence[:60]}...")
                            if self.tts_available and self.kokoro:
                                sentence_queue.put(sentence)

            # Flush remaining text
            if pending.strip():
                sentence = pending.strip()
                full_response_parts.append(sentence)
                self.logger.info(f"Final fragment: {sentence[:60]}...")
                if self.tts_available and self.kokoro:
                    sentence_queue.put(sentence)

        except Exception as e:
            if self._abort_event.is_set():
                self.logger.info("Request aborted")
                sentence_queue.put(None)
                return None, None, None
            self.logger.error(f"OpenClaw API error: {e}")
            sentence_queue.put(None)
            return "Sorry, I couldn't process that request.", tts_thread, play_thread

        # Signal end of sentences
        sentence_queue.put(None)

        if self._abort_event.is_set():
            return None, None, None

        full_response = " ".join(full_response_parts).strip()
        if not full_response:
            return "Sorry, I got an empty response.", tts_thread, play_thread

        return full_response, tts_thread, play_thread

    def _speak_espeak(self, text):
        subprocess.run(
            ["espeak", "-s", "150", "-v", "en+f3", text],
            capture_output=True, check=False,
        )

    # ------------------------------------------------------------------
    # Main listen loop
    # ------------------------------------------------------------------

    async def _listen_loop(self):
        loop = asyncio.get_event_loop()
        stream = None

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

            # Open persistent audio stream
            if stream is None:
                try:
                    stream = self.audio.open(
                        format=AUDIO_FORMAT,
                        channels=CHANNELS,
                        rate=SAMPLE_RATE,
                        input=True,
                        input_device_index=self.input_device,
                        frames_per_buffer=CHUNK_SIZE,
                    )
                    self.logger.info("Audio stream opened")
                except OSError as e:
                    self.logger.error(f"Failed to open audio stream: {e}")
                    await asyncio.sleep(1)
                    continue

            if self.is_processing:
                # Drain buffer while processing
                try:
                    await loop.run_in_executor(None, self._read_chunk, stream, 0.1)
                except OSError:
                    pass
                await asyncio.sleep(0.05)
                continue

            # VAD check
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
                continue

            if self._detect_speech(audio_chunk):
                self._set_waybar_status("listening")
                self.logger.info("Speech detected, recording...")
                self._notify(
                    "🔊 Listening...",
                    replace_id=NOTIFY_ID_LISTENING,
                )

                full_audio = await loop.run_in_executor(
                    None, self._record_until_silence, stream, audio_chunk
                )

                self.is_processing = True
                self._set_waybar_status("thinking")
                asyncio.create_task(self._process_audio(full_audio))

            await asyncio.sleep(0.05)

    async def _process_audio(self, audio_data):
        loop = asyncio.get_event_loop()
        try:
            # Transcribe
            transcription = await loop.run_in_executor(None, self._transcribe, audio_data)
            if not transcription:
                self.is_processing = False
                return

            # Ding
            await loop.run_in_executor(None, self._play_chime, "processing")

            self.logger.info(f"Transcription: {transcription}")
            # Replace the "Listening..." notification with what you said
            self._notify(
                f"🎤 {transcription}",
                title="You Said",
                replace_id=NOTIFY_ID_LISTENING,
            )

            # Stream LLM response and speak concurrently
            response, tts_thread, play_thread = await loop.run_in_executor(
                None, self._stream_and_speak, transcription
            )

            if response is None:
                return

            self.logger.info(f"Response: {response}")
            # Show full reply notification immediately (TTS may still be playing)
            self._notify(
                f"🧙 {response}\n",
                title="Clawbook",
                timeout_ms=10000,
            )

            # Wait for TTS playback to finish
            if tts_thread and play_thread:
                await loop.run_in_executor(None, tts_thread.join)
                await loop.run_in_executor(None, play_thread.join)
            elif not self.tts_available and self.is_active:
                self._set_waybar_status("speaking")
                await loop.run_in_executor(None, self._speak_espeak, response)

        except Exception as e:
            self.logger.error(f"Processing error: {e}")
        finally:
            self.is_processing = False
            if self.is_active:
                self._set_waybar_status("ready")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _cleanup(self):
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
