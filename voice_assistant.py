#!/usr/bin/env python3
"""
Hyprland Voice Assistant
A voice-activated assistant for Hyprland with VAD, STT, LLM, and TTS.

Pipeline: VAD → Whisper STT → OpenClaw LLM (streaming via WS) → Kokoro TTS (streaming) → PipeWire playback

All communication with the OpenClaw gateway uses the native WebSocket protocol.
Thinking events stream as 🧠 notifications; response sentences stream to TTS.
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
import uuid
import wave
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

NOTIFY_ID_LISTENING = 59001

CHIME_SAMPLE_RATE = 44100
CHIME_NOTE_DURATION = 0.2


class VoiceAssistant:
    def __init__(self):
        self.is_active = False
        self.is_processing = False
        self._abort_event = threading.Event()
        self._tts_process: Optional[subprocess.Popen] = None

        # WebSocket state
        self._ws: Optional[websocket.WebSocket] = None
        self._ws_lock = threading.Lock()
        self._ws_connected = threading.Event()
        self._ws_req_id = 0
        self._ws_req_lock = threading.Lock()

        # Per-run state for streaming
        self._active_run_id: Optional[str] = None
        self._sentence_queue: Optional[queue.Queue] = None
        self._thinking_text = ""
        self._assistant_pending = ""
        self._run_done_event = threading.Event()
        self._run_response_parts: list[str] = []

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

        signal.signal(signal.SIGUSR1, self._toggle_handler)
        self.pid_file.write_text(str(os.getpid()))

        # Start WebSocket connection (handles thinking + queries + reasoning setup)
        self._start_ws()

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
            self.logger.info("Chimes already cached")
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
    # Notifications (swaync)
    # ------------------------------------------------------------------

    def _notify(self, message, title="Voice Assistant", replace_id=None, timeout_ms=None):
        cmd = ["notify-send", title, message]
        if replace_id is not None:
            cmd.extend(["-r", str(replace_id)])
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
    # WebSocket — single persistent connection for everything
    # ------------------------------------------------------------------

    def _start_ws(self):
        thread = threading.Thread(target=self._ws_loop, daemon=True)
        thread.start()
        # Wait for initial connection
        self._ws_connected.wait(timeout=10)

    def _ws_loop(self):
        """Persistent WS connection with auto-reconnect."""
        token = os.getenv("OPENCLAW_GATEWAY_TOKEN", "")
        while True:
            try:
                self._ws_connect(token)
            except Exception as e:
                self.logger.warning(f"WS error: {e}")
            self._ws_connected.clear()
            time.sleep(3)

    def _next_req_id(self):
        with self._ws_req_lock:
            self._ws_req_id += 1
            return str(self._ws_req_id)

    def _ws_connect(self, token):
        ws = websocket.WebSocket()
        ws.connect(GATEWAY_URL)
        self.logger.info("WS connected")

        # Wait for challenge
        challenge = json.loads(ws.recv())
        if challenge.get("event") != "connect.challenge":
            self.logger.warning(f"Expected connect.challenge, got: {challenge.get('event')}")
            ws.close()
            return

        # Handshake
        ws.send(json.dumps({
            "type": "req",
            "id": "connect",
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
        }))

        resp = json.loads(ws.recv())
        if not (resp.get("type") == "res" and resp.get("ok")):
            self.logger.warning(f"WS handshake failed: {resp}")
            ws.close()
            return
        self.logger.info("WS handshake OK")

        self._ws = ws
        self._ws_connected.set()

        # Enable extended thinking (low budget) and stream thinking tokens
        ws.send(json.dumps({
            "type": "req",
            "id": "set-reasoning",
            "method": "sessions.patch",
            "params": {
                "key": GATEWAY_SESSION_KEY,
                "thinking": "low",
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
                msg = json.loads(raw)
                self._handle_ws_message(msg)
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
        # Signal any waiting run so it doesn't hang
        if self._active_run_id:
            if self._sentence_queue:
                self._sentence_queue.put(None)
            self._run_done_event.set()
        self.logger.info("WS disconnected")

    def _handle_ws_message(self, msg):
        msg_type = msg.get("type")

        # Handle RPC responses
        if msg_type == "res":
            req_id = msg.get("id")
            if req_id == "set-reasoning":
                if msg.get("ok"):
                    self.logger.info("Reasoning level set to 'stream' via WS")
                else:
                    self.logger.warning(f"Failed to set reasoning: {msg.get('error')}")
            elif req_id and req_id.startswith("voice-"):
                payload = msg.get("payload", {})
                status = payload.get("status", "")
                self.logger.info(f"Query {req_id}: status={status} payload_keys={list(payload.keys())}")
                # Capture runId from the chat.send response if available
                resp_run_id = payload.get("runId")
                if resp_run_id and self._active_req_id == req_id:
                    self._active_run_id = resp_run_id
                    self.logger.info(f"Captured run ID from response: {resp_run_id}")
            return

        # Handle agent events
        if msg_type == "event" and msg.get("event") == "agent":
            self._handle_agent_event(msg.get("payload", {}))

    def _handle_agent_event(self, payload):
        stream = payload.get("stream")
        data = payload.get("data", {})
        run_id = payload.get("runId")

        # Debug: log every agent event type we receive
        delta_preview = ""
        if data.get("delta"):
            delta_preview = f" delta={data['delta'][:60]}..."
        elif data.get("phase"):
            delta_preview = f" phase={data['phase']}"
        self.logger.info(f"Agent event: stream={stream} runId={run_id}{delta_preview}")

        # Track the run ID from the first agent event after we send a query
        if self._active_run_id is None:
            if self._active_req_id and run_id:
                self._active_run_id = run_id
                self.logger.info(f"Captured run ID: {run_id}")
            else:
                return
        elif run_id != self._active_run_id:
            return

        if stream == "thinking":
            delta = data.get("delta", "")
            if delta:
                self._thinking_text += delta
                # Fire notification per sentence
                while True:
                    match = re.search(r"[.!?](?:\s|$)", self._thinking_text)
                    if not match:
                        break
                    pos = match.end()
                    sentence = self._thinking_text[:pos].strip()
                    self._thinking_text = self._thinking_text[pos:]
                    if sentence:
                        self._notify(
                            f"🧠 {sentence}\n",
                            title="Thinking...",
                            timeout_ms=8000,
                        )
                        self.logger.info(f"Thinking: {sentence[:60]}...")

        elif stream == "assistant":
            delta = data.get("delta", "")
            if delta:
                self._assistant_pending += delta
                # Split completed sentences and queue for TTS
                while True:
                    match = re.search(r"[.!?](?:\s|$)", self._assistant_pending)
                    if not match:
                        break
                    pos = match.end()
                    sentence = self._assistant_pending[:pos].strip()
                    self._assistant_pending = self._assistant_pending[pos:]
                    if sentence:
                        self._run_response_parts.append(sentence)
                        self.logger.info(f"Sentence: {sentence[:60]}...")
                        if self._sentence_queue:
                            self._sentence_queue.put(sentence)

        elif stream == "lifecycle":
            phase = data.get("phase")
            if phase in ("end", "error"):
                # Flush remaining thinking
                if self._thinking_text.strip():
                    self._notify(
                        f"🧠 {self._thinking_text.strip()}\n",
                        title="Thinking...",
                        timeout_ms=8000,
                    )
                    self._thinking_text = ""

                # Flush remaining assistant text
                if self._assistant_pending.strip():
                    sentence = self._assistant_pending.strip()
                    self._run_response_parts.append(sentence)
                    self.logger.info(f"Final fragment: {sentence[:60]}...")
                    if self._sentence_queue:
                        self._sentence_queue.put(sentence)
                    self._assistant_pending = ""

                # Signal end of TTS sentences
                if self._sentence_queue:
                    self._sentence_queue.put(None)

                self._run_done_event.set()

    # ------------------------------------------------------------------
    # Query via WebSocket
    # ------------------------------------------------------------------

    def _query_and_speak(self, text):
        """Send query via WS, stream thinking as notifications, stream response to TTS.
        Returns full response text or None if aborted."""
        req_id = f"voice-{uuid.uuid4().hex[:12]}"
        self._active_run_id = None  # Will be set from first agent event
        self._active_req_id = req_id
        self._thinking_text = ""
        self._assistant_pending = ""
        self._run_response_parts = []
        self._run_done_event.clear()

        # Set up TTS pipeline
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
                    path = self.chimes_dir / f"tts_chunk_{idx}.wav"
                    sf.write(str(path), samples, sr)
                    audio_q.put(path)
                    idx += 1
                except Exception as e:
                    self.logger.error(f"TTS error: {e}")
                    audio_q.put(None)
                    return

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
                self.logger.info(f"Playing: {path.name}")
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

        # Send query via WS
        if not self._ws_connected.is_set() or not self._ws:
            self.logger.error("WS not connected, cannot send query")
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
        except Exception as e:
            self.logger.error(f"Failed to send query: {e}")
            self._sentence_queue = None
            sentence_q.put(None)
            return "Sorry, I couldn't send that query."

        # Wait for run to complete (or abort)
        while not self._run_done_event.wait(timeout=0.5):
            if self._abort_event.is_set():
                self._sentence_queue = None
                sentence_q.put(None)
                self._active_run_id = None
                return None

        self._sentence_queue = None
        self._active_run_id = None
        self._active_req_id = None

        if self._abort_event.is_set():
            return None

        full_response = " ".join(self._run_response_parts).strip()
        if not full_response:
            full_response = "Sorry, I got an empty response."

        # Show reply notification now (TTS may still be playing)
        self.logger.info(f"Response: {full_response}")
        self._notify(f"🧙 {full_response}\n", title="Clawbook", timeout_ms=10000)

        # Wait for TTS to finish
        if tts_thread:
            tts_thread.join()
        if play_thread:
            play_thread.join()

        return full_response

    # ------------------------------------------------------------------
    # Abort / Toggle
    # ------------------------------------------------------------------

    def _abort_inflight(self):
        self._abort_event.set()

        if self._tts_process and self._tts_process.poll() is None:
            self._tts_process.kill()
            self.logger.info("Killed TTS playback")

        subprocess.run(["pkill", "-f", "pw-play.*tts_chunk"], capture_output=True, check=False)

        # Send /stop via WS
        def send_stop():
            try:
                if self._ws and self._ws_connected.is_set():
                    with self._ws_lock:
                        self._ws.send(json.dumps({
                            "type": "req",
                            "id": self._next_req_id(),
                            "method": "chat.send",
                            "params": {
                                "sessionKey": GATEWAY_SESSION_KEY,
                                "message": "/stop",
                                "deliver": False,
                                "idempotencyKey": f"stop-{uuid.uuid4().hex[:8]}",
                            },
                        }))
                    self.logger.info("Sent /stop via WS")
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

            if self.is_processing:
                try:
                    await loop.run_in_executor(None, self._read_chunk, stream, 0.1)
                except OSError:
                    pass
                await asyncio.sleep(0.05)
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
                continue

            if self._detect_speech(audio_chunk):
                self._set_waybar_status("listening")
                self.logger.info("Speech detected, recording...")
                self._notify("🔊 Listening...", replace_id=NOTIFY_ID_LISTENING)

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
            transcription = await loop.run_in_executor(None, self._transcribe, audio_data)
            if not transcription:
                self.is_processing = False
                return

            await loop.run_in_executor(None, self._play_chime, "processing")

            self.logger.info(f"Transcription: {transcription}")
            self._notify(f"🎤 {transcription}", title="You Said", replace_id=NOTIFY_ID_LISTENING)

            # Query and speak via WS (blocking — runs TTS pipeline internally)
            response = await loop.run_in_executor(None, self._query_and_speak, transcription)

            if response is None:
                return

            # espeak fallback
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
