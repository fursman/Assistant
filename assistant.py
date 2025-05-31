#!/usr/bin/env python3
"""
Enhanced realtime voice assistant for Hyprland/NixOS integration.
----------------------------------------------------------------
Full‑feature edition (IPC, notify2, VAD tuning, CSV logging, key‑ring,
welcome sounds, etc.) **plus the Realtime‑beta handshake fixes**:

* Uses `OpenAI‑Beta: realtime=v1` (mandatory header)
* Pins model to `gpt‑4o‑realtime‑preview‑2024‑12‑17`
* Sends the required **configuration** frame first, before `session.update`
* Switches to `additional_headers=` so it works on `websockets` ≥14

Everything else from your original long script is preserved.
"""

import asyncio
import os
import json
import base64
import sys
import threading
import getpass
import csv
import datetime
import time
import queue
import signal
import atexit
from pathlib import Path
from enum import Enum
from typing import Optional

# Third‑party libs
import websockets  # >=14
import notify2
import sounddevice as sd
import numpy as np
from pydub import AudioSegment
import keyring

# ---------- Global configuration ----------------------------------------- #
PLAYBACK_SPEED = 1.04  # 4 % faster playback to feel snappier
SAMPLERATE = 16000      # Match OpenAI spec (16 kHz)
ASSISTANT_SAMPLERATE = 16000
CHANNELS = 1
BLOCKSIZE = 1600        # 100 ms @16 kHz mono int16
SOCKET_PATH = "/tmp/assistant.sock"
LOG_CSV_PATH = Path.home() / "assistant_interactions.csv"

API_URL = "wss://api.openai.com/v1/realtime"
DEFAULT_MODEL = "gpt-4o-realtime-preview-2024-12-17"
API_BETA_VERSION = "v1"          # << key fix

_shutdown_requested = False

class AssistantState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    SHUTTING_DOWN = "shutting_down"

# ----------------------- Helper utilities -------------------------------- #

def load_api_key():
    api_key = keyring.get_password("NixOSAssistant", "APIKey")
    if not api_key:
        api_key = getpass.getpass("Please enter your OpenAI API Key: ").strip()
        if api_key:
            keyring.set_password("NixOSAssistant", "APIKey", api_key)
        else:
            print("No API Key provided. Exiting.")
            sys.exit(1)
    return api_key

def play_audio_file(file_path: Path, volume: float = 1.0):
    try:
        audio = AudioSegment.from_file(file_path)
        if volume != 1.0:
            audio += 20 * np.log10(volume)
        samples = np.array(audio.get_array_of_samples()).reshape((-1, audio.channels))
        sd.play(samples, samplerate=audio.frame_rate)
        sd.wait()
    except Exception as e:
        print(f"Error playing audio file {file_path}: {e}")

def log_interaction(question: str, response: str):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        new_file = not LOG_CSV_PATH.exists()
        with open(LOG_CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["Timestamp", "Type", "Content"])
            writer.writerow([now, "Question", question])
            writer.writerow([now, "Response", response])
    except Exception as e:
        print(f"CSV log error: {e}")

def send_notification(title: str, message: str, timeout: int = 5000):
    try:
        if not notify2.is_initted():
            notify2.init("Assistant")
        n = notify2.Notification(title, message)
        n.set_timeout(timeout)
        n.show()
    except Exception as e:
        print(f"Notification error: {e}")

# ----------------------- Assistant session -------------------------------- #
class AssistantSession:
    def __init__(self, api_key: str, assets_dir: Path, welcome_file: Path, gotit_file: Path):
        self.api_key = api_key
        self.assets_dir = assets_dir
        self.welcome_file = welcome_file
        self.gotit_file = gotit_file
        self.api_url = API_URL

        self.state = AssistantState.IDLE
        self.shutdown_event = asyncio.Event()
        self.audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self.assistant_output_stream = None
        self.mic_stream = None

        self.current_response = ""
        self.current_question = ""
        self.response_id: Optional[str] = None

        self.tasks = []
        self.session_start_time = time.time()

    # ------------------- State helpers ----------------------------------- #
    def set_state(self, new_state: AssistantState):
        if self.state != new_state:
            print(f"State transition: {self.state.value} -> {new_state.value}")
            self.state = new_state

    # ------------------- Audio callback ---------------------------------- #
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        if self.state in (AssistantState.LISTENING, AssistantState.IDLE):
            try:
                self.audio_queue.put_nowait(indata.copy())
            except queue.Full:
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(indata.copy())
                except queue.Empty:
                    pass

    def flush_audio_queue(self):
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    # ------------------- WebSocket helpers ------------------------------- #
    async def send_audio(self, websocket):
        loop = asyncio.get_event_loop()
        try:
            while not self.shutdown_event.is_set():
                try:
                    indata = await asyncio.wait_for(loop.run_in_executor(None, self.audio_queue.get), 0.1)
                except asyncio.TimeoutError:
                    continue
                if self.state in (AssistantState.LISTENING, AssistantState.IDLE):
                    b64_audio = base64.b64encode(indata.astype(np.int16).tobytes()).decode()
                    await websocket.send(json.dumps({"type": "input_audio_buffer.append", "audio": b64_audio}))
        except asyncio.CancelledError:
            pass

    async def receive_messages(self, websocket):
        try:
            while not self.shutdown_event.is_set():
                try:
                    msg = await asyncio.wait_for(websocket.recv(), 0.1)
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    print("WebSocket closed")
                    self.shutdown_event.set()
                    break
                await self.handle_event(json.loads(msg))
        except asyncio.CancelledError:
            pass

    # ------------------- Event dispatcher -------------------------------- #
    async def handle_event(self, event):
        etype = event.get("type", "")
        if etype == "session.created":
            self.set_state(AssistantState.LISTENING)
        elif etype == "input_audio_buffer.speech_started":
            self.set_state(AssistantState.PROCESSING)
        elif etype == "conversation.item.input_audio_transcription.completed":
            t = event.get("transcript", "").strip()
            if t:
                self.current_question = t
                print(f"\n👤 You: {t}")
                send_notification("You said", t)
        elif etype == "response.created":
            self.response_id = event.get("response", {}).get("id")
            self.current_response = ""
            print("🤖 Thinking …")
        elif etype == "response.audio_transcript.delta":
            delta = event.get("delta", "")
            self.current_response += delta
            print(delta, end="", flush=True)
        elif etype == "response.audio.delta":
            await self.handle_audio_delta(event)
        elif etype == "response.audio.done":
            await self.finish_audio_playback()
        elif etype == "response.done":
            await self.handle_response_complete()
        elif etype == "error":
            print("❌ API error:", event.get("error"))

    async def handle_audio_delta(self, event):
        delta = event.get("delta", "")
        if not delta:
            return
        chunk = base64.b64decode(delta)
        if self.assistant_output_stream is None:
            self.set_state(AssistantState.SPEAKING)
            self.assistant_output_stream = sd.RawOutputStream(
                samplerate=ASSISTANT_SAMPLERATE, channels=1, dtype="int16", blocksize=BLOCKSIZE)
            self.assistant_output_stream.start()
            self.flush_audio_queue()
        self.assistant_output_stream.write(chunk)

    async def finish_audio_playback(self):
        if self.assistant_output_stream:
            await asyncio.sleep(0.2)
            self.assistant_output_stream.stop()
            self.assistant_output_stream.close()
            self.assistant_output_stream = None
            self.set_state(AssistantState.LISTENING)
            self.flush_audio_queue()

    async def handle_response_complete(self):
        print("\n✓ Response complete")
        if self.current_question and self.current_response:
            log_interaction(self.current_question, self.current_response)
            send_notification("Assistant", self.current_response[:120])
        self.current_question = self.current_response = ""

    # ------------------- IPC server -------------------------------------- #
    async def ipc_server(self):
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)
        async def handle(reader, writer):
            data = (await reader.read(100)).decode().strip()
            if data == "shutdown":
                self.shutdown_event.set()
                writer.write(b"ack")
                await writer.drain()
            writer.close()
            await writer.wait_closed()
        return await asyncio.start_unix_server(handle, path=SOCKET_PATH)

    # ------------------- Cleanup ----------------------------------------- #
    async def cleanup(self):
        print("🧹 Cleaning up …")
        for t in self.tasks:
            t.cancel()
        if self.assistant_output_stream:
            try:
                self.assistant_output_stream.abort(); self.assistant_output
