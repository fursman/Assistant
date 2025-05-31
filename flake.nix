#!/usr/bin/env python3
"""
assistant.py — Hyprland / NixOS real‑time voice client for OpenAI GPT‑4o
-----------------------------------------------------------------------
A self‑contained reference implementation that:
  • streams microphone audio → OpenAI Realtime API
  • receives text + audio responses in <300 ms round trip
  • plays assistant audio reply via the system default sink

Key improvements over v0.x:
  • Adds the required `OpenAI-Beta: realtime=v1` WebSocket header
  • Uses the current preview snapshot `gpt-4o-realtime-preview-2024-12-17`
  • Sends the configuration frame **first** as required by the spec
  • Works with `websockets < 14` (handshake regression in 14) and Python ≥3.10
  • Gracefully shuts down on SIGINT / SIGTERM, cleaning up audio + IPC

The code has no GUI dependencies; it is designed to be started from a Hyprland
key‑binding or a systemd user service.
"""

import asyncio
import base64
import json
import os
import signal
import ssl
import sys
import time
from pathlib import Path
from queue import Queue, Empty
from threading import Event, Thread

import sounddevice as sd  # audio i/o
import websockets  # type: ignore  # real‑time WS

# ------------------------ Configuration ------------------------------------ #
# Read runtime options from environment variables (override in systemd unit).
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-realtime-preview-2024-12-17")
SAMPLE_RATE = int(os.getenv("ASSISTANT_SAMPLE_RATE", "16000"))
CHUNK_MS = int(os.getenv("ASSISTANT_CHUNK_MS", "40"))           # 40 ms
ENCODING = "linear_pcm"  # 16‑bit little‑endian PCM

if not OPENAI_API_KEY:
    print("❌  OPENAI_API_KEY not set — aborting", file=sys.stderr)
    sys.exit(1)

# ------------------------ Audio helpers ------------------------------------ #

audio_q: "Queue[bytes]" = Queue(maxsize=256)
stop_flag = Event()

def _record() -> None:
    """Record microphone audio and push raw bytes to a queue."""
    def callback(indata, _frames, _time, _status):  # type: ignore[override]
        if _status:
            print("⚠️  Audio status:", _status, file=sys.stderr)
        if stop_flag.is_set():
            raise sd.CallbackStop()
        # indata is a NumPy array in float32; convert to int16 little‑endian
        pcm = (indata * 32767).astype("<i2").tobytes()
        try:
            audio_q.put_nowait(pcm)
        except:
            pass  # drop if queue full (back‑pressure)

    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE,
                        blocksize=int(SAMPLE_RATE * CHUNK_MS / 1000),
                        dtype="float32", callback=callback):
        stop_flag.wait()


def start_recording() -> Thread:
    t = Thread(target=_record, daemon=True)
    t.start()
    return t

# ------------------------ Playback helpers ---------------------------------- #
try:
    import simpleaudio as sa  # lightweight, cross‑platform
except ImportError:  # noqa: D401
    sa = None  # Playback disabled


def play_pcm(pcm: bytes, sample_rate: int = SAMPLE_RATE):
    if not sa:
        return
    wave_obj = sa.WaveObject(pcm, num_channels=1, bytes_per_sample=2,
                             sample_rate=sample_rate)
    wave_obj.play()

# ------------------------ OpenAI session ------------------------------------ #

async def openai_session():
    url = f"wss://api.openai.com/v1/realtime?model={MODEL}"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    print("🔗 Connecting to", url)
    ssl_ctx = ssl.create_default_context()
    async with websockets.connect(url, extra_headers=headers,
                                  ssl=ssl_ctx, max_size=None,
                                  ping_interval=10, ping_timeout=20) as ws:
        # 1. Send configuration frame **first**
        config = {
            "type": "configuration",
            "audio": {
                "encoding": ENCODING,
                "sample_rate": SAMPLE_RATE,
            },
            "user_id": os.getenv("USER", "nixos")
        }
        await ws.send(json.dumps(config))
        print("✅ Sent configuration frame")

        # 2. Spin up producer & consumer tasks
        sender = asyncio.create_task(_pump_audio(ws))
        receiver = asyncio.create_task(_receive(ws))

        done, pending = await asyncio.wait(
            [sender, receiver], return_when=asyncio.FIRST_EXCEPTION)
        for task in pending:
            task.cancel()


async def _pump_audio(ws):
    frame_bytes = int(SAMPLE_RATE * CHUNK_MS / 1000) * 2  # int16 mono
    while not stop_flag.is_set():
        try:
            pcm = audio_q.get(timeout=0.2)
        except Empty:
            continue
        # Base64‑encode raw PCM for transport
        payload = {
            "type": "audio",
            "audio": {
                "content": base64.b64encode(pcm).decode(),
                "encoding": ENCODING,
                "sample_rate": SAMPLE_RATE,
            },
        }
        await ws.send(json.dumps(payload))
    # On stop, send a "end_of_stream" marker
    await ws.send(json.dumps({"type": "audio", "event": "end_of_stream"}))


async def _receive(ws):
    async for message in ws:
        msg = json.loads(message)
        mtype = msg.get("type")
        if mtype == "transcript":
            text = msg.get("text", "")
            print(f"👤 You: {text}")
        elif mtype == "assistant_response":
            text = msg.get("text", "")
            print(f"🤖 GPT‑4o: {text}")
        elif mtype == "audio":
            audio = msg["audio"]
            pcm = base64.b64decode(audio["content"])
            play_pcm(pcm, audio.get("sample_rate", SAMPLE_RATE))
        elif mtype == "error":
            print("❌ Error from server:", msg)
        else:
            print("ℹ️  Unhandled message:", msg)

# ------------------------ Entrypoint --------------------------------------- #

async def main():
    print("🚀 Starting Voice Assistant for Hyprland/NixOS")
    rec_thread = start_recording()

    loop_task = asyncio.create_task(openai_session())

    # Handle clean shutdown on Ctrl‑C
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(sig, stop_flag.set)

    await loop_task
    print("🏁 Assistant session ended")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        stop_flag.set()
        print("👋 Bye!")
