#!/usr/bin/env python3
"""
assistant.py — Hyprland / NixOS real‑time voice client for OpenAI GPT‑4o
-----------------------------------------------------------------------
A lean reference implementation that:
  • streams microphone audio → OpenAI Realtime API
  • receives text + audio responses in <300 ms round‑trip
  • plays assistant audio reply via the system default sink

Key points:
  • Adds **OpenAI‑Beta: realtime=v1** header (mandatory)
  • Uses current preview **gpt‑4o‑realtime‑preview‑2024‑12‑17**
  • Sends configuration frame first
  • Works on `websockets` ≥14 by using **additional_headers**
  • Minimal dependencies (`websockets`, `sounddevice`, `simpleaudio`)

You can layer notifications, VAD tuning, IPC, etc. back on once the
WebSocket handshake is proven stable.
"""

import asyncio
import base64
import json
import os
import signal
import ssl
import sys
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread

import sounddevice as sd  # microphone + playback
import websockets  # type: ignore

# ------------------- User‑tweakable constants -------------------- #
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-realtime-preview-2024-12-17")
SAMPLE_RATE = int(os.getenv("ASSISTANT_SAMPLE_RATE", "16000"))
CHUNK_MS = int(os.getenv("ASSISTANT_CHUNK_MS", "40"))  # 40 ms frames
ENCODING = "linear_pcm"  # 16‑bit PCM little‑endian

if not OPENAI_API_KEY:
    print("❌  OPENAI_API_KEY env‑var not set — aborting", file=sys.stderr)
    sys.exit(1)

# -------------------------- Audio I/O ---------------------------- #
audio_q: "Queue[bytes]" = Queue(maxsize=256)
stop_flag = Event()

def _record() -> None:
    """Capture microphone audio and push int16 LE frames onto a Queue."""
    def callback(indata, _frames, _time, status):  # type: ignore[override]
        if status:
            print("⚠️  Audio status:", status, file=sys.stderr)
        if stop_flag.is_set():
            raise sd.CallbackStop()
        pcm = (indata * 32767).astype("<i2").tobytes()
        try:
            audio_q.put_nowait(pcm)
        except:
            pass

    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE,
                        blocksize=int(SAMPLE_RATE * CHUNK_MS / 1000),
                        dtype="float32", callback=callback):
        stop_flag.wait()

def start_recording() -> Thread:
    t = Thread(target=_record, daemon=True)
    t.start()
    return t

# Playback helper (optional, skip if simpleaudio missing)
try:
    import simpleaudio as sa
except ImportError:
    sa = None

def play_pcm(pcm: bytes, sample_rate: int = SAMPLE_RATE):
    if not sa:
        return
    wave = sa.WaveObject(pcm, 1, 2, sample_rate)
    wave.play()

# ----------------------- OpenAI session -------------------------- #
async def openai_session():
    url = f"wss://api.openai.com/v1/realtime?model={MODEL}"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }
    print("🔗 Connecting to", url)
    ssl_ctx = ssl.create_default_context()

    async with websockets.connect(
        url,
        additional_headers=headers,  # NOTE: works on websockets ≥14
        ssl=ssl_ctx,
        max_size=None,
        ping_interval=10,
        ping_timeout=20,
    ) as ws:
        # 1) config frame first
        await ws.send(json.dumps({
            "type": "configuration",
            "audio": {"encoding": ENCODING, "sample_rate": SAMPLE_RATE},
            "user_id": os.getenv("USER", "nixos"),
        }))
        print("✅ Sent configuration header")

        sender = asyncio.create_task(_pump_audio(ws))
        receiver = asyncio.create_task(_consume(ws))
        done, pending = await asyncio.wait(
            [sender, receiver], return_when=asyncio.FIRST_EXCEPTION)
        for task in pending:
            task.cancel()

async def _pump_audio(ws):
    while not stop_flag.is_set():
        try:
            pcm = audio_q.get(timeout=0.2)
        except Empty:
            continue
        await ws.send(json.dumps({
            "type": "audio",
            "audio": {
                "content": base64.b64encode(pcm).decode(),
                "encoding": ENCODING,
                "sample_rate": SAMPLE_RATE,
            },
        }))
    await ws.send(json.dumps({"type": "audio", "event": "end_of_stream"}))

async def _consume(ws):
    async for msg in ws:
        data = json.loads(msg)
        if data.get("type") == "audio":
            pcm = base64.b64decode(data["audio"]["content"])
            play_pcm(pcm, data["audio"].get("sample_rate", SAMPLE_RATE))
        elif data.get("type") == "transcript":
            print("👤", data.get("text", ""))
        elif data.get("type") == "assistant_response":
            print("🤖", data.get("text", ""))
        elif data.get("type") == "error":
            print("❌", data)

# -------------------------- Main -------------------------------- #
async def main():
    print("🚀 Starting Voice Assistant for Hyprland/NixOS")
    start_recording()

    task = asyncio.create_task(openai_session())
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_flag.set)

    await task
    print("🏁 Assistant session ended")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        stop_flag.set()
        print("👋 Bye!")
