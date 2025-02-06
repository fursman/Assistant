#!/usr/bin/env python3
"""
Optimized assistant script for realtime streaming and improved microphone input.
Modifications:
  • Streams assistant audio as soon as audio deltas arrive using a RawOutputStream.
  • Removes aggressive muting in the microphone callback to avoid cutting off user input.
  • After assistant playback completes, flushes the mic audio queue and temporarily ignores further mic data.
  
References:
  OpenAI Realtime API docs: :contentReference[oaicite:2]{index=2}
  Streaming completions guidelines: :contentReference[oaicite:3]{index=3}
"""

import asyncio
import os
import json
import websockets
import base64
import sys
import signal
from pathlib import Path
import getpass
import io
import threading
import keyring
import sounddevice as sd
import numpy as np
from pydub import AudioSegment
import queue  # Thread-safe queue for audio data
import datetime
import csv
import notify2
import time  # used for time.time() in our ignore-mic timer

# Global configuration and state
PLAYBACK_SPEED = 1.04  # Increase playback speed by 4%; set to 1.0 for normal speed.
SAMPLERATE = 16000           # Microphone input sample rate (Hz)
ASSISTANT_SAMPLERATE = 24000  # Assistant's audio output sample rate (Hz)
CHANNELS = 1
BLOCKSIZE = 2400  # Block size for audio recording
AUDIO_QUEUE = queue.Queue()  # Queue for mic audio data

# Global flag to temporarily ignore mic input
IGNORE_MIC = False

# Unix domain socket path for IPC
SOCKET_PATH = '/tmp/assistant.sock'

# Log file path (CSV)
LOG_CSV_PATH = Path.home() / 'assistant_interactions.csv'

def load_api_key():
    """Load the API key from keyring or prompt the user."""
    api_key = keyring.get_password("NixOSAssistant", "APIKey")
    if not api_key:
        api_key = getpass.getpass("Please enter your OpenAI API Key: ").strip()
        if api_key:
            keyring.set_password("NixOSAssistant", "APIKey", api_key)
        else:
            print("No API Key provided. Exiting.")
            sys.exit(1)
    return api_key

def play_audio_file(file_path):
    """Play an audio file using pydub and sounddevice."""
    try:
        audio = AudioSegment.from_file(file_path)
        samples = np.array(audio.get_array_of_samples())
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
        else:
            samples = samples.reshape((-1, 1))
        sd.play(samples, samplerate=audio.frame_rate)
        sd.wait()
    except Exception as e:
        print(f"Error playing audio file {file_path}: {e}")

def log_interaction(question, response):
    """Log a conversation interaction (question and reply) to a CSV file."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(LOG_CSV_PATH, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([now, "Question", question])
            writer.writerow([now, "Response", response])
    except Exception as e:
        print(f"Error logging interaction: {e}")

def send_notification(title, message):
    """Send a desktop notification using notify2."""
    try:
        n = notify2.Notification(title, message)
        n.set_timeout(30000)
        n.show()
    except Exception as e:
        print(f"Notification error: {e}")

# Load API key and set up assets
API_KEY = load_api_key()
ASSETS_DIRECTORY = Path(os.getenv('AUDIO_ASSETS', Path(__file__).parent / "assets-audio"))
ASSETS_DIRECTORY.mkdir(parents=True, exist_ok=True)
WELCOME_FILE_PATH = ASSETS_DIRECTORY / "welcome.mp3"
GOTIT_FILE_PATH = ASSETS_DIRECTORY / "gotit.mp3"

if not WELCOME_FILE_PATH.is_file():
    print(f"Welcome audio file not found at {WELCOME_FILE_PATH}")
    sys.exit(1)
if not GOTIT_FILE_PATH.is_file():
    print(f"Gotit audio file not found at {GOTIT_FILE_PATH}")
    sys.exit(1)

# WebSocket URL and headers for OpenAI Realtime API
URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "OpenAI-Beta": "realtime=v1",
}

def change_speed(sound, speed=1.0):
    """
    Change the playback speed of an AudioSegment.
    This adjusts the frame_rate to speed up (or slow down) playback.
    Note: This method will also alter the pitch.
    """
    sound_altered = sound._spawn(sound.raw_data, overrides={
        "frame_rate": int(sound.frame_rate * speed)
    })
    return sound_altered.set_frame_rate(sound.frame_rate)

def flush_audio_queue():
    """Empty the microphone audio queue."""
    while not AUDIO_QUEUE.empty():
        try:
            AUDIO_QUEUE.get_nowait()
        except queue.Empty:
            break

def disable_mic_temporarily(duration_sec):
    """Temporarily ignore microphone input for duration_sec seconds."""
    global IGNORE_MIC
    IGNORE_MIC = True
    def enable_mic():
        global IGNORE_MIC
        IGNORE_MIC = False
    timer = threading.Timer(duration_sec, enable_mic)
    timer.start()

# --- Updated microphone input ---
def audio_callback(indata, frames, time_info, status):
    """
    Audio callback for recording.
    Always enqueues microphone data UNLESS we are temporarily ignoring mic input.
    """
    if status:
        print(f"Audio callback status: {status}", file=sys.stderr)
    if IGNORE_MIC:
        return  # Skip enqueuing data if we're ignoring mic input
    AUDIO_QUEUE.put(indata.copy())

# Global variable for streaming assistant output.
assistant_output_stream = None  # Will hold our output stream for assistant audio

# --- Updated send_audio: mic audio is always sent ---
async def send_audio(websocket, shutdown_event):
    """Continuously send recorded audio chunks to the WebSocket."""
    loop = asyncio.get_event_loop()
    try:
        while not shutdown_event.is_set():
            indata = await loop.run_in_executor(None, AUDIO_QUEUE.get)
            audio_bytes = indata.tobytes()
            print(f"Sending {len(audio_bytes)} bytes of audio data.")
            base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
            audio_event = {
                "type": "input_audio_buffer.append",
                "audio": base64_audio,
            }
            await websocket.send(json.dumps(audio_event))
    except asyncio.CancelledError:
        print("Audio sending task cancelled.")
    except Exception as e:
        print(f"Error in send_audio: {e}")

# --- Updated receive_messages to stream assistant audio immediately ---
async def receive_messages(websocket, shutdown_event):
    """
    Receive events from the WebSocket.
    Streams assistant audio immediately as chunks arrive.
    Accumulates transcript text concurrently.
    """
    global assistant_output_stream
    assistant_response = ''
    user_question = ''
    try:
        while not shutdown_event.is_set():
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                print("WebSocket connection closed.")
                shutdown_event.set()
                break
            try:
                event = json.loads(message)
            except json.JSONDecodeError:
                print("Received invalid JSON message.")
                continue
            event_type = event.get("type", "")
            if event_type == "session.created":
                print("Session created event received.")
            elif event_type == "session.updated":
                print(f"Session updated: {event.get('session')}")
            elif event_type == "conversation.item.input_audio_transcription.completed":
                transcript = event.get("transcript", "")
                user_question = transcript
                print(f"\n You: {transcript}")
                send_notification("You said:", transcript)
            elif event_type == "response.audio.delta":
                delta = event.get("delta", "")
                if delta:
                    try:
                        chunk = base64.b64decode(delta)
                        # Create output stream on first delta if not already created.
                        if assistant_output_stream is None:
                            assistant_output_stream = sd.RawOutputStream(
                                samplerate=ASSISTANT_SAMPLERATE,
                                channels=1,
                                dtype='int16'
                            )
                            assistant_output_stream.start()
                        # Write chunk immediately to output stream.
                        assistant_output_stream.write(chunk)
                    except Exception as e:
                        print(f"Error processing audio delta: {e}")
            elif event_type == "response.text.delta":
                delta = event.get("delta", "")
                assistant_response += delta
                print(f"\n Assistant (text): {delta}", end='', flush=True)
            elif event_type == "response.audio_transcript.delta":
                transcript_delta = event.get("delta", "")
                assistant_response += transcript_delta
                print(f"\n Assistant (audio transcript): {transcript_delta}", end='', flush=True)
            elif event_type == "response.audio_transcript.done":
                print("\n Assistant audio transcript complete.")
            elif event_type in ["response.audio.done", "response.content_part.done"]:
                # End of assistant audio response: wait briefly to allow any buffered audio to play,
                # then stop and close the stream.
                if assistant_output_stream is not None:
                    # Wait 300ms to allow remaining audio to be played
                    sd.sleep(300)
                    try:
                        assistant_output_stream.stop()
                        assistant_output_stream.close()
                        assistant_output_stream = None
                        print("Assistant audio playback complete.")
                    except Exception as e:
                        print(f"Error finishing assistant audio playback: {e}")
                    # Flush mic input that might have captured assistant output
                    flush_audio_queue()
                    # Temporarily ignore mic input for 0.5 seconds to let the environment settle
                    disable_mic_temporarily(0.5)
            elif event_type == "response.done":
                print("\n Assistant response complete.")
                log_interaction(user_question, assistant_response)
                send_notification("Assistant", assistant_response if assistant_response else "No transcript available.")
                # Reset conversation variables.
                assistant_response = ''
                user_question = ''
            elif event_type in [
                "input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped",
                "input_audio_buffer.committed", "conversation.item.created",
                "response.created", "rate_limits.updated", "response.output_item.added",
                "response.output_item.done"
            ]:
                print(f"Unhandled event type: {event_type}")
                print(event)
            else:
                print(f"Unhandled event type: {event_type}")
                print(event)
    except asyncio.CancelledError:
        print("Message receiving task cancelled.")
    except Exception as e:
        print(f"Error in receive_messages: {e}")

async def ipc_server(shutdown_event):
    """Set up an IPC server to allow external shutdown commands."""
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)
    async def handle_client(reader, writer):
        try:
            data = await reader.read(100)
            message = data.decode().strip()
            print(f"Received IPC message: {message}")
            if message == "shutdown":
                shutdown_event.set()
                writer.write(b"ack")
                await writer.drain()
        except Exception as e:
            print(f"Error in IPC server: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
    server = await asyncio.start_unix_server(handle_client, path=SOCKET_PATH)
    return server

async def realtime_api():
    send_task = None
    receive_task = None
    ipc_srv = None
    stream = None
    shutdown_event = asyncio.Event()
    try:
        ipc_srv = await ipc_server(shutdown_event)
        async with websockets.connect(URL, extra_headers=HEADERS) as websocket:
            print("Connected to OpenAI Realtime Assistant API.")
            play_audio_file(WELCOME_FILE_PATH)
            # Send session update with instructions.
            session_update = {
                "type": "session.update",
                "session": {
                    "modalities": ["audio", "text"],
                    "voice": "shimmer",
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "input_audio_transcription": {"model": "whisper-1"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.75,
                        "prefix_padding_ms": 100,
                        "silence_duration_ms": 1000,
                        "create_response": True
                    },
                    "instructions": (
                        "Your knowledge cutoff is 2023-10. You are a helpful, witty, and friendly AI. "
                        "Act like a human, but remember that you aren't a human. "
                        "Speak quickly in your responses."
                    )
                }
            }
            await websocket.send(json.dumps(session_update))
            # Start audio input stream; microphone always records.
            stream = sd.InputStream(
                samplerate=SAMPLERATE,
                channels=CHANNELS,
                dtype='int16',
                callback=audio_callback,
                blocksize=BLOCKSIZE
            )
            stream.start()
            send_task = asyncio.create_task(send_audio(websocket, shutdown_event))
            receive_task = asyncio.create_task(receive_messages(websocket, shutdown_event))
            await shutdown_event.wait()
            print("Shutdown event received.")
    except Exception as e:
        print(f"Exception in realtime_api: {e}")
    finally:
        print("Cleaning up...")
        if send_task and not send_task.done():
            send_task.cancel()
        if receive_task and not receive_task.done():
            receive_task.cancel()
        if stream and stream.active:
            stream.abort()
            stream.close()
            print("Audio recording stopped.")
        if ipc_srv:
            ipc_srv.close()
            await ipc_srv.wait_closed()
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)

async def send_shutdown_command():
    """Send a shutdown command via the IPC socket to stop any running instance."""
    try:
        reader, writer = await asyncio.open_unix_connection(SOCKET_PATH)
        writer.write(b"shutdown")
        await writer.drain()
        data = await reader.read(100)
        if data.decode().strip() == "ack":
            print("Shutdown command acknowledged.")
        writer.close()
        await writer.wait_closed()
    except Exception as e:
        print(f"Error sending shutdown command: {e}")

def main():
    print("Starting the assistant.")
    notify2.init('Assistant')
    if os.path.exists(SOCKET_PATH):
        print("Another instance detected. Sending shutdown command.")
        try:
            asyncio.run(send_shutdown_command())
            # Wait briefly for the other instance to shut down.
            for _ in range(20):
                if not os.path.exists(SOCKET_PATH):
                    break
                asyncio.run(asyncio.sleep(0.05))
            play_audio_file(GOTIT_FILE_PATH)
        except Exception as e:
            print(f"Error communicating with the running instance: {e}")
        sys.exit(0)
    else:
        try:
            asyncio.run(realtime_api())
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
