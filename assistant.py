#!/usr/bin/env python3
import asyncio
import os
import json
import websockets
import base64
import sys
import signal
from pathlib import Path
import getpass

import keyring
import sounddevice as sd
import numpy as np
from pydub import AudioSegment
import queue  # Thread-safe queue for audio data
import datetime
import csv
import notify2

# Audio configuration
SAMPLERATE = 16000            # Microphone input sample rate
ASSISTANT_SAMPLERATE = 24000  # Assistant's audio output sample rate
CHANNELS = 1
BLOCKSIZE = 2400             # Block size for audio recording
AUDIO_QUEUE = queue.Queue()  # Thread-safe queue for audio data

# Unix domain socket path for IPC
SOCKET_PATH = '/tmp/assistant.sock'

# Log file path
LOG_CSV_PATH = Path.home() / 'assistant_interactions.csv'

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

# Function to play an audio file
def play_audio(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        samples = np.array(audio.get_array_of_samples())
        # If stereo, reshape accordingly
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
        else:
            samples = samples.reshape((-1, 1))
        sd.play(samples, samplerate=audio.frame_rate)
        sd.wait()
    except Exception as e:
        print(f"Error playing audio file {file_path}: {e}")

# Function to log interactions to CSV
def log_interaction(question, response):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(LOG_CSV_PATH, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([now, "Question", question])
            writer.writerow([now, "Response", response])
    except Exception as e:
        print(f"Error logging interaction: {e}")

# Function to send desktop notifications
def send_notification(title, message):
    try:
        n = notify2.Notification(title, message)
        n.set_timeout(30000)
        n.show()
    except Exception as e:
        print(f"Notification error: {e}")

# Load the API key from keyring or prompt the user.
API_KEY = load_api_key()

# Define assets directory and audio file paths.
ASSETS_DIRECTORY = Path(os.getenv('AUDIO_ASSETS', Path(__file__).parent / "assets-audio"))
ASSETS_DIRECTORY.mkdir(parents=True, exist_ok=True)
WELCOME_FILE_PATH = ASSETS_DIRECTORY / "welcome.mp3"
GOTIT_FILE_PATH = ASSETS_DIRECTORY / "gotit.mp3"

# Ensure required audio files exist.
if not WELCOME_FILE_PATH.is_file():
    print(f"Welcome audio file not found at {WELCOME_FILE_PATH}")
    sys.exit(1)
if not GOTIT_FILE_PATH.is_file():
    print(f"Gotit audio file not found at {GOTIT_FILE_PATH}")
    sys.exit(1)

# Define the WebSocket URL and headers.
URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "OpenAI-Beta": "realtime=v1",
}

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio callback status: {status}", file=sys.stderr)
    AUDIO_QUEUE.put(indata.copy())

async def send_audio(websocket, shutdown_event):
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

async def receive_messages(websocket, shutdown_event):
    assistant_response_started = False
    assistant_response = ''
    user_question = ''
    assistant_audio_transcript = ''
    output_stream = None
    try:
        # Start an output stream for audio playback.
        output_stream = sd.OutputStream(samplerate=ASSISTANT_SAMPLERATE, channels=1, dtype='int16')
        output_stream.start()
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
                # Handle session.created silently or log as debug.
                print("Session created event received.")
            elif event_type == "session.updated":
                print(f"Session updated: {event.get('session')}")
            elif event_type == "conversation.item.input_audio_transcription.completed":
                transcript = event.get("transcript", "")
                user_question = transcript
                print(f"\nYou: {transcript}")
                send_notification("You said:", transcript)
            elif event_type == "response.audio.delta":
                delta = event.get("delta", "")
                try:
                    audio_data = base64.b64decode(delta)
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    output_stream.write(audio_array)
                except Exception as e:
                    print(f"Error processing audio delta: {e}")
            elif event_type == "response.text.delta":
                delta = event.get("delta", "")
                assistant_response += delta
                if not assistant_response_started:
                    print("\nAssistant:", end=' ', flush=True)
                    assistant_response_started = True
                print(delta, end='', flush=True)
            elif event_type == "response.audio_transcript.delta":
                assistant_audio_transcript += event.get("delta", "")
            elif event_type == "response.audio_transcript.done":
                assistant_audio_transcript = event.get("transcript", "")
            elif event_type == "response.done":
                assistant_response_started = False
                print("\nAssistant response complete.")
                log_interaction(user_question, assistant_response)
                send_notification("Assistant", assistant_response)
                user_question = ''
                assistant_response = ''
                assistant_audio_transcript = ''
            elif event_type == "error":
                error_info = event.get("error", {})
                error_message = error_info.get("message", "")
                print(f"Error: {error_message}")
                print(f"Full error info: {error_info}")
            else:
                # For other events, print for debugging.
                print(f"Unhandled event type: {event_type}")
                print(event)
    except asyncio.CancelledError:
        print("Message receiving task cancelled.")
    except Exception as e:
        print(f"Error in receive_messages: {e}")
    finally:
        if output_stream:
            output_stream.abort()
            output_stream.close()
            print("Audio playback stopped.")

async def ipc_server(shutdown_event):
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
        async with websockets.connect(URL, extra_headers=HEADERS, ping_interval=20, ping_timeout=60) as websocket:
            print("Connected to OpenAI Realtime Assistant API.")
            play_audio(WELCOME_FILE_PATH)
            # Send a session update without the unsupported "enabled" key.
            session_update = {
                "type": "session.update",
                "session": {
                    "modalities": ["audio", "text"],
                    "voice": "shimmer",
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "model": "whisper-1"
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.75,
                        "prefix_padding_ms": 100,
                        "silence_duration_ms": 1000,
                    },
                },
            }
            await websocket.send(json.dumps(session_update))
            # Start the audio input stream.
            stream = sd.InputStream(samplerate=SAMPLERATE, channels=CHANNELS, dtype='int16',
                                    callback=audio_callback, blocksize=BLOCKSIZE)
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
            # Wait a bit for the other instance to shut down.
            for _ in range(20):
                if not os.path.exists(SOCKET_PATH):
                    break
                asyncio.run(asyncio.sleep(0.05))
            play_audio(GOTIT_FILE_PATH)
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
