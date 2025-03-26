#!/usr/bin/env python3
"""
Refactored assistant script for realtime streaming and improved microphone input.
This version uses a new realtime API endpoint and updated model parameters,
while preserving all functionality from the original script.

Modifications:
  • Organized code into a dedicated AssistantSession class for clarity.
  • Updated API endpoint and session update payload to reflect new realtime API docs.
  • Retained audio streaming, CSV logging, notifications, and IPC shutdown.
  
References:
  OpenAI Next Generation Audio Models: https://openai.com/index/introducing-our-next-generation-audio-models/
  OpenAI Realtime API: https://openai.com/index/introducing-the-realtime-api/
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
from pathlib import Path

import websockets
import notify2
import sounddevice as sd
import numpy as np
from pydub import AudioSegment
import keyring

# Global configuration
PLAYBACK_SPEED = 1.04        # Increase playback speed by 4%
SAMPLERATE = 16000           # Microphone sample rate (Hz)
ASSISTANT_SAMPLERATE = 24000 # Assistant's audio output sample rate (Hz)
CHANNELS = 1
BLOCKSIZE = 2400           # Block size for recording
SOCKET_PATH = '/tmp/assistant.sock'
LOG_CSV_PATH = Path.home() / 'assistant_interactions.csv'

# New API configuration (updated per the realtime API documentation)
API_URL = "wss://api.openai.com/v1/realtime/assistant"  # Updated endpoint
DEFAULT_MODEL = "gpt-4-audio-realtime-2025-03-26"         # Updated model name

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

def change_speed(sound, speed=1.0):
    """
    Change the playback speed of an AudioSegment.
    Note: This method alters both speed and pitch.
    """
    sound_altered = sound._spawn(sound.raw_data, overrides={
        "frame_rate": int(sound.frame_rate * speed)
    })
    return sound_altered.set_frame_rate(sound.frame_rate)

class AssistantSession:
    def __init__(self, api_key, assets_directory, welcome_file, gotit_file):
        self.api_key = api_key
        self.assets_directory = assets_directory
        self.welcome_file = welcome_file
        self.gotit_file = gotit_file
        self.api_url = API_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            # Any additional headers as required by the new API can be added here.
        }
        self.shutdown_event = asyncio.Event()
        self.audio_queue = queue.Queue()  # Thread-safe queue for mic data
        self.ignore_mic = False
        self.assistant_output_stream = None
        self.mic_stream = None

    def audio_callback(self, indata, frames, time_info, status):
        """
        Callback for audio recording. Enqueues mic data unless temporarily ignoring input.
        """
        if status:
            print(f"Audio callback status: {status}", file=sys.stderr)
        if self.ignore_mic:
            return
        self.audio_queue.put(indata.copy())

    def disable_mic_temporarily(self, duration_sec):
        """Temporarily ignore microphone input for the given duration."""
        self.ignore_mic = True
        def enable_mic():
            self.ignore_mic = False
        timer = threading.Timer(duration_sec, enable_mic)
        timer.start()

    def flush_audio_queue(self):
        """Empty the microphone audio queue."""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    async def send_audio(self, websocket):
        """
        Continuously send recorded audio chunks to the websocket.
        """
        loop = asyncio.get_event_loop()
        try:
            while not self.shutdown_event.is_set():
                indata = await loop.run_in_executor(None, self.audio_queue.get)
                audio_bytes = indata.tobytes()
                print(f"Sending {len(audio_bytes)} bytes of audio data.")
                b64_audio = base64.b64encode(audio_bytes).decode('utf-8')
                event = {
                    "type": "input_audio_buffer.append",
                    "audio": b64_audio,
                }
                await websocket.send(json.dumps(event))
        except asyncio.CancelledError:
            print("Audio sending task cancelled.")
        except Exception as e:
            print(f"Error in send_audio: {e}")

    async def receive_messages(self, websocket):
        """
        Receive events from the websocket.
        Streams assistant audio as soon as chunks arrive and handles transcript updates.
        """
        assistant_response = ""
        user_question = ""
        try:
            while not self.shutdown_event.is_set():
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    print("WebSocket connection closed.")
                    self.shutdown_event.set()
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
                            # Start audio output stream if not already running.
                            if self.assistant_output_stream is None:
                                self.assistant_output_stream = sd.RawOutputStream(
                                    samplerate=ASSISTANT_SAMPLERATE,
                                    channels=1,
                                    dtype='int16'
                                )
                                self.assistant_output_stream.start()
                            self.assistant_output_stream.write(chunk)
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
                    # End of audio response: allow buffered audio to play, then clean up.
                    if self.assistant_output_stream is not None:
                        sd.sleep(300)
                        try:
                            self.assistant_output_stream.stop()
                            self.assistant_output_stream.close()
                            self.assistant_output_stream = None
                            print("Assistant audio playback complete.")
                        except Exception as e:
                            print(f"Error finishing assistant audio playback: {e}")
                        self.flush_audio_queue()
                        self.disable_mic_temporarily(0.5)
                elif event_type == "response.done":
                    print("\n Assistant response complete.")
                    log_interaction(user_question, assistant_response)
                    send_notification("Assistant", assistant_response if assistant_response else "No transcript available.")
                    assistant_response = ""
                    user_question = ""
                else:
                    print(f"Unhandled event type: {event_type}")
                    print(event)
        except asyncio.CancelledError:
            print("Message receiving task cancelled.")
        except Exception as e:
            print(f"Error in receive_messages: {e}")

    async def ipc_server(self):
        """
        Set up an IPC server on a UNIX domain socket to handle external shutdown commands.
        """
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)

        async def handle_client(reader, writer):
            try:
                data = await reader.read(100)
                message = data.decode().strip()
                print(f"Received IPC message: {message}")
                if message == "shutdown":
                    self.shutdown_event.set()
                    writer.write(b"ack")
                    await writer.drain()
            except Exception as e:
                print(f"Error in IPC server: {e}")
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_unix_server(handle_client, path=SOCKET_PATH)
        return server

    async def run(self):
        """
        Main entry point for the session. Connects to the realtime API, starts
        the IPC server, audio stream, and processing tasks.
        """
        ipc_srv = await self.ipc_server()
        try:
            async with websockets.connect(self.api_url, extra_headers=self.headers) as websocket:
                print("Connected to OpenAI Realtime Assistant API.")
                play_audio_file(self.welcome_file)
                # Send a session update with the new model and settings.
                session_update = {
                    "type": "session.update",
                    "session": {
                        "model": DEFAULT_MODEL,
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
                # Start microphone recording.
                self.mic_stream = sd.InputStream(
                    samplerate=SAMPLERATE,
                    channels=CHANNELS,
                    dtype='int16',
                    callback=self.audio_callback,
                    blocksize=BLOCKSIZE
                )
                self.mic_stream.start()
                send_task = asyncio.create_task(self.send_audio(websocket))
                receive_task = asyncio.create_task(self.receive_messages(websocket))
                await self.shutdown_event.wait()
                print("Shutdown event received.")
        except Exception as e:
            print(f"Exception in run: {e}")
        finally:
            print("Cleaning up...")
            if self.mic_stream and self.mic_stream.active:
                self.mic_stream.abort()
                self.mic_stream.close()
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
    assets_directory = Path(os.getenv('AUDIO_ASSETS', Path(__file__).parent / "assets-audio"))
    assets_directory.mkdir(parents=True, exist_ok=True)
    welcome_file = assets_directory / "welcome.mp3"
    gotit_file = assets_directory / "gotit.mp3"
    if not welcome_file.is_file():
        print(f"Welcome audio file not found at {welcome_file}")
        sys.exit(1)
    if not gotit_file.is_file():
        print(f"Gotit audio file not found at {gotit_file}")
        sys.exit(1)
    api_key = load_api_key()
    # Check for an existing instance and shut it down if necessary.
    if os.path.exists(SOCKET_PATH):
        print("Another instance detected. Sending shutdown command.")
        try:
            asyncio.run(send_shutdown_command())
            for _ in range(20):
                if not os.path.exists(SOCKET_PATH):
                    break
                asyncio.run(asyncio.sleep(0.05))
            play_audio_file(gotit_file)
        except Exception as e:
            print(f"Error communicating with the running instance: {e}")
        sys.exit(0)
    else:
        session = AssistantSession(api_key, assets_directory, welcome_file, gotit_file)
        try:
            asyncio.run(session.run())
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
