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

# Audio configuration
samplerate = 16000  # Microphone input sample rate
assistant_samplerate = 24000  # Assistant's audio output sample rate
channels = 1
blocksize = 2400  # Block size for audio recording

audio_queue = queue.Queue()  # Thread-safe queue for audio data

# Unix domain socket path for IPC
socket_path = '/tmp/assistant.sock'

# Function to load the API key using keyring
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

# Function to play audio file
def play_audio(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        samples = np.array(audio.get_array_of_samples())
        # If stereo, reshape
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
        else:
            samples = samples.reshape((-1, 1))
        sd.play(samples, samplerate=audio.frame_rate)
        sd.wait()
    except Exception as e:
        print(f"Error playing audio file {file_path}: {e}")

# Use the function to load the API key
api_key = load_api_key()

# Define the assets directory and audio file paths
assets_directory = Path(os.getenv('AUDIO_ASSETS', Path(__file__).parent / "assets-audio"))
assets_directory.mkdir(parents=True, exist_ok=True)
welcome_file_path = assets_directory / "welcome.mp3"
gotit_file_path = assets_directory / "gotit.mp3"  # Define gotit.mp3 path

# Ensure the welcome audio file exists
if not welcome_file_path.is_file():
    print(f"Welcome audio file not found at {welcome_file_path}")
    sys.exit(1)

# Ensure the gotit audio file exists
if not gotit_file_path.is_file():
    print(f"Gotit audio file not found at {gotit_file_path}")
    sys.exit(1)

# Define the URL and headers
url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
headers = {
    "Authorization": f"Bearer {api_key}",
    "OpenAI-Beta": "realtime=v1",
}

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata.copy())  # Put data into thread-safe queue

async def send_audio(websocket, shutdown_event):
    silence_threshold = 0.01  # Normalized threshold
    silence_duration = 0
    silence_duration_limit = 1.0  # seconds
    chunk_duration = blocksize / samplerate  # seconds per chunk

    try:
        while not shutdown_event.is_set():
            indata = await asyncio.get_event_loop().run_in_executor(None, audio_queue.get)
            rms_value = np.sqrt(np.mean((indata.astype(np.float32) / 32768.0) ** 2))
            print(f"RMS value: {rms_value}")

            if rms_value < silence_threshold:
                silence_duration += chunk_duration
            else:
                silence_duration = 0

            audio_bytes = indata.tobytes()
            print(f"Sending {len(audio_bytes)} bytes of audio data.")
            base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
            audio_event = {
                "type": "input_audio_buffer.append",
                "audio": base64_audio,
            }
            await websocket.send(json.dumps(audio_event))

            if silence_duration >= silence_duration_limit:
                commit_event = {"type": "input_audio_buffer.commit"}
                await websocket.send(json.dumps(commit_event))
                silence_duration = 0
                print("Silence detected. Sent input_audio_buffer.commit")
    except asyncio.CancelledError:
        print("Audio sending task cancelled.")
    except Exception as e:
        print(f"Error in send_audio: {e}")

async def receive_messages(websocket, shutdown_event):
    assistant_response_started = False
    try:
        # Initialize the OutputStream for continuous playback
        with sd.OutputStream(samplerate=assistant_samplerate, channels=1, dtype='int16') as output_stream:
            while not shutdown_event.is_set():
                message = await websocket.recv()
                event = json.loads(message)

                if event["type"] == "response.audio.delta":
                    delta = event["delta"]
                    audio_data = base64.b64decode(delta)
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    output_stream.write(audio_array)
                elif event["type"] == "response.text.delta":
                    delta = event.get("delta", "")
                    if not assistant_response_started:
                        print("\nAssistant:", end=' ', flush=True)
                        assistant_response_started = True
                    print(delta, end='', flush=True)
                elif event["type"] == "response.done":
                    assistant_response_started = False
                    print("\nAssistant response complete.")
                elif event["type"] == "error":
                    error_info = event.get("error", {})
                    error_message = error_info.get("message", "")
                    print(f"Error: {error_message}")
                    print(f"Full error info: {error_info}")
                else:
                    pass  # Handle other event types if necessary
    except asyncio.CancelledError:
        print("Message receiving task cancelled.")
    except Exception as e:
        print(f"Error in receive_messages: {e}")

async def ipc_server(shutdown_event):
    # Remove existing socket file if it exists
    if os.path.exists(socket_path):
        os.remove(socket_path)

    async def handle_client(reader, writer):
        try:
            data = await reader.read(100)
            message = data.decode()
            print(f"Received IPC message: {message}")
            if message.strip() == "shutdown":
                shutdown_event.set()
                writer.write(b"ack")
                await writer.drain()
        except Exception as e:
            print(f"Error in IPC server: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_unix_server(handle_client, path=socket_path)
    return server

async def realtime_api():
    stream = None
    send_task = None
    receive_task = None
    ipc_server_task = None
    shutdown_event = asyncio.Event()

    try:
        # Start IPC server
        ipc_server_task = await ipc_server(shutdown_event)

        async with websockets.connect(url, extra_headers=headers) as websocket:
            print("Connected to OpenAI Realtime Assistant API.")

            # Play the welcome audio at the start
            play_audio(welcome_file_path)
            
            # Send session update
            session_update = {
                "type": "session.update",
                "session": {
                    "modalities": ["audio", "text"],
                    "voice": "shimmer",
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 100,
                        "silence_duration_ms": 1000,
                    },
                },
            }
            await websocket.send(json.dumps(session_update))

            # Start audio stream
            stream = sd.InputStream(samplerate=samplerate, channels=channels, dtype='int16',
                                    callback=audio_callback, blocksize=blocksize)
            stream.start()

            send_task = asyncio.create_task(send_audio(websocket, shutdown_event))
            receive_task = asyncio.create_task(receive_messages(websocket, shutdown_event))

            await shutdown_event.wait()
            print("Shutdown event received.")

    finally:
        print("Cleaning up...")
        if send_task and not send_task.done():
            send_task.cancel()
        if receive_task and not receive_task.done():
            receive_task.cancel()
        if stream and not stream.stopped:
            stream.stop()
            stream.close()
        if ipc_server_task:
            ipc_server_task.close()
            await ipc_server_task.wait_closed()
        if os.path.exists(socket_path):
            os.remove(socket_path)

async def send_shutdown_command():
    try:
        reader, writer = await asyncio.open_unix_connection(socket_path)
        writer.write(b"shutdown")
        await writer.drain()
        data = await reader.read(100)
        if data.decode() == "ack":
            print("Shutdown command acknowledged.")
        writer.close()
        await writer.wait_closed()
    except Exception as e:
        print(f"Error sending shutdown command: {e}")

def main():
    print("Starting the assistant.")

    if os.path.exists(socket_path):
        # Another instance is running, send shutdown command
        print("Another instance detected. Sending shutdown command.")
        try:
            asyncio.run(send_shutdown_command())
            # Wait for the running process to shut down
            for _ in range(10):
                if not os.path.exists(socket_path):
                    break
                asyncio.run(asyncio.sleep(0.1))
            # Play "got it" audio
            play_audio(gotit_file_path)
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
