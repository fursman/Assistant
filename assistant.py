#!/usr/bin/env python3

import asyncio
import os
import json
import websockets
import base64
import sys

from dotenv import load_dotenv
import keyring
from pathlib import Path
import getpass

import sounddevice as sd
import numpy as np
import queue  # Import queue module for thread-safe queue
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all levels of logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Log to stdout
    ]
)

# Audio configuration
samplerate = 16000  # Microphone input sample rate
assistant_samplerate = 24000  # Assistant's audio output sample rate
channels = 1
blocksize = 1600  # 0.1 seconds at 16 kHz

audio_queue = queue.Queue()  # Use thread-safe queue

# Function to load the API key using keyring
def load_api_key():
    api_key = keyring.get_password("NixOSAssistant", "APIKey")
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Please enter your OpenAI API Key: ").strip()
        if api_key:
            keyring.set_password("NixOSAssistant", "APIKey", api_key)
        else:
            logging.error("No API Key provided. Exiting.")
            sys.exit(1)
    return api_key

# Function to play audio file
def play_audio(file_path):
    try:
        from pydub import AudioSegment  # Moved import here to avoid import error if not needed
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
        logging.error(f"Error playing audio file {file_path}: {e}")

# Load environment variables
load_dotenv()

# Use the function to load the API key
api_key = load_api_key()

if not api_key:
    logging.error("Please set the OPENAI_API_KEY in your .env file or enter it when prompted.")
    sys.exit(1)

# Define the welcome_file_path
assets_directory = Path(os.getenv('AUDIO_ASSETS', Path(__file__).parent / "assets-audio"))
assets_directory.mkdir(parents=True, exist_ok=True)
welcome_file_path = assets_directory / "welcome.mp3"

# Ensure the welcome audio file exists
if not welcome_file_path.is_file():
    logging.error(f"Welcome audio file not found at {welcome_file_path}")
    sys.exit(1)

# Define the URL and headers
url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
headers = {
    "Authorization": f"Bearer {api_key}",
    "OpenAI-Beta": "realtime=v1",
}

def audio_callback(indata, frames, time_info, status):
    if status:
        logging.warning(f"Audio callback status: {status}")
    audio_queue.put(indata.copy())  # Put data into thread-safe queue

async def send_audio(websocket, loop):
    silence_threshold = 0.01  # Adjusted normalized threshold
    silence_duration = 0
    silence_duration_limit = 1.0  # seconds
    chunk_duration = blocksize / samplerate  # seconds per chunk

    try:
        while True:
            indata = await loop.run_in_executor(None, audio_queue.get)

            # Normalize the audio data
            normalized_data = indata.astype(np.float32) / 32768
            rms_value = np.sqrt(np.mean(normalized_data ** 2))

            logging.debug(f"RMS value: {rms_value}")

            if rms_value < silence_threshold:
                silence_duration += chunk_duration
            else:
                silence_duration = 0

            # Only send audio if there's sound detected
            if rms_value >= silence_threshold:
                audio_bytes = indata.tobytes()
                logging.debug(f"Sending {len(audio_bytes)} bytes of audio data.")
                base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
                audio_event = {
                    "type": "input_audio_buffer.append",
                    "audio": base64_audio,
                }
                await websocket.send(json.dumps(audio_event))
            else:
                logging.debug("Silence detected, not sending audio chunk.")

            if silence_duration >= silence_duration_limit:
                commit_event = {"type": "input_audio_buffer.commit"}
                await websocket.send(json.dumps(commit_event))
                silence_duration = 0
                logging.info("Silence duration exceeded limit. Sent input_audio_buffer.commit")
    except asyncio.CancelledError:
        logging.info("Audio sending task cancelled.")
    except Exception as e:
        logging.exception(f"Exception in send_audio: {e}")

async def receive_messages(websocket):
    try:
        # Initialize the OutputStream for continuous playback
        with sd.OutputStream(samplerate=assistant_samplerate, channels=1, dtype='int16') as output_stream:
            while True:
                message = await websocket.recv()
                event = json.loads(message)
                event_type = event.get("type")

                logging.debug(f"Received event: {event_type}")

                if event_type == "response.audio.delta":
                    delta = event["delta"]
                    audio_data = base64.b64decode(delta)
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    output_stream.write(audio_array)
                elif event_type == "response.text.delta":
                    delta = event.get("delta", "")
                    logging.info(f"Assistant (delta): {delta}")
                    print(f"{delta}", end='', flush=True)
                elif event_type == "response.done":
                    logging.info("Assistant response complete.")
                    print("\nAssistant response complete.")
                elif event_type == "error":
                    error_info = event.get("error", {})
                    error_message = error_info.get("message", "")
                    logging.error(f"Error: {error_message}")
                    logging.error(f"Full error info: {error_info}")
                else:
                    logging.debug(f"Unhandled event type: {event_type}")
    except asyncio.CancelledError:
        logging.info("Message receiving task cancelled.")
    except Exception as e:
        logging.exception(f"Error in receive_messages: {e}")

async def realtime_api():
    stream = None
    try:
        async with websockets.connect(url, extra_headers=headers) as websocket:
            logging.info("Connected to OpenAI Realtime Assistant API.")

            # Send session update
            session_update = {
                "type": "session.update",
                "session": {
                    "modalities": ["audio", "text"],
                    "voice": "alloy",
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 100,
                        "silence_duration_ms": 500,
                    },
                },
            }
            await websocket.send(json.dumps(session_update))
            logging.debug("Sent session update.")

            # Start audio stream
            stream = sd.InputStream(samplerate=samplerate, channels=channels, dtype='int16',
                                    callback=audio_callback, blocksize=blocksize)
            stream.start()
            logging.debug("Started audio input stream.")

            loop = asyncio.get_running_loop()  # Get the current event loop

            send_task = asyncio.create_task(send_audio(websocket, loop))
            receive_task = asyncio.create_task(receive_messages(websocket))

            await asyncio.gather(send_task, receive_task)
    except KeyboardInterrupt:
        logging.info("Program terminated by user.")
    except Exception as e:
        logging.exception(f"Unexpected error in realtime_api: {e}")
    finally:
        logging.info("Cleaning up...")
        if stream and not stream.stopped:
            stream.stop()
            stream.close()

def main():
    logging.info("Press Ctrl+C to exit the program.")

    # Play the welcome audio at the start
    play_audio(welcome_file_path)

    logging.info("Starting the assistant. Speak into your microphone.")

    try:
        asyncio.run(realtime_api())
    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
