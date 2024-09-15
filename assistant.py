#!/usr/bin/env python3

import os
import pyaudio
import wave
import subprocess
import notify2
import datetime
import signal
import numpy as np
import sys
import csv
import re
import logging
import json
import keyring
from pathlib import Path
from openai import OpenAI, AssistantEventHandler
from typing import Optional
import threading
import queue
import contextlib
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Audio recording settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
THRESHOLD = 30
SILENCE_LIMIT = 4

# Paths and directories
BASE_LOG_DIR = Path(os.getenv('LOG_DIR', "/tmp/logs/assistant/"))
BASE_LOG_DIR.mkdir(parents=True, exist_ok=True)

ASSETS_DIR = Path(os.getenv('AUDIO_ASSETS', Path(__file__).parent / "assets-audio"))
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

NOW = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_CSV_PATH = BASE_LOG_DIR / "interaction_log.csv"
RECORDED_AUDIO_PATH = BASE_LOG_DIR / f"input_{NOW}.wav"
LOCK_FILE_PATH = BASE_LOG_DIR / "script.lock"
ASSISTANT_DATA_FILE = BASE_LOG_DIR / "assistant_data.json"

AUDIO_FILES = {
    'welcome': ASSETS_DIR / "welcome.mp3",
    'processing': ASSETS_DIR / "process.mp3",
    'apikey': ASSETS_DIR / "apikey.mp3",
}

def signal_handler(sig, frame):
    """Handle termination signals to clean up resources."""
    delete_lock()
    sys.exit(0)

def load_api_key() -> str:
    """Load the OpenAI API key from keyring or prompt the user."""
    api_key = keyring.get_password("NixOSAssistant", "APIKey")
    if not api_key:
        play_audio(AUDIO_FILES['apikey'])
        api_key = prompt_for_api_key()
        if api_key:
            keyring.set_password("NixOSAssistant", "APIKey", api_key)
        else:
            send_notification("NixOS Assistant Error", "No API Key provided.")
            sys.exit(1)
    return api_key

def prompt_for_api_key() -> Optional[str]:
    """Prompt the user to enter their OpenAI API key."""
    try:
        input_cmd = 'zenity --entry --text="To begin, please enter your OpenAI API Key:" --hide-text'
        api_key = subprocess.check_output(input_cmd, shell=True, text=True).strip()
        return api_key
    except subprocess.CalledProcessError:
        return None

def handle_api_error():
    """Handle invalid API key errors."""
    keyring.delete_password("NixOSAssistant", "APIKey")
    send_notification("NixOS Assistant Error", "Invalid API Key. Please re-enter your API key.")

def check_and_kill_existing_process():
    """Ensure that no other instance of the script is running."""
    if LOCK_FILE_PATH.exists():
        with LOCK_FILE_PATH.open('r') as lock_file:
            try:
                lock_data = json.load(lock_file)
                script_pid = lock_data.get('script_pid')
                if script_pid and script_pid != os.getpid():
                    os.kill(script_pid, signal.SIGTERM)
                    send_notification("NixOS Assistant", "Previous instance terminated.")
            except Exception as e:
                logger.error(f"Error terminating existing process: {e}")
        delete_lock()

def create_lock():
    """Create a lock file with the current process ID."""
    lock_data = {'script_pid': os.getpid()}
    with LOCK_FILE_PATH.open('w') as lock_file:
        json.dump(lock_data, lock_file)

def delete_lock():
    """Delete the lock file if it exists."""
    LOCK_FILE_PATH.unlink(missing_ok=True)

def log_interaction(question: str, response: str):
    """Log the question and response to a CSV file."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with LOG_CSV_PATH.open(mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([now, "Question", question])
            writer.writerow([now, "Response", response])
    except Exception as e:
        logger.error(f"Failed to log interaction: {e}")

def send_notification(title: str, message: str):
    """Send a desktop notification."""
    notify2.init('Assistant')
    notification = notify2.Notification(title, message)
    notification.set_timeout(30000)
    notification.show()

def calculate_rms(data: bytes) -> float:
    """Calculate the RMS of audio data."""
    as_ints = np.frombuffer(data, dtype=np.int16)
    rms = np.sqrt(np.mean(np.square(as_ints)))
    return rms

def is_silence(data_chunk: bytes, threshold: int = THRESHOLD) -> bool:
    """Determine if the audio chunk is silent."""
    return calculate_rms(data_chunk) < threshold

def record_audio(file_path: Path):
    """Record audio from the microphone until silence is detected."""
    audio = pyaudio.PyAudio()
    frames = []
    silent_chunks = 0
    silence_chunk_limit = int(RATE / CHUNK * SILENCE_LIMIT)

    try:
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            if is_silence(data):
                silent_chunks += 1
                if silent_chunks > silence_chunk_limit:
                    break
            else:
                silent_chunks = 0
    except Exception as e:
        logger.error(f"Error recording audio: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

    with wave.open(str(file_path), 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

def create_assistant(client: OpenAI) -> str:
    """Create an assistant instance via the OpenAI API."""
    assistant = client.assistant.create(
        name="NixOS Assistant",
        description="An assistant integrated with NixOS.",
        model="gpt-4"
    )
    return assistant.id

def create_thread(client: OpenAI) -> str:
    """Create a conversation thread."""
    thread = client.thread.create()
    return thread.id

def add_message(client: OpenAI, thread_id: str, content: str):
    """Add a message to the conversation thread."""
    client.message.create(thread_id=thread_id, role="user", content=content)

class CustomEventHandler(AssistantEventHandler):
    """Custom event handler for processing assistant responses."""

    def __init__(self):
        super().__init__()
        self.response_text = ""
        self.text_queue = queue.Queue()

    def on_text_delta(self, delta, snapshot):
        self.response_text += delta.value
        self.text_queue.put(delta.value)

async def run_assistant(client: OpenAI, thread_id: str, assistant_id: str, is_text_input: bool = False) -> str:
    """Run the assistant and handle the response."""
    event_handler = CustomEventHandler()
    tts_task = asyncio.create_task(stream_speech(client, event_handler.text_queue))

    await client.assistant.run(
        thread_id=thread_id,
        assistant_id=assistant_id,
        event_handler=event_handler
    )

    event_handler.text_queue.put(None)
    await tts_task

    return event_handler.response_text

async def stream_speech(client: OpenAI, text_queue: queue.Queue):
    """Stream text-to-speech as text is generated."""
    buffer = ""
    process = None

    try:
        while True:
            text_chunk = text_queue.get()
            if text_chunk is None:
                break
            buffer += text_chunk

            if len(buffer.split()) >= 10:
                await synthesize_and_play(client, buffer)
                buffer = ""
        if buffer:
            await synthesize_and_play(client, buffer)
    except Exception as e:
        logger.error(f"Error in stream_speech: {e}")
    finally:
        if process:
            process.terminate()

async def synthesize_and_play(client: OpenAI, text: str):
    """Synthesize text to speech and play it."""
    try:
        audio_data = await client.audio.synthesize(text=text)
        play_audio_stream(audio_data)
    except Exception as e:
        logger.error(f"Error synthesizing speech: {e}")

def play_audio(file_path: Path):
    """Play an audio file."""
    subprocess.run(['ffplay', '-autoexit', '-nodisp', str(file_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def play_audio_stream(audio_data: bytes):
    """Play audio data using ffplay."""
    process = subprocess.Popen(['ffplay', '-autoexit', '-nodisp', '-'], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        process.stdin.write(audio_data)
        process.stdin.close()
        process.wait()
    except Exception as e:
        logger.error(f"Error playing audio stream: {e}")
    finally:
        process.terminate()

def get_context(question: str) -> str:
    """Gather additional context if the question pertains to NixOS or clipboard."""
    context = question
    if "nixos" in question.lower():
        try:
            with open('/etc/nixos/flake.nix', 'r') as file:
                nixos_config = file.read()
            context += f"\n\nCurrent flake.nix configuration:\n{nixos_config}"
        except Exception as e:
            logger.error(f"Error reading flake.nix: {e}")
    if "clipboard" in question.lower():
        try:
            clipboard_content = subprocess.check_output(['wl-paste'], text=True)
            context += f"\n\nClipboard content:\n{clipboard_content}"
        except Exception as e:
            logger.error(f"Error accessing clipboard: {e}")
    return context

def main():
    """Main function to execute the assistant script."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    check_and_kill_existing_process()
    create_lock()

    try:
        api_key = load_api_key()
        client = OpenAI(api_key=api_key)

        if ASSISTANT_DATA_FILE.exists():
            with ASSISTANT_DATA_FILE.open('r') as f:
                assistant_data = json.load(f)
            assistant_id = assistant_data['assistant_id']
            thread_id = assistant_data['thread_id']
        else:
            assistant_id = create_assistant(client)
            thread_id = create_thread(client)
            with ASSISTANT_DATA_FILE.open('w') as f:
                json.dump({'assistant_id': assistant_id, 'thread_id': thread_id}, f)

        if len(sys.argv) > 1:
            transcript = " ".join(sys.argv[1:])
            is_text_input = True
        else:
            is_text_input = False
            play_audio(AUDIO_FILES['welcome'])
            send_notification("NixOS Assistant", "Recording...")
            record_audio(RECORDED_AUDIO_PATH)
            play_audio(AUDIO_FILES['processing'])

            with RECORDED_AUDIO_PATH.open("rb") as audio_file:
                transcript = client.audio.transcribe(audio_file)
        
        context = get_context(transcript)
        add_message(client, thread_id, context)

        if not is_text_input:
            send_notification("You asked", transcript)

        response = asyncio.run(run_assistant(client, thread_id, assistant_id, is_text_input))

        if is_text_input:
            print(response)
        else:
            send_notification("NixOS Assistant", response)
        
        log_interaction(transcript, response)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        send_notification("NixOS Assistant Error", f"An error occurred: {e}")
    finally:
        delete_lock()

if __name__ == "__main__":
    main()
