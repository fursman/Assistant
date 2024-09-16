#!/usr/bin/env python3

import os
import sys
import signal
import threading
import queue
import logging
import json
import csv
import datetime
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import openai
import keyring
import notify2

# Configuration for silence detection and recording
CHUNK_DURATION = 0.1  # Duration of each audio chunk in seconds
FORMAT = 'int16'
CHANNELS = 1
RATE = 22050
THRESHOLD = 30
SILENCE_LIMIT = 4  # Seconds of silence before stopping recording

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

# Audio asset files
WELCOME_AUDIO_PATH = ASSETS_DIR / "welcome.mp3"
PROCESS_AUDIO_PATH = ASSETS_DIR / "process.mp3"
APIKEY_AUDIO_PATH = ASSETS_DIR / "apikey.mp3"

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def signal_handler(sig, frame):
    delete_lock()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def load_api_key() -> str:
    api_key = keyring.get_password("NixOSAssistant", "APIKey")
    if not api_key:
        play_audio(APIKEY_AUDIO_PATH)
        api_key = subprocess.check_output(
            ['zenity', '--entry', '--text=Please enter your OpenAI API Key:', '--hide-text'],
            text=True
        ).strip()
        if api_key:
            keyring.set_password("NixOSAssistant", "APIKey", api_key)
        else:
            send_notification("NixOS Assistant Error", "No API Key provided.")
            sys.exit(1)
    return api_key

def check_and_kill_existing_process():
    if LOCK_FILE_PATH.exists():
        try:
            with LOCK_FILE_PATH.open('r') as lock_file:
                lock_data = json.load(lock_file)
            script_pid = lock_data.get('script_pid')
            if script_pid and script_pid != os.getpid():
                os.kill(script_pid, signal.SIGTERM)
                send_notification("NixOS Assistant", "Previous instance terminated.")
        except Exception as e:
            logger.error(f"Error handling lock file: {e}")
        finally:
            LOCK_FILE_PATH.unlink(missing_ok=True)

def create_lock():
    lock_data = {'script_pid': os.getpid()}
    with LOCK_FILE_PATH.open('w') as lock_file:
        json.dump(lock_data, lock_file)

def delete_lock():
    LOCK_FILE_PATH.unlink(missing_ok=True)

def log_interaction(question: str, response: str):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_CSV_PATH.open('a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([now, "Question", question])
        writer.writerow([now, "Response", response])

def send_notification(title: str, message: str):
    notify2.init('NixOS Assistant')
    notification = notify2.Notification(title, message)
    notification.set_timeout(30000)
    notification.show()

def calculate_rms(data: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square(data.astype(np.float32))))

def is_silence(data_chunk: np.ndarray) -> bool:
    rms = calculate_rms(data_chunk)
    return rms < THRESHOLD

def record_audio(file_path: Path):
    q = queue.Queue()
    silence_counter = 0
    max_silence_frames = int(SILENCE_LIMIT / CHUNK_DURATION)

    def callback(indata, frames, time, status):
        if status:
            logger.warning(f"Recording status: {status}")
        q.put(indata.copy())

    with sf.SoundFile(file_path, mode='x', samplerate=RATE, channels=CHANNELS, subtype='PCM_16') as file:
        with sd.InputStream(samplerate=RATE, channels=CHANNELS, callback=callback):
            while True:
                data = q.get()
                file.write(data)
                if is_silence(data):
                    silence_counter += 1
                    if silence_counter > max_silence_frames:
                        break
                else:
                    silence_counter = 0

def play_audio(file_path: Path):
    try:
        data, fs = sf.read(file_path)
        sd.play(data, fs)
        sd.wait()
    except Exception as e:
        logger.error(f"Error playing audio: {e}")

def get_context(question: str) -> str:
    context = question
    if "nixos" in question.lower():
        try:
            with open('/etc/nixos/flake.nix', 'r') as file:
                nixos_config = file.read()
            context += f"\n\nCurrent flake.nix configuration:\n{nixos_config}"
        except Exception as e:
            context += f"\n\nError reading flake.nix: {str(e)}"
    if "clipboard" in question.lower():
        try:
            clipboard_content = subprocess.check_output(['wl-paste'], text=True)
            context += f"\n\nClipboard content:\n{clipboard_content}"
        except Exception as e:
            context += f"\n\nError retrieving clipboard content: {str(e)}"
    return context

def main():
    check_and_kill_existing_process()
    create_lock()

    try:
        api_key = load_api_key()
        openai.api_key = api_key

        # Load or create assistant and thread
        if ASSISTANT_DATA_FILE.exists():
            with ASSISTANT_DATA_FILE.open('r') as f:
                assistant_data = json.load(f)
            assistant_id = assistant_data['assistant_id']
            thread_id = assistant_data['thread_id']
        else:
            assistant = openai.Assistant.create(
                name="NixOS Assistant",
                instructions="You are a helpful assistant integrated with NixOS. Provide concise and accurate information. If asked about system configurations or clipboard content, refer to the additional context provided.",
                model="gpt-4"
            )
            assistant_id = assistant['id']
            thread = openai.Thread.create()
            thread_id = thread['id']
            with ASSISTANT_DATA_FILE.open('w') as f:
                json.dump({'assistant_id': assistant_id, 'thread_id': thread_id}, f)

        if len(sys.argv) > 1:
            # Command-line input
            transcript = " ".join(sys.argv[1:])
            is_text_input = True
        else:
            # Audio input
            is_text_input = False
            play_audio(WELCOME_AUDIO_PATH)
            send_notification("NixOS Assistant", "Recording audio...")
            record_audio(RECORDED_AUDIO_PATH)
            play_audio(PROCESS_AUDIO_PATH)

            with open(RECORDED_AUDIO_PATH, "rb") as audio_file:
                transcript_data = openai.Audio.transcribe("whisper-1", audio_file)
                transcript = transcript_data['text']

        context = get_context(transcript)
        if not is_text_input:
            send_notification("You asked", transcript)

        # Add message to thread
        openai.ThreadMessage.create(
            thread_id=thread_id,
            role="user",
            content=context
        )

        # Run assistant and get response
        response = openai.Assistant.run(
            assistant_id=assistant_id,
            thread_id=thread_id
        )

        assistant_response = response['content']

        if is_text_input:
            print(assistant_response)
        else:
            send_notification("NixOS Assistant", assistant_response)

            # Text-to-speech playback
            tts_response = openai.Audio.synthesize(
                text=assistant_response,
                voice="nova"
            )
            with sf.SoundFile(tts_response['audio'], 'rb') as f:
                data, fs = sf.read(f)
                sd.play(data, fs)
                sd.wait()

        log_interaction(transcript, assistant_response)

    except Exception as e:
        send_notification("NixOS Assistant Error", f"An error occurred: {str(e)}")
    finally:
        delete_lock()

if __name__ == "__main__":
    main()
