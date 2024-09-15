#!/usr/bin/env python3

import os
import sys
import json
import csv
import re
import signal
import logging
import threading
import queue
import datetime
import subprocess
from pathlib import Path
from typing_extensions import override

import pyaudio
import wave
import numpy as np
import keyring
import notify2

# Import the OpenAI SDK and handle exceptions
import openai
from openai import OpenAI, AssistantEventHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Audio configurations
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
THRESHOLD = 30  # Initial silence threshold
SILENCE_LIMIT = 4

# Paths and directories
BASE_LOG_DIR = Path(os.getenv('LOG_DIR', "/tmp/logs/assistant/"))
BASE_LOG_DIR.mkdir(parents=True, exist_ok=True)

ASSETS_DIR = Path(os.getenv('AUDIO_ASSETS', Path(__file__).parent / "assets-audio"))
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_CSV_PATH = BASE_LOG_DIR / "interaction_log.csv"
RECORDED_AUDIO_PATH = BASE_LOG_DIR / f"input_{now}.wav"
LOCK_FILE_PATH = BASE_LOG_DIR / "script.lock"
ASSISTANT_DATA_FILE = BASE_LOG_DIR / "assistant_data.json"

WELCOME_FILE_PATH = ASSETS_DIR / "welcome.mp3"
PROCESS_FILE_PATH = ASSETS_DIR / "process.mp3"
APIKEY_FILE_PATH = ASSETS_DIR / "apikey.mp3"

# Global variable to store ffmpeg process ID
ffmpeg_process = None

# Signal handling for graceful exit
def signal_handler(sig, frame):
    delete_lock()
    if ffmpeg_process and ffmpeg_process.poll() is None:
        ffmpeg_process.terminate()
    sys.exit(0)

# Load or prompt for OpenAI API key
def load_api_key():
    api_key = keyring.get_password("NixOSAssistant", "APIKey")
    if not api_key:
        play_audio(APIKEY_FILE_PATH)
        api_key = prompt_for_api_key()
        if api_key:
            keyring.set_password("NixOSAssistant", "APIKey", api_key)
        else:
            send_notification("NixOS Assistant Error", "No API Key provided.")
            sys.exit(1)
    return api_key

def prompt_for_api_key():
    input_cmd = 'zenity --entry --text="To begin, please enter your OpenAI API Key:" --hide-text'
    try:
        api_key = subprocess.check_output(input_cmd, shell=True, text=True).strip()
        return api_key
    except subprocess.CalledProcessError:
        return None

# Handle invalid API key
def handle_api_error():
    keyring.delete_password("NixOSAssistant", "APIKey")
    send_notification("NixOS Assistant Error", "Invalid API Key. Please re-enter your API key.")

# Ensure only one instance is running
def check_and_kill_existing_process():
    if LOCK_FILE_PATH.exists():
        with open(LOCK_FILE_PATH, 'r') as lock_file:
            try:
                lock_data = json.load(lock_file)
                script_pid = lock_data.get('script_pid')
                ffmpeg_pid = lock_data.get('ffmpeg_pid')
                if ffmpeg_pid:
                    try:
                        os.kill(ffmpeg_pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                if script_pid and script_pid != os.getpid():
                    os.kill(script_pid, signal.SIGTERM)
                    send_notification("NixOS Assistant:", "Previous instance terminated.")
            except (json.JSONDecodeError, ProcessLookupError, PermissionError) as e:
                logger.error(f"Error handling lock file: {e}")
                LOCK_FILE_PATH.unlink()

# Create and delete lock files
def create_lock(ffmpeg_pid=None):
    lock_data = {
        'script_pid': os.getpid(),
        'ffmpeg_pid': ffmpeg_pid
    }
    with open(LOCK_FILE_PATH, 'w') as lock_file:
        json.dump(lock_data, lock_file)

def delete_lock():
    if LOCK_FILE_PATH.exists():
        LOCK_FILE_PATH.unlink()

# Log user interactions
def log_interaction(question, response):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_CSV_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([now, "Question", question])
        writer.writerow([now, "Response", response])

# Send desktop notifications
def send_notification(title, message):
    notify2.init('NixOS Assistant')
    n = notify2.Notification(title, message)
    n.set_timeout(30000)
    n.show()

# Calculate RMS for silence detection
def calculate_rms(data):
    as_ints = np.frombuffer(data, dtype=np.int16)
    rms = np.sqrt(np.mean(np.square(as_ints)))
    return rms

def is_silence(data_chunk, threshold):
    rms = calculate_rms(data_chunk)
    return rms < threshold

# Record ambient noise to adjust threshold
def adjust_threshold():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    try:
        data = stream.read(CHUNK, exception_on_overflow=False)
        rms = calculate_rms(data)
        adjusted_threshold = rms + 10  # Add a buffer to the ambient noise level
        logger.info(f"Adjusted silence threshold: {adjusted_threshold}")
        return adjusted_threshold
    except Exception as e:
        logger.error(f"Error adjusting threshold: {e}")
        return THRESHOLD  # Use default if adjustment fails
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

# Record audio until silence is detected
def record_audio(file_path):
    threshold = adjust_threshold()
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    silent_frames = 0
    silence_threshold = int(RATE / CHUNK * SILENCE_LIMIT)

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            if is_silence(data, threshold):
                silent_frames += 1
                if silent_frames >= silence_threshold:
                    break
            else:
                silent_frames = 0
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

# Create assistant using OpenAI Assistants API
def create_assistant(client):
    assistant = client.beta.assistants.create(
        name="NixOS Assistant",
        instructions=(
            "You are a helpful assistant integrated with NixOS. Provide concise and accurate information. "
            "If asked about system configurations or clipboard content, refer to the additional context provided."
        ),
        tools=[],
        model="gpt-4"
    )
    return assistant.id

# Create a new conversation thread
def create_thread(client):
    thread = client.beta.threads.create()
    return thread.id

# Add user message to the thread
def add_message(client, thread_id, content):
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content
    )
    return message

# Custom event handler for assistant responses
class CustomEventHandler(AssistantEventHandler):
    def __init__(self, is_text_input=False):
        super().__init__()
        self.response_text = ""
        self.text_queue = queue.Queue()
        self.is_text_input = is_text_input

    @override
    def on_text_created(self, text) -> None:
        if self.is_text_input:
            print("\nAssistant:", end=' ', flush=True)

    @override
    def on_text_delta(self, delta, snapshot):
        self.response_text += delta.value
        if self.is_text_input:
            print(delta.value, end='', flush=True)
        else:
            self.text_queue.put(delta.value)

    @override
    def on_run_end(self):
        if not self.is_text_input:
            self.text_queue.put(None)  # Signal TTS thread to complete

# Run the assistant and handle responses
def run_assistant(client, thread_id, assistant_id, is_text_input=False):
    event_handler = CustomEventHandler(is_text_input)
    tts_thread = None
    if not is_text_input:
        tts_thread = threading.Thread(target=stream_speech, args=(client, event_handler.text_queue))
        tts_thread.start()

    try:
        with client.beta.threads.runs.stream(
            thread_id=thread_id,
            assistant_id=assistant_id,
            event_handler=event_handler,
        ) as stream:
            stream.until_done()
    except openai.error.AuthenticationError:
        handle_api_error()
        sys.exit(1)
    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI API Error: {e}")
        send_notification("NixOS Assistant Error", f"OpenAI API Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        send_notification("NixOS Assistant Error", f"An unexpected error occurred: {e}")
        sys.exit(1)

    if tts_thread:
        tts_thread.join()

    return event_handler.response_text

# Stream speech output using OpenAI's TTS API
def stream_speech(client, text_queue):
    global ffmpeg_process
    full_text = ""
    buffer = ""
    process = None
    sentence_end_pattern = re.compile(r'(?<=[.!?])\s+')

    def send_chunk_to_tts(chunk):
        nonlocal process
        try:
            response = client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=chunk
            )
            if not process:
                process = subprocess.Popen(
                    ['ffplay', '-autoexit', '-nodisp', '-'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                ffmpeg_process = process  # Store global ffmpeg_process
                create_lock(ffmpeg_pid=process.pid)
            for audio_chunk in response.iter_bytes(chunk_size=4096):
                if audio_chunk:
                    process.stdin.write(audio_chunk)
                    process.stdin.flush()
        except Exception as e:
            logger.error(f"Error in TTS or audio playback: {str(e)}")

    try:
        while True:
            try:
                text_chunk = text_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if text_chunk is None:
                break

            full_text += text_chunk
            buffer += text_chunk

            # Split the buffer into sentences
            sentences = sentence_end_pattern.split(buffer)
            # Process complete sentences
            while len(sentences) > 1:
                chunk_to_send = sentences.pop(0)
                send_chunk_to_tts(chunk_to_send)
            buffer = sentences[0] if sentences else ''
        # Send any remaining text in the buffer
        if buffer:
            send_chunk_to_tts(buffer)
    except Exception as e:
        logger.error(f"Unexpected error in stream_speech: {str(e)}")
    finally:
        if process:
            process.stdin.close()
            process.wait()
            ffmpeg_process = None
            create_lock()  # Update lock file without ffmpeg_pid

# Augment user question with additional context
def get_context(question):
    context = question
    if "nixos" in question.lower():
        try:
            with open('/etc/nixos/flake.nix', 'r') as file:
                nixos_config = file.read()
            context += f"\n\nFor additional context, this is the system's current flake.nix configuration:\n{nixos_config}"
        except FileNotFoundError:
            context += "\n\nUnable to find the flake.nix configuration file."
        except Exception as e:
            context += f"\n\nAn error occurred while reading the flake.nix configuration: {str(e)}"
    if "clipboard" in question.lower():
        try:
            clipboard_content = subprocess.check_output(['wl-paste'], text=True)
            context += f"\n\nFor additional context, this is the current clipboard content:\n{clipboard_content}"
        except subprocess.CalledProcessError:
            context += "\n\nClipboard is empty or unavailable."
        except Exception as e:
            context += f"\n\nAn unexpected error occurred while retrieving clipboard content: {e}"
    return context

# Play audio files
def play_audio(file_path):
    subprocess.run(['ffplay', '-autoexit', '-nodisp', str(file_path)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Main function to orchestrate the assistant interaction
def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    check_and_kill_existing_process()
    create_lock()

    try:
        api_key = load_api_key()
        client = OpenAI(api_key=api_key)

        if ASSISTANT_DATA_FILE.exists():
            with open(ASSISTANT_DATA_FILE, 'r') as f:
                assistant_data = json.load(f)
            assistant_id = assistant_data['assistant_id']
            thread_id = assistant_data['thread_id']
        else:
            assistant_id = create_assistant(client)
            thread_id = create_thread(client)
            with open(ASSISTANT_DATA_FILE, 'w') as f:
                json.dump({'assistant_id': assistant_id, 'thread_id': thread_id}, f)

        if len(sys.argv) > 1:
            # Command-line input
            transcript = " ".join(sys.argv[1:])
            is_text_input = True
        else:
            # Audio input
            is_text_input = False
            play_audio(WELCOME_FILE_PATH)
            send_notification("NixOS Assistant:", "Recording")
            record_audio(RECORDED_AUDIO_PATH)
            play_audio(PROCESS_FILE_PATH)

            with open(RECORDED_AUDIO_PATH, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )

        context = get_context(transcript)
        if not is_text_input:
            send_notification("You asked:", transcript)
        add_message(client, thread_id, context)

        response = run_assistant(client, thread_id, assistant_id, is_text_input=is_text_input)
        if is_text_input:
            print()
        else:
            send_notification("NixOS Assistant:", response)

        log_interaction(transcript, response)

    except openai.error.AuthenticationError:
        handle_api_error()
        sys.exit(1)
    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI API Error: {e}")
        send_notification("NixOS Assistant Error", f"OpenAI API Error: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        send_notification("NixOS Assistant Error", f"An error occurred: {e}")
    finally:
        delete_lock()
        if ffmpeg_process and ffmpeg_process.poll() is None:
            ffmpeg_process.terminate()

if __name__ == "__main__":
    main()
