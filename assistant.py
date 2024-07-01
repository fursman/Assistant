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
import json
import keyring
from pathlib import Path
from openai import OpenAI, AssistantEventHandler
from typing_extensions import override
import threading
import queue

# Configuration for silence detection and volume meter
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
THRESHOLD = 20
SILENCE_LIMIT = 1
PREV_AUDIO_DURATION = 0.5

# Determine the base directory for logs based on an environment variable or fallback to a directory in /tmp
base_log_dir = Path(os.getenv('LOG_DIR', "/tmp/logs/assistant/"))
base_log_dir.mkdir(parents=True, exist_ok=True)

# Determine the base directory for assets based on an environment variable or fallback to a default path
assets_directory = Path(os.getenv('AUDIO_ASSETS', Path(__file__).parent / "assets-audio"))
assets_directory.mkdir(parents=True, exist_ok=True)

# Define file paths
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_csv_path = base_log_dir / "interaction_log.csv"
recorded_audio_path = base_log_dir / f"input_{now}.wav"
lock_file_path = base_log_dir / "script.lock"
assistant_data_file = base_log_dir / "assistant_data.json"

welcome_file_path = assets_directory / "welcome.mp3"
process_file_path = assets_directory / "process.mp3"
gotit_file_path = assets_directory / "gotit.mp3"
apikey_file_path = assets_directory / "apikey.mp3"

def signal_handler(sig, frame):
    delete_lock()
    sys.exit(0)

def load_api_key():
    api_key = keyring.get_password("NixOSAssistant", "APIKey")
    if not api_key:
        play_audio(apikey_file_path)
        input_cmd = 'zenity --entry --text="To begin, please enter your OpenAI API Key:" --hide-text'
        api_key = subprocess.check_output(input_cmd, shell=True, text=True).strip()
        if api_key:
            keyring.set_password("NixOSAssistant", "APIKey", api_key)
        else:
            send_notification("NixOS Assistant Error", "No API Key provided.")
            sys.exit(1)
    return api_key

def handle_api_error():
    keyring.delete_password("NixOSAssistant", "APIKey")
    send_notification("NixOS Assistant Error", "Invalid API Key. Please re-enter your API key.")

def check_and_kill_existing_process():
    if lock_file_path.exists():
        with open(lock_file_path, 'r') as lock_file:
            try:
                lock_data = json.load(lock_file)
                script_pid = lock_data.get('script_pid')
                ffplay_pid = lock_data.get('ffplay_pid')
                if ffplay_pid:
                    os.kill(ffplay_pid, signal.SIGTERM)
                if script_pid:
                    os.kill(script_pid, signal.SIGTERM)
                    send_notification("NixOS Assistant:", "Silencing output and standing by for your next request!")
                    sys.exit("Exiting.")
            except (json.JSONDecodeError, ProcessLookupError, PermissionError):
                sys.exit(1)

def create_lock(ffplay_pid=None):
    lock_data = {
        'script_pid': os.getpid(),
        'ffplay_pid': ffplay_pid
    }
    with open(lock_file_path, 'w') as lock_file:
        json.dump(lock_data, lock_file)

def update_lock_for_ffplay_completion():
    if lock_file_path.exists():
        with open(lock_file_path, 'r+') as lock_file:
            lock_data = json.load(lock_file)
            lock_data['ffplay_pid'] = None
            lock_file.seek(0)
            json.dump(lock_data, lock_file)
            lock_file.truncate()

def delete_lock():
    try:
        lock_file_path.unlink()
    except OSError:
        pass

def log_interaction(question, response):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([now, "Question", question])
        writer.writerow([now, "Response", response])

def send_notification(title, message):
    notify2.init('Assistant')
    n = notify2.Notification(title, message)
    n.set_timeout(30000)
    n.show()

def calculate_rms(data):
    as_ints = np.frombuffer(data, dtype=np.int16)
    rms = np.sqrt(np.mean(np.square(as_ints)))
    return rms

def is_silence(data_chunk, threshold=THRESHOLD):
    rms = calculate_rms(data_chunk)
    return rms < threshold

def record_audio(file_path, format=FORMAT, channels=CHANNELS, rate=RATE, chunk=CHUNK, silence_limit=SILENCE_LIMIT):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)

    frames = []
    silent_frames = 0
    silence_threshold = int(rate / chunk * silence_limit)

    while True:
        data = stream.read(chunk, exception_on_overflow=False)
        frames.append(data)
        if is_silence(data):
            silent_frames += 1
            if silent_frames >= silence_threshold:
                break
        else:
            silent_frames = 0

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(str(file_path), 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(audio.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

def create_assistant(client):
    assistant = client.beta.assistants.create(
        name="NixOS Assistant",
        instructions="You are a helpful assistant integrated with NixOS. Provide concise and accurate information. If asked about system configurations or clipboard content, refer to the additional context provided.",
        tools=[],
        model="gpt-4o"
    )
    return assistant.id

def create_thread(client):
    thread = client.beta.threads.create()
    return thread.id

def add_message(client, thread_id, content):
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content
    )
    return message

class CustomEventHandler(AssistantEventHandler):
    def __init__(self, tts_queue):
        super().__init__()
        self.response_text = ""
        self.tts_queue = tts_queue

    @override
    def on_text_created(self, text) -> None:
        print(f"\nassistant > ", end="", flush=True)

    @override
    def on_text_delta(self, delta, snapshot):
        print(delta.value, end="", flush=True)
        self.response_text += delta.value
        self.tts_queue.put(delta.value)

def run_assistant(client, thread_id, assistant_id, tts_queue):
    event_handler = CustomEventHandler(tts_queue)

    with client.beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant_id,
        event_handler=event_handler,
    ) as stream:
        stream.until_done()

    tts_queue.put(None)  # Signal end of text
    return event_handler.response_text
    
def stream_speech(client, text_queue):
    full_text = ""
    ffplay_process = None
    end_of_sentence_punctuation = ('.', '!', '?')

    try:
        while True:
            text_chunk = text_queue.get()
            if text_chunk is None:
                break
            full_text += text_chunk

            # Check if we have a complete sentence or a significant amount of text
            if (any(full_text.endswith(punct) for punct in end_of_sentence_punctuation) and len(full_text) > 50) or len(full_text) > 200:
                response = client.audio.speech.create(
                    model="tts-1-hd",
                    voice="nova",
                    input=full_text
                )

                if ffplay_process is None or ffplay_process.poll() is not None:
                    ffplay_process = subprocess.Popen(['ffplay', '-nodisp', '-autoexit', '-'], stdin=subprocess.PIPE)
                    create_lock(ffplay_process.pid)

                for chunk in response.iter_bytes(chunk_size=4096):
                    if chunk:
                        ffplay_process.stdin.write(chunk)
                        ffplay_process.stdin.flush()

                full_text = ""  # Reset the text buffer

        # Process any remaining text
        if full_text:
            response = client.audio.speech.create(
                model="tts-1-hd",
                voice="nova",
                input=full_text
            )

            if ffplay_process is None or ffplay_process.poll() is not None:
                ffplay_process = subprocess.Popen(['ffplay', '-nodisp', '-autoexit', '-'], stdin=subprocess.PIPE)
                create_lock(ffplay_process.pid)

            for chunk in response.iter_bytes(chunk_size=4096):
                if chunk:
                    ffplay_process.stdin.write(chunk)
                    ffplay_process.stdin.flush()

    finally:
        if ffplay_process and ffplay_process.poll() is None:
            ffplay_process.stdin.close()
            ffplay_process.wait()
        update_lock_for_ffplay_completion()

# ... [rest of the code remains unchanged]

if __name__ == "__main__":
    main()
