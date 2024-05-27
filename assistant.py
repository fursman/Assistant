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
speech_file_path = base_log_dir / f"response_{now}.mp3"
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
                ffmpeg_pid = lock_data.get('ffmpeg_pid')
                if ffmpeg_pid:
                    os.kill(ffmpeg_pid, signal.SIGTERM)
                if script_pid:
                    os.kill(script_pid, signal.SIGTERM)
                    send_notification("NixOS Assistant:", "Silencing output and standing by for your next request!")
                    sys.exit("Exiting.")
            except (json.JSONDecodeError, ProcessLookupError, PermissionError):
                sys.exit(1)

def create_lock(ffmpeg_pid=None):
    lock_data = {
        'script_pid': os.getpid(),
        'ffmpeg_pid': ffmpeg_pid
    }
    with open(lock_file_path, 'w') as lock_file:
        json.dump(lock_data, lock_file)

def update_lock_for_ffmpeg_completion():
    if lock_file_path.exists():
        with open(lock_file_path, 'r+') as lock_file:
            lock_data = json.load(lock_file)
            lock_data['ffmpeg_pid'] = None
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

def transcribe_audio(client, audio_file_path):
    try:
        with open(audio_file_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        return response
    except Exception:
        keyring.delete_password("NixOSAssistant", "APIKey")
        send_notification("NisOS Assistant: API Key Error", "Failed to authenticate with the provided API Key. It has been deleted. Please rerun the script and enter a valid API Key.")
        sys.exit("Failed to authenticate with OpenAI. Exiting.")

def create_assistant(client):
    assistant = client.beta.assistants.create(
        name="NixOS Assistant",
        instructions="You are a helpful assistant integrated with NixOS.",
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
    def __init__(self):
        super().__init__()
        self.response_text = ""

    def on_text_created(self, text) -> None:
        print(f"\nassistant > ", end="", flush=True)

    def on_text_delta(self, delta, snapshot):
        print(delta.value, end="", flush=True)
        self.response_text += delta.value

def run_assistant(client, thread_id, assistant_id):
    event_handler = CustomEventHandler()

    with client.beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant_id,
        event_handler=event_handler,
    ) as stream:
        stream.until_done()

    return event_handler.response_text

def synthesize_speech(client, text, speech_file_path):
    response = client.audio.speech.create(
        model="tts-1-hd",
        voice="nova",
        response_format="opus",
        input=text
    )
    with open(speech_file_path, 'wb') as f:
        f.write(response.content)

def play_audio(speech_file_path):
    process = subprocess.Popen(['ffmpeg', '-i', str(speech_file_path), '-f', 'alsa', 'default'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    create_lock(ffmpeg_pid=process.pid)
    process.wait()
    update_lock_for_ffmpeg_completion()

def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    check_and_kill_existing_process()

    try:
        create_lock()

        api_key = load_api_key()
        client = OpenAI(api_key=api_key)

        if assistant_data_file.exists():
            with open(assistant_data_file, 'r') as f:
                assistant_data = json.load(f)
            assistant_id = assistant_data['assistant_id']
            thread_id = assistant_data['thread_id']
        else:
            assistant_id = create_assistant(client)
            thread_id = create_thread(client)
            with open(assistant_data_file, 'w') as f:
                json.dump({'assistant_id': assistant_id, 'thread_id': thread_id}, f)

        play_audio(welcome_file_path)
        send_notification("NixOS Assistant:", "Recording")
        record_audio(recorded_audio_path)
        play_audio(process_file_path)

        transcript = transcribe_audio(client, recorded_audio_path)
        send_notification("You asked:", transcript)
        add_message(client, thread_id, transcript)
        response_text = run_assistant(client, thread_id, assistant_id)
        send_notification("NixOS Assistant:", response_text)
        log_interaction(transcript, response_text)

        play_audio(gotit_file_path)
        synthesize_speech(client, response_text, speech_file_path)
        send_notification("NixOS Assistant:", "Audio Received")
        play_audio(speech_file_path)

    finally:
        delete_lock()

if __name__ == "__main__":
    main()
