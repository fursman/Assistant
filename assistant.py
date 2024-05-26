#!/usr/bin/env python3

import os
import time
import requests
import pyaudio
import wave
import subprocess
import notify2
import datetime
import signal
import numpy as np
import keyring
from pathlib import Path
import json
import csv

# Configuration for silence detection and volume meter
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
THRESHOLD = 1000
SILENCE_LIMIT = 1

# Determine the base directory for logs based on an environment variable or fallback to a directory in /tmp
base_log_dir = Path(os.getenv('LOG_DIR', "/tmp/logs/assistant/"))
print(f"Attempting to create log directory at: {base_log_dir}")
base_log_dir.mkdir(parents=True, exist_ok=True)

# Determine the base directory for assets based on an environment variable or fallback to a default path
assets_directory = Path(os.getenv('AUDIO_ASSETS', Path(__file__).parent / "assets-audio"))
assets_directory.mkdir(parents=True, exist_ok=True)

# Define file paths
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_csv_path = base_log_dir / "interaction_log.csv"
recorded_audio_path = base_log_dir / f"input_{now}.wav"
lock_file_path = base_log_dir / "script.lock"

welcome_file_path = assets_directory / "welcome.mp3"
process_file_path = assets_directory / "process.mp3"
gotit_file_path = assets_directory / "gotit.mp3"
apikey_file_path = assets_directory / "apikey.mp3"

# Set up OpenAI API key
api_key = keyring.get_password("NixOSAssistant", "APIKey")
headers = {
    "Authorization": f"Bearer {api_key}",
    "OpenAI-Beta": "assistants=v1",
}
base_url = "https://api.openai.com/v1"

def signal_handler(sig, frame):
    print(f"Received signal {sig}, cleaning up...")
    delete_lock()
    sys.exit(0)

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
                    print(f"Terminated existing ffmpeg process with PID {ffmpeg_pid}.")
                if script_pid:
                    os.kill(script_pid, signal.SIGTERM)
                    print(f"Terminated existing script process with PID {script_pid}.")
                    send_notification("NixOS Assistant", "Silencing output and standing by for your next request!")
                    sys.exit("Exiting.")
            except json.JSONDecodeError:
                print("Lock file is corrupt. Exiting.")
                sys.exit(1)
            except ProcessLookupError:
                print("No process found. Possibly already terminated.")
            except PermissionError:
                sys.exit("Permission denied to kill process. Exiting.")

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
    with open(log_csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([now, "Question", question])
        writer.writerow([now, "Response", response])

def send_notification(title, message):
    notify2.init('Assistant')
    n = notify2.Notification(str(title), str(message))
    n.set_timeout(30000)
    n.show()

def calculate_rms(data):
    rms = np.sqrt(np.mean(np.square(np.frombuffer(data, dtype=np.int16))))
    return rms

def is_silence(data_chunk, threshold=THRESHOLD):
    as_ints = np.frombuffer(data_chunk, dtype=np.int16)
    if np.max(np.abs(as_ints)) < threshold:
        print(np.max(np.abs(as_ints)))
        return True
    print(np.max(np.abs(as_ints)))
    return False

def record_audio(file_path, format=FORMAT, channels=CHANNELS, rate=RATE, chunk=CHUNK, silence_limit=SILENCE_LIMIT):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
    print("Recording...")

    frames = []
    silent_frames = 0
    silence_threshold = int(rate / chunk * silence_limit)

    while True:
        data = stream.read(chunk, exception_on_overflow=False)
        frames.append(data)
        
        if is_silence(data):
            silent_frames += 1
            if silent_frames >= silence_threshold:
                print("Silence detected, stopping recording.")
                break
        else:
            silent_frames = 0

    stream.stop_stream()
    stream.close()
    audio.terminate()
    print("Recording stopped.")

    wf = wave.open(str(file_path), 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(audio.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

def transcribe_audio(audio_file_path):
    url = f"{base_url}/audio/transcriptions"
    with open(audio_file_path, "rb") as audio_file:
        files = {"file": audio_file}
        data = {"model": "whisper-1"}
        response = requests.post(url, headers=headers, files=files, data=data)
        if response.status_code == 200:
            return response.json().get("text")
        else:
            raise Exception(f"Failed to transcribe audio: {response.text}")

def create_assistant():
    url = f"{base_url}/assistants"
    data = {
        "model": "gpt-4-1106-preview",
        "instructions": "You are a helpful assistant.",
    }
    response = requests.post(url, json=data, headers=headers)
    return response.json()

def create_thread():
    url = f"{base_url}/threads"
    response = requests.post(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to create thread: {response.text}")

def generate_response(thread_id, assistant_id, transcript):
    url = f"{base_url}/threads/{thread_id}/messages"
    data = {"role": "user", "content": transcript}
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        message = response.json()

        url = f"{base_url}/threads/{thread_id}/runs"
        data = {"assistant_id": assistant_id}
        run_response = requests.post(url, json=data, headers=headers)
        if run_response.status_code == 200:
            run_id = run_response.json()["id"]

            while True:
                run = requests.get(f"{base_url}/threads/{thread_id}/runs/{run_id}", headers=headers).json()
                status = run.get("status")
                if status == "completed":
                    messages = requests.get(url, headers=headers).json()["data"]
                    assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
                    if assistant_messages:
                        return assistant_messages[-1]["content"]
                elif status in ["failed", "cancelled"]:
                    raise Exception(f"Run failed or was cancelled: {run}")
                time.sleep(1)
        else:
            raise Exception(f"Failed to run assistant: {run_response.text}")
    else:
        raise Exception(f"Failed to add message to thread: {response.text}")

def synthesize_speech(text):
    url = f"{base_url}/audio/speech"
    data = {"model": "tts-1", "input": text, "voice": "alloy", "response_format": "mp3"}
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to synthesize speech: {response.text}")

def play_audio(audio_content):
    with open("/tmp/temp_audio.mp3", "wb") as temp_audio_file:
        temp_audio_file.write(audio_content)
    subprocess.run(["ffmpeg", "-i", "/tmp/temp_audio.mp3", "-f", "alsa", "default"])

def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    check_and_kill_existing_process()
    create_lock()

    try:
        play_audio(open(welcome_file_path, "rb").read())
        record_audio(recorded_audio_path)
        update_lock_for_ffmpeg_completion()

        transcript = transcribe_audio(recorded_audio_path)
        print(f"Transcript: {transcript}")

        assistant = create_assistant()
        thread = create_thread()

        response_text = generate_response(thread["id"], assistant["id"], transcript)
        if not response_text:
            response_text = "I'm sorry, I couldn't generate a response."
        send_notification("NixOS Assistant", response_text)
        print(f"Response: {response_text}")
        log_interaction(transcript, response_text)

        play_audio(open(gotit_file_path, "rb").read())

        audio_response = synthesize_speech(response_text)
        send_notification("NixOS Assistant", "Audio Received")
        play_audio(audio_response)

    finally:
        delete_lock()

if __name__ == "__main__":
    main()
