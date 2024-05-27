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
import select
import keyring
from collections import deque
from pathlib import Path
from openai import OpenAI

# Configuration for silence detection and volume meter
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
THRESHOLD = 1000
SILENCE_LIMIT = 1
PREV_AUDIO_DURATION = 0.5

base_log_dir = Path(os.getenv('LOG_DIR', "/tmp/logs/assistant/"))
base_log_dir.mkdir(parents=True, exist_ok=True)

assets_directory = Path(os.getenv('AUDIO_ASSETS', Path(__file__).parent / "assets-audio"))
assets_directory.mkdir(parents=True, exist_ok=True)

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_csv_path = base_log_dir / "interaction_log.csv"
recorded_audio_path = base_log_dir / f"input_{now}.wav"
speech_file_path = base_log_dir / f"response_{now}.mp3"
lock_file_path = base_log_dir / "script.lock"

welcome_file_path = assets_directory / "welcome.mp3"
process_file_path = assets_directory / "process.mp3"
gotit_file_path = assets_directory / "gotit.mp3"
apikey_file_path = assets_directory / "apikey.mp3"

def signal_handler(sig, frame):
    print(f"Received signal {sig}, cleaning up...")
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
                    print(f"Terminated existing ffmpeg process with PID {ffmpeg_pid}.")
                if script_pid:
                    os.kill(script_pid, signal.SIGTERM)
                    print(f"Terminated existing script process with PID {script_pid}.")
                    send_notification("NixOS Assistant:", "Silencing output and standing by for your next request!")
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
    n = notify2.Notification(title, message)
    n.set_timeout(30000)
    n.show()

def is_silence(data_chunk, threshold=THRESHOLD):
    as_ints = np.frombuffer(data_chunk, dtype=np.int16)
    if np.max(np.abs(as_ints)) < threshold:
        return True
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

def transcribe_audio(client, audio_file_path):
    try:
        with open(audio_file_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        return response
    except Exception as e:
        keyring.delete_password("NixOSAssistant", "APIKey")
        notify2.init('Assistant Error')
        n = notify2.Notification("NixOS Assistant: API Key Error",
                                 "Failed to authenticate with the provided API Key. It has been deleted. Please rerun the script and enter a valid API Key.")
        n.show()
        sys.exit("Failed to authenticate with OpenAI. Exiting.")

def create_assistant(client):
    assistant = client.beta.assistants.create(
        name="NixOS Assistant",
        instructions="You are a helpful assistant.",
        model="gpt-4-turbo-preview"
    )
    return assistant

def get_or_create_assistant(client):
    assistant_id = keyring.get_password("NixOSAssistant", "AssistantID")

    if not assistant_id:
        assistant = create_assistant(client)
        keyring.set_password("NixOSAssistant", "AssistantID", assistant.id)
    else:
        assistant = client.beta.assistants.retrieve(assistant_id)

    return assistant

def create_thread(client):
    thread = client.beta.threads.create()
    return thread

def generate_response(client, assistant_id, thread_id, transcript):
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=transcript
    )

    response = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id
    )

    # Improved error handling and logging
    try:
        response_data = response.to_dict()
        if 'choices' in response_data and response_data['choices']:
            response_text = response_data['choices'][0]['message']['content'].strip()
        else:
            response_text = "I'm sorry, I could not generate a response. Please try again."
            log_interaction("Error", json.dumps(response_data))
    except Exception as e:
        response_text = f"An error occurred: {str(e)}"
        log_interaction("Exception", response_text)

    return response_text

def synthesize_speech(client, text, speech_file_path):
    response = client.audio.speech.create(
        model="tts-1-hd",
        voice="nova",
        response_format="opus",
        input=text
    )
    response.stream_to_file(speech_file_path)

def play_audio(speech_file_path):
    process = subprocess.Popen(['ffmpeg', '-i', str(speech_file_path), '-f', 'alsa', 'default'])
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

        play_audio(welcome_file_path)

        send_notification("NixOS Assistant:", "Recording")
        record_audio(recorded_audio_path)

        play_audio(process_file_path)

        transcript = transcribe_audio(client, recorded_audio_path)
        send_notification("You asked:", transcript)
        print(f"Transcript: {transcript}")

        assistant = get_or_create_assistant(client)
        thread = create_thread(client)
        
        response_text = generate_response(client, assistant.id, thread.id, transcript)
        send_notification("NixOS Assistant:", response_text)
        print(f"Response: {response_text}")
        log_interaction(transcript, response_text)

        play_audio(gotit_file_path)

        synthesize_speech(client, response_text, speech_file_path)

        send_notification("NixOS Assistant:", "Audio Received")
        play_audio(speech_file_path)

    finally:
        delete_lock()

if __name__ == "__main__":
    main()
