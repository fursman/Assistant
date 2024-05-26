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
import json
import keyring
from pathlib import Path
import openai
from openai import OpenAI, AssistantEventHandler

# Configuration for silence detection
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
THRESHOLD = 1000
SILENCE_LIMIT = 1

# Directories for logs and assets
base_log_dir = Path(os.getenv('LOG_DIR', "/tmp/logs/assistant/"))
base_log_dir.mkdir(parents=True, exist_ok=True)
assets_directory = Path(os.getenv('AUDIO_ASSETS', Path(__file__).parent / "assets-audio"))
assets_directory.mkdir(parents=True, exist_ok=True)

# File paths
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
recorded_audio_path = base_log_dir / f"input_{now}.wav"
speech_file_path = base_log_dir / f"response_{now}.mp3"
lock_file_path = base_log_dir / "script.lock"

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
        api_key = subprocess.check_output('zenity --entry --text="Enter OpenAI API Key:" --hide-text', shell=True, text=True).strip()
        if api_key:
            keyring.set_password("NixOSAssistant", "APIKey", api_key)
        else:
            send_notification("NixOS Assistant Error", "No API Key provided.")
            sys.exit(1)
    return api_key

def handle_api_error():
    keyring.delete_password("NixOSAssistant", "APIKey")
    send_notification("NixOS Assistant Error", "Invalid API Key. Please re-enter your API key.")
    sys.exit(1)

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
            except json.JSONDecodeError:
                sys.exit("Lock file is corrupt. Exiting.")
            except ProcessLookupError:
                pass
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

def send_notification(title, message):
    notify2.init('Assistant')
    n = notify2.Notification(title, message)
    n.set_timeout(30000)
    n.show()

def calculate_rms(data):
    as_ints = np.frombuffer(data, dtype=np.int16)
    if as_ints.size == 0:
        return 0
    mean_square = np.mean(np.square(as_ints))
    rms = np.sqrt(mean_square)
    return rms if np.isfinite(rms) else 0

def is_silence(data_chunk, threshold=THRESHOLD):
    return calculate_rms(data_chunk) < threshold

def record_audio(file_path):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    silent_frames = 0
    silence_threshold = int(RATE / CHUNK * SILENCE_LIMIT)

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
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

    with wave.open(str(file_path), 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

def transcribe_audio(client, audio_file_path):
    try:
        with open(audio_file_path, "rb") as audio_file:
            response = client.Audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        return response['text']
    except openai.error.OpenAIError:
        handle_api_error()

def create_assistant(client):
    return client.Assistants.create(
        name="NixOS Assistant",
        instructions="You are an assistant helping with various tasks.",
        tools=[{"type": "code_interpreter"}],
        model="gpt-4"
    )

def create_thread(client):
    return client.Threads.create()

def add_message_to_thread(client, thread_id, message_content):
    return client.Threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=message_content
    )

class MyEventHandler(AssistantEventHandler):
    def on_text_created(self, text):
        print(f"\nassistant > ", end="", flush=True)
    def on_text_delta(self, delta, snapshot):
        print(delta.value, end="", flush=True)
    def on_tool_call_created(self, tool_call):
        print(f"\nassistant > {tool_call.type}\n", flush=True)
    def on_tool_call_delta(self, delta, snapshot):
        if delta.type == 'code_interpreter':
            if delta.code_interpreter.input:
                print(delta.code_interpreter.input, end="", flush=True)
            if delta.code_interpreter.outputs:
                print(f"\n\noutput >", flush=True)
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        print(f"\n{output.logs}", flush=True)

def stream_run(client, thread_id, assistant_id):
    event_handler = MyEventHandler()
    with client.Threads.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant_id,
        event_handler=event_handler,
    ) as stream:
        stream.until_done()

def play_audio(file_path):
    process = subprocess.Popen(['ffmpeg', '-i', str(file_path), '-f', 'alsa', 'default'])
    create_lock(ffmpeg_pid=process.pid)
    process.wait()
    update_lock_for_ffmpeg_completion()

def synthesize_speech(client, text, speech_file_path):
    response = client.Audio.speech.create(
        model="tts-1-hd",
        voice="nova",
        response_format="opus",
        input=text
    )
    response.stream_to_file(speech_file_path)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    check_and_kill_existing_process()
    
    try:
        create_lock()
        api_key = load_api_key()
        openai.api_key = api_key

        assistant = create_assistant(openai)
        thread = create_thread(openai)

        play_audio(welcome_file_path)
        send_notification("NixOS Assistant:", "Recording")
        record_audio(recorded_audio_path)
        play_audio(process_file_path)

        transcript = transcribe_audio(openai, recorded_audio_path)
        send_notification("You asked:", transcript)
        add_message_to_thread(openai, thread.id, transcript)

        stream_run(openai, thread.id, assistant.id)

        play_audio(gotit_file_path)
        response_text = transcribe_audio(openai, recorded_audio_path)
        synthesize_speech(openai, response_text, speech_file_path)
        send_notification("NixOS Assistant:", "Audio Received")
        play_audio(speech_file_path)

    finally:
        delete_lock()

if __name__ == "__main__":
    main()

# V 0.6
