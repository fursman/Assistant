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

    try:
        while True:
            text_chunk = text_queue.get()
            if text_chunk is None:
                break
            full_text += text_chunk

            # Start streaming when we have enough text or encounter sentence-ending punctuation
            if len(full_text) >= 50 or text_chunk.endswith(('.', '!', '?', '\n')):
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
    finally:
        if ffplay_process and ffplay_process.poll() is None:
            ffplay_process.stdin.close()
            ffplay_process.wait()
        update_lock_for_ffplay_completion()

def get_context(question):
    context = question
    if "nixos" in question.lower():
        try:
            with open('/etc/nixos/flake.nix', 'r') as file:
                nixos_config = file.read()
            context += f"\n\nFor additional context, this is the system's current flake.nix configuration:\n{nixos_config}"
        except FileNotFoundError:
            context += "\n\nNOTE: The flake.nix file was not found in the expected location."
        except Exception as e:
            context += f"\n\nNOTE: An error occurred while trying to read the flake.nix file: {str(e)}"
    if "clipboard" in question.lower():
        try:
            clipboard_content = subprocess.check_output(['wl-paste'], text=True)
            context += f"\n\nFor additional context, this is the current clipboard content:\n{clipboard_content}"
        except subprocess.CalledProcessError:
            context += "\n\nNOTE: Failed to retrieve clipboard content. The clipboard might be empty or contain non-text data."
        except Exception as e:
            context += f"\n\nNOTE: An unexpected error occurred while retrieving clipboard content: {str(e)}"
    return context

def play_audio(file_path):
    subprocess.run(['ffplay', '-autoexit', '-nodisp', str(file_path)], check=True)

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

        with open(recorded_audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )

        context = get_context(transcript)
        send_notification("You asked:", transcript)
        add_message(client, thread_id, context)

        tts_queue = queue.Queue()
        assistant_thread = threading.Thread(target=run_assistant, args=(client, thread_id, assistant_id, tts_queue))
        tts_thread = threading.Thread(target=stream_speech, args=(client, tts_queue))

        assistant_thread.start()
        tts_thread.start()

        assistant_thread.join()
        tts_thread.join()

        log_interaction(transcript, "Response logged (streaming)")

    except Exception as e:
        send_notification("NixOS Assistant Error", f"An error occurred: {str(e)}")
    finally:
        delete_lock()

if __name__ == "__main__":
    main()
