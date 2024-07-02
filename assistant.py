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
from typing_extensions import override
import threading
import queue

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
THRESHOLD = 20
SILENCE_LIMIT = 1
PREV_AUDIO_DURATION = 0.5

base_log_dir = Path(os.getenv('LOG_DIR', "/tmp/logs/assistant/"))
base_log_dir.mkdir(parents=True, exist_ok=True)

assets_directory = Path(os.getenv('AUDIO_ASSETS', Path(__file__).parent / "assets-audio"))
assets_directory.mkdir(parents=True, exist_ok=True)

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
                ffmpeg_pid = lock_data.get('ffmpeg_pid')
                for pid in [ffmpeg_pid, script_pid]:
                    if pid:
                        try:
                            os.kill(pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                send_notification("NixOS Assistant:", "Silencing output and standing by for your next request!")
                sys.exit("Exiting.")
            except (json.JSONDecodeError, PermissionError):
                sys.exit(1)

def create_lock(ffmpeg_pid=None):
    lock_data = {
        'script_pid': os.getpid(),
        'ffmpeg_pid': ffmpeg_pid
    }
    with open(lock_file_path, 'w') as lock_file:
        json.dump(lock_data, lock_file)

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
        model="gpt-4-1106-preview"
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
        pass

    @override
    def on_text_delta(self, delta, snapshot):
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

    return event_handler.response_text

def stream_speech(client, text_queue):
    buffer = ""
    process = None
    sentence_end_pattern = re.compile(r'(?<=[.!?])\s+')

    def send_chunk_to_tts(chunk):
        nonlocal process
        try:
            response = client.audio.speech.create(
                model="tts-1-hd",
                voice="nova",
                input=chunk
            )
            if not process:
                process = subprocess.Popen(['ffplay', '-autoexit', '-nodisp', '-'], stdin=subprocess.PIPE)
            for audio_chunk in response.iter_bytes(chunk_size=4096):
                if audio_chunk:
                    process.stdin.write(audio_chunk)
                    process.stdin.flush()
        except Exception as e:
            logger.error(f"Error in TTS or audio playback: {str(e)}")

    try:
        while True:
            try:
                text_chunk = text_queue.get(timeout=0.05)  # Reduced timeout for faster processing
            except queue.Empty:
                if buffer:
                    sentences = sentence_end_pattern.split(buffer)
                    if len(sentences) > 1 and len(buffer.split()) >= 50:
                        chunk_to_send = sentence_end_pattern.join(sentences[:-1])
                        send_chunk_to_tts(chunk_to_send.strip())
                        buffer = sentences[-1]
                continue

            if text_chunk is None:
                break

            buffer += text_chunk

            sentences = sentence_end_pattern.split(buffer)
            complete_sentences = sentences[:-1]
            
            if complete_sentences and len(' '.join(complete_sentences).split()) >= 50:
                chunk_to_send = sentence_end_pattern.join(complete_sentences)
                send_chunk_to_tts(chunk_to_send.strip())
                buffer = sentences[-1]

        if buffer:
            send_chunk_to_tts(buffer.strip())

    except Exception as e:
        logger.error(f"Unexpected error in stream_speech: {str(e)}")
    finally:
        if process:
            process.stdin.close()
            process.wait()

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
            clipboard_content = subprocess.check_output(['wl-paste'], text=True, timeout=2)
            context += f"\n\nFor additional context, this is the current clipboard content:\n{clipboard_content}"
        except subprocess.CalledProcessError:
            context += "\n\nFailed to retrieve clipboard content. The clipboard might be empty or contain non-text data."
        except subprocess.TimeoutExpired:
            context += "\n\nTimeout occurred while trying to retrieve clipboard content."
        except Exception as e:
            context += f"\n\nAn unexpected error occurred while retrieving clipboard content: {e}"
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

        if len(sys.argv) > 1:
            # Command-line input
            transcript = " ".join(sys.argv[1:])
            is_text_input = True
        else:
            # Audio input
            is_text_input = False
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
        if not is_text_input:
            send_notification("You asked:", transcript)
        add_message(client, thread_id, context)

        tts_queue = queue.Queue()
        tts_thread = threading.Thread(target=stream_speech, args=(client, tts_queue))
        tts_thread.start()

        response = run_assistant(client, thread_id, assistant_id, tts_queue)
        
        if is_text_input:
            print(f"assistant > {response.strip()}")
        else:
            send_notification("NixOS Assistant:", response)
        
        tts_queue.put(None)  # Signal the TTS thread to finish
        tts_thread.join()

        log_interaction(transcript, response)

    except Exception as e:
        send_notification("NixOS Assistant Error", f"An error occurred: {str(e)}")
    finally:
        delete_lock()

if __name__ == "__main__":
    main()
