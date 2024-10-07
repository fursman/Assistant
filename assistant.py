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
import websocket
import threading
import queue
import base64
from pathlib import Path
from pydub import AudioSegment

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Configuration for silence detection and volume meter
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
THRESHOLD = 30
SILENCE_LIMIT = 4
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
    if ws_app:
        ws_app.close()
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

def start_realtime_session(api_key):
    ws_url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "realtime=v1"
    }
    ws = websocket.WebSocketApp(ws_url, header=headers, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.on_open = on_open
    return ws

def on_open(ws):
    logger.info("Connected to Realtime API.")
    ws.send(json.dumps({
        "type": "response.create",
        "response": {
            "modalities": ["text", "audio"],
            "instructions": "Please assist the user."
        }
    }))

def on_message(ws, message):
    logger.debug(f"Received message: {message}")
    response = json.loads(message)
    if response.get("type") == "conversation.item.create":
        content = response["item"]["content"][0]
        if content["type"] == "input_text":
            logger.info(f"Received text response: {content['text']}")
            print("Response from Assistant:", content["text"])
        elif content["type"] == "input_audio":
            logger.info("Received audio response, playing audio.")
            audio_bytes = base64.b64decode(content["audio"])
            play_audio_from_bytes(audio_bytes)
    else:
        logger.warning(f"Unexpected message type received: {response.get('type')}")

def on_error(ws, error):
    logger.error(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")

def play_audio_from_bytes(audio_bytes):
    process = subprocess.Popen(['ffplay', '-autoexit', '-nodisp', '-'], stdin=subprocess.PIPE)
    process.stdin.write(audio_bytes)
    process.stdin.close()
    process.wait()

def audio_to_base64_chunks(audio_bytes, chunk_size=32000):
    for i in range(0, len(audio_bytes), chunk_size):
        yield base64.b64encode(audio_bytes[i:i+chunk_size]).decode()

def main():
    global ws_app
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    check_and_kill_existing_process()

    try:
        create_lock()

        api_key = load_api_key()
        ws_app = start_realtime_session(api_key)

        ws_thread = threading.Thread(target=ws_app.run_forever, daemon=True)
        ws_thread.start()

        if len(sys.argv) > 1:
            # Command-line input
            transcript = " ".join(sys.argv[1:])
            logger.info(f"Sending text input: {transcript}")
            ws_app.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{
                        "type": "input_text",
                        "text": transcript
                    }]
                }
            }))
        else:
            # Audio input
            play_audio(welcome_file_path)
            send_notification("NixOS Assistant:", "Recording")
            record_audio(recorded_audio_path)
            play_audio(process_file_path)

            with open(recorded_audio_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
                for audio_chunk in audio_to_base64_chunks(audio_bytes):
                    logger.debug("Sending audio chunk to server.")
                    ws_app.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": audio_chunk
                    }))
                logger.info("Committing audio buffer and requesting response.")
                ws_app.send(json.dumps({"type": "input_audio_buffer.commit"}))
                ws_app.send(json.dumps({"type": "response.create"}))

        ws_thread.join()
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        send_notification("NixOS Assistant Error", f"An error occurred: {str(e)}")
    finally:
        if ws_app:
            ws_app.close()
        delete_lock()

if __name__ == "__main__":
    ws_app = None
    main()