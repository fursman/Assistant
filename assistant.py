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
import time
from pathlib import Path
from pydub import AudioSegment
from threading import Event

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

# Event to track when the WebSocket connection is open
ws_open_event = Event()

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
    ws_open_event.set()  # Signal that the WebSocket is open

def on_message(ws, message):
    logger.debug(f"Received message: {message}")
    try:
        response = json.loads(message)
    except json.JSONDecodeError:
        logger.error("Failed to decode message from server.")
        return

    response_type = response.get("type")

    if response_type == "conversation.item.create":
        content = response.get("item", {}).get("content", [{}])[0]
        if content.get("type") == "input_text":
            text = content.get("text", "")
            logger.info(f"Received text response: {text}")
            print("Response from Assistant:", text)
            ws.close()  # Close the WebSocket after receiving the text response
        else:
            logger.warning(f"Unexpected content type received: {content.get('type')}")
    elif response_type == "response.done":
        logger.info("Response completed.")
        ws.close()
    else:
        logger.warning(f"Unexpected message type received: {response_type}")

def on_error(ws, error):
    logger.error(f"WebSocket error: {error}")
    # Attempt to reconnect if the socket is closed unexpectedly
    if isinstance(error, websocket.WebSocketConnectionClosedException):
        logger.info("WebSocket closed unexpectedly, attempting to reconnect...")
        reconnect()

def on_close(ws, close_status_code, close_msg):
    logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
    ws_open_event.clear()  # Clear the event when the WebSocket is closed

def play_audio_from_bytes(audio_bytes):
    process = subprocess.Popen(['ffplay', '-autoexit', '-nodisp', '-'], stdin=subprocess.PIPE)
    try:
        process.stdin.write(audio_bytes)
        process.stdin.close()
        process.wait()
    except BrokenPipeError:
        logger.error("Failed to play audio due to broken pipe.")

def audio_to_base64_chunks(audio_bytes, chunk_size=32000):
    for i in range(0, len(audio_bytes), chunk_size):
        yield base64.b64encode(audio_bytes[i:i+chunk_size]).decode()

def reconnect():
    global ws_app
    if ws_app:
        ws_app.close()
    api_key = load_api_key()
    ws_app = start_realtime_session(api_key)
    ws_thread = threading.Thread(target=ws_app.run_forever, daemon=True)
    ws_thread.start()

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

        # Wait until the WebSocket is open before proceeding
        ws_open_event.wait(timeout=10)
        if not ws_open_event.is_set():
            logger.error("WebSocket connection timed out.")
            return

        if len(sys.argv) > 1:
            # Command-line input
            transcript = " ".join(sys.argv[1:])
            logger.info(f"Sending text input: {transcript}")
            if ws_app and ws_app.sock and ws_app.sock.connected:
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
                logger.error("WebSocket is not connected. Unable to send message.")
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
                    if ws_app and ws_app.sock and ws_app.sock.connected:
                        ws_app.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": audio_chunk
                        }))
                    else:
                        logger.error("WebSocket is not connected. Unable to send audio chunk.")
                        return
                logger.info("Committing audio buffer and requesting response.")
                if ws_app and ws_app.sock and ws_app.sock.connected:
                    ws_app.send(json.dumps({"type": "input_audio_buffer.commit"}))
                    ws_app.send(json.dumps({"type": "response.create"}))
                else:
                    logger.error("WebSocket is not connected. Unable to commit audio buffer.")

        # Wait for a reasonable amount of time for the response
        start_time = time.time()
        while ws_app.sock and ws_app.sock.connected:
            if time.time() - start_time > 30:  # Timeout after 30 seconds
                logger.error("Timeout waiting for response from server.")
                ws_app.close()
                break
            time.sleep(1)

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