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
response_received_event = Event()

class AssistantSession:
    def __init__(self):
        self.ws_app = None
        self.response_text = []

def signal_handler(sig, frame, session):
    delete_lock()
    if session.ws_app:
        session.ws_app.close()
    sys.exit(0)

def load_api_key():
    api_key = keyring.get_password("NixOSAssistant", "APIKey")
    if not api_key:
        if not shutil.which("zenity"):
            logger.error("Zenity is not installed, and no API key is provided.")
            sys.exit(1)
        play_audio(apikey_file_path)
        input_cmd = 'zenity --entry --text="To begin, please enter your OpenAI API Key:" --hide-text'
        try:
            api_key = subprocess.check_output(input_cmd, shell=True, text=True).strip()
        except subprocess.CalledProcessError:
            logger.error("Failed to obtain API key from user.")
            sys.exit(1)
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
    if not notify2.is_initted():
        notify2.init('Assistant')
    n = notify2.Notification(title, message)
    n.set_timeout(30000)
    n.show()

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

    if response_type == "response.content_part.added":
        content = response.get("content", [{}])[0]
        if content.get("type") == "text":
            text = content.get("text", "")
            logger.info(f"Received partial text response: {text}")
            session.response_text.append(text)
    elif response_type == "response.done":
        logger.info("Response completed.")
        response_received_event.set()
    elif response_type in ("session.created", "conversation.item.created", "response.created"):
        logger.info(f"Received session-related message: {response_type}")
    else:
        logger.warning(f"Unexpected message type received: {response_type}")
        logger.debug(f"Full response received: {response}")

def on_error(ws, error):
    logger.error(f"WebSocket error: {error}")
    # Attempt to reconnect if the socket is closed unexpectedly
    if isinstance(error, websocket.WebSocketConnectionClosedException):
        logger.info("WebSocket closed unexpectedly, attempting to reconnect...")
        reconnect()

def on_close(ws, close_status_code, close_msg):
    logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
    ws_open_event.clear()  # Clear the event when the WebSocket is closed

def reconnect():
    if session.ws_app:
        session.ws_app.close()
    api_key = load_api_key()
    session.ws_app = start_realtime_session(api_key)
    ws_thread = threading.Thread(target=session.ws_app.run_forever, daemon=True)
    ws_thread.start()

def is_ws_connected():
    return session.ws_app and session.ws_app.sock and session.ws_app.sock.connected

def main():
    global session
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, session))
    signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame, session))

    check_and_kill_existing_process()

    try:
        create_lock()

        api_key = load_api_key()
        session.ws_app = start_realtime_session(api_key)

        ws_thread = threading.Thread(target=session.ws_app.run_forever, daemon=True)
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
            if is_ws_connected():
                session.ws_app.send(json.dumps({
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
                # Wait for the response to be received
                response_received_event.wait(timeout=30)
                if response_received_event.is_set():
                    print("Response from Assistant:", " ".join(session.response_text).strip())
                else:
                    logger.error("Timeout waiting for response from server.")
                    session.ws_app.close()
            else:
                logger.error("WebSocket is not connected. Unable to send message.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        send_notification("NixOS Assistant Error", f"An error occurred: {str(e)}")
    finally:
        if session.ws_app:
            session.ws_app.close()
        delete_lock()

if __name__ == "__main__":
    session = AssistantSession()
    main()