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

# ... [Previous imports and configurations remain the same]

def signal_handler(sig, frame):
    print("\nExiting gracefully...")
    delete_lock()
    sys.exit(0)

# ... [Other functions remain the same]

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
    process = None

    try:
        while True:
            text_chunk = text_queue.get()
            if text_chunk is None:
                break
            full_text += text_chunk

            if len(full_text) >= 5 or text_chunk.endswith(('.', '!', '?', '\n')):
                if process is None:
                    process = subprocess.Popen(['ffplay', '-autoexit', '-nodisp', '-'], stdin=subprocess.PIPE)

                response = client.audio.speech.create(
                    model="tts-1-hd",
                    voice="nova",
                    input=full_text
                )
                
                for chunk in response.iter_bytes(chunk_size=1024):
                    if chunk:
                        process.stdin.write(chunk)
                        process.stdin.flush()
                
                full_text = ""

    finally:
        if process:
            process.stdin.close()
            process.wait()

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

# ... [The rest of the script remains the same]
