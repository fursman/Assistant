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

# ... [previous configurations remain unchanged]

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
                text_chunk = text_queue.get(timeout=0.1)
            except queue.Empty:
                if buffer:
                    send_chunk_to_tts(buffer)
                    buffer = ""
                continue

            if text_chunk is None:
                break

            buffer += text_chunk

            sentences = sentence_end_pattern.split(buffer)
            chunk_to_send = ""

            while len(sentences) > 1:
                chunk_to_send += sentences.pop(0) + " "
                if len(chunk_to_send.split()) >= 50:  # Reduced minimum words for quicker response
                    send_chunk_to_tts(chunk_to_send.strip())
                    chunk_to_send = ""

            buffer = chunk_to_send + ''.join(sentences)

        if buffer:
            send_chunk_to_tts(buffer)

    except Exception as e:
        logger.error(f"Unexpected error in stream_speech: {str(e)}")
    finally:
        if process:
            process.stdin.close()
            process.wait()

def main():
    # ... [previous setup code remains unchanged]

    try:
        create_lock()

        api_key = load_api_key()
        client = OpenAI(api_key=api_key)

        # ... [assistant and thread setup remains unchanged]

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
