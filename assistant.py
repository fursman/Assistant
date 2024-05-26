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

# Configuration for silence detection and volume meter
CHUNK = 1024  # Number of bytes to read from the mic per sample
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Number of audio channels
RATE = 22050  # Sampling rate
THRESHOLD = 1000  # Threshold for silence/noise
SILENCE_LIMIT = 1  # Maximum length of silence in seconds before stopping
PREV_AUDIO_DURATION = 0.5  # Duration of audio to keep before detected speech

# Determine the base directory for logs based on an environment variable or fallback to a directory in /tmp
base_log_dir = Path(os.getenv('LOG_DIR', "/tmp/logs/assistant/"))
print(f"Attempting to create log directory at: {base_log_dir}")
base_log_dir.mkdir(parents=True, exist_ok=True)

# Determine the base directory for assets based on an environment variable or fallback to a default path
assets_directory = Path(os.getenv('AUDIO_ASSETS', Path(__file__).parent / "assets-audio"))
assets_directory.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists, in case the default is used and doesn't exist

# Define file paths
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_csv_path = base_log_dir / "interaction_log.csv"
recorded_audio_path = base_log_dir / f"input_{now}.wav"
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
    # Attempt to retrieve the API key from secure storage
    api_key = keyring.get_password("NixOSAssistant", "APIKey")

    # If the API key does not exist, prompt the user to input it
    if not api_key:
        # Play API key audio
        play_audio(apikey_file_path)
        input_cmd = 'zenity --entry --text="To begin, please enter your OpenAI API Key:" --hide-text'
        api_key = subprocess.check_output(input_cmd, shell=True, text=True).strip()
        
        # Store the new API key securely
        if api_key:
            keyring.set_password("NixOSAssistant", "APIKey", api_key)
        else:
            send_notification("NixOS Assistant Error", "No API Key provided.")
            sys.exit(1)

    return api_key

def handle_api_error():
    # Delete the stored API key
    keyring.delete_password("NixOSAssistant", "APIKey")

    # Notify the user
    send_notification("NixOS Assistant Error", "Invalid API Key. Please re-enter your API key.")

def check_and_kill_existing_process():
    """Check if the lock file exists and kill the existing processes if they do."""
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
    """Create or update a lock file to indicate that the script is running."""
    lock_data = {
        'script_pid': os.getpid(),
        'ffmpeg_pid': ffmpeg_pid
    }
    with open(lock_file_path, 'w') as lock_file:
        json.dump(lock_data, lock_file)

def update_lock_for_ffmpeg_completion():
    """Update the lock file to remove the ffmpeg PID upon completion."""
    if lock_file_path.exists():
        with open(lock_file_path, 'r+') as lock_file:
            lock_data = json.load(lock_file)
            lock_data['ffmpeg_pid'] = None  # Remove ffmpeg PID
            lock_file.seek(0)
            json.dump(lock_data, lock_file)
            lock_file.truncate()

def delete_lock():
    """Delete the lock file to clean up on script completion."""
    try:
        lock_file_path.unlink()
    except OSError:
        pass

def log_interaction(question, response):
    with open(log_csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Get the current date-time in a readable format
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([now, "Question", question])
        writer.writerow([now, "Response", response])

def send_notification(title, message):
    # Initialize notify2 using the application name
    notify2.init('Assistant')

    # Create a Notification object
    n = notify2.Notification(str(title), str(message))
    n.set_timeout(30000)  # Time in milliseconds

    # Display the notification
    n.show()

def calculate_rms(data):
    """Calculate the root mean square of the audio data."""
    rms = np.sqrt(np.mean(np.square(np.frombuffer(data, dtype=np.int16))))
    return rms

def is_silence(data_chunk, threshold=THRESHOLD):
    """Check if the given audio data_chunk contains silence defined by the threshold."""
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

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()
    print("Recording stopped.")

    # Save the recorded data as a WAV file
    wf = wave.open(str(file_path), 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(audio.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

def play_audio(speech_file_path):
    """Play audio using ffmpeg and update lock file for process management."""
    process = subprocess.Popen(['ffmpeg', '-i', str(speech_file_path), '-f', 'alsa', 'default'])
    create_lock(ffmpeg_pid=process.pid)  # Update lock file with ffmpeg PID
    process.wait()  # Wait for the ffmpeg process to finish
    update_lock_for_ffmpeg_completion()  # Remove ffmpeg PID from lock file

class AudioEventHandler(AssistantEventHandler):
    def __init__(self, audio_output_path):
        self.audio_output_path = audio_output_path
        self.audio_file = open(audio_output_path, 'wb')

    def on_audio_delta(self, delta, snapshot):
        self.audio_file.write(delta.audio.data)

    def on_complete(self):
        self.audio_file.close()
        play_audio(self.audio_output_path)

def main():
    # Register signal handlers to ensure clean exit
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Check for existing lock file and attempt to kill the existing process
    check_and_kill_existing_process()
    
    try:
        # Create a lock file to indicate the script is running
        create_lock()
        
        # Load API key
        api_key = load_api_key()

        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)

        # Create an assistant
        assistant = client.beta.assistants.create(
          name="NixOS Assistant",
          instructions="You are a helpful assistant.",
          tools=[],
          model="gpt-4o",
        )

        # Create a thread
        thread = client.beta.threads.create()

        # Play welcome audio
        play_audio(welcome_file_path)

        # Record audio
        send_notification("NixOS Assistant", "Recording")
        record_audio(recorded_audio_path)

        # Play processing audio
        play_audio(process_file_path)

        # Read the recorded audio file
        with open(recorded_audio_path, "rb") as audio_file:
            audio_data = audio_file.read()

        # Create an event handler for streaming audio
        audio_output_path = base_log_dir / f"response_audio_{now}.wav"
        event_handler = AudioEventHandler(audio_output_path)

        # Stream the audio to the assistant and receive the audio response
        with client.beta.threads.runs.stream(
            thread_id=thread.id,
            assistant_id=assistant.id,
            audio=audio_data,
            event_handler=event_handler,
        ) as stream:
            stream.until_done()

    finally:
        # Ensure the lock file is deleted even if an error occurs
        delete_lock()

if __name__ == "__main__":
    main()
