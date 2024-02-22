#!/usr/bin/env python3

import os
import pyaudio
import wave
import subprocess
import notify2
import datetime
import signal
import numpy as np
import audioop  # For calculating the RMS
import sys
import csv
import json  # For handling lock file content as JSON
import select
import keyring
from collections import deque
from pathlib import Path
from openai import OpenAI

# Configuration for silence detection and volume meter
CHUNK = 1024  # Number of bytes to read from the mic per sample
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Number of audio channels
RATE = 22050  # Sampling rate
THRESHOLD = 1000  # Threshold for silence/noise
SILENCE_LIMIT = 1  # Maximum length of silence in seconds before stopping
PREV_AUDIO_DURATION = 0.5  # Duration of audio to keep before detected speech

# Function to get environment variable or default
def ensure_dir_exists(directory: Path):
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

log_dir = Path("/var/log/assistant")
ensure_dir_exists(log_dir)

assets_directory = get_path_or_default("AUDIO_ASSETS", Path(__file__).parent / "assets-audio")
lock_file_path = log_directory / "script.lock"

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_csv_path = log_directory / "interaction_log.csv"
recorded_audio_path = log_directory / f"input_{now}.wav"
speech_file_path = log_directory / f"response_{now}.mp3"
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
                    send_notification("NixOS Assistant:","Silencing output and standing by for your next request!")
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
    n = notify2.Notification(title, message)
    n.set_timeout(30000)  # Time in milliseconds

    # Display the notification
    n.show()

def calculate_rms(data):
    """Calculate the root mean square of the audio data."""
    rms = audioop.rms(data, 2)  # Calculate RMS of the given audio chunk
    return rms

def is_silence(data_chunk, threshold=THRESHOLD):
    """Check if the given audio data_chunk contains silence defined by the threshold.
    Simple implementation could be based on average volume."""
    # Assuming data_chunk is in format pyaudio.paInt16
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
        rms = audioop.rms(data, 2)  # Calculate RMS of the audio chunk, 2 because FORMAT is paInt16
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

def transcribe_audio(client, audio_file_path):
    try:
        with open(audio_file_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        return response
    except Exception as e:  # Catch a broad exception if you're not using openai.error
        # Delete the stored API key using keyring
        keyring.delete_password("NixOSAssistant", "APIKey")
        
        # Notify the user of the error
        notify2.init('Assistant Error')
        n = notify2.Notification("NisOS Assistant: API Key Error",
                                 "Failed to authenticate with the provided API Key. It has been deleted. Please rerun the script and enter a valid API Key.")
        n.show()
        
        # Exit the script
        sys.exit("Failed to authenticate with OpenAI. Exiting.")

def generate_response(client, transcript):
    response = client.chat.completions.create(
      model="gpt-4-turbo-preview",
      messages=[
        {
          "role": "user",
          "content": transcript,
        },
      ],
      temperature=1,
      max_tokens=1514,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    response_text = response.choices[0].message.content.strip()
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
    """Play audio using ffmpeg and update lock file for process management."""
    process = subprocess.Popen(['ffmpeg', '-i', str(speech_file_path), '-f', 'alsa', 'default'])
    create_lock(ffmpeg_pid=process.pid)  # Update lock file with ffmpeg PID
    process.wait()  # Wait for the ffmpeg process to finish
    update_lock_for_ffmpeg_completion()  # Remove ffmpeg PID from lock file

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

        # Play welcome audio
        play_audio(welcome_file_path)

        # Record audio
        send_notification("NixOS Assistant:","Recording")
        record_audio(recorded_audio_path)

        # Play processing audio
        play_audio(process_file_path)

        # Transcribe audio to text
        transcript = transcribe_audio(client, recorded_audio_path)
        send_notification("You asked:",transcript)
        print(f"Transcript: {transcript}")

        # Generate a response
        response_text = generate_response(client, transcript)
        send_notification("NixOS Assistant:",response_text)
        print(f"Response: {response_text}")
        log_interaction(transcript, response_text)

        # Play gotit audio
        play_audio(gotit_file_path)

        # Synthesize speech from the response
        synthesize_speech(client, response_text, speech_file_path)

        # Play the synthesized speech
        send_notification("NixOS Assistant:","Audio Received")
        play_audio(speech_file_path)

    finally:
        # Ensure the lock file is deleted even if an error occurs
        delete_lock()

if __name__ == "__main__":
    main()
