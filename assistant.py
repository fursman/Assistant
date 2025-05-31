#!/usr/bin/env python3
"""
Enhanced realtime voice assistant for Hyprland/NixOS integration.
This version includes improved state management, better microphone control,
and seamless integration with the SUPER key for instant start/stop.

Key improvements:
  • Enhanced state management to prevent listening during assistant speech
  • Improved cleanup and shutdown handling
  • Better VAD (Voice Activity Detection) configuration
  • Optimized audio streaming with proper buffering
  • More reliable IPC communication for instant shutdown
  • Enhanced error handling and recovery
  • Better notification system integration
"""

import asyncio
import os
import json
import base64
import sys
import threading
import getpass
import csv
import datetime
import time
import queue
import signal
from pathlib import Path
from enum import Enum
from typing import Optional

# Audio processing imports
import websockets
import notify2
import sounddevice as sd
import numpy as np
from pydub import AudioSegment
import keyring

# Global configuration
PLAYBACK_SPEED = 1.04        # Increase playback speed by 4%
SAMPLERATE = 24000           # Updated to match OpenAI's preferred sample rate
ASSISTANT_SAMPLERATE = 24000 # Assistant's audio output sample rate (Hz)
CHANNELS = 1
BLOCKSIZE = 2400             # Block size for recording
SOCKET_PATH = '/tmp/assistant.sock'
LOG_CSV_PATH = Path.home() / 'assistant_interactions.csv'

# OpenAI Realtime API configuration (updated for 2025)
API_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
DEFAULT_MODEL = "gpt-4o-realtime-preview-2024-10-01"

class AssistantState(Enum):
    """Enum to track the current state of the assistant"""
    IDLE = "idle"
    LISTENING = "listening" 
    PROCESSING = "processing"
    SPEAKING = "speaking"
    SHUTTING_DOWN = "shutting_down"

def load_api_key():
    """Load the API key from keyring or prompt the user."""
    api_key = keyring.get_password("NixOSAssistant", "APIKey")
    if not api_key:
        api_key = getpass.getpass("Please enter your OpenAI API Key: ").strip()
        if api_key:
            keyring.set_password("NixOSAssistant", "APIKey", api_key)
        else:
            print("No API Key provided. Exiting.")
            sys.exit(1)
    return api_key

def play_audio_file(file_path, volume=1.0):
    """Play an audio file using pydub and sounddevice with volume control."""
    try:
        audio = AudioSegment.from_file(file_path)
        if volume != 1.0:
            audio = audio + (20 * np.log10(volume))  # Adjust volume in dB
        
        samples = np.array(audio.get_array_of_samples())
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
        else:
            samples = samples.reshape((-1, 1))
        
        sd.play(samples, samplerate=audio.frame_rate)
        sd.wait()
    except Exception as e:
        print(f"Error playing audio file {file_path}: {e}")

def log_interaction(question, response):
    """Log a conversation interaction (question and reply) to a CSV file."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        # Ensure the CSV file has headers
        if not LOG_CSV_PATH.exists():
            with open(LOG_CSV_PATH, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "Type", "Content"])
        
        with open(LOG_CSV_PATH, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([now, "Question", question])
            writer.writerow([now, "Response", response])
    except Exception as e:
        print(f"Error logging interaction: {e}")

def send_notification(title, message, timeout=5000):
    """Send a desktop notification using notify2."""
    try:
        if not notify2.is_initted():
            notify2.init('Assistant')
        n = notify2.Notification(title, message)
        n.set_timeout(timeout)
        n.show()
    except Exception as e:
        print(f"Notification error: {e}")

class AssistantSession:
    def __init__(self, api_key, assets_directory, welcome_file, gotit_file):
        self.api_key = api_key
        self.assets_directory = assets_directory
        self.welcome_file = welcome_file
        self.gotit_file = gotit_file
        self.api_url = API_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        
        # State management
        self.state = AssistantState.IDLE
        self.shutdown_event = asyncio.Event()
        self.audio_queue = queue.Queue()
        self.assistant_output_stream = None
        self.mic_stream = None
        
        # Response tracking
        self.current_response = ""
        self.current_question = ""
        self.response_id = None
        
        # Audio control
        self.audio_buffer = []
        self.speaking_start_time = None
        
        # Tasks tracking
        self.tasks = []

    def set_state(self, new_state: AssistantState):
        """Thread-safe state management"""
        if self.state != new_state:
            print(f"State transition: {self.state.value} -> {new_state.value}")
            self.state = new_state

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio recording with state-aware processing."""
        if status:
            print(f"Audio callback status: {status}", file=sys.stderr)
        
        # Only process audio when in listening state
        if self.state in [AssistantState.LISTENING, AssistantState.IDLE]:
            try:
                self.audio_queue.put(indata.copy(), block=False)
            except queue.Full:
                # Drop oldest audio if queue is full
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put(indata.copy(), block=False)
                except queue.Empty:
                    pass

    def flush_audio_queue(self):
        """Empty the microphone audio queue."""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    async def send_audio(self, websocket):
        """Continuously send recorded audio chunks to the websocket."""
        loop = asyncio.get_event_loop()
        try:
            while not self.shutdown_event.is_set():
                try:
                    # Use a timeout to check shutdown periodically
                    indata = await asyncio.wait_for(
                        loop.run_in_executor(None, self.audio_queue.get), 
                        timeout=0.1
                    )
                    
                    if self.state in [AssistantState.LISTENING, AssistantState.IDLE]:
                        audio_bytes = indata.tobytes()
                        b64_audio = base64.b64encode(audio_bytes).decode('utf-8')
                        event = {
                            "type": "input_audio_buffer.append",
                            "audio": b64_audio,
                        }
                        await websocket.send(json.dumps(event))
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    if not self.shutdown_event.is_set():
                        print(f"Error in send_audio: {e}")
                    break
        except asyncio.CancelledError:
            print("Audio sending task cancelled.")

    async def receive_messages(self, websocket):
        """Receive and process events from the websocket."""
        try:
            while not self.shutdown_event.is_set():
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    print("WebSocket connection closed.")
                    self.shutdown_event.set()
                    break
                
                try:
                    event = json.loads(message)
                    await self.handle_event(event)
                except json.JSONDecodeError:
                    print("Received invalid JSON message.")
                    continue
                except Exception as e:
                    print(f"Error handling event: {e}")
                    continue
                    
        except asyncio.CancelledError:
            print("Message receiving task cancelled.")

    async def handle_event(self, event):
        """Handle individual WebSocket events."""
        event_type = event.get("type", "")
        
        if event_type == "session.created":
            print("✓ Session created successfully")
            self.set_state(AssistantState.LISTENING)
            
        elif event_type == "session.updated":
            print("✓ Session configuration updated")
            
        elif event_type == "input_audio_buffer.speech_started":
            print("🎤 Speech detected")
            self.set_state(AssistantState.PROCESSING)
            
        elif event_type == "input_audio_buffer.speech_stopped":
            print("🔇 Speech ended, processing...")
            
        elif event_type == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript", "").strip()
            if transcript:
                self.current_question = transcript
                print(f"\n👤 You: {transcript}")
                send_notification("You said:", transcript, timeout=3000)
            
        elif event_type == "response.created":
            self.response_id = event.get("response", {}).get("id")
            print("🤖 Assistant is preparing response...")
            self.current_response = ""
            
        elif event_type == "response.audio.delta":
            await self.handle_audio_delta(event)
            
        elif event_type == "response.audio_transcript.delta":
            delta = event.get("delta", "")
            self.current_response += delta
            print(delta, end='', flush=True)
            
        elif event_type == "response.audio.done":
            await self.finish_audio_playback()
            
        elif event_type == "response.done":
            await self.handle_response_complete()
            
        elif event_type == "error":
            error_info = event.get("error", {})
            print(f"❌ API Error: {error_info}")
            send_notification("Assistant Error", str(error_info))

    async def handle_audio_delta(self, event):
        """Handle streaming audio from the assistant."""
        delta = event.get("delta", "")
        if not delta:
            return
            
        try:
            chunk = base64.b64decode(delta)
            
            # Initialize audio stream if needed
            if self.assistant_output_stream is None:
                self.set_state(AssistantState.SPEAKING)
                self.speaking_start_time = time.time()
                self.assistant_output_stream = sd.RawOutputStream(
                    samplerate=ASSISTANT_SAMPLERATE,
                    channels=1,
                    dtype='int16',
                    blocksize=BLOCKSIZE
                )
                self.assistant_output_stream.start()
                # Clear any queued microphone input
                self.flush_audio_queue()
            
            # Stream the audio chunk
            self.assistant_output_stream.write(chunk)
            
        except Exception as e:
            print(f"Error processing audio delta: {e}")

    async def finish_audio_playback(self):
        """Clean up after assistant finishes speaking."""
        if self.assistant_output_stream is not None:
            try:
                # Allow some time for audio to finish playing
                await asyncio.sleep(0.2)
                self.assistant_output_stream.stop()
                self.assistant_output_stream.close()
                self.assistant_output_stream = None
                print("\n🔊 Assistant audio complete")
                
                # Brief pause before resuming listening
                await asyncio.sleep(0.3)
                self.flush_audio_queue()  # Clear any audio captured during speech
                
            except Exception as e:
                print(f"Error finishing audio playback: {e}")
            finally:
                self.assistant_output_stream = None

    async def handle_response_complete(self):
        """Handle completion of assistant response."""
        print(f"\n✓ Response complete")
        
        # Log the interaction
        if self.current_question and self.current_response:
            log_interaction(self.current_question, self.current_response)
            
        # Send notification
        response_preview = self.current_response[:100] + "..." if len(self.current_response) > 100 else self.current_response
        send_notification("Assistant", response_preview or "Response completed")
        
        # Reset for next interaction
        self.current_question = ""
        self.current_response = ""
        self.response_id = None
        
        # Return to listening state
        self.set_state(AssistantState.LISTENING)

    async def ipc_server(self):
        """Set up IPC server for external shutdown commands."""
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)

        async def handle_client(reader, writer):
            try:
                data = await reader.read(100)
                message = data.decode().strip()
                print(f"📨 Received IPC message: {message}")
                
                if message == "shutdown":
                    print("🛑 Shutdown command received")
                    self.set_state(AssistantState.SHUTTING_DOWN)
                    self.shutdown_event.set()
                    writer.write(b"ack")
                    await writer.drain()
            except Exception as e:
                print(f"Error in IPC server: {e}")
            finally:
                writer.close()
                await writer.wait_closed()

        try:
            server = await asyncio.start_unix_server(handle_client, path=SOCKET_PATH)
            print(f"📡 IPC server listening on {SOCKET_PATH}")
            return server
        except Exception as e:
            print(f"Failed to start IPC server: {e}")
            return None

    async def cleanup(self):
        """Perform thorough cleanup of all resources."""
        print("🧹 Cleaning up resources...")
        
        self.set_state(AssistantState.SHUTTING_DOWN)
        
        # Cancel all running tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Stop audio streams
        if self.assistant_output_stream and not self.assistant_output_stream.closed:
            try:
                self.assistant_output_stream.abort()
                self.assistant_output_stream.close()
            except:
                pass
                
        if self.mic_stream and self.mic_stream.active:
            try:
                self.mic_stream.abort()
                self.mic_stream.close()
            except:
                pass
        
        # Clean up IPC socket
        if os.path.exists(SOCKET_PATH):
            try:
                os.remove(SOCKET_PATH)
            except:
                pass

    async def run(self):
        """Main entry point for the assistant session."""
        ipc_server = None
        websocket = None
        
        try:
            # Start IPC server
            ipc_server = await self.ipc_server()
            
            # Connect to OpenAI
            print("🔗 Connecting to OpenAI Realtime API...")
            websocket = await websockets.connect(
                self.api_url, 
                extra_headers=self.headers,
                ping_interval=20,
                ping_timeout=10
            )
            print("✅ Connected to OpenAI Realtime API")
            
            # Play welcome sound
            play_audio_file(self.welcome_file, volume=0.7)
            
            # Configure session
            session_config = {
                "type": "session.update",
                "session": {
                    "modalities": ["audio", "text"],
                    "voice": "alloy",
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "model": "whisper-1"
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 800
                    },
                    "instructions": (
                        "You are a helpful, efficient desktop assistant integrated with NixOS and Hyprland. "
                        "Keep responses concise but informative. Speak naturally and quickly. "
                        "You can help with system tasks, development questions, and general assistance."
                    ),
                    "temperature": 0.8,
                    "max_response_output_tokens": 4096
                }
            }
            
            await websocket.send(json.dumps(session_config))
            
            # Start microphone
            self.mic_stream = sd.InputStream(
                samplerate=SAMPLERATE,
                channels=CHANNELS,
                dtype='int16',
                callback=self.audio_callback,
                blocksize=BLOCKSIZE
            )
            self.mic_stream.start()
            print("🎤 Microphone started")
            
            # Start main processing tasks
            send_task = asyncio.create_task(self.send_audio(websocket))
            receive_task = asyncio.create_task(self.receive_messages(websocket))
            self.tasks = [send_task, receive_task]
            
            # Send initial notification
            send_notification("Voice Assistant", "Ready to listen! Press SUPER again to stop.")
            
            # Wait for shutdown
            await self.shutdown_event.wait()
            
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
        finally:
            await self.cleanup()
            if websocket:
                await websocket.close()
            if ipc_server:
                ipc_server.close()
                await ipc_server.wait_closed()

async def send_shutdown_command():
    """Send shutdown command to running assistant instance."""
    try:
        reader, writer = await asyncio.open_unix_connection(SOCKET_PATH)
        writer.write(b"shutdown")
        await writer.drain()
        
        # Wait for acknowledgment
        data = await asyncio.wait_for(reader.read(100), timeout=2.0)
        if data.decode().strip() == "ack":
            print("✅ Assistant stopped successfully")
            return True
        else:
            print("⚠️  Unexpected response from assistant")
            return False
    except asyncio.TimeoutError:
        print("⏰ Timeout waiting for assistant response")
        return False
    except FileNotFoundError:
        print("ℹ️  No running assistant instance found")
        return False
    except Exception as e:
        print(f"❌ Error sending shutdown command: {e}")
        return False
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except:
            pass

def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}, shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """Main function that handles assistant lifecycle."""
    setup_signal_handlers()
    
    print("🚀 Starting Voice Assistant for Hyprland/NixOS")
    
    # Check if assistant is already running
    if os.path.exists(SOCKET_PATH):
        print("🔄 Assistant already running, sending shutdown command...")
        if await send_shutdown_command():
            # Wait a moment for cleanup
            await asyncio.sleep(0.5)
            return
        else:
            print("⚠️  Failed to stop existing instance, trying to start anyway...")
    
    # Initialize notifications
    try:
        notify2.init('Assistant')
    except Exception as e:
        print(f"Warning: Could not initialize notifications: {e}")
    
    # Set up audio assets directory
    assets_directory = Path(os.getenv('AUDIO_ASSETS', Path(__file__).parent / "assets-audio"))
    assets_directory.mkdir(parents=True, exist_ok=True)
    
    welcome_file = assets_directory / "welcome.mp3"
    gotit_file = assets_directory / "gotit.mp3"
    
    # Check for required audio files
    if not welcome_file.is_file():
        print(f"❌ Welcome audio file not found at {welcome_file}")
        print("Please ensure welcome.mp3 exists in the assets directory")
        sys.exit(1)
    
    if not gotit_file.is_file():
        print(f"❌ Gotit audio file not found at {gotit_file}")
        print("Please ensure gotit.mp3 exists in the assets directory")
        sys.exit(1)
    
    # Load API key
    try:
        api_key = load_api_key()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    
    # Create and run assistant session
    session = AssistantSession(api_key, assets_directory, welcome_file, gotit_file)
    
    try:
        await session.run()
    except KeyboardInterrupt:
        print("\n👋 Assistant stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        print("🏁 Assistant session ended")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
