#!/usr/bin/env python3
"""
Hyprland Voice Assistant
A voice-activated assistant for Hyprland with VAD, STT, LLM, and TTS.
"""

import asyncio
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import pyaudio
import torch
from faster_whisper import WhisperModel
from openai import OpenAI
import onnxruntime as ort
from silero_vad import load_silero_vad, get_speech_timestamps


class VoiceAssistant:
    def __init__(self):
        self.is_active = False
        self.is_listening = False
        self.is_processing = False
        self._abort_event = threading.Event()
        
        # Paths
        self.state_dir = Path.home() / ".local/state/voice-assistant"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.state_dir / "voice-assistant.log"
        self.pid_file = self.state_dir / "voice-assistant.pid"
        self.chimes_dir = self.state_dir / "chimes"
        self.chimes_dir.mkdir(exist_ok=True)
        
        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        self.audio_format = pyaudio.paInt16
        
        # Initialize components
        self.setup_logging()
        self.setup_audio()
        self.setup_models()
        self.generate_chimes()
        
        # Signal handler
        signal.signal(signal.SIGUSR1, self.toggle_handler)
        
        # Write PID file
        with open(self.pid_file, 'w') as f:
            f.write(str(os.getpid()))
        
        # Initialize waybar status
        self.set_waybar_status("off")
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Voice Assistant starting...")
    
    def setup_audio(self):
        """Initialize audio system."""
        self.audio = pyaudio.PyAudio()
        
        # Use default PipeWire input device (handles resampling automatically)
        default_info = self.audio.get_default_input_device_info()
        self.input_device = default_info['index']
        self.logger.info(f"Using audio input device: {self.input_device} ({default_info['name']}, native rate: {int(default_info['defaultSampleRate'])})")
    
    def setup_models(self):
        """Initialize AI models."""
        try:
            # Silero VAD
            self.vad_model = load_silero_vad(onnx=False)
            self.logger.info("Silero VAD model loaded")
            
            # Faster Whisper
            self.whisper_model = WhisperModel(
                "small",  # or "base" for faster processing
                device="cuda",
                compute_type="float16"
            )
            self.logger.info("Faster Whisper model loaded")
            
            # OpenClaw Gateway (talk to Clawbook!)
            self.openai = OpenAI(
                base_url="http://127.0.0.1:18789/v1",
                api_key=os.getenv('OPENCLAW_GATEWAY_TOKEN', ''),
            )
            self.logger.info("OpenClaw gateway client initialized")
            
            # Kokoro ONNX TTS
            self.setup_tts()
            
            # Warmup: run dummy inferences to pre-compile CUDA/ONNX kernels
            self.logger.info("Warming up models...")
            dummy_audio = np.zeros(self.sample_rate, dtype=np.float32)  # 1 second of silence
            temp_path = self.state_dir / "warmup.wav"
            dummy_int16 = (dummy_audio * 32767).astype(np.int16)
            with wave.open(str(temp_path), 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(dummy_int16.tobytes())
            segments, _ = self.whisper_model.transcribe(str(temp_path))
            list(segments)  # force evaluation
            temp_path.unlink(missing_ok=True)
            self.logger.info("Whisper warmup complete")
            
            # Warmup Kokoro TTS
            if self.tts_available and self.kokoro:
                self.logger.info("Warming up Kokoro TTS...")
                import soundfile as sf
                samples, sample_rate = self.kokoro.create("Hello.", voice="af_heart", speed=1.0)
                self.logger.info("Kokoro TTS warmup complete")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            sys.exit(1)
    
    def setup_tts(self):
        """Setup Kokoro ONNX TTS model."""
        self.tts_available = False
        self.kokoro = None
        try:
            from kokoro_onnx import Kokoro
            self.kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
            self.tts_available = True
            self.logger.info("Kokoro TTS loaded")
        except ImportError:
            self.logger.warning("kokoro-onnx not installed, using espeak fallback")
        except Exception as e:
            self.logger.warning(f"Kokoro TTS setup failed (will download models on first use): {e}")
            # Try lazy init - models may need downloading
            try:
                from kokoro_onnx import Kokoro
                self.kokoro = Kokoro.from_pretrained()
                self.tts_available = True
                self.logger.info("Kokoro TTS loaded from pretrained")
            except Exception as e2:
                self.logger.warning(f"Kokoro TTS fallback also failed: {e2}, using espeak")
    
    def generate_chimes(self):
        """Generate pleasant chime WAV files."""
        # Listening chime - ascending notes
        listening_chime = self.create_chime_sequence([440, 523.25, 659.25], 0.2)
        listening_path = self.chimes_dir / "listening.wav"
        self.save_audio(listening_chime, listening_path)
        
        # Processing chime - single ding matching the first note of the intro
        processing_chime = self.create_chime_sequence([440], 0.2, fade=True)
        processing_path = self.chimes_dir / "processing.wav"
        self.save_audio(processing_chime, processing_path)
        
        # Deactivation chime - descending notes (reverse of listening)
        deactivate_chime = self.create_chime_sequence([659.25, 523.25, 440], 0.2)
        deactivate_path = self.chimes_dir / "deactivate.wav"
        self.save_audio(deactivate_chime, deactivate_path)
        
        self.logger.info("Chimes generated")
    
    def create_chime_sequence(self, frequencies, duration_per_note, fade=False):
        """Create a chime sequence with given frequencies."""
        sample_rate = 44100
        total_samples = int(sample_rate * duration_per_note * len(frequencies))
        audio_data = np.zeros(total_samples)
        
        samples_per_note = int(sample_rate * duration_per_note)
        
        for i, freq in enumerate(frequencies):
            start_idx = i * samples_per_note
            end_idx = start_idx + samples_per_note
            
            t = np.linspace(0, duration_per_note, samples_per_note)
            note = 0.3 * np.sin(2 * np.pi * freq * t)
            
            # Apply envelope
            envelope = np.exp(-2 * t)  # Exponential decay
            if fade and i == len(frequencies) - 1:
                envelope *= np.linspace(1, 0, len(envelope))
            
            audio_data[start_idx:end_idx] = note * envelope
        
        # Convert to 16-bit PCM
        audio_data = (audio_data * 32767).astype(np.int16)
        return audio_data
    
    def save_audio(self, audio_data, filepath):
        """Save audio data to WAV file."""
        with wave.open(str(filepath), 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(44100)
            wav_file.writeframes(audio_data.tobytes())
    
    def play_chime(self, chime_type):
        """Play a chime sound."""
        chime_path = self.chimes_dir / f"{chime_type}.wav"
        if chime_path.exists():
            # Use pw-play for PipeWire
            subprocess.run(['pw-play', str(chime_path)], 
                         capture_output=True, check=False)
    
    def notify(self, message):
        """Send desktop notification."""
        subprocess.run(['notify-send', 'Voice Assistant', message], 
                      capture_output=True, check=False)
    
    def set_waybar_status(self, state):
        """Update waybar status indicator. States: off, ready, listening, thinking, speaking."""
        status_file = self.state_dir / "waybar-status"
        symbols = {"off": "◯", "ready": "●", "listening": "●", "thinking": "●", "speaking": "●"}
        symbol = symbols.get(state, "")
        import json
        status_file.write_text(json.dumps({"text": symbol, "class": state}))
    
    def abort_inflight(self):
        """Abort any in-flight OpenClaw request — both client-side and server-side."""
        self._abort_event.set()
        # Send /stop to the session to kill server-side tool execution
        try:
            self.openai.chat.completions.create(
                model="openclaw:main",
                max_tokens=32,
                messages=[{"role": "user", "content": "/stop"}],
                extra_headers={"x-openclaw-agent-id": "main"},
                user="voice-assistant",
            )
            self.logger.info("Sent /stop to abort server-side processing")
        except Exception as e:
            self.logger.warning(f"Failed to send /stop: {e}")
        self.logger.info("Signalled abort for in-flight request")

    def toggle_handler(self, signum, frame):
        """Handle SIGUSR1 signal to toggle voice assistant."""
        self.is_active = not self.is_active
        
        if self.is_active:
            self.logger.info("Voice Assistant activated")
            self._abort_event.clear()
            self.notify("🎤 Voice Mode ON")
            self.set_waybar_status("ready")
            # Play activation chime in background thread (non-blocking)
            chime_path = self.chimes_dir / "listening.wav"
            if chime_path.exists():
                threading.Thread(target=lambda: subprocess.run(
                    ['pw-play', str(chime_path)], capture_output=True, check=False
                ), daemon=True).start()
        else:
            self.logger.info("Voice Assistant deactivated")
            self.notify("🎤 Voice Mode OFF")
            self.set_waybar_status("off")
            # Abort any in-flight API request
            self.abort_inflight()
            # Play descending chime in background thread (non-blocking)
            chime_path = self.chimes_dir / "deactivate.wav"
            if chime_path.exists():
                threading.Thread(target=lambda: subprocess.run(
                    ['pw-play', str(chime_path)], capture_output=True, check=False
                ), daemon=True).start()
    
    def record_audio(self, duration=10):
        """Record audio from microphone."""
        frames = []
        stream = None
        try:
            stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.input_device,
                frames_per_buffer=self.chunk_size
            )
            
            for _ in range(int(self.sample_rate / self.chunk_size * duration)):
                if not self.is_active:  # Stop if deactivated
                    break
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                frames.append(data)
        except OSError as e:
            self.logger.debug(f"Audio recording interrupted: {e}")
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except OSError:
                    pass
        
        if not frames:
            return np.zeros(self.sample_rate, dtype=np.float32)
        
        # Convert to numpy array
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        return audio_data.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
    
    def detect_speech(self, audio_data):
        """Use Silero VAD to detect speech in audio."""
        audio_tensor = torch.from_numpy(audio_data)
        speech_timestamps = get_speech_timestamps(
            audio_tensor, 
            self.vad_model,
            sampling_rate=self.sample_rate
        )
        return len(speech_timestamps) > 0
    
    def transcribe_audio(self, audio_data):
        """Transcribe audio using Faster Whisper."""
        try:
            # Save temporary audio file
            temp_path = self.state_dir / "temp_audio.wav"
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            with wave.open(str(temp_path), 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            # Transcribe
            segments, _ = self.whisper_model.transcribe(str(temp_path))
            transcription = " ".join([segment.text for segment in segments])
            
            # Clean up
            temp_path.unlink(missing_ok=True)
            
            return transcription.strip()
        
        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
            return ""
    
    def query_claude(self, text):
        """Send query to Clawbook via OpenClaw gateway. Abortable on deactivation."""
        self._abort_event.clear()
        try:
            response = self.openai.chat.completions.create(
                model="openclaw:main",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                extra_headers={
                    "x-openclaw-agent-id": "main",
                },
                user="voice-assistant",
            )
            if self._abort_event.is_set():
                self.logger.info("Request completed but assistant was deactivated — discarding response")
                return None
            return response.choices[0].message.content
        
        except Exception as e:
            if self._abort_event.is_set():
                self.logger.info("Request aborted due to deactivation")
                return None
            self.logger.error(f"OpenClaw API error: {e}")
            return "Sorry, I couldn't process that request."
    
    def speak_text(self, text):
        """Convert text to speech and play it."""
        if self.tts_available and self.kokoro:
            try:
                self.logger.info(f"Speaking with Kokoro: {text[:50]}...")
                import soundfile as sf
                samples, sample_rate = self.kokoro.create(text, voice="af_heart", speed=1.0)
                tmp_path = self.chimes_dir / "tts_output.wav"
                sf.write(str(tmp_path), samples, sample_rate)
                subprocess.run(['pw-play', str(tmp_path)], capture_output=True, check=False)
            except Exception as e:
                self.logger.error(f"Kokoro TTS failed: {e}, falling back to espeak")
                self.speak_with_espeak(text)
        else:
            self.speak_with_espeak(text)
    
    def speak_with_espeak(self, text):
        """Fallback TTS using espeak."""
        subprocess.run(['espeak', '-s', '150', '-v', 'en+f3', text], 
                      capture_output=True, check=False)
    
    def record_until_silence(self, max_duration=15, silence_timeout=1.5, pre_audio=None):
        """Record audio until silence is detected after speech, or max duration."""
        frames = []
        stream = None
        chunk_duration = 0.5  # seconds per VAD check
        chunk_samples = int(self.sample_rate * chunk_duration)
        silence_chunks = 0
        silence_limit = int(silence_timeout / chunk_duration)
        had_speech = pre_audio is not None
        
        if pre_audio is not None:
            # Include the audio that triggered VAD
            frames.append(pre_audio)
        
        try:
            stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.input_device,
                frames_per_buffer=self.chunk_size
            )
            
            max_chunks = int(max_duration / chunk_duration)
            for _ in range(max_chunks):
                if not self.is_active:
                    break
                
                # Record one chunk
                chunk_frames = []
                for _ in range(int(chunk_samples / self.chunk_size)):
                    if not self.is_active:
                        break
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    chunk_frames.append(data)
                
                if not chunk_frames:
                    break
                
                chunk_audio = np.frombuffer(b''.join(chunk_frames), dtype=np.int16)
                chunk_float = chunk_audio.astype(np.float32) / 32768.0
                frames.append(chunk_float)
                
                # Check VAD on this chunk
                if self.detect_speech(chunk_float):
                    had_speech = True
                    silence_chunks = 0
                else:
                    if had_speech:
                        silence_chunks += 1
                        if silence_chunks >= silence_limit:
                            self.logger.info("Silence detected, stopping recording")
                            break
        except OSError as e:
            self.logger.debug(f"Audio recording interrupted: {e}")
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except OSError:
                    pass
        
        if not frames:
            return np.zeros(self.sample_rate, dtype=np.float32)
        
        return np.concatenate(frames)
    
    def _read_chunk(self, stream, duration=0.5):
        """Read a chunk of audio from an open stream."""
        num_frames = int(self.sample_rate * duration)
        frames = []
        for _ in range(int(num_frames / self.chunk_size)):
            data = stream.read(self.chunk_size, exception_on_overflow=False)
            frames.append(data)
        if not frames:
            return np.zeros(int(self.sample_rate * duration), dtype=np.float32)
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        return audio_data.astype(np.float32) / 32768.0

    def _record_until_silence_stream(self, stream, max_duration=30, silence_timeout=2.0, pre_audio=None):
        """Record from an already-open stream until silence is detected."""
        frames = []
        chunk_duration = 0.5
        silence_chunks = 0
        silence_limit = int(silence_timeout / chunk_duration)
        had_speech = pre_audio is not None

        if pre_audio is not None:
            frames.append(pre_audio)

        max_chunks = int(max_duration / chunk_duration)
        for _ in range(max_chunks):
            if not self.is_active:
                break
            chunk_float = self._read_chunk(stream, chunk_duration)
            frames.append(chunk_float)

            if self.detect_speech(chunk_float):
                had_speech = True
                silence_chunks = 0
            else:
                if had_speech:
                    silence_chunks += 1
                    if silence_chunks >= silence_limit:
                        self.logger.info("Silence detected, stopping recording")
                        break

        if not frames:
            return np.zeros(self.sample_rate, dtype=np.float32)
        return np.concatenate(frames)

    async def listen_loop(self):
        """Main listening loop with persistent audio stream."""
        loop = asyncio.get_event_loop()
        stream = None

        while True:
            if not self.is_active:
                # Close stream when inactive
                if stream is not None:
                    try:
                        stream.stop_stream()
                        stream.close()
                    except OSError:
                        pass
                    stream = None
                await asyncio.sleep(0.1)
                continue

            # Open persistent stream when active
            if stream is None:
                try:
                    stream = self.audio.open(
                        format=self.audio_format,
                        channels=self.channels,
                        rate=self.sample_rate,
                        input=True,
                        input_device_index=self.input_device,
                        frames_per_buffer=self.chunk_size
                    )
                    self.logger.info("Audio stream opened")
                except OSError as e:
                    self.logger.error(f"Failed to open audio stream: {e}")
                    await asyncio.sleep(1)
                    continue

            if self.is_processing:
                # Drain audio buffer while processing so we don't get stale data,
                # but keep the stream open so there's no gap
                try:
                    await loop.run_in_executor(None, self._read_chunk, stream, 0.1)
                except OSError:
                    pass
                await asyncio.sleep(0.05)
                continue

            # Read a VAD chunk from the persistent stream
            try:
                audio_chunk = await loop.run_in_executor(None, self._read_chunk, stream, 0.5)
            except OSError as e:
                self.logger.error(f"Audio read error: {e}")
                try:
                    stream.close()
                except OSError:
                    pass
                stream = None
                continue

            if not self.is_active:
                continue

            # Check for speech
            if self.detect_speech(audio_chunk):
                self.is_listening = True
                self.set_waybar_status("listening")
                self.logger.info("Speech detected, recording until silence...")
                self.notify("🔊 Listening...")

                # Record until they stop talking, reusing the same stream
                full_audio = await loop.run_in_executor(
                    None, self._record_until_silence_stream, stream, 30, 1.5, audio_chunk
                )

                self.is_listening = False
                self.is_processing = True
                self.set_waybar_status("thinking")

                # Process the audio
                asyncio.create_task(self.process_audio(full_audio))
            
            await asyncio.sleep(0.05)
    
    async def process_audio(self, audio_data):
        """Process recorded audio through the pipeline."""
        try:
            # Transcribe
            transcription = self.transcribe_audio(audio_data)
            if not transcription:
                self.is_processing = False
                return
            
            # Ding to signal we heard you
            self.play_chime("processing")
            
            self.logger.info(f"Transcription: {transcription}")
            self.notify(f"💭 {transcription}")
            
            # Query Claude
            response = self.query_claude(transcription)
            
            # If aborted (deactivated while waiting), bail out silently
            if response is None:
                return
            
            self.logger.info(f"Response: {response}")
            self.notify(f"🧙 {response}")
            
            # Speak response (only if still active)
            if self.is_active:
                self.set_waybar_status("speaking")
                self.speak_text(response)
            
        except Exception as e:
            self.logger.error(f"Processing error: {e}")
        finally:
            self.is_processing = False
            if self.is_active:
                self.set_waybar_status("ready")
    
    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'audio'):
            self.audio.terminate()
        
        # Remove PID file
        self.pid_file.unlink(missing_ok=True)
        
        self.logger.info("Voice Assistant stopped")
    
    def run(self):
        """Main run loop."""
        try:
            self.logger.info("Voice Assistant ready - send SIGUSR1 to toggle")
            asyncio.run(self.listen_loop())
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()


def main():
    """Entry point."""
    assistant = VoiceAssistant()
    assistant.run()


if __name__ == "__main__":
    main()