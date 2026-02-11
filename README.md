# 🎤 Hyprland Voice Assistant

A local voice assistant for Linux that listens for speech, transcribes it, queries an LLM, and speaks the response — all running on your own hardware.

**Stack:** Silero VAD → Faster Whisper STT → OpenClaw LLM Gateway → Kokoro TTS

## How It Works

1. **SUPER key** toggles voice mode on/off
2. **Silero VAD** continuously monitors your mic for speech
3. When you speak, **Faster Whisper** transcribes your audio on the GPU
4. The transcription is sent to your local **OpenClaw** gateway (which routes to your AI agent)
5. **Kokoro ONNX TTS** speaks the response through PipeWire
6. Waybar shows the current state (off / ready / listening / thinking / speaking)

## Features

- **Fully local** — no cloud APIs required (STT and TTS run on-device)
- **CUDA accelerated** — Whisper and Kokoro run on your NVIDIA GPU
- **VAD-based listening** — only records when you're actually speaking
- **OpenClaw integration** — talks to your AI agent with full tool access
- **Systemd service** — runs as a user service, starts with your session
- **Chime feedback** — ascending chime on activate, ding on processing, descending on deactivate
- **Waybar integration** — status indicator in your bar
- **Abortable** — press SUPER again to cancel mid-response

## Requirements

- Linux with Hyprland (or any Wayland compositor)
- NVIDIA GPU with CUDA support
- PipeWire audio
- Python 3.13+
- [OpenClaw](https://github.com/openclaw/openclaw) gateway running locally

## Quick Start

```bash
# Clone the repo
git clone https://github.com/fursman/Assistant.git
cd Assistant

# Run setup (installs deps, downloads models, configures systemd)
chmod +x setup.sh
./setup.sh

# Set your OpenClaw gateway token
export OPENCLAW_GATEWAY_TOKEN='your-token-here'

# Start the service
systemctl --user start voice-assistant.service

# Add to Hyprland config for SUPER key toggle:
# bindr = SUPER, SUPER_L, exec, pkill -USR1 voice-assistant
```

## Control Script

```bash
./voice-assistant-ctl status    # Check if running
./voice-assistant-ctl start     # Start the service
./voice-assistant-ctl stop      # Stop the service
./voice-assistant-ctl toggle    # Toggle voice mode
./voice-assistant-ctl logs      # View recent logs
./voice-assistant-ctl logs -f   # Follow logs live
```

## Architecture

```
┌──────────┐    ┌───────────┐    ┌──────────┐    ┌─────────┐    ┌───────────┐
│ Silero   │───▶│  Faster   │───▶│ OpenClaw │───▶│ Kokoro  │───▶│ PipeWire  │
│ VAD      │    │  Whisper  │    │ Gateway  │    │ TTS     │    │ Speaker   │
│ (detect) │    │ (transcr) │    │ (LLM)   │    │ (speak) │    │ (output)  │
└──────────┘    └───────────┘    └──────────┘    └─────────┘    └───────────┘
     ▲                                                               │
     │              PipeWire Microphone Input                        │
     └──────────────────────────────────────────────────────────────┘
```

## Models

| Component | Model | Runs On |
|-----------|-------|---------|
| VAD | Silero VAD v4 | CPU |
| STT | Faster Whisper (small) | CUDA |
| LLM | Via OpenClaw gateway | Configurable |
| TTS | Kokoro v1.0 ONNX | CUDA/CPU |

## Files

| File | Description |
|------|-------------|
| `voice_assistant.py` | Main application |
| `setup.sh` | One-shot installation script |
| `voice-assistant-ctl` | Control script (start/stop/status/toggle) |
| `voice-assistant.service` | Systemd user service unit |
| `test_installation.py` | Installation verification tests |
| `requirements.txt` | Python dependencies |
| `pyproject.toml` | Package metadata |

## License

MIT
