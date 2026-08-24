#!/bin/bash
set -e

# Hyprland Voice Assistant Setup Script
# Auto-detects GPU: local mode with full models, or remote mode connecting to a server

echo "🎤 Setting up Hyprland Voice Assistant..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if we're in the right directory
if [[ ! -f "voice_assistant.py" ]]; then
    log_error "Please run this script from the voice-assistant directory"
    exit 1
fi

# ── GPU Detection ────────────────────────────────────────────────────────

INSTALL_MODE="local"
REMOTE_SERVER=""

if nvidia-smi &>/dev/null; then
    log_success "NVIDIA GPU detected — installing in LOCAL mode (full pipeline)"
else
    log_warning "No GPU detected — installing in REMOTE mode (thin client)"
    INSTALL_MODE="remote"
    echo
    echo "Remote mode offloads STT, Claude, and TTS to a server with a GPU."
    echo "You need a RemoteVoice server running (see RemoteVoice/server.py)."
    echo
    read -p "Enter voice server address (e.g. clawbox.local): " REMOTE_SERVER
    if [[ -z "$REMOTE_SERVER" ]]; then
        log_error "Server address is required for remote mode"
        exit 1
    fi
    log_info "Remote server: $REMOTE_SERVER"
fi

# ── System Requirements ──────────────────────────────────────────────────

log_info "Checking system requirements..."

# Check for Python 3.13+
if ! python3 --version | grep -q "Python 3.1[3-9]"; then
    log_error "Python 3.13+ is required"
    exit 1
fi

# Check for PipeWire
if ! systemctl --user is-active pipewire &>/dev/null; then
    log_warning "PipeWire is not running. Starting it..."
    systemctl --user start pipewire
fi

# Required system packages
log_info "Checking for required system packages..."
REQUIRED_PACKAGES=(
    "python3-venv"
    "python3-dev"
    "portaudio19-dev"
    "espeak"
    "pipewire-bin"
    "libportaudio2"
    "pkg-config"
)

MISSING_PACKAGES=()
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if ! dpkg -l | grep -q "^ii  $pkg "; then
        MISSING_PACKAGES+=("$pkg")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
    log_warning "Installing missing system packages: ${MISSING_PACKAGES[*]}"
    sudo apt update
    sudo apt install -y "${MISSING_PACKAGES[@]}"
fi

# ── Directories ──────────────────────────────────────────────────────────

log_info "Creating directories..."
mkdir -p ~/.local/bin
mkdir -p ~/.local/state/voice-assistant
mkdir -p ~/.config/systemd/user

# ── Python Virtual Environment ───────────────────────────────────────────

log_info "Creating Python virtual environment..."
if [[ -d ".venv" ]]; then
    log_warning "Virtual environment already exists, removing..."
    rm -rf .venv
fi

python3 -m venv .venv
source .venv/bin/activate

log_info "Upgrading pip..."
pip install --upgrade pip wheel setuptools

# ── Install Dependencies (mode-dependent) ────────────────────────────────

if [[ "$INSTALL_MODE" == "local" ]]; then
    # LOCAL MODE: full GPU stack
    log_info "Installing PyTorch with CUDA 12.8 support..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

    log_info "Installing Python dependencies (full)..."
    pip install -r requirements.txt

    # Install project
    log_info "Installing voice-assistant package..."
    pip install -e .

    # Download and cache Whisper model
    log_info "Pre-downloading Whisper models..."
    python3 -c "
from faster_whisper import WhisperModel
print('Downloading Whisper small model...')
model = WhisperModel('small', device='cuda', compute_type='float16')
print('Whisper model downloaded and cached')
"

    # Test CUDA
    log_info "Testing CUDA availability..."
    python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name()}')
    print(f'CUDA version: {torch.version.cuda}')
else:
    print('WARNING: CUDA not available for PyTorch')
"
else
    # REMOTE MODE: lightweight deps only (CPU torch for Silero VAD)
    log_info "Installing CPU-only PyTorch (for Silero VAD)..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

    log_info "Installing Python dependencies (remote mode)..."
    pip install numpy pyaudio silero-vad websockets

    # Install project
    log_info "Installing voice-assistant package..."
    pip install -e .

    log_info "Skipping Whisper/Kokoro (handled by remote server)"
fi

# ── Executable Script ────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log_info "Installing voice-assistant executable..."
cat > ~/.local/bin/voice-assistant << EXECEOF
#!/bin/bash
cd "$SCRIPT_DIR"
source .venv/bin/activate
exec python3 voice_assistant.py "\$@"
EXECEOF

chmod +x ~/.local/bin/voice-assistant

# ── Systemd Service ──────────────────────────────────────────────────────

log_info "Installing systemd user service..."

if [[ "$INSTALL_MODE" == "remote" ]]; then
    # Generate service file with remote env vars
    cat > ~/.config/systemd/user/voice-assistant.service << SVCEOF
[Unit]
Description=Hyprland Voice Assistant (Remote Mode)
Documentation=https://github.com/fursman/Assistant
After=pipewire.service
Wants=pipewire.service

[Service]
Type=simple
ExecStart=%h/.local/bin/voice-assistant
Restart=on-failure
RestartSec=5
Environment="PATH=%h/.local/bin:%h/.npm-global/bin:/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONPATH=%h/voice-assistant/.venv/lib/python3.13/site-packages"
Environment="WAYLAND_DISPLAY=wayland-1"
Environment="XDG_RUNTIME_DIR=/run/user/1000"
Environment="VOICE_REMOTE=$REMOTE_SERVER"
WorkingDirectory=$SCRIPT_DIR

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=voice-assistant

[Install]
WantedBy=default.target
SVCEOF
    log_info "Service configured for remote mode (VOICE_REMOTE=$REMOTE_SERVER)"
else
    cp voice-assistant.service ~/.config/systemd/user/
fi

systemctl --user daemon-reload
systemctl --user enable voice-assistant.service

# ── Hyprland Configuration ───────────────────────────────────────────────

log_info "Creating Hyprland configuration snippet..."
cat > hyprland-voice-assistant.conf << 'EOF'
# Hyprland Voice Assistant Configuration
# Add these lines to your ~/.config/hypr/hyprland.conf

# Auto-start voice assistant
exec-once = systemctl --user start voice-assistant.service

# Bind SUPER key solo press to toggle voice assistant
bindr = SUPER, SUPER_L, exec, pkill -USR1 voice-assistant
EOF

# ── Summary ──────────────────────────────────────────────────────────────

log_success "🎉 Voice Assistant setup complete! (${INSTALL_MODE} mode)"
echo
if [[ "$INSTALL_MODE" == "local" ]]; then
    echo "Next steps:"
    echo "1. Add the Hyprland configuration:"
    echo "   cat hyprland-voice-assistant.conf >> ~/.config/hypr/hyprland.conf"
    echo
    echo "2. Reload Hyprland config or restart Hyprland"
    echo
    echo "3. Start the service:"
    echo "   systemctl --user start voice-assistant.service"
    echo
    echo "4. Toggle voice mode by pressing the SUPER key alone"
else
    echo "Next steps:"
    echo "1. Ensure RemoteVoice server is running on $REMOTE_SERVER:"
    echo "   ssh $REMOTE_SERVER 'systemctl status remote-voice'"
    echo
    echo "2. Add the Hyprland configuration:"
    echo "   cat hyprland-voice-assistant.conf >> ~/.config/hypr/hyprland.conf"
    echo
    echo "3. Reload Hyprland config or restart Hyprland"
    echo
    echo "4. Start the service:"
    echo "   systemctl --user start voice-assistant.service"
    echo
    echo "5. Toggle voice mode by pressing the SUPER key alone"
fi
echo
echo "Logs: journalctl --user -u voice-assistant -f"
echo "Status: systemctl --user status voice-assistant"
