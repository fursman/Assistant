#!/bin/bash
set -e

# Hyprland Voice Assistant Setup Script
# This script sets up everything needed for the voice assistant

echo "🎤 Setting up Hyprland Voice Assistant..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [[ ! -f "voice_assistant.py" ]]; then
    log_error "Please run this script from the voice-assistant directory"
    exit 1
fi

# Check system requirements
log_info "Checking system requirements..."

# Check for Python 3.13+
if ! python3 --version | grep -q "Python 3.1[3-9]"; then
    log_error "Python 3.13+ is required"
    exit 1
fi

# Check for NVIDIA driver
if ! nvidia-smi &>/dev/null; then
    log_error "NVIDIA driver not found or not working"
    exit 1
fi

# Check for CUDA (nvidia-smi is sufficient, we don't need nvcc)
if ! nvidia-smi &>/dev/null; then
    log_error "NVIDIA GPU not found (nvidia-smi failed)"
    exit 1
fi

# Check for PipeWire
if ! systemctl --user is-active pipewire &>/dev/null; then
    log_warning "PipeWire is not running. Starting it..."
    systemctl --user start pipewire
fi

# Check for required system packages
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

# Create directories
log_info "Creating directories..."
mkdir -p ~/.local/bin
mkdir -p ~/.local/state/voice-assistant
mkdir -p ~/.config/systemd/user

# Create Python virtual environment
log_info "Creating Python virtual environment..."
if [[ -d ".venv" ]]; then
    log_warning "Virtual environment already exists, removing..."
    rm -rf .venv
fi

python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
log_info "Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA support first
log_info "Installing PyTorch with CUDA 12.8 support..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install other requirements
log_info "Installing Python dependencies..."
pip install -r requirements.txt

# Install project in development mode
log_info "Installing voice-assistant package..."
pip install -e .

# Download models
log_info "Downloading AI models..."

# Silero VAD - this will be downloaded automatically on first use
log_info "Silero VAD will be downloaded on first use"

# Faster Whisper models
log_info "Pre-downloading Whisper models..."
python3 -c "
from faster_whisper import WhisperModel
import logging
logging.basicConfig(level=logging.INFO)
print('Downloading Whisper small model...')
model = WhisperModel('small', device='cuda', compute_type='float16')
print('Whisper model downloaded and cached')
"

# Test CUDA availability
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

# Create executable script
log_info "Installing voice-assistant executable..."
cat > ~/.local/bin/voice-assistant << 'EOF'
#!/bin/bash
cd /home/user/voice-assistant
source .venv/bin/activate
exec python3 voice_assistant.py "$@"
EOF

chmod +x ~/.local/bin/voice-assistant

# Install systemd service
log_info "Installing systemd user service..."
cp voice-assistant.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable voice-assistant.service

# Voice assistant talks to OpenClaw gateway (Clawbook!) on localhost:18789
log_info "Voice assistant will connect to OpenClaw gateway at http://127.0.0.1:18789"

# Create Hyprland configuration snippet
log_info "Creating Hyprland configuration snippet..."
cat > hyprland-voice-assistant.conf << 'EOF'
# Hyprland Voice Assistant Configuration
# Add these lines to your ~/.config/hypr/hyprland.conf

# Auto-start voice assistant
exec-once = systemctl --user start voice-assistant.service

# Bind SUPER key solo press to toggle voice assistant
# Note: This requires the voice-assistant daemon to be running
bindr = SUPER, SUPER_L, exec, pkill -USR1 voice-assistant
EOF

# Test installation
log_info "Testing installation..."
if ~/.local/bin/voice-assistant --help &>/dev/null || [[ $? -eq 130 ]]; then
    log_success "Installation completed successfully!"
else
    log_error "Installation test failed"
    exit 1
fi

# Summary
log_success "🎉 Voice Assistant setup complete!"
echo
echo "Next steps:"
echo "1. Set your Anthropic API key:"
echo "   export ANTHROPIC_API_KEY='your-api-key-here'"
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
echo
echo "Logs can be found at: ~/.local/state/voice-assistant/voice-assistant.log"
echo "Service status: systemctl --user status voice-assistant.service"
echo
log_info "Enjoy your new voice assistant! 🎤✨"