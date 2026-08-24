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

# No GPU detection: the pipeline is CPU-only by design.
#
# Moonshine's streaming STT returns ~0.03s after you stop speaking and Kokoro
# TTS is a small non-autoregressive ONNX model, so a GPU buys nothing here --
# and the one remaining network call (the `claude` CLI) is an API request that
# no local hardware speeds up. Staying on the CPU also means the assistant keeps
# working while the dGPU is passed through to a VM.
log_info "Installing CPU pipeline (Moonshine STT + Kokoro TTS + claude CLI)"

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

# ── Install Dependencies ─────────────────────────────────────────────────

# CPU-only, deliberately. torch is here purely for Silero VAD, so the CUDA
# build would be ~3.7GB of dead weight (it took the venv from 1.5GB to 5.2GB).
log_info "Installing CPU-only PyTorch (for Silero VAD)..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

log_info "Installing Python dependencies..."
pip install -r requirements.txt

# Install project
log_info "Installing voice-assistant package..."
pip install -e .

# The `claude` CLI is a hard dependency, not an optional extra -- it IS the
# LLM half of the pipeline, and the assistant's preflight fails without it.
# A fresh install has no reason to already have it, so install it here rather
# than letting the first run die on a missing binary.
if command -v claude &>/dev/null; then
    log_info "claude CLI: $(command -v claude)"
else
    log_info "Installing claude CLI (native installer)..."
    if curl -fsSL https://claude.ai/install.sh | bash; then
        # The installer drops a symlink in ~/.local/bin, which may not be on
        # PATH yet in this shell.
        export PATH="$HOME/.local/bin:$PATH"
        command -v claude &>/dev/null \
            && log_success "claude CLI installed: $(claude --version 2>&1 | head -1)" \
            || log_warning "claude installed but not on PATH — add ~/.local/bin to PATH"
    else
        log_error "claude CLI install failed — the assistant cannot answer without it"
        log_error "  install manually: https://claude.ai/install.sh"
    fi
fi

# Pre-download the STT model so the first run is not a surprise download.
log_info "Pre-downloading Moonshine STT model..."
python3 -c "
import moonshine_voice as mv
mv.get_model_for_language('en', mv.ModelArch.SMALL_STREAMING)
print('Moonshine streaming model cached')
"

# Kokoro TTS weights. These are ~340MB of binary and are NOT in this repo, so
# a fresh clone has no voice at all until they are fetched -- download them
# rather than leaving the first run to fail into the espeak fallback.
KOKORO_RELEASE="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
for f in kokoro-v1.0.onnx voices-v1.0.bin; do
    if [[ -s "$f" ]]; then
        log_info "$f already present"
    else
        log_info "Downloading $f ..."
        if ! curl -fL --retry 3 -o "$f.part" "$KOKORO_RELEASE/$f"; then
            rm -f "$f.part"
            log_error "Failed to download $f — TTS will fall back to espeak"
            log_error "  fetch manually from $KOKORO_RELEASE/$f"
        else
            mv "$f.part" "$f"      # only rename once complete, so a killed
        fi                          # download never looks like a good file
    fi
done

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

cp voice-assistant.service ~/.config/systemd/user/

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

log_success "🎉 Voice Assistant setup complete!"
echo
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
echo
echo "Logs: journalctl --user -u voice-assistant -f"
echo "Status: systemctl --user status voice-assistant"
