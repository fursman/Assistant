#!/bin/bash
set -e

# Hyprland Voice Assistant setup.
#
# Installs the CPU pipeline (VAD, STT, end-of-turn model, TTS) on any machine,
# and additionally builds the local LLM stack when this machine has an NVIDIA
# GPU with enough VRAM to hold Qwen3.8-27B. Without one, the assistant answers
# with the `claude` CLI, and everything else is identical.
#
# Idempotent: re-running skips what is already done and never re-downloads.
#
#   ./setup.sh                 full install
#   ./setup.sh --no-llm        skip the local model even if the GPU qualifies
#   ./setup.sh --rebuild-llama rebuild llama.cpp
#   ./setup.sh --reinstall     recreate the Python venv from scratch
#   ./setup.sh --sleep-units   only (re)install the suspend/resume units

NO_LLM=0; REBUILD_LLAMA=0; REINSTALL=0; SLEEP_UNITS_ONLY=0
for a in "$@"; do
    case "$a" in
        --no-llm) NO_LLM=1 ;;
        --rebuild-llama) REBUILD_LLAMA=1 ;;
        --reinstall) REINSTALL=1 ;;
        --sleep-units) SLEEP_UNITS_ONLY=1 ;;
        -h|--help) sed -n '3,20p' "$0"; exit 0 ;;
        *) echo "unknown option: $a (try --help)"; exit 2 ;;
    esac
done

echo "🎤 Setting up Hyprland Voice Assistant..."

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

if [[ ! -f "voice_assistant.py" ]]; then
    log_error "Please run this script from the voice-assistant directory"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$HOME/.config/voice-assistant/env"
UNIT_DIR="$HOME/.config/systemd/user"

# ── Local-LLM constants (used by install_local_llm and the summary) ──────
LLM_MIN_VRAM_MIB=15000        # "16 GB" cards report 16303-16384; the model needs ~15.3 GB
LLAMA_DIR="$HOME/llama.cpp"
MODEL_REPO="unsloth/Qwen3.8-27B-GGUF"
MODEL_FILE="Qwen3.8-27B-UD-IQ4_XS.gguf"
MODEL_DIR="$HOME/models/$MODEL_REPO"
MODEL_MIN_BYTES=14000000000   # the real file is 14,252,845,984 bytes

install_sleep_units() {
    # Only needed when the driver preserves VRAM across suspend: with a 15 GB
    # model resident that is an 84 s suspend and a black screen on resume.
    if ! grep -qs 'PreserveVideoMemoryAllocations: 1' /proc/driver/nvidia/params 2>/dev/null; then
        log_info "Driver does not preserve VRAM across suspend — sleep/wake units not needed"
        return 0
    fi
    log_info "Installing qwen38 sleep/wake units (unload the model around suspend)"
    sudo install -m 0755 contrib/qwen38-sleep /usr/local/sbin/qwen38-sleep
    sudo install -m 0644 contrib/qwen38-sleep.service contrib/qwen38-wake.service \
        /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable qwen38-sleep.service qwen38-wake.service
    log_success "Suspend/resume units installed"
}

if [[ "$SLEEP_UNITS_ONLY" == 1 ]]; then
    install_sleep_units
    exit 0
fi

# ── System Requirements ──────────────────────────────────────────────────

log_info "Checking system requirements..."

if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
    log_error "Python 3.11+ is required (found $(python3 --version))"
    exit 1
fi

if ! systemctl --user is-active pipewire &>/dev/null; then
    log_warning "PipeWire is not running. Starting it..."
    systemctl --user start pipewire || log_warning "could not start pipewire"
fi

# libnotify-bin and a notification daemon are not optional extras: the
# assistant's entire visual channel is notify-send, and a missing binary used
# to surface as an unexplained crash in the toggle handler.
REQUIRED_PACKAGES=(
    "python3-venv"
    "python3-dev"
    "portaudio19-dev"
    "libportaudio2"
    "espeak"
    "pipewire-bin"
    "pipewire-pulse"
    "libnotify-bin"
    "pkg-config"
    "build-essential"
    "curl"
)

MISSING_PACKAGES=()
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    dpkg -s "$pkg" &>/dev/null || MISSING_PACKAGES+=("$pkg")
done

if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
    log_warning "Installing missing system packages: ${MISSING_PACKAGES[*]}"
    sudo apt-get update
    sudo apt-get install -y "${MISSING_PACKAGES[@]}"
fi

# A notification daemon has to be running for anything to be visible. Any
# implementation will do; swaync is what the Hyprland setup uses.
if ! command -v swaync-client &>/dev/null && ! command -v mako &>/dev/null \
     && ! command -v dunst &>/dev/null; then
    log_warning "No notification daemon found (swaync / mako / dunst) — the"
    log_warning "  assistant will still speak, but nothing will be shown on screen."
fi

# ── Directories ──────────────────────────────────────────────────────────

log_info "Creating directories..."
mkdir -p ~/.local/bin ~/.local/state/voice-assistant ~/.cache/voice-assistant "$UNIT_DIR"

# ── Python Virtual Environment ───────────────────────────────────────────

if [[ -x ".venv/bin/python" && "$REINSTALL" != 1 ]]; then
    log_info "Reusing existing virtual environment (--reinstall to rebuild)"
else
    [[ -d ".venv" ]] && { log_warning "Recreating virtual environment"; rm -rf .venv; }
    log_info "Creating Python virtual environment..."
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

log_info "Upgrading pip..."
pip install --upgrade pip wheel setuptools

# ── Install Dependencies ─────────────────────────────────────────────────

# CPU-only, deliberately. torch is here purely for Silero VAD, so the CUDA
# build would be ~3.7GB of dead weight (it took the venv from 1.5GB to 5.2GB),
# and it is exactly the dependency that broke this app while the dGPU was
# passed through to the Windows VM.
log_info "Installing CPU-only PyTorch (for Silero VAD)..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

log_info "Installing Python dependencies..."
pip install -r requirements.txt

log_info "Installing voice-assistant package..."
pip install -e . --no-deps

# The `claude` CLI is not an optional extra: it is one of the two LLM backends
# and the only one on a machine with no big GPU. A fresh install has no reason
# to already have it.
if command -v claude &>/dev/null; then
    log_info "claude CLI: $(command -v claude)"
else
    log_info "Installing claude CLI (native installer)..."
    if curl -fsSL https://claude.ai/install.sh | bash; then
        export PATH="$HOME/.local/bin:$PATH"
        command -v claude &>/dev/null \
            && log_success "claude CLI installed: $(claude --version 2>&1 | head -1)" \
            || log_warning "claude installed but not on PATH — add ~/.local/bin to PATH"
    else
        log_error "claude CLI install failed — install it manually from https://claude.ai/install.sh"
    fi
fi

# ── Runtime configuration (never clobbered once it exists) ───────────────

mkdir -p "$(dirname "$ENV_FILE")"
if [[ -s "$ENV_FILE" ]]; then
    log_info "Keeping existing $ENV_FILE"
else
    log_info "Writing $ENV_FILE (backend=auto)"
    cat > "$ENV_FILE" << 'ENVEOF'
# Voice assistant runtime configuration (read by voice-assistant.service).
# Switch backends with:  voice-llm auto | voice-llm claude | voice-llm qwen
#
# auto   = local llama-server when this machine has a >= 16 GB NVIDIA GPU and
#          qwen38.service is installed, else the Claude Code CLI
# claude = Claude Code CLI (tools, filesystem, web; needs network)
# local  = llama-server on this machine (Qwen3.8-27B + run_shell tool, offline)
VOICE_ASSISTANT_LLM_BACKEND=auto
# Answer with Claude for a single query while the local model is loading or
# down, instead of telling the user the model did not answer.
VOICE_ASSISTANT_LLM_FALLBACK=1

# --- local backend (ignored when the backend resolves to claude) ---
VOICE_ASSISTANT_LOCAL_URL=http://127.0.0.1:8081/v1
VOICE_ASSISTANT_LOCAL_MODEL=qwen3.8-27b
VOICE_ASSISTANT_LOCAL_MAX_TOKENS=512
VOICE_ASSISTANT_LOCAL_HISTORY_TURNS=12
# Qwen3.8 reasons by default; for voice that is pure latency, so it is off.
VOICE_ASSISTANT_LOCAL_THINK=0

# --- tool access for the local model ---
# 1 = the model can run arbitrary shell commands (parity with the Claude
# backend, which runs with --dangerously-skip-permissions). 0 = answer only.
VOICE_ASSISTANT_LOCAL_TOOLS=1
VOICE_ASSISTANT_LOCAL_TOOL_TIMEOUT=30
VOICE_ASSISTANT_LOCAL_MAX_TOOL_ITERS=5

# --- speech recognition ---
# medium_streaming, not small: on REAL speech (LibriSpeech) it cuts word error
# rate from 11.5% to 7.8%, and on command audio from 9.2% to 5.7%. Because it
# transcribes while you talk, the extra size costs nothing after you stop.
VOICE_ASSISTANT_MOONSHINE_MODEL=medium_streaming

# --- end of turn ---
# smart-turn v3 decides when you have finished speaking, instead of waiting a
# fixed timeout. Set to 0 to go back to the silence timeout alone.
VOICE_ASSISTANT_SMART_TURN=1
# "<silence seconds>:<probability needed to end the turn>". The bar comes down
# as the pause lengthens, so a mid-sentence breath does not end your turn but a
# pause that keeps going does. Raise the early numbers if it still cuts you off;
# a false "unfinished" only costs the wait to the next checkpoint.
VOICE_ASSISTANT_SMART_TURN_CHECKPOINTS=0.35:0.90,0.70:0.75,1.10:0.60,1.60:0.50
# Ends the turn regardless, when smart-turn keeps saying "unfinished".
VOICE_ASSISTANT_SILENCE_TIMEOUT=2.5
ENVEOF
fi

# ── Models ───────────────────────────────────────────────────────────────

# Pre-download exactly the STT model the env file selects. Getting this wrong
# means the first run downloads ~291 MB inside the service with no progress
# indicator, or fails outright on a machine that is offline.
STT_ARCH=$(grep -E '^VOICE_ASSISTANT_MOONSHINE_MODEL=' "$ENV_FILE" 2>/dev/null | tail -1 | cut -d= -f2)
STT_ARCH=${STT_ARCH:-medium_streaming}
log_info "Pre-downloading Moonshine STT model: $STT_ARCH"
if ! python3 - "$STT_ARCH" << 'PYEOF'
import sys
import moonshine_voice as mv
name = sys.argv[1].strip().upper().replace("-", "_")
arch = getattr(mv.ModelArch, name, None)
if arch is None:
    sys.exit(f"{name}: not a Moonshine architecture; "
             f"try one of {[a.name.lower() for a in mv.ModelArch]}")
path, _ = mv.get_model_for_language("en", arch)
print(f"cached at {path}")
PYEOF
then
    log_error "Moonshine $STT_ARCH could not be fetched"
    log_error "  (BASE_STREAMING is in the enum but has no published English model)"
    log_error "  The first run will retry, or fall back to Whisper."
fi

# smart-turn v3.2: 8.7 MB, decides when the user has stopped talking.
SMART_TURN_FILE="$HOME/.cache/voice-assistant/smart-turn-v3.2-cpu.onnx"
if [[ -s "$SMART_TURN_FILE" ]]; then
    log_info "smart-turn model already present"
else
    log_info "Downloading smart-turn v3.2 end-of-turn model (8.7 MB)..."
    if curl -fL --retry 3 -o "$SMART_TURN_FILE.part" \
        "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx"
    then
        mv "$SMART_TURN_FILE.part" "$SMART_TURN_FILE"
        log_success "smart-turn model downloaded"
    else
        rm -f "$SMART_TURN_FILE.part"
        log_warning "smart-turn download failed — the assistant will fall back to a"
        log_warning "  fixed silence timeout (slower, but it works). It retries at startup."
    fi
fi

# Kokoro TTS weights: ~340 MB of binary that are NOT in this repo, so a fresh
# clone has no voice at all until they are fetched.
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

# ── Local LLM (needs an NVIDIA GPU with >= 16 GB) ────────────────────────

install_local_llm() {
    if ! command -v nvidia-smi &>/dev/null; then
        if lspci 2>/dev/null | grep -qi 'nvidia'; then
            log_warning "NVIDIA GPU present but no driver (nvidia-smi missing)."
            log_warning "  Install it first (sudo ubuntu-drivers install; reboot; enrol the"
            log_warning "  MOK if Secure Boot asks), then re-run ./setup.sh"
        else
            log_info "No NVIDIA GPU — the assistant will answer with the claude CLI"
        fi
        return 0
    fi

    local row name total used cc
    row=$(nvidia-smi --query-gpu=name,memory.total,memory.used,compute_cap \
            --format=csv,noheader,nounits 2>/dev/null | sort -t, -k2 -nr | head -1) || row=""
    if [[ -z "$row" ]]; then
        log_warning "nvidia-smi found no device (driver not loaded, or the card is"
        log_warning "  passed through to a VM) — skipping the local model"
        return 0
    fi
    IFS=, read -r name total used cc <<< "$row"
    name="${name#"${name%%[![:space:]]*}"}"
    total=${total// /}; used=${used// /}; cc=${cc// /}
    if (( total < LLM_MIN_VRAM_MIB )); then
        log_info "$name has ${total} MiB VRAM (< ${LLM_MIN_VRAM_MIB})."
        log_info "  Qwen3.8-27B UD-IQ4_XS needs ~15.3 GB; using the claude CLI instead."
        return 0
    fi
    log_success "$name: ${total} MiB VRAM, compute capability $cc — installing the local model"

    # --- CUDA toolkit and a host compiler nvcc will accept ---
    # Ubuntu 26.04 ships nvidia-cuda-toolkit 12.4. nvcc 12.4 refuses gcc > 13
    # (crt/host_config.h) while the distro default is gcc 15, so pull
    # nvidia-cuda-toolkit-gcc, which depends on the matching g++-13.
    if ! command -v nvcc &>/dev/null || ! command -v g++-13 &>/dev/null; then
        log_info "Installing CUDA toolkit + g++-13 ..."
        sudo apt-get install -y nvidia-cuda-toolkit-gcc libcurl4-openssl-dev cmake git \
            || { log_error "CUDA toolkit install failed"; return 1; }
    fi
    local nvcc_ver arch
    nvcc_ver=$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')
    arch=${cc/./}                       # 8.6 -> 86, 12.0 -> 120
    log_info "nvcc $nvcc_ver, host compiler $(g++-13 --version | head -1), target sm_$arch"
    # Blackwell (cc 12.x) needs CUDA >= 12.8, which Ubuntu's package is not.
    if (( arch >= 120 )) && [[ "$(printf '%s\n12.8\n' "$nvcc_ver" | sort -V | head -1)" != "12.8" ]]; then
        log_error "compute capability $cc needs CUDA >= 12.8, but this is $nvcc_ver."
        log_error "  Install cuda-toolkit from NVIDIA's apt repo, then ./setup.sh --rebuild-llama"
        return 1
    fi

    # --- llama.cpp with CUDA ---
    if [[ -x "$LLAMA_DIR/build/bin/llama-server" && "$REBUILD_LLAMA" != 1 ]]; then
        log_info "llama-server already built ($("$LLAMA_DIR/build/bin/llama-server" --version 2>&1 | head -1))"
        log_info "  use --rebuild-llama to rebuild"
    else
        if [[ -d "$LLAMA_DIR/.git" ]]; then
            log_info "Updating $LLAMA_DIR"
            git -C "$LLAMA_DIR" pull --ff-only || log_warning "git pull failed — building what is there"
        else
            log_info "Cloning llama.cpp"
            git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
        fi
        # CMAKE_CUDA_HOST_COMPILER is passed explicitly: without it CMake omits
        # -ccbin and nvcc picks whichever g++ the PATH shim finds first.
        cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
            -DCMAKE_BUILD_TYPE=Release \
            -DGGML_CUDA=ON -DGGML_NATIVE=ON \
            -DCMAKE_CUDA_ARCHITECTURES="$arch" \
            -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
            -DLLAMA_CURL=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
            || { log_error "cmake configure failed"; return 1; }
        # -j8 rather than nproc: this class of laptop throttles to ~1.1 GHz at
        # 100 C with 16 jobs, and builds slower than with 8.
        log_info "Building llama-server (10-30 minutes) ..."
        cmake --build "$LLAMA_DIR/build" --config Release -j"${LLAMA_JOBS:-8}" --target llama-server \
            || { log_error "llama-server build failed"; return 1; }
        log_success "$("$LLAMA_DIR/build/bin/llama-server" --version 2>&1 | head -1)"
    fi

    # --- model weights ---
    local size avail
    size=$(stat -c %s "$MODEL_DIR/$MODEL_FILE" 2>/dev/null || echo 0)
    if (( size >= MODEL_MIN_BYTES )); then
        log_info "$MODEL_FILE already present ($((size / 1073741824)) GiB)"
    else
        avail=$(df --output=avail -B1 "$HOME" | tail -1)
        if (( avail < 16000000000 )); then
            log_error "Need ~15 GB free under $HOME, have $((avail / 1073741824)) GiB — skipping"
            return 1
        fi
        log_info "Downloading $MODEL_REPO/$MODEL_FILE (13.3 GiB, resumable) ..."
        # `hf` is huggingface_hub 1.x's CLI (huggingface-cli is gone). A
        # complete file is a no-op and a partial one resumes, so re-running
        # setup.sh after an interrupted download is the recovery path.
        if ! hf download "$MODEL_REPO" "$MODEL_FILE" --local-dir "$MODEL_DIR"; then
            log_error "Download failed — re-run ./setup.sh to resume"
            return 1
        fi
        size=$(stat -c %s "$MODEL_DIR/$MODEL_FILE" 2>/dev/null || echo 0)
        (( size >= MODEL_MIN_BYTES )) || { log_error "$MODEL_FILE is only $size bytes"; return 1; }
        log_success "Model downloaded"
    fi

    # --- unit + VRAM guard ---
    install -m 0755 contrib/qwen38-gpu-ok "$HOME/.local/bin/qwen38-gpu-ok"
    install -m 0644 contrib/qwen38.service "$UNIT_DIR/qwen38.service"
    systemctl --user daemon-reload
    systemctl --user enable qwen38.service &>/dev/null
    log_info "qwen38.service installed and enabled"

    install_sleep_units || log_warning "sleep/wake units not installed (sudo needed)"

    if systemctl --user is-active qwen38.service &>/dev/null; then
        log_info "qwen38.service already running"
    else
        log_info "Starting qwen38.service (model load: ~5 s warm, ~35 s cold)"
        systemctl --user start qwen38.service \
            || log_warning "start failed: journalctl --user -u qwen38 -n 40"
    fi
    return 0
}

if [[ "$NO_LLM" == 1 ]]; then
    log_info "--no-llm: skipping the local model (the backend resolves to claude)"
elif install_local_llm; then
    :
else
    log_warning "Local model install incomplete — the assistant uses claude until"
    log_warning "  ./setup.sh is run again"
fi

# ── Executables ──────────────────────────────────────────────────────────

log_info "Installing voice-assistant executable..."
cat > ~/.local/bin/voice-assistant << EXECEOF
#!/bin/bash
cd "$SCRIPT_DIR"
source .venv/bin/activate
exec python3 voice_assistant.py "\$@"
EXECEOF
chmod +x ~/.local/bin/voice-assistant

install -m 0755 voice-llm ~/.local/bin/voice-llm
install -m 0755 voice-assistant-ctl ~/.local/bin/voice-assistant-ctl

# ── Systemd Service ──────────────────────────────────────────────────────

log_info "Installing systemd user service..."
cp voice-assistant.service "$UNIT_DIR/"
systemctl --user daemon-reload
systemctl --user enable voice-assistant.service

# ── Hyprland Configuration ───────────────────────────────────────────────

log_info "Creating Hyprland configuration snippet..."
cat > hyprland-voice-assistant.conf << 'EOF'
# Hyprland Voice Assistant Configuration
# Add these lines to your ~/.config/hypr/hyprland.conf

# Auto-start voice assistant
exec-once = systemctl --user start voice-assistant.service

# SUPER alone toggles voice mode; SUPER+SHIFT+V starts a new conversation.
# These signal the pid file rather than pkill'ing a name: the process is
# `python3 voice_assistant.py`, so `pkill -USR1 voice-assistant` matches nothing.
bindr = SUPER, SUPER_L, exec, kill -USR1 $(cat ~/.local/state/voice-assistant/voice-assistant.pid)
bind = SUPER SHIFT, V, exec, kill -USR2 $(cat ~/.local/state/voice-assistant/voice-assistant.pid)
EOF

# ── Summary ──────────────────────────────────────────────────────────────

log_success "🎉 Voice Assistant setup complete!"
echo
echo "Backend: $(cd "$SCRIPT_DIR" && ./voice-llm status 2>/dev/null | grep -E '^Resolved' || echo 'run: voice-llm status')"
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
echo "Verify:  ./test_installation.py"
echo "Backend: voice-llm status"
echo "Logs:    journalctl --user -u voice-assistant -f"
