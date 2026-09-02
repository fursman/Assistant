#!/usr/bin/env python3
"""
Voice Assistant installation check.

Verifies the stack the assistant actually runs today. Exits 0 when everything
the assistant needs is present; warnings (⚠) are things that degrade the
experience without stopping it.

Run it from the repo with the venv active, or via `voice-assistant-ctl test`.
"""

import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

PASS, WARN, FAIL = "✅", "⚠️ ", "❌"
results = {"pass": 0, "warn": 0, "fail": 0}


def ok(msg):
    print(f"{PASS} {msg}")
    results["pass"] += 1


def warn(msg, fix=None):
    print(f"{WARN} {msg}")
    if fix:
        print(f"   Fix: {fix}")
    results["warn"] += 1


def fail(msg, fix=None):
    print(f"{FAIL} {msg}")
    if fix:
        print(f"   Fix: {fix}")
    results["fail"] += 1


def check_import(module, description, required=True):
    try:
        importlib.import_module(module)
        ok(f"{description}")
        return True
    except ImportError as e:
        (fail if required else warn)(f"{description}: {e}",
                                     "pip install -r requirements.txt")
        return False


def check_command(name, description, required=True):
    path = shutil.which(name)
    if path:
        ok(f"{description}: {path}")
        return True
    (fail if required else warn)(f"{description}: {name} not found")
    return False


def main():
    print("🧪 Voice Assistant Installation Check")
    print("=" * 44)

    print("\n📦 Python dependencies")
    for module, desc in [
        ("numpy", "NumPy"),
        ("scipy", "SciPy (resampling)"),
        ("pyaudio", "PyAudio (capture and playback)"),
        ("soundfile", "SoundFile"),
        ("torch", "PyTorch (Silero VAD only)"),
        ("silero_vad", "Silero VAD"),
        ("onnxruntime", "ONNX Runtime (TTS + end-of-turn model)"),
        ("kokoro_onnx", "Kokoro TTS"),
        ("moonshine_voice", "Moonshine STT"),
        ("openai", "OpenAI client (local LLM backend)"),
    ]:
        check_import(module, desc)
    check_import("faster_whisper", "Faster Whisper (fallback STT)", required=False)

    try:
        import torch
        if torch.version.cuda:
            warn(f"PyTorch is the CUDA build ({torch.__version__})",
                 "the CPU build is intended here: pip install torch torchaudio "
                 "--index-url https://download.pytorch.org/whl/cpu")
        else:
            ok(f"PyTorch is the CPU build ({torch.__version__})")
    except Exception:
        pass

    try:
        import onnxruntime
        if importlib.util.find_spec("onnxruntime") and Path(
                onnxruntime.__file__).parent.joinpath("capi").exists():
            providers = onnxruntime.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                warn("onnxruntime-gpu appears to be installed",
                     "pip uninstall onnxruntime-gpu — it shares an import "
                     "namespace with onnxruntime and fails when the dGPU is "
                     "passed through to a VM")
            else:
                ok("ONNX Runtime is the CPU build")
    except Exception:
        pass

    print("\n🔧 System commands")
    check_command("pw-play", "PipeWire playback")
    check_command("notify-send", "Desktop notifications")
    check_command("espeak", "eSpeak (TTS fallback)", required=False)
    check_command("gdbus", "gdbus (closing our own notifications)", required=False)
    if not any(shutil.which(d) for d in ("swaync-client", "mako", "dunst")):
        warn("No notification daemon found (swaync / mako / dunst)")
    else:
        ok("Notification daemon present")

    print("\n📁 Files")
    here = Path(__file__).parent
    for path, desc, required in [
        (here / "voice_assistant.py", "Main script", True),
        (here / "requirements.txt", "Requirements", True),
        (here / "voice-assistant.service", "Systemd unit", True),
        (here / "kokoro-v1.0.onnx", "Kokoro model (~325 MB)", True),
        (here / "voices-v1.0.bin", "Kokoro voices (~28 MB)", True),
        (Path.home() / ".local/bin/voice-assistant", "Installed launcher", True),
        (Path.home() / ".local/bin/voice-llm", "Backend switcher", False),
        (Path.home() / ".cache/voice-assistant/smart-turn-v3.2-cpu.onnx",
         "smart-turn end-of-turn model", False),
    ]:
        if Path(path).exists():
            ok(f"{desc}")
        elif required:
            fail(f"{desc}: missing ({path})", "./setup.sh")
        else:
            warn(f"{desc}: missing ({path})", "./setup.sh")

    print("\n🎙  Audio")
    try:
        import pyaudio
        audio = pyaudio.PyAudio()
        try:
            src = audio.get_default_input_device_info()
            ok(f"Default input: {src['name']}")
        except Exception as e:
            fail(f"No default input device: {e}")
        names = {audio.get_device_info_by_index(i)["name"]
                 for i in range(audio.get_device_count())}
        if "pipewire" in names:
            ok("PortAudio sees the 'pipewire' device")
        else:
            warn("No 'pipewire' PortAudio device; playback falls back to the default")
        audio.terminate()
    except Exception as e:
        fail(f"Audio: {e}")

    print("\n🧠 LLM backend")
    env_file = Path.home() / ".config/voice-assistant/env"
    configured = "auto"
    if env_file.exists():
        m = re.search(r"^VOICE_ASSISTANT_LLM_BACKEND=(\S+)",
                      env_file.read_text(), re.M)
        configured = m.group(1) if m else "auto"
        ok(f"Config: {env_file} (backend={configured})")
    else:
        warn(f"No {env_file}; defaults apply (backend=auto)", "./setup.sh")

    gpu = None
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            rows = [r.split(",") for r in out.stdout.strip().splitlines() if r.strip()]
            if rows:
                gpu = max(((r[0].strip(), int(r[1])) for r in rows), key=lambda g: g[1])
                if gpu[1] >= 15000:
                    ok(f"GPU: {gpu[0]}, {gpu[1]} MiB — big enough for the local model")
                else:
                    ok(f"GPU: {gpu[0]}, {gpu[1]} MiB — too small for Qwen3.8-27B "
                       f"(needs ~15.3 GB); the assistant will use Claude")
        except Exception:
            pass
    if gpu is None:
        ok("No NVIDIA GPU visible — the assistant will use the Claude CLI")

    url = os.getenv("VOICE_ASSISTANT_LOCAL_URL", "http://127.0.0.1:8081/v1")
    health = re.sub(r"/v1/?$", "", url.rstrip("/")) + "/health"
    try:
        with urllib.request.urlopen(health, timeout=2) as r:
            ok(f"Local model server: serving ({health})")
    except urllib.error.HTTPError as e:
        if e.code == 503:
            ok("Local model server: loading (503 until the weights are in)")
        else:
            warn(f"Local model server: HTTP {e.code}")
    except Exception:
        if configured == "local":
            fail("Local model server is not responding but the backend is forced to 'local'",
                 "voice-llm qwen, or voice-llm auto")
        else:
            ok("Local model server: not running (Claude will answer)")

    if check_command("claude", "Claude Code CLI", required=(configured != "local")):
        settings = Path.home() / ".claude/settings.json"
        try:
            if json.loads(settings.read_text()).get("skipDangerousModePermissionPrompt"):
                ok("Claude skipDangerousModePermissionPrompt: enabled")
            else:
                warn("Claude skipDangerousModePermissionPrompt is not set",
                     'add \'"skipDangerousModePermissionPrompt": true\' to '
                     "~/.claude/settings.json — the assistant runs claude "
                     "non-interactively")
        except FileNotFoundError:
            warn("~/.claude/settings.json not found",
                 "echo '{\"skipDangerousModePermissionPrompt\": true}' > ~/.claude/settings.json")
        except Exception as e:
            warn(f"~/.claude/settings.json could not be parsed: {e}")

    print("\n🔑 Permissions")
    try:
        if subprocess.run(["sudo", "-n", "true"], capture_output=True,
                          timeout=5).returncode == 0:
            ok("Passwordless sudo: available")
        else:
            user = os.getenv("USER", "user")
            warn("Passwordless sudo not available — system commands will fail",
                 f"echo '{user} ALL=(ALL) NOPASSWD: ALL' | sudo tee /etc/sudoers.d/{user}")
    except Exception as e:
        warn(f"sudo check failed: {e}")

    print("\n🔄 Service")
    try:
        out = subprocess.run(["systemctl", "--user", "is-active", "voice-assistant.service"],
                             capture_output=True, text=True, timeout=10)
        state = out.stdout.strip()
        if state == "active":
            ok("voice-assistant.service: running")
        else:
            warn(f"voice-assistant.service: {state}",
                 "systemctl --user start voice-assistant.service")
    except Exception as e:
        warn(f"Service check failed: {e}")

    print("\n" + "=" * 44)
    print(f"📊 {results['pass']} passed, {results['warn']} warnings, "
          f"{results['fail']} failures")
    if results["fail"]:
        print("\nSomething the assistant needs is missing. Re-run ./setup.sh, "
              "then check the notes above.")
        return 1
    if results["warn"]:
        print("\nReady to use. The warnings above are optional or degraded paths.")
    else:
        print("\n🎉 Everything checks out.")
    print("\nStart:  systemctl --user start voice-assistant.service")
    print("Toggle: press SUPER alone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
