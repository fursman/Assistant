#!/usr/bin/env python3
"""
Voice Assistant Installation Test Script
Run this to verify that all components are working correctly.
"""

import sys
import os
import subprocess
import importlib
from pathlib import Path

def test_import(module_name, description):
    """Test if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✅ {description}: OK")
        return True
    except ImportError as e:
        print(f"❌ {description}: FAILED - {e}")
        return False

def test_command(command, description):
    """Test if a command exists and can be run."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ {description}: OK")
            return True
        else:
            print(f"❌ {description}: FAILED - {result.stderr}")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"❌ {description}: FAILED - {e}")
        return False

def test_file_exists(filepath, description):
    """Test if a file exists."""
    if Path(filepath).exists():
        print(f"✅ {description}: OK")
        return True
    else:
        print(f"❌ {description}: MISSING")
        return False

def test_cuda():
    """Test CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name()
            print(f"✅ CUDA: OK - {device_name}")
            return True
        else:
            print("❌ CUDA: Not available in PyTorch")
            return False
    except ImportError:
        print("❌ CUDA: PyTorch not installed")
        return False

def test_audio():
    """Test audio system."""
    try:
        import pyaudio
        audio = pyaudio.PyAudio()
        device_count = audio.get_device_count()
        audio.terminate()
        print(f"✅ Audio: OK - {device_count} devices found")
        return True
    except Exception as e:
        print(f"❌ Audio: FAILED - {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Voice Assistant Installation Test")
    print("=" * 40)
    
    tests_passed = 0
    total_tests = 0
    
    # Test Python modules
    print("\n📦 Python Dependencies:")
    modules = [
        ("numpy", "NumPy"),
        ("torch", "PyTorch"),
        ("faster_whisper", "Faster Whisper"),
        ("anthropic", "Anthropic API"),
        ("silero_vad", "Silero VAD"),
        ("pyaudio", "PyAudio"),
        ("onnxruntime", "ONNX Runtime"),
    ]
    
    for module, description in modules:
        if test_import(module, description):
            tests_passed += 1
        total_tests += 1
    
    # Test system commands
    print("\n🔧 System Commands:")
    commands = [
        (["nvidia-smi"], "NVIDIA Driver"),
        (["nvcc", "--version"], "CUDA Compiler"),
        (["espeak", "--version"], "eSpeak TTS"),
        (["pw-play", "--help"], "PipeWire pw-play"),
        (["notify-send", "--version"], "Desktop Notifications"),
    ]
    
    for command, description in commands:
        if test_command(command, description):
            tests_passed += 1
        total_tests += 1
    
    # Test file structure
    print("\n📁 File Structure:")
    files = [
        ("voice_assistant.py", "Main Script"),
        ("requirements.txt", "Requirements"),
        ("pyproject.toml", "Project Config"),
        ("voice-assistant.service", "Systemd Service"),
        ("setup.sh", "Setup Script"),
        (Path.home() / ".local/bin/voice-assistant", "Installed Executable"),
    ]
    
    for filepath, description in files:
        if test_file_exists(filepath, description):
            tests_passed += 1
        total_tests += 1
    
    # Test voice assistant runtime requirements
    print("\n🔑 Runtime Requirements:")

    # Claude Code --dangerously-skip-permissions
    settings_file = Path.home() / ".claude/settings.json"
    if settings_file.exists():
        import json as _json
        try:
            _settings = _json.loads(settings_file.read_text())
            if _settings.get("skipDangerousModePermissionPrompt"):
                print("✅ Claude skipDangerousModePermissionPrompt: enabled")
                tests_passed += 1
            else:
                print("❌ Claude skipDangerousModePermissionPrompt: not set")
                print("   Fix: Add '\"skipDangerousModePermissionPrompt\": true' to ~/.claude/settings.json")
        except Exception:
            print("❌ Claude settings.json: could not parse")
    else:
        print("❌ Claude settings.json: not found")
        print("   Fix: echo '{\"skipDangerousModePermissionPrompt\": true}' > ~/.claude/settings.json")
    total_tests += 1

    # Passwordless sudo
    try:
        sudo_result = subprocess.run(
            ["sudo", "-n", "true"], capture_output=True, text=True, timeout=5
        )
        if sudo_result.returncode == 0:
            print("✅ Passwordless sudo: available")
            tests_passed += 1
        else:
            user = os.getenv("USER", "user")
            print("❌ Passwordless sudo: not available")
            print(f"   Fix: echo '{user} ALL=(ALL) NOPASSWD: ALL' | sudo tee /etc/sudoers.d/{user}")
    except Exception as e:
        print(f"❌ Passwordless sudo: FAILED - {e}")
    total_tests += 1

    # Test special components
    print("\n🧠 AI Components:")
    if test_cuda():
        tests_passed += 1
    total_tests += 1
    
    if test_audio():
        tests_passed += 1
    total_tests += 1
    
    # Test API key
    print("\n🔐 Configuration:")
    if os.getenv('ANTHROPIC_API_KEY'):
        print("✅ Anthropic API Key: Set")
        tests_passed += 1
    else:
        print("❌ Anthropic API Key: Not set")
    total_tests += 1
    
    # Test service
    print("\n🔄 Service Status:")
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "voice-assistant.service"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            status = result.stdout.strip()
            if status == "active":
                print("✅ Voice Assistant Service: Running")
                tests_passed += 1
            else:
                print(f"⚠️ Voice Assistant Service: {status}")
        else:
            print("❌ Voice Assistant Service: Not found")
    except Exception as e:
        print(f"❌ Service Check: FAILED - {e}")
    total_tests += 1
    
    # Summary
    print("\n" + "=" * 40)
    print(f"📊 Test Summary: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Voice Assistant is ready to use.")
        print("\nTo start:")
        print("1. systemctl --user start voice-assistant.service")
        print("2. Press SUPER key to toggle voice mode")
        return 0
    else:
        print("⚠️  Some tests failed. Check the setup and try again.")
        print("\nFor help:")
        print("1. Check the README.md file")
        print("2. Run the setup script again: ./setup.sh")
        print("3. Check logs: journalctl --user -u voice-assistant.service")
        return 1

if __name__ == "__main__":
    sys.exit(main())