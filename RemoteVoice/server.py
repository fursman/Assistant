#!/usr/bin/env python3
"""
RemoteVoice Server — WebSocket gateway for headless voice assistant clients.

Runs on ClawBox (GPU machine). Thin clients send recorded audio over WebSocket;
this server runs the full pipeline: Whisper STT → Claude CLI → Kokoro TTS,
streaming back events and audio as they happen.

WebSocket protocol:
  Client → Server:
    Binary frame: WAV audio of recorded speech

  Server → Client:
    {"type": "stt", "text": "..."}              — transcription result
    {"type": "thinking", "text": "..."}         — Claude thinking update
    {"type": "tool", "name": "Bash", "detail": "..."} — tool use notification
    {"type": "tts", "text": "sentence text"}    — sentence about to play
    Binary frame                                — WAV audio for preceding tts sentence
    {"type": "done"}                            — query complete, ready for next

Usage:
  python3 server.py [--port 8767] [--host 0.0.0.0]
"""

import asyncio
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np
import requests
from faster_whisper import WhisperModel

try:
    import websockets
    from websockets.asyncio.server import serve
except ImportError:
    print("ERROR: websockets not installed. Run: pip install websockets")
    sys.exit(1)

# ── Config ───────────────────────────────────────────────────────────────

HOST = os.getenv("REMOTEVOICE_HOST", "0.0.0.0")
PORT = int(os.getenv("REMOTEVOICE_PORT", "8767"))

WHISPER_MODEL = "small.en"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE = "int8"

TTS_HOST = os.getenv("REMOTEVOICE_TTS_HOST", "http://localhost:8766")
TTS_VOICE = "af_heart"
TTS_SPEED = 1.0

CLAUDE_MODEL = os.getenv("VOICE_ASSISTANT_MODEL", "sonnet")
CLAUDE_VOICE_PROMPT = (
    "You are a voice assistant integrated into a Linux desktop. "
    "The user speaks to you and hears your responses via text-to-speech. "
    "Keep responses concise and conversational — avoid code blocks, markdown formatting, "
    "and long lists unless specifically asked. Prefer natural spoken language. "
    "You have full access to the system and can run commands, read/write files, search the web, and more. "
    "Be helpful, proactive, and efficient. When the user asks you to do something on their system, just do it. "
    "Give brief confirmations rather than lengthy explanations."
)

SESSION_DIR = os.path.expanduser("~/.local/state/remote-voice")
os.makedirs(SESSION_DIR, exist_ok=True)

# Sentence boundary regex for streaming TTS
_SENTENCE_END_RE = re.compile(r'(?<=[.!?])\s')

# Regex to strip markdown formatting before TTS
_MD_STRIP = re.compile(
    r"\*\*(.+?)\*\*"
    r"|\*(.+?)\*"
    r"|__(.+?)__"
    r"|_(.+?)_"
    r"|`([^`]+)`"
    r"|```[\s\S]*?```"
    r"|\[([^\]]+)\]\([^)]+\)"
    r"|^#{1,6}\s+"
    r"|^[-*]\s+"
    r"|^>\s+"
    , re.MULTILINE
)

# Number words for speech normalization
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
         "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]

def _num_to_words(n: int) -> str:
    if n < 0:
        return "negative " + _num_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " and " + _num_to_words(rest))
    if n < 1000000:
        rest = n % 1000
        return _num_to_words(n // 1000) + " thousand" + ("" if rest == 0 else " " + _num_to_words(rest))
    if n < 1000000000:
        rest = n % 1000000
        return _num_to_words(n // 1000000) + " million" + ("" if rest == 0 else " " + _num_to_words(rest))
    if n < 1000000000000:
        rest = n % 1000000000
        return _num_to_words(n // 1000000000) + " billion" + ("" if rest == 0 else " " + _num_to_words(rest))
    return str(n)

_ABBREVS = [
    (re.compile(r"\bvs\."), "versus"),
    (re.compile(r"\bvs\b"), "versus"),
    (re.compile(r"\betc\."), "et cetera"),
    (re.compile(r"\be\.g\.\s?"), "for example, "),
    (re.compile(r"\bi\.e\.\s?"), "that is, "),
    (re.compile(r"\bw/o\b"), "without"),
    (re.compile(r"\bw/\b"), "with"),
    (re.compile(r"\bGPUs\b"), "G P Us"),
    (re.compile(r"\bGPU\b"), "G P U"),
    (re.compile(r"\bCPUs\b"), "C P Us"),
    (re.compile(r"\bCPU\b"), "C P U"),
    (re.compile(r"\bVRAM\b"), "V ram"),
    (re.compile(r"\bAPIs\b"), "A P Is"),
    (re.compile(r"\bAPI\b"), "A P I"),
    (re.compile(r"\bLLMs\b"), "L L Ms"),
    (re.compile(r"\bLLM\b"), "L L M"),
    (re.compile(r"\bTTS\b"), "T T S"),
    (re.compile(r"\bSTT\b"), "S T T"),
    (re.compile(r"\bUI\b"), "U I"),
    (re.compile(r"\bURLs\b"), "U R Ls"),
    (re.compile(r"\bURL\b"), "U R L"),
    (re.compile(r"\bGB\b"), "gigabytes"),
    (re.compile(r"\bMB\b"), "megabytes"),
    (re.compile(r"\bTB\b"), "terabytes"),
    (re.compile(r"\bSSH\b"), "S S H"),
    (re.compile(r"\bNVLink\b"), "N V Link"),
    (re.compile(r"\bRTX\b"), "R T X"),
    (re.compile(r"\bRAM\b"), "ram"),
    (re.compile(r"\bEDA\b"), "E D A"),
    (re.compile(r"\bIMO\b"), "in my opinion"),
    (re.compile(r"\bSOTA\b"), "state of the art"),
    (re.compile(r"\bINT4\b"), "int four"),
    (re.compile(r"\bINT8\b"), "int eight"),
    (re.compile(r"\bFP16\b"), "F P sixteen"),
    (re.compile(r"\bFP32\b"), "F P thirty-two"),
]


def _strip_markdown(text):
    text = re.sub(r"```[\s\S]*?```", "code block", text)
    def _pick(m):
        for g in m.groups():
            if g is not None:
                return g
        return ""
    return _MD_STRIP.sub(_pick, text).strip()


def _prepare_for_speech(text):
    """Strip markdown and normalize text for natural TTS pronunciation."""
    text = re.sub(r"```[\s\S]*?```", "code block", text)
    def _pick(m):
        for g in m.groups():
            if g is not None:
                return g
        return ""
    text = _MD_STRIP.sub(_pick, text).strip()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[\U0001f300-\U0001f9ff\U00002600-\U000027bf\U0000fe00-\U0000feff]", "", text)
    def _currency(m):
        sign = m.group(1) or ""
        whole = m.group(2).replace(",", "")
        cents = m.group(4) or ""
        prefix = "negative " if sign == "-" else ""
        w = int(whole) if whole else 0
        result = prefix + _num_to_words(w) + (" dollar" if w == 1 else " dollars")
        if cents:
            c = int(cents)
            if c > 0:
                result += " and " + _num_to_words(c) + (" cent" if c == 1 else " cents")
        return result
    text = re.sub(r"(-?)\$([0-9,]+)(\.(\d{1,2}))?", _currency, text)
    def _percent(m):
        whole = m.group(1)
        dec = m.group(3)
        result = _num_to_words(int(whole))
        if dec:
            result += " point " + " ".join(_ONES[int(d)] for d in dec)
        return result + " percent"
    text = re.sub(r"(\d+)(\.(\d+))?%", _percent, text)
    def _multiplier(m):
        whole = m.group(1)
        dec = m.group(3)
        result = _num_to_words(int(whole))
        if dec:
            result += " point " + " ".join(_ONES[int(d)] for d in dec)
        return result + " x"
    text = re.sub(r"(\d+)(\.(\d+))x\b", _multiplier, text)
    def _decimal(m):
        whole = m.group(1).replace(",", "")
        dec = m.group(2)
        result = _num_to_words(int(whole))
        result += " point " + " ".join(_ONES[int(d)] for d in dec)
        return result
    text = re.sub(r"(\d[\d,]*)\.([\d]+)", _decimal, text)
    def _big_num(m):
        n = int(m.group(0).replace(",", ""))
        return _num_to_words(n)
    text = re.sub(r"\d{1,3}(?:,\d{3})+", _big_num, text)
    text = re.sub(r"(\d)([A-Z]{2,})\b", r"\1 \2", text)
    def _plain_num(m):
        n = int(m.group(0))
        if n <= 999999:
            return _num_to_words(n)
        return m.group(0)
    text = re.sub(r"\b\d{1,6}\b", _plain_num, text)
    for pattern, replacement in _ABBREVS:
        text = pattern.sub(replacement, text)
    text = re.sub(r"(\w)/(\w)", r"\1 or \2", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Whisper STT ──────────────────────────────────────────────────────────

print(f"Loading Whisper '{WHISPER_MODEL}' on {WHISPER_DEVICE}...")
t0 = time.time()
whisper_model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
print(f"Whisper ready ({time.time() - t0:.1f}s)")

# Warmup
_dummy = np.zeros(16000, dtype=np.float32)
segments, _ = whisper_model.transcribe(_dummy)
list(segments)
print("Whisper warmup complete")


def transcribe_wav(wav_bytes):
    """Transcribe WAV audio bytes to text. Returns (text, duration_secs, stt_time_secs)."""
    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        tmp = f.name
    try:
        segments, info = whisper_model.transcribe(tmp, beam_size=5, language="en")
        text = " ".join(s.text for s in segments).strip()
        stt_time = time.time() - t0
        return text, info.duration, stt_time
    finally:
        os.unlink(tmp)


# ── TTS via Kokoro HTTP ──────────────────────────────────────────────────

def tts_generate(text):
    """Generate WAV audio bytes for text via Kokoro TTS server."""
    try:
        resp = requests.post(
            f"{TTS_HOST}/v1/audio/speech",
            json={"input": text, "voice": TTS_VOICE, "speed": TTS_SPEED},
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.content
        print(f"  TTS error: {resp.status_code}")
        return None
    except Exception as e:
        print(f"  TTS error: {e}")
        return None


# ── Session management ───────────────────────────────────────────────────

def _session_file(client_id):
    return os.path.join(SESSION_DIR, f"session_{client_id}.txt")


def load_session_id(client_id):
    path = _session_file(client_id)
    try:
        if os.path.exists(path):
            with open(path) as f:
                sid = f.read().strip()
            if sid:
                return sid
    except Exception:
        pass
    return None


def save_session_id(client_id, sid):
    with open(_session_file(client_id), "w") as f:
        f.write(sid)


# ── Claude CLI streaming ────────────────────────────────────────────────

async def run_claude_pipeline(ws, text, client_id):
    """Run Claude CLI, stream events to client, generate TTS and send audio."""
    loop = asyncio.get_event_loop()

    session_id = load_session_id(client_id)

    cmd = [
        "claude", "-p", text,
        "--output-format", "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--dangerously-skip-permissions",
        "--model", CLAUDE_MODEL,
        "--append-system-prompt", CLAUDE_VOICE_PROMPT,
    ]
    if session_id:
        cmd.extend(["--resume", session_id])
    else:
        cmd.extend(["--continue"])

    print(f"  Claude cmd: {' '.join(cmd[:6])}...")

    env = os.environ.copy()
    env.pop("CLAUDECODE", None)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env=env,
        cwd=os.path.expanduser("~"),
    )

    # State for streaming sentence extraction
    assistant_text = ""
    spoken_pos = 0
    thinking_text = ""
    thinking_shown_len = 0
    last_thinking_notify = 0.0
    last_tool_notify = 0.0
    current_tool_idx = None
    current_tool_name = ""
    current_tool_input = ""

    # TTS pipeline: sentence queue → generate → send
    tts_queue = asyncio.Queue()
    tts_done = asyncio.Event()

    async def tts_worker():
        while True:
            item = await tts_queue.get()
            if item is None:
                break
            sentence = item
            wav = await loop.run_in_executor(None, tts_generate, sentence)
            if wav:
                try:
                    await ws.send(json.dumps({"type": "tts", "text": sentence}))
                    await ws.send(wav)
                except Exception:
                    break
        tts_done.set()

    tts_task = asyncio.create_task(tts_worker())

    def flush_sentences(final=False):
        nonlocal assistant_text, spoken_pos
        remaining = assistant_text[spoken_pos:]
        if not remaining:
            return

        while True:
            match = re.search(r'(?<=[.!?])\s', remaining)
            if not match:
                break
            end = match.end()
            sentence = _prepare_for_speech(remaining[:end].strip())
            remaining = remaining[end:]
            spoken_pos += end
            if sentence:
                tts_queue.put_nowait(sentence)

        if final and remaining.strip():
            sentence = _prepare_for_speech(remaining.strip())
            spoken_pos += len(remaining)
            if sentence:
                tts_queue.put_nowait(sentence)

    try:
        async for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type")

            if etype == "system":
                sid = event.get("session_id")
                if sid:
                    save_session_id(client_id, sid)
                    print(f"  Session: {sid[:12]}...")

            elif etype == "stream_event":
                se = event.get("event", {})
                se_type = se.get("type")

                if se_type == "content_block_start":
                    cb = se.get("content_block", {})
                    idx = se.get("index")
                    if cb.get("type") == "tool_use":
                        current_tool_idx = idx
                        current_tool_name = cb.get("name", "")
                        current_tool_input = ""

                elif se_type == "content_block_delta":
                    delta = se.get("delta", {})
                    dt = delta.get("type")

                    if dt == "thinking_delta":
                        thinking_text += delta.get("thinking", "")
                        # Rate-limited thinking notifications
                        now = time.time()
                        if now - last_thinking_notify >= 2.0:
                            new = thinking_text[thinking_shown_len:]
                            # Find last sentence boundary
                            last_b = -1
                            for m in re.finditer(r'(?<!\d)[.!?](?:\s|$)', new):
                                last_b = m.end()
                            if last_b > 0:
                                chunk = new[:last_b].strip()
                                if len(chunk) >= 20:
                                    last_thinking_notify = now
                                    thinking_shown_len += last_b
                                    try:
                                        await ws.send(json.dumps({
                                            "type": "thinking", "text": chunk
                                        }))
                                    except Exception:
                                        pass

                    elif dt == "text_delta":
                        assistant_text += delta.get("text", "")
                        flush_sentences(final=False)

                    elif dt == "input_json_delta":
                        if se.get("index") == current_tool_idx:
                            current_tool_input += delta.get("partial_json", "")

                elif se_type == "content_block_stop":
                    idx = se.get("index")
                    if idx == current_tool_idx and current_tool_name:
                        # Send tool notification
                        detail = ""
                        try:
                            args = json.loads(current_tool_input) if current_tool_input else {}
                        except json.JSONDecodeError:
                            args = {}
                        if current_tool_name == "Bash":
                            cmd_str = args.get("command", "")
                            if cmd_str:
                                detail = cmd_str.split("\n")[0].strip()[:120]
                        elif current_tool_name in ("Read", "Edit", "Write"):
                            detail = args.get("file_path", "")
                        elif current_tool_name == "WebSearch":
                            detail = args.get("query", "")

                        now = time.time()
                        if now - last_tool_notify >= 2.0:
                            last_tool_notify = now
                            try:
                                await ws.send(json.dumps({
                                    "type": "tool",
                                    "name": current_tool_name,
                                    "detail": detail,
                                }))
                            except Exception:
                                pass
                        current_tool_idx = None
                        current_tool_name = ""
                        current_tool_input = ""

            elif etype == "result":
                if event.get("is_error"):
                    print(f"  Claude error: {event.get('result', '')}")
                else:
                    cost = event.get("total_cost_usd", 0)
                    turns = event.get("num_turns", 1)
                    print(f"  Claude done (${cost:.4f}, {turns} turns)")

    except Exception as e:
        print(f"  Claude stream error: {e}")

    # Reap process
    try:
        await asyncio.wait_for(proc.wait(), timeout=5)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()

    # Flush remaining text
    if not assistant_text.strip():
        assistant_text = "Done."
    flush_sentences(final=True)

    # Signal TTS worker to stop, wait for it to drain
    await tts_queue.put(None)
    await asyncio.wait_for(tts_done.wait(), timeout=120)
    tts_task.cancel()

    # Send remaining thinking as final notification
    if thinking_text:
        remaining_think = thinking_text[thinking_shown_len:].strip()
        if remaining_think:
            try:
                await ws.send(json.dumps({"type": "thinking", "text": remaining_think}))
            except Exception:
                pass

    # Done
    try:
        await ws.send(json.dumps({"type": "done"}))
    except Exception:
        pass


# ── WebSocket handler ────────────────────────────────────────────────────

# Hallucination patterns to filter
_HALLUCINATION_PATTERNS = {
    "thank you for watching", "thanks for watching", "subscribe",
    "like and subscribe", "please subscribe", "see you next time",
    "bye bye", "thank you", "thanks for listening",
    "you", "the end", "so",
}


async def handle_client(ws):
    """Handle a single WebSocket client connection."""
    remote = ws.remote_address
    client_id = f"{remote[0]}_{remote[1]}" if remote else "unknown"
    print(f"Client connected: {client_id}")

    try:
        async for message in ws:
            if isinstance(message, bytes):
                # Binary frame = WAV audio to process
                print(f"  Received {len(message)} bytes of audio from {client_id}")

                text, duration, stt_time = await asyncio.get_event_loop().run_in_executor(
                    None, transcribe_wav, message
                )
                print(f"  STT: \"{text}\" ({duration:.1f}s audio, {stt_time:.1f}s)")

                # Filter noise/hallucinations
                words = text.split()
                lower = text.lower().strip().rstrip(".")
                if not text or len(words) < 3 or lower in _HALLUCINATION_PATTERNS:
                    print(f"  Filtered: \"{text}\"")
                    await ws.send(json.dumps({"type": "done"}))
                    continue

                # Send transcription to client
                await ws.send(json.dumps({"type": "stt", "text": text}))

                # Run full pipeline
                await run_claude_pipeline(ws, text, client_id)

            elif isinstance(message, str):
                # JSON control messages
                try:
                    msg = json.loads(message)
                except json.JSONDecodeError:
                    continue

                if msg.get("type") == "new_session":
                    path = _session_file(client_id)
                    try:
                        os.unlink(path)
                    except FileNotFoundError:
                        pass
                    await ws.send(json.dumps({"type": "session_cleared"}))
                    print(f"  Session cleared for {client_id}")

                elif msg.get("type") == "ping":
                    await ws.send(json.dumps({"type": "pong"}))

    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        print(f"  Client error: {e}")
    finally:
        print(f"Client disconnected: {client_id}")


# ── Main ─────────────────────────────────────────────────────────────────

async def main():
    # Verify TTS server is reachable
    try:
        resp = requests.get(f"{TTS_HOST}/health", timeout=5)
        print(f"TTS server: {resp.json()}")
    except Exception as e:
        print(f"WARNING: TTS server not reachable at {TTS_HOST}: {e}")

    # Verify Claude CLI
    claude_path = subprocess.run(
        ["which", "claude"], capture_output=True, text=True
    ).stdout.strip()
    if claude_path:
        print(f"Claude CLI: {claude_path}")
    else:
        print("ERROR: 'claude' CLI not found in PATH")
        sys.exit(1)

    print(f"\nRemoteVoice server starting on ws://{HOST}:{PORT}")
    print(f"  STT: Whisper {WHISPER_MODEL} ({WHISPER_DEVICE} {WHISPER_COMPUTE})")
    print(f"  LLM: Claude CLI ({CLAUDE_MODEL})")
    print(f"  TTS: Kokoro via {TTS_HOST}")
    print()

    async with serve(handle_client, HOST, PORT, max_size=10_000_000):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
