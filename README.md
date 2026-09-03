# 🎤 Hyprland Voice Assistant

A voice assistant for Linux that listens, transcribes, thinks and speaks — with
speech recognition, end-of-turn detection and synthesis all running on your own
CPU, and the language model running either on your own GPU or through Claude.

```
mic ─▶ capture thread ─▶ Silero VAD ─▶ smart-turn v3 (has the user finished?)
                      └▶ Moonshine STT (transcribes while you talk)
                                    │
                                    ▼
                      Claude Code CLI  ·or·  local Qwen3.8-27B
                                    │
                                    ▼
                      Kokoro TTS ─▶ one persistent PipeWire stream
```

Tap **SUPER** to start listening. Speak. It answers — out loud, and it can act
on your machine.

## How it works

1. **SUPER** toggles voice mode. Waybar shows the state; chimes mark the edges.
   **SUPER+M** swaps between the local model and Claude; **SUPER+SHIFT+V**
   starts a new conversation. You can also just type: `assistant <question>`
   joins the same conversation from a terminal.
2. **Silero VAD** watches the mic. Speech has to persist for ~96 ms before a
   turn starts, so a cough does not trigger one.
3. **Moonshine** transcribes *while you are still speaking*, so when you stop
   only the tail is left to decode — about 0.4 s for a short command.
4. **smart-turn v3** decides you have finished, from the sound of the sentence
   rather than a stopwatch. It hears the difference between "turn the lights
   off" and "turn the lights off and…".
5. The transcript goes to **Claude Code** or a **local Qwen3.8-27B**, whichever
   this machine is set up for.
6. **Kokoro** synthesises the reply clause by clause and plays it through a
   single PipeWire stream, so speech starts as soon as the first few words
   exist and there are no gaps between sentences.

## Features

- **Semantic end of turn** — an 8.7 MB audio classifier, not a silence timer.
  ~0.4 s of dead air after you stop instead of ~2.7 s, and it holds through a
  mid-sentence pause instead of cutting you off.
- **Two backends, chosen automatically** — the local model when this machine
  can hold it, Claude otherwise, with a per-query fallback either way.
- **Speech recognition and synthesis on the CPU** — they keep working while the
  dGPU is passed through to a VM.
- **Streaming everywhere** — transcription during speech, synthesis during
  generation, playback during synthesis.
- **Runs commands** — both backends can act on the system, and both are told
  the input is a transcript that might be wrong.
- **Abortable** — tap SUPER again; speech stops mid-word without a click and
  the model is interrupted rather than killed.
- **Audible turn-taking** — the same chime that arms voice mode plays again
  when the microphone goes live after a reply, so you know when it is your turn.
- **Systemd user service**, **waybar module**, **desktop notifications** for
  what it heard, what it is thinking and what it is doing.

## Requirements

- Linux with Wayland (Hyprland assumed for the key binding; any compositor works)
- PipeWire, and a notification daemon (swaync, mako or dunst)
- Python 3.11+
- **Either** the [`claude` CLI](https://claude.ai/install.sh) (installed for you
  by `setup.sh`) **or** an NVIDIA GPU with ≥ 16 GB of VRAM for the local model
- ~1.5 GB for the Python venv and ~700 MB of speech models; a further ~14 GB if
  you install the local LLM

## Quick start

```bash
git clone https://github.com/fursman/Assistant.git ~/voice-assistant
cd ~/voice-assistant
./setup.sh

# Add the key bindings, then start it
cat hyprland-voice-assistant.conf >> ~/.config/hypr/hyprland.conf
systemctl --user start voice-assistant.service
```

`setup.sh` installs the CPU pipeline on any machine and, if it finds an NVIDIA
GPU with enough VRAM, additionally builds llama.cpp with CUDA, downloads the
model and installs the model server. It is idempotent — re-run it whenever
something is missing. `./setup.sh --no-llm` skips the local model.

Verify with `./test_installation.py`.

## Control

```bash
voice-assistant-ctl status       # service, voice state, backend, session
voice-assistant-ctl toggle       # same as tapping SUPER
voice-assistant-ctl new-session  # start a fresh conversation
voice-assistant-ctl logs -f      # follow
voice-assistant-ctl test         # installation checks

voice-llm status                 # what backend is configured, and what it resolves to
voice-llm auto                   # local model if this machine can run it, else Claude
voice-llm qwen                   # force the local model (starts the server)
voice-llm claude                 # force Claude and stop the server (frees the GPU)
```

Say **"new conversation"** (or "start over", "forget everything") to clear the
context by voice.

### Asking from a terminal

`assistant` talks to the running service over a control socket, so a typed
question lands in the **same conversation** as a spoken one -- same history,
same session, same tools. Ask out loud, follow up by typing, and either can
refer to what the other said.

```bash
assistant what is using all my disk space
assistant "remind me what we decided about the tomatoes"
echo "summarise this" | assistant          # reads stdin when given no words
assistant --speak "and read that one out"  # typed questions are silent by default
assistant --status                         # model, local server, voice state, session
assistant --new                            # start a fresh conversation
```

### Swapping models

**SUPER+M** swaps which model answers -- the local Qwen3.8 or Claude -- and
shows you which one you landed on. The same thing from a terminal:

```bash
assistant --swap                 # local <-> Claude
assistant --backend local        # or claude, or auto
```

The choice is written to `~/.config/voice-assistant/env`, so it survives a
restart and `voice-llm status` agrees with it. Switching *to* the local model
starts the server if it is not up, and Claude answers the ~35 s it takes to
load. Nothing is ever stopped by the swap: `voice-llm claude` remains the
deliberate way to free the card for GPU passthrough.

Each backend keeps its own thread. Swapping mid-conversation means the model
you switch to has not heard what the other one did.

## LLM backend: Claude or a local Qwen3.8-27B

The backend is `auto` by default:

1. A model server that already answers wins.
2. Otherwise, an NVIDIA GPU with ≥ 15000 MiB and an installed `qwen38.service`
   means **local**.
3. Otherwise **Claude**.

While the local server is loading — 5 s warm, ~35 s from cold, so most of one
boot — individual questions go to Claude with a distinct chime and a
notification saying so, and the assistant switches over on its own the moment
the model is ready. No restart, and no dead assistant during boot.

| | Claude Code CLI | local Qwen3.8-27B |
|---|---|---|
| first token | ~0.9 s (persistent process) | 0.45–0.71 s |
| shell / filesystem | yes | yes (`run_shell`) |
| web access | yes | no |
| network required | yes | no |
| VRAM while running | 0 | ~15.3 GB of 16 GB |

**The local model has shell access too.** A `run_shell` tool gives it the same
reach as the Claude backend, which already runs with
`--dangerously-skip-permissions`. Every command is logged:

```bash
grep run_shell ~/.local/state/voice-assistant/voice-assistant.log
```

(`journalctl -p warning` does *not* filter these: systemd tags everything the
service writes to stdout as `info`, so the Python level is only text in the
line.) Set `VOICE_ASSISTANT_LOCAL_TOOLS=0` for an answer-only assistant.

Both backends are told that their input is an automatic transcript that can
contain recognition errors, and to confirm before anything irreversible.

### Why this model and these flags

- **`unsloth/Qwen3.8-27B-GGUF`, `UD-IQ4_XS` (13.27 GiB).** The largest quant
  that still fits in 16 GB. Measured against `UD-Q3_K_XL` (12.24 GiB) it is
  both *better quality and faster* — 35.2 vs 32.9 tok/s — because the i-quant
  kernels unpack faster than Q3_K on Ampere. `UD-Q4_K_S` (14.30 GiB) leaves no
  room for the KV cache.
- **MTP speculative decoding is worth ~900 MB of VRAM**: 24.4 → 35.2 tok/s
  (+48%), draft acceptance ~0.56. The draft head is embedded in the Unsloth
  GGUF (`blk.64.nextn.*`), so no separate draft model is needed.
- **`--spec-draft-n-max 2`, not 3.** Depth 3 *lowers* throughput to 31.2 tok/s
  as acceptance falls to 0.40.
- **Only 16 of 64 layers hold KV** (`full_attention_interval=4`; the other 48
  are Gated DeltaNet with a context-independent state), so 32K context costs
  only ~1.1 GB at `q8_0`.
- **`--ctx-checkpoints 0`.** This is a hybrid recurrent model, so llama-server
  snapshots ~150 MB of recurrent state at fixed points of every prompt pass —
  measured ~195 ms each, twice per turn, which was most of what a warm turn
  spent before its first token. Turning them off drops the prompt phase from
  0.4–0.9 s to ~0.1 s. The cost is that starting a new conversation
  re-processes the ~590-token prefix, which it already did in practice.
- **Thinking is off by default.** Qwen3.8 reasons by default and for voice that
  is pure latency. `/no_think` in the prompt does *not* work on this model; the
  `enable_thinking` jinja kwarg is what gates it
  (`VOICE_ASSISTANT_LOCAL_THINK=1` to turn it back on).

## Speech recognition

Moonshine, `medium_streaming` by default. Two reasons it beats Whisper here:
its encoder is variable-length (Whisper zero-pads every clip to 30 s, so a
2-second command costs the same as a 30-second one), and its streaming models
transcribe *during* speech, leaving only a flush when you stop.

Measured on **real human speech** (24 LibriSpeech utterances) and on command
audio synthesised with a different TTS than this project's own:

| model | LibriSpeech WER | command WER | flush after speech |
|---|---|---|---|
| tiny_streaming | 16.1% | 9.8% | 0.02 s |
| small_streaming | 11.5% | 9.2% | 0.17 s |
| **medium_streaming** | **7.8%** | **5.7%** | 0.76 s |

An earlier round of testing concluded medium bought nothing over small. That
was an artifact: the test audio came from Kokoro, this project's own TTS, and
it is clean and uniform enough that even *tiny* nearly matches *small* on it.
**Do not rank ASR models on audio produced by your own synthesiser.**

Because the model runs while you talk, its size is nearly free in perceived
latency. Those decoding passes are not optional bookkeeping — they *are* the
transcription. Suppressing them and asking for the text at the end returns an
empty string, because the decoder is incremental. What
`VOICE_ASSISTANT_MOONSHINE_UPDATE_INTERVAL` (0.25 s) controls is how much is
left to decode when you stop talking:

| interval | wait after the last word (2 s / 10.4 s utterance) |
|---|---|
| 0.50 s | 0.65 s / 2.04 s |
| **0.25 s** | **0.46 s / 1.71 s** |

for the same total CPU. The library raises the interval on its own when a pass
costs more than it, so a slow machine degrades to batch behaviour rather than
falling further behind on every pass.

## End of turn

`smart-turn v3.2` from [pipecat](https://github.com/pipecat-ai/smart-turn)
(BSD-2-Clause): a Whisper-tiny encoder and a classifier head, 8.7 MB, int8. It
is asked "did that sound finished?" at a schedule of checkpoints, and **the bar
comes down as the pause lengthens**:

| silence so far | probability needed to end the turn |
|---|---|
| 0.35 s | 0.90 |
| 0.70 s | 0.75 |
| 1.10 s | 0.60 |
| 1.60 s | 0.50 |
| 2.50 s | ends regardless |

That shape is the point. A mid-sentence breath is short, so early on the model
has to be nearly certain before it cuts you off; a pause that keeps going is
itself evidence the turn is over. A flat threshold at 0.2 s ended turns on
ordinary pauses in natural speech. A finished sentence scores 0.98-0.99, so it
still ends at the first checkpoint.

Measured on 650 real human utterances from the project's own test set: **92.9%
accurate** (7.9% false-complete, 6.3% false-incomplete), 60–120 ms per call on
this CPU. v3.0 scores 82.5% on the same data — v3.2 is the one to use.

The effect on a turn: **~2.7 s of dead air after the last word becomes ~0.4 s**,
while mid-sentence pauses of 0.3, 0.5 and 0.8 s all survive (measured). A false
trigger, which used to cost a full silence timeout, resolves in about the same
0.4 s, because silence scores 0.99 "complete" and the empty transcript is
discarded.

If it still cuts you off, raise the early bars:
`VOICE_ASSISTANT_SMART_TURN_CHECKPOINTS="0.5:0.95,0.9:0.8,1.4:0.6,1.9:0.5"`.
A false "unfinished" only costs the wait to the next checkpoint.

Set `VOICE_ASSISTANT_SMART_TURN=0` to go back to the timeout alone.

## Speech synthesis

Kokoro v1.0 ONNX at full precision on the CPU. The quantized community builds
(q8f16, ~83 MB) load and run but benchmarked **4.6× slower** on this CPU — int8
needs hardware acceleration to pay off, and this chip has AVX2 but no VNNI.

ONNX Runtime is pinned to *physical* cores, not hyperthreads. Measured on 6.2 s
of audio: 16 logical threads RTF 2.41, 8 physical RTF 1.18. Above 1.0 the
queue drains mid-reply and you hear it as chunky, stop-start speech.

Output goes through one persistent PipeWire stream rather than a `pw-play` per
sentence: measured 85 ms of silence at every sentence boundary before, 0 ms
now, first word at ~30 ms instead of ~150 ms, and aborting fades out in 5 ms
instead of cutting the waveform mid-cycle.

The first unit of a reply is allowed to end at a comma. Kokoro costs ~0.5 s of
fixed overhead plus ~0.1 s per word, so waiting for a full stop put seconds
between the model's first token and the first sound.

## Configuration

`~/.config/voice-assistant/env` is read by the service.

| variable | default | meaning |
|---|---|---|
| `VOICE_ASSISTANT_LLM_BACKEND` | `auto` | `auto`, `claude` or `local` |
| `VOICE_ASSISTANT_LLM_FALLBACK` | `1` | answer with Claude while the local model is down |
| `VOICE_ASSISTANT_LOCAL_MIN_VRAM_MIB` | `15000` | VRAM needed to pick the local model |
| `VOICE_ASSISTANT_LOCAL_URL` | `http://127.0.0.1:8081/v1` | any OpenAI-compatible endpoint |
| `VOICE_ASSISTANT_LOCAL_MAX_TOKENS` | `512` | reply cap |
| `VOICE_ASSISTANT_LOCAL_HISTORY_TURNS` | `12` | conversation turns kept |
| `VOICE_ASSISTANT_LOCAL_THINK` | `0` | let the local model reason first |
| `VOICE_ASSISTANT_LOCAL_TOOLS` | `1` | give the local model a shell |
| `VOICE_ASSISTANT_MODEL` / `_EFFORT` | `opus` / `max` | Claude model and effort |
| `VOICE_ASSISTANT_CLAUDE_PERSISTENT` | `1` | keep one `claude` process alive across turns |
| `VOICE_ASSISTANT_MOONSHINE_MODEL` | `medium_streaming` | STT model |
| `VOICE_ASSISTANT_SMART_TURN` | `1` | semantic end-of-turn detection |
| `VOICE_ASSISTANT_SMART_TURN_CHECKPOINTS` | `0.35:0.90,…` | `silence:probability` pairs; raise to be cut off less |
| `VOICE_ASSISTANT_SILENCE_TIMEOUT` | `2.5` | ends the turn regardless |
| `VOICE_ASSISTANT_MOONSHINE_UPDATE_INTERVAL` | `0.25` | seconds of audio between decoding passes |
| `VOICE_ASSISTANT_LISTEN_CHIME` | `1` | chime when the mic goes live again after a reply |
| `VOICE_ASSISTANT_TTS_ENGINE` | `kokoro` | `kokoro`, `pocket` or `supertonic` |
| `VOICE_ASSISTANT_TTS_VOICE` | `af_heart` | Kokoro voice |
| `VOICE_ASSISTANT_TTS_THREADS` | physical cores | ONNX threads for synthesis |

## Waybar

The status file is JSON with a class array of `[state, backend]`, so both can
be styled:

```jsonc
"custom/voice": {
    "exec": "cat ~/.local/state/voice-assistant/waybar-status",
    "return-type": "json",
    "interval": 1,
    "on-click": "kill -USR1 $(cat ~/.local/state/voice-assistant/voice-assistant.pid)",
    "on-click-right": "kill -USR2 $(cat ~/.local/state/voice-assistant/voice-assistant.pid)"
}
```

States are `off ◯`, `ready ●`, `listening ◉`, `thinking ◈`, `speaking ◆`, and
the second class is `claude` or `local`.

## Suspend and resume

If the local model is installed, `setup.sh` also installs
`qwen38-sleep.service` and `qwen38-wake.service`, which unload the model before
suspend and reload it after resume. This is not tidiness: the NVIDIA driver
runs with `NVreg_PreserveVideoMemoryAllocations=1`, so it copies *all* of VRAM
to disk on suspend. With the model resident that is 15.2 GB — measured at 84 s
inside `nvidia-suspend.service` plus a 40 s filesystem sync, an
`NVRM: Error in service of callback`, and a black screen on resume.

A `/usr/lib/systemd/system-sleep/` hook does **not** work for this: those run
from `systemd-suspend.service`, and `nvidia-suspend.service` is ordered
`Before=` it, so the dump has already happened. The unit must be ordered
`Before=nvidia-suspend.service`.

Before passing the GPU through to a VM, run `voice-llm claude` (stops the
server now) or `voice-llm claude --disable` (and keeps it from returning at
boot). The assistant keeps working either way.

## Files

| File | Description |
|------|-------------|
| `voice_assistant.py` | The assistant |
| `setup.sh` | One-shot installer, including the local LLM stack |
| `voice-assistant-ctl` | start / stop / status / toggle / logs / test |
| `voice-llm` | Switch and inspect the LLM backend |
| `assistant` | Ask from a terminal, in the same conversation as the voice |
| `voice-assistant.service` | Systemd user unit |
| `test_installation.py` | Installation checks |
| `contrib/qwen38.service` | Model server unit (installed when the GPU qualifies) |
| `contrib/qwen38-gpu-ok` | VRAM guard, so one unit file is safe on every machine |
| `contrib/qwen38-sleep*` | Unload/reload the model around suspend |

## Credits

- [Silero VAD](https://github.com/snakers4/silero-vad) — voice activity detection
- [Moonshine](https://github.com/usefulsensors/moonshine) — streaming speech recognition
- [smart-turn v3](https://huggingface.co/pipecat-ai/smart-turn-v3) — end-of-turn detection (BSD-2-Clause, © Daily)
- [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) — speech synthesis
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — local model serving

## License

MIT
