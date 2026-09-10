# Voice Assistant for Linux

**A hands-free voice assistant for the Linux desktop.** Speech recognition,
end-of-turn detection and speech synthesis all run locally on your CPU. The
language model runs on your own GPU, or through Claude, or both — whichever
this machine can manage. Tap a key, talk, and it answers out loud and acts on
your system.

Built for **Wayland**, with key bindings and a status indicator for **Hyprland**
and **GNOME**. No cloud speech services, no wake word listening to the room, no
account required for the parts that hear you.

```
mic ─▶ capture thread ─▶ Silero VAD ─▶ smart-turn v3 (has the user finished?)
                      └▶ Moonshine STT (transcribes while you talk)
                                    │
                                    ▼
         Claude Code CLI  ·or·  local Qwen3.8-27B (own loop ·or· DeepSeek Harness)
                                    │
                                    ▼
                      Pocket TTS ─▶ one persistent PipeWire stream
```

Tap **SUPER** to start listening. Speak. It answers.

Below, **SUPER** means the toggle key: bare SUPER on Hyprland, the SUPER+ALT
chord on GNOME — see [Desktop](#desktop).

---

## Is this what you are looking for?

**Yes, if you want:** a voice interface to your Linux desktop that can run
shell commands; speech recognition and synthesis that never leave the machine;
a local LLM assistant with a voice; a hands-free front end to Claude Code; or
something to talk to while your hands are busy.

**No, if you want:** a wake word ("hey computer") — this is push-to-talk by
design, so nothing is transcribed until you ask; a phone or smart-speaker
appliance; Windows or macOS support; or X11.

**Partly offline.** Speech in and speech out are always local. The *thinking*
is local only if you have the GPU for it; otherwise it goes to Claude. See
[Privacy](#privacy-and-what-leaves-the-machine) for exactly what crosses the
network in each mode.

## Contents

- [Quick start](#quick-start) · [Requirements](#requirements) · [How it works](#how-it-works) · [Features](#features)
- [Control](#control) · [LLM backends](#llm-backends-claude-or-a-local-qwen38-27b)
- [Speech recognition](#speech-recognition) · [End of turn](#end-of-turn) · [Speech synthesis](#speech-synthesis) · [Making text speakable](#making-text-speakable)
- [Knowing that time passed](#knowing-that-time-passed) · [Web search](#web-search) · [Desktop](#desktop) · [Configuration](#configuration)
- [Privacy](#privacy-and-what-leaves-the-machine) · [Power and GPU passthrough](#power-suspend-and-gpu-passthrough) · [Troubleshooting](#troubleshooting)
- [Addendum: for the next reader, human or model](#addendum-for-the-next-reader-human-or-model)

## Quick start

```bash
git clone https://github.com/fursman/Assistant.git ~/voice-assistant
cd ~/voice-assistant
./setup.sh
systemctl --user start voice-assistant.service
```

`setup.sh` installs the CPU pipeline on any machine and, if it finds an NVIDIA
GPU with enough VRAM, additionally builds llama.cpp with CUDA, downloads the
model and installs the model server. It also sets up whichever desktop it finds
itself on: on GNOME the key bindings and the top-bar indicator are installed
outright (log out and in once to load the indicator); on Hyprland it prints the
one line to add to `hyprland.conf`.

It is idempotent — re-run it whenever something is missing.

```bash
./setup.sh --no-llm         # skip the local model even if the GPU qualifies
./setup.sh --desktop        # redo only the key bindings and indicator
./setup.sh --rebuild-llama  # rebuild llama.cpp
./setup.sh --reinstall      # recreate the Python venv from scratch
./setup.sh --sleep-units    # only the suspend/resume units
```

Verify with `./test_installation.py`.

## Requirements

- **Linux with Wayland.** Hyprland and GNOME get key bindings and a status
  indicator from `setup.sh`; any other compositor works with two commands bound
  by hand.
- **PipeWire**, and a notification server. GNOME and KDE have one built in; on
  Hyprland or Sway run swaync, mako or dunst.
- **Python 3.11+**
- **Either** the [`claude` CLI](https://claude.ai/install.sh) (installed for you
  by `setup.sh`) **or** an NVIDIA GPU with ≥ 16 GB of VRAM for the local model.
- **Disk:** ~1.5 GB for the Python venv, ~700 MB of speech models, a further
  ~14 GB if you install the local LLM.

Neither backend is mandatory at build time — install both and the assistant
picks per query.

## How it works

1. **SUPER** toggles voice mode. The status bar shows the state (a waybar module
   on Hyprland, a top-bar indicator on GNOME); chimes mark the edges.
   **SUPER+M** swaps which backend answers; **SUPER+SHIFT+V** starts a new
   conversation. You can also just type: `assistant <question>` joins the same
   conversation from a terminal.
2. **Silero VAD** watches the mic. Speech has to persist for ~96 ms before a
   turn starts, so a cough does not trigger one.
3. **Moonshine** transcribes *while you are still speaking*, so when you stop
   only the tail is left to decode — about 0.4 s for a short command.
4. **smart-turn v3** decides you have finished, from the sound of the sentence
   rather than a stopwatch. It hears the difference between "turn the lights
   off" and "turn the lights off and…".
5. The transcript goes to **Claude Code** or a **local Qwen3.8-27B**, whichever
   this machine is set up for, and the model can run commands to answer.
6. **Pocket TTS** synthesises the reply clause by clause and plays it through a
   single PipeWire stream, so speech starts as soon as the first few words exist
   and there are no gaps between sentences.
7. **Everything lands in the top-bar transcript as it happens** — what you said,
   the model's reasoning when it reasons, each tool call with the reason and the
   exact command, and the reply clause by clause. Notifications only carry
   status. Left click the robot to read along; right click for the controls.

## Features

- **Semantic end of turn** — an 8.7 MB audio classifier, not a silence timer.
  ~0.4 s of dead air after you stop instead of ~2.7 s, and it holds through a
  mid-sentence pause instead of cutting you off.
- **Two backends, chosen automatically** — the local model when this machine can
  hold it, Claude otherwise, with a per-query fallback either way.
- **Two harnesses for the local model** — the assistant's own tool loop, or the
  [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) running
  locally against the same llama-server, with the assistant's web tools plugged
  in over MCP.
- **Speech recognition and synthesis on the CPU** — they keep working while the
  dGPU is passed through to a VM.
- **Streaming everywhere** — transcription during speech, synthesis during
  generation, playback during synthesis.
- **Runs commands** — both backends can act on the system, and both are told the
  input is a transcript that might be wrong.
- **Repairs proper nouns** — a vocabulary file fixes names the recogniser cannot
  be expected to know.
- **Abortable** — tap SUPER again; speech stops mid-word without a click and the
  model is interrupted rather than killed.
- **Audible turn-taking** — the same chime that arms voice mode plays again when
  the microphone goes live after a reply.
- **Ask by voice or by keyboard**, in one conversation.
- **A live transcript in the top bar** — you, the model's thinking, every tool
  call headed by its reason with the exact command beneath, and the reply as it
  streams. Select across it with the mouse and Ctrl+C; click a code block to
  copy it; one button copies the whole thing.
- **Change the Claude model or effort out loud** — "switch to fable", "set the
  effort to high". The session is resumed, so nothing said so far is lost.
- **Knows when time passed** — after a long gap, a restart or a reboot, the
  model is told so, and only then, so it stops describing last night's browser
  windows as though they were still open.
- **Systemd user service**, a **status indicator** whose header says what is
  happening and which model is answering, and **desktop notifications** for
  status only: tool calls, model changes, errors. They dismiss themselves; the
  transcript is the record.

## Control

```bash
voice-assistant-ctl status       # service, voice state, backend, session
voice-assistant-ctl start|stop   # the systemd user unit
voice-assistant-ctl restart      # reload it
voice-assistant-ctl toggle       # same as tapping SUPER
voice-assistant-ctl new-session  # start a fresh conversation
voice-assistant-ctl logs -f      # follow
voice-assistant-ctl test         # installation checks
voice-assistant-ctl backend …    # passes through to voice-llm

voice-llm status                 # what backend is configured, and what it resolves to
voice-llm auto                   # local model if this machine can run it, else Claude
voice-llm qwen                   # force the local model (starts the server)
voice-llm dsh                    # the local model, driven by the DeepSeek Harness
voice-llm claude                 # force Claude and stop the server (frees the GPU)
voice-llm logs                   # the model server's own log
```

Say **"new conversation"** (or "start over", "forget everything", "fresh start")
to clear the context by voice. Say **"switch to fable"** (or opus, sonnet,
haiku) or **"set the effort to extra high"** (low, medium, high, xhigh, max) to
change what answers; see [Swapping models](#swapping-models).

### Asking from a terminal

`assistant` talks to the running service over a control socket, so a typed
question lands in the **same conversation** as a spoken one — same history, same
session, same tools. Ask out loud, follow up by typing, and either can refer to
what the other said.

```bash
assistant what is using all my disk space
assistant "remind me what we decided about the tomatoes"
echo "summarise this" | assistant          # reads stdin when given no words
assistant --speak "and read that one out"  # typed questions are silent by default
assistant --status                         # model, local server, voice state, session
assistant --new                            # start a fresh conversation
assistant --json                           # the raw reply, for scripting
```

### Swapping models

**SUPER+M** cycles which model answers — the local Qwen3.8 in the assistant's own
loop, the same model under the DeepSeek Harness, then Claude — and shows you
which one you landed on. The same thing from a terminal:

```bash
assistant --swap                 # local -> dsh -> Claude -> local
assistant --backend local        # or dsh, claude, auto
assistant --model fable          # or opus, sonnet, haiku
assistant --effort xhigh         # or low, medium, high, max
```

**Claude's model and effort change the same way**, by voice ("switch to fable",
"set the effort to extra high"), from the indicator's Model and Effort submenus
with the current one ticked, or with the flags above. Both are handed to the CLI
at spawn, so the persistent process is replaced when either changes -- but its
session id is kept and passed to `--resume`, so the new model picks the
conversation up where the old one left off. Switching mid-thought costs nothing.

The choice is written to `~/.config/voice-assistant/env`, so it survives a
restart and `voice-llm status` agrees with it. Switching *to* the local model
starts the server if it is not up, and Claude answers the ~35 s it takes to
load. Nothing is ever stopped by the swap: `voice-llm claude` remains the
deliberate way to free the card for GPU passthrough.

Each backend keeps its own thread. Swapping mid-conversation means the model you
switch to has not heard what the other one did.

## LLM backends: Claude or a local Qwen3.8-27B

The backend is `auto` by default:

1. A model server that already answers wins.
2. Otherwise, an NVIDIA GPU with ≥ 15000 MiB and an installed `qwen38.service`
   means **local**.
3. Otherwise **Claude**.

When `auto` lands on the local model it uses the assistant's own tool loop; set
`VOICE_ASSISTANT_LOCAL_HARNESS=dsh` to have it use the DeepSeek Harness instead.

While the local server is loading — 5 s warm, ~35 s from cold, so most of one
boot — individual questions go to Claude with a distinct chime and a
notification saying so, and the assistant switches over on its own the moment
the model is ready. No restart, and no dead assistant during boot.

| | Claude Code CLI | local Qwen3.8-27B | Qwen3.8-27B under the DeepSeek Harness |
|---|---|---|---|
| first token | ~0.9 s (persistent process) | 0.45–0.71 s | ~0.15 s warm, 3–4 s on a fresh runtime |
| shell / filesystem | yes | yes (`run_shell`) | yes (persistent `bash`, `str_replace_editor`) |
| web access | yes | `web_search` / `fetch_page` | the same two, over MCP |
| network required | yes | no (search excepted) | no (search excepted) |
| VRAM while running | 0 | ~15.3 GB of 16 GB | ~15.3 GB of 16 GB |

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

### The DeepSeek Harness (`dsh`)

`voice-llm dsh` (or `assistant --backend dsh`, or SUPER+M) drives the **same
local llama-server** through the
[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness), DeepSeek's
open-source agent runtime, instead of the assistant's own tool loop. Same model,
different harness: its agent loop, a persistent `bash`, a file editor, and the
assistant's `web_search` / `fetch_page` served to it as an MCP server
(`dsh_web_mcp.py`). Nothing leaves the machine except the searches.

`setup.sh` installs it with `pip install deepseek-harness-sdk`; the wheel bundles
the `dsh` runtime (~260 MB unpacked), so no Node.js is needed. The assistant
launches `dsh --profile sdk-minimal` as a subprocess speaking JSON-RPC on stdio,
points the stock DeepSeek adapter at `http://127.0.0.1:8081` (it speaks
OpenAI-style chat completions, which is what llama-server serves) and overlays a
patch it writes into `~/.local/state/voice-assistant/dsh/` at every start.
Measured here: the runtime boots in ~1 s; the first prompt on a fresh runtime
(~2K tokens of tool schemas and persona, nothing cached) takes 3–4 s to first
token; warm turns ~0.15 s.

Four properties of the runtime shape how it is driven:

- **No cancel in the SDK protocol.** The only way to stop a turn is to close the
  runtime, so an abort (tap SUPER) costs a respawn on the next question.
- **A new runtime does not resume its session.** The session log is written, but
  a fresh process given the same id starts from nothing (measured). So the
  assistant keeps the transcript itself (`dsh_transcript.json`) and seeds every
  new runtime with a recap of the recent exchanges. That also covers the minimal
  profile's lack of compaction: past `VOICE_ASSISTANT_DSH_ROTATE_TOKENS` (24000)
  of prompt, the harness gets a fresh session with a recap rather than running
  into the 32K context.
- **No tool-call cap of its own.** Two runaways were measured: the same Wikipedia
  URL fetched 25 times until the prompt outgrew the context, and the same
  Gutenberg search run 21 times for a book that is not on Gutenberg. So an
  identical call may now run twice in a turn and the third ends it, with
  `VOICE_ASSISTANT_DSH_MAX_TOOL_CALLS` (500) as the backstop for loops that vary
  their wording. The assistant says why it stopped, and the exchange goes into
  the transcript so the respawned runtime does not start over.
- **A context overflow used to be permanent.** Rotation only ran after a
  *successful* turn, so one overflow left every later turn failing until the
  service was restarted. An overflow now rotates the session and retries once.

Every command it runs is logged like the native loop's: `grep 'dsh tool'` in the
log. The harness's minimal profile runs with full access, the same stance as the
other two backends.

### Why this model and these flags

- **`unsloth/Qwen3.8-27B-GGUF`, `UD-IQ4_XS` (13.27 GiB).** The largest quant that
  still fits in 16 GB. Measured against `UD-Q3_K_XL` (12.24 GiB) it is both
  *better quality and faster* — 35.2 vs 32.9 tok/s — because the i-quant kernels
  unpack faster than Q3_K on Ampere. `UD-Q4_K_S` (14.30 GiB) leaves no room for
  the KV cache.
- **MTP speculative decoding is worth ~900 MB of VRAM**: 24.4 → 35.2 tok/s
  (+48%), draft acceptance ~0.56. The draft head is embedded in the Unsloth GGUF
  (`blk.64.nextn.*`), so no separate draft model is needed.
- **`--spec-draft-n-max 2`, not 3.** Depth 3 *lowers* throughput to 31.2 tok/s as
  acceptance falls to 0.40.
- **Only 16 of 64 layers hold KV** (`full_attention_interval=4`; the other 48 are
  Gated DeltaNet with a context-independent state), so 32K context costs only
  ~1.1 GB at `q8_0`.
- **`--ctx-checkpoints 0`.** This is a hybrid recurrent model, so llama-server
  snapshots ~150 MB of recurrent state at fixed points of every prompt pass —
  measured ~195 ms each, twice per turn, which was most of what a warm turn spent
  before its first token. Turning them off drops the prompt phase from 0.4–0.9 s
  to ~0.1 s. The cost is re-processing the ~590-token prefix when a conversation
  starts, which it already did in practice.
- **Thinking is off by default.** Qwen3.8 reasons by default and for voice that is
  pure latency. `/no_think` in the prompt does *not* work on this model; the
  `enable_thinking` jinja kwarg is what gates it
  (`VOICE_ASSISTANT_LOCAL_THINK=1` to turn it back on).

## Speech recognition

[Moonshine](https://github.com/usefulsensors/moonshine), `medium_streaming` by
default. Two reasons it beats Whisper here: its encoder is variable-length
(Whisper zero-pads every clip to 30 s, so a 2-second command costs the same as a
30-second one), and its streaming models transcribe *during* speech, leaving
only a flush when you stop.

Measured on **real human speech** (24 LibriSpeech utterances) and on command
audio synthesised with a different TTS than this project's own:

| model | LibriSpeech WER | command WER | flush after speech |
|---|---|---|---|
| tiny_streaming | 16.1% | 9.8% | 0.02 s |
| small_streaming | 11.5% | 9.2% | 0.17 s |
| **medium_streaming** | **7.8%** | **5.7%** | 0.76 s |

An earlier round of testing concluded medium bought nothing over small. That was
an artifact: the test audio came from Kokoro, this project's own TTS, and it is
clean and uniform enough that even *tiny* nearly matches *small* on it.
**Do not rank ASR models on audio produced by your own synthesiser.**

Because the model runs while you talk, its size is nearly free in perceived
latency. Those decoding passes are not optional bookkeeping — they *are* the
transcription. Suppressing them and asking for the text at the end returns an
empty string, because the decoder is incremental. What
`VOICE_ASSISTANT_MOONSHINE_UPDATE_INTERVAL` (0.25 s) controls is how much is left
to decode when you stop talking:

| interval | wait after the last word (2 s / 10.4 s utterance) |
|---|---|
| 0.50 s | 0.65 s / 2.04 s |
| **0.25 s** | **0.46 s / 1.71 s** |

for the same total CPU. The library raises the interval on its own when a pass
costs more than it, so a slow machine degrades to batch behaviour rather than
falling further behind on every pass.

### Names the recogniser cannot know

Moonshine takes no hotword list and no initial prompt, so proper nouns come back
as whatever ordinary words they sound like. In one session it produced
"Communication and Securities Establishment" for the Communications Security
Establishment. Those are close misses, and close misses are repairable after the
fact.

Put one term per line in `~/.config/voice-assistant/vocabulary.txt` (`#` for
comments) — people, places, products, jargon, your own surname:

```
Hyprland
PipeWire
Moonshine
```

Matching is deliberately timid, because a wrong correction is worse than the
original. Only near-misses are touched, an exact word is left alone, and a
window that is already correct is never rewritten. Terms are matched against
windows of one to four words, not just their own length, because the recogniser
splits and joins words as readily as it mishears them — "Moonshine" arrives as
"moon shine". `VOICE_ASSISTANT_VOCABULARY_RATIO` (0.80) is how close a near-miss
has to be; lower starts rewriting ordinary words that merely rhyme with
something on the list. It cannot help with a miss that is not phonetically
close.

### The Whisper fallback

If Moonshine fails to load, the assistant falls back to
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (`small` by default)
without stopping. The device is decided at runtime: CUDA when the dGPU is
actually usable, CPU otherwise. A hardcoded `cuda` would abort the whole
assistant the moment the card was handed to a VM. Set
`VOICE_ASSISTANT_STT_ENGINE=whisper` to use it deliberately, which is also the
better choice for long-form dictation.

## End of turn

`smart-turn v3.2` from [pipecat](https://github.com/pipecat-ai/smart-turn)
(BSD-2-Clause): a Whisper-tiny encoder and a classifier head, 8.7 MB, int8. It is
asked "did that sound finished?" at a schedule of checkpoints, and **the bar
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
ordinary pauses in natural speech. A finished sentence scores 0.98–0.99, so it
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

**[Pocket TTS](https://github.com/kyutai-labs/pocket-tts)** (kyutai, ~100M) is
the default, because it is the one engine that stays comfortably faster than
realtime on a thermally capped laptop CPU. Measured on the same four sentences
with the CPU held at ~1.4 GHz:

| engine | real-time factor | verdict |
|---|---|---|
| **Pocket** | **0.48** | default |
| Kokoro v1.0 (fp32) | 1.23 | slower than speech itself |
| Supertonic 3 | 1.90 | optional, `pip install supertonic` |

Anything at or above 1.0 drains the playback queue at every sentence boundary,
and no amount of buffering or thread splitting fixes a synthesiser slower than
the speaker. Below 1.0 the same machine sounds fluent. Weights (~440 MB)
download from Hugging Face on first load.

Output goes through **one persistent PipeWire stream** rather than a `pw-play`
per sentence: measured 85 ms of silence at every sentence boundary before, 0 ms
now, first word at ~30 ms instead of ~150 ms, and aborting fades out in 5 ms
instead of cutting the waveform mid-cycle.

The first unit of a reply is allowed to end at a comma, because waiting for a
full stop put seconds between the model's first token and the first sound. A
short prebuffer (`VOICE_ASSISTANT_TTS_PREBUFFER`, 0.35 s, bounded by a 1.2 s
deadline) absorbs a thermal dip without stalling the reply.

After playback the mic is held shut for `VOICE_ASSISTANT_TTS_TAIL_GATE` (0.35 s)
to swallow the room's tail rather than transcribe the assistant's own voice.
This used to be "wait for three quiet chunks", which threw away your reply
whenever you answered promptly — exactly when the assistant had asked a
question.

### Voices, and cloning one

The default voice is **Kokoro's `af_heart`, cloned into Pocket**: 16.5 s of
Kokoro speech run through Pocket's audio-prompt encoder and exported as
`voices/af_heart.safetensors`. A saved state loads in ~0 s and does **not**
require the gated voice-cloning weights — only *making* a new one does — so it
works on a machine that never logged in to Hugging Face.
`voices/af_heart_reference.wav` is the clip it came from, for regenerating it.

`VOICE_ASSISTANT_POCKET_VOICE` takes any of three things: a name matching a
`.safetensors` file in `voices/`, one of Pocket's presets (`alba`, `azelma`, …),
or a path to an audio clip to clone on the spot.

### The other two engines

`VOICE_ASSISTANT_TTS_ENGINE=kokoro` selects
[Kokoro](https://github.com/thewh1teagle/kokoro-onnx) v1.0 ONNX at **full
precision, deliberately**. The onnx-community quantized builds (q8f16 ~83 MB,
quantized ~89 MB) load and run but benchmarked **4.6× slower** on this CPU
(3.4 s vs 0.74 s per utterance): int8 needs hardware acceleration to pay off,
and this chip has AVX2 but no VNNI. They also name their token input
`input_ids` where kokoro-onnx feeds `tokens`, so they are not drop-in anyway.

ONNX Runtime is pinned to *physical* cores, not hyperthreads. Measured on 6.2 s
of audio: 16 logical threads RTF 2.41, 8 physical RTF 1.18.

`VOICE_ASSISTANT_TTS_ENGINE=supertonic` selects Supertonic 3 (~99M, ONNX, 31
languages), an extra install. Whichever engine is chosen, Kokoro is the
fallback if it fails to load, and espeak is the fallback after that.

## Making text speakable

A language model writes for the eye. Before anything reaches the synthesiser it
is rewritten for the ear:

- **Markdown is stripped** — headers, bold, links, list markers. A code block
  becomes the words "code block" rather than being read out symbol by symbol.
- **Numbers become words**, with the specific patterns first so they do not eat
  their own operands: currency with a scale word, times, percentages, years,
  decimals, ordinals, then plain integers.
- **Bare URLs and emoji are dropped**, because both are noise aloud.
- **`input/output` becomes "input or output"**, but not inside a path.

There is also a **tool-markup gate**. A model several tool calls deep will
sometimes emit the next one as plain prose, imitating the transcript it can see,
and llama-server only parses tool syntax while the request carried a tool
schema — so on a request that withheld one, the markup arrives as ordinary
content and heads for the speaker. That is how the assistant came to read a curl
pipeline out loud, one sentence at a time. Deltas are a few characters wide, so
anything that could still grow into a tool-call opener is held back until the
next delta settles it, and released untouched if it does not.

## Knowing that time passed

A conversation resumes seamlessly across a restart, and across days. The session
id is reused, so the model sees an unbroken exchange and answers as though the
last turn were seconds ago. Measured here: one conversation ran through two
service restarts and a thirteen-hour overnight shutdown with nothing in the
prompt saying so, and the assistant kept describing browser windows it had
opened the night before as though they were still on screen.

So the joins get marked, and only the joins:

```
[context: 13h 18m since the last exchange, now Wed 09 Sep 12:29 PDT; the machine rebooted since then]
```

A timestamp on *every* turn would be worse than none. Anything sitting next to
the user's words invites acknowledgement, and you would get "good evening" and
"still at it, I see" on turns where nothing had changed. A marker that appears
only on a change has nothing to acknowledge on an ordinary turn, and means
something when it does appear.

Three things can put one there:

| trigger | why it matters |
|---|---|
| more than `VOICE_ASSISTANT_CONTEXT_GAP` since the last turn | what the model called current may be stale |
| the assistant restarted, or the machine rebooted | windows it opened are gone, background work died, the mic was reopened |
| the timezone changed | the user travelled |

The reboot and the restart are told apart by `/proc/sys/kernel/random/boot_id`,
and the restart is the more useful of the two: the clock only says time passed,
a restart says what stopped being true.

The system prompt tells every backend that the line is generated rather than
spoken, and not to greet the user about it. State lives in
`~/.local/state/voice-assistant/turn_context.json`; a missing or corrupt file
means no marker rather than a wrong one, and the first turn after an install is
always silent.

## Web search

The local backends get two tools, `web_search` and `fetch_page`, shared by the
native loop and the DeepSeek Harness (over MCP) so there is one implementation
and one set of tuned thresholds.

They exist because the model kept trying to search by hand and could not. In one
session 18 of its 30 shell commands were curl-and-grep against Bing, DuckDuckGo
and Google, and they produced almost nothing: the DuckDuckGo endpoints answer a
bot challenge, and Bing returns real results in markup no one-shot regex is
going to match. It burned every tool call guessing and then told the user a
real, well-covered company did not exist.

Searches are the one thing that leaves the machine in local mode. Claude, being
a hosted model, has its own web access.

## Desktop

The assistant does not care which desktop it is on: it talks to the screen
through `notify-send`, takes SIGUSR1 (toggle) and SIGUSR2 (new conversation) on
its pid file, and publishes its phase to a JSON status file. What differs is how
the keys are bound and what draws the phase. The model is told which desktop it
is on, so it reaches for `hyprctl` or `gsettings` as appropriate;
`VOICE_ASSISTANT_DESKTOP` overrides the detection.

| | Hyprland | GNOME |
|---|---|---|
| Toggle voice mode | **SUPER** (tap alone) | **SUPER+ALT** (either order) |
| New conversation | **SUPER+SHIFT+V** | **SUPER+SHIFT+V** |
| Swap backend | **SUPER+M** | **SUPER+M** |
| Status | waybar module | top-bar indicator (Shell extension) |
| Installed by | `contrib/hyprland/hyprland-voice-assistant.conf` | `setup.sh` (gsettings + `contrib/gnome/`) |

Bare SUPER is GNOME's activities key, which is why the toggle is a two-modifier
chord there, bound in both orders as `<Super>Alt_L` and `<Alt>Super_L`; GNOME's
own SUPER+M (the notification list) is moved off, and stays on SUPER+V.

**Any other desktop:** bind `voice-assistant-ctl toggle` and
`voice-assistant-ctl new-session` to whatever you like. That is the whole
integration.

### Notifications

GNOME shows **one banner at a time** and queues the rest, and it treats the two
ways a notification ends very differently. One that times out stays in the
message list. One the app closes is erased from it. Both measured here.

That decided the design twice. The first pass closed nothing and let
everything expire, so the whole turn stayed in the message list. Then the
transcript took over the conversation *and* the commands, and a popup repeating
what the transcript already records became noise that GNOME never dismisses on
its own. So now **every notification closes itself after a few seconds**
(`VOICE_ASSISTANT_NOTIFY_TRANSIENT`), with one exception: errors, which nothing
else records, stay until you dismiss them. The message list is not the history
any more. The transcript is.

GNOME is a poor display, so it is used only for glances. **The conversation is
not in the notifications at all** -- it lives in the indicator (see below).
Notifications carry only status: tool calls, voice mode on and off, model
changes, errors.

Three findings forced that split, all measured on GNOME 50:

| finding | consequence |
|---|---|
| banners are **strictly queued**; a new one never preempts | the screen falls behind during a busy turn |
| GNOME largely **ignores the expiry** an app asks for | the one lever for pacing them does not exist |
| an expired notification stays in the message list, a **closed one is erased** | the only way to clear a banner early also deletes the history |

Together those make "always show the newest thing" and "keep everything"
mutually exclusive on that surface. Tuning cannot reconcile them; attempts at
5 s, 4 s, 2 s and 1 s each failed from a different direction.

`--replace-id` is not used, and this is why: it updates a notification *in
place in the tray* without raising a banner again. Once GNOME had retired the
first banner, a replacement was simply invisible.

Stacked banners, several on screen at once, are not possible on GNOME at all.
mako, dunst and swaync all do it, which is one thing the Hyprland side gets for
free.

### Status file

`~/.local/state/voice-assistant/status` is rewritten atomically on every phase
change, in waybar's custom-module format so it can be used verbatim:

```json
{"text": "◉", "class": ["listening", "claude"], "tooltip": "Voice Assistant — listening (claude)"}
```

States are `off ◯`, `ready ●`, `listening ◉`, `thinking ◈`, `speaking ◆`, and the
second class is `claude`, `local` or `dsh`. (`waybar-status` is a symlink to the
same file, for configs written against the old name.)

### GNOME indicator

`contrib/gnome/voice-assistant-indicator@fursman.com` puts a robot in the top
bar: dimmed when off, white when ready, and recoloured red, blue or green with
the word *listening*, *thinking* or *speaking* beside it. The button is a fixed
width, so the word appearing and disappearing never nudges it sideways.

**Left click opens the conversation.** A header line says what is happening
and who is answering -- `Listening · Claude · fable · xhigh` -- with three
buttons beside it: copy the whole transcript to the clipboard, new
conversation, and voice mode on or off. Under that the transcript fills the
rest of the screen; nothing sits below it. **Right click is the quick menu**
with the full set of controls: toggle, new conversation, swap backend, and on
the Claude backend the Model and Effort submenus with the current choice
marked. The mute chord works while either is
open, even though an open menu holds the keyboard grab, because the menu itself
watches for it. It watches the state directory with a file monitor, so it
changes the moment the assistant does.

**The conversation lives here.** The menu holds a scrolling transcript, oldest
at the top so it reads downwards, scrolled to the newest automatically. The
assistant writes `transcript.json` atomically on every turn, and the extension
re-renders only when the content actually changed, so it never fights your
scrolling. This is a surface the project owns, which is the whole point: no
queue, no expiry policy, no history semantics to work around, so it can simply
show the newest thing.

**Tool entries lead with the reason.** The popup for a command shows the
human-written description of why it is being run; the transcript entry is
headed by that same description, with the exact command beneath it. So the
phrase you glimpse in a bubble is the line you scroll to, and what actually
ran is directly under it.

**Thinking is in there too.** When the model is one that reasons first, its
reasoning streams into the transcript at the same sentence boundaries the
popup uses, in the tool lines' grey but the normal typeface, since it is prose.
It interleaves with speech and tool calls in the order it happened.

**You can select it with the mouse and Ctrl+C.** Consecutive entries are
rendered into one text actor, with each role's colour and face carried as
Pango markup rather than as separate labels. That is what makes a selection
run across entries and across changes of font: a selection lives inside a
single text object, so separate labels could never be selected together. A
code block is the one thing that breaks a run, because it is a button.

**Code blocks are rows you can click.** A fenced block cannot be spoken -- the
speech layer says only "code block" -- so in the transcript it renders as a
boxed, monospaced row that copies itself when clicked. It uses `St.Clipboard`,
the Shell's own, which matters: a Wayland clipboard is served by the process that
set it, and anything the assistant spawns dies with the command, which is why an
earlier attempt at copying from the assistant kept losing the selection. The
Shell outlives everything.

`setup.sh` installs and enables it; **GNOME on Wayland loads new extensions only
at login**, so log out and back in once.

### Waybar

```jsonc
"custom/voice": {
    "exec": "cat ~/.local/state/voice-assistant/status",
    "return-type": "json",
    "interval": 1,
    "on-click": "kill -USR1 $(cat ~/.local/state/voice-assistant/voice-assistant.pid)",
    "on-click-right": "kill -USR2 $(cat ~/.local/state/voice-assistant/voice-assistant.pid)"
}
```

Both classes can be styled, e.g. `#custom-voice.listening { color: #e01b24; }`.

## Configuration

`~/.config/voice-assistant/env` is read by the service. Everything below is
optional.

### Backend

| variable | default | meaning |
|---|---|---|
| `VOICE_ASSISTANT_LLM_BACKEND` | `auto` | `auto`, `claude`, `local` or `dsh` |
| `VOICE_ASSISTANT_LOCAL_HARNESS` | `native` | what `auto` uses for the local model: `native` or `dsh` |
| `VOICE_ASSISTANT_LLM_FALLBACK` | `1` | answer with Claude while the local model is down |
| `VOICE_ASSISTANT_LOCAL_MIN_VRAM_MIB` | `15000` | VRAM needed to pick the local model |
| `VOICE_ASSISTANT_MODEL` | `opus` | `opus`, `sonnet`, `haiku` or `fable`; changeable at runtime |
| `VOICE_ASSISTANT_EFFORT` | `max` | `low`, `medium`, `high`, `xhigh` or `max`; changeable at runtime |
| `VOICE_ASSISTANT_CLAUDE_PERSISTENT` | `1` | keep one `claude` process alive across turns |
| `VOICE_ASSISTANT_CLI_TIMEOUT` | | seconds a `claude` turn may take |
| `VOICE_ASSISTANT_CONTEXT_MARKERS` | `1` | tell the model when time passed or the machine restarted |
| `VOICE_ASSISTANT_CONTEXT_GAP` | `900` | seconds of silence before a gap is worth mentioning |

### Local model

| variable | default | meaning |
|---|---|---|
| `VOICE_ASSISTANT_LOCAL_URL` | `http://127.0.0.1:8081/v1` | any OpenAI-compatible endpoint |
| `VOICE_ASSISTANT_LOCAL_MODEL` | `qwen3.8-27b` | model name sent to that endpoint |
| `VOICE_ASSISTANT_LOCAL_API_KEY` | `none` | if your endpoint wants one |
| `VOICE_ASSISTANT_LOCAL_UNIT` | `qwen38.service` | the unit to start when switching to local |
| `VOICE_ASSISTANT_LOCAL_MAX_TOKENS` | `512` | reply cap |
| `VOICE_ASSISTANT_LOCAL_TIMEOUT` | `120` | seconds one completion may take |
| `VOICE_ASSISTANT_LOCAL_HISTORY_TURNS` | `12` | conversation turns kept |
| `VOICE_ASSISTANT_LOCAL_HISTORY_CHARS` | `24000` | and the character bound on them |
| `VOICE_ASSISTANT_LOCAL_THINK` | `0` | let the local model reason first |
| `VOICE_ASSISTANT_LOCAL_TEMP` | `0.7` | sampling temperature |
| `VOICE_ASSISTANT_LOCAL_TOP_P` / `_TOP_K` | `0.8` / `20` | nucleus and top-k |
| `VOICE_ASSISTANT_LOCAL_PRESENCE_PENALTY` | `1.5` | presence penalty |

### Local model tools

| variable | default | meaning |
|---|---|---|
| `VOICE_ASSISTANT_LOCAL_TOOLS` | `1` | give the local model a shell |
| `VOICE_ASSISTANT_LOCAL_TOOL_TIMEOUT` | `30` | seconds one command may run |
| `VOICE_ASSISTANT_LOCAL_MAX_TOOL_ITERS` | `5` | tool rounds in one turn |
| `VOICE_ASSISTANT_LOCAL_TOOL_MAX_OUTPUT` | `4000` | characters of output the model sees |
| `VOICE_ASSISTANT_LOCAL_TOOL_HISTORY_OUTPUT` | `600` | and what is kept in later turns |
| `VOICE_ASSISTANT_WEB_RESULTS` | `6` | search results returned |
| `VOICE_ASSISTANT_WEB_PAGE_CHARS` | `4000` | characters of a fetched page |
| `VOICE_ASSISTANT_WEB_TIMEOUT` | `12` | seconds for a search or fetch |

### DeepSeek Harness

| variable | default | meaning |
|---|---|---|
| `VOICE_ASSISTANT_DSH_MAX_TOKENS` | `2048` | reply cap under the harness |
| `VOICE_ASSISTANT_DSH_CONTEXT_WINDOW` | `32768` | context declared to the runtime |
| `VOICE_ASSISTANT_DSH_ROTATE_TOKENS` | `24000` | prompt size at which it gets a fresh, recapped session |
| `VOICE_ASSISTANT_DSH_RECAP_CHARS` | `6000` | size of that recap |
| `VOICE_ASSISTANT_DSH_TOOL_TIMEOUT` | `45` | seconds one harness `bash` command may run |
| `VOICE_ASSISTANT_DSH_MAX_TOOL_CALLS` | `500` | tool calls in one turn before it is cut short |
| `VOICE_ASSISTANT_DSH_MAX_REPEATS` | `2` | times an identical call may run; one more ends the turn |
| `VOICE_ASSISTANT_DSH_HOME` | `~/.local/state/voice-assistant/dsh` | profile, patch and session logs |

### Listening

| variable | default | meaning |
|---|---|---|
| `VOICE_ASSISTANT_STT_ENGINE` | `moonshine` | `moonshine` or `whisper` |
| `VOICE_ASSISTANT_MOONSHINE_MODEL` | `medium_streaming` | `tiny`, `base`, or a `*_streaming` variant |
| `VOICE_ASSISTANT_MOONSHINE_UPDATE_INTERVAL` | `0.25` | seconds of audio between decoding passes |
| `VOICE_ASSISTANT_WHISPER_MODEL` | `small` | fallback engine's model |
| `VOICE_ASSISTANT_WHISPER_DEVICE` | `auto` | `auto`, `cuda` or `cpu` |
| `VOICE_ASSISTANT_VOCABULARY` | `~/.config/voice-assistant/vocabulary.txt` | proper nouns to repair |
| `VOICE_ASSISTANT_VOCABULARY_RATIO` | `0.80` | how close a near-miss must be |
| `VOICE_ASSISTANT_VAD_START` / `_STOP` | `0.5` / `0.35` | Silero speech thresholds |
| `VOICE_ASSISTANT_VAD_WINDOWS` | `3` | 32 ms windows before a turn starts |
| `VOICE_ASSISTANT_SMART_TURN` | `1` | semantic end-of-turn detection |
| `VOICE_ASSISTANT_SMART_TURN_CHECKPOINTS` | `0.35:0.90,…` | `silence:probability` pairs; raise to be cut off less |
| `VOICE_ASSISTANT_SMART_TURN_THREADS` | `4` | ONNX threads for the classifier |
| `VOICE_ASSISTANT_SILENCE_TIMEOUT` | `2.5` | ends the turn regardless |

### Speaking

| variable | default | meaning |
|---|---|---|
| `VOICE_ASSISTANT_TTS_ENGINE` | `pocket` | `pocket`, `kokoro` or `supertonic` |
| `VOICE_ASSISTANT_POCKET_VOICE` | `af_heart` | a voice in `voices/`, a Pocket preset, or a path to a clip |
| `VOICE_ASSISTANT_TTS_VOICE` | `af_heart` | Kokoro voice, when Kokoro is selected |
| `VOICE_ASSISTANT_SUPERTONIC_VOICE` | `M1` | Supertonic voice style |
| `VOICE_ASSISTANT_TTS_SPEED` | `1.0` | Kokoro only; Pocket ignores it |
| `VOICE_ASSISTANT_TTS_THREADS` | physical cores | ONNX threads for synthesis |
| `VOICE_ASSISTANT_TTS_PREBUFFER` | `0.35` | seconds queued before the first word plays |
| `VOICE_ASSISTANT_TTS_PREBUFFER_WAIT` | `1.2` | deadline on that wait |
| `VOICE_ASSISTANT_TTS_TAIL_GATE` | `0.35` | silence held after playback before the mic is trusted |
| `VOICE_ASSISTANT_LISTEN_CHIME` | `1` | chime when the mic goes live again after a reply |

### Desktop

| variable | default | meaning |
|---|---|---|
| `VOICE_ASSISTANT_DESKTOP` | detected | `hyprland`, `gnome`, … |
| `VOICE_ASSISTANT_NOTIFY_EXPIRE` | `1000` | ms a notification stays up. This is also the lag before the next one can show, so raising it puts the banner behind reality |
| `VOICE_ASSISTANT_NOTIFY_TRANSIENT` | `4` | seconds before a status notification closes itself; errors never do |
| `VOICE_ASSISTANT_TRANSCRIPT_TURNS` | `200` | entries kept in the transcript file; every tool call is one, so a tool-heavy turn must not evict the speech around it |
| `VOICE_ASSISTANT_SOCKET` | under `~/.local/state` | control socket `assistant` connects to |

## Privacy and what leaves the machine

| | audio | transcript | commands run |
|---|---|---|---|
| **local backend** | never leaves | never leaves | never leave |
| **dsh backend** | never leaves | never leaves | never leave |
| **Claude backend** | never leaves | sent to Anthropic | sent to Anthropic |

Audio is never uploaded in any mode: VAD, transcription, end-of-turn detection
and synthesis are all local, always. In local mode the only outbound traffic is
`web_search` / `fetch_page`, and only when the model chooses to use them. Model
weights are downloaded once, on first run.

Nothing is recorded until you press the key. There is no wake word, which is a
deliberate trade: you give up "hey computer" and get a microphone that is
provably idle the rest of the time.

**Both backends run with full shell access** and are told to confirm before
anything irreversible, but that is a prompt, not a sandbox. Every command is
logged. `VOICE_ASSISTANT_LOCAL_TOOLS=0` turns the local model into an
answer-only assistant.

## Power, suspend and GPU passthrough

If the local model is installed, `setup.sh` also installs `qwen38-sleep.service`
and `qwen38-wake.service`, which unload the model before suspend and reload it
after resume. This is not tidiness: the NVIDIA driver runs with
`NVreg_PreserveVideoMemoryAllocations=1`, so it copies *all* of VRAM to disk on
suspend. With the model resident that is 15.2 GB — measured at 84 s inside
`nvidia-suspend.service` plus a 40 s filesystem sync, an `NVRM: Error in service
of callback`, and a black screen on resume.

A `/usr/lib/systemd/system-sleep/` hook does **not** work for this: those run
from `systemd-suspend.service`, and `nvidia-suspend.service` is ordered
`Before=` it, so the dump has already happened. The unit must be ordered
`Before=nvidia-suspend.service`.

Before passing the GPU through to a VM, run `voice-llm claude` (stops the server
now) or `voice-llm claude --disable` (and keeps it from returning at boot). The
assistant keeps working either way, because nothing in the speech pipeline wants
a GPU.

## Troubleshooting

**It cuts me off mid-sentence.** Raise the early checkpoints:
`VOICE_ASSISTANT_SMART_TURN_CHECKPOINTS="0.5:0.95,0.9:0.8,1.4:0.6,1.9:0.5"`. A
false "unfinished" only costs the wait to the next checkpoint.

**It waits too long after I stop.** Lower them, or check that smart-turn loaded
at all — without it you get the flat `VOICE_ASSISTANT_SILENCE_TIMEOUT` (2.5 s).
`voice-assistant-ctl logs` says which.

**Speech is chunky and stop-start.** Synthesis is slower than realtime on this
machine. Confirm the engine is `pocket`, not `kokoro`, and check whether the CPU
is thermally throttling. Raising `VOICE_ASSISTANT_TTS_PREBUFFER` buys a little
headroom.

**It mishears the same name every time.** Add it to
`~/.config/voice-assistant/vocabulary.txt`. If that does not take, the miss is
not phonetically close enough; try a lower `VOICE_ASSISTANT_VOCABULARY_RATIO`,
carefully.

**It reads tool calls or code out loud.** That is the markup gate missing a
pattern. Open an issue with the log line.

**The GNOME indicator does not appear.** Extensions load only at login on
Wayland. Log out and back in. `./setup.sh --desktop` reinstalls it.

**SUPER does nothing on GNOME.** Bare SUPER is the activities key; the toggle is
SUPER+ALT there. See [Desktop](#desktop).

**No sound, or the mic goes dead after a reply.** Check PipeWire is running and
that the service can see a notification daemon. The assistant reopens the
capture stream when the driver goes silent, and holds the playback stream open
across turns, because closing it froze the microphone.

**The local model never starts.** `voice-llm status` shows whether the server is
serving, loading, or failed, and `voice-llm logs` shows why. Loading from cold
takes ~35 s, during which questions go to Claude if the fallback is on.

**A menu item in the indicator does nothing.** The extension calls the installed
`assistant` command, and if that copy is older than the checkout it rejects the
flag and exits, silently. `./test_installation.py` names any installed copy that
differs from the checkout and how to refresh it; the extension also logs what
the command said to the journal (`journalctl --user -b -S -5min`).

**I edited the extension and nothing changed.** JavaScript loads only at login.
`gnome-extensions disable` then `enable` reloads the *stylesheet* only. Copy it
into place with `./setup.sh --desktop`, then log out and back in.

**The transcript shows no thinking.** Only models that reason first produce any:
Fable, or the local model with `VOICE_ASSISTANT_LOCAL_THINK=1`. Opus at the
default settings answers directly.

**Everything is slow on battery.** Unrelated to this project, but worth knowing:
power-profiles-daemon uses a more power-saving CPU preference when unplugged.

## Files

| File | Description |
|------|-------------|
| `voice_assistant.py` | The assistant |
| `web_tools.py` | Keyless web search and page fetch, shared by the local backends |
| `dsh_web_mcp.py` | The same two tools served to the DeepSeek Harness over MCP |
| `setup.sh` | One-shot installer, including the local LLM stack |
| `voice-assistant-ctl` | start / stop / status / toggle / logs / test |
| `voice-llm` | Switch and inspect the LLM backend |
| `assistant` | Ask from a terminal, in the same conversation as the voice |
| `voice-assistant.service` | Systemd user unit |
| `test_installation.py` | Installation checks |
| `voices/` | Saved Pocket voices and their reference clips |
| `contrib/gnome/` | GNOME Shell indicator extension |
| `contrib/hyprland/` | Hyprland key bindings |
| `contrib/qwen38.service` | Model server unit (installed when the GPU qualifies) |
| `contrib/qwen38-gpu-ok` | VRAM guard, so one unit file is safe on every machine |
| `contrib/qwen38-sleep*` | Unload/reload the model around suspend |

## Credits

- [Silero VAD](https://github.com/snakers4/silero-vad) — voice activity detection
- [Moonshine](https://github.com/usefulsensors/moonshine) — streaming speech recognition
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — fallback recognition
- [smart-turn v3](https://huggingface.co/pipecat-ai/smart-turn-v3) — end-of-turn detection (BSD-2-Clause, © Daily)
- [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) — speech synthesis (kyutai)
- [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) — alternative synthesis
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — local model serving
- [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) — alternative agent runtime

## Addendum: for the next reader, human or model

This is for whoever picks the project up cold: a person deciding whether to
run it, or an assistant asked to change it. It is the shortest true account of
what the thing does, and of the things that are not visible in the code.

### What it is, in one breath

A push-to-talk voice assistant for a Wayland desktop. Voice detection,
streaming recognition, semantic end-of-turn and synthesis all run locally on
the CPU. The thinking is Claude Code, or a local Qwen served by llama.cpp,
either with full shell access. The conversation is shown live in a top-bar
indicator; notifications are only for status. Nothing needs a GPU except the
optional local model.

### Where things are

- **One program**, `voice_assistant.py`. `VoiceAssistant` owns the loop.
  `ClaudeSession` and `DshSession` are the persistent backends. Every question,
  spoken or typed, goes through `_query_and_speak`; `_run_turn` picks the
  backend; `_finish_turn` ends the turn. `_notify` is status popups.
  `_append_transcript` and `_stream_transcript` write the indicator's
  transcript. `_set_status` publishes the status file. `set_claude_setting`
  changes model or effort. `_control_command` is the socket protocol the
  `assistant` command speaks.
- **State**, in `~/.local/state/voice-assistant/`: `status` (phase, backend,
  model, effort), `transcript.json` (the last 200 entries, roles `you`,
  `assistant`, `thinking`, `tool`, `system`), `turn_context.json` (for the
  time-gap marker), `session_id`, `voice-assistant.log`, `voice-assistant.pid`.
  The control socket lives in `$XDG_RUNTIME_DIR`.
- **Config**, in `~/.config/voice-assistant/`: `env` holds every
  `VOICE_ASSISTANT_*` above, and the backend, model and effort are written back
  into it when changed at runtime; `vocabulary.txt` holds names to repair.
- **Copies.** The extension under `contrib/gnome/` is installed as a copy in
  `~/.local/share/gnome-shell/extensions/`, and `assistant`, `voice-llm` and
  `voice-assistant-ctl` as copies in `~/.local/bin`. The service itself runs the
  checkout directly. Editing a copied file's source changes nothing until it is
  copied again; `./test_installation.py` names any copy that has drifted.

### What shows where

| surface | shows |
|---|---|
| indicator header | `Listening · Claude · fable · xhigh`, plus copy, new conversation, on/off |
| left click | the transcript, full height, selectable, code blocks clickable |
| right click | toggle, new conversation, swap backend, Model and Effort submenus |
| notifications | status only: tool calls, model changes, voice mode; all transient except errors |
| status file | phase and backend for waybar; also model, effort and the valid choices |
| `voice-assistant-ctl status` | service, process, phase, backend, Claude model and effort, session |
| the `[context: …]` line | prepended to a user turn only after a long gap, a restart or a reboot |

### Say it

- **"new conversation"**, "start over", "forget everything", "fresh start".
- **"switch to fable"** -- or opus, sonnet, haiku. Needs a switching verb and a
  short sentence; "use fable to explain the encoder" is a request, not a switch.
- **"set the effort to extra high"** -- or low, medium, high, max. Needs the
  word effort, thinking or reasoning.
- Tap the toggle key mid-reply to abort it.

### Things the code will not tell you

1. **If you are an assistant running inside this service**, restarting
   `voice-assistant.service` kills the process that is answering, mid-sentence.
   Either let the user do it, or schedule it detached with a delay so the spoken
   reply finishes first: `nohup bash -c 'sleep 12; systemctl --user restart
   voice-assistant.service' &`. The same pattern with `gnome-session-quit
   --logout --no-prompt` logs the user out cleanly, and that restarts *both*
   GNOME Shell and the assistant. A logout done by hand sometimes leaves the
   assistant running, because the new session overlaps the old one.
2. **Extension JavaScript loads only at login.** `gnome-extensions disable`
   then `enable` reloads the stylesheet and nothing else.
3. **`journalctl -S` silently rejects** the timestamp format `ps -o lstart=`
   prints, and with stderr discarded an empty result looks exactly like "no
   errors". Use `-S "-5min"` or `YYYY-MM-DD HH:MM:SS`. Extension `console.log`
   lines appear in `journalctl --user -b` as `gnome-shell[pid]: …`.
4. **`Meta` and `Clutter` cannot be introspected** from a plain `gjs` outside
   the shell process; only GLib and Gio can. Pure functions can be lifted into a
   temporary module and tested with `gjs -m` -- the selection highlight builder
   was.
5. **`timeout` on Ubuntu 26.04 is the uutils rewrite** and it crashed under us.
   Do not wrap commands in it.
6. **`sudo` logs its arguments** to `auth.log`. Never put a secret on a sudo
   command line; pass it on stdin.
7. **The `[context: …]` line** before a user turn is generated, not spoken. It
   means "what you remember may be stale", not "greet the user about the time".
8. **The recogniser hears the room.** Short fragments that make no sense --
   "post mode on", "you're a little" -- are usually the user talking to someone
   else. Ask; do not act on a guess. The user's surname arrives as "Fersman";
   that is what `vocabulary.txt` is for.

### Decisions, and the measurement behind each

- **The conversation lives in the indicator, not in notifications.** GNOME
  queues banners, ignores the expiry an app asks for, and the only way to clear
  one early also deletes it from the history. No tuning satisfies "newest thing
  visible" and "keep everything" at once; 5 s, 4 s, 2 s and 1 s each failed
  from a different direction.
- **Notifications are transient except errors.** Everything else is recorded
  better elsewhere, and GNOME never dismisses them itself.
- **`--replace-id` is never used.** It edits the tray entry in place and never
  raises a banner again; once the first banner has gone the update is invisible.
- **The transcript streams per clause**, and a change of role or a tool call
  closes the entry, so thinking, speech and commands appear in the order they
  happened rather than as a record afterwards.
- **A tool entry is the popup's description over the verbatim command.** The
  phrase glimpsed in a bubble is the line to scroll to.
- **Code blocks are clickable rows, not auto-copied.** Auto-copy clobbered the
  clipboard on every reply, and a Wayland selection dies with the process that
  set it; the Shell's own clipboard outlives everything.
- **Selection is driven from the capture phase and painted by us.** A press
  reaches the text in capture but its own handler never fires, and ClutterText
  paints a selection only while focused, which an open menu takes back. The
  highlight is a background span in the markup the run is already rendered
  with; copy slices the run's own character array.
- **Model and effort changes keep the session id**, so `--resume` carries the
  conversation across the switch. They are refused while an answer is in
  flight, because replacing the process would cut it off.
- **The time-gap marker fires only on a change.** A timestamp on every turn
  gets echoed back as small talk.
- **The transcript keeps 200 entries.** Every tool call is one, and at 40 a
  tool-heavy turn evicted the speech it was serving.
- **Pocket TTS is the default** because it stays faster than realtime on a
  throttled laptop CPU (RTF 0.48 against Kokoro's 1.23, measured at ~1.4 GHz).
- **This MacBook's hardware quirks** -- palm rejection, the lid, the sound card
  -- live in [fursman/Ubuntu](https://github.com/fursman/Ubuntu) under
  `macbook/`, not here.

### How to check a change

- `./test_installation.py`: 42 checks, including whether every installed copy
  matches the checkout.
- Python: bind the methods onto a stub object and call them against a temporary
  `TRANSCRIPT_FILE`. To test chunking, feed `_flush_sentences` a reply in small
  deltas, the way it actually arrives; testing `_prepare_for_speech` on a whole
  reply passes while the real path fails.
- JavaScript: lift pure functions into a temporary module and run `gjs -m`.
  Anything touching St, Clutter or Meta needs a login and the journal.
- When something silently does nothing, put one log line at the point of
  decision and read the journal with a valid timestamp. Do not guess twice.

## License

MIT

---

<sub>Keywords: Linux voice assistant · offline voice assistant · local voice
assistant · self-hosted AI assistant · Wayland · Hyprland · GNOME · Ubuntu ·
speech to text · text to speech · local LLM · llama.cpp · Qwen · Claude Code ·
voice control for Linux · push to talk · privacy-first voice assistant</sub>
