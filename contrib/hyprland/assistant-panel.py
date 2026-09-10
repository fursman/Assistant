#!/usr/bin/env python3
"""A live terminal view of the voice-assistant conversation, for Hyprland.

The GNOME build shows the conversation in a Shell-extension side pane. Hyprland
has no equivalent, so this renders the very same data -- the transcript and the
status file the assistant already publishes -- in a terminal, and redraws the
instant either file changes. It is meant to run in kitty on a special
workspace (a scratchpad) toggled with SUPER+A, exactly like the music pane.

Pure stdlib on purpose: no GTK, no rich, nothing to install. It only reads the
state files, so several copies can run and none of them can disturb the
assistant.
"""
import json, os, sys, time, shutil

STATE      = os.path.expanduser("~/.local/state/voice-assistant")
TRANSCRIPT = os.path.join(STATE, "transcript.json")
STATUS     = os.path.join(STATE, "status")

R="\x1b[0m"; B="\x1b[1m"; D="\x1b[2m"
CLR="\x1b[2J\x1b[H"; HOME="\x1b[H"; HIDE="\x1b[?25l"; SHOW="\x1b[?25h"; EOL="\x1b[K"
def fg(n): return f"\x1b[38;5;{n}m"
GREY=fg(244); WHITE=fg(252); BLUE=fg(39); CYAN=fg(44); MAGENTA=fg(170)
GREEN=fg(42); YELLOW=fg(220); ORANGE=fg(208)

STATES = {  # glyph + colour for the phase published in status.class[0]
    "off":       ("○", "off",       GREY),
    "ready":     ("●", "ready",     GREEN),
    "listening": ("◉", "listening", BLUE),
    "thinking":  ("◈", "thinking",  YELLOW),
    "speaking":  ("◆", "speaking",  MAGENTA),
}
ROLES = {  # colour + label + whether it is dimmed secondary text
    "you":       (BLUE,    "You",       False),
    "assistant": (WHITE,   "Assistant", False),
    "thinking":  (GREY,    "· thinking",True),
    "tool":      (CYAN,    "· tool",    True),
    "system":    (ORANGE,  "· system",  True),
}

def read_json(path, default):
    try:
        with open(path) as f: return json.load(f)
    except Exception:
        return default

def wrap(text, width):
    out = []
    for para in text.replace("\t", "  ").split("\n"):
        line = ""
        for word in para.split(" "):
            if line and len(line) + len(word) + 1 > width:
                out.append(line); line = word
            else:
                line = f"{line} {word}".strip() if line else word
        out.append(line)
    return out or [""]

def render(cols, rows):
    status = read_json(STATUS, {})
    turns  = read_json(TRANSCRIPT, {}).get("turns", [])
    cls    = status.get("class") or []
    phase  = cls[0] if len(cls) > 0 else "off"
    backend= cls[1] if len(cls) > 1 else ""
    model  = status.get("model", "")
    effort = status.get("effort", "")
    glyph, label, colour = STATES.get(phase, ("·", phase, WHITE))
    state = label or phase                       # "listening", or "ready"/"off"
    ident = " ".join(p for p in (backend.title(), model.title()) if p)

    left_plain  = f"Assistant · {state}"
    right_plain = ident + (f" · {effort}" if effort else "")
    pad = max(2, (cols - 2) - len(left_plain) - len(right_plain))
    left_col  = f"{B}{MAGENTA}Assistant{R} {D}·{R} {colour}{state}{R}"
    right_col = f"{WHITE}{ident}{R}" + (f" {D}·{R} {YELLOW}{effort}{R}" if effort else "")

    head = [
        "  " + left_col + " " * pad + right_col,
        "",
        f"{D}{GREY}  {'─' * (cols - 2)}{R}",
    ]

    body = []
    for t in turns:
        col, name, dim = ROLES.get(t.get("role", ""), (WHITE, t.get("role", ""), False))
        body.append(f"{col}{B if not dim else D}{name}{R}")
        for ln in wrap(t.get("text", ""), cols - 3):
            body.append(f"  {col}{D if dim else ''}{ln}{R}")
        body.append("")

    foot = f"{D}{GREY}  SUPER tap: mic   SUPER+M: model   SUPER+SHIFT+V: new{R}"
    avail = rows - len(head) - 1
    body = body[-avail:] if avail > 0 else []

    out = [HOME]
    for ln in head + body:
        out.append(ln + EOL + "\n")
    # pad to the bottom so stale lines from a longer previous frame are cleared
    for _ in range(rows - 1 - len(head) - len(body)):
        out.append(EOL + "\n")
    out.append(foot + EOL)
    sys.stdout.write("".join(out)); sys.stdout.flush()

def sig():
    def mt(p):
        try: return os.stat(p).st_mtime
        except OSError: return 0.0
    return (mt(TRANSCRIPT), mt(STATUS), shutil.get_terminal_size())

def main():
    sys.stdout.write(HIDE + CLR)
    last = None
    try:
        while True:
            s = sig()
            if s != last:
                render(s[2].columns, s[2].lines); last = s
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write(SHOW + R + "\n")

if __name__ == "__main__":
    main()
