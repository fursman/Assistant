#!/usr/bin/env python3
"""A Jabra speakerphone's call buttons as the voice assistant's on/off switch.

Green (answer) turns voice mode on, red (hang up) turns it off, mute toggles
it. The Jabra shows the state: its green lights while voice mode is on, blinking
while the assistant hears speech, flashing red while it thinks and answers, the
red ring while it is off -- and that red also mutes its microphone, measured as
digital silence -- and its own idle light while the assistant is not running.

The green blink is the Jabra's incoming-call flash, the Ring LED, lit while the
status file says "listening" -- the phase the indicator shows while someone is
speaking. It makes no sound: the ringtone is a separate output (Ringer) that
this leaves alone, and the Jabra stays in the call throughout.

The red flash has no such LED of its own, so it is timed here: the mute LED on
and off while the assistant thinks or speaks. A Speak 510's Hold LED, lit with
the off-hook one, shows nothing but the steady green.

Why the Jabra is kept "in a call" the whole time: out of a call a Speak 510
reports none of its call buttons. Green does nothing and mute only mutes its own
microphone, unseen by the host. In a call (the host has set the off-hook LED)
all three arrive: green as a Flash pulse, mute as a Phone Mute pulse, red as the
hook switch dropping to on-hook, after which the device leaves the call. So this
holds it off-hook, shows "off" as off-hook and muted, and puts it straight back
into the call after a hang-up. That takes the off-hook LED going out and on
again: relit while still lit, it tells the Jabra nothing, and its hook switch
keeps reading on-hook -- in every later report, green presses included.

The buttons exist only as raw HID reports -- the kernel makes no key events of
them, because Jabra declares the telephony fields constant -- so this reads
/dev/hidraw*, which needs 70-jabra-hidraw.rules (next to this file). Report
layouts come from the device's own descriptor, not a table. Voice mode is
toggled with SIGUSR1, as the hotkey does, and the status file keeps the lights
in step when it is switched some other way.

Pure stdlib, like the Hyprland panel. Installed as ~/.local/bin/voice-assistant-jabra
and run by voice-assistant-jabra.service.
"""
import errno, glob, json, os, re, select, signal, sys, time

STATE   = os.path.expanduser("~/.local/state/voice-assistant")
STATUS  = os.path.join(STATE, "status")
PIDFILE = os.path.join(STATE, "voice-assistant.pid")

HOOK, FLASH, MUTE = 0x000B0020, 0x000B0021, 0x000B002F    # Telephony page inputs
LED_OFFHOOK, LED_MUTE, LED_RING = 0x00080017, 0x00080009, 0x00080018  # LED page outputs
SETTLE = 3.0   # seconds a toggle may take to reach the status file
FLASH_PERIOD = 1.0   # seconds per red flash, on and off, while the assistant has the turn


def log(msg):
    print(msg, flush=True)


def parse_descriptor(desc):
    """Where every one-bit Variable field lives, {(kind, usage): (report_id, bit)}
    with kind "in" or "out", and each output report's length in bytes (ID excluded).

    Constant fields count: Jabra declares its telephony inputs constant, and they
    carry the buttons all the same. Only padding, which has no usages, is skipped.
    """
    fields, bits = {}, {}
    page = rid = size = count = umin = 0
    usages = []
    i = 0
    while i < len(desc):
        b = desc[i]
        if b == 0xFE:                               # long item
            i += 3 + desc[i + 1]
            continue
        n = (0, 1, 2, 4)[b & 3]
        val = int.from_bytes(desc[i + 1:i + 1 + n], "little")
        i += 1 + n
        kind, tag = b >> 2 & 3, b >> 4
        if kind == 1:                               # global
            if tag == 0:
                page = val
            elif tag == 7:
                size = val
            elif tag == 8:
                rid = val
            elif tag == 9:
                count = val
        elif kind == 2:                             # local
            usage = val if n == 4 else page << 16 | val
            if tag == 0:
                usages.append(usage)
            elif tag == 1:
                umin = usage
            elif tag == 2:
                usages.extend(range(umin, usage + 1))
        elif kind == 0:                             # main: Input, Output, Feature, collections
            if tag in (8, 9):
                io = "in" if tag == 8 else "out"
                start = bits.get((io, rid), 0)
                if val & 2 and size == 1:           # Variable: one bit per usage
                    for k, usage in enumerate(usages[:count]):
                        fields.setdefault((io, usage), (rid, start + k))
                bits[(io, rid)] = start + size * count
            usages = []
    return fields, {r: (n + 7) // 8 for (io, r), n in bits.items() if io == "out"}


def hung_up(pressed, before):
    """Red: the hook switch dropping to on-hook. An edge, not a level -- until
    the Jabra is back in the call every report reads on-hook, and reading that
    as red turned a green press straight back off on its release."""
    return bool(before.get(HOOK)) and pressed.get(HOOK) is False


def wanted(pressed, before, on, in_call):
    """The voice mode a report asks for (True/False), or None.

    Green and mute act on the press, not the release; red counts only while the
    device is held in the call.
    """
    down = {u for u, p in pressed.items() if p and not before.get(u)}
    if FLASH in down:
        return True
    if MUTE in down:
        return not on
    if in_call and hung_up(pressed, before):
        return False
    return None


class Jabra:
    def __init__(self, path, name, fields, lengths):
        self.path, self.name, self.fields, self.lengths = path, name, fields, lengths
        self.fd = None
        self.held = None        # (off_hook, muted, ringing) as last written

    def open(self):
        self.fd = os.open(self.path, os.O_RDWR)
        self.held = None

    def close(self):
        if self.fd is None:
            return
        try:
            self.show(False, False)             # hand it back idle
        except OSError:
            pass                                # unplugged
        os.close(self.fd)
        self.fd = None

    def show(self, off_hook, muted, ringing=False):
        reports = {}
        for usage, lit in ((LED_OFFHOOK, off_hook), (LED_MUTE, muted), (LED_RING, ringing)):
            if ("out", usage) not in self.fields:
                continue
            rid, bit = self.fields[("out", usage)]
            buf = reports.setdefault(rid, bytearray([rid]) + bytearray(self.lengths[rid]))
            if lit:
                buf[1 + bit // 8] |= 1 << bit % 8
        for buf in reports.values():
            os.write(self.fd, bytes(buf))
        self.held = (off_hook, muted, ringing)

    def buttons(self, report):
        """{usage: pressed} for the call buttons this report carries."""
        out = {}
        for usage in (HOOK, FLASH, MUTE):
            if ("in", usage) not in self.fields:
                continue
            rid, bit = self.fields[("in", usage)]
            data = report[1:] if rid else report
            if (not rid or report[:1] == bytes([rid])) and len(data) > bit // 8:
                out[usage] = bool(data[bit // 8] >> bit % 8 & 1)
        return out


def find_jabra():
    """A Jabra hidraw node with a hook switch and an off-hook LED, preferring one
    this user may open: the udev rule covers USB, not a Jabra paired over Bluetooth."""
    found = []
    for node in sorted(glob.glob("/sys/class/hidraw/hidraw*")):
        try:
            with open(f"{node}/device/uevent") as f:
                uevent = f.read()
            if not re.search(r"^HID_ID=\w+:0*B0E:", uevent, re.M | re.I):
                continue
            with open(f"{node}/device/report_descriptor", "rb") as f:
                fields, lengths = parse_descriptor(f.read())
        except OSError:
            continue
        if ("in", HOOK) in fields and ("out", LED_OFFHOOK) in fields:
            name = re.search(r"^HID_NAME=(.*)$", uevent, re.M)
            found.append(Jabra("/dev/" + os.path.basename(node),
                               name.group(1) if name else "Jabra", fields, lengths))
    found.sort(key=lambda d: not os.access(d.path, os.R_OK | os.W_OK))
    return found[0] if found else None


def assistant():
    """(pid, phase) for the running assistant, or None. The phase is the status
    file's: off, ready, listening, thinking or speaking."""
    try:
        with open(PIDFILE) as f:
            pid = int(f.read())
        # A stale pid file can name some other process by now, and SIGUSR1
        # would kill it.
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            if b"voice_assistant" not in f.read():
                return None
    except (OSError, ValueError):
        return None
    try:
        with open(STATUS) as f:
            phase = (json.load(f).get("class") or ["off"])[0]
    except (OSError, ValueError):
        phase = "off"
    return pid, phase


def serve(dev):
    """Run one device until it goes away (OSError)."""
    before = {}
    expect = None               # (on, deadline): our toggle, until the status file shows it
    while True:
        state = assistant()
        on = bool(state) and state[1] != "off"
        if expect:
            if state is None or on == expect[0] or time.monotonic() > expect[1]:
                expect = None
            else:
                on = expect[0]
        # Its turn -- thinking or speaking -- flashes the red: the mute LED on
        # and off. That mutes the Jabra's microphone half the time, harmless
        # while the assistant has its own capture muted anyway.
        flash = (on and state[1] in ("thinking", "speaking")
                 and time.monotonic() % FLASH_PERIOD < FLASH_PERIOD / 2)
        want = (True, not on or flash, on and state[1] == "listening") if state else (False, False, False)
        if dev.held != want:
            dev.show(*want)
        if not select.select([dev.fd], [], [], 0.25)[0]:
            continue
        report = os.read(dev.fd, 64)
        if not report:
            raise OSError(errno.ENODEV, os.strerror(errno.ENODEV))
        pressed = dev.buttons(report)
        target = wanted(pressed, before, on, in_call=bool(state))
        if state and hung_up(pressed, before):
            dev.show(False, False)      # out, so the next pass relights it: back in the call
        before.update(pressed)
        # No assistant, nothing to switch: a press then is not an error.
        if state and target is not None and target != on:
            try:
                os.kill(state[0], signal.SIGUSR1)
            except ProcessLookupError:
                continue        # it exited after the pid was read
            expect = (target, time.monotonic() + SETTLE)
            log(f"voice mode {'on' if target else 'off'}")


def main():
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    warned = None
    while True:
        dev = find_jabra()
        if dev is None:
            time.sleep(2)
            continue
        try:
            dev.open()
        except PermissionError:
            if warned != dev.path:
                log(f"no access to {dev.path}: install contrib/jabra/70-jabra-hidraw.rules")
                warned = dev.path
            time.sleep(10)
            continue
        except OSError:
            time.sleep(2)
            continue
        warned = None
        log(f"{dev.name} on {dev.path}: green = on, red = off, mute = toggle")
        try:
            serve(dev)
        except OSError as e:
            log(f"{dev.name} went away ({e.strerror or e})")
        finally:
            dev.close()
        time.sleep(1)


if __name__ == "__main__":
    main()
