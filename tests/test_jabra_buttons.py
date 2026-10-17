"""Jabra call buttons (contrib/jabra/jabra-buttons.py): the report layout read
from a Speak 510's own descriptor, and what the reports it sent in a call do.

    .venv/bin/python -m pytest tests
"""
import importlib.util
import os
import signal
import socket
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "jabra_buttons", ROOT / "contrib" / "jabra" / "jabra-buttons.py")
jb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(jb)

# Jabra SPEAK 510 USB (0b0e:0420): /sys/class/hidraw/hidraw*/device/report_descriptor
SPEAK_510 = bytes.fromhex(
    "050c0901a1018501050c1500250109ea09e909cd09b709b509b609b309b4750195088102"
    "c00600ff0901a10185020901150026ff00750895209202010901150026ff007508952082"
    "020185050620ff1500250109ea09e909e27501950381077501950d8101150025010afbff"
    "0afaff7501950291067501950691010affff75019501b12275019507b10185040630ff15"
    "00250109200997092b092a750195048123092f092109240afdff09507501950581070afc"
    "ff7501950181230906a10219b029bb1500250c750495018140c07501950281010affff15"
    "00250175019501b12275019507b1010640ff150025010917091e0909091809200921092a"
    "7501950791220630ff15002501099e750195019122750195089101c0050b0905a1018503"
    "050b1500250109200997092b092a750195048123092f0921092409507501950481070906"
    "a10219b029bb1500250c750495018140c009070509090115002501750195018102950381"
    "010508150025010917091e0909091809200921092a750195079122050b15002501099e75"
    "0195019122750195089101c0"
)

# What it sent for each button while held in a call, read from /dev/hidraw.
GREEN = [b"\x03\x23\x00", b"\x03\x03\x00"]    # Flash, pressed and released
MUTE = [b"\x03\x13\x00", b"\x03\x03\x00"]     # Phone Mute, pressed and released
RED = [b"\x03\x03\x00", b"\x03\x02\x00"]      # the hook switch going on-hook
# Green straight after a hang-up, before the Jabra is back in the call: the hook
# switch still reads on-hook, in the press and in the release.
GREEN_AFTER_RED = [b"\x03\x22\x00", b"\x03\x02\x00"]


@pytest.fixture
def dev():
    fields, lengths = jb.parse_descriptor(SPEAK_510)
    return jb.Jabra("/dev/null", "Jabra SPEAK 510 USB", fields, lengths)


def asks(dev, reports, on, in_call=True):
    """The voice mode a run of reports asks for, as serve() would read it."""
    before, asked = {}, None
    for report in reports:
        pressed = dev.buttons(report)
        target = jb.wanted(pressed, before, on, in_call)
        before.update(pressed)
        asked = asked if target is None else target
    return asked


def test_layout_comes_from_the_descriptor():
    fields, lengths = jb.parse_descriptor(SPEAK_510)
    # The standard Telephony report, not its vendor-page copy (report 4),
    # though Jabra declares both constant.
    assert fields[("in", jb.HOOK)] == (3, 0)
    assert fields[("in", jb.MUTE)] == (3, 4)
    assert fields[("in", jb.FLASH)] == (3, 5)
    assert fields[("out", jb.LED_OFFHOOK)] == (3, 0)
    assert fields[("out", jb.LED_MUTE)] == (3, 2)
    assert fields[("out", jb.LED_RING)] == (3, 3)
    assert lengths[3] == 2


def test_buttons(dev):
    assert asks(dev, GREEN, on=False) is True
    assert asks(dev, MUTE, on=True) is False
    assert asks(dev, MUTE, on=False) is True
    assert asks(dev, RED, on=True) is False


def test_green_after_a_hang_up_is_not_red_too(dev):
    # Reading on-hook as red, rather than the drop to it, took the release of
    # this press for a hang-up: voice mode on, then straight back off.
    assert asks(dev, RED[-1:] + GREEN_AFTER_RED, on=False) is True


def test_what_is_not_a_press(dev):
    # The off-hook echo the device sends when put in a call.
    assert asks(dev, [b"\x03\x03\x00"], on=False) is None
    # On-hook when it was released, not hung up (the assistant is not running).
    assert asks(dev, [b"\x03\x02\x00"], on=True, in_call=False) is None
    # The vendor-page copy of the telephony report.
    assert asks(dev, [b"\x04\x23\x00"], on=False) is None


def test_lights(dev):
    r, w = os.pipe()
    dev.fd = w
    try:
        dev.show(True, True)
        assert os.read(r, 8) == b"\x03\x05\x00"   # in a call and muted: red ring
        dev.show(True, False)
        assert os.read(r, 8) == b"\x03\x01\x00"   # in a call: green lights
        dev.show(True, False, True)
        assert os.read(r, 8) == b"\x03\x09\x00"   # and ringing: green blinks, silently
        dev.show(False, False)
        assert os.read(r, 8) == b"\x03\x00\x00"   # idle
    finally:
        os.close(r)
        os.close(w)


def test_serve(dev, monkeypatch):
    """Buttons toggle voice mode, the lights follow, a hang-up goes straight
    back into the call (the off-hook LED out, then on), the green blinks while
    the assistant hears speech, the red flashes while it has the turn, and a
    vanished device ends serve()."""
    here, there = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    there.settimeout(2)
    dev.fd = here.fileno()
    voice = {"phase": "off"}
    sent = []

    def kill(pid, sig):             # the assistant toggles and its status file says so
        sent.append((pid, sig))
        voice["phase"] = "ready" if voice["phase"] == "off" else "off"

    monkeypatch.setattr(jb, "assistant", lambda: (4242, voice["phase"]))
    monkeypatch.setattr(jb.os, "kill", kill)
    ended = []

    def run():
        try:
            jb.serve(dev)
        except OSError as e:
            ended.append(e)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    assert there.recv(8) == b"\x03\x05\x00"       # off: in a call, muted
    for steps, lights in ((GREEN, [b"\x03\x01\x00"]),
                          (MUTE, [b"\x03\x05\x00"]),
                          (MUTE, [b"\x03\x01\x00"]),
                          (RED, [b"\x03\x00\x00", b"\x03\x05\x00"]),
                          (GREEN_AFTER_RED, [b"\x03\x01\x00"])):
        for report in steps:
            there.send(report)
        for light in lights:
            assert there.recv(8) == light
    voice["phase"] = "listening"
    assert there.recv(8) == b"\x03\x09\x00"
    for phase in ("thinking", "speaking"):      # the red flashes: mute on, off
        voice["phase"] = phase
        assert {there.recv(8), there.recv(8)} == {b"\x03\x01\x00", b"\x03\x05\x00"}
    voice["phase"] = "ready"
    # Steady green from here. If the flash was in its green half, that is
    # already lit and nothing is written, so wait for the writes to stop
    # rather than for one more: a flash still going writes every half period.
    there.settimeout(jb.FLASH_PERIOD)
    tail = []
    try:
        while len(tail) < 4:
            tail.append(there.recv(8))
    except socket.timeout:
        pass
    assert tail in ([], [b"\x03\x01\x00"]) and dev.held == (True, False, False)
    there.settimeout(2)
    assert sent == [(4242, signal.SIGUSR1)] * 5
    there.close()
    worker.join(2)
    assert ended and not worker.is_alive()


def test_a_press_with_no_assistant_is_ignored(dev, monkeypatch):
    """Green or mute while the assistant is not running signals nothing and
    keeps serving (it used to index the missing pid and crash)."""
    here, there = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    there.settimeout(2)
    dev.fd = here.fileno()
    sent = []
    monkeypatch.setattr(jb, "assistant", lambda: None)
    monkeypatch.setattr(jb.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    ended = []

    def run():
        try:
            jb.serve(dev)
        except BaseException as e:
            ended.append(e)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    assert there.recv(8) == b"\x03\x00\x00"       # no assistant: idle
    for report in GREEN + MUTE:
        there.send(report)
    worker.join(0.6)
    assert worker.is_alive() and not ended and not sent
    there.close()
    worker.join(2)
    assert isinstance(ended[0], OSError)
