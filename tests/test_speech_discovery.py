"""Speech boards that announce themselves (speech_discovery.py, and its use in
voice_assistant.py and remote_speech.Pool): found over mDNS in the background,
added to the board pools, shown by `assistant --status`; a list in the env
file always wins. No real network: a fake mDNS responder on localhost, and
stand-ins for the boards and this machine's engines.

    .venv/bin/python -m pytest tests
"""
import asyncio
import contextlib
import importlib.machinery
import importlib.util
import io
import logging
import socket
import struct
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import remote_speech  # noqa: E402
import speech_discovery as sd  # noqa: E402
import voice_assistant as va  # noqa: E402

va.VOICE_UNLOCK = False


# --- a fake mDNS responder ------------------------------------------------------

def _name(n):
    return b"".join(bytes([len(p)]) + p.encode() for p in n.split(".")) + b"\0"


def _rr(name, rtype, rdata, ttl=120):
    return _name(name) + struct.pack(">HHIH", rtype, 0x8001, ttl, len(rdata)) + rdata


def _ptr(name, target):
    return _rr(name, 12, _name(target))


def _srv(name, port, target):
    return _rr(name, 33, struct.pack(">HHH", 0, 0, port) + _name(target))


def _txt(name, items):
    return _rr(name, 16, b"".join(bytes([len(i)]) + i.encode() for i in items))


def _a(name, address):
    return _rr(name, 1, socket.inet_aton(address))


def _reply(qid, records):
    return struct.pack(">HHHHHH", qid, 0x8400, 0, len(records), 0, 0) + b"".join(records)


class Responder:
    """Answers mDNS queries on 127.0.0.1:<port> from a table of services.
    `split`: the PTR answer alone, SRV/TXT/A only when asked (as a responder
    with a full packet would)."""

    def __init__(self, services, split=False):
        self.services, self.split = services, split
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.port = self.sock.getsockname()[1]
        self.questions = []
        self.stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def close(self):
        self.stop = True
        self.sock.close()

    def _records(self, qname, qtype):
        out = []
        for s in self.services:
            inst = f"{s['instance']}.{s['type']}.local"
            full = [_srv(inst, s["port"], s["host"]), _txt(inst, s["txt"]), _a(s["host"], s["address"])]
            if qtype == 12 and qname == f"{s['type']}.local":
                out += [_ptr(qname, inst)] + ([] if self.split else full)
            elif qtype == 33 and qname == inst:
                out.append(full[0])
            elif qtype == 16 and qname == inst:
                out.append(full[1])
            elif qtype == 1 and qname == s["host"]:
                out.append(full[2])
        return out

    def _run(self):
        while not self.stop:
            try:
                data, src = self.sock.recvfrom(9000)
            except OSError:
                return
            qid, _, qd, _, _, _ = struct.unpack(">HHHHHH", data[:12])
            off, recs = 12, []
            for _ in range(qd):
                qname, off = sd._read_name(data, off)
                qtype = struct.unpack(">H", data[off:off + 2])[0]
                off += 4
                self.questions.append((qname, qtype))
                recs += self._records(qname, qtype)
            if recs:
                self.sock.sendto(_reply(qid, recs), src)


def board(n, kind, address, port, proto=None, **txt):
    t = {"stt": "_speech-stt._tcp", "tts": "_speech-tts._tcp"}[kind]
    label = {"stt": "speech recognition", "tts": "speech synthesis"}[kind]
    items = [f"proto={proto or sd.PROTOCOLS[kind]}", f"board=rk1-{n}", "engine=e", "version=v"] + \
            [f"{k}={v}" for k, v in txt.items()]
    return {"instance": f"rk1-{n} {label}", "type": t, "port": port, "host": f"rk1-{n}.local",
            "address": address, "txt": items}


@pytest.fixture
def lan():
    servers = []

    def start(services, split=False):
        r = Responder(services, split)
        servers.append(r)
        return r
    yield start
    for r in servers:
        r.close()


# --- speech_discovery ------------------------------------------------------------

def test_finds_boards_with_everything_in_one_reply(lan):
    r = lan([board(1, "stt", "10.0.0.11", 8772), board(2, "stt", "10.0.0.12", 9000, capacity=1),
             board(1, "tts", "10.0.0.11", 8771)])
    found = sd.browse("stt", timeout=0.4, targets=["127.0.0.1"], port=r.port)
    assert [(f.name, f.address, f.port) for f in found] == [("rk1-1", "10.0.0.11", 8772),
                                                            ("rk1-2", "10.0.0.12", 9000)]
    assert found[1].spec == "10.0.0.12:9000" and found[1].txt["capacity"] == "1"
    tts = sd.browse("tts", timeout=0.4, targets=["127.0.0.1"], port=r.port)
    assert [f.spec for f in tts] == ["http://10.0.0.11:8771"]


def test_asks_for_what_the_first_reply_left_out(lan):
    r = lan([board(3, "tts", "10.0.0.13", 8771)], split=True)
    found = sd.browse("tts", timeout=0.4, targets=["127.0.0.1"], port=r.port)
    assert [(f.name, f.spec) for f in found] == [("rk1-3", "http://10.0.0.13:8771")]
    asked = {t for _, t in r.questions}
    assert {12, 33, 16, 1} <= asked, "PTR, then SRV and TXT, then A"


def test_a_server_speaking_another_protocol_is_left_out(lan):
    r = lan([board(1, "stt", "10.0.0.11", 8772, proto="moonshine-tcp/2"),
             board(2, "stt", "10.0.0.12", 8772)])
    found = sd.browse("stt", timeout=0.4, targets=["127.0.0.1"], port=r.port)
    assert [f.name for f in found] == ["rk1-2"]


def test_nothing_answering_is_an_empty_list(lan):
    r = lan([])
    assert sd.browse("stt", timeout=0.3, targets=["127.0.0.1"], port=r.port) == []


def test_parse_follows_compressed_names():
    head = struct.pack(">HHHHHH", 7, 0x8400, 0, 1, 0, 0)
    name = _name("_speech-stt._tcp.local")
    # PTR whose target points back at the owner name (offset 12) behind one label.
    rdata = bytes([5]) + b"rk1-9" + b"\xc0\x0c"
    rec = name + struct.pack(">HHIH", 12, 1, 120, len(rdata)) + rdata
    qid, is_reply, recs = sd.parse(head + rec)
    assert qid == 7 and is_reply
    assert recs == [("_speech-stt._tcp.local", 12, 120, "rk1-9._speech-stt._tcp.local")]


# --- remote_speech.Pool.add ----------------------------------------------------------

class Up:
    def __init__(self, name=None, probe=None, up=True): self._up = up
    def up(self): return self._up
    def mark_down(self, why): self._up = False


class Srv:
    def __init__(self, host, port=8772):
        self.host, self.port, self.name = host, port, host

    def probe(self): return True


def test_a_pool_takes_new_servers_once_each():
    pool = remote_speech.Pool("stt", [Srv("a")], [Up()])
    assert pool.add(Srv("b"), Up())
    assert not pool.add(Srv("b"), Up()), "already there"
    assert pool.add(Srv("b", 9000), Up(up=False)), "another port is another server"
    assert [s.host for _, s, _ in pool.order()] == ["a", "b"], "offered once it answers"
    pool.acquire(1)
    pool.release(1)
    assert pool.has("b", 9000) and not pool.has("c", 8772)


def test_an_empty_pool_falls_back_to_this_machine():
    class Here:
        def __init__(self): self.calls = []
        def begin_utterance(self): self.calls.append("begin")
        def finish(self): return "here"
    stt = remote_speech.FallbackStt(Here(), [], lambda: True, health=[])
    stt.begin_utterance()
    assert stt.finish() == "here" and stt.remote is None


# --- the assistant ---------------------------------------------------------------------

class Board:
    """A board's recogniser or synthesiser, recording what it did."""
    log = []

    def __init__(self, spec, *a, **k):
        self.spec = spec
        rest = spec.split("://")[-1]
        self.host, _, port = rest.rpartition(":")
        self.port, self.name = int(port), self.host

    def begin(self): self.log.append(("begin", self.spec))
    def feed(self, chunk, rate): pass
    def partial(self): return ""
    def finish(self): return f"heard on {self.host}"
    def close(self): pass
    def probe(self): return True

    def stream(self, text, voice=None, speed=None):
        self.log.append(("say", self.spec))
        yield np.ones(10, np.float32), 24000


class Here:
    def __init__(self): self.calls = []
    def begin_utterance(self): self.calls.append("begin")
    def feed(self, chunk, rate): pass
    def partial(self): return ""
    def finish(self): return "heard here"
    def transcribe(self, audio, **k): return [remote_speech._Segment("here")], None

    def stream(self, text, voice=None, speed=None):
        self.calls.append(("say", text))
        yield np.zeros(10, np.float32), 24000


class Host(va.VoiceAssistant):
    """The speech wiring alone: no audio, no models."""
    def __init__(self, state_dir, mode="network"):
        self.logger = logging.getLogger("test-speech-discovery")
        self.state_dir = state_dir
        self.backend, self.llm_tier, self.backend_pref = "claude", "local", "auto"
        self._status_state = "ready"
        self.is_active = self.is_processing = False
        self._dsh, self._dsh_available, self._claude_available = None, False, True
        self.voice_only, self._voiceprint, self._session_id = False, None, "s"
        self.speech_mode = mode
        self.notes, self.transcript = [], []
        self.stt, self.kokoro, self.tts_available = Here(), Here(), True

    def _append_transcript(self, role, text): self.transcript.append((role, text))
    def _notify(self, text, **k): self.notes.append((text, k))


def found(kind, n, address, port):
    return sd.Found(kind, f"rk1-{n}", address, port, {"proto": sd.PROTOCOLS[kind], "engine": "e",
                                                       "version": "v"})


@pytest.fixture
def discovering(monkeypatch):
    """No lists in the env file, discovery on; what the next scan finds, and the kinds it was asked for."""
    lan = {"found": {"stt": [], "tts": []}, "asked": []}

    def scan(kinds=("stt", "tts"), *a, **k):
        lan["asked"].append(list(kinds))
        return {k: list(lan["found"].get(k, [])) for k in kinds}
    monkeypatch.setattr(va, "REMOTE_STT", "")
    monkeypatch.setattr(va, "REMOTE_TTS", "")
    monkeypatch.setattr(va, "SPEECH_DISCOVER_INTERVAL", 300.0)
    monkeypatch.setattr(va, "REMOTE_SPEECH_TIERS", {"network", "claude"})
    monkeypatch.setattr(va, "_SPEECH_FOUND", {"stt": [], "tts": []})
    monkeypatch.setattr(va, "_SPEECH_SCANNED_AT", 0.0)
    monkeypatch.setattr(va, "_write_env_setting", lambda key, value: None)
    monkeypatch.setattr(va, "_local_llm_health", lambda timeout=1.0: "down")
    monkeypatch.setattr(va, "NETWORK_LLM_URL", "")
    monkeypatch.setattr(va.speech_discovery, "scan", scan)
    monkeypatch.setattr(remote_speech, "Health", Up)
    monkeypatch.setattr(remote_speech, "RemoteStt", Board)
    monkeypatch.setattr(remote_speech, "RemoteTts", Board)
    Board.log = []
    return lan


def test_boards_found_on_the_lan_are_used_without_any_list(tmp_path, discovering):
    host = Host(tmp_path)
    here_stt, here_tts = host.stt, host.kokoro
    host._setup_remote_speech()
    host.stt.begin_utterance()
    assert host.stt.finish() == "heard here", "nothing found yet: this machine"
    discovering["found"] = {"stt": [found("stt", 1, "10.0.0.11", 8772)],
                            "tts": [found("tts", 1, "10.0.0.11", 8771), found("tts", 2, "10.0.0.12", 8771)]}
    assert host._discover_speech_servers()
    host.stt.begin_utterance()
    assert host.stt.finish() == "heard on 10.0.0.11"
    list(host.kokoro.stream("Hi."))
    assert ("say", "http://10.0.0.11:8771") in Board.log or ("say", "http://10.0.0.12:8771") in Board.log
    assert here_stt.calls == ["begin"] and here_tts.calls == []
    assert host.stt.pool.servers[0].name == "rk1-1", "logged by its board name"
    # The next scan finds the same boards: nothing is added twice.
    host._discover_speech_servers()
    assert len(host.stt.pool.servers) == 1 and len(host.kokoro.pool.servers) == 2


def test_a_list_in_the_env_file_wins_for_its_kind(tmp_path, discovering, monkeypatch):
    monkeypatch.setattr(va, "REMOTE_STT", "10.0.0.50:8772")
    host = Host(tmp_path)
    host._setup_remote_speech()
    discovering["found"] = {"stt": [found("stt", 1, "10.0.0.11", 8772)],
                            "tts": [found("tts", 1, "10.0.0.11", 8771)]}
    host._discover_speech_servers()
    assert discovering["asked"] == [["tts"]], "the listed kind is not even looked for"
    assert [s.spec for s in host.stt.pool.servers] == ["10.0.0.50:8772"]
    assert [s.spec for s in host.kokoro.pool.servers] == ["http://10.0.0.11:8771"]
    monkeypatch.setattr(va, "REMOTE_TTS", "http://10.0.0.50:8771")
    discovering["asked"].clear()
    assert not host._discover_speech_servers() and discovering["asked"] == [], "both listed: no scan"


def test_discovery_off_and_no_lists_means_this_machine(tmp_path, discovering, monkeypatch):
    monkeypatch.setattr(va, "SPEECH_DISCOVER_INTERVAL", 0.0)
    host = Host(tmp_path, mode="network")
    here = host.stt
    host._setup_remote_speech()
    assert host.stt is here, "not wrapped at all"
    assert va._initial_speech_mode() == "local"
    assert host.set_speech_mode("network")["label"] == "Speech: local (no network speech configured)"


def test_network_speech_can_be_chosen_before_a_board_is_found(tmp_path, discovering, monkeypatch):
    monkeypatch.setattr(va, "SPEECH_MODE", "network")
    assert va._initial_speech_mode() == "network", "a board may turn up after start"
    host = Host(tmp_path, mode="local")
    host._speech_rescan = threading.Event()
    res = host.set_speech_mode("network")
    assert res["speech"] == "network" and res["label"] == "Speech: network (no boards found yet)"
    assert host._speech_rescan.is_set(), "looks again at once"


def test_status_says_where_the_servers_came_from(tmp_path, discovering):
    host = Host(tmp_path)
    host._setup_remote_speech()
    r = asyncio.run(host._control_command({"cmd": "status"}))
    assert r["speech_discovery"] == {"stt": "mdns", "tts": "mdns", "scan": "pending"}
    assert r["speech_servers"] == {"stt": [], "tts": []} and r["speech_boards"] == []
    discovering["found"] = {"stt": [found("stt", 1, "10.0.0.11", 8772)],
                            "tts": [found("tts", 1, "10.0.0.11", 8771)]}
    host._discover_speech_servers()
    r = asyncio.run(host._control_command({"cmd": "status"}))
    assert r["speech_discovery"]["scan"] in (0, 1)
    assert r["speech_servers"] == {"stt": ["10.0.0.11:8772"], "tts": ["http://10.0.0.11:8771"]}
    assert r["speech_boards"] == [{"address": "10.0.0.11", "name": "rk1-1",
                                   "stt": "10.0.0.11:8772", "stt_up": True, "stt_source": "mdns",
                                   "tts": "10.0.0.11:8771", "tts_up": True, "tts_source": "mdns"}]


# --- `assistant --status` ----------------------------------------------------------------

def _cli():
    loader = importlib.machinery.SourceFileLoader("assistant_cli", str(ROOT / "assistant"))
    spec = importlib.util.spec_from_loader("assistant_cli", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


STATUS = {"ok": True, "label": "Claude", "preference": "auto", "local_llm": "down",
          "network_llm": None, "dsh": "installed", "voice_active": True, "busy": False,
          "voice_only": False, "session": "s", "speech_tiers": ["claude", "network"]}


def _status(monkeypatch, reply):
    cli = _cli()
    monkeypatch.setattr(cli, "request", lambda payload: reply)
    monkeypatch.setattr(sys, "argv", ["assistant", "--status"])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert cli.main() == 0
    return out.getvalue()


def test_cli_status_lists_the_boards_found(monkeypatch):
    out = _status(monkeypatch, dict(
        STATUS, speech="network",
        speech_servers={"stt": ["10.0.0.11:8772", "10.0.0.12:8772"], "tts": ["http://10.0.0.11:8771"]},
        speech_discovery={"stt": "mdns", "tts": "mdns", "scan": 130},
        speech_boards=[
            {"address": "10.0.0.11", "name": "rk1-1", "stt": "10.0.0.11:8772", "stt_up": True,
             "stt_source": "mdns", "tts": "10.0.0.11:8771", "tts_up": False, "tts_source": "mdns"},
            {"address": "10.0.0.12", "name": "rk1-2", "stt": "10.0.0.12:8772", "stt_up": True,
             "stt_source": "mdns"}]))
    assert "speech     network  (stt 2 found on the LAN; tts 1 found on the LAN; claude/network tiers)\n" in out
    assert "           rk1-1      10.0.0.11       stt up  tts down\n" in out
    assert "           rk1-2      10.0.0.12       stt up\n" in out
    assert "           (looked 2 min ago)\n" in out


def test_cli_status_before_any_board_is_found(monkeypatch):
    reply = dict(STATUS, speech_servers={"stt": [], "tts": []}, speech_boards=[],
                 speech_discovery={"stt": "mdns", "tts": "mdns", "scan": "pending"})
    out = _status(monkeypatch, dict(reply, speech="network"))
    assert "speech     network  (stt none found yet; tts none found yet; claude/network tiers)\n" in out
    out = _status(monkeypatch, dict(reply, speech="local"))
    assert "speech     local  (no speech boards found on the LAN yet)\n" in out


def test_cli_status_with_lists_is_as_before(monkeypatch):
    out = _status(monkeypatch, dict(
        STATUS, speech="network", speech_servers={"stt": ["a:8772"], "tts": ["http://a:8771"]},
        speech_discovery={"stt": "configured", "tts": "configured", "scan": "off"},
        speech_boards=[{"address": "a", "name": "", "stt": "a:8772", "stt_up": True, "stt_source": "configured",
                        "tts": "a:8771", "tts_up": True, "tts_source": "configured"}]))
    assert "speech     network  (stt a:8772; tts http://a:8771; claude/network tiers)\n" in out
    assert "found" not in out and "looked" not in out


def test_a_malformed_address_record_is_ignored(lan):
    """An A record of the wrong length: no server at a garbage address."""
    r = lan([board(1, "stt", "10.0.0.11", 8772)])
    orig = r._records

    def records(qname, qtype):
        out = orig(qname, qtype)
        return [_rr("rk1-1.local", 1, b"\x0a\x3a\x00") if rec.startswith(_name("rk1-1.local")) else rec
                for rec in out]
    r._records = records
    assert sd.browse("stt", timeout=0.4, targets=["127.0.0.1"], port=r.port) == []


def test_a_bad_find_does_not_end_discovery(tmp_path, discovering, monkeypatch):
    host = Host(tmp_path)
    host._setup_remote_speech()
    discovering["found"] = {"stt": [found("stt", 1, "10.0.0.11", 8772)], "tts": []}

    def boom(*a, **k):
        raise ValueError("bad spec")
    monkeypatch.setattr(remote_speech, "RemoteStt", boom)
    assert host._discover_speech_servers() is True, "logged, and the thread carries on"


def test_no_synthesis_pool_means_no_synthesis_scan(tmp_path, discovering):
    host = Host(tmp_path)
    host.tts_available = False          # this machine has no synthesiser to fall back to
    host._setup_remote_speech()
    host._discover_speech_servers()
    assert discovering["asked"] == [["stt"]]
