"""Speech servers on the LAN: boards that announce themselves over mDNS.

rk1-speech (github.com/fursman/rk1-speech) announces each board's recogniser
as `_speech-stt._tcp` and its synthesiser as `_speech-tts._tcp`, with TXT
records naming the protocol. voice_assistant.py looks for them in the
background, as it looks for model servers (llm_discovery.py), and hands what
it finds to the board pools in remote_speech.py: a kind of server listed in
VOICE_ASSISTANT_REMOTE_STT / _TTS is never looked for, so a list in the env
file always wins.

Only servers speaking a protocol this assistant knows are taken:

    _speech-stt._tcp  proto=moonshine-tcp/1   moonshine-rknn's framed TCP (RemoteStt)
    _speech-tts._tcp  proto=kokoro-http/1     kokoro-rknn's POST /tts (RemoteTts)

Standard library only, and no avahi needed here: one-shot mDNS queries (RFC
6762 section 5.1) sent from an ephemeral port, which responders answer by
unicast, so there is no multicast group to join and no port 5353 to share with
a local avahi-daemon. About a second for both kinds; nothing here blocks a turn.

    python3 speech_discovery.py          one '<kind> <name> <address>:<port> <engine> <version>' line each
"""
from __future__ import annotations

import random
import select
import socket
import struct
import sys
import threading
import time
from typing import NamedTuple

MDNS_GROUP = "224.0.0.251"
MDNS_PORT = 5353
TYPES = {"stt": "_speech-stt._tcp", "tts": "_speech-tts._tcp"}
PROTOCOLS = {"stt": "moonshine-tcp/1", "tts": "kokoro-http/1"}
TIMEOUT = 1.0
_A, _PTR, _TXT, _SRV = 1, 12, 16, 33


class Found(NamedTuple):
    kind: str           # "stt" | "tts"
    name: str           # the board's name (TXT board=, else its host name)
    address: str        # IPv4
    port: int
    txt: dict           # every TXT key=value (engine, version, capacity, npu_cores, ...)

    @property
    def spec(self) -> str:
        """What VOICE_ASSISTANT_REMOTE_STT / _TTS would say for it."""
        return f"{self.address}:{self.port}" if self.kind == "stt" else f"http://{self.address}:{self.port}"


# -- DNS wire format ----------------------------------------------------------

def _name(name: str) -> bytes:
    return b"".join(bytes([len(p.encode())]) + p.encode() for p in name.rstrip(".").split(".")) + b"\0"


def query(questions, qid: int = 0) -> bytes:
    """One packet asking every (name, qtype) in `questions`."""
    return struct.pack(">HHHHHH", qid, 0, len(questions), 0, 0, 0) + b"".join(
        _name(n) + struct.pack(">HH", t, 1) for n, t in questions)


def _read_name(data: bytes, off: int):
    labels, end, hops = [], None, 0
    while True:
        n = data[off]
        if n & 0xC0 == 0xC0:
            end = off + 2 if end is None else end
            off = ((n & 0x3F) << 8) | data[off + 1]
            hops += 1
            if hops > 64:
                raise ValueError("pointer loop")
            continue
        if n == 0:
            return ".".join(labels), (off + 1 if end is None else end)
        labels.append(data[off + 1:off + 1 + n].decode("utf-8", "replace"))
        off += 1 + n


def parse(data: bytes):
    """(id, is_response, [(name, type, ttl, value)]) for every record of a reply.
    value: PTR -> name, SRV -> (port, target), TXT -> [bytes], A -> address."""
    qid, flags, qd, an, ns, ar = struct.unpack(">HHHHHH", data[:12])
    off, out = 12, []
    for _ in range(qd):
        off = _read_name(data, off)[1] + 4
    for _ in range(an + ns + ar):
        name, off = _read_name(data, off)
        rtype, _cls, ttl, size = struct.unpack(">HHIH", data[off:off + 10])
        off += 10
        rd = data[off:off + size]
        if len(rd) != size:
            raise ValueError("truncated record")
        value = rd
        if rtype == _PTR:
            value = _read_name(data, off)[0]
        elif rtype == _SRV:
            value = (struct.unpack(">H", rd[4:6])[0], _read_name(data, off + 6)[0])
        elif rtype == _TXT:
            value, i = [], 0
            while i < size:
                value.append(rd[i + 1:i + 1 + rd[i]])
                i += 1 + rd[i]
        elif rtype == _A and size == 4:
            value = socket.inet_ntoa(rd)
        out.append((name.lower().rstrip("."), rtype, ttl, value))
        off += size
    return qid, bool(flags & 0x8000), out


def txt_dict(strings) -> dict:
    out = {}
    for s in strings or ():
        key, _, value = s.decode("utf-8", "replace").partition("=")
        if key:
            out.setdefault(key.lower(), value)
    return out


# -- asking -------------------------------------------------------------------

def _interface_address() -> str:
    """The address this machine reaches the LAN from ("" without a route)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))       # TEST-NET-1: connect() sends nothing
            return s.getsockname()[0]
    except OSError:
        return ""


def exchange(questions, timeout: float = TIMEOUT, targets=(), port: int = MDNS_PORT) -> list:
    """Every record of every reply within `timeout` (the query goes out twice,
    0.25 s apart: mDNS loses packets). `targets` asks those hosts by unicast
    instead of the multicast group."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 255)
        iface = _interface_address()
        if iface:
            try:
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(iface))
            except OSError:
                pass
        sock.bind(("0.0.0.0", 0))
        qid = random.randint(1, 0xFFFF)
        packet = query(questions, qid)
        dests = [(t, port) for t in targets] or [(MDNS_GROUP, port)]
        start = time.monotonic()
        sends, records = [start, start + 0.25], []
        while True:
            now = time.monotonic()
            while sends and sends[0] <= now:
                sends.pop(0)
                for d in dests:
                    try:
                        sock.sendto(packet, d)
                    except OSError:
                        pass
            left = start + timeout - now
            if left <= 0:
                return records
            ready = select.select([sock], [], [], max(0.0, min(left, sends[0] - now) if sends else left))[0]
            if ready:
                try:
                    rid, is_reply, recs = parse(sock.recv(9000))
                except (OSError, ValueError, struct.error, IndexError):
                    continue
                if is_reply and rid in (qid, 0):
                    records += recs
    finally:
        sock.close()


def browse(kind: str, timeout: float = TIMEOUT, targets=(), port: int = MDNS_PORT) -> list:
    """The servers of one kind that answer, as Found, sorted by name and address.
    A server announcing another protocol is left out. Never raises for network
    trouble: no answer is an empty list."""
    fqtype = f"{TYPES[kind]}.local"
    ptr, srv, txt, addr = set(), {}, {}, {}

    def absorb(records):
        for name, rtype, ttl, value in records:
            if rtype == _PTR and name == fqtype and ttl > 0:
                ptr.add(value.lower().rstrip("."))
            elif rtype == _SRV:
                srv[name] = (value[0], value[1].lower().rstrip("."))
            elif rtype == _TXT:
                txt[name] = txt_dict(value)
            elif rtype == _A and isinstance(value, str):      # a malformed A record stays bytes
                addr.setdefault(name, value)
    try:
        absorb(exchange([(fqtype, _PTR)], timeout, targets, port))
        missing = [(i, t) for i in ptr for t, have in ((_SRV, srv), (_TXT, txt)) if i not in have]
        if missing:
            absorb(exchange(missing, timeout / 2, targets, port))
        hosts = [(srv[i][1], _A) for i in ptr if i in srv and srv[i][1] not in addr]
        if hosts:
            absorb(exchange(hosts, timeout / 2, targets, port))
    except (OSError, ValueError):
        return []
    out = []
    for inst in ptr:
        if inst not in srv or srv[inst][1] not in addr:
            continue
        t = txt.get(inst, {})
        if t.get("proto") != PROTOCOLS[kind]:
            continue
        name = t.get("board") or srv[inst][1].split(".")[0]
        out.append(Found(kind, name, addr[srv[inst][1]], srv[inst][0], t))
    return sorted(out, key=lambda f: (f.name, f.address, f.port))


def scan(kinds=("stt", "tts"), timeout: float = TIMEOUT, targets=()) -> dict:
    """{kind: [Found]} for each kind asked, both looked for at once."""
    found = {k: [] for k in kinds}

    def look(kind):
        found[kind] = browse(kind, timeout, targets)
    threads = [threading.Thread(target=look, args=(k,), daemon=True) for k in kinds]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout * 3)
    return found


def main(argv=None) -> int:
    targets = list(argv if argv is not None else sys.argv[1:])
    for kind, servers in scan(targets=targets).items():
        for f in servers:
            print(f"{kind} {f.name} {f.address}:{f.port} {f.txt.get('engine', '-')} "
                  f"{f.txt.get('version', '-')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
