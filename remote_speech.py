"""Speech on another machine, with this one's engines as the fallback.

The network and Claude tiers can hand speech recognition and synthesis to
servers on the LAN, one or a pool of several (the Turing Pi's RK1 boards: Moonshine on one, Kokoro with
its decoder on the NPU on another), so this machine's CPU is left alone while
you talk and while it answers. The local tier never leaves the machine: it is
the one that has to work with the network down.

Both wrappers present the local engine's own interface, so voice_assistant.py
does not learn which one ran:

    stt: begin_utterance() / feed() / partial() / finish() / transcribe()
    tts: create() / stream()

and both fall back to the local engine whenever the server is not known to be
up, refuses a connection, or goes quiet. A server that fails is marked down at
once and probed in the background; the next turn uses it again as soon as it
answers. Nothing here ever waits on a down server for longer than its connect
timeout.

Standard library only: plain TCP frames for the recogniser (moonshine-rknn's
server), chunked HTTP for the synthesiser (kokoro-rknn's). The servers come
from VOICE_ASSISTANT_REMOTE_STT / _TTS, or from their mDNS announcements
(speech_discovery.py), added to a pool as they are found.
"""
from __future__ import annotations

import collections
import http.client
import json
import logging
import queue
import socket
import struct
import threading
import time
import urllib.parse

import numpy as np

log = logging.getLogger("voice-assistant")

HDR = struct.Struct(">cI")
CONNECT_TIMEOUT = 0.5       # seconds; a LAN server answers in a few ms
PROBE_UP = 15.0             # seconds between health checks while it answers
PROBE_DOWN = 5.0            # ... and while it does not


class Health:
    """Whether a server answered its last health check, kept fresh by a
    daemon thread so asking costs nothing on the audio path."""

    def __init__(self, name: str, probe):
        self.name = name
        self._probe = probe
        self._up = False
        self._checked = threading.Event()
        self._wake = threading.Event()
        threading.Thread(target=self._run, name=f"health-{name}", daemon=True).start()

    def up(self) -> bool:
        return self._up

    def wait_first(self, timeout: float) -> bool:
        """Block until the first check is in (start-up only)."""
        self._checked.wait(timeout)
        return self._up

    def mark_down(self, why: str):
        if self._up:
            log.warning(f"{self.name} speech server down ({why}); using this machine's")
        self._up = False
        self._wake.set()

    def _run(self):
        while True:
            try:
                ok = bool(self._probe())
            except Exception:
                ok = False
            if ok != self._up:
                log.info(f"{self.name} speech server {'up' if ok else 'down'}")
            self._up = ok
            self._checked.set()
            self._wake.wait(PROBE_UP if ok else PROBE_DOWN)
            self._wake.clear()


def _host_port(spec: str, default_port: int):
    if "://" in spec:
        u = urllib.parse.urlsplit(spec)
        return u.hostname, u.port or default_port
    host, _, port = spec.rpartition(":")
    if not host:
        return spec, default_port
    return host, int(port)


def _send(sock, kind: bytes, payload: bytes = b""):
    sock.sendall(HDR.pack(kind, len(payload)) + payload)


def _recv_exact(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        part = sock.recv(n - len(buf))
        if not part:
            raise ConnectionError("server closed the connection")
        buf += part
    return bytes(buf)


def _recv(sock):
    kind, size = HDR.unpack(_recv_exact(sock, HDR.size))
    return kind, _recv_exact(sock, size) if size else b""


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------

class RemoteStt:
    """One streamed utterance at a time against an rk1-stt server."""

    def __init__(self, spec: str, rate: int, finish_timeout: float = 4.0,
                 batch_timeout: float = 15.0):
        self.host, self.port = _host_port(spec, 8772)
        self.name = self.host
        self.rate = rate
        self.finish_timeout = finish_timeout
        self.batch_timeout = batch_timeout
        self._sock = None
        self._out = None
        self._partial = ""
        self._final = None
        self._done = threading.Event()
        self._error = None

    def _connect(self):
        sock = socket.create_connection((self.host, self.port), timeout=CONNECT_TIMEOUT)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return sock

    def probe(self) -> bool:
        with self._connect() as sock:
            sock.settimeout(CONNECT_TIMEOUT * 2)
            _send(sock, b"H")
            kind, payload = _recv(sock)
            return kind == b"O" and json.loads(payload).get("ok", False)

    # -- streaming --
    def begin(self):
        self.close()
        sock = self._connect()
        sock.settimeout(None)
        _send(sock, b"B", json.dumps({"rate": self.rate}).encode())
        self._sock = sock
        self._partial, self._final, self._error = "", None, None
        self._done = threading.Event()
        # The record loop must never block on the network: frames go out
        # from their own thread, replies come in on another.
        self._out = queue.Queue()
        threading.Thread(target=self._writer, args=(sock, self._out), name="stt-send",
                         daemon=True).start()
        threading.Thread(target=self._reader, args=(sock, self._done), name="stt-recv",
                         daemon=True).start()

    def _writer(self, sock, out):
        while True:
            item = out.get()
            if item is None:
                return
            try:
                _send(sock, *item)
            except OSError as e:
                self._error = self._error or f"send failed: {e}"
                self._done.set()
                return

    def _reader(self, sock, done):
        try:
            while True:
                kind, payload = _recv(sock)
                if kind == b"P":
                    self._partial = payload.decode(errors="replace")
                elif kind == b"F":
                    self._final = payload.decode(errors="replace")
                    return
                elif kind == b"X":
                    self._error = payload.decode(errors="replace") or "server error"
                    return
        except (OSError, ConnectionError, struct.error) as e:
            if self._final is None:
                self._error = self._error or f"connection lost: {e}"
        finally:
            done.set()

    def feed(self, chunk, sample_rate):
        out = self._out
        if out is None or self._done.is_set():
            return
        out.put((b"A", np.asarray(chunk, dtype="<f4").tobytes()))

    def partial(self) -> str:
        return self._partial

    def finish(self):
        """The final transcript, or None when the server did not deliver one
        (the caller then transcribes the recording locally)."""
        out = self._out
        if out is None:
            return None
        out.put((b"E", b""))
        out.put(None)
        self._done.wait(self.finish_timeout)
        final, error = self._final, self._error
        self.close()
        if final is None:
            raise ConnectionError(error or f"no transcript within {self.finish_timeout:.0f}s")
        return final.strip()

    def close(self):
        out, self._out = self._out, None
        if out is not None:
            out.put(None)
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    # -- batch --
    def batch(self, audio) -> str:
        with self._connect() as sock:
            sock.settimeout(self.batch_timeout)
            _send(sock, b"T", np.asarray(audio, dtype="<f4").tobytes())
            kind, payload = _recv(sock)
            if kind != b"F":
                raise ConnectionError(payload.decode(errors="replace") or "server error")
            return payload.decode(errors="replace").strip()


class _Segment:
    __slots__ = ("text",)

    def __init__(self, text):
        self.text = text


# Requests from this machine in flight, by kind and host. Shared by the
# recogniser and synthesiser pools: a board doing both at once shares its CPU
# between them (measured: recognition unaffected, synthesis ~0.25 s slower to
# start), so a synthesiser prefers a board that is not transcribing right now.
_IN_FLIGHT = {"stt": collections.Counter(), "tts": collections.Counter()}
_IN_FLIGHT_LOCK = threading.Lock()


class Pool:
    """The servers that can do one kind of job, in the order to try them.

    First the ones answering their health checks; among those, ones with no
    request of this kind in flight before busy ones, hosts in `avoid` last,
    and otherwise the least recently used first, so the boards take turns.

    Servers found on the LAN later join with add(); none ever leaves, so a
    server's index stays valid while a request holds it (one that stops
    answering is simply not offered)."""

    def __init__(self, kind: str, servers, health=None):
        self.kind = kind
        self.servers = list(servers)
        if health is None:
            health = [Health(srv.name, srv.probe) for srv in self.servers]
        elif not isinstance(health, (list, tuple)):
            health = [health]
        self.health = list(health)
        self._used = [0.0] * len(self.servers)
        self._add_lock = threading.Lock()

    def has(self, host: str, port: int) -> bool:
        return any(s.host == host and s.port == port for s in list(self.servers))

    def add(self, server, health=None) -> bool:
        """Take another server (one discovery found); False when it is already
        here. Its health is checked from now on, and it is offered once it
        answers. Appended in the order order() relies on: a health entry
        never exists before its server."""
        with self._add_lock:
            if self.has(server.host, server.port):
                return False
            self.servers.append(server)
            self._used.append(0.0)
            self.health.append(health if health is not None else Health(server.name, server.probe))
            return True

    def order(self, avoid=()):
        with _IN_FLIGHT_LOCK:
            busy = dict(_IN_FLIGHT[self.kind])
        up = [i for i, h in enumerate(list(self.health)) if h.up()]
        up.sort(key=lambda i: (busy.get(self.servers[i].host, 0),
                               self.servers[i].host in avoid, self._used[i]))
        return [(i, self.servers[i], self.health[i]) for i in up]

    def acquire(self, i):
        self._used[i] = time.monotonic()
        with _IN_FLIGHT_LOCK:
            _IN_FLIGHT[self.kind][self.servers[i].host] += 1

    def release(self, i):
        with _IN_FLIGHT_LOCK:
            c = _IN_FLIGHT[self.kind]
            c[self.servers[i].host] -= 1
            if c[self.servers[i].host] <= 0:
                del c[self.servers[i].host]


def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


class FallbackStt:
    """The local recogniser's interface; a server from the pool does the work
    when `use_remote()` says this turn may leave the machine. A server that
    refuses is marked down and the next one tried; with none left, or none
    allowed, this machine's engine does it."""

    def __init__(self, local, remote, use_remote, health=None):
        self.local = local
        self.pool = Pool("stt", _as_list(remote), health)
        self.use_remote = use_remote
        self.is_streaming = True
        self._active = None            # pool index of this utterance's server

    @property
    def remote(self):
        if self._active is not None:
            return self.pool.servers[self._active]
        return self.pool.servers[0] if self.pool.servers else None

    def _candidates(self):
        return self.pool.order() if self.use_remote() else []

    def begin_utterance(self):
        self._drop_active()
        for i, server, health in self._candidates():
            try:
                server.begin()
            except OSError as e:
                health.mark_down(str(e))
                continue
            self._active = i
            self.pool.acquire(i)
            return
        begin = getattr(self.local, "begin_utterance", None)
        if begin is not None:
            begin()

    def _drop_active(self):
        if self._active is not None:
            self.pool.servers[self._active].close()
            self.pool.release(self._active)
            self._active = None

    def feed(self, chunk, sample_rate):
        if self._active is not None:
            self.pool.servers[self._active].feed(chunk, sample_rate)
            return
        feed = getattr(self.local, "feed", None)
        if feed is not None:
            feed(chunk, sample_rate)

    def partial(self) -> str:
        if self._active is not None:
            return self.pool.servers[self._active].partial()
        partial = getattr(self.local, "partial", None)
        return partial() if partial is not None else ""

    def finish(self):
        if self._active is not None:
            i, self._active = self._active, None
            try:
                return self.pool.servers[i].finish()
            except Exception as e:
                # None sends the caller to transcribe(), which goes to
                # another server, or runs here.
                self.pool.health[i].mark_down(str(e))
                return None
            finally:
                self.pool.release(i)
        finish = getattr(self.local, "finish", None)
        return finish() if finish is not None else None

    def transcribe(self, audio, **kwargs):
        for i, server, health in self._candidates():
            self.pool.acquire(i)
            try:
                return [_Segment(server.batch(audio))], None
            except Exception as e:
                health.mark_down(str(e))
            finally:
                self.pool.release(i)
        return self.local.transcribe(audio, **kwargs)

    def close(self):
        self._drop_active()
        for server in self.pool.servers:
            server.close()
        close = getattr(self.local, "close", None)
        if close is not None:
            close()


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

class RemoteTts:
    """Streamed synthesis from an rk1-tts server (POST /tts, float32 chunks)."""

    def __init__(self, url: str, voice: str = "", read_timeout: float = 5.0):
        u = urllib.parse.urlsplit(url if "://" in url else f"http://{url}")
        self.host, self.port = u.hostname, u.port or 8771
        self.name = self.host
        self.voice = voice
        self.read_timeout = read_timeout

    def probe(self) -> bool:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=CONNECT_TIMEOUT * 2)
        try:
            conn.request("GET", "/health")
            resp = conn.getresponse()
            return resp.status == 200 and json.loads(resp.read()).get("ok", False)
        finally:
            conn.close()

    def stream(self, text, voice=None, speed=None):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=CONNECT_TIMEOUT)
        try:
            conn.connect()
            conn.sock.settimeout(self.read_timeout)
            body = json.dumps({"text": text, "voice": self.voice or voice or "",
                               "speed": float(speed or 1.0)})
            conn.request("POST", "/tts", body=body, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            if resp.status != 200:
                raise ConnectionError(f"HTTP {resp.status}: {resp.read()[:200]!r}")
            rate = int(resp.getheader("X-Sample-Rate", "24000"))
            spare = b""
            while True:
                data = resp.read1(65536)
                if not data:
                    break
                data = spare + data
                cut = len(data) - len(data) % 4
                spare = data[cut:]
                if cut:
                    yield np.frombuffer(data[:cut], dtype="<f4").astype(np.float32), rate
        finally:
            # Closing mid-reply (barge-in) is what tells the server to stop.
            conn.close()


class FallbackTts:
    """The local synthesiser's interface (create/stream); a server from the
    pool speaks when `use_remote()` allows it, preferring a board that is not
    transcribing the user at the moment."""

    def __init__(self, local, remote, use_remote, health=None):
        self.local = local
        self.pool = Pool("tts", _as_list(remote), health)
        self.use_remote = use_remote
        # Several units of a reply may be synthesising at once (one per board),
        # each streamed by its own thread: per-thread state, and this machine's
        # engine -- the fallback -- used by one of them at a time.
        self._tls = threading.local()
        self._local_lock = threading.Lock()
        # Whether the sentence being streamed is coming from a board, so the
        # player can hold a longer cushion for it (TTS_REMOTE_PREBUFFER).
        self.speaking_remote = False

    @property
    def speaking_remote(self) -> bool:
        """For the sentence this thread is streaming."""
        return getattr(self._tls, "remote", False)

    @speaking_remote.setter
    def speaking_remote(self, value):
        self._tls.remote = bool(value)

    @property
    def remote(self):
        return self.pool.servers[0] if self.pool.servers else None

    def _local_stream(self, text, voice, speed):
        with self._local_lock:
            stream = getattr(self.local, "stream", None)
            if stream is not None:
                yield from stream(text, voice=voice, speed=speed)
            else:
                yield self.local.create(text, voice=voice, speed=speed)

    def stream(self, text, voice=None, speed=None):
        self.speaking_remote = False
        if self.use_remote():
            with _IN_FLIGHT_LOCK:
                listening = set(_IN_FLIGHT["stt"])
            for i, server, health in self.pool.order(avoid=listening):
                self.pool.acquire(i)
                try:
                    gen = server.stream(text, voice=voice, speed=speed)
                    try:
                        first = next(gen)
                    except StopIteration:
                        return
                    except Exception as e:
                        # Nothing has been heard yet: the next server, or this
                        # machine, can still say all of it.
                        health.mark_down(str(e))
                        continue
                    self.speaking_remote = True
                    yield first
                    try:
                        yield from gen
                    except Exception as e:
                        # Part of it was heard; repeating it elsewhere would
                        # say those words twice. The rest of this unit is lost.
                        health.mark_down(f"{e} (mid-sentence)")
                    finally:
                        gen.close()
                    return
                finally:
                    self.pool.release(i)
        yield from self._local_stream(text, voice, speed)

    def create(self, text, voice=None, speed=None):
        pieces, rate = [], None
        for samples, sr in self.stream(text, voice=voice, speed=speed):
            pieces.append(samples)
            rate = sr
        if not pieces:
            return np.zeros(0, np.float32), rate or 24000
        return np.concatenate(pieces), rate
