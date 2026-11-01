"""Speech on the LAN (remote_speech.py): the network and Claude tiers use the
servers while they answer; everything falls back to this machine's engines."""
import http.server
import json
import socket
import socketserver
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import remote_speech as rs  # noqa: E402

HDR = struct.Struct(">cI")


# -- fakes ------------------------------------------------------------------

class FakeHealth:
    def __init__(self, up=True):
        self._up = up
        self.downs = []

    def up(self):
        return self._up

    def mark_down(self, why):
        self._up = False
        self.downs.append(why)


class LocalStt:
    """The local recogniser: records what it was asked to do."""

    def __init__(self):
        self.calls = []
        self.fed = 0

    def begin_utterance(self):
        self.calls.append("begin")

    def feed(self, chunk, sr):
        self.fed += len(chunk)

    def partial(self):
        return "local partial"

    def finish(self):
        self.calls.append("finish")
        return "local stream"

    def transcribe(self, audio, **kw):
        self.calls.append("batch")
        return [rs._Segment("local batch")], None


def _recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        part = conn.recv(n - len(buf))
        if not part:
            raise ConnectionError
        buf += part
    return buf


class SttServer(socketserver.ThreadingTCPServer):
    """Speaks the rk1-stt frames. `mode`: ok | die (close mid-utterance)."""
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, mode="ok"):
        self.mode = mode
        self.samples = 0
        super().__init__(("127.0.0.1", 0), self.Handler)
        threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def spec(self):
        return f"127.0.0.1:{self.server_address[1]}"

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            srv, conn = self.server, self.request

            def send(kind, payload=b""):
                conn.sendall(HDR.pack(kind, len(payload)) + payload)
            try:
                while True:
                    kind, size = HDR.unpack(_recv_exact(conn, HDR.size))
                    payload = _recv_exact(conn, size) if size else b""
                    if kind == b"H":
                        send(b"O", b'{"ok": true}')
                    elif kind == b"A":
                        srv.samples += size // 4
                        if srv.mode == "die":
                            conn.close()
                            return
                        send(b"P", f"heard {srv.samples}".encode())
                    elif kind == b"E":
                        send(b"F", b"remote stream")
                    elif kind == b"T":
                        send(b"F", f"remote batch {size // 4}".encode())
            except (ConnectionError, OSError):
                pass


def closed_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_for(cond, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


# -- recognition -------------------------------------------------------------

def stt(server_spec, use=True, up=True):
    local = LocalStt()
    health = FakeHealth(up)
    return rs.FallbackStt(local, rs.RemoteStt(server_spec, 16000, finish_timeout=1.0),
                          lambda: use, health), local, health


def test_streams_to_the_server_when_the_tier_allows_it():
    srv = SttServer()
    w, local, health = stt(srv.spec)
    w.begin_utterance()
    w.feed(np.zeros(1600, np.float32), 16000)
    w.feed(np.zeros(1600, np.float32), 16000)
    assert wait_for(lambda: w.partial() == "heard 3200")
    assert w.finish() == "remote stream"
    assert local.calls == [] and local.fed == 0, "the local engine did no work"
    assert health.downs == []


def test_the_local_tier_never_leaves_the_machine():
    srv = SttServer()
    w, local, _ = stt(srv.spec, use=False)
    w.begin_utterance()
    w.feed(np.zeros(1600, np.float32), 16000)
    assert w.partial() == "local partial"
    assert w.finish() == "local stream"
    assert w.transcribe(np.zeros(16000, np.float32))[0][0].text == "local batch"
    assert srv.samples == 0


def test_a_server_known_to_be_down_is_not_tried():
    srv = SttServer()
    w, local, _ = stt(srv.spec, up=False)
    w.begin_utterance()
    w.feed(np.zeros(1600, np.float32), 16000)
    assert w.finish() == "local stream"
    assert srv.samples == 0 and local.calls == ["begin", "finish"]


def test_a_refused_connection_falls_back_before_the_first_word():
    w, local, health = stt(f"127.0.0.1:{closed_port()}")
    w.begin_utterance()
    w.feed(np.zeros(1600, np.float32), 16000)
    assert local.fed == 1600, "the utterance went to the local stream instead"
    assert w.finish() == "local stream"
    assert len(health.downs) == 1


def test_a_server_that_dies_mid_utterance_hands_the_recording_back():
    srv = SttServer(mode="die")
    w, local, health = stt(srv.spec)
    w.begin_utterance()
    w.feed(np.zeros(1600, np.float32), 16000)
    # None tells _transcribe to run the whole recording through transcribe(),
    # which is local now that the server is marked down.
    assert w.finish() is None
    assert health.downs
    segments, _ = w.transcribe(np.zeros(16000, np.float32))
    assert segments[0].text == "local batch"


def test_batch_goes_to_the_server_too():
    srv = SttServer()
    w, local, _ = stt(srv.spec)
    segments, _ = w.transcribe(np.zeros(8000, np.float32))
    assert segments[0].text == "remote batch 8000" and local.calls == []


def test_probe():
    assert rs.RemoteStt(SttServer().spec, 16000).probe()
    with pytest.raises(OSError):
        rs.RemoteStt(f"127.0.0.1:{closed_port()}", 16000).probe()


# -- synthesis ---------------------------------------------------------------

class LocalTts:
    def __init__(self):
        self.spoken = []

    def stream(self, text, voice=None, speed=None):
        self.spoken.append(text)
        yield np.full(100, 0.5, np.float32), 24000


class TtsServer(http.server.ThreadingHTTPServer):
    """POST /tts -> float32 chunks; `chunks` of 1000 samples, `die_after` cuts the reply."""
    daemon_threads = True

    def __init__(self, chunks=3, die_after=None, status=200):
        self.chunks, self.die_after, self.status = chunks, die_after, status
        self.requests = []
        self.left_early = threading.Event()
        super().__init__(("127.0.0.1", 0), self.Handler)
        threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_GET(self):
            body = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            srv = self.server
            srv.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            if srv.status != 200:
                self.send_response(srv.status)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("X-Sample-Rate", "24000")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                for i in range(srv.chunks):
                    if srv.die_after is not None and i == srv.die_after:
                        self.connection.close()
                        return
                    data = np.full(1000, i, "<f4").tobytes()
                    self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                    self.wfile.flush()
                    time.sleep(0.02)
                self.wfile.write(b"0\r\n\r\n")
            except OSError:
                srv.left_early.set()


def tts(url, use=True, up=True):
    local = LocalTts()
    health = FakeHealth(up)
    return rs.FallbackTts(local, rs.RemoteTts(url, "af_heart"), lambda: use, health), local, health


def test_speaks_on_the_server_with_its_own_voice():
    srv = TtsServer()
    w, local, _ = tts(srv.url)
    pieces = list(w.stream("Hello there.", voice="alba", speed=1.0))
    assert [len(p) for p, _ in pieces] == [1000, 1000, 1000]
    assert all(sr == 24000 for _, sr in pieces)
    assert srv.requests == [{"text": "Hello there.", "voice": "af_heart", "speed": 1.0}]
    assert local.spoken == []


def test_local_tier_speaks_locally():
    srv = TtsServer()
    w, local, _ = tts(srv.url, use=False)
    assert len(list(w.stream("Hi."))) == 1
    assert local.spoken == ["Hi."] and srv.requests == []


def test_refused_or_failing_server_falls_back_for_the_whole_sentence():
    for w, local, health in (tts(f"http://127.0.0.1:{closed_port()}"),
                             tts(TtsServer(status=500).url)):
        pieces = list(w.stream("Hi."))
        assert local.spoken == ["Hi."] and len(pieces) == 1
        assert len(health.downs) == 1


def test_dying_mid_sentence_does_not_repeat_what_was_heard():
    srv = TtsServer(chunks=4, die_after=2)
    w, local, health = tts(srv.url)
    pieces = list(w.stream("A longer sentence."))
    assert len(pieces) == 2, "what arrived was played"
    assert local.spoken == [], "and not said a second time locally"
    assert health.downs


def test_barge_in_hangs_up_on_the_server():
    srv = TtsServer(chunks=50)
    w, _, _ = tts(srv.url)
    gen = w.stream("A very long reply.")
    next(gen)
    gen.close()
    assert srv.left_early.wait(3), "the server saw the client leave"


def test_create_joins_the_stream():
    w, _, _ = tts(TtsServer().url)
    samples, sr = w.create("Hi.")
    assert samples.shape == (3000,) and sr == 24000


def test_health_tracks_the_probe():
    state = {"ok": False}
    h = rs.Health("t", lambda: state["ok"])
    assert h.wait_first(2) is False
    state["ok"] = True
    h.mark_down("poke")          # wakes the prober
    assert wait_for(h.up)
