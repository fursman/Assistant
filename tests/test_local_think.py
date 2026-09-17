"""The local model's turn under the router's hard verdict (voice_assistant.py:
_stream_local_llm): thinking on demand with a budget and a spoken filler, or
the hard backend when one answers, with a local retry when it does not.

No server: the OpenAI client is a fake that records the request and streams a
few chunks; the host has only the state the local turn touches.

    .venv/bin/python -m pytest tests
"""
import logging
import queue
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import voice_assistant as va  # noqa: E402


# --- fakes -------------------------------------------------------------------

class _Delta:
    def __init__(self, content=None, reasoning=None):
        self.content = content
        self.reasoning_content = reasoning
        self.tool_calls = None


class _Choice:
    def __init__(self, delta, finish=None):
        self.delta = delta
        self.finish_reason = finish


class _Chunk:
    def __init__(self, delta, finish=None):
        self.choices = [_Choice(delta, finish)]


class _Stream:
    def __init__(self, chunks):
        self._it = iter(chunks)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)

    def close(self):
        self.closed = True


class FakeCompletions:
    def __init__(self, reply="Forty-two.", reasoning="", fail=False):
        self.calls = []
        self.reply = reply
        self.reasoning = reasoning
        self.fail = fail

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise ConnectionError("no route to host")
        chunks = []
        if self.reasoning:
            chunks.append(_Chunk(_Delta(reasoning=self.reasoning)))
        chunks.append(_Chunk(_Delta(content=self.reply)))
        chunks.append(_Chunk(_Delta(), finish="stop"))
        return _Stream(chunks)


class FakeClient:
    def __init__(self, **kw):
        self.chat = type("Chat", (), {})()
        self.chat.completions = FakeCompletions(**kw)


class FakeLedger:
    def __init__(self):
        self.remembered = []

    def unseen(self, backend):
        return []

    def caught_up(self, backend):
        pass

    def remember(self, user, reply, backend=None):
        self.remembered.append((user, reply, backend))


class FakeVerdict:
    def __init__(self, hard=False):
        self.hard = hard
        self.backend = None


class Host(va.VoiceAssistant):
    """A VoiceAssistant with only what a local turn touches."""

    def __init__(self, client, verdict=None):
        self.logger = logging.getLogger("test-local-think")
        self.client = client
        self.ledger = FakeLedger()
        self._router = None
        self._router_verdict = verdict
        self._local_history = []
        self._assistant_text = ""
        self._assistant_spoken_pos = 0
        self._thinking_text = ""
        self._sentence_queue = queue.Queue()
        self._local_stream = None
        self._hard_client = None
        self._hard_health = (0.0, False)
        self._turn_tool_calls = 0
        self._turn_think = False
        self._turn_model = ""
        self.used = []

    def _local_llm_client(self):
        return self.client

    def _ledger(self):
        return self.ledger

    def _flush_sentences(self, final=False):
        pass

    def _stream_transcript(self, text, role="assistant"):
        pass

    def _maybe_notify_thinking(self):
        pass

    def _router_tool_choice(self):
        return "auto"

    def _shrink_for_history(self, m):
        return m

    def _trim_local_history(self):
        pass

    def _save_local_history(self):
        pass

    def _router_used(self, backend):
        self.used.append(backend)


def _queued(q):
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(va, "LOCAL_LLM_THINK", False)
    monkeypatch.setattr(va, "ROUTER_THINK", True)
    monkeypatch.setattr(va, "LOCAL_THINK_BUDGET", 512)
    monkeypatch.setattr(va, "LOCAL_THINK_FILLER", "Let me think about that.")
    monkeypatch.setattr(va, "HARD_URL", "")
    monkeypatch.setattr(va, "HARD_MODEL", "")
    monkeypatch.setattr(va, "LOCAL_TOOLS_ENABLED", True)


# --- thinking on demand ------------------------------------------------------------

def test_plain_turn_does_not_think(env):
    client = FakeClient()
    host = Host(client, FakeVerdict(hard=False))
    assert host._stream_local_llm("what is two plus two", threading.Event()) is True
    kw = client.chat.completions.calls[0]
    assert kw["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning_budget_tokens" not in kw["extra_body"]
    assert kw["model"] == va.LOCAL_LLM_MODEL and "tools" in kw
    assert _queued(host._sentence_queue) == [], "no filler on an ordinary turn"
    assert host._turn_think is False
    assert host._assistant_text == "Forty-two."


def test_no_verdict_means_no_thinking(env):
    client = FakeClient()
    host = Host(client, None)
    assert host._stream_local_llm("hello", threading.Event()) is True
    assert client.chat.completions.calls[0]["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_hard_turn_thinks_under_a_budget_and_says_so(env):
    client = FakeClient(reasoning="Hmm, the user asks about entropy...")
    host = Host(client, FakeVerdict(hard=True))
    assert host._stream_local_llm("why does entropy always increase", threading.Event()) is True
    kw = client.chat.completions.calls[0]
    assert kw["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert kw["extra_body"]["reasoning_budget_tokens"] == 512
    assert kw["model"] == va.LOCAL_LLM_MODEL and "tools" in kw, "same model, same tools"
    filler = _queued(host._sentence_queue)
    assert len(filler) == 1 and "think" in filler[0].lower()
    assert host._turn_think is True
    assert host._thinking_text.startswith("Hmm"), "the reasoning stream is captured, not spoken"
    assert host._assistant_text == "Forty-two."
    assert host.ledger.remembered == [("why does entropy always increase", "Forty-two.", "local")]


def test_router_think_switch_off(env, monkeypatch):
    monkeypatch.setattr(va, "ROUTER_THINK", False)
    client = FakeClient()
    host = Host(client, FakeVerdict(hard=True))
    host._stream_local_llm("why does entropy always increase", threading.Event())
    kw = client.chat.completions.calls[0]
    assert kw["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert _queued(host._sentence_queue) == []


def test_global_thinking_has_no_budget_and_no_filler(env, monkeypatch):
    monkeypatch.setattr(va, "LOCAL_LLM_THINK", True)
    client = FakeClient()
    host = Host(client, FakeVerdict(hard=False))
    host._stream_local_llm("hello", threading.Event())
    kw = client.chat.completions.calls[0]
    assert kw["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert "reasoning_budget_tokens" not in kw["extra_body"]
    assert _queued(host._sentence_queue) == []


# --- the hard backend --------------------------------------------------------------

@pytest.fixture
def hard_env(env, monkeypatch):
    monkeypatch.setattr(va, "HARD_URL", "http://pile.local:8092/v1")
    monkeypatch.setattr(va, "HARD_MODEL", "flash-next")
    monkeypatch.setattr(va, "HARD_HISTORY_TURNS", 2)
    monkeypatch.setattr(va, "HARD_THINK", False)


def test_hard_turn_goes_to_the_hard_backend_without_tools(hard_env, monkeypatch):
    local, remote = FakeClient(), FakeClient(reply="Because of the second law.")
    host = Host(local, FakeVerdict(hard=True))
    monkeypatch.setattr(host, "_hard_backend_ready", lambda: True)
    monkeypatch.setattr(host, "_hard_llm_client", lambda: remote)
    host._local_history = [
        {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"}, {"role": "assistant", "content": "a3"},
    ]
    assert host._stream_local_llm("why does entropy always increase", threading.Event()) is True
    assert local.chat.completions.calls == [], "the local model was not asked"
    kw = remote.chat.completions.calls[0]
    assert kw["model"] == "flash-next"
    assert "tools" not in kw and "tool_choice" not in kw
    assert kw["messages"][0]["role"] == "system" and "three tools" not in kw["messages"][0]["content"]
    assert "no tools on this turn" in kw["messages"][0]["content"]
    assert [m["content"] for m in kw["messages"][1:]] == ["q2", "a2", "q3", "a3", "why does entropy always increase"]
    assert kw["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert host.used == ["hard"]
    assert host._turn_model == "flash-next"
    assert len(_queued(host._sentence_queue)) == 1, "the filler covers the remote round trip"
    assert host._assistant_text == "Because of the second law."
    # the exchange lands in the local history too, so the local model knows what was said
    assert host._local_history[-2:] == [{"role": "user", "content": "why does entropy always increase"},
                                        {"role": "assistant", "content": "Because of the second law."}]


def test_hard_backend_failure_falls_back_to_local_thinking(hard_env, monkeypatch):
    local, remote = FakeClient(reply="Locally: entropy."), FakeClient(fail=True)
    host = Host(local, FakeVerdict(hard=True))
    monkeypatch.setattr(host, "_hard_backend_ready", lambda: True)
    monkeypatch.setattr(host, "_hard_llm_client", lambda: remote)
    assert host._stream_local_llm("why does entropy always increase", threading.Event()) is True
    assert len(remote.chat.completions.calls) == 1
    kw = local.chat.completions.calls[0]
    assert kw["model"] == va.LOCAL_LLM_MODEL
    assert kw["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert kw["extra_body"]["reasoning_budget_tokens"] == 512
    assert host.used == ["hard", "local"]
    assert host._hard_health[1] is False, "the backend is remembered as down"
    assert host._assistant_text == "Locally: entropy."


def test_hard_backend_not_ready_stays_local(hard_env, monkeypatch):
    local = FakeClient()
    host = Host(local, FakeVerdict(hard=True))
    monkeypatch.setattr(host, "_hard_backend_ready", lambda: False)
    host._stream_local_llm("why does entropy always increase", threading.Event())
    kw = local.chat.completions.calls[0]
    assert kw["model"] == va.LOCAL_LLM_MODEL
    assert kw["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert host.used == []


def test_hard_backend_health_is_cached(monkeypatch):
    monkeypatch.setattr(va, "HARD_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setattr(va, "HARD_HEALTH_TTL", 30.0)
    host = Host(FakeClient(), None)
    probes = []

    def fake_open(url, timeout=0.0):
        probes.append(url)
        raise OSError("refused")
    monkeypatch.setattr(va.urllib.request, "urlopen", fake_open)
    assert host._hard_backend_ready() is False
    assert host._hard_backend_ready() is False
    assert probes == ["http://127.0.0.1:1/health"], "one probe, then the cached answer"


def test_history_tail():
    tail = va.VoiceAssistant._history_tail
    h = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
         {"role": "user", "content": "q2"}, {"role": "tool", "content": "out"},
         {"role": "assistant", "content": "a2"}]
    assert tail(h, 1) == h[2:]
    assert tail(h, 2) == h
    assert tail(h, 5) == h
    assert tail(h, 0) == [] and tail([], 3) == []
    assert tail([{"role": "assistant", "content": "orphan"}], 2) == []
