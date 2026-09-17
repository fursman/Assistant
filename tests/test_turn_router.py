"""Tests for turn_router.py. No network: a threaded http.server stands in for
the llama-server's /judge endpoint, and the rest is arithmetic.

    .venv/bin/python -m pytest tests
"""
import json
import math
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import turn_router as tr  # noqa: E402
from turn_router import (ROUTER_QUESTIONS, ROUTER_SYSTEM, QUESTION_IDS, TurnRouter, Verdict,  # noqa: E402
                         build_context, calibrate, judge_url, load_thresholds)

REPO_CAL = ROOT / "router_calibration.json"
SYSTEMONE_CAL = Path.home() / "Claude/system-one/data/router_calibration.json"
SYSTEMONE_DATA = Path.home() / "Claude/system-one/systemone/data.py"


# --- a stand-in /judge server -------------------------------------------------

class _State:
    def __init__(self):
        self.status = 200
        self.body = {}
        self.delay = 0.0
        self.raw_body = None      # bytes to send verbatim instead of json
        self.requests = []
        self.health = 200


class _Handler(BaseHTTPRequestHandler):
    state: _State = None

    def log_message(self, *a):  # keep pytest output clean
        pass

    def do_GET(self):
        self.send_response(self.state.health)
        self.end_headers()
        self.wfile.write(b"{}")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.state.requests.append((self.path, json.loads(self.rfile.read(n) or b"{}")))
        if self.state.delay:
            time.sleep(self.state.delay)
        body = self.state.raw_body if self.state.raw_body is not None \
            else json.dumps(self.state.body).encode()
        self.send_response(self.state.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def server():
    state = _State()
    handler = type("H", (_Handler,), {"state": state})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", state
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def router(server, tmp_path):
    url, state = server
    r = TurnRouter(url + "/v1", calibration_path=str(REPO_CAL), timeout=1.0,
                   log_path=tmp_path / "router.jsonl")
    return r, state


def _logit(p):
    return math.log(p / (1.0 - p))


def judge_response(router: TurnRouter, targets: dict, mass=0.98, total_ms=412.0):
    """A /judge body whose calibrated P(yes) comes out at `targets` (default 0.5)."""
    qs = []
    for qid in QUESTION_IDS:
        a, c = router.calibration[qid]
        p = min(max(targets.get(qid, 0.5), 1e-6), 1 - 1e-6)
        ly = (_logit(p) - c) / a       # with logit_no = 0
        ln = 0.0
        py = 1.0 / (1.0 + math.exp(ln - ly))
        qs.append({"id": qid, "options": [{"text": "yes", "logprob": ly, "prob": py},
                                          {"text": "no", "logprob": ln, "prob": 1 - py}],
                   "mass": mass, "argmax": "yes" if py >= 0.5 else "no"})
    return {"questions": qs, "timings": {"total_ms": total_ms}, "n_prefix_tokens": 150}


def _verdict(probs, typed=False, thresholds=None):
    full = {q: 0.5 for q in QUESTION_IDS}
    full.update(probs)
    return Verdict(text="x", typed=typed, context="", probs=full, raw=dict(full),
                   mass={q: 0.98 for q in QUESTION_IDS},
                   thresholds=thresholds or load_thresholds(json.loads(REPO_CAL.read_text())["thresholds"]),
                   latency_ms=1.0)


# --- the rubric is the v3 rubric, verbatim --------------------------------------

def test_rubric_matches_calibration_file_and_systemone():
    cal = json.loads(REPO_CAL.read_text())
    assert cal["system"] == ROUTER_SYSTEM
    assert list(cal["questions"]) == QUESTION_IDS
    for qid, text in ROUTER_QUESTIONS:
        assert cal["questions"][qid]["text"] == text + " Answer yes or no."
    if SYSTEMONE_CAL.exists():
        assert json.loads(SYSTEMONE_CAL.read_text()) == cal, "repo calibration drifted from system-one"
    if SYSTEMONE_DATA.exists():
        ns = {}
        src = SYSTEMONE_DATA.read_text()
        # Execute just the two definitions, not the module (it imports pandas).
        for name in ("ROUTER_Q3", "ROUTER_SYSTEM"):
            m = re.search(rf"^{name} = (\[.*?^\]|\(.*?\))\n", src, re.S | re.M)
            assert m, name
            exec(f"{name} = {m.group(1)}", ns)
        assert ns["ROUTER_SYSTEM"] == ROUTER_SYSTEM
        assert ns["ROUTER_Q3"] == ROUTER_QUESTIONS


def test_request_body_is_exactly_the_judge_contract(router):
    r, _ = router
    body = r.request_body("hello there")
    assert body["mode"] == "chat" and body["system"] == ROUTER_SYSTEM
    assert body["separator"] == "\n\n" and body["enable_thinking"] is False and body["top_k"] == 0
    assert [q["id"] for q in body["questions"]] == QUESTION_IDS
    assert all(q["options"] == ["yes", "no"] for q in body["questions"])
    assert body["questions"][3]["text"].endswith("counts. Answer yes or no.")
    assert body["context"] == "Latest utterance (spoken):\nhello there"


# --- calibration math ---------------------------------------------------------------

def test_calibrate_is_platt_on_the_logit_difference():
    assert calibrate(1.0, 0.0, 0.0, 0.0) == pytest.approx(0.5)
    # identity fit == softmax over the two options
    ly, ln = -0.1, -2.3
    assert calibrate(1.0, 0.0, ly, ln) == pytest.approx(math.exp(ly) / (math.exp(ly) + math.exp(ln)))
    a, c = 1.2430087029684174, 2.6624229195053832      # the "addressed" fit
    z = a * (ly - ln) + c
    assert calibrate(a, c, ly, ln) == pytest.approx(1 / (1 + math.exp(-z)))
    # a strong "no" with a big positive intercept is pulled back toward yes
    assert calibrate(a, c, -5.0, 0.0) > 1 / (1 + math.exp(5.0))
    # extreme inputs do not overflow
    assert calibrate(1.0, 0.0, 1000.0, -1000.0) == pytest.approx(1.0)
    assert calibrate(1.0, 0.0, -1000.0, 1000.0) == pytest.approx(0.0)


def test_judge_applies_the_calibration_from_the_file(router):
    r, state = router
    state.body = judge_response(r, {"needs_shell": 0.91, "simple": 0.08})
    v = r.judge("what is using my disk")
    assert v is not None
    assert v.probs["needs_shell"] == pytest.approx(0.91)
    assert v.probs["simple"] == pytest.approx(0.08)
    # raw is the engine's own two-option probability, not the calibrated one
    a, c = r.calibration["needs_shell"]
    assert v.raw["needs_shell"] != pytest.approx(0.91)
    assert v.mass["needs_shell"] == pytest.approx(0.98)
    assert v.server_ms == 412.0 and v.n_prefix_tokens == 150
    assert v.latency_ms > 0


# --- decisions at the thresholds --------------------------------------------------------

def test_thresholds_come_from_file_then_env(monkeypatch, tmp_path):
    thr = load_thresholds({"drop_junk": 0.9, "bogus": 1})
    assert thr["drop_junk"] == 0.9 and "bogus" not in thr and thr["needs_web"] == 0.2
    monkeypatch.setenv("VOICE_ASSISTANT_ROUTER_THR_DROP", "0.55")
    monkeypatch.setenv("VOICE_ASSISTANT_ROUTER_THR_WEB", "0.33")
    monkeypatch.setenv("VOICE_ASSISTANT_ROUTER_THR_SHELL", "0.44")
    monkeypatch.setenv("VOICE_ASSISTANT_ROUTER_THR_RISKY", "0.66")
    monkeypatch.setenv("VOICE_ASSISTANT_ROUTER_THR_SIMPLE", "0.77")
    r = TurnRouter("http://127.0.0.1:1", calibration_path=str(REPO_CAL), log_path=tmp_path / "r.jsonl")
    assert r.thresholds == {"drop_junk": 0.55, "needs_web": 0.33, "needs_shell": 0.44,
                            "risky": 0.66, "simple_local": 0.77,
                            "escalate_web": 0.6, "escalate_shell": 0.6, "hard": 0.3, "caution": 0.1}
    monkeypatch.setenv("VOICE_ASSISTANT_ROUTER_THR_ESCALATE_SHELL", "0.9")
    monkeypatch.setenv("VOICE_ASSISTANT_ROUTER_THR_HARD", "0.25")
    r = TurnRouter("http://127.0.0.1:1", calibration_path=str(REPO_CAL), log_path=tmp_path / "r.jsonl")
    assert r.thresholds["escalate_shell"] == 0.9 and r.thresholds["hard"] == 0.25
    monkeypatch.setenv("VOICE_ASSISTANT_ROUTER_THR_DROP", "not a number")
    r = TurnRouter("http://127.0.0.1:1", calibration_path=str(REPO_CAL), log_path=tmp_path / "r.jsonl")
    assert r.thresholds["drop_junk"] == 0.8


def test_drop_at_the_junk_threshold():
    # junk = 1 - min(addressed, intelligible); dropped at >= 0.8
    assert _verdict({"addressed": 0.2, "intelligible": 0.99}).drop
    assert _verdict({"addressed": 0.99, "intelligible": 0.2}).drop
    assert _verdict({"addressed": 0.2, "intelligible": 0.99}).junk == pytest.approx(0.8)
    assert not _verdict({"addressed": 0.21, "intelligible": 0.99}).drop
    assert not _verdict({"addressed": 0.99, "intelligible": 0.99}).drop


def test_typed_input_is_never_dropped():
    v = _verdict({"addressed": 0.0, "intelligible": 0.0}, typed=True)
    assert v.junk == pytest.approx(1.0)
    assert not v.drop


def test_tool_decisions_at_their_thresholds():
    assert _verdict({"needs_web": 0.2}).needs_web and not _verdict({"needs_web": 0.199}).needs_web
    assert _verdict({"needs_shell": 0.2}).needs_shell and not _verdict({"needs_shell": 0.199}).needs_shell
    assert _verdict({"risky": 0.3}).risky and not _verdict({"risky": 0.299}).risky
    assert _verdict({"simple": 0.7}).simple and not _verdict({"simple": 0.699}).simple
    assert _verdict({"needs_web": 0.9, "needs_shell": 0.0}).tools == ["web_search", "fetch_page"]
    assert _verdict({"needs_web": 0.0, "needs_shell": 0.9}).tools == ["run_shell"]
    assert _verdict({"needs_web": 0.9, "needs_shell": 0.9}).tools == ["web_search", "fetch_page", "run_shell"]
    assert _verdict({"needs_web": 0.0, "needs_shell": 0.0}).tools == []
    assert _verdict({"followup": 0.5}).followup and not _verdict({"followup": 0.49}).followup
    assert _verdict({"question": 0.5}).question and not _verdict({"question": 0.49}).question


def test_route_leaves_local_only_for_risky_turns_and_real_tool_tasks():
    quiet = {"needs_web": 0.0, "needs_shell": 0.0, "risky": 0.0}
    # simple stays local, tool or no tool
    assert _verdict({**quiet, "simple": 0.7}).route == "local"
    assert _verdict({**quiet, "simple": 0.99, "needs_shell": 0.9}).route == "local"
    assert _verdict({**quiet, "simple": 0.99, "needs_web": 0.9}).route == "local"
    # not simple and no strong tool need: local (the hard lane)
    assert _verdict({**quiet, "simple": 0.1}).route == "local"
    # a glance at the machine is not a task; past the escalation bar it is
    assert _verdict({**quiet, "simple": 0.3, "needs_shell": 0.59}).route == "local"
    assert _verdict({**quiet, "simple": 0.3, "needs_shell": 0.6}).route == "claude"
    assert _verdict({**quiet, "simple": 0.3, "needs_web": 0.6}).route == "claude"
    assert _verdict({**quiet, "simple": 0.699, "needs_web": 0.6}).route == "claude"
    # risky always leaves
    assert _verdict({**quiet, "simple": 0.99, "risky": 0.3}).route == "claude"
    assert _verdict({**quiet, "simple": 0.99, "risky": 0.29}).route == "local"
    assert _verdict({**quiet, "simple": 0.0, "risky": 0.3}).escalate is False


def test_the_casual_questions_that_went_to_claude_now_stay_local():
    # Three real verdicts from 2026-09-16 that cost a Claude round trip each.
    for shell, simple in ((0.29, 0.98), (0.24, 0.76), (0.31, 0.57)):
        v = _verdict({"needs_web": 0.05, "needs_shell": shell, "risky": 0.02, "simple": simple})
        assert v.route == "local" and v.needs_shell and v.tools == ["run_shell"]


def test_hard_and_caution_flags():
    quiet = {"needs_web": 0.0, "needs_shell": 0.0, "risky": 0.0}
    assert _verdict({**quiet, "simple": 0.3}).hard
    assert not _verdict({**quiet, "simple": 0.31}).hard
    assert not _verdict({**quiet, "simple": 0.1, "needs_shell": 0.2}).hard, "a tool turn does not think"
    assert not _verdict({**quiet, "simple": 0.1, "needs_web": 0.2}).hard
    assert not _verdict({**quiet, "simple": 0.1, "risky": 0.3}).hard, "not local, so not hard"
    assert _verdict({**quiet, "simple": 0.9, "risky": 0.1}).caution
    assert not _verdict({**quiet, "simple": 0.9, "risky": 0.099}).caution
    assert not _verdict({**quiet, "simple": 0.9, "risky": 0.3}).caution, "risky itself, not caution"
    d = _verdict({**quiet, "simple": 0.2, "risky": 0.15}).decisions()
    assert d["hard"] and d["caution"] and d["route"] == "local" and not d["escalate"]
    line = _verdict({**quiet, "simple": 0.2, "risky": 0.15}).log_line()
    assert line.endswith("tools=none hard caution (1 ms)")


def test_decisions_end_to_end_over_http(router):
    r, state = router
    state.body = judge_response(r, {"addressed": 0.99, "intelligible": 0.97, "needs_web": 0.05,
                                    "needs_shell": 0.91, "risky": 0.12, "simple": 0.08,
                                    "followup": 0.80, "question": 0.30})
    v = r.judge("what is using all my disk space")
    assert v.route == "claude" and v.tools == ["run_shell"] and not v.drop and not v.risky
    assert v.log_line() == ("Router: addressed=0.99 intelligible=0.97 web=0.05 shell=0.91 risky=0.12 "
                            f"simple=0.08 -> route=claude tools=shell escalate "
                            f"({v.latency_ms:.0f} ms)")

    state.body = judge_response(r, {"addressed": 0.99, "intelligible": 0.99, "needs_web": 0.02,
                                    "needs_shell": 0.03, "risky": 0.01, "simple": 0.95,
                                    "followup": 0.1, "question": 0.9})
    v = r.judge("how many legs does a spider have")
    assert v.route == "local" and v.tools == [] and not v.drop
    assert "tools=none" in v.log_line()

    state.body = judge_response(r, {"addressed": 0.1, "intelligible": 0.5})
    v = r.judge("mumble mumble")
    assert v.drop and v.junk == pytest.approx(0.9)
    v = r.judge("mumble mumble", typed=True)
    assert not v.drop

    state.body = judge_response(r, {"addressed": 0.99, "intelligible": 0.99, "needs_web": 0.0,
                                    "needs_shell": 0.9, "risky": 0.8, "simple": 0.1})
    v = r.judge("yes go ahead and reboot it")
    assert v.risky and v.route == "claude" and v.log_line().endswith(f"tools=shell risky escalate ({v.latency_ms:.0f} ms)")


# --- fail open --------------------------------------------------------------------------

def test_fail_open_on_http_errors(router):
    r, state = router
    state.body = judge_response(r, {})
    for status in (503, 404, 500, 400):
        state.status = status
        assert r.judge("hello") is None
    state.status = 200
    assert r.judge("hello") is not None


def test_fail_open_on_timeout(server, tmp_path):
    url, state = server
    r = TurnRouter(url, calibration_path=str(REPO_CAL), timeout=0.2, log_path=tmp_path / "r.jsonl")
    state.body = judge_response(r, {})
    state.delay = 0.8
    t0 = time.monotonic()
    assert r.judge("hello") is None
    assert time.monotonic() - t0 < 0.7


def test_fail_open_when_nothing_listens(tmp_path):
    r = TurnRouter("http://127.0.0.1:1/v1", calibration_path=str(REPO_CAL), timeout=0.5,
                   log_path=tmp_path / "r.jsonl")
    assert r.judge("hello") is None
    assert r.healthy() is False


def test_fail_open_on_malformed_answers(router):
    r, state = router
    state.raw_body = b"this is not json"
    assert r.judge("hello") is None
    state.raw_body = None
    body = judge_response(r, {})
    body["questions"] = body["questions"][:-1]                 # one question missing
    state.body = body
    assert r.judge("hello") is None
    body = judge_response(r, {})
    body["questions"][0]["options"] = [{"text": "maybe", "logprob": -1.0}]   # wrong options
    state.body = body
    assert r.judge("hello") is None
    state.body = {"error": "context too long"}
    assert r.judge("hello") is None


def test_disabled_router_makes_no_request(server, tmp_path):
    url, state = server
    r = TurnRouter(url, calibration_path=str(REPO_CAL), enabled=False, log_path=tmp_path / "r.jsonl")
    assert r.judge("hello") is None
    assert state.requests == []
    r.enabled = True
    assert r.judge("   ") is None
    assert state.requests == []


def test_monkeypatched_post_exception_fails_open(router, monkeypatch):
    r, _ = router

    def boom(*a, **k):
        raise tr.requests.ConnectionError("down")
    monkeypatch.setattr(r.session, "post", boom)
    assert r.judge("hello") is None


# --- context construction ----------------------------------------------------------

def test_context_without_previous_exchange():
    assert build_context("hello there") == "Latest utterance (spoken):\nhello there"
    assert build_context("  hello there \n", typed=True) == "Latest utterance (typed):\nhello there"
    assert build_context("hello", previous=None) == "Latest utterance (spoken):\nhello"


def test_context_with_previous_exchange_matches_the_dataset_format():
    prev = ("what time is it", "It is half past four.", 12.0)
    assert build_context("and the date?", previous=prev) == (
        "Previous exchange:\nUser: what time is it\nAssistant: It is half past four.\n\n"
        "Latest utterance (spoken):\nand the date?")
    # typed latest, spoken previous: only the latest is labelled
    assert build_context("and the date?", typed=True, previous=prev).endswith(
        "Latest utterance (typed):\nand the date?")


def test_context_previous_exchange_rules():
    long_reply = "x" * 500
    ctx = build_context("ok", previous=("q", long_reply, 1.0))
    assert "Assistant: " + "x" * 300 + "\n\n" in ctx and "x" * 301 not in ctx
    assert "Assistant: (no reply)\n\n" in build_context("ok", previous=("q", "", 1.0))
    assert "Assistant: (no reply)\n\n" in build_context("ok", previous=("q", None, 1.0))
    # too old, or of unknown age: no previous exchange at all
    assert build_context("ok", previous=("q", "a", 900.0)).startswith("Previous exchange:")
    assert build_context("ok", previous=("q", "a", 900.1)) == "Latest utterance (spoken):\nok"
    assert build_context("ok", previous=("q", "a", -1.0)) == "Latest utterance (spoken):\nok"
    assert build_context("ok", previous=("q", "a", None)) == "Latest utterance (spoken):\nok"
    # the assistant's own marker lines never reach the judge
    marked = ("[context: 2h since the last exchange, now Wed 09 Sep 12:29 PDT]\n"
              "[router: this request may change or disrupt the machine or the user's data; confirm before acting]\n"
              "delete the old backups")
    assert build_context("yes", previous=(marked, "Are you sure?", 5.0)).startswith(
        "Previous exchange:\nUser: delete the old backups\nAssistant: Are you sure?\n\n")


def test_judge_sends_the_context_it_records(router):
    r, state = router
    state.body = judge_response(r, {})
    v = r.judge("and the date?", previous=("what time is it", "Half past four.", 3.0))
    sent = state.requests[-1][1]["context"]
    assert sent == v.context == ("Previous exchange:\nUser: what time is it\nAssistant: Half past four.\n\n"
                                 "Latest utterance (spoken):\nand the date?")


# --- urls, calibration files, the record -----------------------------------------------

def test_judge_url_shapes():
    assert judge_url("http://127.0.0.1:8081/v1") == "http://127.0.0.1:8081/judge"
    assert judge_url("http://127.0.0.1:8081/v1/") == "http://127.0.0.1:8081/judge"
    assert judge_url("http://127.0.0.1:8081") == "http://127.0.0.1:8081/judge"
    assert judge_url("http://127.0.0.1:8081/judge") == "http://127.0.0.1:8081/judge"
    r = TurnRouter("http://pile:8081/v1", calibration_path=str(REPO_CAL))
    assert r.url == "http://pile:8081/judge" and r.health_url == "http://pile:8081/health"


def test_calibration_user_copy_wins_then_repo(monkeypatch, tmp_path):
    monkeypatch.setattr(tr, "USER_CALIBRATION", tmp_path / "missing.json")
    r = TurnRouter("http://127.0.0.1:1", log_path=tmp_path / "r.jsonl")
    assert r.calibration_path == tr.REPO_CALIBRATION
    user = tmp_path / "user.json"
    cal = json.loads(REPO_CAL.read_text())
    cal["thresholds"]["simple_local"] = 0.42
    cal["questions"]["simple"]["a"] = 2.0
    user.write_text(json.dumps(cal))
    monkeypatch.setattr(tr, "USER_CALIBRATION", user)
    r = TurnRouter("http://127.0.0.1:1", log_path=tmp_path / "r.jsonl")
    assert r.calibration_path == user
    assert r.thresholds["simple_local"] == 0.42 and r.calibration["simple"][0] == 2.0
    # a corrupt user copy falls back to the repo's
    user.write_text("{not json")
    r = TurnRouter("http://127.0.0.1:1", log_path=tmp_path / "r.jsonl")
    assert r.calibration_path == tr.REPO_CALIBRATION
    # no calibration anywhere is an error the caller can catch (and disable the router)
    with pytest.raises(ValueError):
        TurnRouter("http://127.0.0.1:1", calibration_path=str(tmp_path / "nope.json"))


def test_record_writes_one_json_line_once(router, tmp_path):
    r, state = router
    state.body = judge_response(r, {"needs_shell": 0.9})
    v = r.judge("free the disk", previous=("hi", "hello", 2.0))
    assert r.record(v, backend="claude") is True
    assert r.record(v, backend="local") is False              # once
    lines = (tmp_path / "router.jsonl").read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["text"] == "free the disk" and row["typed"] is False and row["backend"] == "claude"
    assert row["context"] == v.context and row["id"] == v.id
    assert row["decisions"]["needs_shell"] is True and row["decisions"]["route"] == "claude"
    assert set(row["probs"]) == set(QUESTION_IDS) and set(row["raw"]) == set(QUESTION_IDS)
    assert row["latency_ms"] > 0 and row["server_ms"] == 412.0
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$", row["ts"])
    assert json.loads(v.to_json()) == row


def test_record_never_raises(router, tmp_path):
    r, state = router
    state.body = judge_response(r, {})
    v = r.judge("hello")
    r.log_path = tmp_path                     # a directory: open() fails
    assert r.record(v) is False
    assert r.record(None) is False


# --- the reply check and the outcome ----------------------------------------------------------

def test_check_reply_asks_the_three_questions_about_the_exchange(router):
    r, state = router
    v = _verdict({"simple": 0.2})
    v.context = "Latest utterance (spoken):\nwho wrote hamlet"
    qs = []
    for qid, ly in (("answered", 2.0), ("unsupported", -2.0), ("needed_tool", 0.0)):
        qs.append({"id": qid, "options": [{"text": "yes", "logprob": ly}, {"text": "no", "logprob": 0.0}],
                   "mass": 0.99})
    state.body = {"questions": qs, "timings": {"total_ms": 300}}
    res = r.check_reply(v, "William Shakespeare.")
    assert set(res) == {"answered", "unsupported", "needed_tool"}
    assert res["answered"] > 0.85 and res["unsupported"] < 0.15 and res["needed_tool"] == pytest.approx(0.5)
    path, body = state.requests[-1]
    assert path == "/judge"
    assert body["system"] == tr.REPLY_CHECK_SYSTEM
    assert body["context"].startswith(v.context) and body["context"].endswith("Assistant's reply:\nWilliam Shakespeare.")
    assert [q["id"] for q in body["questions"]] == ["answered", "unsupported", "needed_tool"]
    assert all(q["options"] == ["yes", "no"] for q in body["questions"])
    assert body["enable_thinking"] is False
    assert r.check_reply(v, "") is None and r.check_reply(None, "x") is None


def test_check_reply_fails_open(router):
    r, state = router
    v = _verdict({})
    v.context = "Latest utterance (spoken):\nhi"
    state.status = 500
    assert r.check_reply(v, "hello") is None
    state.status = 200
    state.raw_body = b"not json"
    assert r.check_reply(v, "hello") is None


def test_record_carries_the_outcome(router, tmp_path):
    r, _ = router
    v = _verdict({"simple": 0.9})
    v.outcome.update({"took_s": 1.2, "check": {"answered": 0.9}})
    assert r.record(v, backend="local")
    row = json.loads((tmp_path / "router.jsonl").read_text().strip())
    assert row["backend"] == "local" and row["outcome"] == {"took_s": 1.2, "check": {"answered": 0.9}}
    assert row["decisions"]["hard"] is False and row["decisions"]["caution"] is False
