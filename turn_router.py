"""Turn router: one prefill-only judgment call in front of every turn.

The local llama-server (the same one that answers as the local model) has a
POST /judge endpoint that answers typed yes/no questions about a context in a
single forward pass and returns the log-probability of each option. Eight
such questions -- the v2 rubric, verbatim from the System One project -- are
asked about every utterance, the answers are calibrated with a per-question
Platt fit (`router_calibration.json`), and the calibrated probabilities make
three decisions:

  drop    the turn is junk (not addressed to the assistant, or unintelligible)
  tools   which tools the local model is offered (web, shell, both, none)
  route   "local" for a simple turn that needs no tools and changes nothing,
          otherwise "claude"

Everything here FAILS OPEN. judge() returns None on any error, timeout, HTTP
failure, missing question or unhealthy server, and None means "no opinion":
the assistant then behaves exactly as it did before the router existed.

The context is built exactly like the dataset the calibration was fitted on
(apps/router_data.py in System One): the previous exchange, if there was one
in the last 900 s, then the latest utterance marked typed or spoken. Change
that format and the calibration no longer applies.

Every verdict is appended as one JSON line to router.jsonl so the turns can
be relabelled and the calibration refitted later.

Only the standard library and `requests`.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# --- The rubric ---------------------------------------------------------------
# Verbatim ROUTER_SYSTEM and ROUTER_Q2 from systemone/data.py. The calibration
# was fitted on these exact strings; a changed word is an uncalibrated router.
ROUTER_SYSTEM = ("You are the front-door router of a voice assistant running on the user's Linux laptop. "
                 "The assistant can answer from its own knowledge, search the web, and run shell commands. "
                 "The transcript comes from speech recognition and may contain errors. "
                 "Answer each question with exactly one word.")

ROUTER_QUESTIONS: List[Tuple[str, str]] = [
    ("addressed", "Is the speaker talking to the voice assistant, rather than to someone else in the room or thinking aloud?"),
    ("intelligible", "Taking the previous exchange into account, is it clear what the user wants?"),
    ("needs_web", "Taking the previous exchange into account, would acting on this turn require looking something up on the internet (news, weather, a specific person, company, product, or anything recent)?"),
    ("needs_shell", "Taking the previous exchange into account, would acting on this turn require running a command on this computer (inspecting or changing hardware, files, processes, services or settings)? A 'yes' or 'go ahead' that confirms a proposed command counts."),
    ("risky", "Taking the previous exchange into account, would acting on this turn change or disrupt the machine or the user's data (turn hardware off, delete or overwrite files, reboot, change passwords, settings or firmware)? Confirming such a proposal counts."),
    ("simple", "Is this a simple question or casual remark that a small local model can answer well without any tools? A confirmation of a proposed action is not simple."),
    ("followup", "Is this utterance a follow-up to the previous exchange rather than a new topic?"),
    ("question", "Is the user asking a question, as opposed to giving an instruction, confirming, or making a remark?"),
]
QUESTION_IDS = [q for q, _ in ROUTER_QUESTIONS]
REQUIRED_QUESTIONS = ("addressed", "intelligible", "needs_web", "needs_shell", "risky", "simple")
OPTIONS = ["yes", "no"]

# How the dataset was built: a previous exchange counts only when it happened
# within this many seconds, and its reply is cut to this many characters.
PREVIOUS_MAX_AGE = 900.0
PREVIOUS_REPLY_CHARS = 300

DEFAULT_THRESHOLDS = {"drop_junk": 0.8, "needs_web": 0.2, "needs_shell": 0.2,
                      "risky": 0.3, "simple_local": 0.7}
THRESHOLD_ENV = {"drop_junk": "VOICE_ASSISTANT_ROUTER_THR_DROP",
                 "needs_web": "VOICE_ASSISTANT_ROUTER_THR_WEB",
                 "needs_shell": "VOICE_ASSISTANT_ROUTER_THR_SHELL",
                 "risky": "VOICE_ASSISTANT_ROUTER_THR_RISKY",
                 "simple_local": "VOICE_ASSISTANT_ROUTER_THR_SIMPLE"}

USER_CALIBRATION = Path.home() / ".config/voice-assistant/router_calibration.json"
REPO_CALIBRATION = Path(__file__).with_name("router_calibration.json")
ROUTER_LOG = Path.home() / ".local/state/voice-assistant/router.jsonl"

# Lines the assistant prepends to the user's words ("[context: ...]",
# "[router: ...]"). They are in the conversation ledger, but were never said.
_MARKER_LINE = re.compile(r"^\[(?:context|router): [^\n]*\]\n?", re.MULTILINE)


def judge_url(url: str) -> str:
    """The /judge endpoint for a llama-server URL given in any of the usual
    shapes: with or without /v1, with or without /judge already on it."""
    base = url.strip().rstrip("/")
    base = re.sub(r"/judge$", "", base)
    base = re.sub(r"/v1$", "", base)
    return base + "/judge"


def strip_markers(text: str) -> str:
    return _MARKER_LINE.sub("", text).strip()


def build_context(text: str, typed: bool = False,
                  previous: Optional[Tuple[str, str, float]] = None) -> str:
    """The context string, byte for byte as the calibration set had it.

    `previous` is (user_text, assistant_text, age_seconds) of the last
    exchange, or None. It is included only when it is recent enough to be the
    same conversation; an empty or missing reply reads "(no reply)".
    """
    head = ""
    if previous is not None:
        prev_user, prev_reply, age = previous
        if age is not None and 0 <= age <= PREVIOUS_MAX_AGE:
            reply = strip_markers(prev_reply or "")
            head = (f"Previous exchange:\nUser: {strip_markers(prev_user or '')}\n"
                    f"Assistant: {reply[:PREVIOUS_REPLY_CHARS] if reply else '(no reply)'}\n\n")
    return head + f"Latest utterance ({'typed' if typed else 'spoken'}):\n{text.strip()}"


def calibrate(a: float, c: float, logit_yes: float, logit_no: float) -> float:
    """Platt-scaled P(yes) from the two option log-probs."""
    z = a * (logit_yes - logit_no) + c
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def load_thresholds(from_file: Optional[dict] = None, env=os.environ) -> Dict[str, float]:
    thr = dict(DEFAULT_THRESHOLDS)
    for k, v in (from_file or {}).items():
        if k in thr:
            try:
                thr[k] = float(v)
            except (TypeError, ValueError):
                pass
    for k, var in THRESHOLD_ENV.items():
        raw = env.get(var, "")
        if raw.strip():
            try:
                thr[k] = float(raw)
            except ValueError:
                pass
    return thr


@dataclass
class Verdict:
    """What the router concluded about one turn."""
    text: str
    typed: bool
    context: str
    probs: Dict[str, float]            # calibrated P(yes) per question
    raw: Dict[str, float]              # the engine's own P(yes) over the two options
    mass: Dict[str, float]             # probability the model put on either option
    thresholds: Dict[str, float]
    latency_ms: float
    server_ms: Optional[float] = None
    n_prefix_tokens: Optional[int] = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    at: float = field(default_factory=time.time)
    backend: Optional[str] = None      # what actually answered, filled in later
    recorded: bool = False

    # -- decisions ------------------------------------------------------------
    @property
    def junk(self) -> float:
        return 1.0 - min(self.probs["addressed"], self.probs["intelligible"])

    @property
    def drop(self) -> bool:
        return (not self.typed) and self.junk >= self.thresholds["drop_junk"]

    @property
    def needs_web(self) -> bool:
        return self.probs["needs_web"] >= self.thresholds["needs_web"]

    @property
    def needs_shell(self) -> bool:
        return self.probs["needs_shell"] >= self.thresholds["needs_shell"]

    @property
    def risky(self) -> bool:
        return self.probs["risky"] >= self.thresholds["risky"]

    @property
    def simple(self) -> bool:
        return self.probs["simple"] >= self.thresholds["simple_local"]

    @property
    def followup(self) -> bool:
        # descriptive only (no policy reads it); a rubric may leave it out
        return self.probs.get("followup", 0.0) >= 0.5

    @property
    def question(self) -> bool:
        return self.probs.get("question", 0.0) >= 0.5

    @property
    def route(self) -> str:
        if self.simple and not self.needs_web and not self.needs_shell and not self.risky:
            return "local"
        return "claude"

    @property
    def tools(self) -> List[str]:
        out = []
        if self.needs_web:
            out += ["web_search", "fetch_page"]
        if self.needs_shell:
            out.append("run_shell")
        return out

    def decisions(self) -> Dict[str, object]:
        return {"drop": self.drop, "needs_web": self.needs_web, "needs_shell": self.needs_shell,
                "risky": self.risky, "simple": self.simple, "followup": self.followup,
                "question": self.question, "route": self.route, "junk": round(self.junk, 4)}

    # -- presentation ---------------------------------------------------------
    def log_line(self) -> str:
        p = self.probs
        tools = {(True, True): "web+shell", (True, False): "web",
                 (False, True): "shell", (False, False): "none"}[(self.needs_web, self.needs_shell)]
        flags = " risky" if self.risky else ""
        short = {"addressed": "addressed", "intelligible": "intelligible", "needs_web": "web", "needs_shell": "shell",
                 "risky": "risky", "simple": "simple", "followup": "followup", "question": "question"}
        parts = " ".join(f"{short.get(k, k)}={v:.2f}" for k, v in p.items())
        return f"Router: {parts} -> route={self.route} tools={tools}{flags} ({self.latency_ms:.0f} ms)"

    def to_dict(self) -> Dict[str, object]:
        return {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.at)),
                "id": self.id, "text": self.text, "typed": self.typed, "context": self.context,
                "probs": {k: round(v, 4) for k, v in self.probs.items()},
                "raw": {k: round(v, 4) for k, v in self.raw.items()},
                "mass": {k: round(v, 4) for k, v in self.mass.items()},
                "decisions": self.decisions(), "thresholds": self.thresholds,
                "latency_ms": round(self.latency_ms, 1), "server_ms": self.server_ms,
                "n_prefix_tokens": self.n_prefix_tokens, "backend": self.backend}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class TurnRouter:
    """Asks the /judge endpoint about a turn and turns the answer into a Verdict.

    url: the llama-server (http://127.0.0.1:8081, with or without /v1 or /judge).
    calibration_path: an explicit calibration file; by default the user's copy
        under ~/.config/voice-assistant, then the one shipped beside this module.
    timeout: seconds for the whole request; past it the router has no opinion.
    """

    def __init__(self, url: str, calibration_path: Optional[str] = None, timeout: float = 1.5,
                 enabled: bool = True, log_path: Optional[Path] = None, logger=None):
        self.url = judge_url(url)
        self.health_url = self.url[: -len("/judge")] + "/health"
        self.timeout = float(timeout)
        self.enabled = bool(enabled)
        self.log_path = Path(log_path) if log_path is not None else ROUTER_LOG
        self.logger = logger
        self.session = requests.Session()
        self.calibration_path, cal = self._load_calibration(calibration_path)
        # The rubric is data: the calibration file names the questions (ids, wording, order) it was
        # fitted on, and those are what get asked. The module's ROUTER_QUESTIONS is the fallback
        # for a file without texts. The policies need addressed, intelligible, needs_web,
        # needs_shell, risky and simple; anything else is descriptive.
        self.calibration: Dict[str, Tuple[float, float]] = {}
        file_qs = cal.get("questions") or {}
        if file_qs and all(isinstance(v, dict) and v.get("text") for v in file_qs.values()):
            rubric = [(qid, v["text"]) for qid, v in file_qs.items()]
        else:
            rubric = [(qid, self._question_text(qid)) for qid, _ in ROUTER_QUESTIONS]
        missing = [q for q in REQUIRED_QUESTIONS if q not in dict(rubric)]
        if missing:
            self._warn(f"router calibration lacks {missing}; falling back to the built-in rubric")
            rubric = [(qid, self._question_text(qid)) for qid, _ in ROUTER_QUESTIONS]
        for qid, _ in rubric:
            q = file_qs.get(qid) or {}
            try:
                self.calibration[qid] = (float(q["a"]), float(q["c"]))
            except (KeyError, TypeError, ValueError):
                self._warn(f"router calibration has no fit for {qid!r}; using the raw answer")
                self.calibration[qid] = (1.0, 0.0)
        if cal.get("system") and cal["system"] != ROUTER_SYSTEM:
            self._warn("router calibration was fitted with a different system prompt")
        self.thresholds = load_thresholds(cal.get("thresholds"))
        self._questions = [{"id": qid, "text": text, "options": list(OPTIONS)} for qid, text in rubric]

    # -- setup ---------------------------------------------------------------
    @staticmethod
    def _question_text(qid: str) -> str:
        return dict(ROUTER_QUESTIONS)[qid] + " Answer yes or no."

    def _load_calibration(self, explicit: Optional[str]):
        candidates = [Path(explicit)] if explicit else [USER_CALIBRATION, REPO_CALIBRATION]
        last_err = None
        for path in candidates:
            try:
                data = json.loads(path.read_text())
                if not isinstance(data, dict) or "questions" not in data:
                    raise ValueError("not a router calibration file")
                return path, data
            except (OSError, ValueError) as e:
                last_err = e
        raise ValueError(f"no usable router calibration ({candidates[-1]}: {last_err})")

    def describe(self) -> str:
        thr = " ".join(f"{k}={v:g}" for k, v in self.thresholds.items())
        return f"{self.url} calibration={self.calibration_path} timeout={self.timeout:g}s {thr}"

    def _warn(self, msg: str):
        if self.logger is not None:
            self.logger.warning(msg)

    def _debug(self, msg: str):
        if self.logger is not None:
            self.logger.debug(msg)

    # -- the call --------------------------------------------------------------
    def healthy(self, timeout: float = 0.5) -> bool:
        try:
            return self.session.get(self.health_url, timeout=timeout).status_code == 200
        except requests.RequestException:
            return False

    def request_body(self, text: str, typed: bool = False,
                     previous: Optional[Tuple[str, str, float]] = None) -> dict:
        return {"mode": "chat", "system": ROUTER_SYSTEM,
                "context": build_context(text, typed=typed, previous=previous),
                "separator": "\n\n", "enable_thinking": False, "top_k": 0,
                "questions": self._questions}

    def judge(self, text: str, typed: bool = False,
              previous: Optional[Tuple[str, str, float]] = None) -> Optional[Verdict]:
        """One verdict, or None when the router has no opinion (disabled, down,
        slow, or answering something other than what was asked)."""
        if not self.enabled or not text or not text.strip():
            return None
        body = self.request_body(text, typed=typed, previous=previous)
        t0 = time.monotonic()
        try:
            r = self.session.post(self.url, json=body, timeout=self.timeout)
        except requests.RequestException as e:
            self._debug(f"router: no answer from {self.url} ({e.__class__.__name__})")
            return None
        latency_ms = (time.monotonic() - t0) * 1000.0
        if r.status_code != 200:
            self._debug(f"router: {self.url} answered {r.status_code}")
            return None
        try:
            data = r.json()
            probs, raw, mass = self._parse(data)
        except Exception as e:
            self._debug(f"router: unusable answer ({e.__class__.__name__}: {e})")
            return None
        timings = data.get("timings") if isinstance(data.get("timings"), dict) else {}
        server_ms = timings.get("total_ms")
        return Verdict(text=text.strip(), typed=bool(typed), context=body["context"],
                       probs=probs, raw=raw, mass=mass, thresholds=dict(self.thresholds),
                       latency_ms=latency_ms,
                       server_ms=float(server_ms) if isinstance(server_ms, (int, float)) else None,
                       n_prefix_tokens=data.get("n_prefix_tokens"))

    def _parse(self, data: dict):
        by_id = {q.get("id"): q for q in data["questions"]}
        probs, raw, mass = {}, {}, {}
        for qid in QUESTION_IDS:
            q = by_id[qid]
            opts = {o["text"]: o for o in q["options"]}
            ly, ln = float(opts["yes"]["logprob"]), float(opts["no"]["logprob"])
            a, c = self.calibration[qid]
            probs[qid] = calibrate(a, c, ly, ln)
            raw[qid] = float(opts["yes"].get("prob", math.exp(ly) / (math.exp(ly) + math.exp(ln))))
            m = q.get("mass")
            mass[qid] = float(m) if isinstance(m, (int, float)) else float("nan")
        return probs, raw, mass

    # -- the record --------------------------------------------------------------
    def record(self, verdict: Verdict, backend: Optional[str] = None) -> bool:
        """Append the verdict to router.jsonl, once. Never raises."""
        if verdict is None or verdict.recorded:
            return False
        if backend is not None:
            verdict.backend = backend
        verdict.recorded = True
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(verdict.to_json() + "\n")
            return True
        except Exception as e:
            self._debug(f"router: could not write {self.log_path} ({e})")
            return False
