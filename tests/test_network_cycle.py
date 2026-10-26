"""The swap key across every model server on the LAN (voice_assistant.py:
_network_servers, _next_stop, _switch_backend, _discover_network_servers) and
the scanner both it and `voice-llm discover` use (llm_discovery.py). No real
network: every server, health check and scan is a stand-in.

    .venv/bin/python -m pytest tests
"""
import asyncio
import contextlib
import io
import ipaddress
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import llm_discovery  # noqa: E402
import voice_assistant as va  # noqa: E402

va.VOICE_UNLOCK = False

CONFIGURED = "http://llm.lan:8080/v1"          # by name, at 192.168.1.105
FLASH = "http://192.168.1.22:8080/v1"
NEMOTRON = "http://192.168.1.22:8082/v1"
QWEN_4060 = "http://192.168.1.112:8080/v1"


class CycleHost(va.VoiceAssistant):
    """The backend choice alone: no audio, no CLI, no servers."""
    def __init__(self, state_dir, local="absent", claude=True):
        self.logger = logging.getLogger("test-network-cycle")
        self.state_dir = state_dir
        self.backend, self.llm_tier, self.backend_pref = "claude", "local", "auto"
        self._status_state = "ready"
        self.is_active = False
        self.is_processing = False
        self._dsh = None
        self._dsh_available = False
        self._claude_available = claude
        self._claude = claude
        self._router_verdict = None
        self._local_state = local
        self._local_history = [{"role": "user", "content": "hi"},
                               {"role": "assistant", "content": "hello"}]
        self._network_rescan = va.threading.Event()
        self.voice_only, self._voiceprint, self._session_id = False, None, "s"
        self.notes = []

    def _local_tier_state(self):
        return self._local_state, f"local is {self._local_state}"

    def _append_transcript(self, role, text): pass
    def _notify(self, text, **k): self.notes.append(text)


@pytest.fixture
def lan(monkeypatch):
    """A configured server plus three found on the LAN, all answering. The
    returned dict is the servers' health by URL, and what was written."""
    world = {"health": {CONFIGURED: "ready", FLASH: "ready", NEMOTRON: "ready",
                        QWEN_4060: "ready"},
             "models": {CONFIGURED: "qwen3.8-flash-next", FLASH: "qwen3.8-flash-next",
                        NEMOTRON: "nemotron-3.5-super-vl", QWEN_4060: "qwen3.8-27b"},
             "written": [], "asked": []}
    monkeypatch.setattr(va, "NETWORK_LLM_CONFIGURED", (CONFIGURED, "qwen3.8-flash-next"))
    monkeypatch.setattr(va, "NETWORK_LLM_URL", CONFIGURED)
    monkeypatch.setattr(va, "NETWORK_LLM_MODEL", "qwen3.8-flash-next")
    monkeypatch.setattr(va, "_DISCOVERED_SERVERS", [
        llm_discovery.Server(FLASH, "qwen3.8-flash-next", ""),
        llm_discovery.Server(NEMOTRON, "nemotron-3.5-super-vl", ""),
        llm_discovery.Server(QWEN_4060, "qwen3.8-27b", "llm-4060.lan"),
    ])
    monkeypatch.setattr(va, "_HOST_ADDRESSES", {"llm.lan": "192.168.1.105"})
    monkeypatch.setattr(va, "_SERVED_MODELS", {})
    monkeypatch.setattr(va, "NETWORK_HARNESS", "native")
    monkeypatch.setattr(va, "LOCAL_HARNESS", "native")
    monkeypatch.setattr(va, "LLM_FALLBACK", True)
    monkeypatch.setattr(va.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(va, "_write_env_setting",
                        lambda key, value: world["written"].append((key, value)))

    def health(url, timeout=1.0):
        world["asked"].append((url, timeout))
        return world["health"].get(url, "down")
    monkeypatch.setattr(va, "_llm_health", health)
    monkeypatch.setattr(va, "_detect_served_model",
                        lambda tier, timeout=1.0: tier.model or world["models"].get(tier.url, ""))
    monkeypatch.setattr(va, "_start_user_unit", lambda unit: True)

    def no_scan(*a, **k):
        raise AssertionError("a keypress must never scan the LAN")
    monkeypatch.setattr(llm_discovery, "scan", no_scan)
    return world


def press(host, n):
    """Press the swap key n times: where each press landed."""
    out = []
    for _ in range(n):
        res = host._switch_backend("toggle")
        assert res["ok"], res
        out.append(va.NETWORK_LLM_URL if res["tier"] == "network" else res["tier"])
    return out


# --- the cycle ---------------------------------------------------------------

def test_every_server_is_a_stop_configured_first_then_by_address(tmp_path, lan):
    host = CycleHost(tmp_path, local="ready")
    assert [s.url for s in va._network_servers()] == [CONFIGURED, FLASH, NEMOTRON, QWEN_4060]
    assert press(host, 7) == ["local", CONFIGURED, FLASH, NEMOTRON, QWEN_4060, "claude", "local"]


def test_a_server_found_by_address_and_configured_by_name_is_one_stop(tmp_path, lan, monkeypatch):
    monkeypatch.setattr(va, "_DISCOVERED_SERVERS", va._DISCOVERED_SERVERS
                        + [llm_discovery.Server("http://192.168.1.105:8080/v1", "x", "llm")])
    assert [s.url for s in va._network_servers()] == [CONFIGURED, FLASH, NEMOTRON, QWEN_4060]
    assert va._network_servers()[0].name == "llm", "its reverse DNS still labels it"


def test_one_server_behaves_as_before(tmp_path, lan, monkeypatch):
    monkeypatch.setattr(va, "_DISCOVERED_SERVERS", [])
    host = CycleHost(tmp_path, local="ready")
    assert press(host, 4) == ["local", CONFIGURED, "claude", "local"]
    assert lan["written"] == [("VOICE_ASSISTANT_LLM_BACKEND", v)
                              for v in ("local", "network", "claude", "local")]
    host._switch_backend("network")
    assert host._backend_label() == "Network · qwen3.8-flash-next", "no server name with one"


def test_each_press_asks_the_new_server_whether_it_sees(tmp_path, lan, monkeypatch):
    """Whether a model takes images (and so gets `look`) is cached per server;
    a press forgets the answer for where it lands, which may now serve another
    model, and asks nothing on the key itself."""
    bases = [u.removesuffix("/v1") for u in (CONFIGURED, FLASH, NEMOTRON, QWEN_4060)]
    monkeypatch.setattr(va, "_VISION", {b: (b.endswith(":8082"), 0.0, True) for b in bases})
    monkeypatch.setattr(va, "_server_vision", lambda *a, **k: pytest.fail("no /props on a keypress"))
    host = CycleHost(tmp_path, local="ready")
    press(host, 1)                              # local
    for landed in press(host, 4):               # each network server
        assert landed.removesuffix("/v1") not in va._VISION
    assert set(va._VISION) == set(), "all four asked afresh on their next turn"


def test_no_network_server_at_all(tmp_path, lan, monkeypatch):
    monkeypatch.setattr(va, "_DISCOVERED_SERVERS", [])
    monkeypatch.setattr(va, "NETWORK_LLM_CONFIGURED", ("", ""))
    monkeypatch.setattr(va, "NETWORK_LLM_URL", "")
    host = CycleHost(tmp_path, local="ready")
    assert press(host, 3) == ["local", "claude", "local"]
    res = host._switch_backend("network")
    assert not res["ok"] and "no network model" in res["error"]


def test_a_server_that_does_not_answer_is_skipped(tmp_path, lan):
    lan["health"][FLASH] = "down"
    host = CycleHost(tmp_path, local="absent")
    assert press(host, 5) == [CONFIGURED, NEMOTRON, QWEN_4060, "claude", CONFIGURED]
    assert all(t == va.NETWORK_SWAP_TIMEOUT for u, t in lan["asked"] if u == FLASH), \
        "only the short check"


def test_a_loading_server_is_offered_with_the_claude_note(tmp_path, lan):
    lan["health"][NEMOTRON] = "loading"
    host = CycleHost(tmp_path)
    host._switch_backend("network")
    host._switch_backend("toggle")                   # -> FLASH
    res = host._switch_backend("toggle")             # -> NEMOTRON, still loading
    assert va.NETWORK_LLM_URL == NEMOTRON and res["tier"] == "network"
    assert res["note"] == "(loading, Claude answers meanwhile)"


def test_without_local_or_claude_it_cycles_the_network_servers(tmp_path, lan, monkeypatch):
    host = CycleHost(tmp_path, local="absent", claude=False)
    monkeypatch.setattr(va.shutil, "which", lambda name: None)
    host.backend, host.llm_tier = "local", "network"
    assert press(host, 5) == [FLASH, NEMOTRON, QWEN_4060, CONFIGURED, FLASH]


def test_nothing_else_available_stays_put(tmp_path, lan):
    for url in lan["health"]:
        lan["health"][url] = "down"
    host = CycleHost(tmp_path, local="absent")
    assert press(host, 2) == ["claude", "claude"]


def test_a_swap_asks_for_a_fresh_scan_but_never_waits_for_one(tmp_path, lan):
    host = CycleHost(tmp_path)
    assert not host._network_rescan.is_set()
    host._switch_backend("toggle")
    assert host._network_rescan.is_set(), "the discovery thread is woken"


# --- what is written, shown and kept -----------------------------------------

def test_another_server_is_written_to_the_env_file(tmp_path, lan):
    host = CycleHost(tmp_path, local="absent")
    host._switch_backend("network")
    assert ("VOICE_ASSISTANT_NETWORK_URL", CONFIGURED) not in lan["written"], "same server"
    host._switch_backend("toggle")
    assert lan["written"][-3:] == [("VOICE_ASSISTANT_NETWORK_URL", FLASH),
                                   ("VOICE_ASSISTANT_NETWORK_MODEL", "qwen3.8-flash-next"),
                                   ("VOICE_ASSISTANT_LLM_BACKEND", "network")]
    for _ in range(3):
        host._switch_backend("toggle")               # NEMOTRON, QWEN_4060, claude, then:
    host._switch_backend("toggle")
    assert va.NETWORK_LLM_URL == CONFIGURED
    assert lan["written"][-3:-1] == [("VOICE_ASSISTANT_NETWORK_URL", CONFIGURED),
                                     ("VOICE_ASSISTANT_NETWORK_MODEL", "qwen3.8-flash-next")]


def test_the_env_file_keeps_its_other_lines(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    env = tmp_path / ".config/voice-assistant/env"
    env.parent.mkdir(parents=True)
    env.write_text("VOICE_ASSISTANT_LLM_BACKEND=network\n"
                   f"VOICE_ASSISTANT_NETWORK_URL={CONFIGURED}\n"
                   "VOICE_ASSISTANT_NETWORK_MODEL=qwen3.8-flash-next\n"
                   "VOICE_ASSISTANT_LOCAL_THINK=0\n")
    monkeypatch.setattr(va, "NETWORK_LLM_CONFIGURED", (CONFIGURED, "qwen3.8-flash-next"))
    monkeypatch.setattr(va, "NETWORK_LLM_URL", CONFIGURED)
    monkeypatch.setattr(va, "NETWORK_LLM_MODEL", "qwen3.8-flash-next")
    va._use_network_server(va.NetworkServer(QWEN_4060, "qwen3.8-27b", "llm-4060"))
    assert env.read_text() == ("VOICE_ASSISTANT_LLM_BACKEND=network\n"
                               f"VOICE_ASSISTANT_NETWORK_URL={QWEN_4060}\n"
                               "VOICE_ASSISTANT_NETWORK_MODEL=qwen3.8-27b\n"
                               "VOICE_ASSISTANT_LOCAL_THINK=0\n")
    assert (va.NETWORK_LLM_URL, va.NETWORK_LLM_MODEL) == (QWEN_4060, "qwen3.8-27b")
    # Back to the configured one: named by what it serves now, which may no
    # longer be what the env file was written for...
    va._use_network_server(va.NetworkServer(CONFIGURED, "nemotron-3.5-super-vl", ""))
    assert va.NETWORK_LLM_MODEL == "nemotron-3.5-super-vl"
    # ...and by the configured name only while it has not said.
    va._use_network_server(va.NetworkServer(CONFIGURED, "", ""))
    assert va.NETWORK_LLM_MODEL == "qwen3.8-flash-next"


def test_the_label_names_the_server(tmp_path, lan):
    host = CycleHost(tmp_path, local="absent")
    host._switch_backend("network")
    assert host._backend_label() == "Network · qwen3.8-flash-next · llm"
    for _ in range(3):
        res = host._switch_backend("toggle")
    assert res["label"] == "Network · qwen3.8-27b · llm-4060", "reverse DNS, domain dropped"
    assert host.notes[-1] == "🔀 Network · qwen3.8-27b · llm-4060"
    assert host._backend_label(spoken=True) == "the network model", "short out loud"
    host._switch_backend("toggle")                   # claude
    host._switch_backend("toggle")                   # configured
    host._switch_backend("toggle")                   # FLASH: no name, and .22 serves two
    assert host._backend_label() == "Network · qwen3.8-flash-next · 192.168.1.22:8080"


def test_a_new_server_is_named_by_what_it_serves_now(tmp_path, lan):
    lan["models"][FLASH] = "qwen3.9-flash"           # reloaded since the scan
    host = CycleHost(tmp_path, local="absent")
    host._switch_backend("network")
    host._switch_backend("toggle")
    assert va.NETWORK_LLM_MODEL == "qwen3.9-flash" and host._llm().model == "qwen3.9-flash"


def test_the_conversation_carries_across_servers(tmp_path, lan):
    host = CycleHost(tmp_path, local="absent")
    history = list(host._local_history)
    host._switch_backend("network")
    host._switch_backend("toggle")
    assert host._local_history == history, "the native loop's history is every server's"
    assert host._llm().url == FLASH


def test_with_none_configured_network_picks_the_first_that_answers(tmp_path, lan, monkeypatch):
    monkeypatch.setattr(va, "NETWORK_LLM_CONFIGURED", ("", ""))
    monkeypatch.setattr(va, "NETWORK_LLM_URL", "")
    monkeypatch.setattr(va, "NETWORK_LLM_MODEL", "")
    lan["health"][FLASH] = "down"
    host = CycleHost(tmp_path)
    res = host._switch_backend("network")
    assert res["ok"] and va.NETWORK_LLM_URL == NEMOTRON
    assert ("VOICE_ASSISTANT_NETWORK_URL", NEMOTRON) in lan["written"]


def test_status_lists_the_servers(tmp_path, lan, monkeypatch):
    monkeypatch.setattr(va, "_DISCOVERED_AT", va.time.time() - 120)
    monkeypatch.setattr(va, "NETWORK_DISCOVER_INTERVAL", 300)
    monkeypatch.setattr(va, "_local_llm_health", lambda timeout=1.0: "down")
    host = CycleHost(tmp_path)
    host._switch_backend("network")
    r = asyncio.run(host._control_command({"cmd": "status"}))
    assert r["network_url"] == CONFIGURED and r["network_llm"] == "ready"
    assert [s["url"] for s in r["network_servers"]] == [CONFIGURED, FLASH, NEMOTRON, QWEN_4060]
    assert [s["current"] for s in r["network_servers"]] == [True, False, False, False]
    assert r["network_servers"][0]["configured"] and r["network_servers"][3]["label"] == "llm-4060"
    assert 119 <= r["network_scan"] <= 125


# --- discovery, in the background -------------------------------------------

def test_discovery_replaces_the_list_and_learns_the_models(tmp_path, lan, monkeypatch):
    seen = {}

    def scan(ports, exclude=()):
        seen.update(ports=ports, exclude=list(exclude))
        return [llm_discovery.Server(QWEN_4060, "qwen3.8-27b", "llm-4060")]
    monkeypatch.setattr(llm_discovery, "scan", scan)
    monkeypatch.setattr(llm_discovery.socket, "gethostbyname",
                        lambda name: {"llm.lan": "192.168.1.105"}[name])
    monkeypatch.setattr(va, "_HOST_ADDRESSES", {})
    monkeypatch.setattr(va, "_DISCOVERED_AT", 0.0)
    monkeypatch.setattr(va, "NETWORK_DISCOVER_PORTS", (8080, 8081, 8082))
    host = CycleHost(tmp_path)
    assert host._discover_network_servers()
    assert seen == {"ports": (8080, 8081, 8082), "exclude": [va.LOCAL_LLM_URL]}
    assert [s.url for s in va._network_servers()] == [CONFIGURED, QWEN_4060], "FLASH went away"
    assert va._SERVED_MODELS[QWEN_4060] == "qwen3.8-27b" and va._DISCOVERED_AT > 0
    assert va._HOST_ADDRESSES == {"llm.lan": "192.168.1.105"}, "the configured name, resolved"

    def broken(*a, **k):
        raise RuntimeError("no default route, so no LAN to look on")
    monkeypatch.setattr(llm_discovery, "scan", broken)
    assert not host._discover_network_servers()
    assert [s.url for s in va._network_servers()] == [CONFIGURED, QWEN_4060], "kept"


def test_a_server_that_changed_models_is_named_by_the_new_one(tmp_path, lan, monkeypatch):
    # The cluster went from Flash-Next to Nemotron on the same URL; the env
    # file still says Flash-Next.
    monkeypatch.setattr(llm_discovery, "scan", lambda ports, exclude=(): [
        llm_discovery.Server("http://192.168.1.105:8080/v1", "nemotron-3.5-super-vl", "")])
    monkeypatch.setattr(llm_discovery.socket, "gethostbyname", lambda name: "192.168.1.105")
    monkeypatch.setattr(va, "_DISCOVERED_SERVERS", [])
    host = CycleHost(tmp_path, local="absent")
    host._switch_backend("network")
    assert host._backend_label() == "Network · qwen3.8-flash-next"
    assert host._discover_network_servers()
    assert host._backend_label() == "Network · nemotron-3.5-super-vl"
    assert va._llm_tier("network").model == "nemotron-3.5-super-vl", "what it is told it is"
    assert ("VOICE_ASSISTANT_NETWORK_MODEL", "nemotron-3.5-super-vl") in lan["written"]
    assert va._network_servers()[0].model == "nemotron-3.5-super-vl"


# --- the scanner -------------------------------------------------------------

@pytest.mark.parametrize("text, want", [
    ("", (8080, 8081, 8082)),
    ("8080 8081", (8080, 8081)),
    ("8082,8080, 8082", (8082, 8080)),
    ("junk 0 70000", (8080, 8081, 8082)),
])
def test_ports(text, want):
    assert llm_discovery.ports_from(text) == want


@pytest.mark.parametrize("data, want", [
    ({"object": "list", "data": [{"id": "qwen3.8-27b"}]}, "qwen3.8-27b"),
    ({"models": [{"name": "qwen3.8-27b", "model": "qwen3.8-27b"}], "data": []}, "qwen3.8-27b"),
    ({"data": [{"id": ""}, {"id": "second"}]}, "second"),
    ({"data": []}, ""),
    ({"error": {"code": 503, "message": "Loading model"}}, ""),
    (["not", "a", "dict"], ""),
    ({"data": "nonsense"}, ""),
])
def test_models_reply(data, want):
    assert llm_discovery.parse_models(data) == want


def test_targets_leave_out_this_machine_and_the_local_server():
    nets = {ipaddress.ip_network("192.168.1.0/24")}
    mine = {ipaddress.ip_address("192.168.1.44")}
    skip = {("192.168.1.10", 8081)}
    t = llm_discovery.targets(nets, mine, (8080, 8081), skip)
    assert len(t) == 254 * 2 - 2 - 1
    assert ("192.168.1.44", 8080) not in t and ("192.168.1.10", 8081) not in t
    assert ("192.168.1.10", 8080) in t
    assert t[:3] == [("192.168.1.1", 8080), ("192.168.1.1", 8081), ("192.168.1.2", 8080)]


def test_endpoint():
    assert llm_discovery.endpoint("http://192.168.1.22:8082/v1") == ("192.168.1.22", 8082)
    assert llm_discovery.endpoint("http://10.0.0.1/v1") == ("10.0.0.1", 80)
    assert llm_discovery.endpoint("https://llm.lan/v1", resolve=False) == ("llm.lan", 443)
    assert llm_discovery.endpoint("") is None


def test_scan_reports_servers_that_name_a_model(monkeypatch):
    monkeypatch.setattr(llm_discovery, "lan", lambda: (
        {ipaddress.ip_address("192.168.1.44"), ipaddress.ip_address("127.0.0.1")},
        {ipaddress.ip_network("192.168.1.0/24")}))
    listening = {("192.168.1.22", 8080), ("192.168.1.22", 8082), ("192.168.1.112", 8080),
                 ("192.168.1.30", 8081), ("192.168.1.44", 8081)}
    tried = []

    def open_port(t, timeout=0.3):
        tried.append(t)
        return t if t in listening else None
    monkeypatch.setattr(llm_discovery, "open_port", open_port)
    models = {"http://192.168.1.22:8082": "nemotron", "http://192.168.1.112:8080": "qwen3.8-27b",
              "http://192.168.1.30:8081": "local-one"}
    monkeypatch.setattr(llm_discovery, "served_model", lambda base, timeout=2: models.get(base, ""))
    monkeypatch.setattr(llm_discovery, "host_names",
                        lambda ips, timeout=1: {"192.168.1.112": "llm-4060.lan"})
    found = llm_discovery.scan((8080, 8081, 8082), exclude=["http://192.168.1.30:8081/v1"])
    assert found == [
        llm_discovery.Server("http://192.168.1.22:8082/v1", "nemotron", ""),
        llm_discovery.Server("http://192.168.1.112:8080/v1", "qwen3.8-27b", "llm-4060.lan"),
    ], "address order; a silent port and the excluded local server are not servers"
    assert ("192.168.1.44", 8081) not in tried and ("192.168.1.30", 8081) not in tried


def test_scan_without_a_lan(monkeypatch):
    monkeypatch.setattr(llm_discovery, "lan", lambda: (set(), set()))
    with pytest.raises(RuntimeError, match="no LAN"):
        llm_discovery.scan()


def test_discover_prints_what_setup_sh_reads(monkeypatch):
    got = {}

    def scan(ports, exclude=()):
        got.update(ports=ports, exclude=exclude)
        return [llm_discovery.Server("http://192.168.1.112:8080/v1", "qwen3.8-27b", ""),
                llm_discovery.Server("http://192.168.1.22:8082/v1", "nemotron", "cluster.lan")]
    monkeypatch.setattr(llm_discovery, "scan", scan)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = llm_discovery.main(["--exclude", "http://127.0.0.1:8081/v1", "--exclude", "",
                                 "8080", "8082"])
    assert rc == 0 and got == {"ports": (8080, 8082), "exclude": ["http://127.0.0.1:8081/v1"]}
    assert out.getvalue() == ("http://192.168.1.112:8080/v1 qwen3.8-27b -\n"
                              "http://192.168.1.22:8082/v1 nemotron cluster.lan\n")
