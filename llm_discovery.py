"""Model servers on the LAN: OpenAI-compatible endpoints that name a model.

Shared by `voice-llm discover` (which setup.sh reads to offer a network model)
and voice_assistant.py (whose swap key cycles through every server it knows),
so there is one scanner. Standard library only: `voice-llm` runs it with the
system python3, before or without the venv.

Polite on purpose: the /24 of the interface the default route uses (a docker
or VM bridge is not the LAN), a few ports, a short connect timeout, and an
HTTP request only to a port that answered. A host counts when its /v1/models
names a model, so a llama-server that is still loading (503 on everything)
turns up on the next scan instead. This machine is left out, and so is any
endpoint the caller names (its configured local server, wherever that is).
"""
import argparse
import concurrent.futures as cf
import ipaddress
import json
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from typing import NamedTuple

# 8082 as well: a second server on a machine that already serves one (the
# cluster's Nemotron sits beside its Flash-Next).
DEFAULT_PORTS = (8080, 8081, 8082)
CONNECT_TIMEOUT = 0.3
HTTP_TIMEOUT = 2.0
# Reverse DNS can hang where there is none; a name is only a label.
NAME_TIMEOUT = 1.0


class Server(NamedTuple):
    url: str        # http://<ip>:<port>/v1
    model: str      # the first id its /v1/models lists
    name: str       # reverse DNS, "" when there is none


def ports_from(text) -> tuple:
    """'8080 8081', '8080,8081' -> (8080, 8081); nothing usable -> DEFAULT_PORTS."""
    ports = []
    for word in str(text or "").replace(",", " ").split():
        try:
            port = int(word)
        except ValueError:
            continue
        if 0 < port < 65536 and port not in ports:
            ports.append(port)
    return tuple(ports) or DEFAULT_PORTS


def _addresses(*args) -> list:
    out = subprocess.run(["ip", "-4", "-o", *args], capture_output=True, text=True).stdout
    return [ipaddress.ip_interface(w[i + 1]) for w in map(str.split, out.splitlines())
            for i, x in enumerate(w) if x == "inet"]


def lan():
    """(this machine's addresses, the networks to look on). No default route,
    no networks."""
    mine = {a.ip for a in _addresses("addr", "show")}
    routes = subprocess.run(["ip", "-4", "route", "show", "default"],
                            capture_output=True, text=True).stdout
    devs = {w[w.index("dev") + 1] for w in map(str.split, routes.splitlines())
            if "dev" in w[:-1]}
    ifaces = [a for d in devs for a in _addresses("addr", "show", "dev", d, "scope", "global")]
    # Never wider than a /24, and never wider than the interface's own subnet.
    nets = {ipaddress.ip_interface(f"{a.ip}/{max(a.network.prefixlen, 24)}").network
            for a in ifaces}
    return mine, nets


def endpoint(url: str, resolve: bool = True):
    """(address, port) of a URL, the host resolved when it is a name; None
    when the URL has no host."""
    u = urllib.parse.urlsplit(url or "")
    if not u.hostname:
        return None
    host = u.hostname
    if resolve:
        try:
            ipaddress.ip_address(host)
        except ValueError:
            try:
                host = socket.gethostbyname(host)
            except OSError:
                pass
    try:
        port = u.port
    except ValueError:
        port = None
    return host, port or (443 if u.scheme == "https" else 80)


def targets(nets, mine, ports, skip=()) -> list:
    """Every (address, port) to try, in address order: hosts of `nets` that
    are not this machine, minus the endpoints in `skip`."""
    skip = set(skip)
    hosts = sorted({h for n in nets for h in n.hosts() if h not in mine})
    return [(str(h), p) for h in hosts for p in ports if (str(h), p) not in skip]


def parse_models(data) -> str:
    """The first model a /v1/models reply names, "" when it names none.
    llama-server and vLLM say `data: [{id}]`; some servers `models: [{name}]`."""
    if not isinstance(data, dict):
        return ""
    items = data.get("data") or data.get("models") or []
    ids = [m.get("id") or m.get("model") or m.get("name")
           for m in items if isinstance(m, dict)] if isinstance(items, list) else []
    return next((str(i) for i in ids if i), "")


def served_model(base: str, timeout: float = HTTP_TIMEOUT, api_key: str = "none") -> str:
    """What the server at `base` (no /v1) lists first, "" when it does not answer."""
    try:
        req = urllib.request.Request(base.rstrip("/") + "/v1/models",
                                     headers={"Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return parse_models(json.loads(r.read().decode("utf-8", "replace")))
    except Exception:
        return ""


def open_port(target, timeout: float = CONNECT_TIMEOUT):
    try:
        with socket.create_connection(target, timeout=timeout):
            return target
    except OSError:
        return None


def host_names(addresses, timeout: float = NAME_TIMEOUT) -> dict:
    """Reverse DNS for each address, as far as it gets within `timeout`.

    Daemon threads rather than a pool: a lookup that hangs must not hold up
    the caller, nor the exit of the process that asked.
    """
    names = {}

    def look(ip):
        try:
            names[ip] = socket.gethostbyaddr(ip)[0]
        except (OSError, UnicodeError):
            pass

    threads = [threading.Thread(target=look, args=(ip,), daemon=True) for ip in set(addresses)]
    for t in threads:
        t.start()
    deadline = time.monotonic() + timeout
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))
    return dict(names)


def scan(ports=DEFAULT_PORTS, exclude=(), connect_timeout: float = CONNECT_TIMEOUT,
         http_timeout: float = HTTP_TIMEOUT, name_timeout: float = NAME_TIMEOUT) -> list:
    """The model servers on the LAN, in address then port order. `exclude`
    holds URLs to leave out (this machine's own server). Raises RuntimeError
    when there is no LAN to look on."""
    mine, nets = lan()
    if not nets:
        raise RuntimeError("no default route, so no LAN to look on")
    skip = {e for e in (endpoint(u) for u in exclude) if e}
    todo = targets(nets, mine, ports, skip)
    # About a second and a half for a /24 on three ports.
    with cf.ThreadPoolExecutor(128) as ex:
        found = [t for t in ex.map(lambda t: open_port(t, connect_timeout), todo) if t]
        models = list(ex.map(lambda t: served_model(f"http://{t[0]}:{t[1]}", http_timeout), found))
    live = [(t, m) for t, m in zip(found, models) if m]
    names = host_names([t[0] for t, _ in live], name_timeout)
    return [Server(f"http://{t[0]}:{t[1]}/v1", m, names.get(t[0], "")) for t, m in live]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="voice-llm discover",
        description="OpenAI-compatible model servers on this machine's /24, one "
                    "'<url> <model> <host name>' line each, in address order "
                    "('-' when the host has no name).")
    p.add_argument("ports", nargs="*", help=f"ports to try (default {' '.join(map(str, DEFAULT_PORTS))})")
    p.add_argument("--exclude", action="append", default=[], metavar="URL",
                   help="an endpoint to leave out, e.g. the configured local server")
    args = p.parse_args(argv)
    try:
        servers = scan(ports_from(" ".join(args.ports)), [u for u in args.exclude if u])
    except RuntimeError as e:
        print(f"discover: {e}", file=sys.stderr)
        return 1
    for s in servers:
        print(f"{s.url} {s.model} {s.name or '-'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
