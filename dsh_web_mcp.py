#!/usr/bin/env python3
"""The assistant's web tools, served to the DeepSeek Harness over MCP.

The harness's minimal profile gives the model a shell and a file editor and
nothing for the web; its own web_search needs an Exa, Perplexity or DeepSeek
API key. This sidecar exposes the assistant's keyless `web_search` and
`fetch_page` (web_tools.py) as an MCP server over stdio, and the harness is
told about it by the patch voice_assistant.py writes into its home. Inside the
harness the tools are named `mcp__web__web_search` and `mcp__web__fetch_page`.

Speaks the MCP JSON-RPC 2.0 stdio transport: one JSON object per line, no
framing headers. Only the methods a tool server needs are implemented.
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import web_tools  # noqa: E402

# stdout is the wire, so it must flush per message; stderr is the log.
sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="dsh-web-mcp %(levelname)s %(message)s")
log = logging.getLogger("dsh-web-mcp")

TOOLS = [{
    "name": "web_search",
    "description": (
        "Search the web and get back titles and short descriptions, already "
        "ranked and filtered. Use this for anything you do not know, anything "
        "recent, and any person, company or product you cannot place. It "
        "queries several sources at once and tells you plainly when there is "
        "nothing, so an empty result means the thing is genuinely obscure -- "
        "say so rather than guessing. Quote an exact phrase to pin it down, "
        "and add a word of context (a place, a field) for a name."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "What to search for."}},
        "required": ["query"],
    },
}, {
    "name": "fetch_page",
    "description": (
        "Fetch one web page and return its readable text, with the markup "
        "removed. Use it after web_search when a result looks like it holds "
        "the detail you need. Give a full URL including https://."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "Full URL of the page."}},
        "required": ["url"],
    },
}]


def call_tool(name: str, args: dict) -> str:
    if name == "web_search":
        return web_tools.web_search(str(args.get("query", "")), log)
    if name == "fetch_page":
        return web_tools.fetch_page(str(args.get("url", "")), log)
    raise KeyError(name)


def handle(req: dict):
    """Return (result, error) for one request."""
    method = req.get("method", "")
    params = req.get("params") or {}
    if method == "initialize":
        return {
            "protocolVersion": params.get("protocolVersion", "2025-06-18"),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "voice-assistant-web", "version": "1.0"},
        }, None
    if method == "ping":
        return {}, None
    if method == "tools/list":
        return {"tools": TOOLS}, None
    if method == "tools/call":
        name = str(params.get("name", ""))
        args = params.get("arguments") or {}
        try:
            text = call_tool(name, args if isinstance(args, dict) else {})
        except KeyError:
            return None, {"code": -32602, "message": f"unknown tool {name!r}"}
        except Exception as e:  # a failed fetch is a tool result, not a protocol error
            log.warning(f"{name} failed: {e}")
            return {"content": [{"type": "text", "text": f"{name} failed: {e}"}],
                    "isError": True}, None
        return {"content": [{"type": "text", "text": text}], "isError": False}, None
    return None, {"code": -32601, "message": f"method not found: {method}"}


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(req, dict):
            continue
        if "id" not in req:          # notifications (initialized, cancelled) need no answer
            continue
        result, error = handle(req)
        reply = {"jsonrpc": "2.0", "id": req["id"]}
        if error is not None:
            reply["error"] = error
        else:
            reply["result"] = result
        sys.stdout.write(json.dumps(reply) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
