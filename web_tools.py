"""Web search and page fetch for the local backends.

Shared by voice_assistant.py (the native tool loop hands these to the model
as `web_search` / `fetch_page`) and dsh_web_mcp.py (the same two tools served
to the DeepSeek Harness over MCP), so there is one implementation and one set
of empirically tuned thresholds.

These exist because the model kept trying to search by hand and could not. In
one session 18 of its 30 commands were curl-and-grep against Bing, DuckDuckGo
and Google, and they produced almost nothing: the DuckDuckGo endpoints answer a
bot challenge, and Bing does return real results but in markup no one-shot
regex is going to match. It burned all five tool calls guessing and then told
the user a real, well-covered company did not exist.
"""
import html
import json
import logging
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

# How long a search or page fetch may take, and how much of a page comes back.
WEB_TIMEOUT = float(os.getenv("VOICE_ASSISTANT_WEB_TIMEOUT", "12"))
WEB_RESULTS = int(os.getenv("VOICE_ASSISTANT_WEB_RESULTS", "6"))
WEB_PAGE_CHARS = int(os.getenv("VOICE_ASSISTANT_WEB_PAGE_CHARS", "4000"))
WEB_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
          "Chrome/120.0.0.0 Safari/537.36")
# Words too common to say anything about whether a result matches the query.
WEB_STOPWORDS = frozenset(
    "a an the of and or in on at to for from by with about is are was were be been being "
    "do does did what who whom which how why when where can could would should will shall "
    "i you it me my your our their his her its this that these those there here as if then "
    "than so such not no nor tell know anything something please just really very".split())

_LOG = logging.getLogger("voice-assistant.web")

# Chat-template control tokens and tool-protocol tags. Command output goes
# back to the model inside a tool message, which the Qwen template renders in
# the USER's turn, and llama-server parses these tokens out of text -- so a
# file or a web page the model reads could otherwise close the tool response
# and forge a user instruction. Breaking the token with a zero-width space
# keeps the text readable and makes it inert.
_CTRL_TOKENS = re.compile(
    r"<\|[A-Za-z0-9_]+\|>|</?tool_response>|</?tool_call>|</?think>"
    r"|</?function\b[^>]*>|</?parameter\b[^>]*>")


def sanitize_tool_output(text: str) -> str:
    return _CTRL_TOKENS.sub(lambda m: m.group(0).replace("<", "<​"), text)


def _get(url: str, timeout=None) -> str:
    req = urllib.request.Request(url, headers={
        "User-Agent": WEB_UA, "Accept-Language": "en-US,en;q=0.9"})
    with urllib.request.urlopen(req, timeout=timeout or WEB_TIMEOUT) as r:
        return r.read().decode("utf-8", "replace")


def _text(markup: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", markup or "")).strip()


def _terms(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(w) > 2 and w not in WEB_STOPWORDS}


def _sources(query: str, log):
    """Each source yields (name, title, snippet, url). Failures are skipped.

    The url matters more than it looks: without one the model invents an
    address from the title and fetches a 404 -- under the DeepSeek Harness,
    seven times in one turn before giving up.

    Three of them because no one source is enough. Bing has the widest reach
    but ranks loosely, so on a query it cannot match it returns confident
    nonsense -- "Communication - Wikipedia" for the Communications Security
    Establishment. Wikipedia is precise on organisations and people. Google
    News carries anything recent. Scoring against the query in web_search is
    what sorts the nonsense back down.
    """
    q = urllib.parse.quote_plus(query)
    try:
        x = ET.fromstring(_get(f"https://www.bing.com/search?q={q}&format=rss"))
        for item in list(x.iter("item"))[:8]:
            yield ("web", (item.findtext("title") or "").strip(),
                   _text(item.findtext("description")), (item.findtext("link") or "").strip())
    except Exception as e:
        log.info(f"web_search: bing unavailable ({e})")
    try:
        d = json.loads(_get(
            "https://en.wikipedia.org/w/api.php?action=query&list=search"
            f"&srsearch={q}&format=json&srlimit=4"))
        for r in d.get("query", {}).get("search", []):
            title = r.get("title", "")
            yield ("wikipedia", title, _text(r.get("snippet")),
                   "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")))
    except Exception as e:
        log.info(f"web_search: wikipedia unavailable ({e})")
    try:
        x = ET.fromstring(_get(
            f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"))
        for item in list(x.iter("item"))[:6]:
            # The link is a news.google.com redirect that only resolves in a
            # browser, so it is offered as provenance, not as something to fetch.
            yield ("news", (item.findtext("title") or "").strip(),
                   _text(item.findtext("source")), (item.findtext("link") or "").strip())
    except Exception as e:
        log.info(f"web_search: google news unavailable ({e})")


def web_search(query: str, log=None) -> str:
    """Ranked, deduplicated results as text the model can read aloud from."""
    log = log or _LOG
    query = (query or "").strip()
    if not query:
        return "(no query given)"
    log.warning(f"local tool web_search: {query}")
    terms = _terms(query)
    seen, gathered = set(), []
    for source, title, snippet, url in _sources(query, log):
        if not title:
            continue
        key = re.sub(r"\W+", "", title.lower())[:60]
        if key in seen:
            continue
        seen.add(key)
        gathered.append((source, title, snippet, url, _terms(f"{title} {snippet}")))

    # Weight each query word by how rare it is among the results. Counting
    # plain overlap made every word equal, so for "best tomato varieties for
    # coastal British Columbia" a film shot in BC scored as well as a gardening
    # page: "british" and "columbia" matched in both. The words that separate a
    # good result from a bad one are the ones most results do NOT contain.
    df = {t: sum(1 for *_, hit in gathered if t in hit) for t in terms}  # noqa: E501
    weight = {t: 1.0 / (1 + df[t]) for t in terms}
    total = sum(weight.values()) or 1.0

    scored = []
    for source, title, snippet, url, hit in gathered:
        score = sum(weight[t] for t in terms & hit) / total if terms else 1.0
        if score > 0:
            scored.append((score, source, title, snippet, url))
    if not scored:
        return (f"No results for {query!r}. Nothing online matches this, so say "
                "you could not find it rather than guessing.")
    scored.sort(key=lambda r: -r[0])
    lines = [f"Results for {query!r}, best match first:"]
    # A weak top score means nothing really matched and the list below is
    # loose word-overlap. Say so, or the model reads noise as fact. The
    # threshold is empirical: across the queries tried here, real hits scored
    # 0.75 to 1.00 and pure noise ("Charlie St. Cloud" for a question about
    # tomato varieties in BC) topped out at 0.58.
    if scored[0][0] < 0.6:
        lines.append("(Weak matches only -- nothing here clearly matches the query. "
                     "Treat these as unreliable and say you could not find it.)")
    for score, source, title, snippet, url in scored[:WEB_RESULTS]:
        lines.append(f"[{source}] {title}")
        if snippet:
            lines.append(f"    {snippet[:240]}")
        if url and source != "news":
            lines.append(f"    {url}")
    log.info(f"web_search: {len(scored)} results, top score {scored[0][0]:.2f}")
    return sanitize_tool_output("\n".join(lines))


def fetch_page(url: str, log=None) -> str:
    """One page as readable text, markup removed, bounded in size."""
    log = log or _LOG
    url = (url or "").strip()
    if not url.startswith(("http://", "https://")):
        return "(url must start with http:// or https://)"
    log.warning(f"local tool fetch_page: {url}")
    try:
        doc = _get(url)
    except Exception as e:
        return f"Could not fetch that page: {e}"
    doc = re.sub(r"(?is)<(script|style|noscript|svg|head)\b.*?</\1>", " ", doc)
    text = re.sub(r"[ \t]+", " ", _text(doc))
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    if not text:
        return "That page had no readable text (it may be a script-driven app)."
    if len(text) > WEB_PAGE_CHARS:
        text = text[:WEB_PAGE_CHARS] + "\n... (truncated)"
    return sanitize_tool_output(text)
