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

The keyless sources are thin for anything live: asked for the current
temperature in Victoria, BC, they returned 17 loose matches topping out at
0.26 and the model said it could not find it. A self-hosted SearXNG
(VOICE_ASSISTANT_SEARXNG_URL) is the primary source when one is configured;
it queries Google, Bing, DuckDuckGo, Wikipedia and more on the searcher's
behalf and often carries a direct answer.
"""
import concurrent.futures
import html
import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

# How long a search or page fetch may take, and how much of a page comes back.
WEB_TIMEOUT = float(os.getenv("VOICE_ASSISTANT_WEB_TIMEOUT", "12"))
WEB_RESULTS = int(os.getenv("VOICE_ASSISTANT_WEB_RESULTS", "6"))
WEB_PAGE_CHARS = int(os.getenv("VOICE_ASSISTANT_WEB_PAGE_CHARS", "4000"))
# A SearXNG instance with its JSON format enabled, e.g. http://searx.lan:8080.
# Unset = not used, and search is the keyless sources alone. Its own engines
# give up after ~3 s by default, so a few seconds more is plenty; past that the
# keyless sources answer instead.
SEARXNG_URL = os.getenv("VOICE_ASSISTANT_SEARXNG_URL", "").strip().rstrip("/")
SEARXNG_TIMEOUT = float(os.getenv("VOICE_ASSISTANT_SEARXNG_TIMEOUT", "6"))
# Wikipedia and Google News results added under SearXNG's, when they match well.
WEB_EXTRA_RESULTS = 2
# The settings the web tools read, so a process that does not inherit the
# environment (the harness's MCP sidecar) can be handed them.
ENV_VARS = ("VOICE_ASSISTANT_WEB_TIMEOUT", "VOICE_ASSISTANT_WEB_RESULTS",
            "VOICE_ASSISTANT_WEB_PAGE_CHARS", "VOICE_ASSISTANT_SEARXNG_URL",
            "VOICE_ASSISTANT_SEARXNG_TIMEOUT", "VOICE_ASSISTANT_WEATHER_PLACE",
            "VOICE_ASSISTANT_WEATHER_UNITS", "VOICE_ASSISTANT_WEATHER_TIMEOUT")
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


def _bing(q: str) -> list:
    x = ET.fromstring(_get(f"https://www.bing.com/search?q={q}&format=rss"))
    return [("web", (item.findtext("title") or "").strip(),
             _text(item.findtext("description")), (item.findtext("link") or "").strip())
            for item in list(x.iter("item"))[:8]]


def _wikipedia(q: str) -> list:
    d = json.loads(_get(
        "https://en.wikipedia.org/w/api.php?action=query&list=search"
        f"&srsearch={q}&format=json&srlimit=4"))
    return [("wikipedia", r.get("title", ""), _text(r.get("snippet")),
             "https://en.wikipedia.org/wiki/"
             + urllib.parse.quote(r.get("title", "").replace(" ", "_")))
            for r in d.get("query", {}).get("search", [])]


def _news(q: str) -> list:
    x = ET.fromstring(_get(
        f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"))
    # The link is a news.google.com redirect that only resolves in a
    # browser, so it is offered as provenance, not as something to fetch.
    return [("news", (item.findtext("title") or "").strip(),
             _text(item.findtext("source")), (item.findtext("link") or "").strip())
            for item in list(x.iter("item"))[:6]]


def _searxng(q: str) -> tuple:
    """(direct, results) from SearXNG, both lists of (name, title, snippet, url).

    `direct` is what SearXNG offers as the answer itself: its answers ("12 °C,
    light rain"; a dict in current releases, a bare string in older ones) and
    its infobox (Wikidata's summary of the thing named). A dated result says
    its date first, so "who won" questions can tell this year's from last.
    """
    base = re.sub(r"/search$", "", SEARXNG_URL)
    # 403 Forbidden, or on some versions the HTML page, is what an instance
    # without `json` under search.formats in its settings.yml answers.
    hint = "is format json enabled in its settings.yml?"
    try:
        raw = _get(f"{base}/search?q={q}&format=json", timeout=SEARXNG_TIMEOUT)
    except urllib.error.HTTPError as e:
        raise ValueError(f"{e} ({hint})" if e.code == 403 else str(e)) from None
    try:
        d = json.loads(raw)
    except ValueError:
        raise ValueError(f"not JSON ({hint})") from None
    direct = []
    for a in (d.get("answers") or [])[:3]:
        if isinstance(a, dict):
            a, url = a.get("answer"), a.get("url")
        else:
            url = None
        if isinstance(a, str) and a.strip():
            direct.append(("answer", _text(a)[:400], "", url or ""))
    for box in (d.get("infoboxes") or [])[:1]:
        facts = "; ".join(f"{_text(f.get('label'))}: {_text(f.get('value'))}"
                          for f in (box.get("attributes") or [])[:6]
                          if isinstance(f, dict) and f.get("label") and f.get("value"))
        snippet = _text(box.get("content"))[:240]
        direct.append(("infobox", _text(box.get("infobox")),
                       f"{snippet} ({facts})" if snippet and facts else snippet or facts,
                       box.get("id") or ""))
    results = []
    for r in (d.get("results") or [])[:12]:
        snippet = _text(r.get("content"))
        date = str(r.get("publishedDate") or "")[:10]
        if re.match(r"\d{4}-\d\d-\d\d$", date):
            snippet = f"{date}: {snippet}" if snippet else date
        results.append(("web", _text(r.get("title")), snippet, (r.get("url") or "").strip()))
    return direct, results


_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        _POOL = concurrent.futures.ThreadPoolExecutor(8, thread_name_prefix="web-search")
    return _POOL


def _gather(query: str, terms: set, log):
    """Yield (group, (name, title, snippet, url)) from every source, asked at once.

    Group "direct" and "primary" are SearXNG's, "extra" is Wikipedia and Google
    News, "fallback" is Bing. Failures and stragglers are skipped. All run in
    parallel, so a search takes as long as the slowest source used rather than
    the sum, and Bing's answer is only waited for when SearXNG gave nothing.

    The url matters more than it looks: without one the model invents an
    address from the title and fetches a 404 -- under the DeepSeek Harness,
    seven times in one turn before giving up.

    Several sources because no one keyless source is enough. Bing has the
    widest reach but ranks loosely, so on a query it cannot match it returns
    confident nonsense -- "Communication - Wikipedia" for the Communications
    Security Establishment. Wikipedia is precise on organisations and people.
    Google News carries anything recent. Scoring against the query in
    web_search is what sorts the nonsense back down. SearXNG, when there is
    one, does what Bing's feed did and better: it asks the real engines, which
    understand "current temperature" as a question about the weather, and Bing's
    RSS feed is often blocked anyway.
    """
    q = urllib.parse.quote_plus(query)
    start = time.monotonic()
    jobs = {"wikipedia": _wikipedia, "news": _news, "bing": _bing}
    if SEARXNG_URL:
        jobs["searxng"] = _searxng
    futures = {name: _pool().submit(fn, q) for name, fn in jobs.items()}

    def take(name, timeout):
        try:
            return futures[name].result(
                timeout=max(0.0, start + timeout - time.monotonic()))
        except Exception as e:
            log.info(f"web_search: {name} unavailable ({e or type(e).__name__})")
            return None

    direct, results = (take("searxng", SEARXNG_TIMEOUT + 1) if SEARXNG_URL else None) or ([], [])
    yield from (("direct", item) for item in direct)
    yield from (("primary", item) for item in results)
    # SearXNG had something to say: an answer, or a result sharing a word with
    # the query. The others then had as long as it did; without, the usual.
    answered = bool(direct) or any(
        not terms or terms & _terms(f"{title} {snippet}") for _, title, snippet, _ in results)
    wait = SEARXNG_TIMEOUT + 1 if answered else WEB_TIMEOUT
    for name in ("wikipedia", "news"):
        yield from (("extra", item) for item in take(name, wait) or ())
    if not answered:
        yield from (("fallback", item) for item in take("bing", WEB_TIMEOUT) or ())


def web_search(query: str, log=None) -> str:
    """Ranked, deduplicated results as text the model can read aloud from.

    With SearXNG: its answers and infobox first, then its results in its own
    order (a consensus of real engines, which beats word overlap at telling a
    live weather page from a climate article), then up to WEB_EXTRA_RESULTS
    strong Wikipedia or news matches it did not have. Without it, or when it
    has nothing that matches: the keyless sources, ranked by score as before.
    """
    log = log or _LOG
    query = (query or "").strip()
    if not query:
        return "(no query given)"
    log.warning(f"local tool web_search: {query}")
    terms = _terms(query)
    seen, gathered = set(), []
    for group, (source, title, snippet, url) in _gather(query, terms, log):
        if not title:
            continue
        key = re.sub(r"\W+", "", title.lower())[:60]
        if key in seen and group != "direct":
            continue
        seen.add(key)
        gathered.append((group, source, title, snippet, url, _terms(f"{title} {snippet}")))

    # Weight each query word by how rare it is among the results. Counting
    # plain overlap made every word equal, so for "best tomato varieties for
    # coastal British Columbia" a film shot in BC scored as well as a gardening
    # page: "british" and "columbia" matched in both. The words that separate a
    # good result from a bad one are the ones most results do NOT contain.
    df = {t: sum(1 for *_, hit in gathered if t in hit) for t in terms}  # noqa: E501
    weight = {t: 1.0 / (1 + df[t]) for t in terms}
    total = sum(weight.values()) or 1.0

    scored = []
    for group, source, title, snippet, url, hit in gathered:
        score = sum(weight[t] for t in terms & hit) / total if terms else 1.0
        # An answer is SearXNG's reply to the question itself; it need not repeat its words.
        if score > 0 or source == "answer":
            scored.append((group, score, source, title, snippet, url))
    direct = [r for r in scored if r[0] == "direct"]
    primary = [r for r in scored if r[0] == "primary"][:WEB_RESULTS]
    rest = sorted((r for r in scored if r[0] in ("extra", "fallback")), key=lambda r: -r[1])
    if primary:
        shown = primary + [r for r in rest if r[0] == "extra" and r[1] >= 0.6][:WEB_EXTRA_RESULTS]
    else:
        shown = rest[:WEB_RESULTS]
    if not (direct or shown):
        return (f"No results for {query!r}. Nothing online matches this, so say "
                "you could not find it rather than guessing.")
    top = max((r[1] for r in shown), default=0.0)
    lines = [f"Results for {query!r}, best match first:"]
    # A weak top score means nothing really matched and the list below is
    # loose word-overlap. Say so, or the model reads noise as fact. The
    # threshold is empirical: across the queries tried here, real hits scored
    # 0.75 to 1.00 and pure noise ("Charlie St. Cloud" for a question about
    # tomato varieties in BC) topped out at 0.58. A direct answer is not noise.
    if top < 0.6 and not direct:
        lines.append("(Weak matches only -- nothing here clearly matches the query. "
                     "Treat these as unreliable and say you could not find it.)")
    for group, score, source, title, snippet, url in direct + shown:
        lines.append(f"[{source}] {title}")
        if snippet:
            lines.append(f"    {snippet[:400 if group == 'direct' else 240]}")
        if url and source != "news":
            lines.append(f"    {url}")
    log.info(f"web_search: {len(scored)} results, {len(direct)} direct and {len(primary)} "
             f"from searxng, top score {top:.2f}")
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
