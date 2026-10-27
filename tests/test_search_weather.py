"""SearXNG in web_search (web_tools.py), the `weather` tool (weather_tools.py),
and both as the native loop and the dsh MCP sidecar offer them
(voice_assistant.py: LOCAL_TOOLS, _WEATHER_HINT, _stream_local_llm,
DshSession._write_patch; dsh_web_mcp.py).

No network: every request goes through web_tools._get, which is replaced by a
fake that answers by URL and records what was asked. The OpenAI client is a
fake that records each request and streams a scripted reply.

    .venv/bin/python -m pytest tests
"""
import json
import logging
import queue
import socket
import subprocess
import sys
import threading
import time
import types
import urllib.error
import urllib.parse
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import voice_assistant as va  # noqa: E402
import weather_tools as wt  # noqa: E402
import web_tools  # noqa: E402

SEARX = "http://searx.test:8080"


@pytest.fixture
def settings(monkeypatch):
    """This machine's VOICE_ASSISTANT_* settings decide nothing here."""
    monkeypatch.setattr(web_tools, "SEARXNG_URL", "")
    monkeypatch.setattr(web_tools, "SEARXNG_TIMEOUT", 2.0)
    monkeypatch.setattr(web_tools, "WEB_TIMEOUT", 2.0)
    monkeypatch.setattr(web_tools, "WEB_RESULTS", 6)
    monkeypatch.setattr(web_tools, "WEB_EXTRA_RESULTS", 2)
    monkeypatch.setattr(wt, "WEATHER_PLACE", "")
    monkeypatch.setattr(wt, "WEATHER_UNITS", "metric")
    monkeypatch.setattr(wt, "WEATHER_TIMEOUT", 2.0)
    monkeypatch.setattr(wt, "_PLACES", {})
    monkeypatch.setattr(va, "LOCAL_LLM_URL", "http://llm.test:8081/v1")
    monkeypatch.setattr(va, "LOCAL_LLM_MODEL", "test-model")
    monkeypatch.setattr(va, "LOCAL_LLM_THINK", False)
    monkeypatch.setattr(va, "LOCAL_HARNESS", "native")
    monkeypatch.setattr(va, "LOCAL_TOOLS_ENABLED", True)
    monkeypatch.setattr(va, "LOCAL_MAX_TOOL_ITERS", 5)
    monkeypatch.setattr(va, "LOCAL_TOOL_HISTORY_OUTPUT", 4000)
    monkeypatch.setattr(va, "LOOK_ENABLED", True)
    monkeypatch.setattr(va, "ROUTER_TOOLS", True)
    monkeypatch.setattr(va, "ROUTER_THINK", True)
    monkeypatch.setattr(va, "HARD_URL", "")
    monkeypatch.setattr(va, "_server_vision", lambda url, timeout=1.0: False)


class Net:
    """web_tools._get, answering by URL prefix. A value that is an exception
    is raised; a callable is called with the url."""

    def __init__(self, routes):
        self.routes, self.asked = routes, []

    def __call__(self, url, timeout=None):
        self.asked.append((url, timeout))
        for prefix, answer in self.routes.items():
            if url.startswith(prefix):
                if callable(answer) and not isinstance(answer, type):
                    answer = answer(url)
                if isinstance(answer, BaseException) or isinstance(answer, type):
                    raise answer
                return answer if isinstance(answer, str) else json.dumps(answer)
        raise urllib.error.URLError(f"no route to {url}")

    def urls(self, prefix):
        return [u for u, _ in self.asked if u.startswith(prefix)]


def _net(monkeypatch, routes):
    net = Net(routes)
    monkeypatch.setattr(web_tools, "_get", net)
    return net


def _rss(*items):
    return "<rss><channel>" + "".join(
        f"<item><title>{t}</title><description>{d}</description><link>{u}</link>"
        f"<source>{d}</source></item>" for t, d, u in items) + "</channel></rss>"


def _wiki(*titles):
    return {"query": {"search": [{"title": t, "snippet": s} for t, s in titles]}}


BING = "https://www.bing.com/"
WIKI = "https://en.wikipedia.org/"
NEWS = "https://news.google.com/"

KEYLESS = {
    BING: _rss(("Victoria weather forecast and conditions", "Current temperature in Victoria",
                "https://bing.example/victoria")),
    WIKI: _wiki(("Victoria, British Columbia", "Victoria is the capital of British Columbia")),
    NEWS: _rss(("Temperature records broken across British Columbia", "CBC", "https://news.example/1")),
}

QUERY = "current temperature Victoria British Columbia"


def _searx(results=(), answers=(), infoboxes=()):
    return {"query": "q", "results": list(results), "answers": list(answers),
            "infoboxes": list(infoboxes)}


SEARX_HIT = _searx(results=[
    {"title": "Victoria, BC - Current Conditions - Environment Canada",
     "url": "https://weather.gc.ca/city/pages/bc-85_metric_e.html",
     "content": "Temperature: 11.2°C. Condition: Light Rain. Victoria Harbour",
     "engine": "google", "engines": ["google", "bing"], "score": 4.0},
    {"title": "Victoria, British Columbia hourly weather",
     "url": "https://www.timeanddate.com/weather/canada/victoria",
     "content": "Current temperature 11 °C in Victoria, British Columbia",
     "engine": "duckduckgo", "score": 2.0},
])


# --- SearXNG --------------------------------------------------------------------------

def test_searxng_unset_is_never_asked(settings, monkeypatch):
    net = _net(monkeypatch, dict(KEYLESS, **{SEARX: SEARX_HIT}))
    out = web_tools.web_search(QUERY)
    assert net.urls(SEARX) == []
    assert "[web] Victoria weather forecast and conditions" in out, "Bing, as before"
    assert "[wikipedia] Victoria, British Columbia" in out


def test_searxng_results_come_first(settings, monkeypatch):
    monkeypatch.setattr(web_tools, "SEARXNG_URL", SEARX)
    net = _net(monkeypatch, dict(KEYLESS, **{SEARX: SEARX_HIT}))
    out = web_tools.web_search(QUERY)
    url, timeout = net.asked[[u for u, _ in net.asked].index(net.urls(SEARX)[0])]
    assert url == f"{SEARX}/search?q=current+temperature+Victoria+British+Columbia&format=json"
    assert timeout == web_tools.SEARXNG_TIMEOUT
    lines = out.splitlines()
    assert lines[1] == "[web] Victoria, BC - Current Conditions - Environment Canada"
    assert "Temperature: 11.2°C" in lines[2]
    assert lines[3].strip() == "https://weather.gc.ca/city/pages/bc-85_metric_e.html"
    assert lines[4] == "[web] Victoria, British Columbia hourly weather", "in SearXNG's order"
    assert "[wikipedia]" not in out, "the city's article is no match for the current temperature"
    assert "bing.example" not in out, "Bing is the fallback, not a blend"
    assert "Weak matches" not in out


def test_extras_must_match_well_under_searxng(settings, monkeypatch):
    monkeypatch.setattr(web_tools, "SEARXNG_URL", SEARX)
    _net(monkeypatch, dict(KEYLESS, **{
        SEARX: SEARX_HIT, WIKI: _wiki(("Columbia Records", "A record label")),
        NEWS: _rss(("Current temperature in Victoria, British Columbia sets a record", "CBC",
                    "https://news.example/2"))}))
    out = web_tools.web_search(QUERY)
    assert "Columbia Records" not in out
    assert "[news] Current temperature in Victoria, British Columbia sets a record" in out


def test_a_trailing_search_path_is_tolerated(settings, monkeypatch):
    monkeypatch.setattr(web_tools, "SEARXNG_URL", SEARX + "/search")
    net = _net(monkeypatch, dict(KEYLESS, **{SEARX: SEARX_HIT}))
    web_tools.web_search(QUERY)
    assert net.urls(SEARX)[0].startswith(f"{SEARX}/search?q=")


def test_answers_and_infobox_lead(settings, monkeypatch):
    monkeypatch.setattr(web_tools, "SEARXNG_URL", SEARX)
    _net(monkeypatch, dict(KEYLESS, **{SEARX: _searx(
        answers=[{"answer": "11 °C, light rain", "url": "https://wttr.in/Victoria",
                  "engine": "wttr.in"}, "Victoria: 11 °C"],
        infoboxes=[{"infobox": "Victoria, British Columbia", "id": "https://en.wikipedia.org/wiki/Victoria",
                    "content": "Capital city of British Columbia, Canada.",
                    "attributes": [{"label": "Population", "value": "91,867"},
                                   {"label": "Elevation", "value": "23 m"},
                                   {"label": "", "value": "dropped"}]}],
        results=[{"title": "Something about Victoria", "url": "https://x.example",
                  "content": "loosely related"}])}))
    out = web_tools.web_search(QUERY)
    lines = out.splitlines()
    assert lines[1] == "[answer] 11 °C, light rain"
    assert lines[2].strip() == "https://wttr.in/Victoria"
    assert lines[3] == "[answer] Victoria: 11 °C", "the older plain-string form"
    assert lines[4] == "[infobox] Victoria, British Columbia"
    assert lines[5].strip() == ("Capital city of British Columbia, Canada. "
                                "(Population: 91,867; Elevation: 23 m)")
    assert "Weak matches" not in out, "a direct answer is not noise"
    assert "[web] Something about Victoria" in out


def test_an_answer_need_not_repeat_the_question(settings, monkeypatch):
    monkeypatch.setattr(web_tools, "SEARXNG_URL", SEARX)
    _net(monkeypatch, dict(KEYLESS, **{SEARX: _searx(answers=["Dodgers, 4 games to 3"])}))
    out = web_tools.web_search("who won the world series 2026")
    assert "[answer] Dodgers, 4 games to 3" in out
    assert "Weak matches" not in out


def test_an_unrelated_infobox_is_dropped(settings, monkeypatch):
    monkeypatch.setattr(web_tools, "SEARXNG_URL", SEARX)
    hit = dict(SEARX_HIT, infoboxes=[{"infobox": "Banana", "content": "A fruit."}])
    _net(monkeypatch, dict(KEYLESS, **{SEARX: hit}))
    assert "Banana" not in web_tools.web_search(QUERY)


@pytest.mark.parametrize("failure", [
    urllib.error.URLError("connection refused"),
    urllib.error.HTTPError(SEARX, 502, "Bad Gateway", {}, None),
    socket.timeout("timed out"),
    "<html><body>SearXNG</body></html>",       # format json not enabled
    "{\"results\": []}",                         # up, but nothing found
])
def test_searxng_failing_falls_back_to_the_keyless_sources(settings, monkeypatch, failure):
    monkeypatch.setattr(web_tools, "SEARXNG_URL", SEARX)
    _net(monkeypatch, dict(KEYLESS, **{SEARX: failure}))
    out = web_tools.web_search(QUERY)
    assert "[web] Victoria weather forecast and conditions" in out
    assert "https://bing.example/victoria" in out


@pytest.mark.parametrize("answer", [
    "<html>SearXNG</html>",
    urllib.error.HTTPError(SEARX, 403, "Forbidden", {}, None),
])
def test_json_not_enabled_says_what_to_fix(settings, monkeypatch, caplog, answer):
    monkeypatch.setattr(web_tools, "SEARXNG_URL", SEARX)
    _net(monkeypatch, dict(KEYLESS, **{SEARX: answer}))
    web_tools.web_search(QUERY)
    assert "format json enabled in its settings.yml" in caplog.text


def test_a_slow_searxng_is_not_waited_for(settings, monkeypatch):
    monkeypatch.setattr(web_tools, "SEARXNG_URL", SEARX)
    monkeypatch.setattr(web_tools, "SEARXNG_TIMEOUT", 0.05)

    def slow(url):
        time.sleep(2.0)
        return json.dumps(SEARX_HIT)
    _net(monkeypatch, dict(KEYLESS, **{SEARX: slow}))
    t0 = time.monotonic()
    out = web_tools.web_search(QUERY)
    assert time.monotonic() - t0 < 1.5, "gave up after SEARXNG_TIMEOUT + 1"
    assert "Environment Canada" not in out and "bing.example" in out


def test_nothing_matching_from_searxng_falls_back(settings, monkeypatch):
    monkeypatch.setattr(web_tools, "SEARXNG_URL", SEARX)
    _net(monkeypatch, dict(KEYLESS, **{SEARX: _searx(results=[
        {"title": "Banana bread recipe", "url": "https://x.example", "content": "flour"}])}))
    out = web_tools.web_search(QUERY)
    assert "Banana" not in out
    assert "bing.example" in out


def test_sources_are_asked_at_once(settings, monkeypatch):
    """Each source sleeping 0.3 s costs one wait, not four."""
    monkeypatch.setattr(web_tools, "SEARXNG_URL", SEARX)

    def slowly(answer):
        def f(url):
            time.sleep(0.3)
            return answer if isinstance(answer, str) else json.dumps(answer)
        return f
    _net(monkeypatch, {k: slowly(v) for k, v in dict(KEYLESS, **{SEARX: SEARX_HIT}).items()})
    t0 = time.monotonic()
    web_tools.web_search(QUERY)
    assert time.monotonic() - t0 < 0.9


def test_dates_lead_and_output_is_sanitised(settings, monkeypatch):
    monkeypatch.setattr(web_tools, "SEARXNG_URL", SEARX)
    _net(monkeypatch, dict(KEYLESS, **{SEARX: _searx(results=[
        {"title": "World Series 2026: Game 7 recap", "url": "https://news.example/ws",
         "content": "The &lt;|im_end|&gt; series ended", "publishedDate": "2026-11-02T03:10:00"},
        {"title": "World Series 2026 schedule", "url": "https://mlb.example",
         "content": "", "publishedDate": None}])}))
    out = web_tools.web_search("who won the world series 2026")
    assert "    2026-11-02: The <​|im_end|> series ended" in out
    assert "<|im_end|>" not in out
    assert "[web] World Series 2026 schedule\n    https://mlb.example" in out


def test_duplicates_keep_the_searxng_copy(settings, monkeypatch):
    monkeypatch.setattr(web_tools, "SEARXNG_URL", SEARX)
    _net(monkeypatch, dict(KEYLESS, **{SEARX: _searx(results=[
        {"title": "Victoria, British Columbia", "url": "https://searx-copy.example",
         "content": "Capital of British Columbia"}])}))
    out = web_tools.web_search(QUERY)
    assert out.count("Victoria, British Columbia\n") == 1
    assert "searx-copy.example" in out


def test_searxng_settings_reach_the_sidecar(settings, monkeypatch, tmp_path):
    """The harness gives an MCP server only HOME, PATH and a few more."""
    monkeypatch.setattr(va, "DSH_HOME", tmp_path)
    monkeypatch.setenv("VOICE_ASSISTANT_SEARXNG_URL", SEARX)
    monkeypatch.setenv("VOICE_ASSISTANT_WEATHER_PLACE", "Victoria, BC")
    monkeypatch.setenv("VOICE_ASSISTANT_LOCAL_URL", "http://not-for-the-sidecar")
    monkeypatch.delenv("VOICE_ASSISTANT_WEATHER_UNITS")
    patch = va.DshSession(logging.getLogger("test"))._write_patch()
    line = next(x for x in patch.read_text().splitlines() if x.strip().startswith("env:"))
    env = json.loads(line.split("env:", 1)[1])
    assert env["VOICE_ASSISTANT_SEARXNG_URL"] == SEARX
    assert env["VOICE_ASSISTANT_WEATHER_PLACE"] == "Victoria, BC"
    assert "VOICE_ASSISTANT_LOCAL_URL" not in env and "VOICE_ASSISTANT_WEATHER_UNITS" not in env
    assert set(env) <= set(web_tools.ENV_VARS)


# --- weather ----------------------------------------------------------------------------

GEO = "https://geocoding-api.open-meteo.com/"
FORECAST = "https://api.open-meteo.com/"

VICTORIA_BC = {"name": "Victoria", "admin1": "British Columbia", "admin2": "Capital Regional District",
               "country": "Canada", "country_code": "CA", "latitude": 48.4359, "longitude": -123.35155}
VITORIA_BR = {"name": "Vitória", "admin1": "Espírito Santo", "country": "Brazil",
              "country_code": "BR", "latitude": -20.3, "longitude": -40.3}
VICTORIA_TX = {"name": "Victoria", "admin1": "Texas", "country": "United States",
               "country_code": "US", "latitude": 28.8, "longitude": -97.0}
SALT_SPRING = {"name": "Salt Spring Island", "admin1": "British Columbia", "country": "Canada",
               "country_code": "CA", "latitude": 48.81852, "longitude": -123.49061}

FORECAST_REPLY = {
    "current": {"time": "2026-10-27T14:00", "temperature_2m": 11.2, "apparent_temperature": 8.6,
                "relative_humidity_2m": 87, "weather_code": 61, "wind_speed_10m": 15.3,
                "wind_direction_10m": 225, "wind_gusts_10m": 30.1, "precipitation": 0.4},
    "daily": {"time": ["2026-10-27", "2026-10-28", "2026-10-29"],
              "weather_code": [61, 3, 0],
              "temperature_2m_max": [12.4, 10.6, 9.0], "temperature_2m_min": [6.8, 5.2, 2.9],
              "precipitation_probability_max": [80, 40, 0], "precipitation_sum": [3.2, 0.0, 0.0]},
}


def _geo(by_name):
    def answer(url):
        name = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["name"][0]
        return {"results": by_name.get(name, [])} if by_name.get(name) else {"generationtime_ms": 0.1}
    return answer


def _weather_net(monkeypatch, places=None, forecast=FORECAST_REPLY):
    return _net(monkeypatch, {GEO: _geo(places or {"Victoria, BC": [VICTORIA_BC]}),
                              FORECAST: forecast})


def _params(url):
    return {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlsplit(url).query).items()}


def test_weather_reads_the_geocoder_and_the_forecast(settings, monkeypatch):
    net = _weather_net(monkeypatch)
    out = wt.weather("Victoria, BC")
    assert out == (
        "Weather for Victoria, British Columbia, Canada, as of 14:00 local time:\n"
        "Now 11°C (feels like 9°C), light rain, humidity 87%, wind 15 km/h from the "
        "southwest, gusts 30 km/h, 0.4 mm of precipitation in the last hour.\n"
        "Today: light rain, high 12°C, low 7°C, 80% chance of precipitation (3.2 mm).\n"
        "Tomorrow (Wednesday): overcast, high 11°C, low 5°C, 40% chance of precipitation.\n"
        "Thursday: clear, high 9°C, low 3°C, 0% chance of precipitation.")
    p = _params(net.urls(FORECAST)[0])
    assert (p["latitude"], p["longitude"]) == ("48.4359", "-123.35155")
    assert p["timezone"] == "auto" and p["forecast_days"] == "3"
    assert "temperature_unit" not in p, "metric is Open-Meteo's own default"
    assert {"temperature_2m", "apparent_temperature", "relative_humidity_2m", "weather_code",
            "wind_speed_10m", "precipitation"} <= set(p["current"].split(","))
    assert {"temperature_2m_max", "temperature_2m_min",
            "precipitation_probability_max"} <= set(p["daily"].split(","))
    assert all(t == wt.WEATHER_TIMEOUT for _, t in net.asked)


@pytest.mark.parametrize("code, words", [
    (0, "clear"), (2, "partly cloudy"), (3, "overcast"), (45, "fog"), (53, "drizzle"),
    (61, "light rain"), (65, "heavy rain"), (66, "light freezing rain"), (73, "snow"),
    (81, "showers"), (86, "heavy snow showers"), (95, "thunderstorms"),
    (99, "thunderstorms with heavy hail"), (42, "unknown conditions"), (None, "unknown conditions"),
])
def test_weather_code_words(settings, monkeypatch, code, words):
    reply = json.loads(json.dumps(FORECAST_REPLY))
    reply["current"]["weather_code"] = code
    _weather_net(monkeypatch, forecast=reply)
    assert wt.weather("Victoria, BC").splitlines()[1].startswith(f"Now 11°C (feels like 9°C), {words},")


def test_every_documented_code_has_words():
    assert set(wt.WEATHER_CODES) == {0, 1, 2, 3, 45, 48, 51, 53, 55, 56, 57, 61, 63, 65, 66, 67,
                                     71, 73, 75, 77, 80, 81, 82, 85, 86, 95, 96, 99}


@pytest.mark.parametrize("asked", ["", "  ", "here", "Home", "my location"])
def test_no_place_means_the_default(settings, monkeypatch, asked):
    monkeypatch.setattr(wt, "WEATHER_PLACE", "Salt Spring Island")
    net = _weather_net(monkeypatch, places={"Salt Spring Island": [SALT_SPRING]})
    out = wt.weather(asked)
    assert out.startswith("Weather for Salt Spring Island, British Columbia, Canada")
    assert _params(net.urls(GEO)[0])["name"] == "Salt Spring Island"


def test_no_place_and_no_default(settings, monkeypatch):
    net = _weather_net(monkeypatch)
    out = wt.weather("")
    assert "VOICE_ASSISTANT_WEATHER_PLACE" in out and "Ask the user" in out
    assert net.asked == []


def test_imperial_units(settings, monkeypatch):
    monkeypatch.setattr(wt, "WEATHER_UNITS", "imperial")
    reply = json.loads(json.dumps(FORECAST_REPLY))
    reply["current"].update(temperature_2m=52.2, apparent_temperature=47.5, precipitation=0.02,
                            wind_speed_10m=9.5, wind_gusts_10m=18.7)
    reply["daily"]["precipitation_sum"] = [0.13, 0.004, 0.0]
    net = _weather_net(monkeypatch, forecast=reply)
    out = wt.weather("Victoria, BC")
    p = _params(net.urls(FORECAST)[0])
    assert (p["temperature_unit"], p["wind_speed_unit"], p["precipitation_unit"]) == \
        ("fahrenheit", "mph", "inch")
    assert "Now 52°F (feels like 48°F)" in out
    assert "wind 10 mph from the southwest" in out and "gusts" not in out
    assert "0.02 in of precipitation in the last hour" in out
    assert "(0.13 in)" in out
    assert "40% chance of precipitation.\n" in out, "a trace that rounds to nothing is left out"


def test_a_qualifier_picks_the_region(settings, monkeypatch):
    _weather_net(monkeypatch, places={"Victoria, BC": [VITORIA_BR, VICTORIA_TX, VICTORIA_BC]})
    assert wt.weather("Victoria, BC").startswith("Weather for Victoria, British Columbia")


def test_a_qualifier_the_geocoder_cannot_read_is_checked_here(settings, monkeypatch):
    net = _weather_net(monkeypatch, places={"Victoria": [VITORIA_BR, VICTORIA_TX, VICTORIA_BC]})
    assert wt.weather("Victoria, Texas").startswith("Weather for Victoria, Texas, United States")
    assert [_params(u)["name"] for u in net.urls(GEO)] == ["Victoria, Texas", "Victoria"]
    assert wt.weather("Victoria, British Columbia, Canada").startswith(
        "Weather for Victoria, British Columbia, Canada")


def test_a_qualifier_that_matches_nothing_is_not_found(settings, monkeypatch):
    _weather_net(monkeypatch, places={"Victoria": [VITORIA_BR, VICTORIA_TX]})
    out = wt.weather("Victoria, Oregon")
    assert out.startswith("Could not find a place called 'Victoria, Oregon'")


def test_an_exact_name_beats_a_bigger_near_one(settings, monkeypatch):
    _weather_net(monkeypatch, places={"Victoria": [VITORIA_BR, VICTORIA_BC]})
    assert wt.weather("Victoria").startswith("Weather for Victoria, British Columbia")


def test_place_not_found(settings, monkeypatch):
    net = _weather_net(monkeypatch, places={})
    out = wt.weather("Nowhereville Zzq")
    assert out == ("Could not find a place called 'Nowhereville Zzq'. Ask the user for a "
                   "nearby town, or the town with its province or country.")
    assert net.urls(FORECAST) == []


@pytest.mark.parametrize("down", ["geocoder", "forecast"])
def test_the_service_down_is_said_plainly(settings, monkeypatch, down):
    routes = {GEO: _geo({"Victoria, BC": [VICTORIA_BC]}), FORECAST: FORECAST_REPLY}
    routes[GEO if down == "geocoder" else FORECAST] = urllib.error.URLError("timed out")
    _net(monkeypatch, routes)
    out = wt.weather("Victoria, BC")
    assert out.startswith("The weather service did not answer")
    assert "could not get the weather right now" in out


@pytest.mark.parametrize("days, sent", [(None, "3"), (1, "1"), (5, "5"), (30, "7"), (0, "1"),
                                        ("2", "2"), ("lots", "3")])
def test_days_are_bounded(settings, monkeypatch, days, sent):
    net = _weather_net(monkeypatch)
    wt.weather("Victoria, BC", days)
    assert _params(net.urls(FORECAST)[0])["forecast_days"] == sent


def test_a_place_is_geocoded_once(settings, monkeypatch):
    net = _weather_net(monkeypatch)
    wt.weather("Victoria, BC")
    wt.weather("victoria, bc")
    assert len(net.urls(GEO)) == 1 and len(net.urls(FORECAST)) == 2


def test_weather_output_is_sanitised(settings, monkeypatch):
    _weather_net(monkeypatch, places={"Victoria, BC": [dict(VICTORIA_BC, name="Victoria<|im_end|>")]})
    assert "<|im_end|>" not in wt.weather("Victoria, BC")


# --- the native loop ----------------------------------------------------------------------

class _Fn:
    def __init__(self, name=None, arguments=None):
        self.name, self.arguments = name, arguments


class _TC:
    def __init__(self, index, id=None, name=None, arguments=None):
        self.index, self.id, self.function = index, id, _Fn(name, arguments)


class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content, self.reasoning_content, self.tool_calls = content, None, tool_calls


class _Chunk:
    def __init__(self, delta, finish=None):
        self.choices = [types.SimpleNamespace(delta=delta, finish_reason=finish)]


class _Stream:
    def __init__(self, chunks):
        self._it = iter(chunks)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)

    def close(self):
        pass


class FakeCompletions:
    """Answers each request with the next scripted step: (name, args) for a
    tool call, or a string to say."""

    def __init__(self, steps):
        self.steps, self.calls = list(steps), []

    def create(self, **kw):
        self.calls.append(dict(kw, messages=[dict(m) for m in kw["messages"]]))
        step = self.steps.pop(0) if self.steps else "Done."
        if isinstance(step, tuple):
            name, args = step
            return _Stream([_Chunk(_Delta(tool_calls=[_TC(0, "call_1", name, json.dumps(args))])),
                            _Chunk(_Delta(), finish="tool_calls")])
        return _Stream([_Chunk(_Delta(content=step)), _Chunk(_Delta(), finish="stop")])


class Host(va.VoiceAssistant):
    """What a local turn touches, with the real history handling."""

    def __init__(self, steps, verdict=None):
        self.logger = logging.getLogger("test-weather")
        self.client = types.SimpleNamespace(chat=types.SimpleNamespace(
            completions=FakeCompletions(steps)))
        self.llm_tier = "local"
        self._router = None
        self._router_verdict = verdict
        self._local_history = []
        self._assistant_text = ""
        self._thinking_text = ""
        self._sentence_queue = queue.Queue()
        self._local_stream = None
        self._hard_client = None
        self._hard_health = (0.0, False)
        self._turn_tool_calls = 0
        self.notified = []
        self.fillers = []
        self.asked = []

    @property
    def calls(self):
        return self.client.chat.completions.calls

    def _local_llm_client(self):
        return self.client

    def _ledger(self):
        return types.SimpleNamespace(recap_for=lambda backend: "",
                                     remember=lambda *a, **k: None)

    def _flush_sentences(self, final=False):
        pass

    def _maybe_notify_thinking(self):
        pass

    def _notify_tool_use(self, name, args):
        self.notified.append((name, json.loads(args)))

    def _save_local_history(self):
        pass

    def _router_used(self, backend):
        pass

    def _say_filler(self, text):
        self.fillers.append(text)

    def _weather_tool(self, place, days=None):
        self.asked.append((place, days))
        return "Weather for Victoria, British Columbia, Canada:\nNow 11°C, light rain."


class Verdict:
    def __init__(self, tools=(), hard=False):
        self.tools, self.hard, self.backend = list(tools), hard, None


def _tool_names(kw):
    return [t["function"]["name"] for t in kw.get("tools") or []]


def test_weather_is_offered_next_to_the_web_tools(settings):
    host = Host(["Hello."])
    host._stream_local_llm("hello", threading.Event())
    assert _tool_names(host.calls[0]) == ["run_shell", "web_search", "fetch_page", "weather"]
    prompt = host.calls[0]["messages"][0]["content"]
    assert "weather gives the current conditions" in prompt and "not web_search" in prompt


def test_the_tool_schema_is_the_same_every_turn(settings):
    """It is rendered into the system message: a changed schema empties the prompt cache."""
    schemas = []
    for text, verdict in [("hello", None), ("what's the weather tomorrow", Verdict(tools=[])),
                          ("what is two plus two", Verdict(tools=[])),
                          ("search for the news", Verdict(tools=["web_search", "fetch_page"]))]:
        host = Host(["Hi."], verdict=verdict)
        host._stream_local_llm(text, threading.Event())
        schemas.append((json.dumps(host.calls[0]["tools"]), host.calls[0]["messages"][0]["content"]))
    assert len(set(schemas)) == 1


def test_a_weather_call_is_dispatched(settings):
    host = Host([("weather", {"place": "Victoria, BC", "days": 2}), "Eleven degrees and raining."])
    assert host._stream_local_llm("what's the weather in Victoria", threading.Event()) is True
    assert host.asked == [("Victoria, BC", 2)]
    assert host.notified == [("Weather", {"place": "Victoria, BC"})]
    tool_msg = host.calls[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["content"].startswith("Weather for Victoria")
    assert host._assistant_text == "Eleven degrees and raining."


def test_a_weather_call_without_a_place_asks_for_the_default(settings):
    host = Host([("weather", {}), "Ten degrees."])
    host._stream_local_llm("what's the weather like", threading.Event())
    assert host.asked == [("", None)]


def test_the_real_weather_tool_is_what_runs(settings, monkeypatch):
    _weather_net(monkeypatch)
    host = Host([])
    out = va.VoiceAssistant._weather_tool(host, "Victoria, BC", 1)
    assert out.startswith("Weather for Victoria, British Columbia, Canada")


def test_a_weather_turn_keeps_its_tools_when_the_router_wanted_none(settings):
    host = Host(["Ten degrees."], verdict=Verdict(tools=[]))
    host._stream_local_llm("what's the weather tomorrow", threading.Event())
    assert "tool_choice" not in host.calls[0], "auto, not none"
    assert "weather" in _tool_names(host.calls[0])


def test_other_toolless_turns_are_still_gated(settings):
    host = Host(["Four."], verdict=Verdict(tools=[]))
    host._stream_local_llm("what is two plus two", threading.Event())
    assert host.calls[0]["tool_choice"] == "none"


def test_a_weather_turn_is_not_forced(settings):
    host = Host(["Ten degrees."])
    host._stream_local_llm("will it rain tomorrow", threading.Event())
    assert "tool_choice" not in host.calls[0]


def test_a_weather_turn_skips_the_hard_lane(settings):
    host = Host(["Rain."], verdict=Verdict(tools=[], hard=True))
    host._stream_local_llm("what's the forecast for the weekend", threading.Event())
    assert host.fillers == []
    assert host.calls[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    host = Host(["Because."], verdict=Verdict(tools=[], hard=True))
    host._stream_local_llm("why does entropy always increase", threading.Event())
    assert host.fillers, "a hard turn that is not about the weather still thinks"


def test_no_tools_no_hint(settings, monkeypatch):
    monkeypatch.setattr(va, "LOCAL_TOOLS_ENABLED", False)
    host = Host(["I can't check."], verdict=Verdict(tools=[], hard=True))
    host._stream_local_llm("what's the weather", threading.Event())
    assert host.fillers, "with no tools to keep, the hint changes nothing"


def test_the_hard_prompt_drops_the_tools_paragraph():
    prompt = va.VoiceAssistant._hard_voice_prompt()
    assert "four tools" not in prompt and "weather gives" not in prompt
    assert "no tools on this turn" in prompt


@pytest.mark.parametrize("text, hint", [
    ("what's the weather like", True),
    ("what's the forecast for Saturday", True),
    ("is it going to rain tomorrow", True),
    ("will it snow tonight", True),
    ("is it raining in Victoria", True),
    ("how cold is it outside", True),
    ("what's the temperature outside", True),
    ("current temperature in Victoria British Columbia", True),
    ("do I need an umbrella", True),
    ("play Purple Rain", False),
    ("whether or not you agree", False),
    ("what is two plus two", False),
    ("restart the network", False),
    ("set a timer for ten minutes", False),
])
def test_weather_hint(text, hint):
    assert bool(va._WEATHER_HINT.search(text)) == hint


def test_dsh_knows_the_weather_tool():
    assert va.VoiceAssistant._DSH_TOOL_LABELS["mcp__web__weather"] == "Weather"
    assert "mcp__web__weather" in va.DSH_VOICE_PROMPT


# --- the dsh MCP sidecar --------------------------------------------------------------------

@pytest.fixture
def mcp(monkeypatch):
    monkeypatch.setattr(logging, "basicConfig", lambda **k: None)
    import dsh_web_mcp
    return dsh_web_mcp


def test_the_sidecar_lists_weather(mcp):
    result, error = mcp.handle({"method": "tools/list"})
    tools = {t["name"]: t for t in result["tools"]}
    assert list(tools) == ["web_search", "fetch_page", "weather"]
    schema = tools["weather"]["inputSchema"]
    assert set(schema["properties"]) == {"place", "days"} and not schema.get("required")
    native = next(t["function"] for t in va.LOCAL_TOOLS if t["function"]["name"] == "weather")
    assert tools["weather"]["description"] == native["description"], "one tool, one description"
    assert schema["properties"] == native["parameters"]["properties"]


def test_the_sidecar_calls_weather(settings, mcp, monkeypatch):
    asked = []
    monkeypatch.setattr(mcp.weather_tools, "weather",
                        lambda place, days, log=None: asked.append((place, days)) or "Sunny.")
    result, error = mcp.handle({"method": "tools/call", "params": {
        "name": "weather", "arguments": {"place": "Salt Spring Island", "days": 2}}})
    assert error is None and result == {"content": [{"type": "text", "text": "Sunny."}],
                                        "isError": False}
    mcp.handle({"method": "tools/call", "params": {"name": "weather", "arguments": {}}})
    assert asked == [("Salt Spring Island", 2), ("", None)]


def test_the_sidecar_searches_through_searxng(settings, mcp, monkeypatch):
    monkeypatch.setattr(web_tools, "SEARXNG_URL", SEARX)
    _net(monkeypatch, dict(KEYLESS, **{SEARX: SEARX_HIT}))
    result, _ = mcp.handle({"method": "tools/call", "params": {
        "name": "web_search", "arguments": {"query": QUERY}}})
    assert "Environment Canada" in result["content"][0]["text"]


def test_the_sidecar_over_stdio_reads_its_settings_from_the_env(tmp_path):
    """As the harness runs it: a bare environment plus what the patch passes."""
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
           "VOICE_ASSISTANT_SEARXNG_URL": SEARX, "VOICE_ASSISTANT_WEATHER_PLACE": "Victoria, BC"}
    script = ("import sys; sys.argv = ['x']; sys.path.insert(0, %r); import dsh_web_mcp, web_tools, "
              "weather_tools; print(web_tools.SEARXNG_URL, '|', weather_tools.WEATHER_PLACE, "
              "file=sys.stderr); sys.exit(dsh_web_mcp.main())" % str(ROOT))
    reqs = "\n".join(json.dumps(r) for r in [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}]) + "\n"
    p = subprocess.run([sys.executable, "-c", script], input=reqs, capture_output=True,
                       text=True, env=env, timeout=30)
    replies = [json.loads(x) for x in p.stdout.splitlines()]
    assert [r["id"] for r in replies] == [1, 2]
    assert "weather" in [t["name"] for t in replies[1]["result"]["tools"]]
    assert f"{SEARX} | Victoria, BC" in p.stderr
