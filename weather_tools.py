"""Current weather and a short forecast, from Open-Meteo.

Shared by voice_assistant.py (the native tool loop's `weather`) and
dsh_web_mcp.py (the same tool, served to the DeepSeek Harness), like the web
tools in web_tools.py.

Its own tool because a search cannot answer it: "current temperature Victoria
British Columbia" through web_search came back as 17 loose matches topping out
at 0.26 -- a Wikipedia article on the province, a news story about records --
and the model said it could not find it. Open-Meteo needs no key, answers in
well under a second, and returns numbers rather than pages to read.
"""
import json
import logging
import os
import unicodedata
import urllib.parse
from datetime import date

import web_tools

# Where "what's the weather" means when the user names no place: a town, with
# its province, state or country when the name is ambiguous ("Victoria, BC").
WEATHER_PLACE = os.getenv("VOICE_ASSISTANT_WEATHER_PLACE", "").strip()
# metric (Celsius, km/h, mm) or imperial (Fahrenheit, mph, inches).
WEATHER_UNITS = os.getenv("VOICE_ASSISTANT_WEATHER_UNITS", "metric").strip().lower()
WEATHER_TIMEOUT = float(os.getenv("VOICE_ASSISTANT_WEATHER_TIMEOUT", "5"))
# Days of forecast, today included, when the model does not ask for a number.
WEATHER_DAYS = 3
WEATHER_MAX_DAYS = 7

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

_LOG = logging.getLogger("voice-assistant.weather")

# WMO weather interpretation codes, as Open-Meteo documents them, in words a
# spoken sentence can carry.
WEATHER_CODES = {
    0: "clear", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "light freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "light freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "violent showers",
    85: "light snow showers", 86: "heavy snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail", 99: "thunderstorms with heavy hail",
}

_COMPASS = ("north", "northeast", "east", "southeast",
            "south", "southwest", "west", "northwest")
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

# What a model passes when it means the user's own location.
_HERE = frozenset({"", "here", "home", "local", "my location", "current location",
                   "my area", "where i am", "where i live"})

# Province and state codes. The geocoder matches place names, so the "BC" in
# "Victoria, BC" is checked against each candidate's region here.
_REGION_CODES = dict(pair.split(":") for pair in (
    "ab:alberta,bc:british columbia,mb:manitoba,nb:new brunswick,"
    "nl:newfoundland and labrador,ns:nova scotia,nt:northwest territories,"
    "nu:nunavut,on:ontario,pe:prince edward island,pei:prince edward island,"
    "qc:quebec,sk:saskatchewan,yt:yukon,"
    "al:alabama,ak:alaska,az:arizona,ar:arkansas,ca:california,co:colorado,"
    "ct:connecticut,de:delaware,fl:florida,ga:georgia,hi:hawaii,id:idaho,"
    "il:illinois,in:indiana,ia:iowa,ks:kansas,ky:kentucky,la:louisiana,"
    "me:maine,md:maryland,ma:massachusetts,mi:michigan,mn:minnesota,"
    "ms:mississippi,mo:missouri,mt:montana,ne:nebraska,nv:nevada,"
    "nh:new hampshire,nj:new jersey,nm:new mexico,ny:new york,"
    "nc:north carolina,nd:north dakota,oh:ohio,ok:oklahoma,or:oregon,"
    "pa:pennsylvania,ri:rhode island,sc:south carolina,sd:south dakota,"
    "tn:tennessee,tx:texas,ut:utah,vt:vermont,va:virginia,wa:washington,"
    "wv:west virginia,wi:wisconsin,wy:wyoming,dc:district of columbia,"
    "usa:united states,uk:united kingdom").split(","))

# Geocoded places, by what was asked. A town does not move.
_PLACES: dict = {}


class WeatherError(Exception):
    """A failure the model should hear in words."""


def _fold(text) -> str:
    """Lower case without accents, so "Quebec" finds "Québec"."""
    text = unicodedata.normalize("NFKD", str(text or ""))
    return "".join(c for c in text if not unicodedata.combining(c)).lower().strip()


def _units() -> dict:
    if WEATHER_UNITS == "imperial":
        return {"query": {"temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
                          "precipitation_unit": "inch"},
                "temp": "°F", "wind": "mph", "precip": "in"}
    return {"query": {}, "temp": "°C", "wind": "km/h", "precip": "mm"}


def _get_json(url: str, params: dict) -> dict:
    return json.loads(web_tools._get(f"{url}?{urllib.parse.urlencode(params)}",
                                     timeout=WEATHER_TIMEOUT))


def _matches(place: dict, qualifier: str) -> bool:
    """Whether a geocoder result is in the region or country the user named."""
    q = _fold(qualifier)
    names = {_fold(place.get(k)) for k in ("admin1", "admin2", "admin3", "country")}
    names.add(_fold(place.get("country_code")))
    return q in names or _REGION_CODES.get(q) in names


def geocode(place: str) -> dict:
    """The best Open-Meteo geocoder match for a place, or WeatherError.

    "Victoria" alone is Vitória in Brazil to the geocoder (it ranks fuzzy
    matches by population), so an exact name beats a near one. A qualifier
    after a comma ("Victoria, BC", "Paris, Texas") must match the result's
    province, state or country; the geocoder itself only reads names, so when
    the whole string finds nothing the name is looked up alone.
    """
    key = _fold(place)
    if key in _PLACES:
        return _PLACES[key]
    name, _, qualifier = (p.strip() for p in place.partition(","))
    qualifiers = [q.strip() for q in qualifier.split(",") if q.strip()]
    try:
        found = _get_json(GEOCODE_URL, {"name": place, "count": 10, "language": "en",
                                        "format": "json"}).get("results") or []
        whole = bool(found)
        if not found and qualifiers:
            found = _get_json(GEOCODE_URL, {"name": name, "count": 10, "language": "en",
                                            "format": "json"}).get("results") or []
    except Exception as e:
        raise WeatherError(f"The weather service did not answer ({e}). Say you could "
                           "not get the weather right now.") from None
    if qualifiers:
        named = [r for r in found if all(_matches(r, q) for q in qualifiers)]
        # A whole-string match the qualifier cannot place is still the
        # geocoder's own reading of it; a name-only match is not.
        found = named or (found if whole else [])
    if not found:
        raise WeatherError(f"Could not find a place called {place!r}. Ask the user for "
                           "a nearby town, or the town with its province or country.")
    exact = [r for r in found if _fold(r.get("name")) == _fold(name)]
    best = (exact or found)[0]
    _PLACES[key] = best
    return best


def _label(place: dict) -> str:
    parts = []
    for k in ("name", "admin1", "country"):
        v = (place.get(k) or "").strip()
        if v and v not in parts:
            parts.append(v)
    return ", ".join(parts)


def _num(v, unit: str = "") -> str:
    """12.4 -> "12", with the unit: a voice does not need tenths of a degree."""
    return f"{round(float(v))}{unit}" if v is not None else "?"


def _amount(v, unit: str) -> str:
    """Precipitation keeps a decimal (two in inches): 0.4 mm is not nothing.
    Empty when it rounds to nothing."""
    n = f"{float(v or 0):.{2 if unit == 'in' else 1}f}".rstrip("0").rstrip(".")
    return "" if n in ("0", "-0") else f"{n} {unit}"


def _day_name(i: int, iso: str) -> str:
    if i == 0:
        return "Today"
    try:
        weekday = _WEEKDAYS[date.fromisoformat(iso).weekday()]
    except ValueError:
        weekday = iso
    return f"Tomorrow ({weekday})" if i == 1 else weekday


def describe(place: dict, d: dict, units: dict) -> str:
    """Open-Meteo's forecast as a few compact lines a model can say in one sentence."""
    t, w, p = units["temp"], units["wind"], units["precip"]
    cur = d.get("current") or {}
    when = str(cur.get("time") or "")
    lines = [f"Weather for {_label(place)}"
             + (f", as of {when[11:16]} local time:" if len(when) >= 16 else ":")]
    if cur:
        now = [f"Now {_num(cur.get('temperature_2m'), t)}"
               f" (feels like {_num(cur.get('apparent_temperature'), t)})",
               WEATHER_CODES.get(cur.get("weather_code"), "unknown conditions")]
        if cur.get("relative_humidity_2m") is not None:
            now.append(f"humidity {_num(cur['relative_humidity_2m'])}%")
        if cur.get("wind_speed_10m") is not None:
            wind = f"wind {_num(cur['wind_speed_10m'])} {w}"
            if cur.get("wind_direction_10m") is not None and round(float(cur["wind_speed_10m"])):
                wind += f" from the {_COMPASS[round(float(cur['wind_direction_10m']) / 45) % 8]}"
            if cur.get("wind_gusts_10m") is not None and \
                    float(cur["wind_gusts_10m"]) >= float(cur["wind_speed_10m"]) + 10:
                wind += f", gusts {_num(cur['wind_gusts_10m'])} {w}"
            now.append(wind)
        if _amount(cur.get("precipitation"), p):
            now.append(f"{_amount(cur['precipitation'], p)} of precipitation in the last hour")
        lines.append(", ".join(now) + ".")
    daily = d.get("daily") or {}
    for i, iso in enumerate(daily.get("time") or []):
        def at(k):
            v = daily.get(k) or []
            return v[i] if i < len(v) else None
        day = [WEATHER_CODES.get(at("weather_code"), "unknown conditions"),
               f"high {_num(at('temperature_2m_max'), t)}",
               f"low {_num(at('temperature_2m_min'), t)}"]
        if at("precipitation_probability_max") is not None:
            chance = f"{_num(at('precipitation_probability_max'))}% chance of precipitation"
            if _amount(at("precipitation_sum"), p):
                chance += f" ({_amount(at('precipitation_sum'), p)})"
            day.append(chance)
        lines.append(f"{_day_name(i, iso)}: " + ", ".join(day) + ".")
    return "\n".join(lines)


def weather(place: str = "", days=None, log=None) -> str:
    """Current conditions and a short forecast as text, or a failure in words."""
    log = log or _LOG
    place = " ".join(str(place or "").split())
    if _fold(place) in _HERE:
        place = WEATHER_PLACE
    if not place:
        return ("No place was given and no home location is set "
                "(VOICE_ASSISTANT_WEATHER_PLACE). Ask the user which town they mean.")
    try:
        days = max(1, min(WEATHER_MAX_DAYS, int(days)))
    except (TypeError, ValueError):
        days = WEATHER_DAYS
    log.warning(f"local tool weather: {place} ({days} days)")
    units = _units()
    try:
        where = geocode(place)
        try:
            d = _get_json(FORECAST_URL, {
                "latitude": where["latitude"], "longitude": where["longitude"],
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                           "weather_code,wind_speed_10m,wind_direction_10m,"
                           "wind_gusts_10m,precipitation",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                         "precipitation_probability_max,precipitation_sum",
                "timezone": "auto", "forecast_days": days, **units["query"]})
        except Exception as e:
            raise WeatherError(f"The weather service did not answer ({e}). Say you "
                               "could not get the weather right now.") from None
        if d.get("error"):
            raise WeatherError(f"The weather service refused the request "
                               f"({d.get('reason', 'no reason given')}).")
    except WeatherError as e:
        log.info(f"weather: {e}")
        return web_tools.sanitize_tool_output(str(e))
    return web_tools.sanitize_tool_output(describe(where, d, units))
