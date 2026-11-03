"""The [context: ...] marker: when it appears, and that its date has a year."""
import json
import time
import types

import voice_assistant as va


def _assistant(exchanges):
    ledger = types.SimpleNamespace(transcript=exchanges)
    return types.SimpleNamespace(_ledger=lambda: ledger, logger=None)


def test_now_includes_the_year():
    at = time.mktime((2026, 11, 2, 14, 5, 0, 0, 0, -1))
    assert va._format_now(at).startswith("Monday 2 November 2026, 14:05")


def test_new_conversation_opens_with_the_date(tmp_path, monkeypatch):
    monkeypatch.setattr(va, "TURN_CONTEXT_FILE", tmp_path / "ctx.json")
    marker = va.VoiceAssistant._turn_context(_assistant([]))
    assert marker.startswith("[context: a new conversation; it is now ")
    assert time.strftime("%Y") in marker


def test_ordinary_turn_has_no_marker(tmp_path, monkeypatch):
    f = tmp_path / "ctx.json"
    f.write_text(json.dumps({"at": time.time(), "boot": va._boot_id(),
                             "run": va.RUN_ID, "tz": va._timezone_name()}))
    monkeypatch.setattr(va, "TURN_CONTEXT_FILE", f)
    assert va.VoiceAssistant._turn_context(_assistant([{"user": "hi"}])) == ""


def test_gap_marker_states_the_full_date(tmp_path, monkeypatch):
    f = tmp_path / "ctx.json"
    f.write_text(json.dumps({"at": time.time() - 7200, "boot": va._boot_id(),
                             "run": va.RUN_ID, "tz": va._timezone_name()}))
    monkeypatch.setattr(va, "TURN_CONTEXT_FILE", f)
    marker = va.VoiceAssistant._turn_context(_assistant([{"user": "hi"}]))
    assert marker.startswith("[context: 2h since the last exchange; it is now ")
    assert time.strftime("%Y") in marker
