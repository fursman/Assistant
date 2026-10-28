"""TTS thread count: physical cores this process may use, capped (_physical_cores)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import voice_assistant as va  # noqa: E402


def cpuinfo(pairs):
    """/proc/cpuinfo for logical cpus given as (physical id, core id)."""
    return "\n\n".join(f"processor\t: {i}\nphysical id\t: {p}\ncore id\t\t: {c}"
                       for i, (p, c) in enumerate(pairs))


# An i9-13900K: 8 P-cores with two threads each (cpus 0-15), then 16 E-cores.
I9 = [(0, c // 2 * 4) for c in range(16)] + [(0, 32 + c) for c in range(16)]


def fake(monkeypatch, pairs, allowed):
    orig = va.Path.read_text
    monkeypatch.setattr(va.Path, "read_text", lambda self, *a, **k:
                        cpuinfo(pairs) if str(self) == "/proc/cpuinfo" else orig(self, *a, **k))
    monkeypatch.setattr(va.os, "sched_getaffinity", lambda pid: set(allowed))
    monkeypatch.delenv("VOICE_ASSISTANT_TTS_THREADS", raising=False)
    monkeypatch.setattr(va, "TTS_MAX_THREADS", 8)


def test_capped_on_a_big_part(monkeypatch):
    fake(monkeypatch, I9, range(32))
    assert va._physical_cores() == 8, "24 physical cores, but no more than the cap"


def test_only_the_cores_this_process_may_use(monkeypatch):
    # The VM holds 2-15: what's left is P-core 0's two threads and 16 E-cores.
    fake(monkeypatch, I9, [0, 1] + list(range(16, 32)))
    monkeypatch.setattr(va, "TTS_MAX_THREADS", 64)
    assert va._physical_cores() == 17


def test_hyperthreads_are_one_core(monkeypatch):
    fake(monkeypatch, [(0, 0), (0, 0), (0, 1), (0, 1)], range(4))
    assert va._physical_cores() == 2


def test_the_override_wins(monkeypatch):
    fake(monkeypatch, I9, range(32))
    monkeypatch.setenv("VOICE_ASSISTANT_TTS_THREADS", "12")
    assert va._physical_cores() == 12
