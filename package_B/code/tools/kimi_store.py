#!/usr/bin/env python3
"""READ helper (local workstation only): per-turn detail from the executor's own session store.

The harness report (`[REDACTED: infrastructure] report --json`) gives each delegated turn's steps and input.
This module re-reads the same sessions' wire logs (written by the executor process, not by the
harness) to add what the report does not carry:

  * each turn's first request input -- every request re-sends the priming block (system
    prompt, tool schemas) plus the conversation so far, so any request of a turn is an upper
    bound on that turn's per-request priming;
  * each turn's first event time (the workload's date range).

Turn segmentation mirrors the report's: an event carrying turnStep "<turn>.<step>" sets the
current turn, llm.request counts a step, usage.record adds inputOther + inputCacheRead to the
current turn. Lines end at "\n" only (raw U+2028/U+2029 occur inside JSON strings).
Every turn is reconciled against the report (steps and input must match exactly); any
mismatch is an error, never a silent skip.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

KIMI_HOME = Path(os.environ.get("KIMI_HOME", Path.home() / ".kimi-code"))
# The executor's subscription models; anything else in a session is a fallback.
SUBSCRIPTION_MODEL = re.compile(r"(?:^|/)(?:kimi-for-coding|k3)$")


@dataclass
class TurnWire:
    steps: int = 0
    total_input: int = 0                 # inputOther + inputCacheRead, as the report counts
    first_request: int | None = None     # all input of the turn's first usage record
    first_ms: int | None = None          # earliest event time in the turn (epoch ms)


def parse_wire(text: str) -> dict[str, TurnWire]:
    turns: dict[str, TurnWire] = {}
    current = "0"
    for line in text.split("\n"):
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        step = ev.get("turnStep")
        if isinstance(step, str) and step:
            current = step.split(".", 1)[0]
        kind = ev.get("type")
        if kind not in ("llm.request", "usage.record"):
            continue
        t = turns.setdefault(current, TurnWire())
        if isinstance(ev.get("time"), int):
            t.first_ms = ev["time"] if t.first_ms is None else min(t.first_ms, ev["time"])
        if kind == "llm.request":
            t.steps += 1
            continue
        u = ev.get("usage") or {}
        other, read, write = (int(u.get(k, 0) or 0) for k in ("inputOther", "inputCacheRead",
                                                              "inputCacheCreation"))
        t.total_input += other + read
        if t.first_request is None:
            t.first_request = other + read + write
    return turns


def session_wire(prefix: str, home: Path = KIMI_HOME) -> Path:
    """The wire log of the one session whose id starts with `prefix` (the report truncates ids)."""
    hits = sorted(p for p in (home / "sessions").glob(f"*/{prefix}*") if p.is_dir())
    if len(hits) != 1:
        raise LookupError(f"{len(hits)} sessions match one report prefix (need exactly 1)")
    wire = hits[0] / "agents" / "main" / "wire.jsonl"
    if not wire.is_file():
        raise LookupError("a reported session has no wire log")
    return wire


def turn_details(report_turns: list[dict[str, Any]], home: Path = KIMI_HOME) -> list[TurnWire]:
    """Wire detail for every report turn, in report order; reconciled turn by turn."""
    cache: dict[str, dict[str, TurnWire]] = {}
    out = []
    for rt in report_turns:
        prefix = rt["session"]
        if prefix not in cache:
            cache[prefix] = parse_wire(session_wire(prefix, home).read_text(encoding="utf-8",
                                                                            errors="replace"))
        w = cache[prefix].get(str(rt["turn"]))
        if w is None or (w.steps, w.total_input) != (rt["steps"], rt["totalInput"]):
            got = None if w is None else (w.steps, w.total_input)
            raise ValueError(f"wire log disagrees with the report for one turn: "
                             f"report {(rt['steps'], rt['totalInput'])}, wire {got}")
        out.append(w)
    return out


def priming_bound(turns: list[TurnWire], constant: int) -> dict[str, Any]:
    """Upper bound on the priming share of executor input.

    Per-request priming in a turn is at most min(the turn's first request, constant), where the
    constant is the one-off measurement taken before leaner tool profiles; a turn's priming is at
    most that times its steps, and never more than the turn's whole input.
    """
    total = sum(t.total_input for t in turns)
    with_usage = [t for t in turns if t.first_request is not None]
    upper = sum(min(min(t.first_request, constant) * t.steps, t.total_input) for t in with_usage)
    return {"total_input": total, "priming_upper": upper,
            "share_upper": upper / total if total else None,
            "turns_with_usage": len(with_usage),
            "first_request_below_constant": sum(t.first_request < constant for t in with_usage),
            "constant_times_steps": constant * sum(t.steps for t in turns),
            "turns_where_constant_exceeds_input": sum(constant * t.steps > t.total_input for t in turns)}


def is_fallback(models: str) -> bool:
    """True when a session was served (at least partly) by a non-subscription model."""
    names = [m for m in models.split(",") if m and m != "?"]
    return any(not SUBSCRIPTION_MODEL.search(m) for m in names)


def utc_date(ms: int) -> str:
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).date().isoformat()
