#!/usr/bin/env python3
"""READ helper (local workstation only): a third meter for the transcript token counts.

Claude Code can export OpenTelemetry log events. Its `api_request` event carries, per API request,
the usage the API response reported (input, output, cache-read and cache-write tokens) and the
request id. This module matches those events, request by request on request id, against the
records the paper builds from the transcripts with tools/claude_usage.py (one record per message
id, input fields from the first line, output the maximum over lines, synthetic messages dropped),
restricted to transcript requests inside the telemetry window, and reports how far they agree.

Both channels come from the same client process: telemetry carries the usage object the API
returned, the transcripts are what the client wrote to disk. Agreement therefore tests the
"transcripts undercount" claim and the paper's line rule against the API's own reported usage.
It is not the provider's billing record.

Inputs
  * telemetry: local copies of the OTLP/HTTP-JSON sink's files (one {"recv_utc", "path", "body"}
    object per line, body = resourceLogs -> scopeLogs -> logRecords). Only `api_request` records
    are kept, and of those only request id, model, query source, the four token counts, duration
    and timestamp, plus the session id hashed at read time (for classifying unmatched requests,
    in memory only). Everything else -- e-mail, organisation and user ids, prompts, responses --
    is never stored. The client version comes from the resource's service.version.
  * transcripts: tools/claude_usage.extract() over the projects root; the files modified since the
    window start (minus SCAN_MARGIN) are re-read line by line for what extract() does not keep:
    each message id's request id, its number of usage lines, its first line's output and its
    per-line sums (the alternative rules below). The same line filters as claude_usage apply.

Window: from the earliest api_request's start (event time minus duration) to the latest event
time. A transcript request belongs to it if its record timestamp (first line) falls inside.

Unmatched classes
  telemetry only (in telemetry, no transcript record with that request id):
    query_source_never_matched  its query source has no matched request at all in this run: side
                                calls (session-title generation and the like) that the client
                                does not write to transcripts
    session_without_transcript  no transcript record of its session was read
    unexplained                 anything else
  transcript only (in the window, request id absent from telemetry):
    no_request_id               the transcript line carries no request id (cannot be matched)
    session_not_emitting        its session never emitted an api_request: a process started
                                before telemetry was enabled
    before_session_first_event  its session emits, but the request precedes the session's first
                                event: an earlier, non-emitting process of a later-resumed session
    in_flight_at_window_end     its first line lies within the longest request duration seen in
                                the telemetry before the window end: the event is emitted when the
                                response completes, the first transcript line mid-stream, so the
                                response may have finished after the telemetry was copied
    after_session_last_event    after the session's last event, earlier than that: not exported
                                (lost at process exit, say)
    unexplained                 inside its session's telemetry span, yet missing

Rules compared on matched requests (ratios are transcript / telemetry)
  paper       the claude_usage record (input fields from the first line, output max over lines)
  first_line  output from the first line only
  line_count  every usage line counted as a request (a line counter), token fields summed

Writes evidence/otel_meter.json: aggregates only, never ids, e-mails, paths or project names.

Re-run (Git Bash, local workstation; EVIDENCE_BOX = [REDACTED: infrastructure]; the raw
copies stay outside the repository):
  OTEL_DIR="$TEMP/claude-otel"; rm -rf "$OTEL_DIR"; mkdir -p "$OTEL_DIR"
  MSYS_NO_PATHCONV=1 scp -q "$EVIDENCE_BOX:[REDACTED: infrastructure]/claude_code_otel.jsonl*" "$OTEL_DIR/"
  python tools/otel_meter.py --telemetry "$OTEL_DIR"/claude_code_otel.jsonl* && python tools/gen_macros.py
--until <ISO time> reproduces an earlier run from a grown file (the sink only appends).
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import claude_usage  # noqa: E402
from evidence_lib import EVIDENCE, figure, sha256_text, utc_now, write_json  # noqa: E402

GENERATOR = "tools/otel_meter.py"
FREEZE = "2026-11-15T00:00:00Z"  # the frozen snapshot's freeze (tools/pull_evidence.py)
FIELDS = ("input", "cache_read", "cache_write", "output")
TRANSCRIPT_KEY = {"input": "input_tokens", "cache_read": "cache_read_input_tokens",
                  "cache_write": "cache_creation_input_tokens", "output": "output_tokens"}
TELEMETRY_KEY = {"input": "input_tokens", "cache_read": "cache_read_tokens",
                 "cache_write": "cache_creation_tokens", "output": "output_tokens"}
# The only api_request attributes ever read; all others are skipped at read time.
KEEP = frozenset({"event.name", "event.timestamp", "request_id", "model", "query_source", "duration_ms",
                  "session.id", *TELEMETRY_KEY.values()})
TELEMETRY_ONLY = ("query_source_never_matched", "session_without_transcript", "unexplained")
TRANSCRIPT_ONLY = ("no_request_id", "session_not_emitting", "before_session_first_event",
                   "in_flight_at_window_end", "after_session_last_event", "unexplained")
SCAN_MARGIN = dt.timedelta(days=1)  # re-read transcripts modified up to a day before the window
TOP_DIFFS = 5
# Labels written out. Anything else becomes "other": a query source could one day carry a
# user-chosen agent name, so only the client's own vocabulary passes.
SAFE_LABEL = {
    "model": re.compile(r"^claude-[a-z0-9.\-]{1,60}$"),
    "query_source": re.compile(r"^(?:sdk|agent:custom|agent:builtin:[A-Za-z0-9\-]{1,40}|[a-z][a-z0-9_]{0,40})$"),
    "version": re.compile(r"^\d{1,4}\.\d{1,4}\.\d{1,6}$"),
}


# ------------------------------------------------------------------ telemetry
@dataclass(frozen=True)
class ApiRequest:
    request_id: str
    model: str
    query_source: str
    input: int
    output: int
    cache_read: int
    cache_write: int
    ts: str             # event.timestamp (ISO-8601 UTC)
    duration_ms: int
    session: str        # sha256 of the session id; in memory only, never written
    version: str        # client version (resource service.version)


def attr_value(v: Any) -> Any:
    """One OTLP AnyValue (JSON encoding) -> a Python value; None if empty or unknown."""
    if not isinstance(v, dict):
        return None
    if "stringValue" in v:
        return v["stringValue"]
    if "intValue" in v:  # int64 arrives as a JSON string per the OTLP/JSON spec, sometimes as a number
        try:
            return int(v["intValue"])
        except (TypeError, ValueError):
            return None
    if "doubleValue" in v:
        try:
            return float(v["doubleValue"])
        except (TypeError, ValueError):
            return None
    if "boolValue" in v:
        b = v["boolValue"]
        return b if isinstance(b, bool) else str(b).lower() == "true"
    if "arrayValue" in v:
        return [attr_value(x) for x in (v["arrayValue"] or {}).get("values") or []]
    if "kvlistValue" in v:
        return {x["key"]: attr_value(x.get("value")) for x in (v["kvlistValue"] or {}).get("values") or []
                if isinstance(x, dict) and isinstance(x.get("key"), str)}
    if "bytesValue" in v:
        return str(v["bytesValue"])
    return None


def read_attributes(attrs: Any, keep: Iterable[str]) -> dict[str, Any]:
    """Only the attributes named in `keep`; everything else is skipped unread."""
    keep = set(keep)
    out: dict[str, Any] = {}
    for a in attrs if isinstance(attrs, list) else []:
        if isinstance(a, dict) and a.get("key") in keep:
            out[a["key"]] = attr_value(a.get("value"))
    return out


def session_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _iso_from_nanos(nanos: Any) -> str | None:
    try:
        secs = int(nanos) / 1e9
    except (TypeError, ValueError):
        return None
    if secs <= 0:
        return None
    t = dt.datetime.fromtimestamp(secs, dt.timezone.utc)
    return t.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def read_telemetry(paths: Iterable[Path], until: str | None = None) -> tuple[list[ApiRequest], dict[str, Any]]:
    """api_request events from the sink's files, de-duplicated by request id (re-sent batches)."""
    paths = [Path(p) for p in paths]
    cut = claude_usage.parse_ts(until) if until else None
    st: dict[str, Any] = {"files": len(paths), "lines": 0, "bad_lines": 0, "log_batches": 0, "records": 0,
                          "api_request_records": 0, "duplicates": 0, "conflicting_duplicates": 0,
                          "no_request_id": 0, "no_timestamp": 0, "after_until": 0, "inputs": []}
    whole = hashlib.sha256()
    by_id: dict[str, ApiRequest] = {}
    for path in paths:
        data = path.read_bytes()
        whole.update(data)
        st["inputs"].append({"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
        for raw in data.decode("utf-8", errors="replace").split("\n"):
            if not raw.strip():
                continue
            st["lines"] += 1
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                st["bad_lines"] += 1
                continue
            body = row.get("body") if isinstance(row, dict) else None
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except json.JSONDecodeError:
                    st["bad_lines"] += 1
                    continue
            rls = body.get("resourceLogs") if isinstance(body, dict) else None
            if not isinstance(rls, list):
                continue
            st["log_batches"] += 1
            for rl in rls:
                if not isinstance(rl, dict):
                    continue
                res = read_attributes((rl.get("resource") or {}).get("attributes"), {"service.version"})
                version = str(res.get("service.version") or "unknown")
                for sl in rl.get("scopeLogs") or []:
                    for rec in (sl or {}).get("logRecords") or []:
                        st["records"] += 1
                        if not isinstance(rec, dict):
                            continue
                        a = read_attributes(rec.get("attributes"), KEEP)
                        if a.get("event.name") != "api_request":
                            continue
                        st["api_request_records"] += 1
                        e = _api_request(a, rec, version, st)
                        if e is None:
                            continue
                        if cut is not None and claude_usage.parse_ts(e.ts) >= cut:
                            st["after_until"] += 1
                            continue
                        if e.request_id in by_id:
                            st["duplicates"] += 1
                            st["conflicting_duplicates"] += by_id[e.request_id] != e
                            continue
                        by_id[e.request_id] = e
    st["sha256"] = whole.hexdigest()
    events = sorted(by_id.values(), key=lambda e: claude_usage.parse_ts(e.ts))
    return events, st


def _api_request(a: dict[str, Any], rec: dict[str, Any], version: str, st: dict[str, Any]) -> ApiRequest | None:
    rid = a.get("request_id")
    if not isinstance(rid, str) or not rid:
        st["no_request_id"] += 1
        return None
    ts = a.get("event.timestamp")
    if not isinstance(ts, str) or not ts:
        ts = _iso_from_nanos(rec.get("timeUnixNano"))
    if ts is None:
        st["no_timestamp"] += 1
        return None
    sid = a.get("session.id")
    n = claude_usage._count  # the instrument's own count normalisation
    return ApiRequest(request_id=rid, model=str(a.get("model") or "?"), query_source=str(a.get("query_source") or "?"),
                      input=n(a.get(TELEMETRY_KEY["input"])), output=n(a.get(TELEMETRY_KEY["output"])),
                      cache_read=n(a.get(TELEMETRY_KEY["cache_read"])),
                      cache_write=n(a.get(TELEMETRY_KEY["cache_write"])),
                      ts=ts, duration_ms=n(a.get("duration_ms")),
                      session=session_key(sid) if isinstance(sid, str) and sid else "", version=version)


# ------------------------------------------------------------------ transcripts
@dataclass
class LineInfo:
    request_id: str | None
    lines: int
    first_output: int
    sums: dict[str, int]
    rid_conflict: bool = False


@dataclass
class TRequest:
    request_id: str | None
    model: str
    ts: str
    session: str                 # sha256 of the session id from the path; in memory only
    paper: dict[str, int]        # the claude_usage record
    first_output: int            # first line's output
    lines: int                   # usage lines of this message id across the re-read files
    line_sums: dict[str, int] = field(default_factory=dict)


def transcript_session(path: Path) -> str:
    """Session key from the transcript's path: <session>.jsonl or <session>/subagents/<agent>.jsonl."""
    parts = list(path.parts)
    low = [p.lower() for p in parts]
    raw = parts[low.index("subagents") - 1] if "subagents" in low and low.index("subagents") > 0 else path.stem
    return session_key(raw)


def line_details(path: Path) -> dict[str, LineInfo]:
    """Per message id: request id, usage lines, first line's output, per-line sums.

    Same line filters as tools/claude_usage.read_transcript (lines split at newline only, a
    non-empty usage dict, synthetic placeholders dropped, a message id and a timestamp)."""
    out: dict[str, LineInfo] = {}
    n = claude_usage._count
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.split("\n"):  # never splitlines(): see tools/claude_usage.py
        if '"usage"' not in line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = ev.get("message") if isinstance(ev, dict) else None
        if not isinstance(msg, dict) or not isinstance(msg.get("usage"), dict) or not msg["usage"]:
            continue
        if msg.get("model") == "<synthetic>":
            continue
        mid, ts = msg.get("id"), ev.get("timestamp")
        if not isinstance(mid, str) or not mid or not isinstance(ts, str):
            continue
        u = msg["usage"]
        rid = ev.get("requestId")
        rid = rid if isinstance(rid, str) and rid else None
        info = out.get(mid)
        if info is None:
            out[mid] = LineInfo(rid, 1, n(u.get("output_tokens")), {f: n(u.get(TRANSCRIPT_KEY[f])) for f in FIELDS})
            continue
        info.lines += 1
        for f in FIELDS:
            info.sums[f] += n(u.get(TRANSCRIPT_KEY[f]))
        if rid and info.request_id and rid != info.request_id:
            info.rid_conflict = True
        if info.request_id is None:
            info.request_id = rid
    return out


def transcript_requests(projects: Path, since: dt.datetime | None = None) -> tuple[list[TRequest], dict[str, Any]]:
    """claude_usage records of every transcript modified since `since` - SCAN_MARGIN, with request ids."""
    projects = Path(projects)
    transcripts, ustats = claude_usage.extract(projects)
    files = sorted(projects.glob("**/*.jsonl"))
    floor = (since - SCAN_MARGIN).timestamp() if since is not None else None
    scanned = [p for p in files if floor is None or p.stat().st_mtime >= floor]
    details = {p: line_details(p) for p in scanned}
    lines_total: collections.Counter[str] = collections.Counter()
    sums_total: dict[str, collections.Counter[str]] = {f: collections.Counter() for f in FIELDS}
    for d in details.values():  # a line counter sees every copy, resumed transcripts included
        for mid, info in d.items():
            lines_total[mid] += info.lines
            for f in FIELDS:
                sums_total[f][mid] += info.sums[f]
    out: list[TRequest] = []
    conflicts = missing = 0
    for t in transcripts:
        d = details.get(t.path)
        if d is None:  # not modified since the window start: no request of it can lie inside
            continue
        skey = transcript_session(t.path)
        for r in t.requests:
            info = d.get(r.msg_id)
            if info is None:
                missing += 1
                continue
            conflicts += info.rid_conflict
            out.append(TRequest(request_id=info.request_id, model=r.model, ts=r.ts, session=skey,
                                paper={f: r.usage[TRANSCRIPT_KEY[f]] for f in FIELDS},
                                first_output=info.first_output, lines=lines_total[r.msg_id],
                                line_sums={f: sums_total[f][r.msg_id] for f in FIELDS}))
    st = {"files_read": ustats.files, "files_rescanned": len(scanned), "records_total": sum(len(t.requests) for t in transcripts),
          "records_rescanned": len(out), "request_id_conflicts": conflicts, "records_without_line_details": missing,
          "synthetic_lines": ustats.synthetic_lines, "cross_file_duplicates": ustats.cross_file_duplicates}
    return out, st


# ------------------------------------------------------------------ matching
def window(events: list[ApiRequest]) -> tuple[dt.datetime, dt.datetime]:
    """Earliest request start (event time minus duration) to the latest event time."""
    if not events:
        raise ValueError("no api_request events")
    starts = [claude_usage.parse_ts(e.ts) - dt.timedelta(milliseconds=e.duration_ms) for e in events]
    return min(starts), max(claude_usage.parse_ts(e.ts) for e in events)


def match(events: list[ApiRequest], trecs: list[TRequest], start: dt.datetime, end: dt.datetime) -> dict[str, Any]:
    tel = {e.request_id: e for e in events}
    trans: dict[str, TRequest] = {}
    duplicate_ids = 0  # transcript records sharing a request id with an earlier one (first kept)
    for r in trecs:
        if r.request_id:
            duplicate_ids += r.request_id in trans
            trans.setdefault(r.request_id, r)
    pairs = [(e, trans[e.request_id]) for e in events if e.request_id in trans]
    pt = claude_usage.parse_ts
    spans: dict[str, list[dt.datetime]] = {}
    for e in events:
        t = pt(e.ts)
        s = spans.setdefault(e.session, [t - dt.timedelta(milliseconds=e.duration_ms), t])
        s[0] = min(s[0], t - dt.timedelta(milliseconds=e.duration_ms))
        s[1] = max(s[1], t)
    matched_sources = {e.query_source for e, _ in pairs}
    transcript_sessions = {r.session for r in trecs}

    tel_only = {c: 0 for c in TELEMETRY_ONLY}
    tel_only_qs: collections.Counter[str] = collections.Counter()
    tel_only_model: collections.Counter[str] = collections.Counter()
    for e in events:
        if e.request_id in trans:
            continue
        if e.query_source not in matched_sources:
            tel_only["query_source_never_matched"] += 1
        elif e.session not in transcript_sessions:
            tel_only["session_without_transcript"] += 1
        else:
            tel_only["unexplained"] += 1
        tel_only_qs[label(e.query_source, "query_source")] += 1
        tel_only_model[label(e.model, "model")] += 1

    in_window = [r for r in trecs if start <= pt(r.ts) <= end]
    longest = max(e.duration_ms for e in events)
    in_flight_from = end - dt.timedelta(milliseconds=longest)
    tr_only = {c: 0 for c in TRANSCRIPT_ONLY}
    tr_only_model: collections.Counter[str] = collections.Counter()
    for r in in_window:
        if r.request_id and r.request_id in tel:
            continue
        span = spans.get(r.session)
        t = pt(r.ts)
        if not r.request_id:
            tr_only["no_request_id"] += 1
        elif span is None:
            tr_only["session_not_emitting"] += 1
        elif t < span[0]:
            tr_only["before_session_first_event"] += 1
        elif t > in_flight_from:
            tr_only["in_flight_at_window_end"] += 1
        elif t > span[1]:
            tr_only["after_session_last_event"] += 1
        else:
            tr_only["unexplained"] += 1
        tr_only_model[label(r.model, "model")] += 1
    return {
        "pairs": pairs,
        "in_window": in_window,
        "longest_request_ms": longest,
        "duplicate_request_ids": duplicate_ids,
        "matched_outside_window":sum(not start <= pt(r.ts) <= end for _, r in pairs),
        "model_mismatch": sum(e.model != r.model for e, r in pairs),
        "telemetry_only": {"total": sum(tel_only.values()), "by_class": tel_only,
                           "by_query_source": dict(sorted(tel_only_qs.items())),
                           "by_model": dict(sorted(tel_only_model.items()))},
        "transcript_only": {"total": sum(tr_only.values()), "by_class": tr_only,
                            "by_model": dict(sorted(tr_only_model.items()))},
    }


def label(s: str, kind: str) -> str:
    return s if SAFE_LABEL[kind].match(s or "") else "other"


# ------------------------------------------------------------------ comparison
def field_stats(pairs: list[tuple[int, int]]) -> dict[str, Any]:
    """(transcript, telemetry) pairs -> exact-agreement share, aggregate ratio, largest differences."""
    n = len(pairs)
    t = sum(a for a, _ in pairs)
    o = sum(b for _, b in pairs)
    diffs = sorted((a - b for a, b in pairs), key=lambda d: (-abs(d), d))
    return {"n": n, "exact_share": (sum(a == b for a, b in pairs) / n) if n else None,
            "ratio": (t / o) if o else None, "transcript_sum": t, "telemetry_sum": o,
            "below": sum(a < b for a, b in pairs), "above": sum(a > b for a, b in pairs),
            "max_abs_diff": abs(diffs[0]) if diffs else 0, "largest_abs_diffs": diffs[:TOP_DIFFS]}


def _div(a: float, b: float) -> float | None:
    return a / b if b else None


def compare(pairs: list[tuple[ApiRequest, TRequest]]) -> dict[str, Any]:
    first_sum = sum(r.first_output for _, r in pairs)
    tel_out = sum(e.output for e, _ in pairs)
    lines = sum(r.lines for _, r in pairs)
    return {
        "paper": {f: field_stats([(r.paper[f], getattr(e, f)) for e, r in pairs]) for f in FIELDS},
        "first_line": {"output": field_stats([(r.first_output, e.output) for e, r in pairs]),
                       "telemetry_over_first": _div(tel_out, first_sum),
                       "max_over_first": _div(sum(r.paper["output"] for _, r in pairs), first_sum)},
        "line_count": {**{f: field_stats([(r.line_sums[f], getattr(e, f)) for e, r in pairs]) for f in FIELDS},
                       "requests": len(pairs), "lines": lines, "lines_per_request": _div(lines, len(pairs)),
                       "single_line_share": _div(sum(r.lines == 1 for _, r in pairs), len(pairs))},
    }


def breakdown(events: list[ApiRequest], pairs: list[tuple[ApiRequest, TRequest]], key: str) -> dict[str, Any]:
    """Per model or per query source (telemetry's value): counts and the paper rule's agreement."""
    out: dict[str, Any] = {}
    for e in events:
        g = out.setdefault(label(getattr(e, key), key), {"telemetry": 0, "matched": 0})
        g["telemetry"] += 1
    grouped: dict[str, list[tuple[ApiRequest, TRequest]]] = collections.defaultdict(list)
    for e, r in pairs:
        grouped[label(getattr(e, key), key)].append((e, r))
    for name, ps in grouped.items():
        c = compare(ps)
        out[name]["matched"] = len(ps)
        out[name]["paper"] = {f: {"exact_share": c["paper"][f]["exact_share"], "ratio": c["paper"][f]["ratio"]}
                              for f in FIELDS}
        out[name]["first_line_output_ratio"] = c["first_line"]["output"]["ratio"]
        out[name]["lines_per_request"] = c["line_count"]["lines_per_request"]
    return dict(sorted(out.items()))


def window_replication(in_window: list[TRequest]) -> dict[str, Any]:
    """The paper's own transcript-only ratios on every in-window transcript request (no telemetry)."""
    first = sum(r.first_output for r in in_window)
    return {"requests": len(in_window), "lines": sum(r.lines for r in in_window),
            "lines_per_request": _div(sum(r.lines for r in in_window), len(in_window)),
            "max_over_first_output": _div(sum(r.paper["output"] for r in in_window), first)}


def telemetry_only_share(events: list[ApiRequest], pairs: list[tuple[ApiRequest, TRequest]]) -> dict[str, Any]:
    """Per token field: the share of all telemetry tokens carried by requests no transcript records.

    A meter that totals every request of a session (as the client's own per-session totals do)
    exceeds the transcripts by this much even where every matched request agrees exactly."""
    matched = {e.request_id for e, _ in pairs}
    out = {}
    for f in FIELDS:
        total = sum(getattr(e, f) for e in events)
        only = sum(getattr(e, f) for e in events if e.request_id not in matched)
        out[f] = {"telemetry_only": only, "all": total, "share": _div(only, total)}
    return out


# ------------------------------------------------------------------ evidence
def _minute(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def _num(v: float | None, fmt: str, source: str, desc: str) -> dict[str, Any]:
    return figure(v, fmt, source, desc) if v is not None else figure("n/a", "text", source, desc)


def build_figures(w: dict[str, Any], tst: dict[str, Any], m: dict[str, Any], rules: dict[str, Any],
                  rep: dict[str, Any], side: dict[str, Any], versions: list[str],
                  models: list[str]) -> dict[str, Any]:
    tel, trs = "otel_telemetry", "otel_transcripts"
    p, fl, lc = rules["paper"], rules["first_line"], rules["line_count"]
    n_tel, n_match = w["telemetry_requests"], len(m["pairs"])
    to, tr = m["telemetry_only"]["by_class"], m["transcript_only"]["by_class"]
    names = {"input": "Input", "cache_read": "Read", "cache_write": "Write", "output": "Output"}
    words = {"input": "uncached input", "cache_read": "cache reads", "cache_write": "cache writes", "output": "output"}
    figs = {
        "OtelWindowStartDate": figure(w["start_minute"][:10], "date", tel, "telemetry window start (UTC date)"),
        "OtelWindowEndDate": figure(w["end_minute"][:10], "date", tel, "telemetry window end (UTC date)"),
        "OtelWindowStartUTC": figure(w["start_minute"].replace("T", " ").rstrip("Z"), "text", tel,
                                     "telemetry window start, UTC, to the minute"),
        "OtelWindowEndUTC": figure(w["end_minute"].replace("T", " ").rstrip("Z"), "text", tel,
                                   "telemetry window end, UTC, to the minute"),
        "OtelWindowHours": figure(w["hours"], "float1", tel, "telemetry window length, hours"),
        "OtelRequests": figure(n_tel, "int", tel, "api_request events in the window (de-duplicated by request id)"),
        "OtelSessions": figure(w["telemetry_sessions"], "word", tel, "sessions that emitted api_request events"),
        "OtelClientVersion": figure(", ".join(versions), "text", tel, "Claude Code versions emitting the events"),
        "OtelModels": figure(", ".join(models), "text", tel, "model ids among matched requests"),
        "OtelMatched": figure(n_match, "int", tel, "requests matched on request id (telemetry and transcript)"),
        "OtelMatchedPct": figure(n_match / n_tel * 100, "pct1", tel, "matched share of telemetry requests"),
        "OtelTelemetryOnly": figure(m["telemetry_only"]["total"], "int", tel,
                                    "telemetry requests with no transcript record"),
        "OtelTelemetryOnlySide": figure(to["query_source_never_matched"], "int", tel,
                                        "telemetry-only requests whose query source never reaches transcripts "
                                        "(side calls such as session-title generation)"),
        "OtelTelemetryOnlyNoTranscript": figure(to["session_without_transcript"], "int", tel,
                                                "telemetry-only requests of sessions with no transcript record"),
        "OtelTelemetryOnlyUnexplained": figure(to["unexplained"], "int", tel, "telemetry-only requests, unexplained"),
        "OtelTranscriptInWindow": figure(w["transcript_in_window"], "int", trs,
                                         "transcript requests (paper rule) inside the telemetry window"),
        "OtelTranscriptOnly": figure(m["transcript_only"]["total"], "int", trs,
                                     "in-window transcript requests absent from telemetry"),
        "OtelTranscriptOnlyNotEmitting": figure(tr["session_not_emitting"] + tr["before_session_first_event"], "int",
                                                trs, "transcript-only requests from processes that did not emit "
                                                "(started before telemetry was enabled)"),
        "OtelTranscriptOnlyInFlight": figure(tr["in_flight_at_window_end"], "int", trs,
                                             "transcript-only requests whose first line lies within the longest "
                                             "request duration of the window end (response may have completed "
                                             "after the telemetry was copied)"),
        "OtelTranscriptOnlyAfterLast": figure(tr["after_session_last_event"], "int", trs,
                                              "transcript-only requests after their session's last exported event"),
        "OtelLongestRequestSec": figure(m["longest_request_ms"] / 1000, "int", tel,
                                        "longest api_request duration in the window, seconds"),
        "OtelTelemetryOnlyInputPct": _num(_pct(side["input"]["share"]), "pctlt1", tel,
                                          "share of all telemetry uncached input on requests no transcript records"),
        "OtelTelemetryOnlyOutputPct": _num(_pct(side["output"]["share"]), "pctlt1", tel,
                                           "share of all telemetry output on requests no transcript records"),
        "OtelTranscriptOnlyNoId": figure(tr["no_request_id"], "int", trs, "transcript-only requests without request id"),
        "OtelTranscriptOnlyUnexplained": figure(tr["unexplained"], "int", trs,
                                                "transcript-only requests inside their session's telemetry span"),
        "OtelModelMismatch": figure(m["model_mismatch"], "int", trs, "matched requests whose model ids differ"),
        "OtelMaxAbsDiff": figure(max(p[f]["max_abs_diff"] for f in FIELDS), "int", trs,
                                 "largest absolute per-request difference, any token field, paper rule"),
        "OtelReadGapPct": _num(abs(p["cache_read"]["ratio"] - 1) * 100 if p["cache_read"]["ratio"] is not None else None,
                               "pct2", trs, "cache reads: |transcript / telemetry - 1|, percent, paper rule"),
        "OtelFirstLineExactPct": _num(_pct(fl["output"]["exact_share"]), "pct1", trs,
                                      "matched requests whose first-line output equals telemetry output"),
        "OtelFirstLineRatio": _num(fl["output"]["ratio"], "float2", trs, "first-line output / telemetry output"),
        "OtelFirstLineFactor": _num(fl["telemetry_over_first"], "float2", trs,
                                    "telemetry output / first-line output (first-line undercount factor)"),
        "OtelMaxOverFirst": _num(fl["max_over_first"], "float2", trs,
                                 "matched requests: output max over lines / first-line output"),
        "OtelLineFactor": _num(lc["lines_per_request"], "float2", trs, "transcript usage lines per telemetry request"),
        "OtelSingleLinePct": _num(_pct(lc["single_line_share"]), "pct1", trs,
                                  "matched requests written as a single usage line"),
        "OtelLineReadRatio": _num(lc["cache_read"]["ratio"], "float2", trs,
                                  "line counter: summed cache reads over all lines / telemetry cache reads"),
        "OtelLineOutputRatio": _num(lc["output"]["ratio"], "float2", trs,
                                    "line counter: summed output over all lines / telemetry output"),
        "OtelWindowLineFactor": _num(rep["lines_per_request"], "float2", trs,
                                     "all in-window transcript requests: usage lines per request"),
        "OtelWindowMaxOverFirst": _num(rep["max_over_first_output"], "float2", trs,
                                       "all in-window transcript requests: output max over lines / first line"),
    }
    for f in FIELDS:
        figs[f"OtelExact{names[f]}Pct"] = _num(_pct(p[f]["exact_share"]), "pct1", trs,
                                               f"matched requests whose {words[f]} agree exactly, paper rule")
        figs[f"OtelRatio{names[f]}"] = _num(p[f]["ratio"], "float3", trs,
                                            f"{words[f]}: transcript / telemetry, paper rule")
    return figs


def _pct(share: float | None) -> float | None:
    return share * 100 if share is not None else None


def _command(paths: list[Path], projects_default: bool, until: str | None) -> str:
    cmd = "python tools/otel_meter.py --telemetry " + " ".join(f"$OTEL_DIR/{p.name}" for p in paths)
    if not projects_default:
        cmd += " --projects $CLAUDE_PROJECTS"
    if until:
        cmd += f" --until {until}"
    return cmd


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--telemetry", type=Path, nargs="+", required=True,
                    help="local copies of the sink's claude_code_otel.jsonl[.*] files")
    ap.add_argument("--projects", type=Path, default=claude_usage.DEFAULT_PROJECTS)
    ap.add_argument("--until", help="drop api_request events at or after this ISO time (reproduce an earlier run)")
    ap.add_argument("--out", type=Path, default=EVIDENCE / "otel_meter.json")
    args = ap.parse_args(argv)

    paths = sorted(args.telemetry, key=lambda p: p.name)
    events, tst = read_telemetry(paths, args.until)
    if not events:
        print("BLOCKED: no api_request events in the telemetry")
        return 1
    start, end = window(events)
    trecs, ust = transcript_requests(args.projects, since=start)
    m = match(events, trecs, start, end)
    if not m["pairs"]:
        print("BLOCKED: no telemetry request matched a transcript request")
        return 1
    rules = compare(m["pairs"])
    rep = window_replication(m["in_window"])
    side = telemetry_only_share(events, m["pairs"])
    versions = sorted({label(e.version, "version") for e in events})
    models = sorted({label(e.model, "model") for e, _ in m["pairs"]})
    after_freeze = start >= claude_usage.parse_ts(FREEZE)
    w = {"start_minute": _minute(start), "end_minute": _minute(end),
         "hours": round((end - start).total_seconds() / 3600, 3), "after_snapshot_freeze": after_freeze,
         "snapshot_freeze": FREEZE, "telemetry_requests": len(events),
         "telemetry_sessions": len({e.session for e in events}), "transcript_in_window": len(m["in_window"])}
    aggregates = {
        "window": w,
        "telemetry": {k: v for k, v in tst.items() if k != "sha256"},
        "transcripts": ust,
        "matching": {"matched": len(m["pairs"]), "matched_outside_window": m["matched_outside_window"],
                     "model_mismatch": m["model_mismatch"], "longest_request_ms": m["longest_request_ms"],
                     "duplicate_request_ids": m["duplicate_request_ids"],
                     "telemetry_only": m["telemetry_only"], "transcript_only": m["transcript_only"]},
        "telemetry_only_token_share": side,
        "rules": rules,
        "transcript_window_replication": rep,
        "by_model": breakdown(events, m["pairs"], "model"),
        "by_query_source": breakdown(events, m["pairs"], "query_source"),
        "by_client_version": breakdown(events, m["pairs"], "version"),
        "client_versions": versions,
    }
    record = json.dumps(aggregates, sort_keys=True)
    cmd = _command(paths, args.projects == claude_usage.DEFAULT_PROJECTS, args.until)
    span = f"{w['start_minute']} to {w['end_minute']} UTC"
    evidence = {
        "schema_version": 1, "kind": "otel_meter",
        "provenance": {"generated_at": utc_now(), "generator": GENERATOR, "sources": [
            {"id": "otel_telemetry", "grade": "measured", "sha256": tst["sha256"], "command": cmd,
             "description": f"Claude Code OpenTelemetry api_request events (the usage the API response reported, "
                            f"as the client exports it), window {span}, {len(events)} requests from "
                            f"{len(paths)} telemetry file(s); sha256 of the files concatenated in file-name order. "
                            f"Read: request id, model, query source, four token counts, duration, timestamp, and "
                            f"the session id hashed in memory for classification; nothing else"},
            {"id": "otel_transcripts", "grade": "measured", "sha256": sha256_text(record), "command": cmd,
             "description": f"{ust['files_read']} transcript files read with tools/claude_usage.py (one record per "
                            f"message id: input fields from the first line, output the maximum over lines, synthetic "
                            f"dropped); {ust['files_rescanned']} files modified since the window start re-read for "
                            f"request ids and the alternative rules; sha256 of the aggregate comparison record"},
        ], "notes": [
            f"Telemetry window (UTC, to the minute): {span}; "
            f"{'later than' if after_freeze else 'overlapping'} the frozen snapshot (freeze {FREEZE}).",
            "Ratios are transcript / telemetry over matched requests; exact share = share of matched requests "
            "whose value is identical.",
            "Both channels come from the same client process: telemetry carries the API response's usage, the "
            "transcripts what the client wrote to disk. Agreement tests the transcripts against the API's reported "
            "usage, not against the provider's billing.",
        ]},
        "aggregates": aggregates,
        "figures": build_figures(w, tst, m, rules, rep, side, versions, models),
    }
    write_json(args.out, evidence)
    p = rules["paper"]
    print(f"wrote {args.out.name}: window {span}; telemetry {len(events)} requests, matched {len(m['pairs'])}, "
          f"telemetry-only {m['telemetry_only']['total']}, transcript-only in window {m['transcript_only']['total']}")
    for f in FIELDS:
        r = p[f]["ratio"]
        print(f"  {f:<11} exact {p[f]['exact_share']:.3f}  ratio {r if r is None else round(r, 4)}  "
              f"max|diff| {p[f]['max_abs_diff']}")
    print(f"  first-line output ratio {rules['first_line']['output']['ratio']}, "
          f"lines per request {rules['line_count']['lines_per_request']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
