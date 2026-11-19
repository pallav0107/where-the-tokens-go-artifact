#!/usr/bin/env python3
"""tokens_contrib.py: aggregate token and prompt-cache figures from your own Claude Code logs.

WHAT IT DOES
  Reads the Claude Code session logs on this computer (by default ~/.claude/projects/**/*.jsonl),
  computes the figures of the workshop paper "Where the Tokens Go: Auditing Cache Reads and
  Re-Writes in a Multi-Agent Coding Harness" (AGENT 2027) with the paper's own rules, prints the
  result and writes it to tokens_contrib_result.json. Nothing is sent anywhere: you read the file
  and decide whether to send it.

WHAT THE RESULT CONTAINS
  Counts, token totals and percentages; the calendar months covered (YYYY-MM); Claude model names;
  Claude Code version numbers; this script's own sha256. Never prompts, replies, tool names or
  tool inputs, file contents, paths, project or folder names, session, message or request ids,
  timestamps finer than the month, user names, emails or host names. Before writing, a guard
  refuses the result if any text in it is not a month, a Claude model name, a version number,
  this script's sha256 or the result's fixed kind label.

USAGE
  python tokens_contrib.py                  read ~/.claude/projects, write tokens_contrib_result.json
  python tokens_contrib.py --self-test      check this script on a built-in synthetic log
  options:  --root DIR       read the logs under DIR instead
            --since DATE     only requests at or after 00:00 UTC on DATE (YYYY-MM-DD)
            --until DATE     only requests before 00:00 UTC on DATE (YYYY-MM-DD)
            --out FILE       write the result to FILE instead

  Python 3.9 or newer, standard library only; Windows, macOS and Linux.

RULES (the paper's, unchanged)
  * One record per API request: Claude Code writes one response as several log lines that repeat
    the message id and its input usage. Input and cache fields come from the id's first line,
    output tokens are the maximum over its lines, model "<synthetic>" (no API call) is dropped,
    and a message id already counted in an earlier transcript (a resumed copy) is skipped.
    Lines end at "\\n" only (never str.splitlines()).
  * Prices: Anthropic list prices per model and token kind (PRICES below, from the official
    pricing page, accessed 2026-11-15). Requests on a model not in the table are counted in every
    token figure but left out of the cost figures, and reported as unpriced.
  * Cache writes, per transcript: after request t-1 the positions [0, read + write) are cached;
    request t writes [read, read + write), and the part below the previous cached length is a
    RE-WRITE, the rest a NEW SUFFIX.
  * Cause of a re-write, in this order: model change; a compaction marker between the two
    requests; a config change (tool, agent or MCP-instructions list delta, changed system-prompt
    snapshot, or changed Claude Code version); else by the idle gap since the previous request
    and the cache lifetime of the entries covering the re-written positions: more than an hour
    is expiry; five to sixty minutes is expiry for five-minute entries and within lifetime for
    one-hour entries; five minutes or less is within lifetime.
  * Line-count factor: usage-carrying log lines per de-duplicated request.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

SCHEMA_VERSION = 1
KIND = "tokens_contrib_result"
RESULT_NAME = "tokens_contrib_result.json"

USAGE_KEYS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")
# Characters str.splitlines() treats as line ends that may appear raw inside a JSON string.
UNICODE_LINE_SEPARATORS = ("\u0085", "\u2028", "\u2029")
# Attachments that record a change to the tool, agent or MCP-instructions list.
LIST_DELTAS = ("deferred_tools_delta", "agent_listing_delta", "mcp_instructions_delta")
FIVE_MIN, ONE_HOUR = 300.0, 3600.0
CAUSES = ("model_change", "compaction", "config_change", "gap_gt_60min", "gap_5_to_60min", "within_lifetime")
GAP_BINS = ("le_5min", "5_to_60min", "gt_60min")
ONE_HOUR_SESSION = 0.99   # a transcript with at least this share of written tokens at the one-hour rate

# Anthropic list prices, USD per million tokens: the paper's evidence/params/anthropic_list_prices.json
# ("Model pricing" table of https://platform.claude.com/docs/en/about-claude/pricing, accessed
# 2026-11-15, retrieved page sha256 048ccb65c7aa39a16f8475d92bec1730bc9e42c63b79085e2f0f1e6e5f867cd1).
PRICES: Dict[str, Dict[str, float]] = {
    "claude-fable-5-1": {"input": 10.0, "cache_write_5m": 12.5, "cache_write_1h": 20.0, "cache_hit": 0.25, "output": 50.0},
    "claude-fable-5": {"input": 10.0, "cache_write_5m": 12.5, "cache_write_1h": 20.0, "cache_hit": 1.0, "output": 50.0},
    "claude-opus-5-5": {"input": 4.0, "cache_write_5m": 5.0, "cache_write_1h": 8.0, "cache_hit": 0.2, "output": 20.0},
    "claude-opus-5": {"input": 5.0, "cache_write_5m": 6.25, "cache_write_1h": 10.0, "cache_hit": 0.5, "output": 25.0},
    "claude-opus-4-8": {"input": 5.0, "cache_write_5m": 6.25, "cache_write_1h": 10.0, "cache_hit": 0.5, "output": 25.0},
    "claude-sonnet-5-5": {"input": 2.0, "cache_write_5m": 2.5, "cache_write_1h": 4.0, "cache_hit": 0.1, "output": 10.0},
    "claude-sonnet-5": {"input": 2.0, "cache_write_5m": 2.5, "cache_write_1h": 4.0, "cache_hit": 0.2, "output": 10.0},
    "claude-sonnet-4-6": {"input": 3.0, "cache_write_5m": 3.75, "cache_write_1h": 6.0, "cache_hit": 0.3, "output": 15.0},
    "claude-haiku-4-5": {"input": 1.0, "cache_write_5m": 1.25, "cache_write_1h": 2.0, "cache_hit": 0.1, "output": 5.0},
}

# The only text the result may carry (the privacy guard in check_result enforces it; always fullmatch).
# A model name is written only if it is a canonical Claude model id (optionally with a cloud region or
# "anthropic." prefix, a date, a Bedrock "-v1:0" or Vertex "@date" suffix); anything else, such as a
# gateway alias or deployment name chosen by a person or company, is written as "other".
SAFE_MONTH = re.compile(r"[0-9]{4}-(?:0[1-9]|1[0-2])", re.ASCII)
SAFE_MODEL = re.compile(
    r"(?:(?:us|eu|apac|global|jp|au|ca|uk|us-gov)\.)?(?:anthropic\.)?claude-"
    r"(?:(?:opus|sonnet|haiku|fable|mythos)(?:-[0-9]{1,2}){0,3}|[0-9](?:-[0-9])?-(?:opus|sonnet|haiku))"
    r"(?:-[0-9]{8})?(?:-v[0-9]{1,2}:[0-9]{1,2})?(?:-fast)?(?:@[0-9]{8})?", re.ASCII)
SAFE_VERSION = re.compile(r"[0-9]{1,4}\.[0-9]{1,4}\.[0-9]{1,6}", re.ASCII)
SAFE_SHA = re.compile(r"[0-9a-f]{64}", re.ASCII)
SAFE_KEY = re.compile(r"[a-z0-9_]{1,40}", re.ASCII)
MAX_COUNT = 10 ** 15      # a usage field above this (or not finite) is not a token count: read as 0
OTHER = "other"


# ---------------------------------------------------------------------------------------------
# Reading the logs (vendored from the paper's tools/claude_usage.py, chain mode)
# ---------------------------------------------------------------------------------------------
_TS = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:\d{2})?$")


def parse_ts(ts: Any) -> Optional[dt.datetime]:
    """An ISO-8601 timestamp as an aware UTC datetime (None if it is not one).

    Same instant as datetime.fromisoformat(ts.replace("Z", "+00:00")) on Python 3.11+, but also
    on 3.9 and 3.10, which accept only three or six fractional digits."""
    m = _TS.match(ts) if isinstance(ts, str) else None
    if not m:
        return None
    y, mo, d, h, mi, s, frac, tz = m.groups()
    try:
        t = dt.datetime(int(y), int(mo), int(d), int(h), int(mi), int(s),
                        int((frac or "0")[:6].ljust(6, "0")), tzinfo=dt.timezone.utc)
        if tz and tz != "Z":
            sign = 1 if tz[0] == "+" else -1
            t -= sign * dt.timedelta(hours=int(tz[1:3]), minutes=int(tz[4:6]))
    except (ValueError, OverflowError):
        return None
    return t


@dataclass
class Request:
    msg_id: str
    ts: str
    t: dt.datetime
    model: str
    usage: Dict[str, Any]
    order: int                       # line position in its transcript
    separator_only: bool = False     # every line of this id carries a Unicode line separator
    parent_id: Optional[str] = None  # nearest earlier request on this request's message chain
    version: Optional[str] = None    # client version on the request's first line


@dataclass
class Transcript:
    order: int    # path order (tie-breaker)
    kind: str     # "main" | "subagent"
    role: str     # kept in memory only (equivalence with the paper's log); never written out
    requests: List[Request] = field(default_factory=list)
    compactions: List[int] = field(default_factory=list)
    deltas: List[int] = field(default_factory=list)
    prompts: List[Tuple[int, str]] = field(default_factory=list)


@dataclass
class Stats:
    files: int = 0
    unreadable_files: int = 0
    usage_lines: int = 0
    synthetic_lines: int = 0
    bad_lines: int = 0
    no_id_lines: int = 0
    bad_ts_lines: int = 0
    repeated_lines: int = 0
    outside_window: int = 0
    cross_file_duplicates: int = 0
    write_split_mismatch: int = 0
    separator_lines: int = 0
    separator_only_requests: int = 0
    files_without_requests: int = 0
    files_only_outside_window: int = 0
    files_emptied_by_dedup: int = 0
    # A reader that counts every usage line as a request (no de-duplication), in the window.
    line_requests: int = 0
    line_input: int = 0
    line_cache_read: int = 0


def _count(v: Any) -> int:
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not 0 < v <= MAX_COUNT:
        return 0       # also NaN and infinity (the comparison is False for NaN; MAX_COUNT stops infinity)
    return int(v)


def _norm_usage(u: Dict[str, Any], stats: Stats) -> Dict[str, Any]:
    out: Dict[str, Any] = {k: _count(u.get(k)) for k in USAGE_KEYS}
    cc = u.get("cache_creation") if isinstance(u.get("cache_creation"), dict) else {}
    w5, w1 = _count(cc.get("ephemeral_5m_input_tokens")), _count(cc.get("ephemeral_1h_input_tokens"))
    if w5 + w1 != out["cache_creation_input_tokens"]:
        # The paper's rule: price any unexplained remainder as 5-minute writes.
        stats.write_split_mismatch += 1
        w5 = max(out["cache_creation_input_tokens"] - w1, 0)
        w1 = out["cache_creation_input_tokens"] - w5
    out["cache_creation"] = {"ephemeral_5m_input_tokens": w5, "ephemeral_1h_input_tokens": w1}
    return out


def _is_compaction(ev: Dict[str, Any]) -> bool:
    return (ev.get("type") == "system" and ev.get("subtype") == "compact_boundary") or ev.get("isCompactSummary") is True


def _nearest_request(start: Any, mid: str, parent: Dict[str, Any], line_msg: Dict[str, str]) -> Optional[str]:
    """Walk parentUuid links from `start` to the first line that belongs to another request."""
    seen = set()
    u = start
    while isinstance(u, str) and u not in seen:
        seen.add(u)
        other = line_msg.get(u)
        if other is not None and other != mid:
            return other
        if u not in parent:
            return None
        u = parent[u]
    return None


def _in_window(t: dt.datetime, since: Optional[dt.datetime], until: Optional[dt.datetime]) -> bool:
    return (since is None or t >= since) and (until is None or t < until)


def read_transcript(path: Path, order: int, stats: Stats, since: Optional[dt.datetime] = None,
                    until: Optional[dt.datetime] = None) -> Transcript:
    kind = "subagent" if "subagents" in [p.lower() for p in path.parts] else "main"
    role = "main" if kind == "main" else "subagent(unattributed)"
    by_id: Dict[str, Request] = {}
    parent: Dict[str, Any] = {}          # line uuid -> parentUuid
    line_msg: Dict[str, str] = {}        # line uuid -> message id of a request line
    first_parent: Dict[str, Any] = {}    # message id -> parentUuid of its first line
    compactions: List[int] = []
    deltas: List[int] = []
    prompts: List[Tuple[int, str]] = []
    versions: Dict[str, Any] = {}        # message id -> client version of its first line
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        stats.unreadable_files += 1
        return Transcript(order, kind, role)
    for n, line in enumerate(text.split("\n")):  # never splitlines(): see the module docstring
        if kind == "subagent" and role == "subagent(unattributed)" and '"attributionAgent"' in line:
            try:
                found = json.loads(line).get("attributionAgent")
            except (ValueError, RecursionError, AttributeError):
                found = None
            if isinstance(found, str) and found:
                role = found
        ev: Any = None
        if '"uuid"' in line:
            try:
                ev = json.loads(line)
            except (ValueError, RecursionError):
                ev = None
            if isinstance(ev, dict):
                if isinstance(ev.get("uuid"), str):
                    parent[ev["uuid"]] = ev.get("parentUuid")
                if _is_compaction(ev):
                    compactions.append(n)
                att = ev.get("attachment") if ev.get("type") == "attachment" else None
                if isinstance(att, dict):
                    if att.get("type") in LIST_DELTAS:
                        deltas.append(n)
                    elif att.get("type") == "prompt_snapshot":
                        try:
                            digest = json.dumps(att.get("systemPrompt"), sort_keys=True, ensure_ascii=False)
                        except (ValueError, RecursionError):
                            digest = "?"
                        prompts.append((n, hashlib.sha256(digest.encode("utf-8")).hexdigest()))
        if '"usage"' not in line:
            continue
        if ev is None:
            try:
                ev = json.loads(line)
            except (ValueError, RecursionError):
                stats.bad_lines += 1
                continue
        msg = ev.get("message") if isinstance(ev, dict) else None
        if not isinstance(msg, dict) or not isinstance(msg.get("usage"), dict) or not msg["usage"]:
            continue
        stats.usage_lines += 1
        has_sep = any(c in line for c in UNICODE_LINE_SEPARATORS)
        stats.separator_lines += has_sep
        if msg.get("model") == "<synthetic>":
            stats.synthetic_lines += 1
            continue
        mid, ts = msg.get("id"), ev.get("timestamp")
        if not isinstance(mid, str) or not mid or not isinstance(ts, str):
            stats.no_id_lines += 1
            continue
        t = parse_ts(ts)
        if t is None:
            stats.bad_ts_lines += 1
            continue
        if isinstance(ev.get("uuid"), str):
            line_msg[ev["uuid"]] = mid
        if _in_window(t, since, until):
            u = msg["usage"]
            stats.line_requests += 1
            stats.line_input += sum(_count(u.get(k)) for k in USAGE_KEYS[:3])
            stats.line_cache_read += _count(u.get("cache_read_input_tokens"))
        if mid in by_id:
            stats.repeated_lines += 1
            prev = by_id[mid]
            prev.usage["output_tokens"] = max(prev.usage["output_tokens"], _count(msg["usage"].get("output_tokens")))
            prev.separator_only = prev.separator_only and has_sep
            continue
        first_parent[mid] = ev.get("parentUuid")
        versions[mid] = ev.get("version")
        by_id[mid] = Request(mid, ts, t, str(msg.get("model") or "?"), _norm_usage(msg["usage"], stats), n, has_sep)
    for mid, r in by_id.items():
        r.parent_id = _nearest_request(first_parent.get(mid), mid, parent, line_msg)
        v = versions.get(mid)
        r.version = v if isinstance(v, str) and v else None
    tr = Transcript(order, kind, role, compactions=compactions, deltas=deltas, prompts=prompts)
    tr.requests = sorted(by_id.values(), key=lambda r: (r.t, r.order))
    return tr


def extract(root: Path, since: Optional[dt.datetime] = None, until: Optional[dt.datetime] = None,
            progress: Optional[Callable[[int, int], None]] = None) -> Tuple[List[Transcript], Stats]:
    """Every transcript with at least one request in the window, de-duplicated across files."""
    stats = Stats()
    files = sorted(root.glob("**/*.jsonl"))
    stats.files = len(files)
    kept_ts: List[Transcript] = []
    for i, path in enumerate(files):
        if progress is not None:
            progress(i, len(files))
        tr = read_transcript(path, i, stats, since, until)
        if not tr.requests:
            stats.files_without_requests += 1
            continue
        if since is not None or until is not None:
            kept = [r for r in tr.requests if _in_window(r.t, since, until)]
            stats.outside_window += len(tr.requests) - len(kept)
            tr.requests = kept
            if not kept:
                stats.files_only_outside_window += 1
                continue
        kept_ts.append(tr)
    kept_ts.sort(key=lambda tr: (tr.requests[0].t, tr.order))
    seen = set()
    out = []
    for tr in kept_ts:
        kept = [r for r in tr.requests if r.msg_id not in seen]
        stats.cross_file_duplicates += len(tr.requests) - len(kept)
        seen.update(r.msg_id for r in kept)
        tr.requests = kept
        if kept:
            out.append(tr)
        else:
            stats.files_emptied_by_dedup += 1
    stats.separator_only_requests = sum(r.separator_only for tr in out for r in tr.requests)
    return out, stats


def records(transcripts: Iterable[Transcript]) -> Iterable[Dict[str, Any]]:
    """The paper's per-request log records: turn = transcript index, step = request index."""
    for ti, tr in enumerate(transcripts):
        for si, r in enumerate(tr.requests):
            yield {"turn": ti, "step": si, "label": tr.role, "kind": tr.kind, "model": r.model,
                   "ts": r.ts, "usage": r.usage}


def _prompt_at(prompts: List[Tuple[int, str]], line: int) -> Optional[str]:
    """Digest of the latest system-prompt snapshot before `line` (None if there is none)."""
    found = None
    for n, digest in prompts:
        if n >= line:
            break
        found = digest
    return found


def annotations(transcripts: Iterable[Transcript]) -> Iterable[Dict[str, Any]]:
    """Per request, in records() order: the markers between it and the previous kept request."""
    for ti, tr in enumerate(transcripts):
        for si, r in enumerate(tr.requests):
            if si == 0:
                yield {"turn": ti, "step": si, "after_compact": False, "chain_from_prev": None,
                       "after_list_delta": False, "prompt_changed": False, "version_changed": False}
                continue
            prev = tr.requests[si - 1]
            lo, hi = sorted((prev.order, r.order))
            p_lo, p_hi = _prompt_at(tr.prompts, lo), _prompt_at(tr.prompts, hi)
            yield {"turn": ti, "step": si, "after_compact": any(lo < b < hi for b in tr.compactions),
                   "chain_from_prev": r.parent_id == prev.msg_id,
                   "after_list_delta": any(lo < b < hi for b in tr.deltas),
                   "prompt_changed": p_lo is not None and p_hi is not None and p_lo != p_hi,
                   "version_changed": bool(prev.version and r.version and prev.version != r.version)}


def annotated_records(transcripts: List[Transcript]) -> List[Dict[str, Any]]:
    recs = list(records(transcripts))
    anns = list(annotations(transcripts))
    if len(recs) != len(anns):
        raise RuntimeError("annotations out of step with the records")
    for r, a in zip(recs, anns):
        if (r["turn"], r["step"]) != (a["turn"], a["step"]):
            raise RuntimeError("annotations out of step with the records")
        for k in ("after_compact", "chain_from_prev", "after_list_delta", "prompt_changed", "version_changed"):
            r[k] = a[k]
    return recs


# ---------------------------------------------------------------------------------------------
# Prices and cost by token kind (vendored from the paper's tools/pull_evidence.py)
# ---------------------------------------------------------------------------------------------
def price_for(model: str, models: Dict[str, Dict[str, float]] = PRICES) -> Optional[Dict[str, float]]:
    return models.get(model) or models.get(re.sub(r"-\d{8}$", "", model))


def request_cost(usage: Dict[str, Any], p: Dict[str, float]) -> Dict[str, float]:
    """List-price cost of one request by token kind, USD."""
    cc = usage["cache_creation"]
    return {
        "fresh": usage["input_tokens"] * p["input"] / 1e6,
        "cacheWrite": (cc["ephemeral_5m_input_tokens"] * p["cache_write_5m"]
                       + cc["ephemeral_1h_input_tokens"] * p["cache_write_1h"]) / 1e6,
        "cacheRead": usage["cache_read_input_tokens"] * p["cache_hit"] / 1e6,
        "output": usage["output_tokens"] * p["output"] / 1e6,
    }


def cost_components(transcripts: List[Transcript]) -> Dict[str, Any]:
    comp = {k: 0.0 for k in ("fresh", "cacheWrite", "cacheRead", "output")}
    unpriced, unpriced_input = 0, 0
    for tr in transcripts:
        for r in tr.requests:
            p = price_for(r.model)
            if p is None:
                unpriced += 1
                unpriced_input += sum(r.usage[k] for k in USAGE_KEYS[:3])
                continue
            for k, v in request_cost(r.usage, p).items():
                comp[k] += v
    return {"components": comp, "unpriced_requests": unpriced, "unpriced_input_tokens": unpriced_input}


# ---------------------------------------------------------------------------------------------
# New suffix or re-write, and the cause of each re-write (vendored from tools/cache_writes.py)
# ---------------------------------------------------------------------------------------------
@dataclass
class Block:
    start: int
    end: int
    f1: float     # share of the block written at the one-hour rate


def _gap_bin(gap: Optional[float]) -> str:
    if gap is None or gap <= FIVE_MIN:
        return "le_5min"
    return "5_to_60min" if gap <= ONE_HOUR else "gt_60min"


def config_changed(r: Dict[str, Any]) -> bool:
    """A change outside the messages: list delta, changed system-prompt snapshot or client version."""
    return bool(r.get("after_list_delta") or r.get("prompt_changed") or r.get("version_changed"))


def rewrite_causes(model_changed: bool, after_compact: bool, config_change: bool, gap: Optional[float],
                   share1: float) -> Dict[str, float]:
    """Observed cause of a re-write, as shares summing to one (see the module docstring)."""
    if model_changed:
        return {"model_change": 1.0}
    if after_compact:
        return {"compaction": 1.0}
    if config_change:
        return {"config_change": 1.0}
    b = _gap_bin(gap)
    if b == "gt_60min":
        return {"gap_gt_60min": 1.0}
    if b == "5_to_60min":
        return {"gap_5_to_60min": 1.0 - share1, "within_lifetime": share1}
    return {"within_lifetime": 1.0}


def analyse(recs: List[Dict[str, Any]], prices: Dict[str, Dict[str, float]] = PRICES) -> Dict[str, Any]:
    """The paper's positional analysis, restricted to the figures this kit reports."""
    by_turn: Dict[Any, List[Dict[str, Any]]] = {}
    for r in recs:
        by_turn.setdefault(r["turn"], []).append(r)
    out: Dict[str, Any] = {
        "write_tokens": 0, "new_suffix": 0, "rewrite": 0,
        "rewrite_by_gap": {b: 0 for b in GAP_BINS},
        "write_cost_usd": {"new_suffix": 0.0, "rewrite": 0.0},
        "rewrite_by_cause": {c: 0 for c in CAUSES},
        "total_cost_usd": 0.0, "unpriced_requests": 0}
    base = {"rewrite_tokens": 0.0, "exposure_tokens": 0, "requests": 0, "sessions": 0}
    gt60: List[Tuple[float, int]] = []    # (re-written tokens, cached prefix) of expiry re-writes
    for reqs in by_turn.values():
        reqs.sort(key=lambda r: r["step"])
        cached, prev_t = 0, None
        prev: Optional[Dict[str, Any]] = None
        wr_all = sum(r["usage"]["cache_creation_input_tokens"] for r in reqs)
        w1_all = sum(r["usage"]["cache_creation"]["ephemeral_1h_input_tokens"] for r in reqs)
        one_hour_session = bool(wr_all) and w1_all >= ONE_HOUR_SESSION * wr_all
        base["sessions"] += one_hour_session
        live: List[Block] = []
        for r in reqs:
            u = r["usage"]
            rd, wr = u["cache_read_input_tokens"], u["cache_creation_input_tokens"]
            w1 = u["cache_creation"]["ephemeral_1h_input_tokens"]
            f1 = w1 / wr if wr else 0.0
            t = parse_ts(r["ts"])
            gap = (t - prev_t).total_seconds() if prev_t is not None else None
            rewrite = max(0, min(cached, rd + wr) - rd)
            new = wr - rewrite
            p = price_for(r["model"], prices)
            out["write_tokens"] += wr
            out["new_suffix"] += new
            out["rewrite"] += rewrite
            unit = 0.0
            if p is None:
                out["unpriced_requests"] += 1
            else:
                unit = (f1 * p["cache_write_1h"] + (1 - f1) * p["cache_write_5m"]) / 1e6
                out["write_cost_usd"]["new_suffix"] += new * unit
                out["write_cost_usd"]["rewrite"] += rewrite * unit
                out["total_cost_usd"] += (wr * unit + (u["input_tokens"] * p["input"] + rd * p["cache_hit"]
                                                       + u["output_tokens"] * p["output"]) / 1e6)
            # Lifetime tier of the live entries covering the re-written positions [rd, rd + rewrite);
            # positions no block of this transcript covers take this request's tier.
            share1 = f1
            if rewrite:
                cov1 = cov5 = 0.0
                for b in live:
                    ov = max(0, min(rd + rewrite, b.end) - max(rd, b.start))
                    cov1 += ov * b.f1
                    cov5 += ov * (1 - b.f1)
                share1 = (cov1 + (rewrite - cov1 - cov5) * f1) / rewrite
            causes: Dict[str, float] = {}
            model_changed = after_compact = config = False
            if prev is not None:
                model_changed = r["model"] != prev["model"]
                after_compact = bool(r.get("after_compact"))
                config = config_changed(r)
                causes = rewrite_causes(model_changed, after_compact, config, gap, share1)
            if prev is not None and gap is not None and gap <= ONE_HOUR and one_hour_session \
                    and not (model_changed or after_compact or config):
                base["rewrite_tokens"] += rewrite * causes.get("within_lifetime", 0.0)
                base["exposure_tokens"] += cached
                base["requests"] += 1
            if rewrite:
                out["rewrite_by_gap"][_gap_bin(gap)] += rewrite
                for c, share in causes.items():
                    out["rewrite_by_cause"][c] += rewrite * share
                if causes.get("gap_gt_60min") == 1.0:
                    gt60.append((rewrite, cached))
                kept = []
                for b in live:              # the cache below `rd` survived; above it was lost
                    if b.end <= rd:
                        kept.append(b)
                    elif b.start < rd:      # split: the part above rd is dead
                        b.end = rd
                        kept.append(b)
                live = kept
            if wr and p is not None:
                live.append(Block(rd, rd + wr, f1))
            cached, prev_t, prev = rd + wr, t, r
    rate = base["rewrite_tokens"] / base["exposure_tokens"] if base["exposure_tokens"] else 0.0
    base["rate"] = rate
    out["base_rate"] = base
    out["rewrite_gt_60min_net_of_base_rate"] = sum(rw - min(rw, rate * c) for rw, c in gt60)
    return out


# ---------------------------------------------------------------------------------------------
# The result: aggregates only
# ---------------------------------------------------------------------------------------------
def script_sha256() -> str:
    """sha256 of this file with CRLF line ends read as LF (a copy that only changed line ends matches)."""
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _pct(num: float, den: float) -> Optional[float]:
    return round(num / den * 100, 4) if den else None


def _safe_model(name: str) -> str:
    return name if SAFE_MODEL.fullmatch(name) else OTHER


def _safe_version(v: str) -> str:
    return v if SAFE_VERSION.fullmatch(v) else OTHER


def build_result(transcripts: List[Transcript], stats: Stats, a: Dict[str, Any], costs: Dict[str, Any],
                 since: Optional[dt.datetime], until: Optional[dt.datetime]) -> Dict[str, Any]:
    reqs = [r for tr in transcripts for r in tr.requests]
    tot = {k: sum(r.usage[k] for r in reqs) for k in USAGE_KEYS}
    w1h = sum(r.usage["cache_creation"]["ephemeral_1h_input_tokens"] for r in reqs)
    w5m = sum(r.usage["cache_creation"]["ephemeral_5m_input_tokens"] for r in reqs)
    total_input = tot["input_tokens"] + tot["cache_creation_input_tokens"] + tot["cache_read_input_tokens"]
    months = sorted({"%04d-%02d" % (r.t.year, r.t.month) for r in reqs})
    comp = costs["components"]
    comp_total = sum(comp.values())
    wcost = sum(a["write_cost_usd"].values())
    rw = a["rewrite"]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "script_sha256": script_sha256(),
        "client_versions_seen": sorted({_safe_version(r.version) for r in reqs if r.version}),
        "selection": {"since_filter": since is not None, "until_filter": until is not None},
        "counts": {
            "requests": len(reqs),
            "sessions": sum(tr.kind == "main" for tr in transcripts),
            "subagent_runs": sum(tr.kind == "subagent" for tr in transcripts),
            "months_covered": len(months),
            "months": months,
            "model_count": len({r.model for r in reqs}),
            "models": sorted({_safe_model(r.model) for r in reqs}),
        },
        "tokens": {
            "uncached_input": tot["input_tokens"],
            "cache_read": tot["cache_read_input_tokens"],
            "cache_write_5m": w5m,
            "cache_write_1h": w1h,
            "output": tot["output_tokens"],
            "input_total": total_input,
        },
        "input_shares_pct": {
            "cache_read": _pct(tot["cache_read_input_tokens"], total_input),
            "cache_write": _pct(tot["cache_creation_input_tokens"], total_input),
            "uncached": _pct(tot["input_tokens"], total_input),
        },
        "cache_write_1h_share_pct": _pct(w1h, tot["cache_creation_input_tokens"]),
        "list_price_cost_shares_pct": {
            "uncached_input": _pct(comp["fresh"], comp_total),
            "cache_read": _pct(comp["cacheRead"], comp_total),
            "cache_write": _pct(comp["cacheWrite"], comp_total),
            "output": _pct(comp["output"], comp_total),
        },
        "cache_read_largest_cost": (comp["cacheRead"] == max(comp.values())) if comp_total else None,
        "unpriced": {
            "requests": costs["unpriced_requests"],
            "input_share_pct": _pct(costs["unpriced_input_tokens"], total_input),
        },
        "rewrites": {
            "written_tokens": a["write_tokens"],
            "new_suffix_tokens": a["new_suffix"],
            "rewrite_tokens": rw,
            "rewrite_share_of_written_pct": _pct(rw, a["write_tokens"]),
            "rewrite_share_of_write_cost_pct": _pct(a["write_cost_usd"]["rewrite"], wcost),
            "rewrite_share_of_total_cost_pct": _pct(a["write_cost_usd"]["rewrite"], a["total_cost_usd"]),
            "by_gap_pct": {b: _pct(a["rewrite_by_gap"][b], rw) for b in GAP_BINS},
            "by_cause_pct": {c: _pct(a["rewrite_by_cause"][c], rw) for c in CAUSES},
            "expiry_gt_60min_net_of_base_rate_pct": _pct(a["rewrite_gt_60min_net_of_base_rate"], rw),
        },
        "line_factor": {
            "usage_lines": stats.line_requests,
            "requests": len(reqs),
            "lines_per_request": round(stats.line_requests / len(reqs), 4) if reqs else None,
        },
        "checks": {
            "log_files": stats.files,
            "unreadable_files": stats.unreadable_files,
            "synthetic_lines_dropped": stats.synthetic_lines,
            "unparseable_lines": stats.bad_lines + stats.bad_ts_lines,
            "lines_without_id": stats.no_id_lines,
            "cross_file_duplicates": stats.cross_file_duplicates,
            "write_split_mismatch": stats.write_split_mismatch,
        },
    }


def check_result(obj: Any, path: str = "") -> None:
    """Privacy guard: raise ValueError unless every text value is one of the permitted kinds."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str) or not SAFE_KEY.fullmatch(k):
                raise ValueError(f"unexpected key at {path or '/'}")
            check_result(v, f"{path}/{k}")
        return
    if isinstance(obj, list):
        for v in obj:
            check_result(v, path + "[]")
        return
    if obj is None or isinstance(obj, (bool, int, float)):
        return
    if not isinstance(obj, str):
        raise ValueError(f"unexpected value type at {path}")
    allowed = {"/kind": lambda s: s == KIND,
               "/script_sha256": lambda s: bool(SAFE_SHA.fullmatch(s)),
               "/counts/months[]": lambda s: bool(SAFE_MONTH.fullmatch(s)),
               "/counts/models[]": lambda s: s == OTHER or bool(SAFE_MODEL.fullmatch(s)),
               "/client_versions_seen[]": lambda s: s == OTHER or bool(SAFE_VERSION.fullmatch(s))}
    ok = allowed.get(path)
    if ok is None or not ok(obj):
        raise ValueError(f"text not permitted at {path}")


def compute(root: Path, since: Optional[dt.datetime] = None, until: Optional[dt.datetime] = None,
            progress: Optional[Callable[[int, int], None]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(result, internals). The result is what gets written; internals stay in memory."""
    transcripts, stats = extract(root, since, until, progress)
    recs = annotated_records(transcripts)
    a = analyse(recs)
    costs = cost_components(transcripts)
    result = build_result(transcripts, stats, a, costs, since, until)
    check_result(result)
    return result, {"transcripts": transcripts, "stats": stats, "records": recs, "analysis": a, "costs": costs}


def render(result: Dict[str, Any]) -> str:
    return json.dumps(result, indent=2, ensure_ascii=True) + "\n"


# ---------------------------------------------------------------------------------------------
# Self-test on a built-in synthetic log (hand-computed answers)
# ---------------------------------------------------------------------------------------------
def _synthetic_logs(root: Path) -> List[str]:
    """Write a tiny synthetic log tree under root; return the planted sentinel strings."""
    s_prompt, s_tool, s_path = "SENTINELPROMPTqx41", "SENTINELTOOLqx42", "SENTINELPATHqx43"
    s_agent, s_session = "SENTINELAGENTqx44", "SENTINELSESSIONqx45"
    s_email = "sentinel.user" + "@" + "example.invalid"
    s_host = "sentinel-host-qx46.corp.invalid"
    t0 = dt.datetime(2026, 10, 31, 20, 0, 0, tzinfo=dt.timezone.utc)

    def ts(sec: float) -> str:
        return (t0 + dt.timedelta(seconds=sec)).strftime("%Y-%m-%dT%H:%M:%S.") + "%03dZ" % (int(sec * 1000) % 1000)

    def asst(mid: str, sec: float, uuid: str, parent: Optional[str], rd: int, w1: int, w5: int = 0,
             out: int = 50, model: str = "claude-opus-5", ver: str = "2.1.1", **extra: Any) -> str:
        usage = {"input_tokens": 10, "cache_read_input_tokens": rd, "cache_creation_input_tokens": w1 + w5,
                 "output_tokens": out,
                 "cache_creation": {"ephemeral_5m_input_tokens": w5, "ephemeral_1h_input_tokens": w1}}
        row = {"type": "assistant", "uuid": uuid, "parentUuid": parent, "timestamp": ts(sec),
               "sessionId": s_session, "cwd": "/work/" + s_path, "version": ver, "requestId": "req_" + uuid,
               "message": {"id": mid, "model": model, "usage": usage,
                           "content": [{"type": "tool_use", "name": s_tool, "input": {"cmd": s_tool, "host": s_host}}]}}
        row.update(extra)
        return json.dumps(row)

    def user(sec: float, uuid: str, parent: Optional[str], **extra: Any) -> str:
        row = {"type": "user", "uuid": uuid, "parentUuid": parent, "timestamp": ts(sec), "sessionId": s_session,
               "message": {"role": "user", "content": s_prompt + " mail " + s_email + " on " + s_host}}
        row.update(extra)
        return json.dumps(row)

    def att(sec: float, uuid: str, parent: str, typ: str, **kw: Any) -> str:
        return json.dumps({"type": "attachment", "uuid": uuid, "parentUuid": parent, "timestamp": ts(sec),
                           "attachment": {"type": typ, **kw}})

    h = 3600
    main = [
        att(0, "p0", "x", "prompt_snapshot", systemPrompt=[s_prompt]),
        user(0, "u0", "p0"),
        asst("m1", 1, "a1", "u0", rd=0, w1=1000, out=50),
        asst("m1", 1.5, "a1b", "a1", rd=0, w1=1000, out=80),                  # second line, same response
        asst("s0", 2, "z0", "a1b", rd=0, w1=5, model="<synthetic>"),          # no API call: dropped
        user(60, "u1", "a1b"),
        asst("m2", 61, "a2", "u1", rd=1000, w1=200),                          # new suffix
        user(2 * h + 61, "u2", "a2"),
        asst("m3", 2 * h + 62, "a3", "u2", rd=0, w1=1300),                    # re-write 1200 after 2 h: expiry
        '{"type":"system","subtype":"compact_boundary","uuid":"cb","parentUuid":null,"timestamp":"%s"}'
        % ts(2 * h + 600),
        user(2 * h + 610, "u3", "cb", isCompactSummary=True),
        asst("m4", 2 * h + 662, "a4", "u3", rd=0, w1=500),                    # re-write 500: compaction
        asst("m5", 2 * h + 722, "a5", "a4", rd=0, w1=0, w5=400, model="claude-sonnet-5"),  # 400: model change
        asst("m6", 2 * h + 1922, "a6", "a5", rd=100, w1=400, model="claude-sonnet-5"),  # 300 after 20 min, 5m entries
        asst("m7", 2 * h + 1952, "a7", "a6", rd=200, w1=350, model="claude-sonnet-5", ver="2.1.2"),  # 300: version
        asst("m8", 2 * h + 1982, "a8", "a7", rd=450, w1=150, model="claude-sonnet-5", ver="2.1.2"),  # 100: within
    ]
    sub = [
        user(300, "v0", None, attributionAgent=s_agent),
        asst("k1", 301, "b1", "v0", rd=0, w1=300),
        asst("k2", 360, "b2", "b1", rd=300, w1=50),
    ]
    resumed = [
        asst("m1", 1, "c1", None, rd=0, w1=1000, out=50),                     # copy of m1: de-duplicated
        asst("r1", 24 * h, "c2", "c1", rd=0, w1=100, ver=s_email),            # next month; junk version string
    ]
    for rel, lines in ((Path("aaa-" + s_path) / (s_session + ".jsonl"), main),
                       (Path("aaa-" + s_path) / s_session / "subagents" / ("agent-" + s_agent + ".jsonl"), sub),
                       (Path("bbb-" + s_host) / ("resumed-" + s_session + ".jsonl"), resumed)):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [s_prompt, s_tool, s_path, s_agent, s_session, s_email, s_host, "req_a1"]


def self_test() -> int:
    """Run the kit on the synthetic log; compare with hand-computed answers. 0 = pass."""
    failures: List[str] = []

    def check(name: str, got: Any, want: Any) -> None:
        ok = (abs(got - want) < 1e-3) if isinstance(want, float) and isinstance(got, float) else got == want
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {got!r}" + ("" if ok else f" (expected {want!r})"))
        if not ok:
            failures.append(name)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sentinels = _synthetic_logs(root)
        result, _ = compute(root)
        text = render(result)
    c, tk, rwr = result["counts"], result["tokens"], result["rewrites"]
    print("Self-test on a built-in synthetic log:")
    check("requests (two log lines of one response, a synthetic line and a resumed copy collapse)", c["requests"], 11)
    check("sessions", c["sessions"], 2)
    check("subagent runs", c["subagent_runs"], 1)
    check("months covered", c["months"], ["2026-10", "2026-11"])
    check("models (names only)", c["models"], ["claude-opus-5", "claude-sonnet-5"])
    check("client versions (an unrecognised string becomes 'other')", result["client_versions_seen"],
          ["2.1.1", "2.1.2", OTHER])
    check("output tokens (max over a response's lines)", tk["output"], 80 + 50 * 10)
    check("cache-write tokens, 1 h", tk["cache_write_1h"], 4750 - 400)
    check("cache-write tokens, 5 min", tk["cache_write_5m"], 400)
    check("written tokens", rwr["written_tokens"], 4750)
    check("re-written tokens", rwr["rewrite_tokens"], 2800)
    check("re-write share of written tokens (%)", rwr["rewrite_share_of_written_pct"], round(2800 / 4750 * 100, 4))
    want = {"model_change": 400, "compaction": 500, "config_change": 300, "gap_gt_60min": 1200,
            "gap_5_to_60min": 300, "within_lifetime": 100}
    check("re-writes by cause (%)", rwr["by_cause_pct"], {k: round(v / 2800 * 100, 4) for k, v in want.items()})
    check("usage lines per request", result["line_factor"]["lines_per_request"], round(13 / 11, 4))
    shares = result["list_price_cost_shares_pct"]
    check("cost shares sum to 100", round(sum(shares.values()), 2), 100.0)
    leaked = [s for s in sentinels if s in text]
    check("planted prompts, tool input, paths, ids, email and host name absent from the result", leaked, [])
    try:
        check_result(json.loads(text))
        guard = True
    except ValueError:
        guard = False
    check("privacy guard accepts the result", guard, True)
    print("SELF-TEST " + ("PASSED" if not failures else f"FAILED ({len(failures)} checks)"))
    return 0 if not failures else 1


# ---------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------
def _date(s: str) -> dt.datetime:
    try:
        return dt.datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a date as YYYY-MM-DD, got {s!r}")


def default_root() -> Path:
    cfg = os.environ.get("CLAUDE_CONFIG_DIR")
    if cfg and (Path(cfg).expanduser() / "projects").is_dir():
        return Path(cfg).expanduser() / "projects"
    return Path.home() / ".claude" / "projects"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Aggregate token and prompt-cache figures from your own Claude Code logs "
                    "(aggregates only; nothing is sent anywhere).")
    ap.add_argument("--root", type=Path, default=None,
                    help="folder holding the logs (default: ~/.claude/projects)")
    ap.add_argument("--since", type=_date, default=None, help="only requests at or after 00:00 UTC on this date")
    ap.add_argument("--until", type=_date, default=None, help="only requests before 00:00 UTC on this date")
    ap.add_argument("--out", type=Path, default=Path(RESULT_NAME), help=f"result file (default: {RESULT_NAME})")
    ap.add_argument("--self-test", action="store_true", help="check this script on a built-in synthetic log")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.since and args.until and args.since >= args.until:
        print("--since must be earlier than --until.")
        return 2
    root = (args.root or default_root()).expanduser()
    if not root.is_dir():
        print(f"No Claude Code logs found: the folder {root} does not exist.\n"
              "If your logs are somewhere else, run:  python tokens_contrib.py --root <folder with your logs>")
        return 1
    if next(root.glob("**/*.jsonl"), None) is None:
        print(f"No Claude Code logs found: there are no .jsonl files under {root}.\n"
              "If your logs are somewhere else, run:  python tokens_contrib.py --root <folder with your logs>")
        return 1

    def progress(i: int, n: int) -> None:
        if i == 0:
            print(f"Reading {n:,} log files (this can take a minute or two)...", file=sys.stderr)
        elif n >= 20 and i % max(1, n // 10) == 0:
            print(f"  {i * 100 // n}%", file=sys.stderr)

    try:
        result, _ = compute(root, args.since, args.until, progress)
    except Exception as exc:  # noqa: BLE001 - report the type only; its text could quote log content
        print(f"The script stopped on an unexpected problem ({type(exc).__name__}). Nothing was written.\n"
              "Please tell the person who sent it this one line; do not send your logs.")
        return 1
    if not result["counts"]["requests"]:
        print("The log files contain no API requests with token usage"
              + (" in the dates you chose." if (args.since or args.until) else ".") + " Nothing to report.")
        return 1
    text = render(result)
    print(f"\n===== {RESULT_NAME}: exactly what will be written =====")
    print(text, end="")
    print("===== end =====\n")
    try:
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    except OSError as exc:
        print(f"Could not write {args.out}: {exc}\nRun it again with  --out <a file you can write>")
        return 1
    print(f"Wrote {args.out.resolve()}")
    print("Please read the numbers above before sending anything. They are counts, token totals,\n"
          "percentages, months (YYYY-MM), Claude model names and Claude Code version numbers only.\n"
          f"If you are happy to share them, send back only the file {RESULT_NAME}.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped.")
        raise SystemExit(130)
