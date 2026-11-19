#!/usr/bin/env python3
"""READ helper (local workstation only): Claude Code transcripts -> one usage record per API request.

Claude Code writes one assistant response as several transcript lines (one per content
block), each repeating the same message id and the same input usage. Counting lines
therefore overcounts requests. This module keeps one record per message id:

  * input fields from the id's first line (they are identical across repeats);
  * output_tokens = the maximum over the id's lines (streamed repeats can be partial);
  * model "<synthetic>" (client-side placeholders, no API call) is dropped;
  * requests at or after the freeze time are dropped;
  * transcripts are ordered by first request time (ties: path order) and a message id
    already attributed to an earlier transcript is skipped (resumed copies).

Lines end at a newline only. The client writes raw U+2028, U+2029 and U+0085 inside JSON strings
(legal JSON); str.splitlines() also breaks at those, which shattered such lines into
unparseable fragments and silently dropped their requests (fixed 16 Nov 2026; the two
counters separator_lines and separator_only_requests record what that reader lost).

Chain mode (extract(..., chain=True), used by tools/cache_writes.py) also reads every line's uuid
and parentUuid, and the client's compaction markers (a system line with subtype
"compact_boundary", or a message flagged isCompactSummary), without changing the records:
annotations() then says, per request, whether a compaction marker lies between the previous kept
request and this one, and whether this request's message chain (parentUuid links) reaches the
previous kept request's response before any other request -- an unbroken message chain (which
cannot see content edited in place, such as microcompact; the transcripts carry no microcompact
marker). Chain mode also records changes outside the messages (rev3, 17 Nov 2026): a tool, agent
or MCP-instructions list delta (attachment types in LIST_DELTAS), a changed system prompt (the
client's prompt_snapshot attachment differing from the transcript's previous snapshot) and a
changed client version between two requests (a restart, which can change the system prompt and
tool definitions).

The output is a JSONL log in tokenledger's "anthropic" format (usage under "usage"), with
"turn" = transcript index and "step" = request index within it, so tokenledger-llm reads it
directly. Records hold token counts, model id, timestamp and agent role only -- never
prompts, paths or session ids -- and are written under evidence/raw/ (gitignored).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

DEFAULT_PROJECTS = Path(os.environ.get("CLAUDE_PROJECTS", Path.home() / ".claude" / "projects"))
USAGE_KEYS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")
# Characters str.splitlines() treats as line ends that may appear raw inside a JSON string.
UNICODE_LINE_SEPARATORS = ("\u0085", "\u2028", "\u2029")
# Attachments that record a change to the tool, agent or MCP-instructions list (chain mode).
LIST_DELTAS = ("deferred_tools_delta", "agent_listing_delta", "mcp_instructions_delta")


def parse_ts(ts: str) -> dt.datetime:
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))


@dataclass
class Request:
    msg_id: str
    ts: str
    model: str
    usage: dict[str, Any]
    order: int  # line position in its transcript
    separator_only: bool = False  # every line of this id carries a Unicode line separator
    parent_id: str | None = None  # chain mode: nearest earlier request on this request's message chain
    version: str | None = None    # chain mode: client version on the request's first line


@dataclass
class Transcript:
    order: int            # path order (tie-breaker)
    kind: str             # "main" | "subagent"
    role: str             # "main", an agent type, or "subagent(unattributed)"
    requests: list[Request] = field(default_factory=list)
    compactions: list[int] = field(default_factory=list)  # chain mode: line positions of compaction markers
    deltas: list[int] = field(default_factory=list)       # chain mode: line positions of list deltas
    prompts: list[tuple[int, str]] = field(default_factory=list)  # chain mode: (line, system-prompt digest)
    path: Path | None = None  # source file, kept in memory only (session clustering); never written out

    @property
    def first_ts(self) -> str:
        return self.requests[0].ts


@dataclass
class Stats:
    files: int = 0
    usage_lines: int = 0
    synthetic_lines: int = 0
    bad_lines: int = 0
    no_id_lines: int = 0
    repeated_lines: int = 0
    after_freeze: int = 0
    cross_file_duplicates: int = 0
    write_split_mismatch: int = 0
    # Usage lines carrying a raw Unicode line separator, and the kept requests whose every
    # line carries one: what a str.splitlines() reader would have failed to parse or lost.
    separator_lines: int = 0
    separator_only_requests: int = 0
    # Exclusion E1 (no request before the freeze), by cause.
    files_without_requests: int = 0
    files_only_after_freeze: int = 0
    files_emptied_by_dedup: int = 0
    # Line-level replication of a reader that counts every usage line as a request
    # (no de-duplication), restricted to lines before the freeze, for reconciliation.
    line_requests: int = 0
    line_input: int = 0
    line_cache_read: int = 0


def _count(v: Any) -> int:
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 else 0


def _norm_usage(u: dict[str, Any], stats: Stats) -> dict[str, Any]:
    out = {k: _count(u.get(k)) for k in USAGE_KEYS}
    cc = u.get("cache_creation") if isinstance(u.get("cache_creation"), dict) else {}
    w5, w1 = _count(cc.get("ephemeral_5m_input_tokens")), _count(cc.get("ephemeral_1h_input_tokens"))
    if w5 + w1 != out["cache_creation_input_tokens"]:
        # Pre-registered rule: price any unexplained remainder as 5-minute writes.
        stats.write_split_mismatch += 1
        w5 = max(out["cache_creation_input_tokens"] - w1, 0)
        w1 = out["cache_creation_input_tokens"] - w5
    out["cache_creation"] = {"ephemeral_5m_input_tokens": w5, "ephemeral_1h_input_tokens": w1}
    return out


def _is_compaction(ev: dict[str, Any]) -> bool:
    return (ev.get("type") == "system" and ev.get("subtype") == "compact_boundary") or ev.get("isCompactSummary") is True


def _nearest_request(start: Any, mid: str, parent: dict[str, Any], line_msg: dict[str, str]) -> str | None:
    """Walk parentUuid links from `start` to the first line that belongs to another request."""
    seen: set[str] = set()
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


def read_transcript(path: Path, order: int, stats: Stats, cut: dt.datetime | None = None,
                    chain: bool = False) -> Transcript:
    kind = "subagent" if "subagents" in [p.lower() for p in path.parts] else "main"
    role = "main" if kind == "main" else "subagent(unattributed)"
    by_id: dict[str, Request] = {}
    parent: dict[str, Any] = {}          # chain mode: line uuid -> parentUuid
    line_msg: dict[str, str] = {}        # chain mode: line uuid -> message id of a request line
    first_parent: dict[str, Any] = {}    # chain mode: message id -> parentUuid of its first line
    compactions: list[int] = []
    deltas: list[int] = []
    prompts: list[tuple[int, str]] = []
    versions: dict[str, Any] = {}        # chain mode: message id -> client version of its first line
    text = path.read_text(encoding="utf-8", errors="replace")
    for n, line in enumerate(text.split("\n")):  # never splitlines(): see the module docstring
        if kind == "subagent" and role == "subagent(unattributed)" and '"attributionAgent"' in line:
            try:
                found = json.loads(line).get("attributionAgent")
            except (json.JSONDecodeError, AttributeError):
                found = None
            if isinstance(found, str) and found:
                role = found
        ev: Any = None
        if chain and '"uuid"' in line:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
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
                        digest = json.dumps(att.get("systemPrompt"), sort_keys=True, ensure_ascii=False)
                        prompts.append((n, hashlib.sha256(digest.encode("utf-8")).hexdigest()))
        if '"usage"' not in line:
            continue
        if ev is None:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
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
        if chain and isinstance(ev.get("uuid"), str):
            line_msg[ev["uuid"]] = mid
        if cut is None or parse_ts(ts) < cut:
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
        by_id[mid] = Request(mid, ts, str(msg.get("model") or "?"), _norm_usage(msg["usage"], stats), n, has_sep)
    if chain:
        for mid, r in by_id.items():
            r.parent_id = _nearest_request(first_parent.get(mid), mid, parent, line_msg)
            v = versions.get(mid)
            r.version = v if isinstance(v, str) and v else None
    t = Transcript(order, kind, role, compactions=compactions, deltas=deltas, prompts=prompts, path=path)
    t.requests = sorted(by_id.values(), key=lambda r: (parse_ts(r.ts), r.order))
    return t


def extract(projects: Path = DEFAULT_PROJECTS, freeze: str | None = None,
            chain: bool = False) -> tuple[list[Transcript], Stats]:
    """Every transcript with at least one request before the freeze, deduplicated."""
    stats = Stats()
    files = sorted(projects.glob("**/*.jsonl"))
    stats.files = len(files)
    cut = parse_ts(freeze) if freeze else None
    ts_list = []
    for i, path in enumerate(files):
        t = read_transcript(path, i, stats, cut, chain)
        if not t.requests:
            stats.files_without_requests += 1
            continue
        if cut is not None:
            kept = [r for r in t.requests if parse_ts(r.ts) < cut]
            stats.after_freeze += len(t.requests) - len(kept)
            t.requests = kept
            if not kept:
                stats.files_only_after_freeze += 1
                continue
        ts_list.append(t)
    ts_list.sort(key=lambda t: (parse_ts(t.first_ts), t.order))
    seen: set[str] = set()
    out = []
    for t in ts_list:
        kept = [r for r in t.requests if r.msg_id not in seen]
        stats.cross_file_duplicates += len(t.requests) - len(kept)
        seen.update(r.msg_id for r in kept)
        t.requests = kept
        if t.requests:
            out.append(t)
        else:
            stats.files_emptied_by_dedup += 1
    stats.separator_only_requests = sum(r.separator_only for t in out for r in t.requests)
    return out, stats


def records(transcripts: Iterable[Transcript]) -> Iterable[dict[str, Any]]:
    """tokenledger 'anthropic'-format records: turn = transcript index, step = request index."""
    for ti, t in enumerate(transcripts):
        for si, r in enumerate(t.requests):
            yield {"turn": ti, "step": si, "label": t.role, "kind": t.kind, "model": r.model,
                   "ts": r.ts, "usage": r.usage}


def annotations(transcripts: Iterable[Transcript]) -> Iterable[dict[str, Any]]:
    """Chain-mode annotations, one per record and in records() order (same turn and step).

    after_compact: a compaction marker lies between the previous kept request's first line and
    this request's first line. chain_from_prev: this request's message chain reaches the
    previous kept request's response before any other request (None for a transcript's first
    request). after_list_delta: a tool, agent or MCP-instructions list delta lies between them.
    prompt_changed: the latest system-prompt snapshot before this request differs from the latest
    one before the previous request (both must exist). version_changed: the two requests' client
    versions differ (both recorded). All need extract(..., chain=True)."""
    for ti, t in enumerate(transcripts):
        for si, r in enumerate(t.requests):
            if si == 0:
                yield {"turn": ti, "step": si, "after_compact": False, "chain_from_prev": None,
                       "after_list_delta": False, "prompt_changed": False, "version_changed": False}
                continue
            prev = t.requests[si - 1]
            lo, hi = sorted((prev.order, r.order))
            p_lo, p_hi = _prompt_at(t.prompts, lo), _prompt_at(t.prompts, hi)
            yield {"turn": ti, "step": si, "after_compact": any(lo < b < hi for b in t.compactions),
                   "chain_from_prev": r.parent_id == prev.msg_id,
                   "after_list_delta": any(lo < b < hi for b in t.deltas),
                   "prompt_changed": p_lo is not None and p_hi is not None and p_lo != p_hi,
                   "version_changed": bool(prev.version and r.version and prev.version != r.version)}


def _prompt_at(prompts: list[tuple[int, str]], line: int) -> str | None:
    """Digest of the latest system-prompt snapshot before `line` (None if there is none)."""
    found = None
    for n, digest in prompts:
        if n >= line:
            break
        found = digest
    return found


def write_log(transcripts: list[Transcript], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records(transcripts):
            f.write(json.dumps(rec, separators=(",", ":"), sort_keys=True) + "\n")
