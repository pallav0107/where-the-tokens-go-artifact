#!/usr/bin/env python3
"""Cache writes: new suffix or re-write, and do one-hour writes pay back?

Reads the de-duplicated per-request log that tools/claude_usage.py wrote for the orchestrator
snapshot (evidence/raw/claude_requests.jsonl; its sha256 must equal the one recorded in
evidence/orchestrator_snapshot.json) and the list prices in evidence/params/.

Positional model, per transcript (the provider caches prefixes, so a request reads [0, read),
writes [read, read + write) and sends its few uncached tokens after that):

  * After request t-1, positions [0, C) are cached, C = its read + its write; request t-1 read or
    wrote all of them, which refreshed every entry's lifetime at t-1's time.
  * Request t writes [r, r + w). The part below C was cached before, so it is a RE-WRITE:
    re-write = max(0, min(C, r + w) - r); the rest is NEW SUFFIX. A re-write after a gap of at most
    five minutes cannot be expiry (invalidation or eviction); after more than an hour it is expiry
    of any entry; in between it is expiry of five-minute entries or invalidation.
  * Every write is a block [r, r + w). A later request of the same transcript reading [0, r') reads
    min(r', end) - start of each live block; a re-write kills the live blocks it overlaps (they no
    longer collect reads). Reads per written token = read tokens / block size.
  * Writing a block at w times base input instead of sending it uncached pays back once it is
    re-read at least (w - 1) / (1 - rho) times, rho = hit price / base input (Section III).
  * Counterfactual five-minute writes for the one-hour blocks: same requests, same times; a use
    more than five minutes after the block's previous use re-writes at the five-minute price.

Splits by TTL are proportional to each request's five-minute and one-hour write tokens.

A resumed session's transcript starts with nothing cached, so its first full write of the resumed
prefix counts as a NEW-SUFFIX write: re-writes are a lower bound.

Re-writes by observed cause (not pre-registered; added 17 Nov 2026, revised in rev3). Each re-write
is assigned, in this order, to: MODEL CHANGE (its model id differs from the transcript's previous
request), COMPACTION (a compaction marker of the client lies between the previous request and this
one), CONFIG CHANGE (rev3: a tool, agent or MCP-instructions list delta, a changed system-prompt
snapshot or a changed client version between them), or else by the idle gap since the previous
request and the TTL tier of the live entries covering the re-written positions (rev3; Block.f1,
positions no block of the transcript covers take the re-writing request's tier): more than an hour
is EXPIRY of any entry; five to sixty minutes is expiry of the five-minute share and WITHIN
LIFETIME (not expiry) for the one-hour share; at most five minutes is within lifetime (eviction,
or a change the markers do not record, such as microcompact, which leaves no marker). The
pre-rev3 rule (no markers, every entry five-minute) is kept in rewrite_by_cause_pre_rev3.
Unbroken message chain: whether the request's parentUuid chain reaches the previous request's
response before any other request (tools/claude_usage.py chain mode); counted per cause. It cannot
see content edited in place, so it is not prefix identity.

Base rate (rev3, not pre-registered): within-lifetime re-written tokens per cached token per
request, over one-hour sessions (at least 99% of written tokens one-hour) and requests with no
model change, compaction or config change. Had the cache been kept alive, a returning request would
still have re-written rate x C tokens (C = the cached prefix); the expiry share and the keepalive
figures are reported net of that.
The cause annotations come from re-reading the transcripts; the re-extracted log must hash to the
snapshot's tokenledger_log, so the annotations align with the measured requests one to one.

Keepalive (not pre-registered). A ping is a request that re-reads a transcript's whole cached
prefix C at the cache-hit price, refreshing its one-hour lifetime. Bridging an idle gap g with the
cache alive takes ceil(g / 1 h) - 1 hourly pings. For each expiry re-write (gap of more than an
hour, no model change, compaction or config change) whose message chain is unbroken, the avoidable
part (the re-write minus the base rate's rate x C) is either re-written as it was or, with pings,
read at the hit price by the returning request; the pings' cost plus that read is set against the
avoidable part's re-write cost (hindsight: the return is known). A full-prefix re-write costs
w / rho = 20 pings at list prices (policy_cap): 19 pings plus the returning read, so the pings
break even at w / rho - 1 = 19 (Khailo's I_max = tau (w / r - 1) at tau = one hour). The capped
policy pings main-loop sessions only, at most that many times per idle period, and also pays for
those pings after each session's last request; it avoids the avoidable part only when the gap is
bridged. Pings' own few uncached and output tokens are ignored.

Writes evidence/cache_writes.json (aggregates only).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
import math
import statistics
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import claude_usage  # noqa: E402
from evidence_lib import EVIDENCE, figure, sha256_bytes, sha256_text, utc_now, write_json  # noqa: E402

GENERATOR = "tools/cache_writes.py"
SNAPSHOT = EVIDENCE / "orchestrator_snapshot.json"
PRICES = EVIDENCE / "params" / "anthropic_list_prices.json"
FIVE_MIN, ONE_HOUR = 300.0, 3600.0
TTLS = ("5m", "1h")
FREEZE = "2026-11-15T00:00:00Z"  # same freeze as tools/pull_evidence.py
CAUSES = ("model_change", "compaction", "config_change", "gap_gt_60min", "gap_5_to_60min", "within_lifetime")
CHAIN_KEY = {True: "intact", False: "broken", None: "unknown"}
ONE_HOUR_SESSION = 0.99   # a transcript whose written tokens are at least this share one-hour (base rate)


def parse_ts(ts: str) -> dt.datetime:
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))


def price_for(model: str, models: dict[str, dict[str, float]]) -> dict[str, float] | None:
    return models.get(model) or models.get(re.sub(r"-\d{8}$", "", model))


@dataclass
class Block:
    start: int
    end: int
    f1: float                 # share of the block written at the one-hour rate
    p: dict[str, float]       # list prices of the request's model, USD per million
    last_use: dt.datetime
    read_tokens: float = 0.0
    cf5_read_cost: float = 0.0   # five-minute counterfactual cost of the uses after the write, USD
    events: int = field(default=0)

    @property
    def size(self) -> int:
        return self.end - self.start


def _gap_bin(gap: float | None) -> str:
    if gap is None or gap <= FIVE_MIN:
        return "le_5min"
    return "5_to_60min" if gap <= ONE_HOUR else "gt_60min"


def config_changed(r: dict[str, Any], version: bool = True) -> bool:
    """A change outside the messages: list delta, changed system-prompt snapshot or client version.

    version=False drops the client-version marker (the cause rule's sensitivity analysis)."""
    return bool(r.get("after_list_delta") or r.get("prompt_changed") or (version and r.get("version_changed")))


def rewrite_causes(model_changed: bool, after_compact: bool, config_change: bool, gap: float | None,
                   share1: float) -> dict[str, float]:
    """Observed cause of a re-write, as shares summing to one.

    Model change, then compaction, then a list/prompt/version change; otherwise by idle gap and the
    TTL tier of the live entries covering the re-written positions (share1 = their one-hour share):
    over an hour every entry has expired; between five and sixty minutes only five-minute entries
    have, so the one-hour share is re-written within its lifetime (not expiry); within five
    minutes nothing has expired (eviction, or a change the markers do not record)."""
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


def pings_to_bridge(gap: float) -> int:
    """Hourly pings that keep a one-hour entry alive across an idle gap of `gap` seconds."""
    return max(0, math.ceil(gap / ONE_HOUR) - 1)


def policy_cap(models: dict[str, dict[str, float]]) -> tuple[float, int]:
    """Reference break-even in pings and the policy's cap.

    A full-prefix one-hour re-write costs w / rho prefix reads (w = one-hour write multiplier, rho
    = the largest hit multiplier): 2.0 / 0.1 = 20 at list prices. With a keepalive the returning
    request still reads the prefix once, so the pings themselves break even at w / rho - 1 = 19:
    19 pings plus the returning read cost as much as the re-write."""
    writes_1h = {m["cache_write_1h"] / m["input"] for m in models.values()}
    if len(writes_1h) != 1:
        sys.exit("BLOCKED: the one-hour write multiplier is not uniform across models")
    ref = writes_1h.pop() / max(m["cache_hit"] / m["input"] for m in models.values())
    return ref, int(round(ref)) - 1


def _weighted_median(pairs: list[tuple[float, float]]) -> float | None:
    tot = sum(w for _, w in pairs)
    if not tot:
        return None
    acc = 0.0
    for v, w in sorted(pairs):
        acc += w
        if acc >= tot / 2:
            return v
    return None


def session_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Main-loop sessions: wall-clock duration, requests and idle gaps of over an hour (medians)."""
    by_turn: dict[Any, list[dt.datetime]] = {}
    for r in records:
        if r.get("kind") == "main":
            by_turn.setdefault(r["turn"], []).append(parse_ts(r["ts"]))
    hours, reqs, gaps = [], [], []
    for ts in by_turn.values():
        ts.sort()
        hours.append((ts[-1] - ts[0]).total_seconds() / ONE_HOUR)
        reqs.append(len(ts))
        gaps.append(sum((b - a).total_seconds() > ONE_HOUR for a, b in zip(ts, ts[1:])))
    if not hours:
        return {"sessions": 0}
    return {"sessions": len(hours), "median_hours": statistics.median(hours),
            "median_requests": statistics.median(reqs), "median_long_gaps": statistics.median(gaps),
            "long_gaps_total": sum(gaps), "max_hours": max(hours), "min_hours": min(hours)}


def analyse(records: list[dict[str, Any]], prices: dict[str, dict[str, float]],
            keepalive_pings: int = 19, version_is_config: bool = True,
            trace: dict[str, list[Any]] | None = None) -> dict[str, Any]:
    """The positional analysis (module docstring). Defaults reproduce the reported figures.

    version_is_config=False stops counting a client-version change as a config change (sensitivity).
    trace, if given (new_trace()), is filled with per-request rows ("requests"), the idle-gap events
    ("events") and the capped policy's pings after each main-loop session's last request ("tails"), so
    a resampling analysis (tools/share_robustness.py) can re-aggregate them without re-implementing."""
    by_turn: dict[Any, list[dict[str, Any]]] = {}
    for r in records:
        by_turn.setdefault(r["turn"], []).append(r)
    out = {"write_tokens": 0, "new_suffix": 0, "rewrite": 0,
           "rewrite_by_gap": {"le_5min": 0, "5_to_60min": 0, "gt_60min": 0},
           "by_ttl": {t: {"new_suffix": 0.0, "rewrite": 0.0} for t in TTLS},
           "write_cost_usd": {"new_suffix": 0.0, "rewrite": 0.0},
           "write_cost_by_ttl_usd": {t: 0.0 for t in TTLS},
           "rewrite_cost_by_gap_usd": {"le_5min": 0.0, "5_to_60min": 0.0, "gt_60min": 0.0},
           "rewrite_by_cause": {c: 0 for c in CAUSES},
           "rewrite_cost_by_cause_usd": {c: 0.0 for c in CAUSES},
           "rewrite_chain_by_cause": {c: {"intact": 0, "broken": 0, "unknown": 0} for c in CAUSES},
           "rewrite_shrunk_by_cause": {c: 0 for c in CAUSES},
           "config_marker_rewrite": {"after_list_delta": 0, "prompt_changed": 0, "version_changed": 0},
           "rewrite_by_cause_pre_rev3": {"model_change": 0, "compaction": 0, "gap_gt_60min": 0,
                                         "gap_5_to_60min": 0, "within_lifetime": 0},
           "annotated_requests": 0,
           "total_cost_usd": 0.0, "unpriced_requests": 0}
    ka = new_keepalive(keepalive_pings)
    base = {"rewrite_tokens": 0.0, "exposure_tokens": 0, "requests": 0, "sessions": 0}
    events: list[dict[str, Any]] = []     # every gap of over an hour, settled once the base rate is known
    gt60: list[tuple[float, int]] = []    # (re-written tokens, cached prefix) of expiry re-writes
    done: list[Block] = []
    for reqs in by_turn.values():
        reqs.sort(key=lambda r: r["step"])
        cached, prev_t = 0, None
        prev: dict[str, Any] | None = None
        prev_hit = 0.0
        main = reqs[0].get("kind") == "main"
        wr_all = sum(r["usage"]["cache_creation_input_tokens"] for r in reqs)
        w1_all = sum(r["usage"]["cache_creation"]["ephemeral_1h_input_tokens"] for r in reqs)
        one_hour_session = bool(wr_all) and w1_all >= ONE_HOUR_SESSION * wr_all
        base["sessions"] += one_hour_session
        live: list[Block] = []
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
            chain = r.get("chain_from_prev")
            out["annotated_requests"] += "chain_from_prev" in r
            for ttl, f in (("1h", f1), ("5m", 1 - f1)):
                out["by_ttl"][ttl]["new_suffix"] += new * f
                out["by_ttl"][ttl]["rewrite"] += rewrite * f
            unit = 0.0
            if p is None:
                out["unpriced_requests"] += 1
            else:
                unit = (f1 * p["cache_write_1h"] + (1 - f1) * p["cache_write_5m"]) / 1e6
                out["write_cost_usd"]["new_suffix"] += new * unit
                out["write_cost_usd"]["rewrite"] += rewrite * unit
                out["write_cost_by_ttl_usd"]["1h"] += wr * f1 * p["cache_write_1h"] / 1e6
                out["write_cost_by_ttl_usd"]["5m"] += wr * (1 - f1) * p["cache_write_5m"] / 1e6
                out["total_cost_usd"] += (wr * unit + (u["input_tokens"] * p["input"] + rd * p["cache_hit"]
                                                       + u["output_tokens"] * p["output"]) / 1e6)
            # TTL tier of the live entries covering the re-written positions [rd, rd + rewrite); positions
            # no block of this transcript covers (written by another transcript) take this request's tier
            share1 = f1
            if rewrite:
                cov1 = cov5 = 0.0
                for b in live:
                    ov = max(0, min(rd + rewrite, b.end) - max(rd, b.start))
                    cov1 += ov * b.f1
                    cov5 += ov * (1 - b.f1)
                share1 = (cov1 + (rewrite - cov1 - cov5) * f1) / rewrite
            causes: dict[str, float] = {}
            model_changed = after_compact = config = False
            if prev is not None:
                model_changed = r["model"] != prev["model"]
                after_compact = bool(r.get("after_compact"))
                config = config_changed(r, version_is_config)
                causes = rewrite_causes(model_changed, after_compact, config, gap, share1)
            expiry_unchanged = causes.get("gap_gt_60min") == 1.0 and chain is not False
            base_rw, base_exp = 0.0, 0
            if prev is not None and gap is not None and gap <= ONE_HOUR and one_hour_session \
                    and not (model_changed or after_compact or config):
                base_rw, base_exp = rewrite * causes.get("within_lifetime", 0.0), cached
                base["rewrite_tokens"] += base_rw
                base["exposure_tokens"] += base_exp
                base["requests"] += 1
            if prev is not None and gap is not None and gap > ONE_HOUR:
                events.append({"main": main, "n": pings_to_bridge(gap), "cached": cached, "prev_hit": prev_hit,
                               "rewrite": rewrite if (expiry_unchanged and p is not None) else 0,
                               "unit": unit, "hit": p["cache_hit"] / 1e6 if p is not None else 0.0,
                               "gap_h": gap / ONE_HOUR, "turn": r["turn"]})
            if trace is not None:
                cost = 0.0 if p is None else (wr * unit + (u["input_tokens"] * p["input"] + rd * p["cache_hit"]
                                                           + u["output_tokens"] * p["output"]) / 1e6)
                trace["requests"].append({
                    "turn": r["turn"], "kind": r.get("kind"), "ts": r["ts"], "model": r["model"],
                    "fresh": u["input_tokens"], "read": rd, "write": wr, "rewrite": rewrite,
                    "write_cost": wr * unit, "rewrite_cost": rewrite * unit, "cost": cost,
                    "causes": dict(causes) if rewrite else {}, "base_rw": base_rw, "base_exp": base_exp})
            if rewrite:
                out["rewrite_by_gap"][_gap_bin(gap)] += rewrite
                out["rewrite_cost_by_gap_usd"][_gap_bin(gap)] += rewrite * unit
                shrunk = rd + wr + u["input_tokens"] < cached     # content removed from the prefix
                for c, share in causes.items():
                    out["rewrite_by_cause"][c] += rewrite * share
                    out["rewrite_cost_by_cause_usd"][c] += rewrite * unit * share
                    out["rewrite_chain_by_cause"][c][CHAIN_KEY[chain]] += rewrite * share
                    out["rewrite_shrunk_by_cause"][c] += rewrite * share if shrunk else 0
                if config:
                    for k in out["config_marker_rewrite"]:
                        out["config_marker_rewrite"][k] += rewrite if r.get(k) else 0
                if prev is not None:
                    for c, share in rewrite_causes(model_changed, after_compact, False, gap, 0.0).items():
                        out["rewrite_by_cause_pre_rev3"][c] += rewrite * share
                if causes.get("gap_gt_60min") == 1.0:
                    gt60.append((rewrite, cached))
                kept = []
                for b in live:              # the cache below `rd` survived; above it was lost
                    if b.end <= rd:
                        kept.append(b)
                    elif b.start >= rd:
                        done.append(b)
                    else:                   # split: the part above rd is dead, same history
                        dead = Block(rd, b.end, b.f1, b.p, b.last_use, b.read_tokens * (b.end - rd) / b.size,
                                     b.cf5_read_cost * (b.end - rd) / b.size, b.events)
                        b.read_tokens -= dead.read_tokens
                        b.cf5_read_cost -= dead.cf5_read_cost
                        b.end = rd
                        done.append(dead)
                        kept.append(b)
                live = kept
            for b in live:
                ov = max(0, min(rd, b.end) - b.start)
                if ov:
                    b.read_tokens += ov
                    b.events += 1
                    since = (t - b.last_use).total_seconds()
                    b.cf5_read_cost += ov * (b.p["cache_write_5m"] if since > FIVE_MIN else b.p["cache_hit"]) / 1e6
                    b.last_use = t
            if wr and p is not None:
                live.append(Block(rd, rd + wr, f1, p, t))
            cached, prev_t, prev = rd + wr, t, r
            prev_hit = p["cache_hit"] if p is not None else 0.0
        if main and keepalive_pings:
            ka["policy_sessions"] += 1
            tail = keepalive_pings * cached * prev_hit / 1e6   # pings after the session's last request
            ka["policy_ping_cost_usd"] += tail
            ka["policy_saving_usd"] -= tail
            if trace is not None:
                trace["tails"].append((reqs[0]["turn"], tail))
        done.extend(live)
    # Base rate of non-expiry invalidations: within-lifetime re-written tokens per cached token per
    # request in one-hour sessions (no model change, compaction or list/prompt/version change). Had
    # the cache been kept alive, a returning request would still have re-written rate x C tokens.
    rate = base["rewrite_tokens"] / base["exposure_tokens"] if base["exposure_tokens"] else 0.0
    base["rate"] = rate
    out["base_rate"] = base
    out["rewrite_gt_60min_net_of_base_rate"] = sum(rw - min(rw, rate * c) for rw, c in gt60)
    if trace is not None:
        trace["events"].extend(events)
    gaps = settle_keepalive(events, rate, keepalive_pings, ka)
    rw = ka["rewrite_tokens"]
    ka["breakeven_pings_weighted"] = ka["breakeven_pings_weighted"] / rw if rw else None
    ka["median_gap_hours_token_weighted"] = _weighted_median(gaps)
    out["keepalive"] = ka
    out["one_hour"] = _payback(done, "1h")
    out["five_minute"] = _payback(done, "5m")
    return out


def new_keepalive(keepalive_pings: int) -> dict[str, Any]:
    """Zeroed keepalive accumulators (analyse, and settle_keepalive on a resample)."""
    return {"rewrite_tokens": 0, "rewrite_cost_usd": 0.0, "ping_cost_usd": 0.0, "cheaper_tokens": 0,
            "oracle_saving_usd": 0.0, "breakeven_pings_weighted": 0.0, "events": 0, "avoidable_tokens": 0.0,
            "policy_pings": keepalive_pings, "policy_saving_usd": 0.0, "policy_ping_cost_usd": 0.0,
            "policy_avoided_rewrite_usd": 0.0, "policy_sessions": 0}


def new_trace() -> dict[str, list[Any]]:
    """An empty trace for analyse(..., trace=...)."""
    return {"requests": [], "events": [], "tails": []}


def settle_keepalive(events: list[dict[str, Any]], rate: float, keepalive_pings: int,
                     ka: dict[str, Any]) -> list[tuple[float, float]]:
    """Settle every idle gap of over an hour against the base rate: the hindsight keepalive and the
    capped policy (module docstring). Adds into `ka`; returns (gap hours, re-written tokens) pairs."""
    gaps: list[tuple[float, float]] = []
    for e in events:
        rw = e["rewrite"]
        avoidable = rw - min(rw, rate * e["cached"])
        pings = e["n"] * e["cached"] * e["prev_hit"] / 1e6
        if e["main"] and keepalive_pings:
            spent = min(e["n"], keepalive_pings) * e["cached"] * e["prev_hit"] / 1e6
            ka["policy_ping_cost_usd"] += spent
            ka["policy_saving_usd"] -= spent
            if e["n"] <= keepalive_pings and rw:
                ka["policy_avoided_rewrite_usd"] += avoidable * (e["unit"] - e["hit"])
                ka["policy_saving_usd"] += avoidable * (e["unit"] - e["hit"])
        if not rw:
            continue
        ping = pings + avoidable * e["hit"]          # the returning request reads the avoidable part
        cost = avoidable * e["unit"]                 # the unavoidable part is re-written either way
        ka["events"] += 1
        ka["rewrite_tokens"] += rw
        ka["avoidable_tokens"] += avoidable
        ka["rewrite_cost_usd"] += rw * e["unit"]
        ka["ping_cost_usd"] += ping
        ka["breakeven_pings_weighted"] += rw * e["unit"] / e["hit"]
        if ping < cost:
            ka["cheaper_tokens"] += rw
            ka["oracle_saving_usd"] += cost - ping
        gaps.append((e["gap_h"], float(rw)))
    return gaps


def _payback(blocks: list[Block], ttl: str) -> dict[str, float]:
    key = "cache_write_1h" if ttl == "1h" else "cache_write_5m"
    tok = reads = never = paid = be_w = actual = uncached = cf5 = 0.0
    for b in blocks:
        f = b.f1 if ttl == "1h" else 1 - b.f1
        if f <= 0 or b.size <= 0:
            continue
        size = b.size * f
        w, rho = b.p[key] / b.p["input"], b.p["cache_hit"] / b.p["input"]
        breakeven = (w - 1) / (1 - rho)
        r = b.read_tokens / b.size
        tok += size
        reads += b.read_tokens * f
        be_w += breakeven * size
        never += size if b.read_tokens == 0 else 0.0
        paid += size if r >= breakeven - 1e-12 else 0.0
        actual += (size * b.p[key] + b.read_tokens * f * b.p["cache_hit"]) / 1e6
        uncached += (size + b.read_tokens * f) * b.p["input"] / 1e6
        cf5 += size * b.p["cache_write_5m"] / 1e6 + b.cf5_read_cost * f
    if not tok:
        return {"tokens": 0.0}
    return {"tokens": tok, "read_tokens": reads, "reads_per_token": reads / tok, "breakeven_reads": be_w / tok,
            "share_never_reread": never / tok, "share_paid_back": paid / tok, "cost_actual_usd": actual,
            "cost_uncached_usd": uncached, "cost_5m_counterfactual_usd": cf5}


def check_log(raw: bytes, snapshot: dict[str, Any]) -> str:
    """The log must be the one the snapshot was measured on."""
    want = next(s["sha256"] for s in snapshot["provenance"]["sources"] if s["id"] == "tokenledger_log")
    got = sha256_bytes(raw)
    if got != want:
        sys.exit(f"BLOCKED: log sha256 {got[:16]} is not the snapshot's tokenledger_log {want[:16]}")
    return got


def annotated_records(projects: Path, snapshot: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """Re-extract the frozen requests in chain mode; the log must be the snapshot's (same sha256)."""
    _, recs, log_sha = annotated_transcripts(projects, snapshot)
    return recs, log_sha


def annotated_transcripts(projects: Path,
                          snapshot: dict[str, Any]) -> tuple[list[Any], list[dict[str, Any]], str]:
    """annotated_records, also returning the transcripts (a record's turn indexes them)."""
    transcripts, _ = claude_usage.extract(projects, freeze=FREEZE, chain=True)
    recs = list(claude_usage.records(transcripts))
    raw = "".join(json.dumps(r, separators=(",", ":"), sort_keys=True) + "\n" for r in recs).encode("utf-8")
    log_sha = check_log(raw, snapshot)
    for r, ann in zip(recs, claude_usage.annotations(transcripts), strict=True):
        if (r["turn"], r["step"]) != (ann["turn"], ann["step"]):
            sys.exit("BLOCKED: annotations out of step with the records")
        for k in ("after_compact", "chain_from_prev", "after_list_delta", "prompt_changed", "version_changed"):
            r[k] = ann[k]
    return transcripts, recs, log_sha


def input_since(records: list[dict[str, Any]], day: str) -> int:
    """Orchestrator input tokens (all kinds) on or after a UTC date."""
    cut = parse_ts(day[:10] + "T00:00:00Z")
    return sum(r["usage"]["input_tokens"] + r["usage"]["cache_read_input_tokens"]
               + r["usage"]["cache_creation_input_tokens"] for r in records if parse_ts(r["ts"]) >= cut)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--projects", type=Path, default=claude_usage.DEFAULT_PROJECTS)
    ap.add_argument("--log", type=Path, default=None,
                    help="optional: also check this saved log against the snapshot's sha256")
    ap.add_argument("--out", type=Path, default=EVIDENCE / "cache_writes.json")
    args = ap.parse_args()
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    records, log_sha = annotated_records(args.projects, snapshot)
    if args.log is not None:
        check_log(args.log.read_bytes(), snapshot)
    praw = PRICES.read_bytes().replace(b"\r\n", b"\n")
    prices = json.loads(praw.decode("utf-8"))
    ref_pings, policy_pings = policy_cap(prices["models"])   # 20 and 19 at list prices
    a = analyse(records, prices["models"], keepalive_pings=policy_pings)
    a["sessions"] = session_stats(records)
    if a["write_tokens"] != snapshot["aggregates"]["input_tokens"]["cache_write"]:
        sys.exit("BLOCKED: written tokens do not match the snapshot")
    if a["annotated_requests"] != len(records):
        sys.exit("BLOCKED: not every request carries a cause annotation")
    hc_path = EVIDENCE / "harness_costs.json"
    hc_raw = hc_path.read_bytes().replace(b"\r\n", b"\n")
    exec_first = json.loads(hc_raw.decode("utf-8"))["figures"]["ExecFirstDate"]["value"]
    a["input_since_executor_first_turn"] = {"from": exec_first, "tokens": input_since(records, exec_first)}
    h, s = a["one_hour"], a["five_minute"]
    wcost = sum(a["write_cost_usd"].values())
    rw = a["rewrite"] or 1
    ka, cause, ccost, chain = a["keepalive"], a["rewrite_by_cause"], a["rewrite_cost_by_cause_usd"], a["rewrite_chain_by_cause"]
    tot = a["total_cost_usd"]
    exp = chain["gap_gt_60min"]
    within = chain["within_lifetime"]
    base, sess = a["base_rate"], a["sessions"]
    figs = {
        "CwNewSharePct": figure(a["new_suffix"] / a["write_tokens"] * 100, "pct0", "tokenledger_log",
                                "share of cache-write tokens that extend the cached prefix (new suffix)"),
        "CwRewriteSharePct": figure(a["rewrite"] / a["write_tokens"] * 100, "pct0", "tokenledger_log",
                                    "share of cache-write tokens re-writing positions cached before"),
        "CwRewriteCostSharePct": figure(a["write_cost_usd"]["rewrite"] / wcost * 100, "pct0", "list_prices",
                                        "share of list-price cache-write cost spent on re-writes"),
        "CwRewriteTotalCostPct": figure(a["write_cost_usd"]["rewrite"] / a["total_cost_usd"] * 100, "pct0",
                                        "list_prices", "re-writes' share of total list-price cost"),
        "CwExpiryTotalCostPct": figure(a["rewrite_cost_by_gap_usd"]["gt_60min"] / a["total_cost_usd"] * 100, "pct0",
                                       "list_prices", "share of total list-price cost spent re-writing a prefix "
                                       "after an idle gap of more than an hour"),
        "CwRewriteShortGapPct": figure(a["rewrite_by_gap"]["le_5min"] / rw * 100, "pct0", "tokenledger_log",
                                       "re-write tokens after a gap of at most five minutes (invalidation)"),
        "CwRewriteMidGapPct": figure(a["rewrite_by_gap"]["5_to_60min"] / rw * 100, "pct0", "tokenledger_log",
                                     "re-write tokens after a gap of five to sixty minutes"),
        "CwRewriteLongGapPct": figure(a["rewrite_by_gap"]["gt_60min"] / rw * 100, "pct0", "tokenledger_log",
                                      "re-write tokens after a gap of more than an hour (expiry)"),
        "CwLongReads": figure(h["reads_per_token"], "float1", "tokenledger_log",
                              "same-transcript re-reads per one-hour-written token"),
        "CwLongBreakeven": figure(h["breakeven_reads"], "float2", "list_prices",
                                  "re-reads a one-hour write needs to pay back, token-weighted over models"),
        "CwLongNeverPct": figure(h["share_never_reread"] * 100, "pct0", "tokenledger_log",
                                 "share of one-hour-written tokens never re-read in their transcript"),
        "CwLongPaidPct": figure(h["share_paid_back"] * 100, "pct0", "tokenledger_log",
                                "share of one-hour-written tokens re-read at least the break-even number of times"),
        "CwLongSavingPct": figure((1 - h["cost_actual_usd"] / h["cost_uncached_usd"]) * 100, "pct0", "list_prices",
                                  "one-hour blocks: list-price saving of writing and re-reading over sending uncached"),
        "CwLongFiveMinDeltaPct": figure((h["cost_5m_counterfactual_usd"] / h["cost_actual_usd"] - 1) * 100,
                                        "signedpct1", "list_prices",
                                        "one-hour blocks: five-minute counterfactual cost over actual, percent change"),
        "CwShortReads": figure(s["reads_per_token"], "float1", "tokenledger_log",
                               "same-transcript re-reads per five-minute-written token"),
        "CwShortNeverPct": figure(s["share_never_reread"] * 100, "pct0", "tokenledger_log",
                                  "share of five-minute-written tokens never re-read in their transcript"),
        # ---- re-writes by observed cause (not pre-registered)
        "CwCauseModelPct": figure(cause["model_change"] / rw * 100, "pct0", "transcripts",
                                  "re-written tokens whose request changed model from the previous request"),
        "CwCauseCompactPct": figure(cause["compaction"] / rw * 100, "pct0", "transcripts",
                                    "re-written tokens after a compaction marker (and no model change)"),
        "CwCauseConfigPct": figure(cause["config_change"] / rw * 100, "pct0", "transcripts",
                                   "re-written tokens after a tool/agent/MCP-list delta, a changed system-prompt "
                                   "snapshot or a changed client version (no model change or compaction)"),
        "CwCauseLongGapPct": figure(cause["gap_gt_60min"] / rw * 100, "pct0", "transcripts",
                                    "re-written tokens after a gap of more than an hour, no observed other cause (expiry)"),
        "CwCauseLongGapNetPct": figure(a["rewrite_gt_60min_net_of_base_rate"] / rw * 100, "pct0", "transcripts",
                                       "expiry re-written tokens net of the base rate of non-expiry invalidations"),
        "CwCauseMidGapPct": figure(cause["gap_5_to_60min"] / rw * 100, "pct0", "transcripts",
                                   "re-written five-minute entries after a gap of five to sixty minutes (expiry)"),
        "CwCauseWithinPct": figure(cause["within_lifetime"] / rw * 100, "pct0", "transcripts",
                                   "re-written tokens within their entries' lifetime (gap at most five minutes, or "
                                   "one-hour entries within the hour), no observed cause: not expiry"),
        "CwCausePreRevLongGapPct": figure(a["rewrite_by_cause_pre_rev3"]["gap_gt_60min"] / rw * 100, "pct0",
                                          "transcripts", "pre-rev3 rule (no list/prompt/version markers): re-written "
                                          "tokens after a gap of more than an hour"),
        "CwCauseModelCostPct": figure(ccost["model_change"] / tot * 100, "pct1", "list_prices",
                                      "model-change re-writes, share of total list-price cost"),
        "CwCauseCompactCostPct": figure(ccost["compaction"] / tot * 100, "pct1", "list_prices",
                                        "compaction re-writes, share of total list-price cost"),
        "CwCauseConfigCostPct": figure(ccost["config_change"] / tot * 100, "pct1", "list_prices",
                                       "list/prompt/version-change re-writes, share of total list-price cost"),
        "CwCauseLongGapCostPct": figure(ccost["gap_gt_60min"] / tot * 100, "pct1", "list_prices",
                                        "expiry re-writes after more than an hour, share of total list-price cost"),
        "CwCauseMidGapCostPct": figure(ccost["gap_5_to_60min"] / tot * 100, "pct1", "list_prices",
                                       "five-minute-entry expiry re-writes (five to sixty minutes), share of total cost"),
        "CwCauseWithinCostPct": figure(ccost["within_lifetime"] / tot * 100, "pct1", "list_prices",
                                       "within-lifetime re-writes (not expiry), share of total list-price cost"),
        "CwLongGapChainPct": figure(exp["intact"] / (sum(exp.values()) or 1) * 100, "pct0", "transcripts",
                                    "expiry re-writes after more than an hour: share whose message chain is unbroken"),
        "CwWithinChainPct": figure(within["intact"] / (sum(within.values()) or 1) * 100, "pct0", "transcripts",
                                   "within-lifetime re-writes: share whose message chain is unbroken"),
        "CwCompactChainBrokenPct": figure(chain["compaction"]["broken"] / (sum(chain["compaction"].values()) or 1) * 100,
                                          "pct0", "transcripts",
                                          "compaction re-writes whose message chain does not reach the previous request"),
        "CwLongGapShrunkPct": figure(a["rewrite_shrunk_by_cause"]["gap_gt_60min"] / (cause["gap_gt_60min"] or 1) * 100,
                                     "pct0", "transcripts", "expiry re-writes whose prompt was shorter than the "
                                     "previous request's cached prefix (content removed, as microcompact would)"),
        "CwBaseRatePct": figure(base["rate"] * 100, "pct2", "transcripts",
                                "base rate: within-lifetime re-written tokens per cached token per request in "
                                "one-hour sessions (no model, compaction or config change)"),
        "CwBaseRateRequests": figure(base["requests"], "int", "transcripts",
                                     "requests in one-hour sessions the base rate is estimated from"),
        "CwOneHourSessions": figure(base["sessions"], "int", "transcripts",
                                    "transcripts with at least 99% of written tokens at the one-hour rate"),
        # ---- main-loop sessions (descriptive)
        "SessMedianHours": figure(sess["median_hours"], "float1", "tokenledger_log",
                                  "main-loop sessions: median wall-clock duration, first to last request (hours)"),
        "SessMedianDays": figure(sess["median_hours"] / 24, "float1", "tokenledger_log",
                                 "main-loop sessions: median wall-clock duration (days)"),
        "SessMedianRequests": figure(sess["median_requests"], "int", "tokenledger_log",
                                     "main-loop sessions: median requests per session"),
        "SessMedianLongGaps": figure(sess["median_long_gaps"], "int", "tokenledger_log",
                                     "main-loop sessions: median idle gaps of over an hour per session"),
        # ---- keepalive (not pre-registered; hindsight and a capped policy)
        "KaRefPings": figure(ref_pings, "int", "list_prices",
                             "pings (prefix reads) costing as much as one full-prefix one-hour re-write at the "
                             "largest hit multiplier (one-hour write multiplier / hit multiplier)"),
        "KaWeightedPings": figure(ka["breakeven_pings_weighted"], "int", "list_prices",
                                  "the same ratio at each re-write's own model and write-tier prices, weighted by re-written tokens"),
        "KaMedianGapHours": figure(ka["median_gap_hours_token_weighted"], "float1", "transcripts",
                                   "median idle gap (hours) of expiry re-writes of an unchanged prefix, token-weighted"),
        "KaCheaperPct": figure(ka["cheaper_tokens"] / (ka["rewrite_tokens"] or 1) * 100, "pct0", "list_prices",
                               "expiry re-written tokens (unbroken chain) for which hourly pings over the gap would have "
                               "cost less, net of the base rate (hindsight)"),
        "KaOracleSavingPct": figure(ka["oracle_saving_usd"] / tot * 100, "pct1", "list_prices",
                                    "hindsight saving of pinging only where cheaper, share of total list-price cost"),
        "KaPolicyPings": figure(policy_pings, "int", "list_prices",
                                "capped policy: most hourly pings per idle period (w / rho - 1: the pings plus the "
                                "returning read cost one full re-write)"),
        "KaPolicySavingPct": figure(ka["policy_saving_usd"] / tot * 100, "pct1", "list_prices",
                                    "capped keepalive policy on main-loop sessions, net of every ping (also after each "
                                    "session's last request): share of total list-price cost saved"),
        # ---- harness split
        "OrchInputSinceExec": figure(a["input_since_executor_first_turn"]["tokens"], "billions1", "tokenledger_log",
                                     "orchestrator input tokens (all kinds) on or after the executor's first delegated turn"),
    }
    write_json(args.out, {
        "schema_version": 1, "kind": "cache_writes",
        "provenance": {"generated_at": utc_now(), "generator": GENERATOR, "sources": [
            {"id": "tokenledger_log", "grade": "measured", "sha256": log_sha,
             "command": "python tools/cache_writes.py --log <the snapshot's evidence/raw/claude_requests.jsonl>",
             "description": "the per-request usage log the orchestrator snapshot was measured on (same sha256)"},
            {"id": "list_prices", "grade": "model-parameter", "sha256": sha256_bytes(praw),
             "command": "evidence/params/anthropic_list_prices.json",
             "description": "Anthropic list prices per model and token kind (as in the orchestrator snapshot)"},
            {"id": "transcripts", "grade": "measured", "sha256": log_sha,
             "command": "python tools/cache_writes.py (tools/claude_usage.py chain mode)",
             "description": "cause annotations (model id, compaction markers, parentUuid chain) from re-reading the "
                            "transcripts; the re-extracted log hashes to the snapshot's tokenledger_log"},
            {"id": "harness_costs", "grade": "measured", "sha256": sha256_bytes(hc_raw),
             "command": "evidence/harness_costs.json",
             "description": "the executor's first delegated-turn date (figures.ExecFirstDate)"},
        ], "notes": [
            "Not pre-registered (17 Nov 2026; revised 17 Nov 2026 for rev3): re-writes by observed cause, the "
            "unbroken-message-chain check, the TTL-tier rule, list/prompt/version markers, the base rate of "
            "non-expiry invalidations, keepalive (hindsight and a capped policy), main-loop session statistics, and "
            "the orchestrator input since the executor's first turn.",
            "Positional model per transcript; re-reads counted within the writing transcript only, so reads by "
            "other transcripts of a shared prefix are missed (re-reads are a lower bound).",
            "Gap = time since the transcript's previous request, which refreshed every live entry.",
            "Cause order: model change, compaction marker, list/prompt/version change, then gap and the TTL tier "
            "of the covering entries; one-hour entries re-written within the hour are not expiry. Microcompact "
            "leaves no marker; the base rate covers unobserved non-expiry invalidations.",
            "A resumed session's first full write counts as a new-suffix write, so re-writes are a lower bound.",
            "Keepalive assumes the prefix is held at the one-hour rate, nets out the base rate and ignores the "
            "pings' own uncached and output tokens; the hindsight figure knows the user returned, the capped "
            "policy does not.",
            f"Analysis record sha256 {sha256_text(json.dumps(a, sort_keys=True))[:16]}.",
        ]},
        "aggregates": a,
        "figures": figs,
    })
    print(json.dumps({k: a[k] for k in ("write_tokens", "new_suffix", "rewrite", "rewrite_by_gap", "rewrite_by_cause",
                                        "rewrite_by_cause_pre_rev3", "config_marker_rewrite", "base_rate",
                                        "rewrite_gt_60min_net_of_base_rate", "sessions")}))
    print("keepalive", json.dumps(ka))
    print("1h", json.dumps(h))
    print("5m", json.dumps(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
