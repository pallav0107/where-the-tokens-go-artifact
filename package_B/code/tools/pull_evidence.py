#!/usr/bin/env python3
"""READ stage. Runs LOCALLY on a workstation (CI cannot see the transcripts).

Writes aggregates only, with provenance, to evidence/*.json:

  harness_costs.json        <- [REDACTED: unrelated]
                               [REDACTED: unrelated]
  orchestrator_snapshot.json <- tokenledger-llm over the de-duplicated per-request log that
                               tools/claude_usage.py extracts from Claude Code transcripts,
                               list prices from evidence/params/, reconciled against
                               `[REDACTED: infrastructure] report --claude --json`
  heldout.json              <- the pre-registered held-out test (tools/heldout.py) on the same log
  baseline_sample.json      <- [REDACTED: unrelated]
  index_ab.json             <- [REDACTED: unrelated]
  failfirst_incidents.json  <- [REDACTED: unrelated]

Run after it (tools/pipeline.py evidence does): tools/usage_audit.py (transcript counts against
the client's own meter, and the cost-split sensitivity), tools/cache_writes.py (new-suffix
writes versus re-writes, one-hour payback) and experiments/setup_v3.py (the controlled test's
registered settings and price sheet).

Raw command output goes to evidence/raw/ (gitignored, never committed); only its
sha256 is recorded. Nothing written here may contain project names, paths,
hostnames, session ids or prompts -- tools/validate_evidence.py and
tools/privacy_scan.py enforce that in CI.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_lib import (EVIDENCE, REPO, figure, sha256_bytes, sha256_text,  # noqa: E402
                          utc_now, write_json)

HARNESS = Path(os.environ.get("HARNESS_ROOT", Path.home() / ".claude"))
KIMI_EXEC = HARNESS / "[REDACTED: infrastructure]"
RAW = EVIDENCE / "raw"
GENERATOR = "tools/pull_evidence.py"
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0
CI_LOW_Q, CI_HIGH_Q = 0.025, 0.975
TOP_SUBAGENTS = 3
TOKENLEDGER_DIST, TOKENLEDGER_VERSION = "tokenledger-llm", "0.2.0"  # tools/requirements-read.txt
FREEZE = "2026-11-15T00:00:00Z"  # papers/agent2027-cost/PREREG_heldout.md
PRICES = EVIDENCE / "params" / "anthropic_list_prices.json"
PREREG = REPO / "papers" / "agent2027-cost" / "PREREG_heldout.md"


def run_report(args: list[str], raw_name: str) -> tuple[dict[str, Any], str, str]:
    """Run [REDACTED: infrastructure] with the given args; keep raw output local, return (data, sha, cmd)."""
    if not KIMI_EXEC.is_file():
        sys.exit(f"BLOCKED: {KIMI_EXEC} not found -- READ must run on the local workstation")
    env = dict(os.environ, PYTHONUTF8="1")
    proc = subprocess.run([sys.executable, str(KIMI_EXEC), *args], capture_output=True,
                          env=env, timeout=1800, check=False)
    if proc.returncode != 0:
        sys.exit(f"BLOCKED: report exited {proc.returncode}: {proc.stderr[-400:]!r}")
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / raw_name).write_bytes(proc.stdout)
    cmd = "python [REDACTED: infrastructure] " + " ".join(args)
    return json.loads(proc.stdout.decode("utf-8")), sha256_bytes(proc.stdout), cmd


def parse_constants() -> tuple[dict[str, Any], str]:
    """Read the cost-model parameters straight from the tool's source (no import)."""
    src = KIMI_EXEC.read_text(encoding="utf-8")
    out: dict[str, Any] = {}
    m = re.search(r'^PRIMING_TOKENS_PER_REQUEST\s*=.*"(\d+)"', src, re.M)
    if not m:
        sys.exit("BLOCKED: could not find PRIMING_TOKENS_PER_REQUEST default")
    out["PRIMING_TOKENS_PER_REQUEST"] = int(m.group(1))
    m = re.search(r'\["peakContext"\]\s*>\s*([\d_]+)', src)
    if not m:
        sys.exit("BLOCKED: could not find the long-session threshold")
    out["LONG_SESSION_THRESHOLD"] = int(m.group(1).replace("_", ""))
    return out, sha256_text(src)


def ols_loglog(points: list[tuple[int, int]]) -> tuple[float, float]:
    xs = [math.log(s) for s, _ in points]
    ys = [math.log(v) for _, v in points]
    n = len(points)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((y - a - b * x) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return b, 1 - ss_res / ss_tot


def harness_costs() -> None:  # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    sys.exit("[REDACTED: unrelated]: this producer is not part of the release")


# ------------------------------------------------------------ orchestrator + held-out
def tokenledger_version() -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        v = version(TOKENLEDGER_DIST)
    except PackageNotFoundError:
        sys.exit(f"BLOCKED: pip install {TOKENLEDGER_DIST}=={TOKENLEDGER_VERSION} (the paper's instrument)")
    if v != TOKENLEDGER_VERSION:
        sys.exit(f"BLOCKED: {TOKENLEDGER_DIST} {v} installed; the paper states {TOKENLEDGER_VERSION}")
    return v


def run_tokenledger(log: Path) -> tuple[dict[str, Any], str, str]:
    """The published CLI over the whole log: one tokenledger turn per transcript."""
    args = ["report", "--format", "anthropic", "--turn-key", "turn", "--step-key", "step", "--json"]
    proc = subprocess.run([sys.executable, "-m", "tokenledger", *args, str(log)], capture_output=True,
                          env=dict(os.environ, PYTHONUTF8="1"), timeout=1800, check=False)
    if proc.returncode != 0:
        sys.exit(f"BLOCKED: tokenledger exited {proc.returncode}: {proc.stdout[-400:]!r} {proc.stderr[-400:]!r}")
    (RAW / "tokenledger_report.json").write_bytes(proc.stdout)
    cmd = f"tokenledger {' '.join(args)} evidence/raw/claude_requests.jsonl"
    return json.loads(proc.stdout.decode("utf-8")), sha256_bytes(proc.stdout), cmd


def ledger_by_transcript(log: Path) -> dict[int, dict[str, int]]:
    """tokenledger's own accounting, run once per transcript (library API, same version)."""
    from tokenledger.account import account
    from tokenledger.adapters import load_provider
    groups: dict[int, list[Any]] = {}
    for r in load_provider(str(log), "anthropic", turn_key="turn", step_key="step"):
        groups.setdefault(r.turn, []).append(r)
    out = {}
    for turn, reqs in groups.items():
        m = account(reqs)
        first = min(reqs, key=lambda r: r.step)
        out[turn] = {"T": m.n_requests, "C": m.max_prefix, "I": m.total_input, "P": first.input_tokens,
                     "cache_read": m.total_cache_read}
    return out


def load_prices() -> tuple[dict[str, Any], str]:
    raw = PRICES.read_bytes().replace(b"\r\n", b"\n")
    return json.loads(raw.decode("utf-8")), sha256_bytes(raw)


def price_for(model: str, models: dict[str, dict[str, float]]) -> dict[str, float] | None:
    return models.get(model) or models.get(re.sub(r"-\d{8}$", "", model))


def request_cost(usage: dict[str, Any], p: dict[str, float]) -> dict[str, float]:
    """List-price cost of one request by token kind, USD."""
    cc = usage["cache_creation"]
    return {
        "fresh": usage["input_tokens"] * p["input"] / 1e6,
        "cacheWrite": (cc["ephemeral_5m_input_tokens"] * p["cache_write_5m"]
                       + cc["ephemeral_1h_input_tokens"] * p["cache_write_1h"]) / 1e6,
        "cacheRead": usage["cache_read_input_tokens"] * p["cache_hit"] / 1e6,
        "output": usage["output_tokens"] * p["output"] / 1e6,
    }


def prereg_record() -> dict[str, str]:
    """The commit that added the pre-registration, and proof the file is unchanged since."""
    rel = PREREG.relative_to(REPO).as_posix()
    log = subprocess.run(["git", "log", "--diff-filter=A", "--format=%H %cI", "--", rel], cwd=REPO,
                         capture_output=True, text=True, check=False).stdout.split()
    if len(log) < 2:
        sys.exit("BLOCKED: the pre-registration is not committed; commit it before computing results")
    commit, when = log[-2], log[-1]
    at_commit = subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=REPO, capture_output=True,
                               check=True).stdout.replace(b"\r\n", b"\n")
    now = PREREG.read_bytes().replace(b"\r\n", b"\n")
    if at_commit != now:
        sys.exit("BLOCKED: PREREG_heldout.md changed after registration; record a deviation instead")
    return {"commit": commit, "committed_at": when, "sha256": sha256_bytes(at_commit), "path": rel}


def _num(v: float, fmt: str = "float2") -> str:
    """float2 for non-negative values, signedfloat2 when the value is negative."""
    return fmt if v >= 0 else "signedfloat2"


def orchestrator() -> None:
    """Orchestrator snapshot and the pre-registered held-out test, both measured with tokenledger."""
    import claude_usage
    import heldout
    version = tokenledger_version()
    prereg = prereg_record()
    consts, consts_sha = parse_constants()
    transcripts, st = claude_usage.extract(freeze=FREEZE)
    log = RAW / "claude_requests.jsonl"
    claude_usage.write_log(transcripts, log)
    log_sha = sha256_bytes(log.read_bytes())
    tl, tl_sha, tl_cmd = run_tokenledger(log)
    per = ledger_by_transcript(log)

    # tokenledger must agree exactly with the log it was given.
    recs = list(claude_usage.records(transcripts))
    tot = {k: sum(r["usage"][k] for r in recs) for k in claude_usage.USAGE_KEYS}
    total_input = tot["input_tokens"] + tot["cache_creation_input_tokens"] + tot["cache_read_input_tokens"]
    checks = {"requests": (tl["n_requests"], len(recs)), "transcripts": (tl["n_turns"], len(transcripts)),
              "total_input": (tl["total_input"], total_input),
              "cache_read": (tl["total_cache_read"], tot["cache_read_input_tokens"]),
              "per_transcript_input": (sum(v["I"] for v in per.values()), tl["total_input"])}
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        sys.exit(f"BLOCKED: tokenledger disagrees with the extracted log: {bad}")

    prices, prices_sha = load_prices()
    comp = {k: 0.0 for k in ("fresh", "cacheWrite", "cacheRead", "output")}
    by_role: dict[str, float] = {}
    input_cost = [0.0] * len(transcripts)
    priced = [True] * len(transcripts)
    unpriced_requests = 0
    for ti, t in enumerate(transcripts):
        for r in t.requests:
            p = price_for(r.model, prices["models"])
            if p is None:
                priced[ti] = False
                unpriced_requests += 1
                continue
            c = request_cost(r.usage, p)
            for k in comp:
                comp[k] += c[k]
            by_role[t.role] = by_role.get(t.role, 0.0) + sum(c.values())
            input_cost[ti] += c["fresh"] + c["cacheWrite"] + c["cacheRead"]
    comp_total = sum(comp.values())
    role_total = sum(by_role.values())

    # Tariff: the page's multipliers over base input, for the models in the data.
    used = sorted({r.model for t in transcripts for r in t.requests if price_for(r.model, prices["models"])})
    mult = {k: sorted({round(price_for(m, prices["models"])[k] / price_for(m, prices["models"])["input"], 4)
                       for m in used}) for k in ("cache_write_5m", "cache_write_1h", "cache_hit")}
    if len(mult["cache_write_5m"]) != 1 or len(mult["cache_write_1h"]) != 1:
        sys.exit("BLOCKED: cache-write multipliers differ across models; the text states one each")
    w1h = sum(r.usage["cache_creation"]["ephemeral_1h_input_tokens"] for t in transcripts for r in t.requests)

    # Sensitivity of the cost split to the pricing semantics (not registered; Threats only):
    # every cache hit at a flat 0.1x base input, every write at the 5-minute rate, and both.
    def split_under(flat_hits: bool, all_5m: bool) -> dict[str, float]:
        c = {k: 0.0 for k in comp}
        for t in transcripts:
            for r in t.requests:
                p = price_for(r.model, prices["models"])
                if p is None:
                    continue
                q = dict(p)
                if flat_hits:
                    q["cache_hit"] = 0.1 * p["input"]
                if all_5m:
                    q["cache_write_1h"] = p["cache_write_5m"]
                for k, v in request_cost(r.usage, q).items():
                    c[k] += v
        tot_c = sum(c.values())
        return {k: v / tot_c for k, v in c.items()}
    variants = {"paper": {k: v / comp_total for k, v in comp.items()},
                "flat_hits_0.1x": split_under(True, False), "all_writes_5m": split_under(False, True),
                "both": split_under(True, True)}
    read_shares = [v["cacheRead"] for v in variants.values()]
    read_largest_everywhere = all(v["cacheRead"] == max(v.values()) for v in variants.values())
    shares = {n: v / role_total * 100 for n, v in sorted(by_role.items(), key=lambda kv: -kv[1])}
    if "main" not in shares:
        sys.exit("BLOCKED: no main-loop transcripts")
    subagents = [(n, s) for n, s in shares.items() if n != "main"][:TOP_SUBAGENTS]

    # Reconciliation with the harness's own report (counts every usage line as a request).
    claude, claude_sha, claude_cmd = run_report(["report", "--claude", "--json"], "claude_report.json")
    line_factor = st.line_requests / len(recs)
    cr_ours = tot["cache_read_input_tokens"] / total_input * 100
    cr_lines = st.line_cache_read / st.line_input * 100
    cr_harness = claude["total"]["cacheRead"] / claude["totalInput"] * 100
    tool_share_sum = sum(claude["costShare"].values())
    recon = {
        "harness_report_requests_unfrozen": claude["total"]["requests"],
        "harness_report_transcript_files_unfrozen": claude["transcripts"],
        "harness_report_input_tokens_unfrozen": claude["totalInput"],
        "harness_report_cache_read_share_pct": round(cr_harness, 4),
        "harness_report_cost_share_sum": round(tool_share_sum, 4),
        "line_level_requests_frozen": st.line_requests, "line_level_input_frozen": st.line_input,
        "line_level_cache_read_share_pct": round(cr_lines, 4),
        "deduplicated_requests_frozen": len(recs), "deduplicated_input_frozen": total_input,
        "deduplicated_cache_read_share_pct": round(cr_ours, 4),
        "line_to_request_factor": round(line_factor, 4),
        "line_to_request_input_factor": round(st.line_input / total_input, 4),
    }

    snap_figs: dict[str, Any] = {
        "SnapshotDate": figure(FREEZE[:10], "date", "tokenledger_log", "freeze: requests before 00:00 UTC on this date"),
        "TokenledgerVersion": figure(version, "text", "tokenledger_report", "tokenledger-llm version used"),
        "TranscriptCount": figure(tl["n_turns"], "int", "tokenledger_report", "transcripts with at least one request"),
        "RequestCount": figure(tl["n_requests"], "int", "tokenledger_report", "orchestrator API requests (deduplicated)"),
        "TotalInputTokens": figure(tl["total_input"], "billions1", "tokenledger_report",
                                   "orchestrator input tokens (all kinds)"),
        "CacheReadSharePct": figure(tl["cache_read_pct"] * 100, "pct1", "tokenledger_report",
                                    "cache-read share of orchestrator input tokens"),
        "CacheWriteSharePct": figure(tot["cache_creation_input_tokens"] / total_input * 100, "pct1",
                                     "tokenledger_log", "cache-write share of orchestrator input tokens"),
        "FreshInputSharePct": figure(tot["input_tokens"] / total_input * 100, "pct3", "tokenledger_log",
                                     "uncached (fresh) share of orchestrator input tokens"),
        "CostShareCacheReadPct": figure(comp["cacheRead"] / comp_total * 100, "pct1", "list_prices",
                                        "cache-read share of list-price cost"),
        "CostShareCacheWritePct": figure(comp["cacheWrite"] / comp_total * 100, "pct1", "list_prices",
                                         "cache-write share of list-price cost"),
        "CostShareOutputPct": figure(comp["output"] / comp_total * 100, "pct1", "list_prices",
                                     "output share of list-price cost"),
        "CostShareFreshPct": figure(comp["fresh"] / comp_total * 100, "pct3", "list_prices",
                                    "uncached-input share of list-price cost"),
        "MainLoopSharePct": figure(shares["main"], "pct1", "list_prices", "main-loop share of list-price cost"),
        "AgentTypeCount": figure(len(by_role), "int", "tokenledger_log", "distinct agent roles seen"),
        "LongSessionCount": figure(sum(1 for v in per.values() if v["C"] > consts["LONG_SESSION_THRESHOLD"]),
                                   "int", "tokenledger_report", "transcripts whose peak context exceeded the threshold"),
        "PriceAccessDate": figure(prices["source"]["accessed"], "date", "list_prices", "pricing page access date"),
        "ReconLineFactor": figure(line_factor, "float1", "tokenledger_log",
                                  "usage lines per deduplicated request (same frozen requests; a line "
                                  "counter, as the harness report then was, sees this many)"),
        "ReconCacheReadGapPts": figure(abs(cr_lines - cr_ours), "float2", "tokenledger_log",
                                       "cache-read share, line counting minus deduplicated, points"),
        "SplitFixLostRequests": figure(st.separator_only_requests, "int", "tokenledger_log",
                                       "requests a str.splitlines() reader dropped (every line carried a raw "
                                       "Unicode line separator); recovered by the 16 Nov 2026 fix"),
        "OrchFirstDate": figure(min(t.first_ts for t in transcripts)[:10], "date", "tokenledger_log",
                                "date of the first orchestrator request in the data (UTC)"),
        "OrchMainTranscripts": figure(sum(t.kind == "main" for t in transcripts), "int", "tokenledger_log",
                                      "main-loop transcripts (sessions)"),
        "OrchSubagentTranscripts": figure(sum(t.kind == "subagent" for t in transcripts), "int",
                                          "tokenledger_log", "subagent transcripts (runs)"),
        "OrchModelCount": figure(len({r.model for t in transcripts for r in t.requests}), "int",
                                 "tokenledger_log", "distinct orchestrator model ids in the data"),
        "PriceWriteShortMult": figure(mult["cache_write_5m"][0], "float2", "list_prices",
                                      "5-minute cache write price over base input (all models in the data)"),
        "PriceWriteLongMult": figure(mult["cache_write_1h"][0], "float1", "list_prices",
                                     "1-hour cache write price over base input (all models in the data)"),
        "PriceHitMultMin": figure(min(mult["cache_hit"]), "float3", "list_prices",
                                  "smallest cache-hit price over base input among models in the data"),
        "PriceHitMultMax": figure(max(mult["cache_hit"]), "float1", "list_prices",
                                  "largest cache-hit price over base input among models in the data"),
        "CacheWriteLongSharePct": figure(w1h / tot["cache_creation_input_tokens"] * 100, "pct0", "tokenledger_log",
                                         "share of cache-write tokens written at the 1-hour rate"),
        "CostShareCacheReadSensMinPct": figure(min(read_shares) * 100, "pct1", "list_prices",
                                               "cache-read cost share, lowest across pricing-semantics variants"),
        "CostShareCacheReadSensMaxPct": figure(max(read_shares) * 100, "pct1", "list_prices",
                                               "cache-read cost share, highest across pricing-semantics variants"),
        "ReconToolShareSum": figure(tool_share_sum, "float2", "harness_claude_report",
                                    "sum of the harness report's printed cost shares"),
    }
    names = ["One", "Two", "Three"]
    for i, (name, share) in enumerate(subagents):
        snap_figs[f"SubagentTop{names[i]}Name"] = figure(name, "text", "tokenledger_log", f"subagent role ranked {i + 1}")
        snap_figs[f"SubagentTop{names[i]}SharePct"] = figure(share, "pct1", "list_prices",
                                                             f"list-price cost share of subagent role ranked {i + 1}")
    tl_desc = (f"tokenledger-llm {version} (PyPI) over the de-duplicated per-request log extracted from "
               "Claude Code transcripts by tools/claude_usage.py (one record per message id; freeze "
               f"{FREEZE})")
    common_sources = [
        {"id": "tokenledger_log", "grade": "measured", "sha256": log_sha,
         "command": "python tools/pull_evidence.py --only orchestrator",
         "description": "per-request usage log (token counts, model, time, role; no content), kept in "
                        "evidence/raw/claude_requests.jsonl"},
        {"id": "tokenledger_report", "grade": "measured", "sha256": tl_sha, "command": tl_cmd,
         "description": tl_desc},
        {"id": "list_prices", "grade": "model-parameter", "sha256": prices_sha,
         "description": "Anthropic list prices per model and token kind from the official pricing page "
                        "(https://platform.claude.com/docs/en/about-claude/pricing, 'Model pricing'), accessed "
                        f"{prices['source']['accessed']}; committed as evidence/params/anthropic_list_prices.json"},
    ]
    write_json(EVIDENCE / "orchestrator_snapshot.json", {
        "schema_version": 1, "kind": "orchestrator_snapshot",
        "provenance": {"generated_at": utc_now(), "generator": GENERATOR, "sources": common_sources + [
            {"id": "harness_claude_report", "grade": "measured", "command": claude_cmd, "sha256": claude_sha,
             "description": "the harness's own Claude report, used only to reconcile against tokenledger"},
            {"id": "cost_model_parameters", "grade": "model-parameter", "sha256": consts_sha,
             "description": "long-session threshold parsed from [REDACTED: infrastructure] source"},
        ], "notes": [
            "Instrument: tokenledger-llm; the harness report is a reconciliation source only.",
            "The harness report counted every transcript line that carries usage as a request (as recorded "
            "in evidence commit d7bf97f: 210,443 requests); one response spans several lines, so its totals "
            "were inflated by the line factor. It has since been corrected; ReconLineFactor and "
            "ReconCacheReadGapPts replicate the line-counting reader on the frozen requests (tokenledger_log).",
            "Transcript lines are split on newline only. Before 16 Nov 2026 the extractor used "
            "str.splitlines(), which also breaks at raw U+2028/U+2029/U+0085 inside JSON strings; "
            "{} usage lines failed to parse and {} requests were lost (extraction.separator_lines, "
            "separator_only_requests).".format(st.separator_lines, st.separator_only_requests),
            "Not registered: cost_split_sensitivity re-prices the same requests with every cache hit at a "
            "flat 0.1x base input, every write at the 5-minute rate, and both; cache reads are the largest "
            "cost component in {} variants.".format("all" if read_largest_everywhere else "NOT all"),
            "Long context: the pricing page lists a prompt-length surcharge only for Claude Haiku 5.5, which "
            "does not occur in the data; every model in the data has one rate per token kind.",
            ("Its printed cost shares sum to {:.4f}, not 1: the model-tier weight is applied in the "
             "denominator only. Not used.".format(tool_share_sum) if abs(tool_share_sum - 1) > 0.01 else
             "Its printed cost shares now sum to {:.4f}. The runs of 15 Nov 2026 (evidence commits d7bf97f, "
             "3dc01a3) recorded a sum of 0.2133: the model-tier weight sat in the denominator only. The "
             "harness has since changed that computation. Not used either way.".format(tool_share_sum)),
            "List prices weight token kinds and models; they are not the operator's bill (subscription).",
        ]},
        "aggregates": {
            "requests": len(recs), "transcripts": len(transcripts),
            "transcripts_by_kind": {k: sum(t.kind == k for t in transcripts) for k in ("main", "subagent")},
            "input_tokens": {"total": total_input, "fresh": tot["input_tokens"],
                             "cache_write": tot["cache_creation_input_tokens"],
                             "cache_read": tot["cache_read_input_tokens"]},
            "output_tokens": tot["output_tokens"],
            "list_price_cost_component_share": {k: round(v / comp_total, 6) for k, v in comp.items()},
            "cost_split_sensitivity": {n: {k: round(x, 6) for k, x in v.items()} for n, v in variants.items()},
            "cache_read_largest_in_every_variant": read_largest_everywhere,
            "price_multipliers_over_base_input": mult, "models_priced": used,
            "cache_write_tokens_1h": w1h,
            "list_price_role_share_pct": {n: round(s, 3) for n, s in shares.items()},
            "unpriced_requests": unpriced_requests,
            "extraction": vars(st), "reconciliation": recon,
        },
        "figures": snap_figs,
    })
    print(f"orchestrator_snapshot.json: {len(transcripts)} transcripts, {len(recs):,} requests "
          f"(line factor {line_factor:.2f}), cache-read {cr_ours:.2f}%, main loop {shares['main']:.2f}%")

    # ---- held-out test (PREREG_heldout.md)
    units = []
    for ti, t in enumerate(transcripts):
        v = per[ti]
        first = claude_usage.parse_ts(t.first_ts).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        units.append(heldout.Unit(first_ts=first, order=ti, kind=t.kind, T=v["T"], C=v["C"], P=v["P"],
                                  I=v["I"], y=input_cost[ti], priced=priced[ti]))
    with (RAW / "heldout_units.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for u in units:
            f.write(json.dumps(vars(u), sort_keys=True) + "\n")
    res = heldout.run(units)
    e1 = {"no_requests": st.files_without_requests, "only_after_freeze": st.files_only_after_freeze,
          "all_requests_in_earlier_transcript": st.files_emptied_by_dedup}
    if sum(e1.values()) + len(transcripts) != st.files:
        sys.exit("BLOCKED: E1 counts do not reconcile files with transcripts")
    res["excluded_E1_no_request_before_freeze"] = e1
    explo = heldout.exploratory(units)
    prim, sec, sub = res["primary_cost"], res["secondary_tokens"], res["test_subsets_cost"]
    # Units holding a request that the earlier str.splitlines() reader dropped (deviation note).
    # Test-set membership is taken from the registered split itself (same exclusions and order).
    fixed = [ti for ti, t in enumerate(transcripts) if any(r.separator_only for r in t.requests)]
    _, test_units = heldout.temporal_split(heldout.eligible(units)[0])
    test_orders = {u.order for u in test_units}
    split_fix = {"requests_recovered": st.separator_only_requests, "units_affected": len(fixed),
                 "units_affected_in_test": sum(1 for ti in fixed if units[ti].order in test_orders)}
    hs = "tokenledger_report"
    hfigs: dict[str, Any] = {
        "HoUnits": figure(res["n_units"], "int", hs, "transcripts before exclusions"),
        "HoEligible": figure(res["n_eligible"], "int", hs, "eligible transcripts"),
        "HoExclShort": figure(res["excluded"]["E2_fewer_than_3_requests"], "int", hs, "excluded: fewer than 3 requests"),
        "HoExclUnpriced": figure(res["excluded"]["E3_unpriced_model"], "int", hs, "excluded: unpriced model"),
        "HoExclZero": figure(res["excluded"]["E4_zero_cost_or_prompt"], "int", hs, "excluded: zero cost or prompt"),
        "HoExclNoRequests": figure(sum(e1.values()), "int", hs, "E1: transcript files with no request before the freeze"),
        "HoConstRmse": figure(explo["const_test"]["rmse_log"], "float2", hs,
                              "exploratory: test RMSE of predicting the training mean of ln cost"),
        "HoCtRatioMedian": figure(explo["ct_over_i_median_test"], "float1", hs,
                                  "exploratory: median of peak context x turns over total input, test"),
        "HoCtLogCorr": figure(explo["corr_ln_ct_ln_i_test"], "float3", hs,
                              "exploratory: correlation of ln(C x T) with ln(total input), test"),
        "HoDayCiLow": figure(explo["d_rmse_log_day_clustered_ci95"][0], "signedfloat2", hs,
                             "exploratory: day-clustered bootstrap 2.5% of RMSE_A - RMSE_B"),
        "HoDayCiHigh": figure(explo["d_rmse_log_day_clustered_ci95"][1], "signedfloat2", hs,
                              "exploratory: day-clustered bootstrap 97.5% of RMSE_A - RMSE_B"),
        "HoTestDays": figure(explo["test_days"], "int", hs, "exploratory: distinct start days in the test set"),
        "HoTrainN": figure(res["n_train"], "int", hs, "training transcripts"),
        "HoTestN": figure(res["n_test"], "int", hs, "test transcripts"),
        "HoTestMainN": figure(res["n_test_main"], "int", hs, "main-loop transcripts in the test set"),
        "HoTrainMainN": figure(res["n_train_main"], "int", hs, "main-loop transcripts in the training set"),
        "HoTestFrom": figure(res["test_first_ts"][:10], "date", hs, "first-request date of the earliest test transcript"),
        "HoTrainPct": figure(heldout.TRAIN_FRACTION * 100, "pct0", "prereg", "training share of the temporal split"),
        "HoMinRequests": figure(heldout.MIN_REQUESTS, "word", "prereg", "minimum requests per transcript"),
        "HoResamples": figure(heldout.RESAMPLES, "int", "prereg", "bootstrap resamples"),
        "HoPreregCommit": figure(prereg["commit"][:7], "text", "prereg", "commit that registered the plan"),
        "HoPreregDate": figure(prereg["committed_at"][:10], "date", "prereg", "registration date"),
        "HoDecision": figure(prim["decision"], "text", hs, "pre-registered decision, primary target"),
        "HoDeltaRmse": figure(prim["a_minus_b"]["d_rmse_log"], "signedfloat2", hs, "RMSE_A - RMSE_B on ln cost, test"),
        "HoDeltaRmseLow": figure(prim["a_minus_b"]["d_rmse_log_ci95"][0], "signedfloat2", hs, "bootstrap 2.5% of the difference"),
        "HoDeltaRmseHigh": figure(prim["a_minus_b"]["d_rmse_log_ci95"][1], "signedfloat2", hs, "bootstrap 97.5% of the difference"),
        "HoTokDecision": figure(sec["decision"], "text", hs, "decision rule applied to the token target"),
        "HoSplitFixUnits": figure(split_fix["units_affected"], "word", hs,
                                  "transcripts holding a request recovered by the line-splitting fix"),
        "HoSplitFixTestUnits": figure(split_fix["units_affected_in_test"], "word", hs,
                                      "of those, transcripts in the test set"),
    }
    tags = {"A": "A", "B": "B", "S1": "SOne", "S2": "STwo"}
    for m, tag in tags.items():
        mm = prim["models"][m]
        hfigs[f"Ho{tag}Rmse"] = figure(mm["test"]["rmse_log"], "float2", hs, f"model {m}: test RMSE on ln cost")
        hfigs[f"Ho{tag}TrainRmse"] = figure(mm["train"]["rmse_log"], "float2", hs,
                                            f"model {m}: training RMSE on ln cost")
        hfigs[f"Ho{tag}Mape"] = figure(mm["test"]["mape"], "pct0", hs, f"model {m}: test MAPE")
        hfigs[f"Ho{tag}Rsq"] = figure(mm["test"]["r2_log"], _num(mm["test"]["r2_log"]), hs, f"model {m}: test R^2 on ln cost")
        hfigs[f"Ho{tag}Slope"] = figure(mm["slope"], _num(mm["slope"]), hs, f"model {m}: fitted slope (training)")
        tm = sec["models"][m]["test"]
        hfigs[f"HoTok{tag}Rmse"] = figure(tm["rmse_log"], "float2", hs, f"model {m}: test RMSE on ln tokens")
        hfigs[f"HoTok{tag}Rsq"] = figure(tm["r2_log"], _num(tm["r2_log"], "float3"), hs, f"model {m}: test R^2 on ln tokens")
    for kind, tag in (("main", "Main"), ("subagent", "Sub")):
        hfigs[f"Ho{tag}TestN"] = figure(sub[kind]["n"], "int", hs, f"{kind} transcripts in the test set")
        for m in ("A", "B"):
            if m in sub[kind]:
                hfigs[f"Ho{tag}{m}Rmse"] = figure(sub[kind][m]["rmse_log"], "float2", hs,
                                                  f"model {m}: test RMSE on ln cost, {kind} subset")
    write_json(EVIDENCE / "heldout.json", {
        "schema_version": 1, "kind": "heldout",
        "provenance": {"generated_at": utc_now(), "generator": GENERATOR, "sources": common_sources + [
            {"id": "prereg", "grade": "documented-record", "sha256": prereg["sha256"],
             "description": f"pre-registration {prereg['path']}, added in commit {prereg['commit'][:12]} at "
                            f"{prereg['committed_at']} before any per-unit value was computed; unchanged since"},
        ], "notes": [
            "One operator, one harness, one provider. Features and target come from the same transcript: "
            "the test concerns the form of the relation, not forecasting a conversation before it runs.",
            "Deviation: the freeze was applied before ordering transcripts for cross-file de-duplication; "
            "the plan did not fix the order of these two steps.",
            "Per-unit rows stay in evidence/raw/heldout_units.jsonl (gitignored); only aggregates here.",
            "Deviation: the plan required every exclusion to be counted; E1 was not counted in the first "
            "run and was added after an independent audit (no change to any registered result). E1 counts "
            "transcript files on disk at read time, so it grows as conversations are recorded after the "
            "freeze; no registered result depends on it.",
            "Not registered: aggregates.exploratory (constant baseline, closeness of model A's feature to "
            "total input, day-clustered interval) was added after the audit to qualify the reading; the "
            "secondary models S1/S2 and the main-loop/subagent subsets have no interval and are descriptive.",
            ("Deviation: the runs of 15 Nov 2026 (evidence commits d7bf97f, 3dc01a3, 8b76f16) split transcript "
             "lines with str.splitlines(), which also breaks at raw U+2028/U+2029/U+0085 inside JSON strings, "
             "so {r} request(s) were lost although the registered rule keeps one request per message id with "
             "a usage block. Fixed on 16 Nov 2026 (newline-only split) and re-run: {u} unit(s) changed, {t} "
             "of them in the test set. Compare the registered aggregates with those commits for the effect."
             ).format(r=split_fix["requests_recovered"], u=split_fix["units_affected"],
                      t=split_fix["units_affected_in_test"]),
            "Code timing: tools/claude_usage.py and tools/heldout.py were committed after the registration "
            "(1e3e96b); the unit-building and pricing code in this file was committed together with the "
            "first results (d7bf97f), so git alone cannot show it predates them.",
        ]},
        "aggregates": {"prereg": prereg, "freeze": FREEZE, "tokenledger_version": version, **res,
                       "exploratory": explo, "split_fix": split_fix},
        "figures": hfigs,
    })
    print(f"heldout.json: n train/test {res['n_train']}/{res['n_test']}; A rmse "
          f"{prim['models']['A']['test']['rmse_log']:.3f}, B rmse {prim['models']['B']['test']['rmse_log']:.3f}; "
          f"{prim['decision']}")


def _section(text: str, heading: str) -> str:
    """Return one markdown section (heading line to the next heading of the same level)."""
    start = text.find(heading)
    if start < 0:
        sys.exit(f"BLOCKED: section {heading!r} not found")
    level = len(heading) - len(heading.lstrip("#"))
    nxt = re.search(rf"^#{{1,{level}}} ", text[start + len(heading):], re.M)
    end = start + len(heading) + (nxt.start() if nxt else len(text))
    return text[start:end]


def _int(s: str) -> int:
    return int(s.replace(",", ""))


def index_ab() -> None:  # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    sys.exit("[REDACTED: unrelated]: this producer is not part of the release")


def baseline_sample() -> None:  # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    sys.exit("[REDACTED: unrelated]: this producer is not part of the release")


def failfirst_incidents() -> None:  # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    # [REDACTED: unrelated]
    sys.exit("[REDACTED: unrelated]: this producer is not part of the release")


STEPS = {"costs": harness_costs, "orchestrator": orchestrator, "ab": index_ab, "baseline": baseline_sample,
         "incidents": failfirst_incidents}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=sorted(STEPS), action="append",
                    help="run only these producers (default: all)")
    args = ap.parse_args()
    for name in args.only or list(STEPS):
        STEPS[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
