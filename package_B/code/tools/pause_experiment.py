#!/usr/bin/env python3
"""Pause experiment: composition, pause-caused re-writes, and the cause rule against ground truth.

Pre-registered in experiments/PREREG_pause.md. Reads the container runs that
experiments/pause/runner.py leaves under --runs (one directory per run: meta.json, the run's
Claude Code transcripts under projects/, the pause hook's log under out/pause/pauses.jsonl) and
the run ledger, and writes evidence/pause_experiment.json (aggregates only).

Every request is measured with the paper's own code: tools/claude_usage.py (one record per API
request, chain-mode annotations) and tools/cache_writes.analyse (the positional re-write model and
the cause rule), run per container run. Ground truth is the hook's log: a request is AFTER AN
INJECTED PAUSE when a pause of positive length started and ended inside the gap since its
transcript's previous request. The rule labels a re-write IDLE by the share it assigns to expiry
by idle gap (gap_gt_60min plus gap_5_to_60min).

  * H1 composition: cache-read share of input (fresh + read + write) and re-write share of written
    tokens, pooled per condition, with task-cluster bootstrap CIs.
  * H2 pause-caused re-writes: per task, condition B minus condition A, over the whole run and at
    the planned pause points (A logs the same points with zero length); mean and bootstrap CI.
  * Direct expiry check (per injected pause): the first request after the pause, its written and read
    tokens against the cached prefix before the pause (full re-write expected).
  * Condition C (Amendment 1: one-hour caching, one 61-minute pause): direct expiry check, H3-C (the
    rule's gap_gt_60min label over C re-write events), composition; never part of the A/B pairing.
  * H3 rule validity: over re-write events (requests with re-written tokens) of valid runs, truth =
    after an injected pause, prediction = idle share >= 0.5: precision and recall (task-cluster
    bootstrap CIs), and the share of condition-A re-written tokens labelled idle.

    python tools/pause_experiment.py --runs <runs dir> --ledger <ledger.jsonl>
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cache_writes  # noqa: E402
import claude_usage  # noqa: E402
from evidence_lib import EVIDENCE, figure, sha256_bytes, utc_now, write_json  # noqa: E402

GENERATOR = "tools/pause_experiment.py"
PRICES = EVIDENCE / "params" / "anthropic_list_prices.json"
SEED = 2027
N_BOOT = 10000
IDLE_CAUSES = ("gap_gt_60min", "gap_5_to_60min")
FULL_REWRITE = 0.95     # direct expiry check: >= 95% of the prefix written, < 5% of it read back
EXPIRED_SHARE = 0.5     # a pause "expired" the cache if the next request re-wrote >= this share of C
# Pre-registered decision thresholds (PREREG_pause.md)
H1_READ_SHARE_LO = 0.80
H3_PRECISION, H3_RECALL = 0.90, 0.90
LIFETIME_S = {"5m": 300.0, "1h": 3600.0}


def parse_ts(ts: str) -> dt.datetime:
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))


def pause_in_gap(prev: dt.datetime, t: dt.datetime, pauses: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The first logged pause that started and ended inside (prev, t], else None."""
    for p in pauses:
        if prev <= p["start"] and p["end"] <= t:
            return p
    return None


def idle_share(causes: dict[str, float]) -> float:
    return sum(causes.get(c, 0.0) for c in IDLE_CAUSES)


def lifetime_tier(w5: int, w1: int) -> str:
    tot = w5 + w1
    if not tot:
        return "none"
    if w1 >= 0.99 * tot:
        return "1h"
    if w5 >= 0.99 * tot:
        return "5m"
    return "mixed"


def read_pauses(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            p = json.loads(line)
            out.append({"call": p["call"], "start": parse_ts(p["start"]), "end": parse_ts(p["end"]),
                        "seconds": p["seconds"]})
    return out


def load_runs(root: Path) -> list[dict[str, Any]]:
    """Every run directory under root (sorted), with its annotated request records and pause log."""
    runs = []
    for meta_path in sorted(Path(root).glob("*/meta.json")):
        d = meta_path.parent
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        recs: list[dict[str, Any]] = []
        projects = d / "projects"
        if projects.is_dir():
            transcripts, _ = claude_usage.extract(projects, chain=True)
            recs = list(claude_usage.records(transcripts))
            for r, ann in zip(recs, claude_usage.annotations(transcripts), strict=True):
                for k in ("after_compact", "chain_from_prev", "after_list_delta", "prompt_changed", "version_changed"):
                    r[k] = ann[k]
        runs.append({**meta, "records": recs, "pauses": read_pauses(d / "out" / "pause" / "pauses.jsonl"),
                     "inputs": sorted(p for p in d.rglob("*") if p.is_file() and p.suffix in (".jsonl", ".json"))})
    return runs


def run_metrics(run: dict[str, Any], prices: dict[str, dict[str, float]]) -> dict[str, Any]:
    """One run through the paper's analysis, joined with the pause log."""
    trace = cache_writes.new_trace()
    a = cache_writes.analyse(run["records"], prices, trace=trace)
    w5 = sum(r["usage"]["cache_creation"]["ephemeral_5m_input_tokens"] for r in run["records"])
    w1 = sum(r["usage"]["cache_creation"]["ephemeral_1h_input_tokens"] for r in run["records"])
    life = lifetime_tier(w5, w1)
    life_s = LIFETIME_S.get(life, LIFETIME_S["5m"])
    injected = [p for p in run["pauses"] if p["seconds"] > 0]
    events, point_rw, natural = [], 0, 0
    point_write = rr_write = 0     # reconciliation: all written tokens at planned points / at any re-writing request
    expired: set[int] = set()
    first_after: dict[int, dict[str, Any]] = {}   # injected pause -> first request whose gap holds it
    prev_by_turn: dict[Any, dict[str, Any]] = {}
    fresh = read = 0
    for row in trace["requests"]:
        fresh += row["fresh"]
        read += row["read"]
        t = parse_ts(row["ts"])
        prev = prev_by_turn.get(row["turn"])
        prev_by_turn[row["turn"]] = row
        if prev is None:
            continue
        pt = parse_ts(prev["ts"])
        gap = (t - pt).total_seconds()
        planned = pause_in_gap(pt, t, run["pauses"])
        inj = pause_in_gap(pt, t, injected)
        if planned is not None:
            point_rw += row["rewrite"]
            point_write += row["write"]
        if prev["read"] + prev["write"] > row["read"]:      # the request did not read back the whole cached prefix
            rr_write += row["write"]
        if inj is not None and row["rewrite"] >= EXPIRED_SHARE * (prev["read"] + prev["write"]) > 0:
            expired.add(inj["call"])
        if inj is not None and (inj["call"] not in first_after or t < first_after[inj["call"]]["t"]):
            first_after[inj["call"]] = {"t": t, "prefix_before": prev["read"] + prev["write"],
                                        "write_after": row["write"], "read_after": row["read"], "gap_s": gap}
        if inj is None and gap > life_s:
            natural += 1
        if row["rewrite"]:
            events.append({"rewrite": row["rewrite"], "rewrite_usd": row["rewrite_cost"], "gap_s": gap,
                           "after_pause": inj is not None, "idle": idle_share(row["causes"]),
                           "natural_gap": inj is None and gap > life_s,
                           "long": row["causes"].get("gap_gt_60min", 0.0)})
    return {"fresh": fresh, "read": read, "write": a["write_tokens"], "rewrite": a["rewrite"],
            "rewrite_usd": a["write_cost_usd"]["rewrite"], "cost_usd": a["total_cost_usd"],
            "w5": w5, "w1": w1, "lifetime": life, "requests": len(trace["requests"]),
            "point_rewrite": point_rw, "point_write": point_write, "rewrite_request_write": rr_write,
            "injected_pauses": len(injected), "planned_points": len(run["pauses"]),
            "expired_after_pause": len(expired), "natural_long_gaps": natural, "events": events,
            "pause_writes": [direct_expiry_row(v) for _, v in sorted(first_after.items())],
            "rewrite_by_cause": a["rewrite_by_cause"]}


def direct_expiry_row(v: dict[str, Any]) -> dict[str, Any]:
    """Direct expiry check for one injected pause: the first request after it against the cached prefix
    before it. Full re-write = nothing of that prefix read back and at least FULL_REWRITE of it written."""
    pre = v["prefix_before"]
    ratio = v["write_after"] / pre if pre else None
    return {"prefix_before": pre, "write_after": v["write_after"], "read_after": v["read_after"],
            "gap_s": v["gap_s"], "write_over_prefix": ratio,
            "full_rewrite": bool(pre) and v["read_after"] < (1 - FULL_REWRITE) * pre and ratio >= FULL_REWRITE,
            # exploratory (not pre-registered): the part of the prefix NOT read back was re-written in full
            "rest_rewritten": bool(pre) and v["write_after"] >= FULL_REWRITE * max(0, pre - v["read_after"])}


def bootstrap_mean_ci(values: list[float], n_boot: int = N_BOOT, seed: int = SEED, level: float = 0.95) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(n_boot))
    k = (1 - level) / 2
    return {"n": n, "mean": sum(values) / n, "lo": means[int(k * (n_boot - 1))], "hi": means[int((1 - k) * (n_boot - 1))]}


def cluster_ci(clusters: list[Any], stat, n_boot: int = N_BOOT, seed: int = SEED, level: float = 0.95) -> dict[str, Any]:
    """Percentile CI of stat(list of clusters) resampling clusters (tasks) with replacement."""
    point = stat(clusters) if clusters else None
    if not clusters:
        return {"value": None, "lo": None, "hi": None}
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        v = stat([rng.choice(clusters) for _ in clusters])
        if v is not None:
            vals.append(v)
    vals.sort()
    k = (1 - level) / 2
    if not vals:
        return {"value": point, "lo": None, "hi": None}
    return {"value": point, "lo": vals[int(k * (len(vals) - 1))], "hi": vals[int((1 - k) * (len(vals) - 1))]}


def _read_share(ms: list[dict[str, Any]]) -> float | None:
    tot = sum(m["fresh"] + m["read"] + m["write"] for m in ms)
    return sum(m["read"] for m in ms) / tot if tot else None


def _rewrite_share(ms: list[dict[str, Any]]) -> float | None:
    w = sum(m["write"] for m in ms)
    return sum(m["rewrite"] for m in ms) / w if w else None


def _confusion(evs: list[dict[str, Any]]) -> dict[str, int]:
    c = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for e in evs:
        pred = e["idle"] >= 0.5
        c[("t" if pred == e["after_pause"] else "f") + ("p" if pred else "n")] += 1
    return c


def _confusion_long(evs: list[dict[str, Any]]) -> dict[str, int]:
    """Condition C (Amendment 1): positive label = share >= 0.5 assigned to gap_gt_60min."""
    return _confusion([{**e, "idle": e["long"]} for e in evs])


def _precision(evs):
    c = _confusion(evs)
    return c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else None


def _recall(evs):
    c = _confusion(evs)
    return c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else None


def select_valid(runs: list[dict[str, Any]], include_pilot: bool = False) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    """Non-pilot runs with status ok; per (task, condition) the latest valid attempt. include_pilot is
    for checking the pilot and dry runs only, never for the registered analysis."""
    excl = {"pilot_runs": 0, "failed_runs": 0}
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for r in runs:
        if r.get("pilot") and not include_pilot:
            excl["pilot_runs"] += 1
            continue
        if r.get("status") != "ok":
            excl["failed_runs"] += 1
            continue
        key = (r["instance_id"], r["condition"])
        if key not in best or r.get("attempt", 1) > best[key].get("attempt", 1):
            best[key] = r
    return best, excl


def evaluate(runs: list[dict[str, Any]], prices: dict[str, dict[str, float]], n_boot: int = N_BOOT,
             seed: int = SEED, include_pilot: bool = False) -> dict[str, Any]:
    best, excl = select_valid(runs, include_pilot)
    tasks = sorted({k[0] for k in best})
    paired = [t for t in tasks if (t, "A") in best and (t, "B") in best]
    excl["unpaired_tasks"] = [t for t in tasks if t not in paired]
    m = {t: {c: run_metrics(best[(t, c)], prices) for c in "AB"} for t in paired}
    comp = {}
    for c in "AB":
        ms = [m[t][c] for t in paired]
        comp[c] = {"fresh": sum(x["fresh"] for x in ms), "cache_read": sum(x["read"] for x in ms),
                   "cache_write": sum(x["write"] for x in ms), "rewrite": sum(x["rewrite"] for x in ms),
                   "read_share": _read_share(ms), "rewrite_share": _rewrite_share(ms),
                   "read_share_ci": cluster_ci(ms, _read_share, n_boot, seed),
                   "rewrite_share_ci": cluster_ci(ms, _rewrite_share, n_boot, seed),
                   "cost_usd": sum(x["cost_usd"] for x in ms), "requests": sum(x["requests"] for x in ms),
                   "w5": sum(x["w5"] for x in ms), "w1": sum(x["w1"] for x in ms)}
    diff = [m[t]["B"]["rewrite"] - m[t]["A"]["rewrite"] for t in paired]
    diff_usd = [m[t]["B"]["rewrite_usd"] - m[t]["A"]["rewrite_usd"] for t in paired]
    pdiff = [m[t]["B"]["point_rewrite"] - m[t]["A"]["point_rewrite"] for t in paired]
    per_session = []
    for t in paired:
        a_, b_ = m[t]["A"], m[t]["B"]
        per_session.append({"session": t, "b_point_rewrite": b_["point_rewrite"], "a_point_rewrite": a_["point_rewrite"],
                            "b_point_write": b_["point_write"], "a_point_write": a_["point_write"],
                            "b_rewrite_request_write": b_["rewrite_request_write"],
                            "a_rewrite_request_write": a_["rewrite_request_write"]})

    def _mean(xs):
        return sum(xs) / len(xs) if xs else None
    reconciliation = {
        "registered": "point_rewrite_diff_tokens",
        "mean_registered": _mean([r["b_point_rewrite"] - r["a_point_rewrite"] for r in per_session]),
        "mean_point_write": _mean([r["b_point_write"] - r["a_point_write"] for r in per_session]),
        "mean_rewrite_request_write": _mean([r["b_rewrite_request_write"] - r["a_rewrite_request_write"] for r in per_session]),
        # exact identity: mean_rewrite_request_write = mean_registered + mean_b_new_suffix_at_points + mean_off_point_term
        "mean_b_new_suffix_at_points": _mean([r["b_point_write"] - r["b_point_rewrite"] for r in per_session]),
        "mean_off_point_term": _mean([r["a_point_rewrite"] + (r["b_rewrite_request_write"] - r["b_point_write"])
                                      - r["a_rewrite_request_write"] for r in per_session]),
        "note": ("Registered H2 outcome = re-written POSITIONS (cache_writes: min(C, read + write) - read, C = the previous "
                 "request's read + write) of requests whose gap holds a planned point; A's planned points are its 0 s markers "
                 "between the same tasks. mean_point_write counts every written token of those requests (adds the new suffix "
                 "written after the cached prefix). mean_rewrite_request_write counts every written token of any request that "
                 "read back less than C, at any position in the session (adds new suffix and re-writes after natural gaps, "
                 "in either condition). Exactly: mean_rewrite_request_write = mean_registered + mean_b_new_suffix_at_points (new suffix "
                 "written by B's post-pause requests) + mean_off_point_term (re-writing requests away from the points, "
                 "B minus A, e.g. after natural gaps). The registered outcome is mean_registered; the others are descriptive."),
        "per_session": per_session,
    }
    ev_by_task = [[e for c in "AB" for e in m[t][c]["events"]] for t in paired]
    evs = [e for x in ev_by_task for e in x]
    conf = _confusion(evs)
    a_rw = sum(e["rewrite"] for t in paired for e in m[t]["A"]["events"])
    a_idle = sum(e["rewrite"] * e["idle"] for t in paired for e in m[t]["A"]["events"])
    tp_tok = sum(e["rewrite"] * e["idle"] for e in evs if e["after_pause"])
    pred_tok = sum(e["rewrite"] * e["idle"] for e in evs)
    true_tok = sum(e["rewrite"] for e in evs if e["after_pause"])
    inj = sum(m[t]["B"]["injected_pauses"] for t in paired)
    w5 = comp["A"]["w5"] + comp["B"]["w5"] if paired else 0
    w1 = comp["A"]["w1"] + comp["B"]["w1"] if paired else 0
    c_runs = [best[k] for k in sorted(best) if k[1] == "C"]
    cm = [run_metrics(r, prices) for r in c_runs]
    c_evs_by_run = [[{**e, "idle": e["long"]} for e in x["events"]] for x in cm]
    c_evs = [e for x in c_evs_by_run for e in x]
    condition_c = {
        "runs": len(cm), "lifetime": lifetime_tier(sum(x["w5"] for x in cm), sum(x["w1"] for x in cm)),
        "injected_pauses": sum(x["injected_pauses"] for x in cm),
        "runs_without_pause": sum(x["injected_pauses"] == 0 for x in cm),
        "composition": {"read_share": _read_share(cm), "rewrite_share": _rewrite_share(cm),
                        "requests": sum(x["requests"] for x in cm), "cost_usd": sum(x["cost_usd"] for x in cm)},
        "rule": {**_confusion(c_evs), "events": len(c_evs), "precision": _precision(c_evs), "recall": _recall(c_evs),
                 "precision_ci": cluster_ci(c_evs_by_run, lambda cl: _precision([e for x in cl for e in x]), n_boot, seed),
                 "recall_ci": cluster_ci(c_evs_by_run, lambda cl: _recall([e for x in cl for e in x]), n_boot, seed)},
        "direct_expiry": direct_expiry_summary([pw for x in cm for pw in x["pause_writes"]]),
    }
    return {
        "condition_c": condition_c,
        "paired_tasks": len(paired), "excluded": excl, "lifetime": lifetime_tier(w5, w1),
        "composition": comp,
        "paired": {"rewrite_diff_tokens": bootstrap_mean_ci(diff, n_boot, seed),
                   "rewrite_diff_usd": bootstrap_mean_ci(diff_usd, n_boot, seed),
                   "point_rewrite_diff_tokens": bootstrap_mean_ci(pdiff, n_boot, seed),
                   "reconciliation": reconciliation},
        "rule": {**conf, "events": len(evs), "precision": _precision(evs), "recall": _recall(evs),
                 "precision_ci": cluster_ci(ev_by_task, lambda cl: _precision([e for x in cl for e in x]), n_boot, seed),
                 "recall_ci": cluster_ci(ev_by_task, lambda cl: _recall([e for x in cl for e in x]), n_boot, seed),
                 "token_precision": tp_tok / pred_tok if pred_tok else None,
                 "token_recall": tp_tok / true_tok if true_tok else None,
                 "false_idle_share_a": a_idle / a_rw if a_rw else 0.0,
                 # secondary: any gap longer than the lifetime counts as truth (natural tool-runtime gaps too)
                 "precision_any_gap": _precision([{**e, "after_pause": e["after_pause"] or e["natural_gap"]} for e in evs]),
                 "recall_any_gap": _recall([{**e, "after_pause": e["after_pause"] or e["natural_gap"]} for e in evs])},
        "pauses": {"injected": inj, "planned_points_a": sum(m[t]["A"]["planned_points"] for t in paired),
                   "expired_after_pause": sum(m[t]["B"]["expired_after_pause"] for t in paired)},
        "natural_long_gaps": sum(m[t][c]["natural_long_gaps"] for t in paired for c in "AB"),
        "direct_expiry": direct_expiry_summary([pw for t in paired for pw in m[t]["B"]["pause_writes"]]),
    }


def direct_expiry_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = sorted(r["write_over_prefix"] for r in rows if r["write_over_prefix"] is not None)
    return {"pauses": rows, "n": len(rows),
            "full_rewrite_share": sum(r["full_rewrite"] for r in rows) / len(rows) if rows else None,
            "exploratory_rest_rewritten_share": sum(r["rest_rewritten"] for r in rows) / len(rows) if rows else None,
            "exploratory_read_after_values": sorted({r["read_after"] for r in rows}),
            "median_write_over_prefix": ratios[len(ratios) // 2] if ratios else None,
            "min_gap_s": min((r["gap_s"] for r in rows), default=None)}


def decisions(agg: dict[str, Any]) -> dict[str, str]:
    """Pre-registered decision rules (PREREG_pause.md)."""
    def lo(c):
        return agg["composition"][c]["read_share_ci"]["lo"]
    h1 = all(lo(c) is not None and lo(c) > H1_READ_SHARE_LO for c in "AB")
    p = agg["paired"]["point_rewrite_diff_tokens"]
    h2 = p["lo"] is not None and p["lo"] > 0
    r = agg["rule"]
    h3 = (r["precision"] or 0) >= H3_PRECISION and (r["recall"] or 0) >= H3_RECALL
    rc = agg.get("condition_c", {}).get("rule", {})
    h3c = (rc.get("precision") or 0) >= H3_PRECISION and (rc.get("recall") or 0) >= H3_RECALL
    word = {True: "supported", False: "not supported"}
    return {"H1": word[h1], "H2": word[h2], "H3": word[h3], "H3C": word[h3c]}


def _pct(x):
    return None if x is None else x * 100


def build_figures(agg: dict[str, Any], dec: dict[str, str], cost: float, ledger_rows: int) -> dict[str, Any]:
    ca, cb, r, p = agg["composition"]["A"], agg["composition"]["B"], agg["rule"], agg["paired"]
    f = {
        "PauseTasksPaired": figure(agg["paired_tasks"], "int", "runs", "SWE-bench Verified tasks with a valid run in both conditions"),
        "PauseLifetime": figure(agg["lifetime"], "text", "runs", "cache lifetime of the writes (5m, 1h or mixed)"),
        "PauseReadSharePctA": figure(_pct(ca["read_share"]), "pct1", "runs", "no-pause runs: cache-read share of input"),
        "PauseReadSharePctB": figure(_pct(cb["read_share"]), "pct1", "runs", "pause runs: cache-read share of input"),
        "PauseRewriteSharePctA": figure(_pct(ca["rewrite_share"]), "pct0", "runs", "no-pause runs: re-write share of written tokens"),
        "PauseRewriteSharePctB": figure(_pct(cb["rewrite_share"]), "pct0", "runs", "pause runs: re-write share of written tokens"),
        "PausePointDiffTokens": figure(p["point_rewrite_diff_tokens"]["mean"], "int", "runs",
                                       "mean paired re-written tokens at the pause points, B minus A"),
        "PausePointDiffLo": figure(p["point_rewrite_diff_tokens"]["lo"], "int", "runs", "lower 95% bootstrap bound"),
        "PausePointDiffHi": figure(p["point_rewrite_diff_tokens"]["hi"], "int", "runs", "upper 95% bootstrap bound"),
        "PauseRunDiffTokens": figure(p["rewrite_diff_tokens"]["mean"], "int", "runs",
                                     "mean paired re-written tokens per run, B minus A"),
        "PauseRunDiffUsd": figure(p["rewrite_diff_usd"]["mean"], "float2", "runs", "mean paired re-write cost per run, B minus A, USD"),
        "PauseRulePrecisionPct": figure(_pct(r["precision"]), "pct0", "runs", "cause rule: re-writes labelled idle that followed an injected pause"),
        "PauseRuleRecallPct": figure(_pct(r["recall"]), "pct0", "runs", "cause rule: re-writes after injected pauses labelled idle"),
        "PauseFalseIdlePctA": figure(_pct(r["false_idle_share_a"]), "pctlt1", "runs", "no-pause runs: re-written tokens labelled idle"),
        "PauseFullRewritePct": figure(_pct(agg["direct_expiry"]["full_rewrite_share"]), "pct0", "runs",
                                      "injected pauses whose next request re-wrote the whole cached prefix (direct check)"),
        "PauseWriteOverPrefix": figure(agg["direct_expiry"]["median_write_over_prefix"], "float2", "runs",
                                       "median, over injected pauses, of the next request's written tokens / cached prefix before"),
        "PausePausesInjected": figure(agg["pauses"]["injected"], "int", "runs", "pauses injected in valid pause runs"),
        "PauseCostUsd": figure(cost, "float2", "ledger", f"API list-price cost of all {ledger_rows} runs incl. pilot and re-runs, USD"),
        "PauseCRuns": figure(agg["condition_c"]["runs"], "int", "runs", "valid one-hour-caching runs with a 61-minute pause (C)"),
        "PauseCRulePrecisionPct": figure(_pct(agg["condition_c"]["rule"]["precision"]), "pct0", "runs",
                                         "C: re-writes labelled expiry after more than an hour that followed the pause"),
        "PauseCRuleRecallPct": figure(_pct(agg["condition_c"]["rule"]["recall"]), "pct0", "runs",
                                      "C: re-writes after the 61-minute pause labelled expiry after more than an hour"),
        "PauseCFullRewritePct": figure(_pct(agg["condition_c"]["direct_expiry"]["full_rewrite_share"]), "pct0", "runs",
                                       "C: 61-minute pauses whose next request re-wrote the whole cached prefix"),
        "PauseCReadSharePct": figure(_pct(agg["condition_c"]["composition"]["read_share"]), "pct1", "runs",
                                     "C (one-hour caching): cache-read share of input"),
        "PauseHThreeC": figure(dec["H3C"], "text", "runs", "H3-C decision (Amendment 1)"),
        "PauseHOne": figure(dec["H1"], "text", "runs", "H1 decision"),
        "PauseHTwo": figure(dec["H2"], "text", "runs", "H2 decision"),
        "PauseHThree": figure(dec["H3"], "text", "runs", "H3 decision"),
    }
    return {k: v for k, v in f.items() if v["value"] is not None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=EVIDENCE / "pause_experiment.json")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--include-pilot", action="store_true",
                    help="checking the pilot or a dry run only: analyse pilot runs (never for the evidence file)")
    args = ap.parse_args(argv)
    if args.include_pilot and args.out.resolve() == (EVIDENCE / "pause_experiment.json").resolve():
        sys.exit("BLOCKED: --include-pilot output may not overwrite the registered evidence file")
    praw = PRICES.read_bytes().replace(b"\r\n", b"\n")
    prices = json.loads(praw.decode("utf-8"))["models"]
    runs = load_runs(args.runs)
    agg = evaluate(runs, prices, n_boot=args.n_boot, include_pilot=args.include_pilot)
    dec = decisions(agg)
    lraw = args.ledger.read_bytes() if args.ledger.exists() else b""
    lrows = [json.loads(x) for x in lraw.decode("utf-8").splitlines() if x.strip()]
    cost = sum(max(float(r.get("usd") or 0), float(r.get("cli_usd") or 0)) for r in lrows)
    h = hashlib.sha256()
    for r in runs:
        for p in r["inputs"]:
            h.update(p.relative_to(args.runs).as_posix().encode() + b"\0" + p.read_bytes())
    agg["decisions"] = dec
    # counts from the ledger, which also holds runs without a run directory (e.g. a pilot run elsewhere)
    pilot_ids = {r["run_id"] for r in lrows if r.get("plan", "full") != "full"} | {r["run_id"] for r in runs if r.get("pilot")}
    agg["excluded"]["pilot_runs"] = len(pilot_ids)
    agg["ledger"] = {"rows": len(lrows), "pilot_rows": sum(r.get("plan", "full") != "full" for r in lrows),
                     "usd_total": cost,
                     "usd_pilots": sum(max(float(r.get("usd") or 0), float(r.get("cli_usd") or 0))
                                       for r in lrows if r.get("plan", "full") != "full")}
    agg["runs_total"] = len({r["run_id"] for r in lrows} | {r["run_id"] for r in runs})
    agg["run_directories"] = len(runs)
    write_json(args.out, {
        "schema_version": 1, "kind": "pause_experiment",
        "provenance": {"generated_at": utc_now(), "generator": GENERATOR, "sources": [
            {"id": "runs", "grade": "measured", "sha256": h.hexdigest(),
             "command": "python tools/pause_experiment.py --runs <runs dir> --ledger <ledger.jsonl>",
             "description": "container runs (transcripts, pause-hook logs, run metadata) of experiments/pause/runner.py; "
                            "sha256 over every run's .json/.jsonl files in path order"},
            {"id": "ledger", "grade": "measured", "sha256": sha256_bytes(lraw),
             "command": "experiments/pause/runner.py ledger", "description": "per-run cost ledger (list prices)"},
            {"id": "list_prices", "grade": "model-parameter", "sha256": sha256_bytes(praw),
             "command": "evidence/params/anthropic_list_prices.json", "description": "Anthropic list prices"},
        ], "notes": [
            "Pre-registered: experiments/PREREG_pause.md. Pilot runs are excluded; failed runs re-run once.",
            "Measured with tools/claude_usage.py and tools/cache_writes.analyse per run (the paper's code).",
            f"Bootstrap: {args.n_boot} resamples of tasks, seed {SEED}, percentile 95% intervals.",
        ]},
        "aggregates": agg,
        "figures": build_figures(agg, dec, cost, len(lrows)),
    })
    print(json.dumps({"paired_tasks": agg["paired_tasks"], "lifetime": agg["lifetime"], "decisions": dec,
                      "rule": {k: agg["rule"][k] for k in ("tp", "fp", "fn", "tn", "precision", "recall")}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
