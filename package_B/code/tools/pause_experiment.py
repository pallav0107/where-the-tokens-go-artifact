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
            for r, changed in zip(recs, shape_changes(transcripts), strict=True):
                r["shape_changed"] = changed    # Amendment 3 exploratory only; the rule never reads it
        inj = d / "out" / "inject" / "injections.jsonl"
        runs.append({**meta, "records": recs, "pauses": read_pauses(d / "out" / "pause" / "pauses.jsonl"),
                     "injections": ([json.loads(x) for x in inj.read_text(encoding="utf-8").splitlines() if x.strip()]
                                    if inj.exists() else []),
                     "inputs": sorted(p for p in d.rglob("*") if p.is_file() and p.suffix in (".jsonl", ".json"))})
    return runs


def shape_changes(transcripts: list[Any]) -> list[bool]:
    """Per request (records() order): the client's latest api-request-shape hash before this request differs
    from the latest one before the previous request (both must exist). Amendment 3, exploratory: a change of
    the request's tools or system blocks that the client records but the published cause rule does not read."""
    out: list[bool] = []
    for t in transcripts:
        shapes: list[tuple[int, str]] = []
        if t.path is not None:
            text = Path(t.path).read_text(encoding="utf-8", errors="replace")
            for n, line in enumerate(text.split("\n")):     # same line numbering as claude_usage.read_transcript
                if '"api-request-shape"' in line:
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(ev, dict) and ev.get("type") == "api-request-shape" and ev.get("shapeHash"):
                        shapes.append((n, str(ev["shapeHash"])))
        for si, r in enumerate(t.requests):
            if si == 0:
                out.append(False)
                continue
            lo, hi = sorted((t.requests[si - 1].order, r.order))
            a, b = claude_usage._prompt_at(shapes, lo), claude_usage._prompt_at(shapes, hi)
            out.append(a is not None and b is not None and a != b)
    return out


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


# ---------------------------------------------------------------- Amendment 3 (PREREG_pause.md)
A3_TYPE = {"model": "MS", "tools": "TL", "mcp": "MCP", "model+pause": "MS+P", "pause+tools": "TL+P",
           "mcp+pause": "MCP+P"}
A3_TYPES = ("MS", "TL", "MCP", "MS+P", "TL+P", "MCP+P")
A3_NOT_IDLE = ("MS", "TL", "MS+P", "TL+P")     # truth: the injected change alone invalidates the prefix
A3_RECORDED = ("MS", "MS+P")                    # ... and the rule reads a marker for it
A3_PASS = 0.90
M1_REWRITE, M2_READ, M2_PASS = 0.5, 0.95, 0.90
K_SUPPORTED, K_NOT_SUPPORTED, K_WINDOW_S = 0.90, 0.10, 300.0
Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float | None, float | None]:
    """Wilson score interval for k successes out of n."""
    if not n:
        return None, None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / den
    return max(0.0, c - h), min(1.0, c + h)


def dominant(causes: dict[str, float]) -> str | None:
    """The cause with the largest share (ties: the rule's cause order)."""
    if not causes:
        return None
    order = cache_writes.CAUSES
    return max(order, key=lambda c: (causes.get(c, 0.0), -order.index(c)))


def _main_sequence(run: dict[str, Any], prices: dict[str, dict[str, float]]) -> tuple[list[tuple], list[dict]]:
    """(ts, row, record, previous row of the same transcript) for main-transcript requests in time order, with
    the cache_writes trace rows (the rule's causes); also every trace row."""
    trace = cache_writes.new_trace()
    cache_writes.analyse(run["records"], prices, trace=trace)
    recs: dict[Any, list[dict[str, Any]]] = {}
    for r in run["records"]:
        recs.setdefault(r["turn"], []).append(r)
    rows: dict[Any, list[dict[str, Any]]] = {}
    for row in trace["requests"]:
        rows.setdefault(row["turn"], []).append(row)
    seq = []
    for turn, rr in recs.items():
        rr.sort(key=lambda r: r["step"])
        for i, (rec, row) in enumerate(zip(rr, rows[turn], strict=True)):
            if rec.get("kind", "main") == "main":
                seq.append((parse_ts(row["ts"]), row, rec, rows[turn][i - 1] if i else None))
    seq.sort(key=lambda x: x[0])
    return seq, trace["requests"]


def a3_boundaries(run: dict[str, Any], prices: dict[str, dict[str, float]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Boundary requests of one S run (the first main-transcript request at or after the start of task b+1)
    with the injected type, the rule's causes, and the exploratory shape-marker variant; plus all-event counts."""
    seq, all_rows = _main_sequence(run, prices)
    tasks = run.get("tasks") or []
    inj = {int(x["boundary"]): x for x in run.get("injections", [])}
    out = []
    for b in sorted(inj):
        if b >= len(tasks) or not tasks[b].get("start"):
            continue                                  # task b+1 never started: boundary not reached
        start = parse_ts(tasks[b]["start"])
        hit = next((x for x in seq if x[0] >= start), None)
        if hit is None or hit[3] is None:
            continue
        t, row, rec, prev = hit
        pre = prev["read"] + prev["write"]
        causes = row["causes"]
        shape_causes = causes
        if row["rewrite"] and rec.get("shape_changed") and not (causes.get("model_change") or causes.get("compaction")):
            shape_causes = {"config_change": 1.0}
        out.append({"session": run.get("instance_id"), "template": run.get("condition"), "boundary": b,
                    "type": A3_TYPE["+".join(sorted(inj[b]["types"]))], "prefix_before": pre,
                    "read": row["read"], "write": row["write"], "rewrite": row["rewrite"],
                    "rewrite_over_prefix": row["rewrite"] / pre if pre else None,
                    "read_over_prefix": row["read"] / pre if pre else None,
                    "gap_s": (t - parse_ts(prev["ts"])).total_seconds(), "model": rec["model"],
                    "after_list_delta": bool(rec.get("after_list_delta")), "shape_changed": bool(rec.get("shape_changed")),
                    "causes": causes, "idle": idle_share(causes), "dominant": dominant(causes),
                    "idle_shape": idle_share(shape_causes), "dominant_shape": dominant(shape_causes)})
    bts = {(x["boundary"]) for x in out}
    ev = [r for r in all_rows if r["rewrite"]]
    n_b = sum(1 for x in out if x["rewrite"])
    other_idle = sum(1 for r in ev if idle_share(r["causes"]) >= 0.5) - sum(1 for x in out if x["rewrite"] and x["idle"] >= 0.5)
    return out, {"events": len(ev), "boundary_events": n_b, "other_events": len(ev) - n_b,
                 "other_labelled_idle": other_idle, "boundaries_reached": len(bts)}


def _share_metric(per_run: list[list[dict[str, Any]]], select, success, n_boot: int, seed: int) -> dict[str, Any]:
    evs = [x for run in per_run for x in run if select(x)]
    k, n = sum(1 for x in evs if success(x)), len(evs)

    def stat(cl):
        e = [x for run in cl for x in run if select(x)]
        return sum(1 for x in e if success(x)) / len(e) if e else None
    ci = cluster_ci(per_run, stat, n_boot, seed) if n else {"lo": None, "hi": None}
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "value": k / n if n else None, "wilson_lo": lo, "wilson_hi": hi,
            "boot_lo": ci["lo"], "boot_hi": ci["hi"]}


def _a3_metrics(per_run: list[list[dict[str, Any]]], idle_key: str, dom_key: str, mcp_ok: bool,
                n_boot: int, seed: int) -> dict[str, Any]:
    def ev(types):
        return lambda x: x["rewrite"] > 0 and x["type"] in types
    spec = _share_metric(per_run, ev(A3_NOT_IDLE), lambda x: x[idle_key] < 0.5, n_boot, seed)
    spec.update(tn=spec["k"], fp=spec["n"] - spec["k"])
    rec = _share_metric(per_run, ev(A3_RECORDED), lambda x: x[idle_key] < 0.5, n_boot, seed)
    rec.update(tn=rec["k"], fp=rec["n"] - rec["k"])
    om = _share_metric(per_run, ev(("MS+P",)), lambda x: x[dom_key] == "model_change", n_boot, seed)
    oc = _share_metric(per_run, ev(("MCP+P",)), lambda x: x[idle_key] >= 0.5, n_boot, seed)
    if not mcp_ok:
        oc = {**oc, "value": None, "excluded": "MCP+P truth ambiguous (manipulation check M2 below 0.90)"}
    conf: dict[str, dict[str, int]] = {}
    for x in (x for run in per_run for x in run):
        key = x[dom_key] if x["rewrite"] else "no_event"
        conf.setdefault(x["type"], {})
        conf[x["type"]][key] = conf[x["type"]].get(key, 0) + 1
    return {"spec": spec, "spec_recorded": rec, "order_model": om, "order_mcp": oc,
            "confusion": {t: dict(sorted(conf[t].items())) for t in A3_TYPES if t in conf}}


def _a3_decision(m: dict[str, Any]) -> str:
    if m.get("excluded"):
        return "excluded (truth ambiguous)"
    if m["value"] is None:
        return "no events"
    return "pass" if m["value"] >= A3_PASS else "fail"


def evaluate_amend3_s(runs: list[dict[str, Any]], prices: dict[str, dict[str, float]], n_boot: int = N_BOOT,
                      seed: int = SEED) -> dict[str, Any]:
    """Specificity arm (Amendment 3): boundary events of valid S1/S2 runs against the injected truth."""
    best, excl = select_valid(runs)
    s_runs = [best[k] for k in sorted(best) if k[1] in ("S1", "S2")]
    per_run, allev = [], {"events": 0, "boundary_events": 0, "other_events": 0, "other_labelled_idle": 0}
    for r in s_runs:
        bds, a = a3_boundaries(r, prices)
        per_run.append(bds)
        for k in allev:
            allev[k] += a[k]
    flat = [x for run in per_run for x in run]
    m1_rows = [x for x in flat if x["type"] in A3_NOT_IDLE]
    m2_rows = [x for x in flat if x["type"] == "MCP"]
    m1k = sum(1 for x in m1_rows if x["prefix_before"] and x["rewrite"] >= M1_REWRITE * x["prefix_before"])
    m2k = sum(1 for x in m2_rows if x["prefix_before"] and x["read"] >= M2_READ * x["prefix_before"])
    m1 = {"k": m1k, "n": len(m1_rows), "value": m1k / len(m1_rows) if m1_rows else None}
    m2 = {"k": m2k, "n": len(m2_rows), "value": m2k / len(m2_rows) if m2_rows else None}
    mcp_ok = m2["value"] is not None and m2["value"] >= M2_PASS
    reg = _a3_metrics(per_run, "idle", "dominant", mcp_ok, n_boot, seed)
    shp = _a3_metrics(per_run, "idle_shape", "dominant_shape", mcp_ok, n_boot, seed)
    by_type = {t: {"boundaries": sum(1 for x in flat if x["type"] == t),
                   "events": sum(1 for x in flat if x["type"] == t and x["rewrite"])} for t in A3_TYPES}
    return {
        "runs": len(s_runs), "excluded": excl, "boundaries_reached": len(flat), "by_type": by_type,
        "manipulation": {"m1": m1, "m2": m2, "mcp_truth_idle": mcp_ok},
        "metrics": {k: reg[k] for k in ("spec", "spec_recorded", "order_model", "order_mcp")},
        "decisions": {k: _a3_decision(reg[k]) for k in ("spec", "spec_recorded", "order_model", "order_mcp")},
        "confusion": reg["confusion"],
        "exploratory_shape_marker": {**{k: shp[k] for k in ("spec", "spec_recorded", "order_model", "order_mcp")},
                                     "confusion": shp["confusion"]},
        "exploratory_all_events": allev,
        "boundaries": flat,
    }


def post_pause_requests(run: dict[str, Any]) -> list[dict[str, Any]]:
    """Per injected pause, the first request whose gap since its transcript's previous request holds it."""
    injected = [p for p in run["pauses"] if p["seconds"] > 0]
    by_turn: dict[Any, list[dict[str, Any]]] = {}
    for r in run["records"]:
        by_turn.setdefault(r["turn"], []).append(r)
    first: dict[int, dict[str, Any]] = {}
    for rr in by_turn.values():
        rr.sort(key=lambda r: r["step"])
        for prev, r in zip(rr, rr[1:]):
            pt, t = parse_ts(prev["ts"]), parse_ts(r["ts"])
            p = pause_in_gap(pt, t, injected)
            if p is not None and (p["call"] not in first or t < first[p["call"]]["t"]):
                pu, u = prev["usage"], r["usage"]
                first[p["call"]] = {"call": p["call"], "t": t, "gap_s": (t - pt).total_seconds(),
                                    "prefix_before": pu["cache_read_input_tokens"] + pu["cache_creation_input_tokens"],
                                    "write_after": u["cache_creation_input_tokens"],
                                    "read_after": u["cache_read_input_tokens"]}
    return [first[c] for c in sorted(first)]


def activity(runs: list[dict[str, Any]], ledger_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every request of every run directory (time, tokens covered = read + write), plus the first-to-last
    request interval of ledger rows that have no run directory (e.g. a pilot run elsewhere)."""
    out: list[dict[str, Any]] = []
    have = {r.get("run_id") for r in runs}
    for r in runs:
        for rec in r["records"]:
            u = rec["usage"]
            out.append({"run_id": r.get("run_id"), "t": parse_ts(rec["ts"]),
                        "tokens": u["cache_read_input_tokens"] + u["cache_creation_input_tokens"]})
    for row in ledger_rows:
        if row.get("run_id") not in have and row.get("first_request") and row.get("last_request"):
            out.append({"run_id": row["run_id"], "start": parse_ts(row["first_request"]),
                        "end": parse_ts(row["last_request"]), "tokens": None})
    return out


def _nearby(acts: list[dict[str, Any]], own: Any, t: dt.datetime, window_s: float,
            min_tokens: int = 0) -> list[float]:
    """Seconds before t of other runs' activity within (t - window_s, t]; an interval counts at its end."""
    lo = t - dt.timedelta(seconds=window_s)
    out = []
    for a in acts:
        if a["run_id"] == own:
            continue
        if "t" in a:
            if lo <= a["t"] < t and a["tokens"] >= min_tokens:
                out.append((t - a["t"]).total_seconds())
        elif a["start"] < t and a["end"] >= lo:
            out.append(max(0.0, (t - min(a["end"], t)).total_seconds()))
    return out


def evaluate_amend3_k(runs: list[dict[str, Any]], other: list[dict[str, Any]]) -> dict[str, Any]:
    """Concurrency arm (Amendment 3): registered full re-write check after each pause of the one-at-a-time
    condition-B sessions; a pause with any other run's request in the 300 s before is excluded."""
    best, excl = select_valid(runs)
    k_runs = [best[k] for k in sorted(best) if k[1] == "K"]
    acts = activity(runs, []) + list(other)
    rows, first = [], []
    for r in k_runs:
        recs = sorted((x for x in r["records"] if x.get("kind", "main") == "main"), key=lambda x: parse_ts(x["ts"]))
        if recs:
            first.append(recs[0]["usage"]["cache_read_input_tokens"])
        for p in post_pause_requests(r):
            near = _nearby(acts, r.get("run_id"), p["t"], K_WINDOW_S)
            rows.append({**direct_expiry_row(p), "session": r.get("instance_id"), "call": p["call"],
                         "valid": not near, "other_activity_s": min(near) if near else None})
    valid = [x for x in rows if x["valid"]]
    k = sum(1 for x in valid if x["full_rewrite"])
    share = k / len(valid) if valid else None
    if share is None:
        decision = "inconclusive"
    elif share >= K_SUPPORTED:
        decision = "supported"
    elif share <= K_NOT_SUPPORTED:
        decision = "not supported"
    else:
        decision = "inconclusive"
    lo, hi = wilson(k, len(valid))
    return {"runs": len(k_runs), "excluded": excl, "pauses": len(rows), "valid_pauses": len(valid),
            "excluded_pauses": len(rows) - len(valid), "full_rewrite_k": k, "full_rewrite_share": share,
            "full_rewrite_wilson_lo": lo, "full_rewrite_wilson_hi": hi, "decision": decision,
            "first_request_read": first, "read_after_values": sorted({x["read_after"] for x in valid}),
            "median_write_over_prefix": direct_expiry_summary(valid)["median_write_over_prefix"], "pauses_detail": rows}


def existing_concurrency_check(runs: list[dict[str, Any]], ledger_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Amendment 3, exploratory, existing data: for each injected pause of the valid B and C runs, whether a
    request of another run covering at least the post-pause read occurred within the cache lifetime before
    the post-pause request; and how many run directories' first request read cached tokens."""
    acts = activity(runs, ledger_rows)
    best, _ = select_valid(runs)
    res: dict[str, dict[str, Any]] = {}
    for key in sorted(best):
        if key[1] not in ("B", "C"):
            continue
        r = best[key]
        w5 = sum(x["usage"]["cache_creation"]["ephemeral_5m_input_tokens"] for x in r["records"])
        w1 = sum(x["usage"]["cache_creation"]["ephemeral_1h_input_tokens"] for x in r["records"])
        life = LIFETIME_S.get(lifetime_tier(w5, w1), LIFETIME_S["5m"])
        c = res.setdefault(key[1], {"pauses": 0, "with_warm_request": 0, "nearest": []})
        for p in post_pause_requests(r):
            near = _nearby(acts, r.get("run_id"), p["t"], life, p["read_after"])
            c["pauses"] += 1
            if near:
                c["with_warm_request"] += 1
                c["nearest"].append(min(near))
    firsts = []
    for r in runs:
        recs = sorted(r["records"], key=lambda x: parse_ts(x["ts"]))
        if recs:
            firsts.append(recs[0]["usage"]["cache_read_input_tokens"])
    return {"label": "exploratory (Amendment 3; existing data, computed before the amendment)",
            "pauses": sum(c["pauses"] for c in res.values()),
            "with_warm_request": sum(c["with_warm_request"] for c in res.values()),
            "by_condition": {k: {"pauses": c["pauses"], "with_warm_request": c["with_warm_request"]} for k, c in res.items()},
            "nearest_s": {k: {"min": min(c["nearest"]) if c["nearest"] else None,
                              "max": max(c["nearest"]) if c["nearest"] else None} for k, c in res.items()},
            "run_directories": len(firsts), "first_request_read_cached": sum(1 for f in firsts if f > 0)}


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
    a3 = agg.get("amendment3", {})
    ex = a3.get("existing_concurrency_check")
    if ex:
        f.update({
            "PauseExistingPauses": figure(ex["pauses"], "int", "runs", "injected pauses of valid B and C runs (existing-data check)"),
            "PauseExistingWarm": figure(ex["with_warm_request"], "int", "runs",
                                        "of those, pauses with another run's request covering the post-pause read within the "
                                        "lifetime before it (exploratory)"),
            "PauseExistingRunDirs": figure(ex["run_directories"], "int", "runs", "run directories of the original runs"),
            "PauseExistingFirstRead": figure(ex["first_request_read_cached"], "int", "runs",
                                             "run directories whose first request read cached tokens (exploratory)"),
        })
    s = a3.get("specificity_arm")
    if s:
        m, sh = s["metrics"], s["exploratory_shape_marker"]
        f.update({
            "PauseSRuns": figure(s["runs"], "int", "runs_a3", "valid specificity-arm runs (Amendment 3)"),
            "PauseSBoundaries": figure(s["boundaries_reached"], "int", "runs_a3", "specificity-arm boundaries reached"),
            "PauseSpecPct": figure(_pct(m["spec"]["value"]), "pct0", "runs_a3",
                                   "S-spec: specificity of the idle label over not-idle boundary events (MS, TL, MS+P, TL+P)"),
            "PauseSpecLoPct": figure(_pct(m["spec"]["wilson_lo"]), "pct0", "runs_a3", "S-spec Wilson 95% lower bound"),
            "PauseSpecHiPct": figure(_pct(m["spec"]["wilson_hi"]), "pct0", "runs_a3", "S-spec Wilson 95% upper bound"),
            "PauseSpecTn": figure(m["spec"]["tn"], "int", "runs_a3", "S-spec true negatives"),
            "PauseSpecFp": figure(m["spec"]["fp"], "int", "runs_a3", "S-spec false positives (labelled idle)"),
            "PauseSpecRecordedPct": figure(_pct(m["spec_recorded"]["value"]), "pct0", "runs_a3",
                                           "S-spec-recorded: specificity over model-switch events (MS, MS+P)"),
            "PauseOrderModelPct": figure(_pct(m["order_model"]["value"]), "pct0", "runs_a3",
                                         "S-order-model: MS+P events whose dominant cause is model_change"),
            "PauseOrderMcpPct": figure(_pct(m["order_mcp"]["value"]), "pct0", "runs_a3",
                                       "S-order-mcp: MCP+P events labelled idle (the true cause)"),
            "PauseSpecDecision": figure(s["decisions"]["spec"], "text", "runs_a3", "S-spec decision (pass mark 0.90)"),
            "PauseSpecRecordedDecision": figure(s["decisions"]["spec_recorded"], "text", "runs_a3", "S-spec-recorded decision"),
            "PauseOrderModelDecision": figure(s["decisions"]["order_model"], "text", "runs_a3", "S-order-model decision"),
            "PauseOrderMcpDecision": figure(s["decisions"]["order_mcp"], "text", "runs_a3", "S-order-mcp decision"),
            "PauseMOnePct": figure(_pct(s["manipulation"]["m1"]["value"]), "pct0", "runs_a3",
                                   "M1: MS/TL/MS+P/TL+P boundaries that re-wrote at least half the cached prefix"),
            "PauseMTwoPct": figure(_pct(s["manipulation"]["m2"]["value"]), "pct0", "runs_a3",
                                   "M2: MCP boundaries that read back at least 95% of the cached prefix"),
            "PauseShapeSpecPct": figure(_pct(sh["spec"]["value"]), "pct0", "runs_a3",
                                        "exploratory: S-spec with the client's request-shape change counted as a config change"),
        })
    k = a3.get("concurrency_arm")
    if k:
        f.update({
            "PauseKRuns": figure(k["runs"], "int", "runs_a3", "valid one-at-a-time condition-B runs (K)"),
            "PauseKValidPauses": figure(k["valid_pauses"], "int", "runs_a3", "K pauses with no other run's request in the 300 s before"),
            "PauseKFullRewritePct": figure(_pct(k["full_rewrite_share"]), "pct0", "runs_a3",
                                           "K-direct: valid K pauses whose next request re-wrote the whole cached prefix"),
            "PauseKDecision": figure(k["decision"], "text", "runs_a3", "K-direct decision on the concurrency explanation"),
        })
    for c, v in sorted(agg.get("resolved", {}).items()):
        f[f"PauseResolved{c}"] = figure(v["resolved"], "int", "grading", f"condition {c}: final diffs resolved (SWE-bench harness)")
        f[f"PauseGraded{c}"] = figure(v["submitted"], "int", "grading", f"condition {c}: final diffs submitted for grading")
    if a3.get("ledger"):
        f["PauseAThreeCostUsd"] = figure(a3["ledger"]["usd_total"], "float2", "ledger_a3",
                                         "API list-price cost of all Amendment 3 runs, USD")
    return {k: v for k, v in f.items() if v["value"] is not None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=EVIDENCE / "pause_experiment.json")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--include-pilot", action="store_true",
                    help="checking the pilot or a dry run only: analyse pilot runs (never for the evidence file)")
    ap.add_argument("--a3-runs", type=Path, default=None, help="Amendment 3 runs directory (S and K arms)")
    ap.add_argument("--a3-ledger", type=Path, default=None, help="Amendment 3 ledger")
    ap.add_argument("--grading", type=Path, default=None,
                    help="summary.json of experiments/pause/grade.py --sessions (registered secondary outcome)")
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
    agg["amendment3"] = {"existing_concurrency_check": existing_concurrency_check(runs, lrows)}
    extra_sources = []
    if args.a3_runs is not None:
        a3runs = load_runs(args.a3_runs)
        l3raw = args.a3_ledger.read_bytes() if args.a3_ledger is not None and args.a3_ledger.exists() else b""
        l3rows = [json.loads(x) for x in l3raw.decode("utf-8").splitlines() if x.strip()]
        s_arm = evaluate_amend3_s(a3runs, prices, n_boot=args.n_boot)
        s_arm.pop("boundaries")                 # per-boundary rows stay out of the evidence file (aggregates only)
        other = activity(runs, lrows) + [a for a in activity(a3runs, l3rows) if "start" in a]
        k_arm = evaluate_amend3_k(a3runs, other)
        k_arm.pop("pauses_detail")
        agg["amendment3"].update({
            "specificity_arm": s_arm, "concurrency_arm": k_arm,
            "ledger": {"rows": len(l3rows), "usd_total": sum(max(float(r.get("usd") or 0), float(r.get("cli_usd") or 0))
                                                              for r in l3rows),
                       "infra_fail_rows": sum(r.get("status") != "ok" for r in l3rows)},
            "run_directories": len(a3runs)})
        h3 = hashlib.sha256()
        for r in a3runs:
            for p in r["inputs"]:
                h3.update(p.relative_to(args.a3_runs).as_posix().encode() + b"\0" + p.read_bytes())
        extra_sources = [
            {"id": "runs_a3", "grade": "measured", "sha256": h3.hexdigest(),
             "command": "python tools/pause_experiment.py ... --a3-runs <runs-a3 dir> --a3-ledger <ledger.jsonl>",
             "description": "Amendment 3 container runs (specificity arm S1/S2, concurrency arm K): transcripts, "
                            "pause and injection logs, run metadata; sha256 as for runs"},
            {"id": "ledger_a3", "grade": "measured", "sha256": sha256_bytes(l3raw),
             "command": "experiments/pause/runner.py run ... --runs-subdir runs-a3", "description": "Amendment 3 cost ledger"}]
    if args.grading is not None:              # registered secondary outcome (no hypothesis): resolved rate
        graw = args.grading.read_bytes().replace(b"\r\n", b"\n")
        agg["resolved"] = {c: {k: v.get(k) for k in ("submitted", "resolved", "completed", "error", "empty_patches")}
                           for c, v in sorted(json.loads(graw.decode("utf-8")).items())}
        extra_sources.append({"id": "grading", "grade": "measured", "sha256": sha256_bytes(graw),
                              "command": "experiments/pause/grade.py --sessions",
                              "description": "official SWE-bench harness reports of the kept final diffs, per condition"})
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
        ] + extra_sources, "notes": [
            "Pre-registered: experiments/PREREG_pause.md. Pilot runs are excluded; failed runs re-run once.",
            "Measured with tools/claude_usage.py and tools/cache_writes.analyse per run (the paper's code).",
            f"Bootstrap: {args.n_boot} resamples of sessions (each chains the registered tasks, Amendment 2), "
            f"seed {SEED}, percentile 95% intervals.",
        ] + ([
            "Amendment 3 (PREREG_pause.md): specificity arm S1/S2 and one-at-a-time arm K under aggregates.amendment3; "
            "Amendment 3 bootstrap intervals resample runs. existing_concurrency_check is exploratory.",
            "Correction to the Amendment 3 text: it states the nearest warm request before a B post-pause read as 2 s "
            "(a rounded figure from the pre-registration computation); the computed value is "
            "aggregates.amendment3.existing_concurrency_check.nearest_s.B.min.",
        ] if args.a3_runs is not None else [])},
        "aggregates": agg,
        "figures": build_figures(agg, dec, cost, len(lrows)),
    })
    print(json.dumps({"paired_tasks": agg["paired_tasks"], "lifetime": agg["lifetime"], "decisions": dec,
                      "rule": {k: agg["rule"][k] for k in ("tp", "fp", "fn", "tn", "precision", "recall")}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
