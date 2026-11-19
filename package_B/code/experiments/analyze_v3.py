"""PREREG_v3 analysis: operating points, H8 (AoB against B*), secondary analyses and the
cross-run cache-isolation check.

    python experiments/analyze_v3.py --input out/h2h3_test --split test --out out/agent_h2h_v3_test.json

Reads runs.jsonl and requests.jsonl of v3 grid dirs (harness "v3"). Cost = mean billed USD per
question (PREREG_v2 Amendment 2, B2; cache writes 1.25x input, cached reads 0.1x input). Two
counterfactual repricings of the same logged usage are reported beside it (section 8): cache-
neutral (every input token at the input rate) and no-write-premium (writes at the input rate).
Every number is computed mechanically from the artifacts; nothing here is tuned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import analyze as az  # noqa: E402  (IO, sources, JSON cleaning)

BOOT_SEED = 7
N_BOOT = 2000
COST_RATIO_MAX = 0.70        # H8 (a), (d)
F1_DIFF_MIN = -0.02          # H8 (b): point estimate >= -2.0 F1 points (inclusive)
F1_CI_LO_MIN = -0.05         # H8 (b): 95% CI lower bound >= -5.0 points (inclusive)
CACHE_MIN_TOKENS = 1024      # provider cache minimum (gpt-6-luna)
EPS = 1e-9                   # inclusive thresholds are compared with this float tolerance
COMPLETE_SHARE = 0.95        # B*-eligible only with runs on >= 95% of the split's questions
PRICE = {"input": 0.10, "cached": 0.01, "write": 0.125, "output": 0.50}  # USD / 1M (B2)
AGENT = "aob"
PUBLISHED = ("react", "notes_reset", "window2", "memgpt", "agentless")  # B*-eligible only
REFERENCES = ("rag1", "closed_book")
ABLATIONS = ("aob_nogate", "aob_nohop", "aob_nopack", "aob_llmq")
SUPERSEDED = ("pbc",)
COSTS = ("cost", "cost_neutral", "cost_nowrite")
NOT_YET = "not yet run"
PREREG_V3 = ROOT / "experiments" / "PREREG_v3.md"
V3_CONFIG = ROOT / "experiments" / "configs" / "h2h_v3.yaml"
V2_DEV = ROOT / "experiments" / "offline" / "v2_dev_round0_run43777353380"
_DIGITS = "Zero One Two Three Four Five Six Seven Eight Nine".split()


# ---------------------------------------------------------------------------------------
# per-run costs
# ---------------------------------------------------------------------------------------

def run_costs(r: dict) -> dict:
    """Billed USD (as logged) and the two counterfactual repricings of the same usage."""
    inp, out = r.get("total_input") or 0, r.get("total_output") or 0
    cached = r.get("total_cached") or 0
    neutral = (inp * PRICE["input"] + out * PRICE["output"]) / 1e6
    nowrite = ((inp - cached) * PRICE["input"] + cached * PRICE["cached"]
               + out * PRICE["output"]) / 1e6
    return {"cost": r.get("usd_list"), "cost_neutral": neutral, "cost_nowrite": nowrite}


def usable(runs, split=None):
    return [r for r in runs if r.get("status") == "ok" and r.get("arm")
            and r.get("harness") == "v3" and (split is None or r.get("split") == split)]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def operating_points(runs):
    groups: dict = defaultdict(list)
    for r in runs:
        groups[(r["arm"], int(r["k"]))].append(r)
    points = []
    for (arm, k), rs in sorted(groups.items()):
        costs = [run_costs(r) for r in rs]
        billed = [c["cost"] for c in costs]
        sup = [r["grounded"] for r in rs if r.get("grounded") is not None]
        inp_sum = sum(r["total_input"] or 0 for r in rs)
        pq = per_question(rs, (arm, k), "cost_neutral")  # repeat-averaged per question
        points.append({
            "arm": arm, "k": k, "n": len(rs),
            "n_questions": len({r["question_id"] for r in rs}),
            "n_repeats": len({r.get("repeat") for r in rs}),
            "cost": None if any(c is None for c in billed) else _mean(billed),
            "cost_neutral": _mean([c["cost_neutral"] for c in costs]),
            "cost_nowrite": _mean([c["cost_nowrite"] for c in costs]),
            # EM and F1: mean over questions of the repeat-averaged per-question values
            "em": _mean([v[2] for v in pq.values()]), "f1": _mean([v[1] for v in pq.values()]),
            "requests": _mean([r.get("n_requests") for r in rs]),
            "input": _mean([r["total_input"] for r in rs]),
            "cached": _mean([r.get("total_cached") for r in rs]),
            "cache_write": _mean([r.get("total_cache_write") for r in rs]),
            "output": _mean([r["total_output"] for r in rs]),
            "latency_s": _mean([r.get("latency_s") for r in rs]),
            "turns": _mean([r.get("turns") for r in rs]),
            # cache hit rate = billed cached-read tokens / all input tokens (provider-reported)
            "cache_hit_rate": (sum(r.get("total_cached") or 0 for r in rs) / inp_sum
                               if inp_sum else None),
            "supported": _mean([float(s) for s in sup]) if sup else None,
            "role": ("published" if arm in PUBLISHED else "reference" if arm in REFERENCES
                     else "agent" if arm == AGENT else "ablation" if arm in ABLATIONS
                     else "superseded" if arm in SUPERSEDED else "other"),
        })
    return points


def frontier(points, cost_key="cost"):
    priced = [p for p in points if p[cost_key] is not None and p["f1"] is not None]

    def dominated(p):
        return any(o[cost_key] <= p[cost_key] and o["f1"] >= p["f1"]
                   and (o[cost_key] < p[cost_key] or o["f1"] > p["f1"]) for o in priced)

    return [p for p in priced if not dominated(p)]


# ---------------------------------------------------------------------------------------
# paired, block-stratified cluster bootstrap
# ---------------------------------------------------------------------------------------

def per_question(runs, cell, cost_key="cost"):
    """{question_id: (mean cost, mean F1, mean EM, block)} over the cell's repeats; the
    question is the resampling unit, so repeats are averaged first (section 8)."""
    acc: dict = defaultdict(list)
    for r in runs:
        if (r["arm"], int(r["k"])) == tuple(cell):
            c = run_costs(r)[cost_key]
            if c is not None:
                acc[r["question_id"]].append((c, r["f1"], r["em"], r.get("block")))
    return {q: (sum(x[0] for x in v) / len(v), sum(x[1] for x in v) / len(v),
                sum(x[2] for x in v) / len(v), v[0][3]) for q, v in acc.items()}


def paired(runs, a, b, cost_key="cost"):
    """Questions answered by both cells: rows (cost_a, cost_b, f1_a, f1_b, em_a, em_b) of
    repeat-averaged values, and the questions' blocks."""
    pa, pb = per_question(runs, a, cost_key), per_question(runs, b, cost_key)
    qids = sorted(set(pa) & set(pb))
    S = np.array([[pa[q][0], pb[q][0], pa[q][1], pb[q][1], pa[q][2], pb[q][2]]
                  for q in qids], dtype=float)
    blocks = [pa[q][3] for q in qids]
    return qids, S.reshape(-1, 6), blocks


def boot_indices(blocks, n_boot=N_BOOT, seed=BOOT_SEED):
    """(n_boot, n) resampled row indices: questions are resampled with replacement within
    each KB block (stratified by block), the same draw for every statistic of a comparison."""
    rng = np.random.default_rng(seed)
    by_block: dict = defaultdict(list)
    for i, b in enumerate(blocks):
        by_block[-1 if b is None else int(b)].append(i)
    parts = []
    for b in sorted(by_block):
        idx = np.array(by_block[b])
        parts.append(idx[rng.integers(0, len(idx), size=(n_boot, len(idx)))])
    return np.concatenate(parts, axis=1)


def _ci(vals):
    vals = np.asarray(vals, dtype=float)
    if not np.all(np.isfinite(vals)):
        return [None, None]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


def compare(runs, a, b, cost_key="cost", n_boot=N_BOOT, seed=BOOT_SEED):
    """cost(a)/cost(b) (ratio of means over paired questions) and F1(a) - F1(b), with 95%
    percentile CIs from the paired block-stratified bootstrap."""
    qids, S, blocks = paired(runs, a, b, cost_key)
    n = len(qids)
    if n < 2:
        return None
    sums = S.sum(axis=0)
    ratio = float(sums[0] / sums[1]) if sums[1] > 0 else math.nan
    diff = float((sums[2] - sums[3]) / n)
    idx = boot_indices(blocks, n_boot, seed)
    T = S[idx]  # (n_boot, n, 4)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_boot = T[:, :, 0].sum(axis=1) / T[:, :, 1].sum(axis=1)
    d_boot = (T[:, :, 2].sum(axis=1) - T[:, :, 3].sum(axis=1)) / n
    e_boot = (T[:, :, 4].sum(axis=1) - T[:, :, 5].sum(axis=1)) / n
    return {"a": list(a), "b": list(b), "n": n, "cost_ratio": ratio,
            "cost_ratio_ci": _ci(r_boot), "f1_diff": diff, "f1_diff_ci": _ci(d_boot),
            "em_diff": float((sums[4] - sums[5]) / n), "em_diff_ci": _ci(e_boot)}


def cell_cis(runs, cell, n_boot=N_BOOT, seed=BOOT_SEED):
    """95% CIs of the cell's mean F1, EM and cost (repeat-averaged per question, block-
    stratified bootstrap)."""
    pq = per_question(runs, cell)
    if len(pq) < 2:
        return None
    qids = sorted(pq)
    S = np.array([[pq[q][0], pq[q][1], pq[q][2]] for q in qids], dtype=float)
    T = S[boot_indices([pq[q][3] for q in qids], n_boot, seed)].mean(axis=1)
    return {"cost_ci": _ci(T[:, 0]), "f1_ci": _ci(T[:, 1]), "em_ci": _ci(T[:, 2])}


def repeat_stats(runs, cell):
    """Variance across repeats (temperature 0 is not deterministic): the SD and range of the
    cell's per-repeat means, and the share of questions whose normalised answer is identical
    in every repeat."""
    from agent.metrics import normalize_answer
    by_rep: dict = defaultdict(list)
    answers: dict = defaultdict(set)
    for r in runs:
        if (r["arm"], int(r["k"])) == tuple(cell):
            by_rep[r.get("repeat")].append(r)
            answers[r["question_id"]].add(normalize_answer(r.get("answer") or ""))
    if len(by_rep) < 2:
        return {"n_repeats": len(by_rep)}
    out = {"n_repeats": len(by_rep)}
    for key, fn in (("f1", lambda r: r["f1"]), ("em", lambda r: r["em"]),
                    ("cost", lambda r: run_costs(r)["cost"])):
        means = [_mean([fn(r) for r in rs]) for _, rs in sorted(by_rep.items())]
        if any(m is None for m in means):
            continue
        out[f"{key}_by_repeat"] = means
        out[f"{key}_sd"] = statistics.stdev(means)
        out[f"{key}_range"] = max(means) - min(means)
    out["answer_agreement"] = sum(len(v) == 1 for v in answers.values()) / len(answers)
    return out


# ---------------------------------------------------------------------------------------
# H8
# ---------------------------------------------------------------------------------------

def select_bstar(points, cost_key="cost"):
    """B*: the published-baseline cell with the highest F1; ties -> lower cost, then name. Only
    cells with runs on at least COMPLETE_SHARE of the questions any cell answered are eligible
    (Amendment 1), so a cell stopped early cannot become B*."""
    n_all = max((p["n_questions"] for p in points), default=0)
    base = [p for p in points if p["arm"] in PUBLISHED and p[cost_key] is not None
            and p["f1"] is not None and p["n_questions"] >= COMPLETE_SHARE * n_all]
    return min(base, key=lambda p: (-p["f1"], p[cost_key], p["arm"], p["k"])) if base else None


def _ratio_ok(c):
    return bool(c and math.isfinite(c["cost_ratio"]) and c["cost_ratio"] <= COST_RATIO_MAX + EPS
                and c["cost_ratio_ci"][1] is not None and c["cost_ratio_ci"][1] < 1.0)


def h8(runs, cost_key="cost", n_boot=N_BOOT):
    """PREREG_v3 section 8. runs: usable runs of one split."""
    points = operating_points(runs)
    agent = [p for p in points if p["arm"] == AGENT]
    if not agent or agent[0][cost_key] is None:
        return {"decision": NOT_YET, "reason": "needs priced aob runs"}
    if len(agent) > 1:
        return {"decision": NOT_YET, "reason": "more than one aob cell"}
    ag = agent[0]
    bstar = select_bstar(points, cost_key)
    if bstar is None:
        return {"decision": NOT_YET, "reason": "needs priced published-baseline runs"}
    cell_a, cell_b = (AGENT, ag["k"]), (bstar["arm"], bstar["k"])
    main = compare(runs, cell_a, cell_b, cost_key, n_boot)
    if main is None:
        return {"decision": NOT_YET, "reason": "fewer than 2 questions paired with B*"}
    rule_a = _ratio_ok(main)
    lo = main["f1_diff_ci"][0]
    rule_b = bool(main["f1_diff"] >= F1_DIFF_MIN - EPS and lo is not None
                  and lo >= F1_CI_LO_MIN - EPS)
    beaten_by = [f"{p['arm']}:{p['k']}" for p in points
                 if p["arm"] in PUBLISHED + REFERENCES and p[cost_key] is not None
                 and p[cost_key] < ag[cost_key] and p["f1"] > ag["f1"]]
    rule_c = not beaten_by
    d_cells, d_fail = [], []
    for p in points:
        if (p["arm"] in PUBLISHED and p[cost_key] is not None
                and p["f1"] >= ag["f1"] + F1_DIFF_MIN - EPS):
            c = compare(runs, cell_a, (p["arm"], p["k"]), cost_key, n_boot)
            d_cells.append(c)
            if not _ratio_ok(c):
                d_fail.append(f"{p['arm']}:{p['k']}")
    rule_d = not d_fail
    unstable = main["cost_ratio_ci"][0] is None or main["f1_diff_ci"][0] is None
    supported = rule_a and rule_b and rule_c and rule_d and not unstable
    return {
        "decision": "supported" if supported else "not supported",
        "reason": "unstable bootstrap CI" if unstable else None,
        "cost_key": cost_key,
        "bstar": {"arm": bstar["arm"], "k": bstar["k"], "f1": bstar["f1"],
                  "cost": bstar[cost_key]},
        "aob": {"k": ag["k"], "f1": ag["f1"], "cost": ag[cost_key]},
        "n_paired_questions": main["n"],
        "cost_ratio": main["cost_ratio"], "cost_ratio_ci": main["cost_ratio_ci"],
        "f1_diff": main["f1_diff"], "f1_diff_ci": main["f1_diff_ci"],
        "rule_a": rule_a, "rule_b": rule_b, "rule_c": rule_c, "rule_d": rule_d,
        "dominated_by": beaten_by, "rule_d_cells": d_cells, "rule_d_failed": d_fail,
    }


# ---------------------------------------------------------------------------------------
# cache isolation, gate statistics, dev-round triggers
# ---------------------------------------------------------------------------------------

def cross_run_cache_reads(reqs, run_ids=None):
    """Provable cross-run cache reads: cached tokens of a request whose own run (same run_id
    and attempt) had no EARLIER request of at least CACHE_MIN_TOKENS prompt tokens, so the
    cached prefix cannot be its own. Registered check: the total is 0 (section 3, H1)."""
    best: dict = {}
    total, by_arm, n_req = 0, defaultdict(int), 0
    for q in reqs:  # file order = execution order within a run attempt
        if run_ids is not None and q.get("run_id") not in run_ids:
            continue
        key = (q.get("run_id"), q.get("attempt"))
        cached = q.get("cached_input_tokens") or 0
        if cached > 0 and best.get(key, 0) < CACHE_MIN_TOKENS:
            total += cached
            by_arm[q.get("arm") or q.get("policy")] += cached
            n_req += 1
        best[key] = max(best.get(key, 0), q.get("input_tokens") or 0)
    return {"cross_run_cached_tokens": total, "n_requests": n_req, "by_arm": dict(by_arm),
            "ok": total == 0}


def gate_stats(runs, reqs, arm=AGENT):
    ids = {r["run_id"] for r in runs if r["arm"] == arm}
    rq = [q for q in reqs if q.get("run_id") in ids]
    ans = [q for q in rq if q.get("aob_request")]
    n_answer = defaultdict(int)
    for q in ans:
        n_answer[q["run_id"]] += 1
    counts = [n_answer[i] for i in ids]
    sup = [r for r in runs if r["arm"] == arm and r.get("grounded") is not None]
    tokens = [q.get("input_tokens") or 0 for q in rq]
    return {
        "arm": arm, "n_runs": len(ids),
        "answer_requests_per_run": _mean(counts),
        "share_committed_at": {str(i): (sum(c == i for c in counts) / len(counts)
                                        if counts else None) for i in (1, 2, 3)},
        "supported_share": _mean([float(r["grounded"]) for r in sup]),
        "f1_supported": _mean([r["f1"] for r in sup if r["grounded"]]),
        "f1_unsupported": _mean([r["f1"] for r in sup if not r["grounded"]]),
        "n_requests": len(rq),
        "share_requests_at_or_over_cache_min": (sum(t >= CACHE_MIN_TOKENS for t in tokens)
                                                / len(tokens) if tokens else None),
        "median_prompt_tokens": statistics.median(tokens) if tokens else None,
        "max_prompt_tokens": max(tokens) if tokens else None,
    }


def dev_triggers(h, gs):
    """PREREG_v3 section 6 dev-round triggers (dev split only)."""
    gap = None if h.get("f1_diff") is None else h["f1_diff"]
    return {
        "f1_gap_vs_bstar_dev": gap,
        "round1_thought_triggered": gap is not None and gap < F1_DIFF_MIN - EPS,
        "share_over_1023": gs.get("share_requests_at_or_over_cache_min"),
        "median_prompt_tokens": gs.get("median_prompt_tokens"),
        "budget_down_triggered": (gs.get("share_requests_at_or_over_cache_min") or 0) > 0.10,
        "budget_up_triggered": (gs.get("median_prompt_tokens") or 1e9) < 750,
    }


# ---------------------------------------------------------------------------------------
# evidence envelope
# ---------------------------------------------------------------------------------------

def _words(k):
    return "".join(_DIGITS[int(c)] if c.isdigit() else c for c in str(k))


def _camel(arm):
    return "".join(p.capitalize() for p in _words(arm).split("_"))


def build_figures(points, hyp, src):
    figs = {}

    def fig(name, value, fmt, desc):
        assert name.isalpha() and name.startswith("Hthree") and len(name) <= 60, name
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            figs[name] = {"value": NOT_YET, "format": "text", "source": src, "desc": desc}
        else:
            figs[name] = {"value": value, "format": fmt, "source": src, "desc": desc}

    ci_r = hyp.get("cost_ratio_ci") or [None, None]
    ci_d = hyp.get("f1_diff_ci") or [None, None]
    fig("HthreeDecision", hyp["decision"], "text", "H8 decision (PREREG_v3 section 8).")
    fig("HthreeCostRatio", hyp.get("cost_ratio"), "float3", "H8 cost(aob) / cost(B*).")
    fig("HthreeCostRatioLo", ci_r[0], "float3", "H8 cost ratio 95% CI lower bound.")
    fig("HthreeCostRatioHi", ci_r[1], "float3", "H8 cost ratio 95% CI upper bound.")
    fig("HthreeFOneDiff", None if hyp.get("f1_diff") is None else 100 * hyp["f1_diff"],
        "signedfloat2", "H8 F1(aob) - F1(B*) in points.")
    fig("HthreeFOneDiffLo", None if ci_d[0] is None else 100 * ci_d[0], "signedfloat2",
        "H8 F1 difference 95% CI lower bound, points.")
    fig("HthreeFOneDiffHi", None if ci_d[1] is None else 100 * ci_d[1], "signedfloat2",
        "H8 F1 difference 95% CI upper bound, points.")
    for rule in "abcd":
        v = hyp.get(f"rule_{rule}")
        fig(f"HthreeRule{rule.upper()}", None if v is None else str(v), "text",
            f"H8 rule ({rule}).")
    bstar = hyp.get("bstar") or {}
    fig("HthreeBstarArm", bstar.get("arm"), "text", "B*: highest-F1 published baseline.")
    fig("HthreeBstarK", bstar.get("k"), "int", "Retrieval k of B*.")
    fig("HthreePairedQuestions", hyp.get("n_paired_questions"), "int",
        "Questions answered by both aob and B*.")
    on_front = {(p["arm"], p["k"]) for p in frontier(points)}
    for p in points:
        tag = f"Hthree{_camel(p['arm'])}K{_words(p['k'])}"
        fig(tag + "Cost", None if p["cost"] is None else 1000 * p["cost"], "float3",
            f"{p['arm']} k={p['k']} billed USD per 1000 questions.")
        fig(tag + "CostNeutral", 1000 * p["cost_neutral"], "float3",
            f"{p['arm']} k={p['k']} cache-neutral USD per 1000 questions.")
        fig(tag + "FOne", None if p["f1"] is None else 100 * p["f1"], "float1",
            f"{p['arm']} k={p['k']} F1 in points.")
        fig(tag + "Frontier", str((p["arm"], p["k"]) in on_front), "text",
            f"{p['arm']} k={p['k']} is on the billed cost-F1 frontier.")
    return figs


def design_figures(prereg_src, v2_src):
    """Registered design constants (documented record: PREREG_v3 and its config) and the
    superseded v2 dev-round-0 facts recomputed from the committed v2 artifact (measured)."""
    import yaml
    with open(V3_CONFIG, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    from agent.data import split_positions
    aob = cfg["aob"]
    figs = {}

    def fig(name, value, fmt, desc, src=prereg_src):
        assert name.isalpha() and name.startswith("Hthree") and len(name) <= 60, name
        figs[name] = {"value": value, "format": fmt, "source": src, "desc": desc}

    m = cfg["models"][0]
    fig("HthreeModel", f"OpenAI {m['model']}", "text", "Model of every v3 arm.")
    fig("HthreeNTest", len(split_positions(cfg["dataset"], "test")), "int", "Test questions.")
    fig("HthreeNDev", len(split_positions(cfg["dataset"], "dev")), "int", "Dev questions.")
    fig("HthreeBlockSize", len(range(*[cfg["dataset"]["blocks"]["test"][0][0],
                                         cfg["dataset"]["blocks"]["test"][0][1] + 1])),
        "int", "Questions per knowledge-base block.")
    fig("HthreeRepeats", int(cfg["agent"]["repeats"]["test"]), "int",
        "Independent repeats of every (cell, question) on test.")
    fig("HthreeReactSteps", int(cfg["react"]["max_steps"]), "int", "ReAct step cap (source).")
    fig("HthreeTurnCap", int(cfg["agent"]["max_steps"]), "int", "Harness step cap.")
    fig("HthreeKOne", int(aob["k1"]), "int", "AoB first-search depth k1.")
    fig("HthreeMTwo", int(aob["m2"]), "int", "AoB PRF-hop chunks m2.")
    fig("HthreeRMax", int(aob["r_max"]), "int", "AoB maximum answer requests.")
    fig("HthreeBudgetChars", int(aob["budget"]), "int", "AoB request character budget.")
    fig("HthreeCacheMin", CACHE_MIN_TOKENS, "int", "Provider cache minimum, tokens.")
    fig("HthreeRatioMax", COST_RATIO_MAX, "float2", "H8 cost-ratio threshold.")
    fig("HthreeFOneMargin", -100 * F1_DIFF_MIN, "float1", "H8 F1 non-inferiority margin, points.")
    fig("HthreeFOneCiMargin", -100 * F1_CI_LO_MIN, "float1",
        "H8 F1 CI lower-bound margin, points.")
    fig("HthreeNBoot", N_BOOT, "int", "Bootstrap replicates.")
    reqs = az._read_jsonl(str(V2_DEV / "requests.jsonl"))
    runs = az._read_jsonl(str(V2_DEV / "runs.jsonl"))
    if reqs and runs and v2_src:
        for r in reqs:
            r.setdefault("arm", r.get("policy"))
        iso = cross_run_cache_reads(reqs)
        cached = sum(q.get("cached_input_tokens") or 0 for q in reqs)
        fig("HthreeVTwoCachedTokens", cached, "int", "v2 dev round 0 cached tokens.", v2_src)
        fig("HthreeVTwoCrossTokens", iso["cross_run_cached_tokens"], "int",
            "v2 dev round 0 cached tokens provably read across runs.", v2_src)
        fig("HthreeVTwoCrossShare", 100 * iso["cross_run_cached_tokens"] / cached, "pct1",
            "Share of v2 dev cached tokens provably read across runs.", v2_src)
        from experiments import analyze_v2 as a2
        h7 = a2.h7(a2.usable(runs, "dev"))
        if h7.get("decision") != a2.NOT_YET:
            fig("HthreeVTwoPbcRatio", h7["cost_ratio"], "float2",
                "Superseded v2 pbc: dev cost ratio to the highest-F1 baseline.", v2_src)
            fig("HthreeVTwoPbcRatioLo", h7["cost_ratio_ci"][0], "float2", "Its CI low.", v2_src)
            fig("HthreeVTwoPbcRatioHi", h7["cost_ratio_ci"][1], "float2", "Its CI high.", v2_src)
            fig("HthreeVTwoPbcFOneDiff", 100 * h7["f1_diff"], "signedfloat2",
                "Superseded v2 pbc: dev F1 difference, points.", v2_src)
            fig("HthreeVTwoPbcFOneDiffLo", 100 * h7["f1_diff_ci"][0], "signedfloat2",
                "Its CI low.", v2_src)
            fig("HthreeVTwoPbcFOneDiffHi", 100 * h7["f1_diff_ci"][1], "signedfloat2",
                "Its CI high.", v2_src)
    return figs


def _sources(dirs):
    sources, _ = az._build_sources(dirs)
    for s in sources:
        s["sha256"] = s["sha256"] or hashlib.sha256(b"").hexdigest()
    sha = az._git(["log", "--diff-filter=A", "--format=%H", "--", "experiments/PREREG_v3.md"])
    sha = (sha or "").splitlines()[-1] if sha else os.environ.get("PREREG_V3_SHA")
    sources[-1] = {
        "id": "prereg", "grade": "documented-record",
        "sha256": az._sha256_file(str(PREREG_V3)) or hashlib.sha256(b"").hexdigest(),
        "command": "experiments/PREREG_v3.md",
        "description": "Pre-registration experiments/PREREG_v3.md, section 8 specifies H8"
                       + (f"; registration commit {sha}" if sha else ""),
    }
    if (V2_DEV / "runs.jsonl").is_file():
        sources.append({
            "id": "v2dev", "grade": "measured",
            "sha256": az._sha256_file(str(V2_DEV / "requests.jsonl")),
            "command": "experiments/offline/v2_dev_round0_run43777353380/requests.jsonl",
            "description": "Superseded v2 dev round 0 (Actions run 43777353380), committed copy; "
                           "DECISIONS.md D31 and D33"})
    return sources


def run_evidence(input_dirs, split="test", n_boot=N_BOOT):
    runs, reqs, not_run = [], [], []
    for d in input_dirs:
        runs += az._read_jsonl(os.path.join(d, "runs.jsonl"))
        reqs += az._read_jsonl(os.path.join(d, "requests.jsonl"))
        not_run += az._read_jsonl(os.path.join(d, "not_run.jsonl"))
    rs = usable(runs, split)
    ids = {r["run_id"] for r in rs}
    points = operating_points(rs)
    iso = cross_run_cache_reads(reqs, ids)
    primary_key = "cost" if iso["ok"] else "cost_neutral"  # registered fallback (section 8)
    hyp = h8(rs, primary_key, n_boot)
    hyp["cache_isolation_ok"] = iso["ok"]
    secondary = {
        "h8_cost_neutral": h8(rs, "cost_neutral", n_boot),
        "h8_cost_nowrite": h8(rs, "cost_nowrite", n_boot),
        "ablations": {}, "references": {},
        "gate": gate_stats(rs, reqs),
        "gate_ablations": {a: gate_stats(rs, reqs, a) for a in ABLATIONS},
        "frontier": [[p["arm"], p["k"]] for p in frontier(points)],
        "frontier_cost_neutral": [[p["arm"], p["k"]] for p in frontier(points, "cost_neutral")],
        "cache_isolation": iso,
    }
    ag = [p for p in points if p["arm"] == AGENT]
    if ag:
        for p in points:
            target = (secondary["ablations"] if p["arm"] in ABLATIONS
                      else secondary["references"] if p["arm"] in REFERENCES + SUPERSEDED
                      else None)
            if target is not None:
                target[f"{p['arm']}:{p['k']}"] = compare(rs, (AGENT, ag[0]["k"]),
                                                         (p["arm"], p["k"]), "cost", n_boot)
    for p in points:  # per-arm table: CIs, repeat variance, paired difference to aob
        cell_p = (p["arm"], p["k"])
        p["cis"] = cell_cis(rs, cell_p, n_boot)
        p["repeats"] = repeat_stats(rs, cell_p)
        p["vs_aob"] = (compare(rs, cell_p, (AGENT, ag[0]["k"]), "cost", n_boot)
                       if ag and p["arm"] != AGENT else None)
    if split == "dev":
        secondary["dev_triggers"] = dev_triggers(hyp, secondary["gate"])
    sources = _sources(input_dirs)
    return {
        "schema_version": 1,
        "kind": "agent_h2h_v3",
        "provenance": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "generator": "experiments/analyze_v3.py",
            "notes": [
                "Cost = mean billed USD per question (PREREG_v2 Amendment 2, B2); cache-neutral "
                "and no-write-premium repricings of the same usage are secondary.",
                "H8: paired cluster bootstrap over questions stratified by KB block, 2000 "
                "replicates, seed 7, percentile 95% CIs; B* = highest-F1 published baseline.",
                f"Analysed split: {split}.",
            ],
            "sources": sources,
        },
        "aggregates": {"split": split, "n_runs": len(rs), "points": points, "H8": hyp,
                       "secondary": secondary,
                       "not_run": [{"arm": r.get("policy"), "k": r.get("k"),
                                    "reason": r.get("reason")} for r in not_run]},
        "figures": {**design_figures("prereg", "v2dev" if any(
            x["id"] == "v2dev" for x in sources) else None),
                    **build_figures(points, hyp, sources[0]["id"])},
    }


def dev_table(points) -> str:
    """Markdown table in the DECISIONS.md layout (token columns: totals; USD per question)."""
    head = ("| arm | k | n | EM | F1 | requests | input | cached | cache write | output | "
            "USD/q | neutral USD/q | cache hit | latency s |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for p in points:
        n = p["n"]
        billed = "n/a" if p["cost"] is None else format(p["cost"], ".6f")
        rows.append(
            f"| {p['arm']} | {p['k']} | {n} | {p['em']:.3f} | {p['f1']:.3f} | "
            f"{round(p['requests'] * n)} | {round(p['input'] * n):,} | "
            f"{round((p['cached'] or 0) * n):,} | {round((p['cache_write'] or 0) * n):,} | "
            f"{round(p['output'] * n):,} | {billed} | {p['cost_neutral']:.6f} | "
            f"{(p['cache_hit_rate'] or 0):.3f} | {(p['latency_s'] or 0):.2f} |")
    return "\n".join([head, *rows])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="experiments/analyze_v3.py",
                                description="PREREG_v3 head-to-head analysis.")
    p.add_argument("--input", action="append", required=True, help="v3 grid dir; repeatable")
    p.add_argument("--out", required=True, help="output evidence JSON path")
    p.add_argument("--split", default="test", choices=["dev", "test"])
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--table", default=None, help="also write the markdown table here")
    args = p.parse_args(argv)
    ev = run_evidence(args.input, args.split, args.n_boot)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(az._clean(ev), indent=2) + "\n")
    if args.table:
        with open(args.table, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(dev_table(ev["aggregates"]["points"]) + "\n")
    agg = ev["aggregates"]
    print(f"wrote {args.out} (split={args.split} runs={agg['n_runs']} "
          f"H8={agg['H8']['decision']} cache_isolation_ok="
          f"{agg['secondary']['cache_isolation']['ok']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
