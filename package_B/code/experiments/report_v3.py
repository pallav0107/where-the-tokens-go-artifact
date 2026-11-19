"""PREREG_v3 test reporting figures (post-test; descriptive, outside the registered decision).

    python experiments/report_v3.py --input DIR_S0 --input DIR_S1 --checks POSTTEST.json \
        --out evidence/agent_h2h_v3_report.json

The registered decision is computed by experiments/analyze_v3.py (evidence/agent_h2h_v3.json,
figures "Hthree..."). This script reuses its functions, unchanged, for the per-arm table the paper
prints (EM and F1 with CIs, AoB's paired differences, tokens, cache hit rate, latency, timed-out
attempts), the billed-cost H8 (secondary when the cache-isolation check fails), the timeout
sensitivity analysis and AoB's request-size share, with figures prefixed "Vthree".
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import analyze as az  # noqa: E402
from experiments import analyze_v3 as av  # noqa: E402

_D = "Zero One Two Three Four Five Six Seven Eight Nine".split()


def _w(k):
    return "".join(_D[int(c)] if c.isdigit() else c for c in str(k))


def _camel(arm):
    return "".join(p.capitalize() for p in _w(arm).split("_"))


def load(dirs):
    runs, reqs = [], []
    for d in dirs:
        last = {}
        for r in az._read_jsonl(os.path.join(d, "runs.jsonl")):
            last[r["run_id"]] = r
        runs += list(last.values())
        reqs += az._read_jsonl(os.path.join(d, "requests.jsonl"))
    return av.usable(runs, "test"), reqs


def build(dirs, checks):
    rs, reqs = load(dirs)
    figs = {}
    src = "testruns"

    def fig(name, value, fmt, desc):
        assert name.isalpha() and name.startswith("Vthree") and len(name) <= 60, name
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            figs[name] = {"value": av.NOT_YET, "format": "text", "source": src, "desc": desc}
        else:
            figs[name] = {"value": value, "format": fmt, "source": src, "desc": desc}

    points = av.operating_points(rs)
    agent = [p for p in points if p["arm"] == av.AGENT][0]
    cell_of = {r["run_id"]: (r["arm"], int(r["k"])) for r in rs}
    timeouts: dict = {}
    for q in reqs:  # timed-out (unconfirmed) attempts per cell, from the request records
        c = cell_of.get(q.get("run_id"))
        if c is not None:
            timeouts[c] = timeouts.get(c, 0) + int(q.get("unconfirmed_attempts") or 0)
    for p in points:
        cell = (p["arm"], p["k"])
        t = f"Vthree{_camel(p['arm'])}K{_w(p['k'])}"
        ci = av.cell_cis(rs, cell)
        fig(t + "FOne", 100 * p["f1"], "float1", f"{cell} F1, points.")
        fig(t + "FOneLo", 100 * ci["f1_ci"][0], "float1", f"{cell} F1 95% CI low.")
        fig(t + "FOneHi", 100 * ci["f1_ci"][1], "float1", f"{cell} F1 95% CI high.")
        fig(t + "EM", 100 * p["em"], "float1", f"{cell} EM, points.")
        fig(t + "Billed", 1000 * p["cost"], "float3", f"{cell} billed USD per 1000 questions.")
        fig(t + "Neutral", 1000 * p["cost_neutral"], "float3",
            f"{cell} cache-neutral USD per 1000 questions.")
        fig(t + "Req", p["requests"], "float2", f"{cell} requests per question.")
        fig(t + "InTok", p["input"], "int", f"{cell} input tokens per question.")
        fig(t + "OutTok", p["output"], "int", f"{cell} output tokens per question.")
        fig(t + "Hit", 100 * (p["cache_hit_rate"] or 0.0), "pct1", f"{cell} cache hit rate.")
        fig(t + "Lat", p["latency_s"], "float1", f"{cell} latency per question, s.")
        fig(t + "Timeouts", int(timeouts.get(cell, 0)), "int",
            f"{cell} timed-out (unconfirmed) attempts.")
        if p["arm"] != av.AGENT:
            c = av.compare(rs, (av.AGENT, agent["k"]), cell, "cost")
            fig(t + "AobRatio", c["cost_ratio"], "float2", f"AoB billed cost / {cell}.")
            fig(t + "AobRatioLo", c["cost_ratio_ci"][0], "float2", "Its CI low.")
            fig(t + "AobRatioHi", c["cost_ratio_ci"][1], "float2", "Its CI high.")
            fig(t + "AobDiff", 100 * c["f1_diff"], "signedfloat2", f"AoB F1 - {cell} F1, points.")
            fig(t + "AobDiffLo", 100 * c["f1_diff_ci"][0], "signedfloat2", "Its CI low.")
            fig(t + "AobDiffHi", 100 * c["f1_diff_ci"][1], "signedfloat2", "Its CI high.")
            # rev3 (17 Nov 2026, reporting only): the same paired comparison on cache-neutral USD,
            # the primary cost once the registered cache-isolation check failed (PREREG_v3 section 8)
            n = av.compare(rs, (av.AGENT, agent["k"]), cell, "cost_neutral")
            fig(t + "AobNRatio", n["cost_ratio"], "float2", f"AoB cache-neutral cost / {cell}.")
            fig(t + "AobNRatioLo", n["cost_ratio_ci"][0], "float2", "Its CI low.")
            fig(t + "AobNRatioHi", n["cost_ratio_ci"][1], "float2", "Its CI high.")
    hb = av.h8(rs, "cost")
    fig("VthreeBilledDecision", hb["decision"], "text", "H8 on billed USD (secondary).")
    fig("VthreeBilledRatio", hb["cost_ratio"], "float3", "Billed cost(aob)/cost(B*).")
    fig("VthreeBilledRatioLo", hb["cost_ratio_ci"][0], "float3", "Its CI low.")
    fig("VthreeBilledRatioHi", hb["cost_ratio_ci"][1], "float3", "Its CI high.")
    sens = checks["sensitivity_timeouts_at_retry_cost"]
    fig("VthreeSensRatio", sens["cost_ratio"], "float3",
        "Billed ratio with each timed-out attempt priced at its retry's cost.")
    fig("VthreeSensRatioLo", sens["cost_ratio_ci"][0], "float3", "Its CI low.")
    fig("VthreeSensRatioHi", sens["cost_ratio_ci"][1], "float3", "Its CI high.")
    fig("VthreeSensDecision", sens["decision"], "text", "H8 decision in the sensitivity analysis.")
    g = checks["gate_without_unrecorded"]
    fig("VthreeAobOverMin", 100 * g["share_requests_at_or_over_cache_min"], "pct1",
        "Share of AoB requests at or above 1,024 prompt tokens.")
    fig("VthreeAobMaxPrompt", int(g["max_prompt_tokens"]), "int", "Largest AoB prompt, tokens.")
    fig("VthreeAobReqPerRun", g["answer_requests_per_run"], "float2", "AoB answer requests per run.")
    fig("VthreeAobSupported", 100 * g["supported_share"], "pct1", "AoB answers supported by the gate.")
    iso = av.cross_run_cache_reads(reqs, {r["run_id"] for r in rs})
    fig("VthreeIsoTokens", iso["cross_run_cached_tokens"], "int",
        "Tokens flagged by the registered cache check (same-request retries, D51).")
    fig("VthreeIsoRequests", iso["n_requests"], "int", "Requests flagged by it.")
    fig("VthreeTimeoutsTotal", int(sum(timeouts.values())), "int", "Timed-out attempts, all arms.")
    fig("VthreeRuns", len(rs), "int", "Test runs.")
    fig("VthreeCiLevel", 95, "pct0", "Confidence level of every interval (percentile bootstrap).")
    # PREREG_v3 section 8 reporting items: repeat variance and the frontiers
    sds, agree = [], []
    for p in points:
        st = av.repeat_stats(rs, (p["arm"], p["k"]))
        if "f1_sd" in st:
            sds.append(100 * st["f1_sd"])
            agree.append(100 * st["answer_agreement"])
    fig("VthreeRepSdMax", max(sds), "float1",
        "Largest SD of a cell's three per-repeat F1 means, points.")
    fig("VthreeAgreeMin", min(agree), "pct0",
        "Smallest share of questions with the same normalised answer in all repeats.")
    fig("VthreeAgreeMax", max(agree), "pct0", "Largest such share.")
    names = {"aob": "AoB", "aob_nogate": "AoB without the gate", "aob_nopack":
             "AoB without packing", "aob_nohop": "AoB without the hop", "aob_llmq":
             "AoB with an LLM query", "agentless": "Agentless", "rag1": "the one-request reference",
             "closed_book": "closed book", "react": "ReAct", "notes_reset": "the notes agent",
             "window2": "the sliding window", "memgpt": "MemGPT", "pbc": "the superseded controller"}
    fr = av.frontier(points, "cost")
    fig("VthreeFrontierBilled", ", ".join(names.get(p["arm"], p["arm"]) for p in fr), "text",
        "Cells on the billed cost-F1 frontier.")
    frn = av.frontier(points, "cost_neutral")
    fig("VthreeFrontierSame", str([(p["arm"], p["k"]) for p in frn] == [(p["arm"], p["k"])
                                                                         for p in fr]), "text",
        "Whether the cache-neutral frontier equals the billed one.")
    # EXPLORATORY (post hoc, not pre-registered): the no-packing ablation against B*
    bstar = av.select_bstar(points, "cost_neutral")
    cn = av.compare(rs, ("aob_nopack", agent["k"]), (bstar["arm"], bstar["k"]), "cost_neutral")
    cb = av.compare(rs, ("aob_nopack", agent["k"]), (bstar["arm"], bstar["k"]), "cost")
    fig("VthreeExplNopackDiff", 100 * cn["f1_diff"], "signedfloat2",
        "EXPLORATORY: F1(aob_nopack) - F1(B*), points.")
    fig("VthreeExplNopackDiffLo", 100 * cn["f1_diff_ci"][0], "signedfloat2", "Its CI low.")
    fig("VthreeExplNopackDiffHi", 100 * cn["f1_diff_ci"][1], "signedfloat2", "Its CI high.")
    fig("VthreeExplNopackNeutral", cn["cost_ratio"], "float2",
        "EXPLORATORY: cache-neutral cost(aob_nopack)/cost(B*).")
    fig("VthreeExplNopackBilled", cb["cost_ratio"], "float2",
        "EXPLORATORY: billed cost(aob_nopack)/cost(B*).")
    return figs, hb, sens


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input", action="append", required=True)
    p.add_argument("--checks", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    checks = json.load(open(a.checks, encoding="utf-8"))
    figs, hb, sens = build(a.input, checks)
    sources, _ = az._build_sources(a.input)
    sources = [s for s in sources if s["id"] != "prereg"]
    shas = sorted(s["sha256"] for s in sources)
    sources = [{"id": "testruns", "grade": "measured",
                "sha256": hashlib.sha256("".join(shas).encode()).hexdigest(),
                "command": "python experiments/report_v3.py --input <test shard dirs>",
                "description": "PREREG_v3 test runs: " + "; ".join(s["description"] for s in sources)},
               {"id": "posttest", "grade": "measured",
                "sha256": az._sha256_file(a.checks),
                "command": "python experiments/offline/posttest_v3_checks.py",
                "description": "Post-test checklist output (DECISIONS.md D52)"}]
    ev = {"schema_version": 1, "kind": "agent_h2h_v3_report",
          "provenance": {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "generator": "experiments/report_v3.py",
                         "notes": ["Descriptive post-test figures; the registered H8 decision is in "
                                   "agent_h2h_v3.json (analyze_v3.py)."],
                         "sources": sources},
          "aggregates": {"billed_h8": hb, "sensitivity": sens},
          "figures": figs}
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(az._clean(ev), indent=2, default=str) + "\n")
    print(f"wrote {a.out}: {len(figs)} figures")


if __name__ == "__main__":
    sys.exit(main())
