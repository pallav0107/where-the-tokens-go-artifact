"""PREREG_v2 analysis: cost-F1 operating points per arm, the empirical frontier and H7.

    python experiments/analyze_v2.py --input out/h2h_test --out evidence/agent_h2h.json

Reads runs.jsonl (and not_run.jsonl, requests.jsonl for the returned model ids) of h2h grid
dirs; writes the evidence envelope with figures prefixed "Hth". Cost = mean USD per question
at the cited list prices (`usd_list`); with OpenAI gpt-6-luna (PREREG_v2 Amendment 2) that is
the billed amount, cache writes priced separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import analyze as az  # noqa: E402  (reuse bootstrap, IO and sources)

BOOT_SEED = 7
N_BOOT = 2000
COST_RATIO_MAX = 0.70       # H7 (a)
F1_DIFF_MIN = -0.02         # H7 (b): -2.0 F1 points
NOT_YET = "not yet run"
PREREG_V2 = ROOT / "experiments" / "PREREG_v2.md"
H2H_CONFIG = ROOT / "experiments" / "configs" / "h2h.yaml"
_PROVIDER_LABEL = {"openai": "OpenAI", "gemini": "Gemini", "dashscope": "Qwen (DashScope)"}
_DIGITS = "Zero One Two Three Four Five Six Seven Eight Nine".split()


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _mean_all(xs):
    """Mean, or None when any value is missing (never a partial mean)."""
    return None if (not xs or any(x is None for x in xs)) else sum(xs) / len(xs)


def registered_model(path=H2H_CONFIG):
    """The model every arm is registered to use (PREREG_v2 Amendment 2, B1)."""
    with open(path, encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)["models"][0]
    return {"provider": spec["provider"], "model": spec["model"],
            "label": f"{_PROVIDER_LABEL.get(spec['provider'], spec['provider'])} "
                     f"{spec['model']}"}


def returned_models(reqs, run_ids):
    """Model ids the provider returned (`response_model`) in the analysed runs' requests."""
    return sorted({str(q["response_model"]) for q in reqs
                   if q.get("run_id") in run_ids and q.get("response_model")})


def usable(runs, split=None):
    """Ok h2h runs (those carrying an arm), optionally of one split."""
    return [r for r in runs if r.get("status") == "ok" and r.get("arm")
            and (split is None or r.get("split") == split)]


def operating_points(runs):
    """One dict per (arm, k): n, mean cost and tokens, EM, F1, grounded rate."""
    groups: dict = {}
    for r in runs:
        groups.setdefault((r["arm"], int(r["k"])), []).append(r)
    points = []
    for (arm, k), rs in sorted(groups.items()):
        cached = [r.get("total_cached") for r in rs]
        costs = [r.get("usd_list") for r in rs]
        grounded = [r["grounded"] for r in rs if r.get("grounded") is not None]
        points.append({
            "arm": arm, "k": k, "n": len(rs),
            "cost": None if any(c is None for c in costs) else _mean(costs),
            "input": _mean([r["total_input"] for r in rs]),
            "cached": None if any(c is None for c in cached) else _mean(cached),
            "output": _mean([r["total_output"] for r in rs]),
            # PREREG_v2 Amendment 2, B5: descriptive only (None when not reported)
            "reasoning": _mean_all([r.get("total_reasoning") for r in rs]),
            "cache_write": _mean_all([r.get("total_cache_write") for r in rs]),
            "turns": _mean([r["turns"] for r in rs]),
            "requests": _mean([r.get("n_requests") for r in rs]),
            "em": _mean([r["em"] for r in rs]),
            "f1": _mean([r["f1"] for r in rs]),
            "grounded": _mean([float(g) for g in grounded]) if grounded else None,
        })
    return points


def frontier(points):
    """Points no other point dominates (dominates = cost <= and F1 >=, one strictly)."""
    priced = [p for p in points if p["cost"] is not None and p["f1"] is not None]

    def dominated(p):
        return any(o["cost"] <= p["cost"] and o["f1"] >= p["f1"]
                   and (o["cost"] < p["cost"] or o["f1"] > p["f1"]) for o in priced)

    return [p for p in priced if not dominated(p)]


def _not_yet(reason, **extra):
    return {"decision": NOT_YET, "reason": reason, **extra}


def h7(runs, n_boot=N_BOOT, seed=BOOT_SEED):
    """PREREG_v2 section 6. runs: usable runs of one split."""
    points = operating_points(runs)
    pbc = [p for p in points if p["arm"] == "pbc"]
    base = [p for p in points if p["arm"] != "pbc" and p["cost"] is not None
            and p["f1"] is not None]
    if not pbc or not base:
        return _not_yet("needs priced pbc and baseline runs")
    if len(pbc) > 1:
        return _not_yet("more than one pbc operating point in the input")
    pbc = pbc[0]
    if pbc["cost"] is None:
        return _not_yet("pbc runs carry no list-price cost")
    # B*: highest F1, ties -> lower cost, then arm/k for determinism
    bstar = min(base, key=lambda p: (-p["f1"], p["cost"], p["arm"], p["k"]))

    def per_q(arm, k):
        return {r["question_id"]: r for r in runs if r["arm"] == arm and int(r["k"]) == k
                and r.get("usd_list") is not None}

    a, b = per_q("pbc", pbc["k"]), per_q(bstar["arm"], bstar["k"])
    qids = sorted(set(a) & set(b))
    if len(qids) < 2:
        return _not_yet("fewer than 2 questions answered by both pbc and B*")
    S = np.array([[a[q]["usd_list"], b[q]["usd_list"], a[q]["f1"], b[q]["f1"]]
                  for q in qids], dtype=float)
    n = len(qids)
    sums = S.sum(axis=0)
    ratio = float(sums[0] / sums[1]) if sums[1] > 0 else math.nan
    diff = float((sums[2] - sums[3]) / n)
    # same seed and shape -> both statistics see the same resampled questions (paired)
    ci_ratio = az._boot_vec(S, n_boot, seed, lambda t: t[:, 0] / t[:, 1])
    ci_diff = az._boot_vec(S, n_boot, seed, lambda t: (t[:, 2] - t[:, 3]) / n)
    rule_a = bool(math.isfinite(ratio) and ratio <= COST_RATIO_MAX
                  and ci_ratio[1] is not None and ci_ratio[1] < 1.0)
    rule_b = diff >= F1_DIFF_MIN
    beaten_by = [f"{p['arm']}:{p['k']}" for p in base
                 if p["cost"] < pbc["cost"] and p["f1"] > pbc["f1"]]
    rule_c = not beaten_by
    unstable = ci_ratio.unstable or ci_diff.unstable
    supported = rule_a and rule_b and rule_c and not unstable
    return {
        "decision": "supported" if supported else "not supported",
        "reason": "unstable bootstrap CI" if unstable and rule_a and rule_b and rule_c else None,
        "bstar": {"arm": bstar["arm"], "k": bstar["k"], "f1": bstar["f1"],
                  "cost": bstar["cost"]},
        "pbc": {"k": pbc["k"], "f1": pbc["f1"], "cost": pbc["cost"]},
        "n_paired_questions": n,
        "cost_ratio": ratio, "cost_ratio_ci": ci_ratio,
        "f1_diff": diff, "f1_diff_ci": ci_diff,
        "rule_a": rule_a, "rule_b": rule_b, "rule_c": rule_c,
        "dominated_by": beaten_by,
    }


# ---------------------------------------------------------------------------------------
# evidence envelope
# ---------------------------------------------------------------------------------------

def _words(k):
    """Digits spelled out: figure names may contain letters only."""
    return "".join(_DIGITS[int(c)] if c.isdigit() else c for c in str(k))


def _camel(arm):
    return "".join(p.capitalize() for p in _words(arm).split("_"))


def build_figures(points, hyp, src, model=None, returned=None, prereg_src="prereg"):
    figs = {}
    if model is not None:  # registered model: a documented record, not a measurement
        figs["HthModel"] = {"value": model["label"], "format": "text", "source": prereg_src,
                            "desc": "Model of every arm (PREREG_v2 Amendment 2; "
                                    "experiments/configs/h2h.yaml)."}

    def fig(name, value, fmt, desc):
        assert re.fullmatch(r"Hth[A-Za-z]{1,57}", name), name
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            figs[name] = {"value": NOT_YET, "format": "text", "source": src, "desc": desc}
        else:
            figs[name] = {"value": value, "format": fmt, "source": src, "desc": desc}

    ran = hyp["decision"] != NOT_YET
    fig("HthDecision", hyp["decision"], "text", "H7 decision (PREREG_v2 section 6).")
    ci_r, ci_d = hyp.get("cost_ratio_ci") or (None, None), hyp.get("f1_diff_ci") or (None, None)
    fig("HthCostRatio", hyp.get("cost_ratio"), "float3", "H7 cost(pbc) / cost(B*).")
    fig("HthCostRatioLo", ci_r[0], "float3", "H7 cost ratio 95% CI lower bound.")
    fig("HthCostRatioHi", ci_r[1], "float3", "H7 cost ratio 95% CI upper bound.")
    fig("HthFOneDiff", None if "f1_diff" not in hyp else 100 * hyp["f1_diff"], "signedfloat2",
        "H7 F1(pbc) - F1(B*) in points.")
    fig("HthFOneDiffLo", None if ci_d[0] is None else 100 * ci_d[0], "signedfloat2",
        "H7 F1 difference 95% CI lower bound, points.")
    fig("HthFOneDiffHi", None if ci_d[1] is None else 100 * ci_d[1], "signedfloat2",
        "H7 F1 difference 95% CI upper bound, points.")
    for rule, text in (("A", "cost ratio <= 0.70 with CI below 1.0"),
                       ("B", "F1 difference >= -2.0 points"),
                       ("C", "pbc on the empirical frontier")):
        v = hyp.get(f"rule_{rule.lower()}")
        fig(f"HthRule{rule}", None if v is None else str(v), "text", f"H7 rule {rule}: {text}.")
    bstar = hyp.get("bstar") or {}
    fig("HthBstarArm", bstar.get("arm"), "text", "Baseline B* with the highest F1.")
    fig("HthBstarK", bstar.get("k"), "int", "Retrieval k of baseline B*.")
    fig("HthPairedQuestions", hyp.get("n_paired_questions"), "int",
        "Questions answered by both pbc and B*.")
    fig("HthRan", str(ran), "text", "Whether H7 could be evaluated.")
    fig("HthModelReturned", "; ".join(returned) if returned else None, "text",
        "Model id(s) returned in the responses of the analysed runs.")
    on_front = {(p["arm"], p["k"]) for p in frontier(points)}
    for p in points:
        tag = f"Hth{_camel(p['arm'])}K{_words(p['k'])}"
        fig(tag + "N", p["n"], "int", f"Runs for {p['arm']} k={p['k']}.")
        fig(tag + "Cost", None if p["cost"] is None else 1000 * p["cost"], "float2",
            f"{p['arm']} k={p['k']} list-price USD per 1000 questions.")
        fig(tag + "FOne", None if p["f1"] is None else 100 * p["f1"], "float1",
            f"{p['arm']} k={p['k']} F1 in points.")
        fig(tag + "Grounded", p["grounded"], "pct1",
            f"{p['arm']} k={p['k']} grounded-answer rate.")
        fig(tag + "Frontier", str((p["arm"], p["k"]) in on_front), "text",
            f"{p['arm']} k={p['k']} is on the empirical cost-F1 frontier.")
    return figs


def _sources(dirs):
    sources, _ = az._build_sources(dirs)
    sha = az._git(["log", "--diff-filter=A", "--format=%H", "--", "experiments/PREREG_v2.md"])
    sha = (sha or "").splitlines()[-1] if sha else os.environ.get("PREREG_V2_SHA")
    for s in sources:  # a dir without runs.jsonl must still satisfy the envelope schema
        s["sha256"] = s["sha256"] or hashlib.sha256(b"").hexdigest()
    sources[-1] = {
        "id": "prereg", "grade": "documented-record",
        "sha256": az._sha256_file(str(PREREG_V2)) or hashlib.sha256(b"").hexdigest(),
        "command": "experiments/PREREG_v2.md",
        "description": "Pre-registration experiments/PREREG_v2.md, section 6 specifies H7"
                       + (f"; registration commit {sha}" if sha else ""),
    }
    return sources


def run_evidence(input_dirs, split="test", n_boot=N_BOOT):
    runs, not_run, reqs = [], [], []
    for d in input_dirs:
        runs += az._read_jsonl(os.path.join(d, "runs.jsonl"))
        not_run += az._read_jsonl(os.path.join(d, "not_run.jsonl"))
        reqs += az._read_jsonl(os.path.join(d, "requests.jsonl"))
    rs = usable(runs, split)
    points = operating_points(rs)
    hyp = h7(rs, n_boot=n_boot)
    sources = _sources(input_dirs)
    model = registered_model()
    returned = returned_models(reqs, {r.get("run_id") for r in rs} - {None})
    aggregates = {
        "split": split, "n_runs": len(rs), "points": points,
        "model": {**model, "returned": returned},
        "frontier": [[p["arm"], p["k"]] for p in frontier(points)],
        "H7": hyp,
        "not_run": [{"arm": r.get("policy"), "k": r.get("k"), "reason": r.get("reason")}
                    for r in not_run],
    }
    return {
        "schema_version": 1,
        "kind": "agent_h2h",
        "provenance": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "generator": "experiments/analyze_v2.py",
            "notes": [
                "Cost = mean USD per question at the cited list prices (usd_list: every "
                "request incl. summariser requests). PREREG_v2 Amendment 2: OpenAI "
                "gpt-6-luna, standard tier, so this is the billed amount; cache writes are "
                "priced at 1.25x input and cached reads at 0.1x.",
                "H7: paired cluster bootstrap over questions, 2000 replicates, seed 7; "
                "B* = highest-F1 baseline point, ties lower cost.",
                f"Analysed split: {split}.",
            ],
            "sources": sources,
        },
        "aggregates": aggregates,
        "figures": build_figures(points, hyp, sources[0]["id"], model, returned,
                                 sources[-1]["id"]),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="experiments/analyze_v2.py",
                                description="PREREG_v2 head-to-head analysis.")
    p.add_argument("--input", action="append", required=True, help="h2h grid dir; repeatable")
    p.add_argument("--out", required=True, help="output evidence JSON path")
    p.add_argument("--split", default="test", choices=["dev", "test"])
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    args = p.parse_args(argv)
    evidence = run_evidence(args.input, args.split, args.n_boot)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    text = json.dumps(az._clean(evidence), indent=2)  # URLs keep their https:// scheme
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text + "\n")
    print(f"wrote {args.out} (split={args.split} runs={evidence['aggregates']['n_runs']} "
          f"H7={evidence['aggregates']['H7']['decision']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
