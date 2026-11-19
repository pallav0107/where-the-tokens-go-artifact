"""PREREG_v3 post-test checklist (DECISIONS D47 (5), D50; the auditor's first-report checks).

    python experiments/offline/posttest_v3_checks.py FINAL_ARTIFACT DISPATCH_ARTIFACT... \
        --parquet PATH --ci-json PATH --out PATH

FINAL_ARTIFACT is the downloaded artifact of the last test dispatch (its h2h3_test_r1_s0/s1 dirs
hold every test run). DISPATCH_ARTIFACT... are the ten dispatch artifacts in order (provenance
table and salt nonces). Read-only; no API call. The registered analysis (analyze_v3.py) is re-run
here outside CI and compared with the CI JSON. A sensitivity analysis prices each timed-out
(unconfirmed) attempt at its retry's actual cost and recomputes H8.
"""
from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from agent.arms import cells_from_config  # noqa: E402
from agent.budget import charged_usd, collect_prior_spend  # noqa: E402
from agent.data import split_positions  # noqa: E402
from agent.run_grid import cell_salt  # noqa: E402
from experiments import analyze_v3 as av  # noqa: E402

PRICE = av.PRICE
RESULTS: list = []


def check(name, ok, detail=""):
    RESULTS.append({"check": name, "pass": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""))


def jl(path):
    if not os.path.isfile(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("final")
    p.add_argument("dispatches", nargs="+")
    p.add_argument("--parquet", required=True)
    p.add_argument("--ci-json", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    cfg = yaml.safe_load(open(ROOT / "experiments" / "configs" / "h2h_v3.yaml", encoding="utf-8"))
    shards = sorted(glob.glob(os.path.join(a.final, "h2h3_test_r1_s?")))
    runs, reqs, failed = [], [], []
    for d in shards:
        last = {}
        for r in jl(os.path.join(d, "runs.jsonl")):
            last[r["run_id"]] = r
        runs += list(last.values())
        reqs += jl(os.path.join(d, "requests.jsonl"))
        failed += jl(os.path.join(d, "requests_failed.jsonl"))
    ok = [r for r in runs if r["status"] == "ok"]

    # 1 dataset and split
    h = hashlib.sha256(open(a.parquet, "rb").read()).hexdigest()
    check("dataset sha256", h == cfg["dataset"]["sha256"], h[:16])
    pos = sorted({r["position"] for r in ok})
    check("test positions = 4000-4239", pos == list(range(4000, 4240)), f"{pos[0]}-{pos[-1]} n={len(pos)}")
    touched = json.load(open(ROOT / "experiments" / "offline" / "TOUCHED_POSITIONS.json"))
    tset = {q for rg in touched["ranges"] for q in range(rg["start"], rg["end"] + 1)}
    check("test disjoint from touched positions", not (set(pos) & tset))
    blocks = {}
    for bi, (s, e) in enumerate(cfg["dataset"]["blocks"]["test"]):
        for q in range(s, e + 1):
            blocks[q] = bi
    check("each run's block matches its position", all(blocks[r["position"]] == r["block"] for r in ok))

    # 2 completeness
    cells = cells_from_config(cfg)
    keys = Counter((r["arm"], int(r["k"]), r["question_id"], r["repeat"]) for r in ok)
    check("10,800 ok runs, one per (cell, question, repeat)",
          len(ok) == 240 * len(cells) * 3 and len(keys) == len(ok) and set(c[:2] for c in keys) == set(cells),
          f"ok={len(ok)} unique={len(keys)} cells={len(set(c[:2] for c in keys))}")
    check("no errored or NOT RUN run", len(runs) == len(ok) and not any(
        jl(os.path.join(d, "not_run.jsonl")) for d in shards), f"runs={len(runs)}")

    # 3 provenance table and salt tracing
    table, nonces = [], []
    for art in a.dispatches:
        ctx = open(os.path.join(art, "ci_context.txt"), encoding="utf-8").read()
        rid = re.search(r"/actions/runs/(\d+)", ctx).group(1)
        q = re.search(r"questions=(\d+)", ctx).group(1)
        row = {"run_id": rid, "questions": int(q), "git_sha": re.search(r"git_sha=(\w+)", ctx).group(1)}
        for sd in sorted(glob.glob(os.path.join(art, "h2h3_test_r1_s?"))):
            pv = json.load(open(os.path.join(sd, "provenance.json")))
            row[os.path.basename(sd)[-2:]] = {"nonce": pv["salt_nonce"], "config_sha256": pv["config_sha256"][:16],
                                             "prereg_v3_sha": (pv.get("prereg_v3_sha") or "")[:7],
                                             "spend_this_run": pv["cloud_spend_this_run"]}
            nonces.append(pv["salt_nonce"])
            last = {}
            for r in jl(os.path.join(sd, "runs.jsonl")):
                last[r["run_id"]] = r
            row.setdefault("cum_ok", 0)
            row["cum_ok"] += sum(1 for r in last.values() if r["status"] == "ok")
        table.append(row)
    check("20 shard nonces, all distinct", len(nonces) == 20 and len(set(nonces)) == 20, str(len(set(nonces))))
    check("cumulative ok runs = 1,080 x k per dispatch",
          [t["cum_ok"] for t in table] == [1080 * (i + 1) for i in range(len(table))],
          str([t["cum_ok"] for t in table]))

    def traced(r):
        for n in nonces:
            for att in (1, 2, 3):
                if cell_salt(n, r["arm"], int(r["k"]), r["repeat"], att) == r["salt"]:
                    return True
        return False
    bad = [r["run_id"] for r in ok if not traced(r)]
    check("every run's salt traces to a shard nonce", not bad, f"untraced={len(bad)}")
    check("salts distinct between cells within each dispatch nonce",
          len({(r["salt"]) for r in ok}) >= len({(r["arm"], r["k"], r["repeat"]) for r in ok}))

    # 4 per-request provider facts
    agent_reqs = [q for q in reqs if q.get("role") != "unrecorded_charge"]
    check("model gpt-6-luna on every request", all(q.get("response_model") == "gpt-6-luna" for q in agent_reqs),
          str(Counter(q.get("response_model") for q in agent_reqs)))
    check("tier default, finish stop", all(q.get("service_tier") == "default" and q.get("finish_reason") == "stop"
                                           for q in agent_reqs),
          str(Counter((q.get("service_tier"), q.get("finish_reason")) for q in agent_reqs)))
    check("no parameter rejected; reasoning tokens 0",
          all(not (q.get("thinking_config") or {}).get("rejected") for q in agent_reqs)
          and sum(q.get("reasoning_tokens") or 0 for q in agent_reqs) == 0)
    check("cache_write_tokens field reported on every request",
          all("cache_write_tokens" in ((q.get("usage_metadata") or {}).get("prompt_tokens_details") or {})
              for q in agent_reqs))

    # 5 cache isolation and USD recomputation
    ids = {r["run_id"] for r in ok}
    iso = av.cross_run_cache_reads(reqs, ids)
    check("cross-run cached tokens = 0", iso["ok"], str(iso["cross_run_cached_tokens"]))
    worst = 0.0
    for q in agent_reqs:
        u = (q["input_tokens"] - q["cached_input_tokens"] - (q["cache_write_tokens"] or 0)) * PRICE["input"] \
            + q["cached_input_tokens"] * PRICE["cached"] + (q["cache_write_tokens"] or 0) * PRICE["write"] \
            + q["output_tokens"] * PRICE["output"]
        worst = max(worst, abs(u / 1e6 - q["usd"]))
    check("request USD = B2 formula from usage", worst < 1e-8, f"max abs diff {worst:.2e}")
    by_run = defaultdict(float)
    for q in agent_reqs:
        by_run[q["run_id"]] += q["usd"]
    check("run usd_list = sum of its request USD", all(abs(by_run[r["run_id"]] - r["usd_list"]) < 2e-6 for r in ok))

    # 6 orphans, failed and unrecorded records
    orphans = [q["run_id"] for q in reqs if q["run_id"] not in ids]
    check("no orphan requests (every requests.jsonl row belongs to an ok run)", not orphans, str(len(orphans)))
    unrec = [q for q in reqs + failed if q.get("role") == "unrecorded_charge"]
    print("unrecorded_charge records:", [(q["run_id"][-40:], q["arm"], q["usd"], q.get("reason", "")[:60])
                                         for q in unrec])
    print("failed request records:", [(q["run_id"][-40:], q.get("arm"), q.get("usd"), q.get("attempt")) for q in failed])

    # 7 gate statistics without unrecorded rows; defect; >= 1,024 share
    clean = [q for q in reqs if q.get("role") != "unrecorded_charge"]
    g = av.gate_stats([r for r in ok], clean)
    aq = [q for q in clean if q.get("arm") == "aob"]
    fails = [q for q in aq if q.get("gate") == "fail"]
    gate = {**g, "aob_requests": len(aq), "gate_fail": len(fails),
            "gate_fail_no_next": sum(1 for q in fails if not q.get("next_queries")),
            "parse_errors": sum(1 for q in aq if q.get("parse_error"))}
    print("aob gate (no unrecorded rows):", json.dumps(gate))

    # 8 attempts per arm (both files)
    att = defaultdict(lambda: Counter())
    for q in reqs + failed:
        arm = q.get("arm")
        att[arm]["unconfirmed_attempts"] += int(q.get("unconfirmed_attempts") or 0)
        att[arm]["unrecorded_charges"] += q.get("role") == "unrecorded_charge"
    for q in failed:
        att[q.get("arm")]["failed_request_records"] += q.get("role") != "unrecorded_charge"
    attempts = {k: dict(v) for k, v in sorted(att.items())}
    print("attempts per arm:", json.dumps(attempts))

    # 9 reproduce the registered analysis outside CI
    ev = av.run_evidence(shards, "test")
    ci = json.load(open(a.ci_json, encoding="utf-8"))
    h_loc, h_ci = ev["aggregates"]["H8"], ci["aggregates"]["H8"]
    same = all(h_loc.get(k) == h_ci.get(k) for k in ("decision", "rule_a", "rule_b", "rule_c", "rule_d"))
    num = all(math.isclose(h_loc[k], h_ci[k], rel_tol=0, abs_tol=1e-12) for k in ("cost_ratio", "f1_diff"))
    cis = all(all(math.isclose(x, y, abs_tol=1e-12) for x, y in zip(h_loc[k], h_ci[k]))
              for k in ("cost_ratio_ci", "f1_diff_ci"))
    pts = {(p["arm"], p["k"]): p for p in ev["aggregates"]["points"]}
    pts_ci = {(p["arm"], p["k"]): p for p in ci["aggregates"]["points"]}
    pt_same = all(math.isclose(pts[c]["f1"], pts_ci[c]["f1"], abs_tol=1e-12)
                  and math.isclose(pts[c]["cost"], pts_ci[c]["cost"], abs_tol=1e-15) for c in pts)
    check("analyze_v3 re-run outside CI reproduces the CI JSON (H8 and every point)",
          same and num and cis and pt_same and set(pts) == set(pts_ci))

    # 10 sensitivity: each timed-out attempt priced at its retry's actual cost
    extra = defaultdict(float)
    for q in agent_reqs:
        n = int(q.get("unconfirmed_attempts") or 0)
        if n:
            extra[q["run_id"]] += n * q["usd"]
    rs2 = copy.deepcopy(av.usable(ok, "test"))
    for r in rs2:
        r["usd_list"] = r["usd_list"] + extra.get(r["run_id"], 0.0)
    h_s = av.h8(rs2, "cost")
    sens = {"runs_with_timeouts": len(extra), "added_usd_total": round(sum(extra.values()), 8),
            "decision": h_s["decision"], "cost_ratio": h_s.get("cost_ratio"),
            "cost_ratio_ci": h_s.get("cost_ratio_ci"), "rules": [h_s.get(f"rule_{x}") for x in "abcd"],
            "bstar": h_s.get("bstar")}
    print("sensitivity (timeouts priced at the retry's cost):", json.dumps(sens))
    spend = collect_prior_spend([a.final])
    out = {"checks": RESULTS, "dispatch_table": table, "gate_without_unrecorded": gate,
           "attempts_per_arm": attempts, "unrecorded_records": len(unrec), "failed_records": len(failed),
           "sensitivity_timeouts_at_retry_cost": sens, "charged_spend_total_from_records": spend["total"],
           "local_h8": {k: h_loc.get(k) for k in ("decision", "cost_ratio", "cost_ratio_ci", "f1_diff",
                                                   "f1_diff_ci", "rule_a", "rule_b", "rule_c", "rule_d",
                                                   "bstar")}}
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=2, default=str)
    print("charged spend total from records:", spend["total"])
    print("ALL CHECKS PASS" if all(r["pass"] for r in RESULTS) else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
