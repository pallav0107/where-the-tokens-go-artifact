"""PREREG_v3 section 8: H8 rules (a)-(d), B* eligibility, block-stratified bootstrap, the
counterfactual repricings and the cross-run cache-isolation check (synthetic runs)."""
import json

import numpy as np
import pytest

from experiments import analyze_v3 as av


def mk(arm, k, i, usd, f1, block=None, inp=1000, cached=0, out=50, split="test"):
    return {"run_id": f"{arm}|k={k}|q{i}", "arm": arm, "policy": arm, "k": k,
            "question_id": f"q{i}", "block": i % 2 if block is None else block,
            "status": "ok", "harness": "v3", "split": split, "usd_list": usd,
            "total_input": inp, "total_cached": cached, "total_cache_write": 0,
            "total_output": out, "em": int(f1 == 1.0), "f1": f1, "n_requests": 1,
            "grounded": None, "latency_s": 1.0}


def cell(arm, k, usd, f1s, **kw):
    return [mk(arm, k, i, usd if not callable(usd) else usd(i), f, **kw)
            for i, f in enumerate(f1s)]


N = 40
F_BASE = [1.0 if i % 3 else 0.5 for i in range(N)]  # mean 0.8375


def world(aob_usd=0.0001, aob_f1=None, extra=()):
    runs = []
    runs += cell("notes_reset", 8, 0.0002, F_BASE)               # B* (highest F1)
    runs += cell("react", 8, 0.00019, [f - 0.1 for f in F_BASE])
    runs += cell("react", 2, 0.00018, [f - 0.2 for f in F_BASE])
    runs += cell("aob", 6, aob_usd, aob_f1 or F_BASE)
    for e in extra:
        runs += e
    return runs


def test_bstar_is_the_highest_f1_published_baseline_only():
    runs = world(extra=[cell("rag1", 8, 0.0001, [1.0] * N),
                        cell("aob_nogate", 6, 0.0001, [1.0] * N),
                        cell("pbc", 3, 0.0003, [1.0] * N),
                        cell("closed_book", 0, 0.00001, [1.0] * N)])
    b = av.select_bstar(av.operating_points(av.usable(runs)))
    assert (b["arm"], b["k"]) == ("notes_reset", 8)


def test_h8_supported_when_half_the_cost_at_equal_f1():
    h = av.h8(av.usable(world()))
    assert h["decision"] == "supported", h
    assert h["cost_ratio"] == pytest.approx(0.5) and h["f1_diff"] == pytest.approx(0.0)
    assert h["n_paired_questions"] == N


def test_rule_a_needs_ratio_at_most_070():
    h = av.h8(av.usable(world(aob_usd=0.00015)))
    assert h["cost_ratio"] == pytest.approx(0.75) and h["rule_a"] is False
    assert h["decision"] == "not supported"


def test_rule_b_point_estimate_is_inclusive_at_minus_two_points():
    h = av.h8(av.usable(world(aob_f1=[f - 0.02 for f in F_BASE])))
    assert h["f1_diff"] == pytest.approx(-0.02) and h["rule_b"] is True


def test_rule_b_also_needs_the_ci_lower_bound_at_least_minus_five_points():
    # mean difference -1 point, but per-question differences of about +-28 points: wide CI
    runs = (cell("notes_reset", 8, 0.0002, [0.5] * N) + cell("react", 8, 0.00019, [0.4] * N)
            + cell("aob", 6, 0.0001, [0.77 if i % 4 >= 2 else 0.21 for i in range(N)]))
    h = av.h8(av.usable(runs))
    assert h["f1_diff"] >= -0.02 and h["f1_diff_ci"][0] < -0.05
    assert h["rule_b"] is False and h["decision"] == "not supported"


def test_rule_c_counts_the_reference_points():
    runs = world(extra=[cell("rag1", 8, 0.00009, [1.0] * N)])
    h = av.h8(av.usable(runs))
    assert h["rule_c"] is False and "rag1:8" in h["dominated_by"]


def test_rule_d_every_close_published_cell_must_also_be_at_most_070():
    # react k=2 is within 2 F1 points of aob and cheap: aob must also be <= 0.70 x its cost
    runs = world(aob_usd=0.00013)
    runs = [r for r in runs if not (r["arm"] == "react" and r["k"] == 2)]
    runs += cell("react", 2, 0.00016, [f - 0.01 for f in F_BASE])
    h = av.h8(av.usable(runs))
    assert h["rule_a"] is True and h["cost_ratio"] == pytest.approx(0.65)
    assert h["rule_d"] is False and h["rule_d_failed"] == ["react:2"]
    assert h["decision"] == "not supported"


def test_not_yet_run_without_aob():
    runs = [r for r in world() if r["arm"] != "aob"]
    assert av.h8(av.usable(runs))["decision"] == av.NOT_YET


def test_bootstrap_resamples_within_blocks():
    blocks = [0] * 5 + [1] * 7 + [2] * 3
    idx = av.boot_indices(blocks, n_boot=200, seed=7)
    assert idx.shape == (200, 15)
    blk = np.array(blocks)[idx]
    for row in blk:
        assert sorted(row.tolist()) == sorted(blocks)  # every replicate keeps block sizes
    again = av.boot_indices(blocks, n_boot=200, seed=7)
    assert (idx == again).all()


def test_counterfactual_repricings():
    r = mk("aob", 6, 0, 0.000999, 1.0, inp=2000, cached=1200, out=100)
    c = av.run_costs(r)
    assert c["cost"] == 0.000999
    assert c["cost_neutral"] == pytest.approx((2000 * 0.10 + 100 * 0.50) / 1e6)
    assert c["cost_nowrite"] == pytest.approx((800 * 0.10 + 1200 * 0.01 + 100 * 0.50) / 1e6)


def _req(run, attempt, inp, cached, arm="react"):
    return {"run_id": run, "attempt": attempt, "input_tokens": inp,
            "cached_input_tokens": cached, "arm": arm}


def test_cross_run_cache_reads_flags_reads_without_an_own_earlier_cacheable_request():
    reqs = [_req("r1", 1, 1100, 0), _req("r1", 1, 1300, 1024),   # own earlier write: fine
            _req("r2", 1, 1100, 1024),                             # first request reads: cross
            _req("r3", 1, 900, 0), _req("r3", 1, 1200, 1024),      # earlier one was < 1024
            _req("r1", 2, 1300, 1024, arm="aob")]                  # new attempt, no earlier
    out = av.cross_run_cache_reads(reqs)
    assert out["cross_run_cached_tokens"] == 3 * 1024 and out["ok"] is False
    assert out["by_arm"] == {"react": 2048, "aob": 1024}
    assert av.cross_run_cache_reads(reqs[:2])["ok"] is True


def test_failed_isolation_check_switches_the_primary_cost_to_cache_neutral(tmp_path):
    d = tmp_path / "grid"
    d.mkdir()
    runs = world()
    (d / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in runs), encoding="utf-8")
    reqs = [_req(runs[0]["run_id"], 1, 1100, 1024)]
    (d / "requests.jsonl").write_text(json.dumps(reqs[0]), encoding="utf-8")
    ev = av.run_evidence([str(d)], "test", n_boot=200)
    assert ev["aggregates"]["H8"]["cost_key"] == "cost_neutral"
    assert ev["aggregates"]["H8"]["cache_isolation_ok"] is False
    (d / "requests.jsonl").write_text("", encoding="utf-8")
    ev = av.run_evidence([str(d)], "test", n_boot=200)
    assert ev["aggregates"]["H8"]["cost_key"] == "cost"
    assert ev["kind"] == "agent_h2h_v3" and "HthreeDecision" in ev["figures"]


def test_rule_a_also_needs_the_cost_ratio_ci_entirely_below_one():
    # mean ratio 0.68, but 6 of 40 questions cost 85x the rest: wide CI
    runs = world(aob_usd=lambda i: 0.00085 if i % 16 in (0, 5) else 0.00001)
    h = av.h8(av.usable(runs))
    assert h["cost_ratio"] == pytest.approx(0.68)
    assert h["cost_ratio_ci"][1] >= 1.0 and h["rule_a"] is False


def _with_repeat(rows, rep):
    for r in rows:
        r["repeat"], r["run_id"] = rep, r["run_id"] + f"|rep{rep}"
    return rows


def test_repeats_are_averaged_per_question_and_their_variance_reported():
    runs = []
    for rep, (usd, f1) in enumerate([(0.0002, 1.0), (0.0002, 0.9), (0.0002, 0.8)], start=1):
        runs += _with_repeat(cell("notes_reset", 8, usd, [f1] * N), rep)
    for rep, usd in enumerate([0.00008, 0.0001, 0.00012], start=1):
        runs += _with_repeat(cell("aob", 6, usd, [0.9] * N), rep)
    rs = av.usable(runs)
    c = av.compare(rs, ("aob", 6), ("notes_reset", 8))
    assert c["n"] == N  # questions, not runs
    assert c["cost_ratio"] == pytest.approx(0.5) and c["f1_diff"] == pytest.approx(0.0)
    st = av.repeat_stats(rs, ("notes_reset", 8))
    assert st["n_repeats"] == 3 and st["f1_by_repeat"] == pytest.approx([1.0, 0.9, 0.8])
    assert st["f1_sd"] == pytest.approx(0.1) and st["f1_range"] == pytest.approx(0.2)
    pts = {(p["arm"], p["k"]): p for p in av.operating_points(rs)}
    assert pts[("aob", 6)]["n_questions"] == N and pts[("aob", 6)]["n_repeats"] == 3


def test_cache_hit_rate_turns_and_cell_cis():
    runs = cell("react", 8, 0.0002, F_BASE, inp=2000, cached=500)
    for r in runs:
        r["turns"] = 2
    rs = av.usable(runs)
    p = av.operating_points(rs)[0]
    assert p["cache_hit_rate"] == pytest.approx(0.25) and p["turns"] == 2
    ci = av.cell_cis(rs, ("react", 8))
    assert ci["f1_ci"][0] <= p["f1"] <= ci["f1_ci"][1]
    assert ci["cost_ci"] == pytest.approx([0.0002, 0.0002])


def test_inclusive_thresholds_are_float_safe():
    # summed over 40 questions, 0.09 vs 0.11 gives -0.020000000000000028: still exactly -2 points
    runs = (cell("notes_reset", 8, 0.0002, [0.11] * N) + cell("react", 8, 0.00019, [0.05] * N)
            + cell("aob", 6, 0.0001, [0.09] * N))
    h = av.h8(av.usable(runs))
    assert h["f1_diff"] < -0.02  # the float artefact is present
    assert h["rule_b"] is True and h["rule_d"] is True


def test_a_partly_run_cell_cannot_be_bstar():
    runs = world()
    runs += cell("memgpt", 8, 0.0003, [1.0] * 30)  # stopped after 30 of 40 questions
    b = av.select_bstar(av.operating_points(av.usable(runs)))
    assert (b["arm"], b["k"]) == ("notes_reset", 8)
