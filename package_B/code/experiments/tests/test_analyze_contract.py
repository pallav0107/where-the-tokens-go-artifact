"""Contract tests for experiments/analyze.py (written before the implementation).

Synthetic runs with KNOWN structure check that each pre-registered decision rule
(PREREG section 7) detects the effect when it exists and does not when it is absent,
and that fail-first controls 2 and 3 (section 8) behave as specified.
"""
import numpy as np
import pytest

import experiments.analyze as az

MODELS = ("m_a", "m_b")
POLICIES = ("carry_all", "reset_summary")
KS = (2, 8)


def synth(carry_grows=True, reset_grows=False, k_matters=True, n_q=48, seed=0, broken_reset=False):
    """Generate runs.jsonl / requests.jsonl-shaped records.

    Per-request input: x_1 = P; x_t = P + (t-1)*D (growing) or P + D (flat) for t >= 2,
    with D = per-step increment that depends on k when k_matters. Turns vary by question.
    """
    rng = np.random.default_rng(seed)
    runs, reqs = [], []
    for pos in range(n_q):
        qid = f"q{pos}"
        base_T = int(rng.integers(3, 9))
        for m_i, m in enumerate(MODELS):
            for pol in POLICIES:
                for k in KS:
                    T = max(2, base_T + int(rng.integers(-1, 2)))
                    P = int(600 + 40 * m_i + rng.integers(-30, 30))
                    D = (120 + 70 * k) if k_matters else 400
                    grows = carry_grows if pol == "carry_all" else (reset_grows or broken_reset)
                    xs = []
                    for t in range(1, T + 1):
                        if t == 1:
                            x = P
                        elif grows:
                            x = P + (t - 1) * D
                        else:
                            x = P + D
                        x = int(x * (1 + rng.normal(0, 0.02)))
                        xs.append(x)
                    rid = f"{m}|{pol}|k{k}|{qid}"
                    for s, x in enumerate(xs):
                        reqs.append({"run_id": rid, "step": s, "input_tokens": x, "cached_input_tokens": None,
                                     "output_tokens": 50, "usd": None})
                    runs.append({"run_id": rid, "provider": "ollama", "model": m, "policy": pol, "k": k,
                                 "control": "broken_reset" if (broken_reset and pol == "reset_summary") else None,
                                 "question_id": qid, "position": pos, "split": "test" if pos % 3 == 2 else "train",
                                 "status": "ok", "turns": T, "total_input": sum(xs), "total_cached": None,
                                 "total_output": 50 * T, "P": xs[0], "latency_s": 1.0, "finished": True,
                                 "em": int(rng.random() < 0.5), "f1": float(rng.random()), "usd": None,
                                 "truncated": False})
    return runs, reqs


def test_h1_supported_when_cost_is_context_times_turns():
    runs, _ = synth()
    out = az.h1(runs, n_boot=300, seed=7)
    assert out["decision"] == "supported", out
    assert out["pooled"]["M_ctx"]["r2"] >= 0.80
    assert out["pooled"]["M_prompt"]["mape"] > out["pooled"]["M_ctx"]["mape"]
    for key in ("M_ctx", "M_prompt", "M_cell", "M_carry"):
        assert {"r2", "mae", "mape"} <= set(out["pooled"][key])
    assert out["delta_mape_ci"][0] > 0


def test_control3_label_shuffle_does_not_support_h1():
    runs, _ = synth()
    out = az.control3(runs, n_perm=40, seed=11, n_boot=100)
    assert out["frac_supported"] <= 0.05 and out["passed"] is True


def test_h2_supported_superlinear_carry_linear_reset():
    runs, reqs = synth()
    out = az.h2(runs, reqs, n_boot=300, seed=7)
    assert out["decision"] == "supported", out
    assert out["ci"]["carry_all"][0] > 1.0
    assert 0.90 <= out["b"]["reset_summary"] <= 1.10
    assert out["diff_ci"][0] > 0


def test_h2_not_supported_when_both_policies_flat():
    runs, reqs = synth(carry_grows=False)
    assert az.h2(runs, reqs, n_boot=300, seed=7)["decision"] == "not supported"


def test_control2_broken_reset_is_detected():
    runs, reqs = synth()
    broken_runs, broken_reqs = synth(broken_reset=True, seed=1)
    broken_runs = [r for r in broken_runs if r["policy"] == "reset_summary"]
    ids = {r["run_id"] for r in broken_runs}
    broken_reqs = [q for q in broken_reqs if q["run_id"] in ids]
    out = az.control2(runs, reqs, broken_runs, broken_reqs, n_boot=300, seed=7)
    assert out["h2_decision_with_broken"] != "supported"
    assert out["passed"] is True


def test_h3_interaction_detected_and_null_rejected():
    runs, _ = synth()
    out = az.h3(runs, n_boot=300, seed=7)
    assert out["decision"] == "supported", out
    assert out["I_ci"][0] > 0 and out["rho_ci"][0] > 1
    runs0, _ = synth(k_matters=False)
    assert az.h3(runs0, n_boot=300, seed=7)["decision"] == "not supported"


def test_h4_reports_without_direction():
    runs, _ = synth()
    out = az.h4(runs, n_boot=200, seed=7)
    cells = out["cells"]
    assert len(cells) == len(MODELS) * len(POLICIES) * len(KS)
    for c in cells:
        assert 0 <= c["em"] <= 1 and len(c["em_ci"]) == 2 and len(c["f1_ci"]) == 2
    assert "decision" not in out  # H4 is descriptive only


def test_excluded_runs_are_not_analysed_but_counted():
    runs, reqs = synth()
    runs[0]["status"] = "error"
    runs[1]["truncated"] = True
    ex = az.exclusions(runs)
    assert ex["n_excluded"] == 2
    assert all(r["status"] == "ok" and not r["truncated"] for r in az.analysable(runs))
