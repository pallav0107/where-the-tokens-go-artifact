"""Regression tests: H3 pair structure, Amendment 1 scope, new figures, fast control 3."""
import json

import numpy as np
import pytest

import experiments.analyze as az
from experiments.tests.test_analyze_contract import synth


def _gemini(runs, reqs=None, model=None):
    out = []
    for r in runs:
        r2 = dict(r, provider="gemini")
        if model:
            r2["model"] = model
        out.append(r2)
    return out


def test_h3_items_unpack_and_mean_T_only_over_paired_questions():
    runs, _ = synth(n_q=6)
    # q5 only has carry_all/k=2 for m_a (incomplete block): not a pair.
    runs = [r for r in runs if not (r["question_id"] == "q5"
                                    and (r["policy"], r["k"]) != ("carry_all", 2))]
    runs[[i for i, r in enumerate(runs) if r["question_id"] == "q5"][0]]["turns"] = 9
    out = az.h3(runs, n_boot=50, seed=7)  # used to raise "too many values to unpack"
    assert out["n_pairs"] == 5 * 2
    paired = [r for r in runs if r["question_id"] != "q5"
              and (r["policy"], r["k"]) == ("carry_all", 2)]
    assert out["mean_T"]["carry_all|k=2"] == pytest.approx(
        float(np.mean([r["turns"] for r in paired])))
    assert set(out["per_model"]) == {"m_a", "m_b"}


def test_amendment1_h1_to_h4_do_not_filter_on_provider():
    runs, reqs = synth(n_q=12)
    for prov in ("gemini", "dashscope", "other"):
        rs = [dict(r, provider=prov) for r in runs]
        assert az.h1(rs, n_boot=30)["n_test"] == az.h1(runs, n_boot=30)["n_test"] > 0
        assert az.h3(rs, n_boot=30)["n_pairs"] == 24
        assert len(az.h4(rs, n_boot=30)["cells"]) == 8
        assert az.h2(rs, reqs, n_boot=30)["n_runs_used"] > 0


def test_controls_never_enter_h1_to_h5():
    runs, reqs = synth(n_q=6)
    runs = _gemini(runs)
    for r in runs[:10]:
        r["control"] = "smoke"
    for r in runs[10:20]:
        r["control"] = "broken_reset"
    used = az.analysable(runs)
    assert len(used) == len(runs) - 20
    assert all(r["control"] not in ("smoke", "broken_reset") for r in used)
    assert az.h1(runs, n_boot=20)["n_train"] + az.h1(runs, n_boot=20)["n_test"] == len(used)


def _with_cache(reqs, runs_ids):
    out = []
    for q in reqs:
        q = dict(q)
        if q["run_id"] in runs_ids:
            q["cached_input_tokens"] = int(q["input_tokens"] * min(0.9, 0.1 * (q["step"] + 1)))
        out.append(q)
    return out


def test_h5_uses_runs_with_cached_tokens_regardless_of_provider():
    runs, reqs = synth(n_q=8)
    # No cached info anywhere -> not run, whatever the provider.
    assert az.h5(_gemini(runs), reqs, n_boot=30)["status"] == "not run"
    gem = {r["run_id"] for r in runs if r["model"] == "m_a"}
    reqs2 = _with_cache(reqs, gem)
    out = az.h5(runs, reqs2, n_boot=50)  # provider is non-cloud: still evaluated
    assert out["status"] == "evaluated"
    assert out["n_runs"] == len(gem)
    assert set(out["per_model"]) == {"m_a"}
    assert out["cached_share_slope"] > 0


def test_h5_excludes_control_runs():
    runs, reqs = synth(n_q=4)
    reqs = _with_cache(reqs, {r["run_id"] for r in runs})
    for r in runs:
        r["control"] = "smoke"
    assert az.h5(runs, reqs, n_boot=20)["status"] == "not run"


def test_figures_model_ids_and_spend(tmp_path):
    runs, reqs = synth(n_q=4)
    ids = {"m_a": "gemini-3.5-flash", "m_b": "gemini-3.5-flash-lite"}
    runs = [dict(r, provider="gemini", model=ids[r["model"]]) for r in runs]
    d = tmp_path / "grid"
    d.mkdir()
    (d / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in runs), encoding="utf-8")
    (d / "requests.jsonl").write_text("".join(json.dumps(q) + "\n" for q in reqs),
                                      encoding="utf-8")
    (d / "provenance.json").write_text(json.dumps({"cloud_spend_total": 1.2345}),
                                       encoding="utf-8")
    ev = az._clean(az.run_evidence([str(d)], n_boot=20, n_perm=2, control1_path=None))
    figs = ev["figures"]
    assert figs["AgentModelFlashId"] == {**figs["AgentModelFlashId"],
                                         "value": "gemini-3.5-flash", "format": "text"}
    assert figs["AgentModelLiteId"]["value"] == "gemini-3.5-flash-lite"
    assert figs["AgentCloudSpendUsd"]["value"] == pytest.approx(1.2345)
    assert figs["AgentCloudSpendUsd"]["format"] == "float2"
    src = {s["id"] for s in ev["provenance"]["sources"]}
    for nm in ("AgentModelFlashId", "AgentModelLiteId", "AgentCloudSpendUsd"):
        assert figs[nm]["source"] in src and figs[nm]["desc"]

    d2 = tmp_path / "empty"
    d2.mkdir()
    (d2 / "runs.jsonl").write_text("", encoding="utf-8")
    (d2 / "requests.jsonl").write_text("", encoding="utf-8")
    ev2 = az._clean(az.run_evidence([str(d2)], n_boot=20, n_perm=2, control1_path=None))
    assert ev2["figures"]["AgentModelFlashId"]["value"] == "not yet run"
    assert ev2["figures"]["AgentModelLiteId"]["value"] == "not yet run"
    assert ev2["figures"]["AgentCloudSpendUsd"]["value"] == 0


def test_control3_vectorised_matches_h1_decision_rule():
    runs, _ = synth(n_q=24, seed=3)
    base = az.analysable(runs)
    Y = np.array([float(r["total_input"]) for r in base])
    rng = np.random.default_rng(5)
    noisy = Y * (1 + rng.normal(0, 0.6, Y.size))
    shuffled = Y.copy()
    idx = np.arange(Y.size)
    shuffled[idx] = Y[rng.permutation(Y.size)]
    M = np.array([Y, noisy, shuffled])
    fast = az._h1_decisions_for(base, M, 100)
    slow = [az.h1([dict(r, total_input=v) for r, v in zip(base, row)],
                  n_boot=100, seed=7)["decision"] for row in M]
    assert fast == slow
    assert fast[0] == "supported"
