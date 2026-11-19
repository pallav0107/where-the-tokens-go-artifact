"""Regression tests for the independent review of experiments/analyze.py.

One test per review item (request-log hygiene, H2 drop counting, CI validity
reporting, H1 same-row scoring, control2 robustness, H3 notes/mean_T,
exclusions split, evidence envelope).
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

import experiments.analyze as az
from experiments.tests.test_analyze_contract import synth


def test_item1_final_attempt_and_request_log_mismatch():
    runs, reqs = synth(n_q=4)
    rid, T0 = runs[0]["run_id"], runs[0]["turns"]
    stale = [{"run_id": rid, "step": s, "input_tokens": 111,
              "cached_input_tokens": None, "output_tokens": 1, "usd": None,
              "attempt": 1} for s in range(2)]
    marked = [dict(q, attempt=2) if q["run_id"] == rid else q for q in reqs]
    kept = az.final_attempt_requests(marked + stale)
    rid_kept = [q for q in kept if q["run_id"] == rid]
    assert len(rid_kept) == T0 and all(q.get("attempt") == 2 for q in rid_kept)
    assert not any(q["input_tokens"] == 111 for q in kept)
    # records without `attempt` are treated as attempt 1 and all kept
    assert len(az.final_attempt_requests(reqs)) == len(reqs)

    rid2, T2 = runs[1]["run_id"], runs[1]["turns"]
    reqs_missing = [q for q in reqs if q["run_id"] != rid2]
    reqs_missing += [q for q in reqs if q["run_id"] == rid2 and q["step"] < T2 - 1]
    bad = az.request_log_mismatch_ids(runs, reqs_missing)
    assert bad == {rid2}
    assert rid2 not in {r["run_id"] for r in az.analysable(runs) if r["run_id"] not in bad}
    ex = az.exclusions(runs, request_mismatch_ids=bad)
    assert ex["n_request_log_mismatch"] == 1
    assert ex["by_reason"]["request log mismatch"] == 1
    assert ex["n_excluded"] == 1


def test_item2_h2_counts_dropped_nonpositive_runs():
    runs, reqs = synth(n_q=6)
    victim = next(r for r in runs if r["turns"] >= 3 and r["policy"] == "carry_all")
    for q in reqs:
        if q["run_id"] == victim["run_id"] and q["step"] == 1:
            q["input_tokens"] = 0
    out = az.h2(runs, reqs, n_boot=50, seed=7)
    assert out["n_runs_dropped_nonpositive"] == 1


def test_item3_boot_questions_reports_validity_and_marks_unstable():
    items = [(f"q{i}", float(i)) for i in range(10)]

    def flaky(ps):
        return float(np.mean(ps)) if any(p == 0.0 for p in ps) else None

    ci = az._boot_questions(items, n_boot=200, seed=1, stat=flaky)
    assert isinstance(ci, az.CI) and len(ci) == 2
    assert 0 < ci.n_valid_boot < 0.95 * 200
    assert ci.unstable
    stable = az._boot_questions(items, n_boot=200, seed=1,
                                stat=lambda ps: float(np.mean(ps)))
    assert stable.n_valid_boot == 200 and not stable.unstable


def test_item3_unstable_ci_demotes_decision(monkeypatch):
    runs, _ = synth(n_q=8)
    real = az._boot_questions

    def fake(items, n_boot, seed, stat):
        ci = real(items, n_boot, seed, stat)
        if ci is None:
            return None
        return az.CI(ci[0], ci[1], n_valid_boot=max(1, int(0.5 * n_boot)), n_boot=n_boot)

    monkeypatch.setattr(az, "_boot_questions", fake)
    out = az.h1(runs, n_boot=100, seed=7)
    assert out["decision"] == "not supported (unstable CI)"


def test_item4_h1_scores_every_model_on_same_rows():
    runs, _ = synth(n_q=12)
    m, pol, k = "m_a", "carry_all", 8
    runs = [r for r in runs if not (r["model"] == m and r["policy"] == pol
                                    and int(r["k"]) == k and r["split"] == "train")]
    out = az.h1(runs, n_boot=50, seed=7)
    n_missing = len([r for r in runs if r["split"] == "test" and r["model"] == m
                     and r["policy"] == pol and int(r["k"]) == k])
    assert n_missing > 0
    pm = out["per_model"][m]
    n_scored = pm["M_ctx"]["n_scored"]
    assert n_scored == pm["M_prompt"]["n_scored"] == pm["M_cell"]["n_scored"] \
        == pm["M_carry"]["n_scored"]
    assert n_scored == pm["M_ctx"]["n_test_total"] - n_missing
    pooled = out["pooled"]
    assert pooled["M_ctx"]["n_scored"] == pooled["M_cell"]["n_scored"]
    assert pooled["M_ctx"]["n_test_total"] == len([r for r in runs if r["split"] == "test"])


def test_item5_control2_filters_broken_and_reports_rules():
    runs, reqs = synth()
    broken_runs, broken_reqs = synth(broken_reset=True, seed=1)
    broken_runs = [r for r in broken_runs if r["policy"] == "reset_summary"]
    broken_runs[0]["status"] = "error"  # must not be used
    ids = {r["run_id"] for r in broken_runs}
    broken_reqs = [q for q in broken_reqs if q["run_id"] in ids]
    out = az.control2(runs, reqs, broken_runs, broken_reqs, n_boot=100, seed=7)
    assert out["status"] == "passed" and out["passed"] is True
    assert out["n_broken_used"] == len(broken_runs) - 1
    assert out["n_questions"] == 48
    assert set(out["rules_failed"]) | set(out["rules_passed"]) == {"a", "b", "c"}
    assert "a" in out["rules_passed"] and "c" in out["rules_failed"]


def test_item5_control2_inconclusive_when_subset_too_small():
    runs, reqs = synth()
    broken_runs, broken_reqs = synth(broken_reset=True, seed=1)
    broken_runs = [r for r in broken_runs if r["policy"] == "reset_summary"]
    for r in broken_runs:
        r["turns"] = 2
    ids = {r["run_id"] for r in broken_runs}
    broken_reqs = [q for q in broken_reqs if q["run_id"] in ids and q["step"] < 2]
    out = az.control2(runs, reqs, broken_runs, broken_reqs, n_boot=100, seed=7)
    assert out["status"] == "inconclusive" and out["passed"] is False
    assert "T >= 3" in out["inconclusive_reason"]


def test_item6_h3_note_and_mean_T_over_paired_questions_only():
    # k_matters=False with identical T per (question, model): the k-effect is
    # exactly 0 in both policies, so the rho denominator CI is exactly [0, 0].
    runs, _ = synth(k_matters=False, n_q=8)
    newruns, newreqs = [], []
    by_qm = {}
    for r in runs:
        by_qm.setdefault((r["question_id"], r["model"]), []).append(r)
    for rs in by_qm.values():
        T, P = 5, rs[0]["P"]
        for r in rs:
            grows = r["policy"] == "carry_all"
            xs = [P] + [(P + (t - 1) * 400 if grows else P + 400)
                        for t in range(2, T + 1)]
            r2 = dict(r, turns=T, total_input=sum(xs), total_output=50 * T)
            newruns.append(r2)
            for s, x in enumerate(xs):
                newreqs.append({"run_id": r2["run_id"], "step": s, "input_tokens": x,
                                "cached_input_tokens": None, "output_tokens": 50,
                                "usd": None})
    extra = {"run_id": "x", "provider": "localx", "model": "m_a", "policy": "carry_all",
             "k": 2, "control": None, "question_id": "q99", "position": 99,
             "split": "train", "status": "ok", "turns": 9, "total_input": 99999,
             "total_cached": None, "total_output": 0, "P": 600, "latency_s": 1.0,
             "finished": True, "em": 0, "f1": 0.0, "usd": None, "truncated": False}
    out = az.h3(newruns + [extra], n_boot=50, seed=7)
    assert out["rho"] is None and out["rho_ci"] is None
    assert out["note"] and "I alone" in out["note"]
    paired = [r for r in newruns if r["question_id"] != "q99"
              and r["policy"] == "carry_all" and int(r["k"]) == 2]
    assert out["mean_T"]["carry_all|k=2"] == pytest.approx(
        float(np.mean([r["turns"] for r in paired])))


def test_item7_exclusions_report_control_runs_separately():
    runs, _ = synth(n_q=4)
    runs[0]["control"] = "broken_reset"
    runs[1]["control"] = "smoke"
    ex = az.exclusions(runs)
    assert ex["n_excluded"] == 0
    assert ex["control_runs"]["n"] == 2
    assert ex["control_runs"]["by_control"] == {"broken_reset": 1, "smoke": 1}
    assert len(az.analysable(runs)) == len(runs) - 2
    runs[2]["status"] = "error"
    runs[3]["truncated"] = True
    ex = az.exclusions(runs)
    assert ex["n_excluded"] == 2
    assert ex["by_reason"] == {"status": 1, "truncated": 1}
    assert ex["control_runs"]["n"] == 2


def test_item8_evidence_envelope(tmp_path):
    runs, reqs = synth(n_q=6)
    d = tmp_path / "gridA"
    d.mkdir()
    (d / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in runs),
                                  encoding="utf-8")
    (d / "requests.jsonl").write_text("".join(json.dumps(q) + "\n" for q in reqs),
                                      encoding="utf-8")
    (d / "not_run.jsonl").write_text(
        json.dumps({"provider": "gemini", "model": "gemini-2.5-flash",
                    "policy": "carry_all", "k": 2, "reason": "no key"}) + "\n",
        encoding="utf-8")
    (d / "provenance.json").write_text(json.dumps({
        "dataset_sha256": "ab" * 32, "git_sha": "c" * 40,
        "github_run_url": "https://github.com/owner/repo/actions/runs/12345"}),
        encoding="utf-8")

    ev = az._clean(az.run_evidence([str(d)], n_boot=20, n_perm=2, control1_path=None))
    json.dumps(ev)  # must be JSON-serializable as emitted

    assert set(ev) == {"schema_version", "kind", "provenance", "aggregates", "figures"}
    assert ev["schema_version"] == 1 and ev["kind"] == "agent_runs"

    prov = ev["provenance"]
    assert set(prov) == {"generated_at", "generator", "notes", "sources"}
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", prov["generated_at"])
    assert prov["generator"] == "experiments/analyze.py"
    assert isinstance(prov["notes"], list) and all(isinstance(n, str) for n in prov["notes"])
    src_ids = [s["id"] for s in prov["sources"]]
    assert "prereg" in src_ids and "grida" in src_ids  # dir "gridA" sanitized
    for s in prov["sources"]:
        assert re.fullmatch(r"[a-z][a-z0-9_]{2,40}", s["id"])
        assert s["grade"] in ("measured", "documented-record")
        assert re.fullmatch(r"[0-9a-f]{64}", s["sha256"])
        assert isinstance(s["command"], str) and s["command"]
        assert isinstance(s["description"], str) and len(s["description"]) >= 10
    meas = next(s for s in prov["sources"] if s["id"] == "grida")
    assert meas["grade"] == "measured"
    assert "https://github.com/owner/repo/actions/runs/12345" in meas["description"]
    assert meas["sha256"] == hashlib.sha256(
        (d / "runs.jsonl").read_bytes()).hexdigest()
    pre = next(s for s in prov["sources"] if s["id"] == "prereg")
    assert pre["grade"] == "documented-record"
    assert pre["sha256"] == hashlib.sha256(
        Path(az.PREREG_PATH).read_bytes()).hexdigest()

    for key in ("prereg", "provenance", "cells", "not_run", "exclusions",
                "tokenledger", "hypotheses", "controls", "cloud_spend_usd"):
        assert key in ev["aggregates"], key
    assert ev["aggregates"]["not_run"][0]["model"] == "gemini-2.5-flash"

    # every CI in the aggregates carries n_valid_boot / unstable metadata
    dci = ev["aggregates"]["hypotheses"]["H1"]["delta_mape_ci"]
    assert dci["n_valid_boot"] >= 1 and dci["unstable"] is False and len(dci["ci"]) == 2
    assert ev["aggregates"]["exclusions"]["control_runs"]["n"] == 0

    figs = ev["figures"]
    assert figs["AgentNQuestions"]["value"] == 48
    assert figs["AgentNQuestions"]["format"] == "int"
    assert figs["AgentNQuestions"]["source"] == "prereg"
    assert figs["AgentTurnCap"]["value"] == 10 and figs["AgentKLow"]["value"] == 2
    assert figs["AgentKHigh"]["value"] == 8 and figs["AgentNTestQuestions"]["value"] == 16
    assert figs["AgentNBoot"]["value"] == 20 and figs["AgentNPerm"]["value"] == 2
    assert figs["AgentPreregShaShort"]["format"] == "text"
    assert figs["AgentPreregShaShort"]["value"] == az._prereg_sha()[:7]
    # H5 has no cloud runs: reported as not-yet-run text figures
    assert figs["AgentHFiveDecision"]["value"] == "not yet run"
    assert figs["AgentHFiveDecision"]["format"] == "text"
    for name, f in figs.items():
        assert re.fullmatch(r"Agent[A-Za-z]+", name) and 3 <= len(name) <= 60
        assert set(f) == {"value", "format", "source", "desc"}
        assert f["format"] in az.FIG_FORMATS
        assert f["source"] in src_ids
        assert isinstance(f["desc"], str) and f["desc"]
        if f["format"] != "text":
            assert isinstance(f["value"], (int, float)), (name, f["value"])
