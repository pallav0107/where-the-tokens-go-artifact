"""H7 and the evidence envelope of experiments/analyze_v2.py on synthetic runs."""
import json
import os
import subprocess
import sys

import pytest

import experiments.analyze_v2 as a2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from evidence_lib import SCHEMA_DIR, load_json, validate  # noqa: E402

N = 20


def point(arm, k, cost, f1, n=N, grounded=None):
    """n runs; the first round(f1*n) questions score F1 = 1, the rest 0; cost wobbles per
    question by a factor shared across arms, so cost ratios are exact per question."""
    runs = []
    for i in range(n):
        runs.append({
            "status": "ok", "arm": arm, "k": k, "split": "test", "question_id": f"q{i}",
            "usd_list": cost * (1 + 0.05 * ((i % 5) - 2)),
            "total_input": 1000, "total_cached": 100, "total_output": 50, "turns": 3,
            "n_requests": 3, "em": 1 if i < round(f1 * n) else 0,
            "f1": 1.0 if i < round(f1 * n) else 0.0, "grounded": grounded})
    return runs


def build(*pts):
    return [r for p in pts for r in p]


def decide(runs):
    return a2.h7(a2.usable(runs, "test"), n_boot=500)


def base_points():  # fresh dicts every call: tests mutate runs
    return [point("react", 8, 1.0, 0.60), point("notes_reset", 2, 0.5, 0.40),
            point("agentless", 8, 0.2, 0.30)]


def test_h7_supported_when_pbc_is_much_cheaper_and_as_accurate():
    out = decide(build(*base_points(), point("pbc", 3, 0.5, 0.60)))
    assert out["decision"] == "supported"
    assert out["bstar"]["arm"] == "react" and out["bstar"]["k"] == 8
    assert out["cost_ratio"] == pytest.approx(0.5) and out["cost_ratio_ci"][1] < 1.0
    assert out["rule_a"] and out["rule_b"] and out["rule_c"]
    assert out["f1_diff"] == pytest.approx(0.0)


def test_h7_not_supported_when_pbc_loses_more_than_two_f1_points():
    out = decide(build(*base_points(), point("pbc", 3, 0.5, 0.45)))
    assert out["decision"] == "not supported"
    assert out["rule_a"] and not out["rule_b"]
    assert out["f1_diff"] == pytest.approx(-0.15)


def test_h7_two_point_boundary_is_inclusive():
    # N=20 questions: 1 question = 5 F1 points, so use 100 questions for a 1-point step
    base = [point("react", 8, 1.0, 0.60, n=100)]
    ok = decide(build(*base, point("pbc", 3, 0.5, 0.58, n=100)))   # exactly -2.0 points
    bad = decide(build(*base, point("pbc", 3, 0.5, 0.57, n=100)))  # -3.0 points
    assert ok["rule_b"] and ok["decision"] == "supported"
    assert not bad["rule_b"] and bad["decision"] == "not supported"


def test_h7_not_supported_when_the_saving_is_too_small():
    out = decide(build(*base_points(), point("pbc", 3, 0.8, 0.60)))  # ratio 0.8 > 0.70
    assert out["decision"] == "not supported" and not out["rule_a"]


def test_h7_not_supported_when_pbc_is_dominated_by_a_baseline():
    # a (ratio 0.5) and b (exactly -2 points) hold, but a cheaper baseline with higher F1 exists
    base = [point("react", 8, 1.0, 0.60, n=100),
            point("agentless", 8, 0.3, 0.59, n=100)]
    out = decide(build(*base, point("pbc", 3, 0.5, 0.58, n=100)))
    assert out["rule_a"] and out["rule_b"] and not out["rule_c"]
    assert out["decision"] == "not supported" and out["dominated_by"] == ["agentless:8"]


def test_h7_bstar_ties_go_to_the_lower_cost_point():
    base = [point("react", 8, 1.0, 0.60), point("notes_reset", 8, 0.7, 0.60)]
    out = decide(build(*base, point("pbc", 3, 0.3, 0.60)))
    assert (out["bstar"]["arm"], out["bstar"]["k"]) == ("notes_reset", 8)


def test_h7_not_yet_run_without_pbc_or_baselines_or_prices():
    assert decide([])["decision"] == "not yet run"
    assert decide(build(*base_points()))["decision"] == "not yet run"
    assert decide(point("pbc", 3, 0.5, 0.6))["decision"] == "not yet run"
    unpriced = build(*base_points(), point("pbc", 3, 0.5, 0.6))
    for r in unpriced:
        r["usd_list"] = None
    assert decide(unpriced)["decision"] == "not yet run"


def test_operating_points_and_frontier():
    pts = a2.operating_points(a2.usable(build(
        point("react", 8, 1.0, 0.6, grounded=None), point("pbc", 3, 0.5, 0.6, grounded=True),
        point("window2", 8, 0.9, 0.5)), "test"))
    by = {(p["arm"], p["k"]): p for p in pts}
    assert by[("pbc", 3)]["grounded"] == 1.0 and by[("react", 8)]["grounded"] is None
    assert by[("react", 8)]["n"] == N and by[("react", 8)]["requests"] == 3
    assert by[("react", 8)]["cost"] == pytest.approx(1.0)
    assert [(p["arm"], p["k"]) for p in a2.frontier(pts)] == [("pbc", 3)]  # dominates both


def test_split_filter_keeps_dev_and_test_apart():
    runs = point("pbc", 3, 0.5, 0.6)
    for r in runs[:5]:
        r["split"] = "dev"
    assert len(a2.usable(runs, "dev")) == 5 and len(a2.usable(runs, "test")) == N - 5


# ------------------------------------------------------------ evidence envelope

def _write_dir(tmp_path, runs, name="h2h_test"):
    d = tmp_path / name
    d.mkdir()
    (d / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in runs), encoding="utf-8")
    return d


def _envelope(tmp_path, runs):
    d = _write_dir(tmp_path, runs)
    out = tmp_path / "agent_h2h.json"
    cwd = os.getcwd()
    os.chdir(tmp_path)  # relative source paths, as in CI
    try:
        assert a2.main(["--input", d.name, "--out", str(out), "--n-boot", "200"]) == 0
    finally:
        os.chdir(cwd)
    return json.loads(out.read_text(encoding="utf-8"))


def test_envelope_is_schema_valid_with_hth_figures(tmp_path):
    env = _envelope(tmp_path, build(*base_points(), point("pbc", 3, 0.5, 0.60, grounded=True)))
    assert validate(env, load_json(SCHEMA_DIR / "envelope.schema.json")) == []
    assert env["kind"] == "agent_h2h" and env["aggregates"]["H7"]["decision"] == "supported"
    figs = env["figures"]
    assert all(n.startswith("Hth") and n.isalpha() for n in figs)
    assert figs["HthDecision"]["value"] == "supported"
    assert figs["HthCostRatio"]["value"] == pytest.approx(0.5)
    assert figs["HthPbcKThreeGrounded"]["value"] == 1.0
    assert figs["HthReactKEightN"]["value"] == N
    assert "http" not in json.dumps(env).replace("github.com", "")


def test_envelope_when_nothing_ran_says_not_yet_run(tmp_path):
    env = _envelope(tmp_path, [])
    assert validate(env, load_json(SCHEMA_DIR / "envelope.schema.json")) == []
    assert env["aggregates"]["H7"]["decision"] == "not yet run"
    for name in ("HthDecision", "HthCostRatio", "HthFOneDiff", "HthRuleA", "HthBstarArm"):
        assert env["figures"][name] == {**env["figures"][name], "value": "not yet run",
                                        "format": "text"}


def test_cli_runs_as_a_script(tmp_path):
    d = _write_dir(tmp_path, build(*base_points(), point("pbc", 3, 0.5, 0.60)))
    out = tmp_path / "x.json"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "analyze_v2.py"),
                        "--input", str(d), "--out", str(out), "--n-boot", "100"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "H7=supported" in r.stdout
