"""Evidence files keep the https:// scheme of Actions run URLs (tools/validate_evidence.py
accepts it since f64acc9); the producers used to strip it."""
import json
import os

import experiments.analyze as az
import experiments.analyze_v2 as a2

URL = "https://github.com/owner/repo/actions/runs/12345"


def _grid_dir(tmp_path, name):
    d = tmp_path / name
    d.mkdir()
    (d / "runs.jsonl").write_text("", encoding="utf-8")
    (d / "provenance.json").write_text(json.dumps({"github_run_url": URL}), encoding="utf-8")
    return d


def _run(tmp_path, main, argv):
    cwd = os.getcwd()
    os.chdir(tmp_path)  # relative source paths, as in CI
    try:
        assert main(argv) == 0
    finally:
        os.chdir(cwd)


def test_analyze_writes_run_urls_with_their_scheme(tmp_path):
    d = _grid_dir(tmp_path, "probe_1")
    out = tmp_path / "agent_runs.json"
    _run(tmp_path, az.main, ["--input", d.name, "--out", str(out), "--n-boot", "20",
                             "--n-perm", "2"])
    text = out.read_text(encoding="utf-8")
    assert URL in text and '"github.com/' not in text and " github.com/" not in text
    ev = json.loads(text)
    assert ev["aggregates"]["provenance"]["inputs"][0]["github_run_url"] == URL


def test_analyze_v2_writes_run_urls_with_their_scheme(tmp_path):
    d = _grid_dir(tmp_path, "h2h_test")
    out = tmp_path / "agent_h2h.json"
    _run(tmp_path, a2.main, ["--input", d.name, "--out", str(out), "--n-boot", "20"])
    text = out.read_text(encoding="utf-8")
    assert URL in text and " github.com/" not in text


def test_h2h_model_figures_registered_and_returned(tmp_path):
    """PREREG_v2 Amendment 2: the paper names the model through HthModel (registered, a
    documented record) and HthModelReturned (the ids the API returned, measured)."""
    d = tmp_path / "h2h_dev"
    d.mkdir()
    run = {"run_id": "r1", "status": "ok", "arm": "react", "split": "dev", "k": 2,
           "question_id": "q", "total_input": 10, "total_cached": 0, "total_output": 1,
           "turns": 1, "n_requests": 1, "em": 1, "f1": 1.0, "usd_list": 0.001,
           "grounded": None, "total_reasoning": 0, "total_cache_write": 8}
    (d / "runs.jsonl").write_text(json.dumps(run) + "\n", encoding="utf-8")
    (d / "requests.jsonl").write_text(json.dumps(
        {"run_id": "r1", "response_model": "gpt-6-luna"}) + "\n", encoding="utf-8")
    out = tmp_path / "h2h.json"
    _run(tmp_path, a2.main, ["--input", d.name, "--split", "dev", "--out", str(out),
                             "--n-boot", "20"])
    ev = json.loads(out.read_text(encoding="utf-8"))
    figs = ev["figures"]
    assert figs["HthModel"]["value"] == "OpenAI gpt-6-luna"
    assert figs["HthModel"]["source"] == "prereg"
    assert figs["HthModelReturned"]["value"] == "gpt-6-luna"
    point = ev["aggregates"]["points"][0]
    assert point["reasoning"] == 0 and point["cache_write"] == 8
    empty = tmp_path / "none"
    empty.mkdir()
    out2 = tmp_path / "h2h0.json"
    _run(tmp_path, a2.main, ["--input", empty.name, "--out", str(out2), "--n-boot", "20"])
    figs0 = json.loads(out2.read_text(encoding="utf-8"))["figures"]
    assert figs0["HthModel"]["value"] == "OpenAI gpt-6-luna"
    assert figs0["HthModelReturned"]["value"] == "not yet run"
