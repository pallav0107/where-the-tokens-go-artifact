"""Regression tests for the budget / spend-guard defects (PREREG Amendment 1, A4).

No network: urllib.request.urlopen is replaced by in-process handlers.
"""
import json
import os
import shutil
import subprocess
import urllib.error
from pathlib import Path

import pytest
import yaml

from agent import run_grid
from agent.agent import run_agent
from agent.budget import collect_prior_spend
from agent.providers import make_provider
from agent.tests.test_impl import (FIXTURE, _gm, _http_error, _patch_http, _quota_setup,
                                   _retry_body)

REPO = Path(__file__).resolve().parents[2]
FINISH = '{"thought":"t","notes":"n","action":"finish","args":{"answer":"x"}}'
SEARCH = '{"thought":"t","notes":"n","action":"search","args":{"query":"x"}}'
FLASH = "gemini-3.8-flash"  # 0.75 USD per 1M input tokens, 3.75 per 1M output


def _cand(text, prompt_tokens, out=0, **extra):
    um = {"promptTokenCount": prompt_tokens, "candidatesTokenCount": out}
    um.update(extra)
    return {"candidates": [{"content": {"parts": [{"text": text}]}}], "usageMetadata": um}


def _is_step0(req):
    body = json.loads(req.data.decode())
    return not any(c.get("role") == "model" for c in body["contents"])


def _grid(tmp_path, models, argv=(), policies=("carry_all",), ks=(2,), n_questions=3,
          out_name="out"):
    cfg = {"experiment": "t", "dataset": {"parquet_url": None, "sha256": None, "cache_dir": None,
                                          "fixture": FIXTURE, "sample_seed": 20261115,
                                          "n_questions": n_questions, "n_smoke": 0},
           "agent": {"max_steps": 10, "cell_order_seed": 20261116},
           "policies": list(policies), "ks": list(ks), "models": models,
           "budget": {"max_usd": 5, "est_input_tokens_per_run": 20000,
                      "est_output_tokens_per_run": 1500}}
    cfg_path = tmp_path / "g.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = tmp_path / out_name
    assert run_grid.main(["--config", str(cfg_path), "--out", str(out), *argv]) == 0
    return out


def _load(out, name):
    p = out / name
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def _prov(out):
    return json.loads((out / "provenance.json").read_text())


GEMINI = [{"provider": "gemini", "model": "auto-flash", "thinking_budget": 0}]


# ---- A1 + A7: end-of-grid retry honours the hard stop; provider resolved once ------------

def test_a1_retry_loop_respects_hard_stop_and_a7_resolves_model_once(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    state = {"posts": 0, "gets": 0, "ok_posts": 0}

    def handler(req):
        if req.get_method() == "GET":
            state["gets"] += 1
            return {"models": [_gm(FLASH)]}
        state["posts"] += 1
        if state["posts"] <= 4:  # the first run's request fails all 4 tries -> error, no spend
            raise _http_error(req.full_url, 500, {"error": "boom"})
        state["ok_posts"] += 1
        return _cand(FINISH, 800_000, 1)  # 0.60 USD per run

    _patch_http(monkeypatch, handler)
    out = _grid(tmp_path, GEMINI, ["--max-usd", "1.0"])
    runs = _load(out, "runs.jsonl")
    by_status = sorted(r["status"] for r in runs)
    assert by_status == ["error", "ok", "ok"], by_status
    # spend reached 1.2 >= 1.0 after the 2nd/3rd run: the errored run must NOT be retried
    assert state["ok_posts"] == 2
    assert _prov(out)["cloud_spend_total"] == pytest.approx(1.2, abs=1e-4)
    nr = _load(out, "not_run.jsonl")
    assert nr and all(n["reason"].startswith("budget exhausted") for n in nr)
    assert state["gets"] == 1  # A7: the model list is fetched once per block, not per run


def test_a1_retry_spend_is_added(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    state = {"posts": 0}

    def handler(req):
        if req.get_method() == "GET":
            return {"models": [_gm(FLASH)]}
        state["posts"] += 1
        if state["posts"] <= 4:
            raise _http_error(req.full_url, 500, {"error": "boom"})
        return _cand(FINISH, 100_000, 1)  # 0.075 USD per run

    _patch_http(monkeypatch, handler)
    out = _grid(tmp_path, GEMINI, ["--max-usd", "5"])
    runs = _load(out, "runs.jsonl")
    assert len(runs) == 3 and all(r["status"] == "ok" for r in runs)  # retry succeeded
    assert _prov(out)["cloud_spend_total"] == pytest.approx(0.225, abs=1e-4)  # 3 runs counted


# ---- A2: spend from failed / partial attempts --------------------------------------------

@pytest.mark.parametrize("failure", ["error", "quota"])
def test_a2_failed_attempt_spend_counts(tmp_path, monkeypatch, failure):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def handler(req):
        if req.get_method() == "GET":
            return {"models": [_gm(FLASH)]}
        if _is_step0(req):
            return _cand(SEARCH, 400_000, 0)  # 0.30 USD
        if failure == "quota":
            raise _http_error(req.full_url, 429, {"error": {
                "message": "PerDay quota", "status": "RESOURCE_EXHAUSTED"}})
        raise _http_error(req.full_url, 500, {"error": "boom"})

    _patch_http(monkeypatch, handler)
    out = _grid(tmp_path, GEMINI, ["--max-usd", "50"], n_questions=1)
    failed = _load(out, "requests_failed.jsonl")
    assert failed and sum(r["usd"] for r in failed) > 0
    assert _prov(out)["cloud_spend_total"] == pytest.approx(sum(r["usd"] for r in failed),
                                                           abs=1e-5)


# ---- A3: --spend-from, prior_spend breakdown ---------------------------------------------

def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_a3_collect_prior_spend_counts_each_file_once(tmp_path):
    d = tmp_path / "prev"
    _write_jsonl(d / "grid" / "requests.jsonl", [{"usd": 1.0}, {"usd": 0.5}])
    _write_jsonl(d / "grid" / "requests_failed.jsonl", [{"usd": 0.25}])
    (d / "literature").mkdir()
    (d / "literature" / "embedding_log.json").write_text(json.dumps({"est_usd": 0.125}))
    # d and a nested dir under it: every file is reachable twice
    info = collect_prior_spend([d, d, d / "grid"], extra_usd=0.1)
    assert info["total"] == pytest.approx(1.975)
    assert len(info["requests"]) == 2 and len(info["embedding_logs"]) == 1
    assert info["flag_usd"] == 0.1


def test_a3_run_grid_spend_from_and_resumed_out_dir(tmp_path):
    d = tmp_path / "prev"
    _write_jsonl(d / "smoke" / "requests.jsonl", [{"usd": 1.0}])
    _write_jsonl(d / "control_broken" / "requests_failed.jsonl", [{"usd": 0.5}])
    (d / "literature").mkdir()
    (d / "literature" / "embedding_log.json").write_text(json.dumps({"est_usd": 0.25}))
    out = tmp_path / "out"
    _write_jsonl(out / "requests_failed.jsonl", [{"usd": 0.125}])  # resumed job's own records
    models = [{"provider": "mock", "model": "mock-scripted"}]
    _grid(tmp_path, models, ["--spend-from", str(d), "--spend-from", str(d),
                             "--prior-spend-usd", "0.5", "--max-usd", "9"])
    prov = _prov(out)
    assert prov["prior_spend_usd"] == pytest.approx(2.375)
    ps = prov["prior_spend"]
    assert ps["flag_usd"] == 0.5 and ps["total"] == pytest.approx(2.375)
    assert sum(ps["embedding_logs"].values()) == pytest.approx(0.25)
    assert len(ps["requests"]) == 3  # smoke + control_broken failed + resumed out/requests_failed
    assert prov["cloud_spend_total"] == pytest.approx(2.375)


def test_a3_prior_spend_blocks_gemini_block_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    d = tmp_path / "prev"
    _write_jsonl(d / "smoke" / "requests.jsonl", [{"usd": 4.99}])
    _patch_http(monkeypatch, lambda req: {"models": [_gm(FLASH)]})
    out = _grid(tmp_path, GEMINI, ["--spend-from", str(d), "--max-usd", "5"])
    nr = _load(out, "not_run.jsonl")
    assert len(nr) == 1 and nr[0]["reason"].startswith("budget:")


def _stub_env(tmp_path):
    """bash + stub pip/python on PATH so job.sh can be exercised without installing anything."""
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash not available")
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    log = tmp_path / "calls.log"
    (stubs / "pip").write_text("#!/bin/sh\nexit 0\n", newline="\n")
    (stubs / "python").write_text(
        '#!/bin/sh\necho "$@" >> "$CALLS_LOG"\n'
        'case "$*" in "-m agent.budget"*) echo 3.000000;; esac\ncat >/dev/null 2>&1 </dev/null\n'
        "exit 0\n", newline="\n")
    for f in ("pip", "python"):
        os.chmod(stubs / f, 0o755)
    env = dict(os.environ, CALLS_LOG=str(log), PATH=str(stubs) + os.pathsep + os.environ["PATH"])
    for v in ("JOB", "MODELS", "QUESTIONS", "FIRST_IN_CHAIN", "MAX_USD", "PRIOR_SPEND_USD"):
        env.pop(v, None)
    return bash, env, log


def _job(tmp_path, bash, env, **vars_):
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    script = (REPO / "experiments" / "ci" / "job.sh").as_posix()
    return subprocess.run([bash, script], cwd=str(work), env={**env, **vars_},
                          capture_output=True, text=True, timeout=60), work


@pytest.mark.parametrize("job", ["smoke", "control-broken", "grid", "literature", "h2h"])
def test_a3_job_refuses_without_prev_unless_first_in_chain(tmp_path, job):
    bash, env, log = _stub_env(tmp_path)
    r, work = _job(tmp_path, bash, env, JOB=job)
    assert r.returncode == 2 and "REFUSING" in r.stderr, (r.returncode, r.stderr)
    assert not log.exists()  # nothing was run
    r2, _ = _job(tmp_path, bash, env, JOB=job, FIRST_IN_CHAIN="true")
    assert "REFUSING" not in r2.stderr


def test_h2h_job_runs_the_split_into_its_own_dir_with_the_spend_guard(tmp_path):
    bash, env, log = _stub_env(tmp_path)
    r, work = _job(tmp_path, bash, env, JOB="h2h", SPLIT="dev", FIRST_IN_CHAIN="true")
    assert r.returncode == 0, r.stderr
    line = [l for l in log.read_text().splitlines() if "agent.run_grid" in l][0]
    assert "--config experiments/configs/h2h.yaml" in line and "--split dev" in line
    assert "--out out/h2h_dev" in line and "--spend-from out" in line
    assert any("analyze_v2.py" in l and "h2h_dev" in l for l in log.read_text().splitlines())
    r, _ = _job(tmp_path, bash, env, JOB="h2h", SPLIT="bogus", FIRST_IN_CHAIN="true")
    assert r.returncode == 2


def test_a3_a4_job_copies_all_prev_passes_spend_from_and_forces_lite(tmp_path):
    bash, env, log = _stub_env(tmp_path)
    work = tmp_path / "work"
    for sub in ("literature", "smoke", "control_broken", "grid"):
        (work / "prev" / sub).mkdir(parents=True)
        (work / "prev" / sub / "marker.txt").write_text(sub)
    r, work = _job(tmp_path, bash, env, JOB="control-broken", MODELS="auto-flash,auto-flash-lite")
    assert r.returncode == 0, r.stderr
    for sub in ("literature", "smoke", "control_broken", "grid"):
        assert (work / "out" / sub / "marker.txt").read_text() == sub
    line = [l for l in log.read_text().splitlines() if "agent.run_grid" in l][0]
    assert "--spend-from out" in line
    assert "--models auto-flash-lite" in line and "auto-flash," not in line  # A4: G-Lite only


def test_a3_literature_job_passes_remaining_budget(tmp_path):
    bash, env, log = _stub_env(tmp_path)
    (tmp_path / "work" / "prev").mkdir(parents=True)
    (tmp_path / "work" / "literature").mkdir()
    for f in ("metadata.json", "refs.bib", "chunks_manifest.jsonl", "embedding_log.json"):
        (tmp_path / "work" / "literature" / f).write_text("{}")
    r, work = _job(tmp_path, bash, env, JOB="literature", MAX_USD="5")
    assert r.returncode == 0, r.stderr
    assert (work / "out" / "literature" / "embedding_log.json").exists()
    calls = log.read_text()
    assert "agent.budget remaining" in calls and "--spend-from out" in calls
    assert "build_index.py --embed gemini --budget-usd 3.000000" in calls


# ---- A5: usage_metadata and thinking_config per request ----------------------------------

def test_a5_request_records_carry_usage_metadata_and_thinking_config(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    um_extra = {"thoughtsTokenCount": 7, "cachedContentTokenCount": 3}
    _patch_http(monkeypatch, lambda req: _cand(FINISH, 50, 4, **um_extra))
    prov = make_provider({"provider": "gemini", "model": "gemini-3.8-flash"})
    recs = []

    class L:
        def log(self, r):
            recs.append(r)

    run_agent("q", prov, "carry_all", 2, tools=None, logger=L(),
              run_meta={"run_id": "r", "question_id": "q", "turn": 1, "control": None})
    assert recs[0]["usage_metadata"] == {"promptTokenCount": 50, "candidatesTokenCount": 4,
                                         **um_extra}
    assert recs[0]["thinking_config"] == {"thinkingLevel": "minimal"}
    mock = make_provider({"provider": "mock", "model": "mock-scripted"})
    recs.clear()
    run_agent("q", mock, "carry_all", 2, tools={"search": lambda q, k: "[a] x",
                                               "read": lambda d: "[a] x: y z"}, logger=L())
    assert recs and all(r["usage_metadata"] is None and r["thinking_config"] is None
                        for r in recs)


# ---- A6: not_run.jsonl history ------------------------------------------------------------

def test_a6_stale_not_run_does_not_block_resumed_job(tmp_path):
    out = tmp_path / "out"
    stale = {"provider": "mock", "model": "mock-scripted", "policy": "carry_all", "k": 2,
             "reason": "budget: earlier invocation"}
    _write_jsonl(out / "not_run.jsonl", [stale])
    models = [{"provider": "mock", "model": "mock-scripted"}]
    _grid(tmp_path, models)
    runs = _load(out, "runs.jsonl")
    assert len(runs) == 3 and all(r["status"] == "ok" for r in runs)  # cell was not dead
    assert _load(out, "not_run.jsonl") == []
    assert _load(out, "not_run_history.jsonl") == [stale]


# ---- A7: unpriced gemini id never counts as 0 --------------------------------------------

def test_a7_unpriced_request_stops_block(tmp_path, monkeypatch):
    import agent.budget as budget
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(budget, "request_usd", lambda *a, **k: None)
    posts = []

    def handler(req):
        if req.get_method() == "GET":
            return {"models": [_gm(FLASH)]}
        posts.append(1)
        return _cand(FINISH, 10, 1)

    _patch_http(monkeypatch, handler)
    out = _grid(tmp_path, GEMINI, ["--max-usd", "5"])
    assert len(posts) == 1  # block stopped after the first (unpriced) run
    nr = _load(out, "not_run.jsonl")
    assert nr and all(n["reason"] == f"price: unpriced model id {FLASH}" for n in nr)


# ---- A8: resolution failure reason --------------------------------------------------------

def test_a8_resolution_failure_is_labelled_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def handler(req):
        raise _http_error(req.full_url, 403, {"error": {"message": "denied"}})

    _patch_http(monkeypatch, handler)
    out = _grid(tmp_path, GEMINI, ["--max-usd", "5"])
    nr = _load(out, "not_run.jsonl")
    assert len(nr) == 1 and nr[0]["reason"].startswith("resolution: "), nr
    assert "price" not in nr[0]["reason"]


# ---- A9: persistent errors do not poison a cell -------------------------------------------

def test_a9_errors_recorded_per_run_and_cell_stops_after_three(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

    def handler(req):
        raise urllib.error.HTTPError(req.full_url, 500, "boom", {}, None)

    _patch_http(monkeypatch, handler)
    models = [{"provider": "dashscope", "model": "qwen-flash", "enable_thinking": False}]
    out = _grid(tmp_path, models, ["--max-usd", "5"], n_questions=4)
    runs = _load(out, "runs.jsonl")
    assert len(runs) == 3 and all(r["status"] == "error" for r in runs)
    nr = _load(out, "not_run.jsonl")
    assert len(nr) == 1 and nr[0]["reason"] == "errors: 3 consecutive"
    assert nr[0]["question_id"] not in {r["question_id"] for r in runs}


def test_a9_intermittent_error_does_not_skip_later_questions(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    state = {"n": 0}

    def handler(req):
        state["n"] += 1
        if state["n"] <= 4:  # first run's request: 4 tries, all 500
            raise urllib.error.HTTPError(req.full_url, 500, "boom", {}, None)
        return {"choices": [{"message": {"content": FINISH}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2}}

    _patch_http(monkeypatch, handler)
    models = [{"provider": "dashscope", "model": "qwen-flash", "enable_thinking": False}]
    out = _grid(tmp_path, models, ["--max-usd", "5"], n_questions=4)
    runs = _load(out, "runs.jsonl")
    assert len(runs) == 4 and all(r["status"] == "ok" for r in runs)  # 3 later + 1 retried
    assert _load(out, "not_run.jsonl") == []


# ---- A11: retryDelay cap -------------------------------------------------------------------

def test_a11_server_retry_delay_is_capped_at_120s(monkeypatch):
    calls, sleeps = _quota_setup(monkeypatch, [(429, _retry_body("9999s"))])
    p = make_provider({"provider": "gemini", "model": "gemini-2.5-flash"})
    p.chat([{"role": "user", "content": "hi"}])
    assert sleeps == [120.0]


# ---- A12: remember the working thinkingLevel ------------------------------------------------

def test_a12_minimal_rejected_then_low_is_remembered(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    levels = []

    def handler(req):
        lvl = json.loads(req.data.decode())["generationConfig"]["thinkingConfig"]["thinkingLevel"]
        levels.append(lvl)
        if lvl == "minimal":
            raise _http_error(req.full_url, 400, {"error": {
                "message": "Thinking level minimal is not supported for this model"}})
        return _cand("{}", 1, 1)

    _patch_http(monkeypatch, handler)
    p = make_provider({"provider": "gemini", "model": "gemini-3.8-flash"})
    r1 = p.chat([{"role": "user", "content": "a"}])
    r2 = p.chat([{"role": "user", "content": "b"}])
    assert levels == ["minimal", "low", "low"]  # second chat goes straight to "low"
    assert r2.usage.raw["thinking_config"] == {"thinkingLevel": "low"}
    assert r1.usage.raw["thinking_config"] == {"thinkingLevel": "low"}
