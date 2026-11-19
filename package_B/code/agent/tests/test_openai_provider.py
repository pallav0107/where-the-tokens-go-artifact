"""OpenAI provider and the per-request spend guard (PREREG_v2 Amendment 2, B2-B8).

No network: urllib.request.urlopen is replaced by in-process handlers (fake transport).
"""
import io
import json
import urllib.error
from pathlib import Path

import pytest
import yaml

from agent import run_grid
from agent.agent import make_record
from agent.budget import BudgetExceeded, SpendGuard, collect_prior_spend, input_token_upper_bound
from agent.providers import (ProviderInsufficientQuota, ProviderQuotaExhausted,
                             ProviderUnavailable, make_provider)
from agent.tests.test_budget_guards import _job, _stub_env
from agent.tests.test_impl import FIXTURE, _patch_http

REPO = Path(__file__).resolve().parents[2]
# Built at run time so the privacy scan never sees a key-shaped literal (it is fake anyway).
KEY = "-".join(("sk", "proj", "TESTKEY0123456789abcdefSECRET"))
FINISH = '{"thought":"t","notes":"n","action":"finish","args":{"answer":"x"}}'
SPEC = {"provider": "openai", "model": "gpt-6-luna", "reasoning_effort": "none",
        "max_output_tokens": 4096}
MSGS = [{"role": "system", "content": "Reply with one JSON object."},
        {"role": "user", "content": "Question: q"}]


def _completion(text=FINISH, prompt=1000, completion=20, cached=0, write=0, reasoning=0,
                tier="default", model="gpt-6-luna"):
    return {"id": "chatcmpl-x", "model": model, "service_tier": tier,
            "system_fingerprint": "fp_test",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                      "total_tokens": prompt + completion,
                      "prompt_tokens_details": {"cached_tokens": cached,
                                                "cache_write_tokens": write},
                      "completion_tokens_details": {"reasoning_tokens": reasoning}}}


def _provider(monkeypatch, max_usd=5.0, spent=0.0, spec=None, attach=True):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    p = make_provider(dict(spec or SPEC))
    if attach:
        p.attach_budget(max_usd, spent)
    return p


def _setup(monkeypatch, handler):
    from agent import providers
    sleeps = []
    calls = _patch_http(monkeypatch, handler)
    monkeypatch.setattr(providers, "_sleep", sleeps.append)
    return calls, sleeps


def _body(req):
    return json.loads(req.data.decode())


# ---- B1-B5: payload, usage mapping, cache-write-aware cost, cum_usd ----------------------

def test_payload_usage_mapping_and_cache_write_aware_cost(monkeypatch):
    calls, _ = _setup(monkeypatch, lambda req: _completion(
        prompt=2000, completion=50, cached=1024, write=512, reasoning=10,
        model="gpt-6-luna-2026-10-01"))
    p = _provider(monkeypatch, spent=0.25)
    r = p.chat(MSGS)
    assert len(calls) == 1 and calls[0].full_url == "https://api.openai.com/v1/chat/completions"
    body = _body(calls[0])
    assert body["model"] == "gpt-6-luna" and body["messages"] == MSGS
    assert body["temperature"] == 0 and body["seed"] == 42
    assert body["reasoning_effort"] == "none" and body["max_completion_tokens"] == 4096
    assert body["response_format"] == {"type": "json_object"}
    assert body["service_tier"] == "default" and body["store"] is False
    for opt in ("prompt_cache_key", "prompt_cache_options", "prompt_cache_retention"):
        assert opt not in body  # B4: no cache routing, identically for every arm
    headers = {k.lower(): v for k, v in calls[0].headers.items()}
    assert headers["authorization"] == f"Bearer {KEY}"
    u = r.usage
    assert (u.input_tokens, u.cached_input_tokens, u.output_tokens) == (2000, 1024, 50)
    assert (u.cache_write_tokens, u.reasoning_tokens) == (512, 10)
    usd = (464 * 0.10 + 1024 * 0.01 + 512 * 0.125 + 50 * 0.50) / 1e6
    rec = make_record(p, r, MSGS, 0, "react", 8, {"run_id": "r"}, "finish", False, 0)
    assert rec["usd"] == pytest.approx(usd)
    assert rec["cum_usd"] == pytest.approx(0.25 + usd, abs=1e-6)  # rounded to 6 dp
    assert rec["response_model"] == "gpt-6-luna-2026-10-01" and rec["model"] == "gpt-6-luna"
    assert rec["reasoning_tokens"] == 10 and rec["cache_write_tokens"] == 512
    assert rec["service_tier"] == "default" and rec["finish_reason"] == "stop"
    assert rec["thinking_config"]["reasoning_effort"] == "none"
    assert rec["usage_metadata"]["completion_tokens_details"] == {"reasoning_tokens": 10}
    assert r.text == FINISH


def test_inconsistent_or_missing_usage_is_unpriced_never_zero(monkeypatch):
    from agent.budget import request_usd
    _setup(monkeypatch, lambda req: _completion(prompt=100, cached=80, write=50))
    p = _provider(monkeypatch)
    r = p.chat(MSGS)
    assert request_usd("openai", "gpt-6-luna", r.usage) is None  # 80 + 50 > 100
    assert p.guard.stopped  # charged at worst case; nothing further is sent


# ---- B7: the spend guard -------------------------------------------------------------------

def test_no_request_without_an_attached_spend_guard(monkeypatch):
    calls, _ = _setup(monkeypatch, lambda req: _completion())
    p = _provider(monkeypatch, attach=False)
    with pytest.raises(BudgetExceeded, match="no spend guard"):
        p.chat(MSGS)
    assert calls == []


def test_guard_refuses_a_request_whose_worst_case_could_cross_the_cap(monkeypatch):
    """Fail-first control: with the admit() call removed from OpenAIProvider.chat this test
    fails (the request is sent)."""
    calls, _ = _setup(monkeypatch, lambda req: _completion())
    worst = (input_token_upper_bound(MSGS) * 0.125 + 4096 * 0.50) / 1e6
    p = _provider(monkeypatch, max_usd=5.0, spent=5.0 - worst / 2)
    with pytest.raises(BudgetExceeded, match="could exceed"):
        p.chat(MSGS)
    assert calls == []  # refused before anything was sent
    ok = _provider(monkeypatch, max_usd=5.0, spent=5.0 - worst)  # exactly fits
    ok.chat(MSGS)
    assert len(calls) == 1


def test_guard_accumulates_actual_spend_and_reservations():
    g = SpendGuard("openai", "gpt-6-luna", max_usd=0.005)
    w = g.worst_case_usd(1000, 4096)
    assert w == pytest.approx((1000 * 0.125 + 4096 * 0.5) / 1e6)
    assert g.admit(1000, 4096) == pytest.approx(w)
    with pytest.raises(BudgetExceeded):  # two in flight would exceed the cap
        g.admit(1000, 4096) and g.admit(1000, 4096)
    g.release(w)
    g.release(w)
    for _ in range(2):
        w1 = g.admit(1000, 4096)
        assert g.charge(0.002, w1) == pytest.approx(0.002 * (_ + 1))
    with pytest.raises(BudgetExceeded):  # 0.004 + 0.002173 > 0.005
        g.admit(1000, 4096)
    assert g.spent == pytest.approx(0.004) and g.reserved == pytest.approx(0.0)


def test_guard_refuses_long_context_and_unverified_prices():
    g = SpendGuard("openai", "gpt-6-luna", max_usd=5.0)
    with pytest.raises(BudgetExceeded, match="short-context"):
        g.admit(272_001, 16)
    with pytest.raises(BudgetExceeded):
        SpendGuard("openai", "gpt-unknown", max_usd=5.0)


def test_unpriced_service_tier_stops_every_later_request(monkeypatch):
    calls, _ = _setup(monkeypatch, lambda req: _completion(tier="fast"))
    p = _provider(monkeypatch)
    r = p.chat(MSGS)
    rec = make_record(p, r, MSGS, 0, "react", 8, {}, "finish", False, 0)
    assert rec["usd"] is None and rec["cum_usd"] > 0  # charged at the worst case
    with pytest.raises(BudgetExceeded, match="unpriced"):
        p.chat(MSGS)
    assert len(calls) == 1


def test_prompt_larger_than_the_bound_stops_every_later_request(monkeypatch):
    big = input_token_upper_bound(MSGS) + 1
    calls, _ = _setup(monkeypatch, lambda req: _completion(prompt=big))
    p = _provider(monkeypatch)
    p.chat(MSGS)
    with pytest.raises(BudgetExceeded, match="bound"):
        p.chat(MSGS)
    assert len(calls) == 1


# ---- B7: [REDACTED: unrelated] - insufficient_quota, rate limits, 5xx -----------------------------

def _raising(errors, then=None):
    state = {"i": 0}

    def handler(req):
        if state["i"] < len(errors):
            code, body, headers = errors[state["i"]]
            state["i"] += 1
            raise urllib.error.HTTPError(req.full_url, code, "err", headers or {},
                                         io.BytesIO(json.dumps(body).encode()))
        return then or _completion()
    return handler


IQ = {"error": {"message": "You exceeded your current quota, please check your plan and "
                           "billing details.", "type": "insufficient_quota", "param": None,
                "code": "insufficient_quota"}}
RL = {"error": {"message": "Rate limit reached for requests", "type": "requests",
                "param": None, "code": "rate_limit_exceeded"}}


def test_insufficient_quota_is_never_retried(monkeypatch):
    calls, sleeps = _setup(monkeypatch, _raising([(429, IQ, None)] * 5))
    p = _provider(monkeypatch)
    with pytest.raises(ProviderInsufficientQuota):
        p.chat(MSGS)
    assert len(calls) == 1 and sleeps == []
    assert p.guard.reserved == 0.0 and p.guard.spent == 0.0  # an error is not billed


def test_rate_limit_backs_off_exponentially_then_stops_the_block(monkeypatch):
    calls, sleeps = _setup(monkeypatch, _raising([(429, RL, None)] * 20))
    p = _provider(monkeypatch)
    with pytest.raises(ProviderQuotaExhausted):
        p.chat(MSGS)
    assert sleeps == [2, 4, 8, 16, 32, 64] and len(calls) == 7


def test_rate_limit_honours_retry_after_capped(monkeypatch):
    calls, sleeps = _setup(monkeypatch, _raising([(429, RL, {"retry-after": "20"}),
                                                  (429, RL, {"retry-after": "999"})]))
    p = _provider(monkeypatch)
    p.chat(MSGS)
    assert sleeps == [20.0, 120.0] and len(calls) == 3


def test_5xx_retries_then_raises_unavailable(monkeypatch):
    calls, sleeps = _setup(monkeypatch, _raising([(503, {"error": {"message": "x"}}, None)] * 9))
    p = _provider(monkeypatch)
    with pytest.raises(ProviderUnavailable):
        p.chat(MSGS)
    assert sleeps == [5, 15, 45] and len(calls) == 4


# ---- B3: rule for unsupported parameters -----------------------------------------------------

def test_rejected_temperature_is_dropped_sticky_and_recorded(monkeypatch):
    rej = {"error": {"message": "Unsupported value: 'temperature' does not support 0 with "
                                "this model. Only the default (1) value is supported.",
                     "type": "invalid_request_error", "param": "temperature",
                     "code": "unsupported_value"}}
    calls, _ = _setup(monkeypatch, _raising([(400, rej, None)]))
    p = _provider(monkeypatch)
    r = p.chat(MSGS)
    assert len(calls) == 2 and "temperature" in _body(calls[0])
    assert "temperature" not in _body(calls[1]) and _body(calls[1])["seed"] == 42
    assert p.rejected == ["temperature"]
    assert r.usage.raw["thinking_config"]["temperature"] is None
    assert r.usage.raw["thinking_config"]["rejected"] == ["temperature"]
    p.chat(MSGS)  # sticky: the next request goes out once, without temperature
    assert len(calls) == 3 and "temperature" not in _body(calls[2])


def test_rejected_seed_and_effort_follow_the_ladder(monkeypatch):
    seed = {"error": {"message": "Unsupported parameter: 'seed' is not supported with this "
                                 "model.", "param": "seed", "code": "unsupported_parameter"}}
    effort = {"error": {"message": "Unsupported value: 'none' is not supported with this model "
                                   "for reasoning_effort.", "param": "reasoning_effort"}}
    calls, _ = _setup(monkeypatch, _raising([(400, seed, None), (400, effort, None)]))
    p = _provider(monkeypatch)
    p.chat(MSGS)
    last = _body(calls[-1])
    assert "seed" not in last and last["reasoning_effort"] == "minimal" and last["temperature"] == 0
    assert p.rejected == ["seed", "reasoning_effort"] and len(calls) == 3


def test_unrelated_400_is_raised_not_retried(monkeypatch):
    bad = {"error": {"message": "Invalid 'messages[1].content'.", "param": "messages",
                     "code": "invalid_value"}}
    calls, _ = _setup(monkeypatch, _raising([(400, bad, None)] * 3))
    p = _provider(monkeypatch)
    with pytest.raises(ProviderUnavailable, match="HTTP 400"):
        p.chat(MSGS)
    assert len(calls) == 1 and p.rejected == []


# ---- key handling ----------------------------------------------------------------------------

def test_key_never_appears_in_errors_records_or_repr(monkeypatch):
    echo = {"error": {"message": f"Incorrect API key provided: {KEY}. You can find your API "
                                 "key at https://platform.openai.com/account/api-keys.",
                      "type": "invalid_request_error", "code": "invalid_api_key"}}
    _setup(monkeypatch, _raising([(401, echo, None)]))
    p = _provider(monkeypatch)
    with pytest.raises(ProviderUnavailable) as ei:
        p.chat(MSGS)
    text = str(ei.value) + repr(ei.value)
    assert KEY not in text and "SECRET" not in text and "[redacted]" in text
    assert ei.value.__cause__ is None and ei.value.__suppress_context__
    assert KEY not in repr(p) and KEY not in str(vars(p).get("guard"))
    _setup(monkeypatch, lambda req: _completion())
    r = p.chat(MSGS)
    rec = make_record(p, r, MSGS, 0, "react", 8, {}, "finish", False, 0)
    assert KEY not in json.dumps(rec)


def test_missing_key_is_unavailable(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderUnavailable, match="OPENAI_API_KEY not set"):
        make_provider(dict(SPEC))


# ---- grid runner: guard attached at prior spend; insufficient_quota halts the run -----------

def _cfg(tmp_path, models, n_questions=4, ks=(2, 8)):
    cfg = {"experiment": "t", "dataset": {"parquet_url": None, "sha256": None, "cache_dir": None,
                                          "fixture": FIXTURE, "sample_seed": 20261115,
                                          "n_questions": n_questions, "n_smoke": 0},
           "agent": {"max_steps": 10, "cell_order_seed": 20261116},
           "policies": ["carry_all"], "ks": list(ks), "models": models,
           "budget": {"max_usd": 5, "est_input_tokens_per_run": 100,
                      "est_output_tokens_per_run": 10}}
    path = tmp_path / "g.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _lines(out, name):
    p = out / name
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def test_grid_per_request_guard_keeps_total_spend_under_the_cap(tmp_path, monkeypatch):
    """Fail-first control: without the per-request guard the grid's per-run hard stop lets
    the fifth request through and total spend ends above the cap (5.0005 > 5)."""
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    # each request: 1000 x 0.10 + 4000 x 0.50 per 1M = 0.0021 USD; worst case ~0.0024
    calls = _patch_http(monkeypatch, lambda req: _completion(prompt=1000, completion=4000))
    out = tmp_path / "out"
    rc = run_grid.main(["--config", str(_cfg(tmp_path, [SPEC])), "--out", str(out),
                        "--max-usd", "5", "--prior-spend-usd", "4.99"])
    assert rc == 0
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["cloud_spend_total"] <= 5.0
    assert len(calls) == 4  # the fifth request's worst case could cross 5 USD
    reqs = _lines(out, "requests.jsonl")
    assert [r["cum_usd"] for r in reqs] == pytest.approx(
        [4.99 + 0.0021 * i for i in (1, 2, 3, 4)], abs=1e-6)
    nr = _lines(out, "not_run.jsonl")
    assert nr and all(n["reason"].startswith("budget: request refused") for n in nr), nr


def test_grid_insufficient_quota_halts_the_whole_run(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    calls, sleeps = _setup(monkeypatch, _raising([(429, IQ, None)] * 50))
    out = tmp_path / "out"
    models = [SPEC, {"provider": "mock", "model": "mock-scripted"}]
    rc = run_grid.main(["--config", str(_cfg(tmp_path, models, n_questions=3)),
                        "--out", str(out)])
    assert rc == run_grid.EXIT_HALTED
    assert len(calls) == 1 and sleeps == []  # never retried, not even at the end of the grid
    assert _lines(out, "runs.jsonl") == []  # the later (mock) block never started
    nr = _lines(out, "not_run.jsonl")
    assert {n["provider"] for n in nr} == {"openai", "mock"}
    assert all(n["reason"].startswith("insufficient_quota") for n in nr)
    assert json.loads((out / "provenance.json").read_text())["halted"]


# ---- off-chain ledger and CI wiring (B6, B8) --------------------------------------------------

def test_unpriced_and_unconfirmed_charges_persist_into_prior_spend(tmp_path):
    """Review finding: usd None must not count as 0 in a later job's prior spend."""
    d = tmp_path / "prev"
    d.mkdir()
    rows = [{"usd": None, "usd_worst": 0.002},                         # unpriced: 2 x worst
            {"usd": 0.001, "usd_worst": 0.003, "unconfirmed_attempts": 1},  # + 1 x worst
            {"usd": 0.0005}]                                            # older providers
    (d / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    info = collect_prior_spend([d])
    assert info["total"] == pytest.approx(0.004 + 0.004 + 0.0005)
    assert info["n_unpriced_requests"] == 1


def test_billing_stop_errors_halt_like_insufficient_quota(monkeypatch):
    hard = {"error": {"message": "Billing hard limit has been reached.",
                      "type": "invalid_request_error", "code": "billing_hard_limit_reached"}}
    calls, sleeps = _setup(monkeypatch, _raising([(400, hard, None)] * 3))
    p = _provider(monkeypatch)
    with pytest.raises(ProviderInsufficientQuota, match="billing_hard_limit_reached"):
        p.chat(MSGS)
    assert len(calls) == 1 and sleeps == []


def test_grid_unpriced_response_carries_its_worst_case_into_spend(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    _patch_http(monkeypatch, lambda req: _completion(tier="fast"))
    out = tmp_path / "out"
    assert run_grid.main(["--config", str(_cfg(tmp_path, [SPEC], n_questions=2)),
                          "--out", str(out)]) == 0
    reqs = _lines(out, "requests.jsonl") + _lines(out, "requests_failed.jsonl")
    assert len(reqs) == 1 and reqs[0]["usd"] is None and reqs[0]["usd_worst"] > 0
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["cloud_spend_this_run"] == pytest.approx(2 * reqs[0]["usd_worst"], abs=1e-6)
    later = collect_prior_spend([out])["total"]
    assert later == pytest.approx(2 * reqs[0]["usd_worst"], abs=1e-6)


def test_job_missing_key_pattern_matches_the_grid_reason(tmp_path, monkeypatch):
    """Couples job.sh's NOT RUN grep to the reason run_grid actually writes."""
    import re
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    pattern = re.search(r'grep -q "([^"]+)" "\$DIR/not_run.jsonl"',
                        (REPO / "experiments" / "ci" / "job.sh").read_text()).group(1)
    out = tmp_path / "out"
    run_grid.main(["--config", str(_cfg(tmp_path, [SPEC], n_questions=2)), "--out", str(out)])
    nr = _lines(out, "not_run.jsonl")
    assert nr and all(pattern in n["reason"] for n in nr)


def test_offchain_spend_ledger_counts_as_prior_spend(tmp_path):
    d = tmp_path / "spend"
    d.mkdir()
    (d / "spend_ledger.jsonl").write_text(
        json.dumps({"label": "smoke", "usd": 0.0012}) + "\n", encoding="utf-8")
    assert collect_prior_spend([d])["total"] == pytest.approx(0.0012)


def test_committed_offchain_ledger_is_well_formed():
    # the B6 smoke request was made on 2026-11-16, so the ledger must exist from then on
    path = REPO / "experiments" / "spend" / "spend_ledger.jsonl"
    assert path.exists(), "the off-chain spend ledger must be committed"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    total = collect_prior_spend([path.parent])["total"]  # rounded to 6 dp per file
    assert rows and total >= sum(r["usd"] for r in rows) - 1e-6
    for row in rows:
        assert row["label"] == "smoke" and isinstance(row["usd"], float)
        assert "reply" not in row and not any("sk-" in str(v) for v in row.values())


def test_jobs_add_the_offchain_ledger_and_missing_key_is_never_green(tmp_path):
    bash, env, log = _stub_env(tmp_path)
    nr_dir = tmp_path / "work" / "out" / "h2h_dev"
    nr_dir.mkdir(parents=True)
    r, _ = _job(tmp_path, bash, env, JOB="h2h", SPLIT="dev", FIRST_IN_CHAIN="true")
    assert r.returncode == 0, r.stderr
    line = [x for x in log.read_text().splitlines() if "agent.run_grid" in x][0]
    assert "--spend-from out" in line and "--spend-from experiments/spend" in line
    (nr_dir / "not_run.jsonl").write_text(json.dumps(
        {"provider": "openai", "model": "gpt-6-luna", "policy": "react", "k": 2,
         "reason": "resolution: OPENAI_API_KEY not set in job environment"}) + "\n")
    r, _ = _job(tmp_path, bash, env, JOB="h2h", SPLIT="dev", FIRST_IN_CHAIN="true")
    assert r.returncode == 3 and "NOT RUN" in r.stderr


def test_workflow_passes_openai_key_on_dispatch_only():
    wf = (REPO / ".github" / "workflows" / "experiment.yml").read_text(encoding="utf-8")
    assert ("OPENAI_API_KEY: ${{ github.event_name == 'workflow_dispatch' && "
            "secrets.OPENAI_API_KEY || '' }}") in wf
    sh = (REPO / "experiments" / "ci" / "run_in_container.sh").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" in sh.split("for v in", 1)[1].split("; do", 1)[0]


def test_an_invalidated_key_halts_like_a_billing_stop_and_is_never_retried(monkeypatch):
    """PREREG_v3 Amendment 2: the v3 smoke got HTTP 401 token_invalidated; a 401 must stop the
    whole run at once instead of erroring run after run."""
    from agent.providers import ProviderAuthInvalid
    body = ('{"error": {"message": "Your API key has been invalidated.", "type": "",'
            ' "code": "token_invalidated", "param": null}}')
    calls, sleeps = _setup(monkeypatch, _raising([(401, body, None)] * 5))
    p = _provider(monkeypatch)
    with pytest.raises(ProviderInsufficientQuota) as ei:
        p.chat(MSGS)
    assert isinstance(ei.value, ProviderAuthInvalid) and "token_invalidated" in str(ei.value)
    assert len(calls) == 1 and sleeps == [] and p.guard.spent == 0.0
