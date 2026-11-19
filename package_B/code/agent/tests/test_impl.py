"""Implementation tests: retrieval determinism, data sampling/KB dedup, budget, providers."""
import hashlib
import json
import os
import urllib.error

import pytest

from agent.budget import (BudgetExceeded, check_budget, ensure_price_verified,
                          estimate_usd, request_usd)
from agent.data import build_kb, load_dataset
from agent.providers import (LLMResponse, Provider, ProviderUnavailable, Usage,
                             make_provider)
from agent.retrieval import Retriever

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "hotpot_tiny.jsonl")


def _dataset(n_q=3, n_sm=1):
    return load_dataset({"fixture": FIXTURE, "parquet_url": None, "sha256": None,
                         "cache_dir": None, "sample_seed": 20261115,
                         "n_questions": n_q, "n_smoke": n_sm})


def test_sampling_deterministic_positions_and_split():
    d1, d2 = _dataset(), _dataset()
    assert [q.id for q in d1.sample] == [q.id for q in d2.sample]
    assert [q.id for q in d1.smoke] == [q.id for q in d2.smoke]
    assert len(d1.sample) == 3 and len(d1.smoke) == 1
    assert [q.position for q in d1.sample] == [0, 1, 2]
    assert d1.sample[2].split == "test"
    assert d1.sample[0].split == "train" and d1.sample[1].split == "train"
    assert {q.id for q in d1.sample}.isdisjoint({q.id for q in d1.smoke})
    assert len(d1.dataset_sha256) == 64


def test_kb_dedup_first_wins_doc_ids_and_chunks():
    d = _dataset()
    titles = [doc["title"] for doc in d.kb.docs]
    assert len(titles) == len(set(titles)), "titles must be deduplicated"
    for doc in d.kb.docs:
        assert doc["doc_id"] == hashlib.sha1(doc["title"].encode()).hexdigest()[:10]
        assert "\n" not in doc["paragraph"] and "  " not in doc["paragraph"]
    for ch in d.kb.chunks:
        assert ch["text"].startswith(ch["title"] + ": ")
    # Gamma City has 3 sentences -> exactly 1 chunk; Delta Lake has 2 -> 1 chunk
    by_title = {doc["title"]: doc for doc in d.kb.docs}
    assert len([c for c in d.kb.chunks if c["doc_id"] == by_title["Gamma City"]["doc_id"]]) == 1
    assert len([c for c in d.kb.chunks if c["doc_id"] == by_title["Delta Lake"]["doc_id"]]) == 1
    # union of the sampled questions' contexts
    assert "Gamma City" in by_title and "Delta Lake" in by_title


def test_build_kb_first_occurrence_wins():
    rows = [
        {"id": "a", "context": {"title": ["T"], "sentences": [["first version."]]}},
        {"id": "b", "context": {"title": ["T"], "sentences": [["second version."]]}},
    ]
    kb = build_kb(rows)
    assert len(kb.docs) == 1
    assert kb.docs[0]["paragraph"] == "first version."


def test_retrieval_deterministic_and_tie_order():
    d = _dataset()
    r1, r2 = Retriever(d.kb), Retriever(d.kb)
    q = d.sample[0].question
    ids1 = [c["doc_id"] for c in r1.search(q, 3)]
    ids2 = [c["doc_id"] for c in r2.search(q, 3)]
    assert ids1 == ids2 and len(ids1) == 3
    # a chunk sharing more query tokens ranks first
    hits = r1.search("Gamma City capital Testland", 8)
    assert hits[0]["title"] == "Gamma City"
    obs = r1.search_observation(q, 2)
    lines = obs.split("\n")
    assert len(lines) == 2 and lines[0].startswith("[") and "] " in lines[0]
    robs = r1.read_observation(hits[0]["doc_id"])
    assert robs.startswith("[") and ": " in robs
    assert r1.read_observation("does-not-exist").startswith("Unknown doc_id")


def test_budget_math_and_refusals():
    est = estimate_usd("gemini", "gemini-2.5-flash", 1_000_000, 1_000_000)
    assert est == pytest.approx(0.30 + 2.50)
    with pytest.raises(BudgetExceeded):
        check_budget(6.0, 5.0)
    with pytest.raises(BudgetExceeded):
        check_budget(2.1, 5.0, spent=3.0)
    check_budget(2.0, 5.0, spent=3.0)
    u = Usage(input_tokens=1000, cached_input_tokens=400, output_tokens=100,
              latency_s=0.0, raw={})
    assert request_usd("gemini", "gemini-2.5-flash", u) == pytest.approx(
        (600 * 0.30 + 400 * 0.03 + 100 * 2.50) / 1_000_000)
    assert request_usd("gemini", "gemini-unknown-model", u) is None
    ensure_price_verified("gemini", "gemini-2.5-flash")  # verified: true
    with pytest.raises(BudgetExceeded):
        ensure_price_verified("dashscope", "qwen-max")  # verified: false
    with pytest.raises(BudgetExceeded):
        ensure_price_verified("gemini", "gemini-unknown-model")  # no price entry


def test_mock_provider_scripted_steps():
    prov = make_provider({"provider": "mock", "model": "mock-scripted"})
    s0 = prov.chat([{"role": "system", "content": "s s"},
                    {"role": "user", "content": "Question: Which mountain?"}])
    a0 = json.loads(s0.text)
    assert a0["action"] == "search" and a0["args"]["query"] == "Which mountain?"
    assert s0.usage.input_tokens == 5  # 2 system words + 3 question words
    obs = "[abc123] Mount Beta: Mount Beta is a peak. It rises 2,100 m."
    s1 = prov.chat([{"role": "user", "content": "Observation: " + obs}])
    assert json.loads(s1.text)["args"]["doc_id"] == "abc123"
    s2 = prov.chat([{"role": "user", "content": "Observation: " + obs}])
    a2 = json.loads(s2.text)
    assert a2["action"] == "finish" and a2["args"]["answer"] == "Mount Beta"


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _patch_http(monkeypatch, handler):
    """handler(request) -> response dict (or raises). Returns the call list."""
    from agent import providers
    calls = []

    def fake(req, timeout=None):
        calls.append(req)
        return _Resp(handler(req))

    monkeypatch.setattr(providers.urllib.request, "urlopen", fake)
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    return calls


def _payload_of(req):
    return json.loads(req.data.decode())


def test_gemini_mapping_usage_and_retry(monkeypatch):
    from agent import providers
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    canned = {
        "candidates": [{"content": {"parts": [{"text": "{}"}, {"text": "thoughts", "thought": True}]}}],
        "usageMetadata": {"promptTokenCount": 20, "cachedContentTokenCount": 5,
                          "candidatesTokenCount": 3, "thoughtsTokenCount": 2},
    }
    state = {"errors": 1}

    def handler(req):
        if state["errors"]:
            state["errors"] -= 1
            raise urllib.error.HTTPError(req.full_url, 503, "try again", {}, None)
        return canned

    calls = _patch_http(monkeypatch, handler)
    p = make_provider({"provider": "gemini", "model": "gemini-2.5-flash",
                       "thinking_budget": 0, "max_output_tokens": 128})
    r = p.chat([{"role": "system", "content": "sys"},
                {"role": "assistant", "content": "prev"},
                {"role": "user", "content": "hi"}])
    assert len(calls) == 2  # one retry after the 503
    body = _payload_of(calls[-1])
    assert body["systemInstruction"]["parts"][0]["text"] == "sys"
    assert body["contents"][0] == {"role": "model", "parts": [{"text": "prev"}]}
    assert body["generationConfig"]["temperature"] == 0
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}
    assert r.usage.input_tokens == 20 and r.usage.cached_input_tokens == 5
    assert r.usage.output_tokens == 5
    assert r.text == "{}" and r.thinking == "thoughts"
    headers = {k.lower(): v for k, v in calls[-1].headers.items()}
    assert headers.get("x-goog-api-key") == "test-key"


def test_gemini_auto_flash_resolution(monkeypatch):
    from agent import providers
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    listed = {"models": [
        {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-2.0-flash-001", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-2.5-flash-preview", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-2.5-flash-lite", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-3-flash", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-3-flash-image", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-2.5-pro", "supportedGenerationMethods": ["generateContent"]},
    ]}

    def handler(req):
        if req.get_method() == "GET":
            return listed
        return {"candidates": [{"content": {"parts": [{"text": "{}"}]}}],
                "usageMetadata": {"promptTokenCount": 1}}

    _patch_http(monkeypatch, handler)
    p = make_provider({"provider": "gemini", "model": "auto-flash"})
    assert p.model == "gemini-3-flash"  # highest exact gemini-<version>-flash


def test_gemini_auto_flash_none_available(monkeypatch):
    from agent import providers
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    _patch_http(monkeypatch, lambda req: {"models": [
        {"name": "models/gemini-2.5-flash-preview", "supportedGenerationMethods": ["generateContent"]}]})
    with pytest.raises(ProviderUnavailable):
        make_provider({"provider": "gemini", "model": "auto-flash"})


def test_dashscope_usage_and_payload(monkeypatch):
    from agent import providers
    monkeypatch.setenv("DASHSCOPE_API_KEY", "ds-key")
    canned = {"choices": [{"message": {"content": "{}"}}],
              "usage": {"prompt_tokens": 30, "completion_tokens": 4,
                        "prompt_tokens_details": {"cached_tokens": 12}}}
    calls = _patch_http(monkeypatch, lambda req: canned)
    p = make_provider({"provider": "dashscope", "model": "qwen-flash",
                       "enable_thinking": False, "max_tokens": 64})
    r = p.chat([{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}])
    body = _payload_of(calls[0])
    assert body["response_format"] == {"type": "json_object"}
    assert body["enable_thinking"] is False and body["max_tokens"] == 64
    assert body["messages"][0]["role"] == "system"
    headers = {k.lower(): v for k, v in calls[0].headers.items()}
    assert headers.get("authorization") == "Bearer ds-key"
    assert r.usage.input_tokens == 30 and r.usage.cached_input_tokens == 12
    assert r.usage.output_tokens == 4


def test_dashscope_absent_cached_defaults_to_zero(monkeypatch):
    from agent import providers
    monkeypatch.setenv("DASHSCOPE_API_KEY", "ds-key")
    canned = {"choices": [{"message": {"content": "{}"}}],
              "usage": {"prompt_tokens": 30, "completion_tokens": 4}}
    _patch_http(monkeypatch, lambda req: canned)
    p = make_provider({"provider": "dashscope", "model": "qwen-flash"})
    r = p.chat([{"role": "user", "content": "hi"}])
    assert r.usage.cached_input_tokens == 0


class _ScriptedProvider(Provider):
    """Replies with fixed texts; input tokens = word count over message contents."""

    name = "mock"

    def __init__(self, replies):
        self.model = "scripted"
        self.replies = list(replies)

    def chat(self, messages, **opts):
        text = self.replies.pop(0)
        n_in = sum(len(str(m.get("content") or "").split()) for m in messages)
        return LLMResponse(text=text, usage=Usage(
            input_tokens=n_in, cached_input_tokens=None,
            output_tokens=len(text.split()), latency_s=0.0, raw={}))


def test_agent_non_dict_args_counts_as_parse_error():
    from agent.agent import run_agent
    bad = '{"thought":"t","notes":"n","action":"search","args":"oops"}'
    good = '{"thought":"t","notes":"n","action":"finish","args":{"answer":"yes"}}'
    prov = _ScriptedProvider([bad, good])
    res = run_agent(question="q q q", provider=prov, policy="carry_all", k=2,
                    tools={"search": lambda query, k: "obs", "read": lambda doc_id: "obs"},
                    system_prompt="s s", max_steps=10, logger=None,
                    run_meta={"run_id": "t", "question_id": "q0", "turn": 1})
    assert res.parse_errors == 1 and res.turns == 2 and res.finished
    assert res.answer == "yes"
    assert res.steps[0]["action"] is None and res.steps[0]["parse_error"] is True


def test_gemini_3_sends_thinking_level_minimal(monkeypatch):
    from agent import providers
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    canned = {"candidates": [{"content": {"parts": [{"text": "{}"}]}}],
              "usageMetadata": {"promptTokenCount": 10}}

    def handler(req):
        if req.get_method() == "GET":  # auto-flash resolution
            return {"models": [{"name": "models/gemini-3.6-flash",
                                "supportedGenerationMethods": ["generateContent"]}]}
        return canned

    calls = _patch_http(monkeypatch, handler)
    p = make_provider({"provider": "gemini", "model": "gemini-3.8-flash",
                       "max_output_tokens": 128})
    r = p.chat([{"role": "user", "content": "hi"}])
    assert _payload_of(calls[0])["generationConfig"]["thinkingConfig"] == {
        "thinkingLevel": "minimal"}
    assert r.usage.raw["thinking_config"] == {"thinkingLevel": "minimal"}
    # resolved auto-flash ids starting with gemini-3 get thinkingLevel too
    p2 = make_provider({"provider": "gemini", "model": "auto-flash"})
    r2 = p2.chat([{"role": "user", "content": "hi"}])
    posts = [c for c in calls if c.get_method() == "POST"]
    assert _payload_of(posts[-1])["generationConfig"]["thinkingConfig"] == {
        "thinkingLevel": "minimal"}
    assert r2.usage.raw["thinking_config"] == {"thinkingLevel": "minimal"}


def test_gemini_3_thinking_400_retries_once_with_low(monkeypatch):
    import io
    from agent import providers
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    canned = {"candidates": [{"content": {"parts": [{"text": "{}"}]}}],
              "usageMetadata": {"promptTokenCount": 10}}
    state = {"fail": True}

    def handler(req):
        if state["fail"]:
            state["fail"] = False
            raise urllib.error.HTTPError(
                req.full_url, 400, "bad request", {},
                io.BytesIO(b'{"error": {"message": "thinkingLevel minimal not supported"}}'))
        return canned

    calls = _patch_http(monkeypatch, handler)
    p = make_provider({"provider": "gemini", "model": "gemini-3.8-flash"})
    r = p.chat([{"role": "user", "content": "hi"}])
    assert len(calls) == 2
    assert _payload_of(calls[0])["generationConfig"]["thinkingConfig"] == {
        "thinkingLevel": "minimal"}
    assert _payload_of(calls[1])["generationConfig"]["thinkingConfig"] == {
        "thinkingLevel": "low"}
    assert r.usage.raw["thinking_config"] == {"thinkingLevel": "low"}


def test_gemini_non_3_keeps_thinking_budget_and_unrelated_400_raises(monkeypatch):
    import io
    from agent import providers
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    canned = {"candidates": [{"content": {"parts": [{"text": "{}"}]}}],
              "usageMetadata": {"promptTokenCount": 10}}
    calls = _patch_http(monkeypatch, lambda req: canned)
    p = make_provider({"provider": "gemini", "model": "gemini-2.5-flash",
                       "thinking_budget": 0})
    r = p.chat([{"role": "user", "content": "hi"}])
    assert _payload_of(calls[0])["generationConfig"]["thinkingConfig"] == {
        "thinkingBudget": 0}
    assert r.usage.raw["thinking_config"] == {"thinkingBudget": 0}

    def bad_handler(req):
        raise urllib.error.HTTPError(req.full_url, 400, "bad request", {},
                                     io.BytesIO(b'{"error": "invalid argument"}'))

    calls2 = _patch_http(monkeypatch, bad_handler)
    with pytest.raises(ProviderUnavailable):
        p.chat([{"role": "user", "content": "hi"}])
    assert len(calls2) == 1  # no retry: the 400 did not mention thinking


def test_dashscope_enable_thinking_sent_only_when_specified(monkeypatch):
    from agent import providers
    monkeypatch.setenv("DASHSCOPE_API_KEY", "ds-key")
    canned = {"choices": [{"message": {"content": "{}"}}],
              "usage": {"prompt_tokens": 30, "completion_tokens": 4}}
    calls = _patch_http(monkeypatch, lambda req: canned)
    p = make_provider({"provider": "dashscope", "model": "qwen-max", "max_tokens": 64})
    p.chat([{"role": "user", "content": "hi"}])
    assert "enable_thinking" not in _payload_of(calls[0])  # qwen-max: no thinking mode
    p2 = make_provider({"provider": "dashscope", "model": "qwen-flash",
                        "enable_thinking": False})
    p2.chat([{"role": "user", "content": "hi"}])
    assert _payload_of(calls[1])["enable_thinking"] is False


def _run_grid(tmp_path, models, max_usd, monkeypatch=None, http_handler=None,
              policies=None, ks=None, smoke=False):
    """Write a temp grid config over the tiny fixture and run agent.run_grid in-process.

    Returns (runs, not_run, requests, failed_requests)."""
    import yaml
    from agent import run_grid
    cfg = {
        "experiment": "test-grid",
        "dataset": {"parquet_url": None, "sha256": None, "cache_dir": None,
                    "fixture": FIXTURE, "sample_seed": 20261115,
                    "n_questions": 3, "n_smoke": 1},
        "agent": {"max_steps": 10, "temperature": 0, "seed": 42,
                  "cell_order_seed": 20261116},
        "policies": policies or ["carry_all", "reset_summary"],
        "ks": ks or [2, 8],
        "models": models,
        "budget": {"max_usd": max_usd, "est_input_tokens_per_run": 20000,
                   "est_output_tokens_per_run": 1500},
    }
    cfg_path = tmp_path / "grid.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = tmp_path / "out"
    calls = _patch_http(monkeypatch, http_handler) if http_handler else None
    argv = ["--config", str(cfg_path), "--out", str(out)] + (["--smoke"] if smoke else [])
    assert run_grid.main(argv) == 0

    def load(name):
        p = out / name
        if not p.exists():
            return []
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

    return load("runs.jsonl"), load("not_run.jsonl"), load("requests.jsonl"), load(
        "requests_failed.jsonl")


def test_run_grid_refuses_block_with_unverified_price(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    models = [{"provider": "mock", "model": "mock-scripted"},
              {"provider": "dashscope", "model": "qwen-max", "max_tokens": 64}]
    runs, not_run, _, _ = _run_grid(tmp_path, models, max_usd=5.0)
    ok = [r for r in runs if r["status"] == "ok"]
    assert len(ok) == 12  # mock block only
    qmax = [n for n in not_run if n["model"] == "qwen-max"]
    assert len(qmax) == 4  # one record per cell of the refused block
    assert all(n["reason"].startswith("price: ") for n in qmax), qmax
    assert not any(r["model"] == "qwen-max" for r in runs)


def test_run_grid_refuses_block_over_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    # qwen-flash block: 12 runs x 0.0016 USD est = 0.0192 > 0.01
    models = [{"provider": "mock", "model": "mock-scripted"},
              {"provider": "dashscope", "model": "qwen-flash", "enable_thinking": False}]
    runs, not_run, _, _ = _run_grid(tmp_path, models, max_usd=0.01)
    ok = [r for r in runs if r["status"] == "ok"]
    assert len(ok) == 12
    qf = [n for n in not_run if n["model"] == "qwen-flash"]
    assert len(qf) == 4 and all(n["reason"].startswith("budget: ") for n in qf), qf
    assert not any(r["model"] == "qwen-flash" for r in runs)


def test_run_grid_stops_block_when_budget_exhausted(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    finish = '{"thought":"t","notes":"n","action":"finish","args":{"answer":"x"}}'
    canned = {"choices": [{"message": {"content": finish}}],
              "usage": {"prompt_tokens": 40000, "completion_tokens": 10}}  # 0.002 USD/run
    calls = _patch_http(monkeypatch, lambda req: canned)
    models = [{"provider": "mock", "model": "mock-scripted"},
              {"provider": "dashscope", "model": "qwen-flash", "enable_thinking": False}]
    runs, not_run, _, _ = _run_grid(tmp_path, models, max_usd=0.02)  # est 0.0192 fits
    ok = [r for r in runs if r["status"] == "ok"]
    qf_ok = [r for r in ok if r["model"] == "qwen-flash"]
    assert len(ok) == 22 and len(qf_ok) == 10  # 12 local + 10 cloud before the cutoff
    # models processed in config order: the whole local block precedes the cloud block
    assert [r["provider"] for r in runs] == ["mock"] * 12 + ["dashscope"] * 10
    qf_nr = [n for n in not_run if n["model"] == "qwen-flash"]
    assert qf_nr and all(n["reason"] == "budget exhausted after 10 runs" for n in qf_nr), qf_nr
    qids = {r["question_id"] for r in ok if r["provider"] == "mock"}
    done = {(r["policy"], r["k"], r["question_id"]) for r in qf_ok}
    unfinished = [(p, k) for p in ("carry_all", "reset_summary") for k in (2, 8)
                  if sum(1 for qid in qids if (p, k, qid) in done) < len(qids)]
    assert len(qf_nr) == len(unfinished)
    assert len(calls) == 10  # one request per finished cloud run, then stopped


def test_dashscope_refuses_coding_plan(monkeypatch):
    import pytest
    from agent.providers import ProviderUnavailable, make_provider
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-sp-notarealkey")
    with pytest.raises(ProviderUnavailable, match="Coding Plan"):
        make_provider({"provider": "dashscope", "model": "qwen-plus"})
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-notarealkey")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://coding-intl.dashscope.aliyuncs.com/v1")
    with pytest.raises(ProviderUnavailable, match="Coding Plan"):
        make_provider({"provider": "dashscope", "model": "qwen-plus"})
    monkeypatch.delenv("DASHSCOPE_BASE_URL")
    assert make_provider({"provider": "dashscope", "model": "qwen-plus"}).model == "qwen-plus"


FINISH_JSON = '{"thought":"t","notes":"n","action":"finish","args":{"answer":"x"}}'
SEARCH_JSON = '{"thought":"t","notes":"n","action":"search","args":{"query":"x"}}'


def _flaky_then_steady_handler(state):
    """First run: step 0 answers search, step 1 dies on HTTP 500 (4 tries); after
    that every request is answered with finish."""
    def handler(req):
        body = json.loads(req.data.decode())
        first_step = not any(m["role"] == "assistant" for m in body["messages"])
        if not state["steady"]:
            if first_step:
                return {"choices": [{"message": {"content": SEARCH_JSON}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
            if state["booms"]:  # burn the 4 attempts of the failing request
                state["booms"] -= 1
                raise urllib.error.HTTPError(req.full_url, 500, "boom", {}, None)
            state["steady"] = True
        return {"choices": [{"message": {"content": FINISH_JSON}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
    return handler


def test_run_grid_failed_attempt_records_go_to_requests_failed(tmp_path, monkeypatch):
    """Defect 1: a failed attempt's buffered records go to requests_failed.jsonl
    (attempt=1); only the successful retry's records (attempt=2) reach
    requests.jsonl under the same run_id."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    handler = _flaky_then_steady_handler({"booms": 4, "steady": False})
    _patch_http(monkeypatch, handler)
    models = [{"provider": "dashscope", "model": "qwen-flash", "enable_thinking": False}]
    runs, not_run, reqs, failed_reqs = _run_grid(tmp_path, models, max_usd=5.0,
                                                 policies=["carry_all", "reset_summary"],
                                                 ks=[2])
    # A9: an errored run does not poison its cell: all 3 questions x 2 cells run, the
    # errored one is retried at the end and is finally ok
    assert len(runs) == 6 and all(r["status"] == "ok" for r in runs)
    assert len(failed_reqs) == 1 and failed_reqs[0]["attempt"] == 1
    rid = failed_reqs[0]["run_id"]
    mine = [q for q in reqs if q["run_id"] == rid]
    assert len(mine) == 1 and mine[0]["attempt"] == 2  # no orphan attempt-1 records
    assert all("attempt" in q for q in reqs)
    assert all(q["attempt"] == 1 for q in reqs if q["run_id"] != rid)


def test_run_grid_retries_every_errored_run_once(tmp_path, monkeypatch):
    """Defect 2: every errored (run, question) is retried once (the old loop
    crashed unpacking the errored set and retried only one question per cell)."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

    def handler(req):
        body = json.loads(req.data.decode())
        first_step = not any(m["role"] == "assistant" for m in body["messages"])
        if first_step:
            return {"choices": [{"message": {"content": SEARCH_JSON}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        raise urllib.error.HTTPError(req.full_url, 500, "boom", {}, None)

    _patch_http(monkeypatch, handler)
    models = [{"provider": "dashscope", "model": "qwen-flash", "enable_thinking": False}]
    runs, not_run, reqs, failed_reqs = _run_grid(tmp_path, models, max_usd=5.0,
                                                 policies=["carry_all", "reset_summary"],
                                                 ks=[2])
    # A9: every question of every cell is recorded as an error (3 per cell, then retried)
    assert len(runs) == 6 and all(r["status"] == "error" for r in runs)
    assert all(r["reason"].endswith("(retried once)") for r in runs)
    by_run = {}
    for rec in failed_reqs:
        by_run.setdefault(rec["run_id"], []).append(rec["attempt"])
    assert len(by_run) == 6, by_run  # every errored run was retried
    assert all(sorted(a) == [1, 2] for a in by_run.values()), by_run
    assert reqs == []  # no run ever finished ok


def test_run_grid_smoke_writes_control_smoke(tmp_path):
    """Defect 3: --smoke runs are written with control 'smoke' so the analysis
    excludes them together with the broken_reset control."""
    models = [{"provider": "mock", "model": "mock-scripted"}]
    runs, not_run, reqs, failed_reqs = _run_grid(tmp_path, models, max_usd=5.0, smoke=True)
    assert len(runs) == 4  # 1 smoke question x 2 policies x 2 ks
    assert all(r["control"] == "smoke" for r in runs), runs
    assert all(r["run_id"].startswith("smoke|") for r in runs)


# ---- Amendment 1: Gemini tiers, quota handling, --est-from, privacy -------------------

def _http_error(url, code, body):
    import io
    return urllib.error.HTTPError(url, code, "err", {}, io.BytesIO(json.dumps(body).encode()))


def _gm(name, methods=("generateContent",)):
    return {"name": "models/" + name, "supportedGenerationMethods": list(methods)}


def test_gemini_auto_tiers_numeric_versions_and_exclusions(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    names = [
        "gemini-3.8-flash", "gemini-3.10-flash", "gemini-3.9-flash-001", "gemini-4-flash-preview",
        "gemini-4-flash-exp", "gemini-4-flash-image", "gemini-4-flash-tts", "gemini-4-flash-live",
        "gemini-4-flash-customtools", "gemini-3.1-flash-lite", "gemini-3.10-flash-lite",
        "gemini-3.12-flash-lite-preview", "gemini-5-flash-lite-001"]
    listed = {"models": [_gm(n) for n in names]
              + [_gm("gemini-9-flash", ["embedContent"]), _gm("gemini-9-flash-lite", ["countTokens"])]}
    _patch_http(monkeypatch, lambda req: listed)
    assert make_provider({"provider": "gemini", "model": "auto-flash"}).model == "gemini-3.10-flash"
    assert (make_provider({"provider": "gemini", "model": "auto-flash-lite"}).model
            == "gemini-3.10-flash-lite")


def test_gemini_model_list_pagination_and_key_header(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    pages = {None: {"models": [_gm("gemini-3.8-flash")], "nextPageToken": "T2"},
             "T2": {"models": [_gm("gemini-3.9-flash")]}}

    def handler(req):
        token = req.full_url.split("pageToken=")[1] if "pageToken=" in req.full_url else None
        return pages[token]

    calls = _patch_http(monkeypatch, handler)
    assert make_provider({"provider": "gemini", "model": "auto-flash"}).model == "gemini-3.9-flash"
    assert len(calls) == 2
    assert {k.lower(): v for k, v in calls[0].headers.items()}["x-goog-api-key"] == "test-key"


def test_gemini_usage_metadata_stored_raw(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    um = {"promptTokenCount": 9, "candidatesTokenCount": 4, "thoughtsTokenCount": 6,
          "totalTokenCount": 19}
    _patch_http(monkeypatch, lambda req: {
        "candidates": [{"content": {"parts": [{"text": "{}"}]}}], "usageMetadata": um})
    r = make_provider({"provider": "gemini", "model": "gemini-2.5-flash"}).chat(
        [{"role": "user", "content": "hi"}])
    assert r.usage.raw["usage_metadata"] == um
    assert r.usage.output_tokens == 10 and r.usage.cached_input_tokens == 0


def _quota_setup(monkeypatch, errors):
    """errors: list of (code, body) raised in order, then success."""
    from agent import providers
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    sleeps = []
    state = {"i": 0}

    def handler(req):
        if state["i"] < len(errors):
            code, body = errors[state["i"]]
            state["i"] += 1
            raise _http_error(req.full_url, code, body)
        return {"candidates": [{"content": {"parts": [{"text": "{}"}]}}],
                "usageMetadata": {"promptTokenCount": 1}}

    calls = _patch_http(monkeypatch, handler)
    monkeypatch.setattr(providers, "_sleep", sleeps.append)
    return calls, sleeps


def _retry_body(delay):
    return {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}]}}


def test_gemini_429_honours_retry_delay_then_succeeds(monkeypatch):
    calls, sleeps = _quota_setup(monkeypatch,
                                 [(429, _retry_body("17s")), (429, _retry_body("2.5s"))])
    p = make_provider({"provider": "gemini", "model": "gemini-2.5-flash"})
    p.chat([{"role": "user", "content": "hi"}])
    assert sleeps == [17.0, 2.5] and len(calls) == 3


def test_gemini_429_default_backoff_then_quota_exhausted(monkeypatch):
    from agent.providers import ProviderQuotaExhausted
    body = {"error": {"code": 429, "message": "slow down"}}
    calls, sleeps = _quota_setup(monkeypatch, [(429, body)] * 10)
    p = make_provider({"provider": "gemini", "model": "gemini-2.5-flash"})
    with pytest.raises(ProviderQuotaExhausted):
        p.chat([{"role": "user", "content": "hi"}])
    assert sleeps == [10, 30, 90, 270] and len(calls) == 5
    assert not issubclass(ProviderQuotaExhausted, ProviderUnavailable)


def test_gemini_429_per_day_quota_raises_immediately(monkeypatch):
    from agent.providers import ProviderQuotaExhausted
    body = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                      "message": "Quota exceeded for GenerateRequestsPerDayPerProjectPerModel"}}
    calls, sleeps = _quota_setup(monkeypatch, [(429, body)] * 3)
    p = make_provider({"provider": "gemini", "model": "gemini-2.5-flash"})
    with pytest.raises(ProviderQuotaExhausted):
        p.chat([{"role": "user", "content": "hi"}])
    assert sleeps == [] and len(calls) == 1


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="unknown provider"):
        make_provider({"provider": "local-x", "model": "m"})


def test_no_private_ips_or_home_paths_in_agent_sources():
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parent.parent
    pat = re.compile(r"\b(10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+)\b|/home/")
    for f in root.rglob("*.py"):
        if "tests" in f.parts:
            continue
        assert not pat.search(f.read_text(encoding="utf-8")), f


def _gemini_grid_handler(models, finish_in=1000, quota_after=None):
    finish = '{"thought":"t","notes":"n","action":"finish","args":{"answer":"x"}}'
    state = {"n": 0}

    def handler(req):
        if req.get_method() == "GET":
            return {"models": [_gm(m) for m in models]}
        state["n"] += 1
        if quota_after is not None and state["n"] > quota_after:
            raise _http_error(req.full_url, 429, {"error": {
                "message": "Quota exceeded ... PerDay ...", "status": "RESOURCE_EXHAUSTED"}})
        return {"candidates": [{"content": {"parts": [{"text": finish}]}}],
                "usageMetadata": {"promptTokenCount": finish_in, "candidatesTokenCount": 10}}

    return handler


def test_run_grid_quota_stops_block_and_continues_with_next_model(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    models = [{"provider": "gemini", "model": "auto-flash", "thinking_budget": 0},
              {"provider": "mock", "model": "mock-scripted"}]
    runs, not_run, reqs, failed = _run_grid(
        tmp_path, models, max_usd=5.0, monkeypatch=monkeypatch,
        http_handler=_gemini_grid_handler(["gemini-3.8-flash"], quota_after=2))
    g_ok = [r for r in runs if r["provider"] == "gemini" and r["status"] == "ok"]
    assert len(g_ok) == 2 and all(r["model"] == "gemini-3.8-flash" for r in g_ok)
    assert not any(r["status"] == "error" for r in runs)
    assert len([r for r in runs if r["provider"] == "mock" and r["status"] == "ok"]) == 12
    nr = [n for n in not_run if n["provider"] == "gemini"]
    assert len(nr) == 4  # every cell has unfinished questions
    assert all(n["reason"].startswith("quota: ") and n["n_done"] == 2
               and n["model"] == "gemini-3.8-flash" for n in nr)
    assert len([r for r in reqs if r["run_id"].startswith("gemini:")]) == 2  # finished runs only


def test_run_grid_est_from_uses_smoke_means_and_records_provenance(tmp_path, monkeypatch):
    import yaml
    from agent import run_grid
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    est_dir = tmp_path / "smoke"
    est_dir.mkdir()
    rows = [{"model": "gemini-3.8-flash", "status": "ok", "total_input": 1000, "total_output": 100},
            {"model": "gemini-3.8-flash", "status": "ok", "total_input": 3000, "total_output": 300},
            {"model": "gemini-3.8-flash", "status": "error", "total_input": 99999, "total_output": 9},
            {"model": "other", "status": "ok", "total_input": 5, "total_output": 5}]
    (est_dir / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    cfg = {"experiment": "t", "dataset": {"parquet_url": None, "sha256": None, "cache_dir": None,
                                          "fixture": FIXTURE, "sample_seed": 20261115,
                                          "n_questions": 3, "n_smoke": 1},
           "agent": {"max_steps": 10, "cell_order_seed": 20261116},
           "policies": ["carry_all"], "ks": [2],
           "models": [{"provider": "gemini", "model": "auto-flash", "thinking_budget": 0}],
           "budget": {"max_usd": 5, "est_input_tokens_per_run": 20000,
                      "est_output_tokens_per_run": 1500}}
    cfg_path = tmp_path / "g.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    _patch_http(monkeypatch, _gemini_grid_handler(["gemini-3.8-flash"]))
    out = tmp_path / "out"
    # est = 3 q x (2000 in, 200 out) = 0.00675 USD < 0.007
    assert run_grid.main(["--config", str(cfg_path), "--out", str(out),
                          "--est-from", str(est_dir), "--max-usd", "0.007"]) == 0
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["budget_estimates"] == {"gemini-3.8-flash": {
        "source": str(est_dir), "est_input": 2000.0, "est_output": 200.0, "n_runs": 2}}
    runs = [json.loads(x) for x in (out / "runs.jsonl").read_text().splitlines()]
    assert len(runs) == 3 and all(r["status"] == "ok" for r in runs)
    # Without --est-from the config estimate (20000 in) exceeds 0.007: block refused.
    out2 = tmp_path / "out2"
    assert run_grid.main(["--config", str(cfg_path), "--out", str(out2), "--max-usd", "0.007"]) == 0
    nr = [json.loads(x) for x in (out2 / "not_run.jsonl").read_text().splitlines()]
    assert nr and nr[0]["reason"].startswith("budget:")
