"""Contract tests for the experimental agent (written before the implementation).

These pin the interface and the fail-first control 1 from experiments/PREREG.md section 8.
Token counts here come from WordCountProvider: input_tokens = number of whitespace-separated
words across all message contents of the request, so the prefix sums can be computed by hand.

Hand computation (S=5 words, "Question: q q q"=4, each scripted action=1 word because the JSON
has no spaces, "Observation: o1 o1 o1 o1"=5, "Observation: o2 o2 o2 o2 o2 o2"=7):
  carry_all:            x = [9, 9+1+5=15, 15+1+7=23]  -> total 47
  reset_summary:        x = [9, 9+1+5=15, 9+1+7=17]   -> total 41
  reset_summary_broken: behaves like carry_all        -> total 47
"""
import json
import os
import tempfile

import pytest

from agent.agent import run_agent
from agent.ledger import RequestLogger, to_tokenledger_rows
from agent.policies import build_messages
from agent.providers import LLMResponse, Provider, ProviderUnavailable, Usage, make_provider

SYSTEM = "s s s s s"
QUESTION = "q q q"
A1 = '{"thought":"t","notes":"n","action":"search","args":{"query":"a"}}'
A2 = '{"thought":"t","notes":"n","action":"read","args":{"doc_id":"d1"}}'
A3 = '{"thought":"t","notes":"n","action":"finish","args":{"answer":"yes"}}'
TOOLS = {
    "search": lambda query, k: "o1 o1 o1 o1",
    "read": lambda doc_id: "o2 o2 o2 o2 o2 o2",
}
HAND = {"carry_all": [9, 15, 23], "reset_summary": [9, 15, 17], "reset_summary_broken": [9, 15, 23]}


class WordCountProvider(Provider):
    name = "mock"

    def __init__(self, script):
        self.model = "wordcount"
        self.script = list(script)
        self.seen = []

    def chat(self, messages, **opts):
        self.seen.append([dict(m) for m in messages])
        n_in = sum(len(m["content"].split()) for m in messages)
        text = self.script.pop(0)
        return LLMResponse(text=text, usage=Usage(input_tokens=n_in, cached_input_tokens=None,
                                                  output_tokens=len(text.split()), latency_s=0.0, raw={}),
                           thinking=None)


def _run(policy, logger_cls=RequestLogger):
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "requests.jsonl")
    logger = logger_cls(path)
    prov = WordCountProvider([A1, A2, A3])
    res = run_agent(question=QUESTION, provider=prov, policy=policy, k=8, tools=TOOLS,
                    system_prompt=SYSTEM, max_steps=10, logger=logger,
                    run_meta={"run_id": f"t-{policy}", "question_id": "q0", "turn": 1})
    logger.close()
    with open(path, encoding="utf-8") as fh:
        recs = [json.loads(line) for line in fh if line.strip()]
    return res, recs, prov, tmp


@pytest.mark.parametrize("policy", ["carry_all", "reset_summary", "reset_summary_broken"])
def test_control1_prefix_sum_matches_hand_computation(policy):
    res, recs, _, _ = _run(policy)
    assert [r["input_tokens"] for r in recs] == HAND[policy]
    assert sum(r["input_tokens"] for r in recs) == sum(HAND[policy])
    assert res.total_input_tokens == sum(HAND[policy])
    assert res.turns == 3 and res.finished and res.answer == "yes"
    assert [r["step"] for r in recs] == [0, 1, 2]
    for r in recs:
        assert r["cached_input_tokens"] is None  # not reported -> null, never 0
        assert r["policy"] == policy and r["run_id"] == f"t-{policy}" and r["k"] == 8


def test_control1_through_tokenledger():
    import tokenledger
    _, recs, _, tmp = _run("carry_all")
    rows = to_tokenledger_rows(recs)
    p = os.path.join(tmp, "tl.jsonl")
    with open(p, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    m = tokenledger.account(tokenledger.load_requests(p))
    assert m.total_input == 47
    assert m.n_requests == 3 and m.n_turns == 1
    assert m.priming_total == 27  # priming = x_1 = 9, re-sent on each of 3 requests
    assert m.conversation_total == 20
    assert all("cache_read_tokens" not in r for r in rows)  # unknown is omitted, not 0


class DroppingLogger(RequestLogger):
    """A deliberately broken logger: loses the second request."""

    def log(self, record):
        if record.get("step") == 1:
            return
        super().log(record)


def test_control1_fails_first_when_logger_drops_a_request():
    _, recs, _, _ = _run("carry_all", logger_cls=DroppingLogger)
    assert sum(r["input_tokens"] for r in recs) != sum(HAND["carry_all"])


def test_policies_send_exactly_the_specified_messages():
    hist = [("A1", "O1"), ("A2", "O2")]
    carry = build_messages("carry_all", SYSTEM, QUESTION, hist)
    reset = build_messages("reset_summary", SYSTEM, QUESTION, hist)
    broken = build_messages("reset_summary_broken", SYSTEM, QUESTION, hist)
    window = build_messages("sliding_window:1", SYSTEM, QUESTION, hist)
    contents = lambda ms: [m["content"] for m in ms]
    assert contents(carry) == [SYSTEM, "Question: q q q", "A1", "Observation: O1", "A2", "Observation: O2"]
    assert contents(reset) == [SYSTEM, "Question: q q q", "A2", "Observation: O2"]
    assert contents(broken) == contents(carry)
    assert contents(window) == contents(reset)
    assert [m["role"] for m in carry] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert contents(build_messages("reset_summary", SYSTEM, QUESTION, [])) == [SYSTEM, "Question: q q q"]


def test_turn_cap_and_parse_errors_count_as_steps():
    tmp = tempfile.mkdtemp()
    logger = RequestLogger(os.path.join(tmp, "r.jsonl"))
    prov = WordCountProvider(["not json"] * 12)
    res = run_agent(question=QUESTION, provider=prov, policy="carry_all", k=2, tools=TOOLS,
                    system_prompt=SYSTEM, max_steps=10, logger=logger,
                    run_meta={"run_id": "cap", "question_id": "q0", "turn": 1})
    logger.close()
    assert res.turns == 10 and not res.finished and res.answer == ""
    assert res.parse_errors == 10


def test_cloud_provider_without_key_is_unavailable(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    for spec in ({"provider": "gemini", "model": "gemini-2.5-flash"},
                 {"provider": "dashscope", "model": "qwen-plus"}):
        with pytest.raises(ProviderUnavailable):
            make_provider(spec)


def test_budget_guard_refuses_over_limit():
    from agent.budget import BudgetExceeded, check_budget, estimate_usd
    est = estimate_usd("gemini", "gemini-2.5-flash", input_tokens=50_000_000, output_tokens=1_000_000)
    assert est > 5
    with pytest.raises(BudgetExceeded):
        check_budget(est, max_usd=5.0)
    check_budget(0.01, max_usd=5.0)  # does not raise


def test_hotpot_metrics():
    from agent.metrics import exact_match, f1_score
    assert exact_match("The Eiffel Tower!", "eiffel tower") == 1
    assert exact_match("yes", "no") == 0
    assert abs(f1_score("Barack Obama", "Obama") - 2 / 3) < 1e-9
    assert f1_score("yes", "no") == 0.0
