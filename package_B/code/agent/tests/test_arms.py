"""PREREG_v2 arms on a scripted provider: exact message lists, no network."""
import json

import pytest
import yaml

from agent.agent import run_agent
from agent.arms import (ARM_POLICY, cells_from_config, evict, run_agentless, run_arm,
                        run_memgpt, run_pbc)
from agent.prompts import (AGENTLESS_ANSWER_INSTRUCTION, AGENTLESS_QUERY_INSTRUCTION,
                           AGENTLESS_SYSTEM_PROMPT, PBC_REJECT_MESSAGE, PBC_SYSTEM_PROMPT,
                           SUMMARIZER_PROMPT, SYSTEM_PROMPT)
from agent.providers import LLMResponse, Provider, Usage


class Scripted(Provider):
    """Replies in order (a str, or a callable(messages) -> str); keeps every message list."""
    name, model = "mock", "scripted"

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, messages, **opts):
        self.calls.append([dict(m) for m in messages])
        r = self.replies.pop(0)
        text = r(messages) if callable(r) else r
        n = sum(len(str(m["content"]).split()) for m in messages)
        return LLMResponse(text=text, usage=Usage(n, None, len(text.split()), 0.0, {}),
                           thinking=None)


def reply(action, args, notes="n", cites=None):
    d = {"thought": "t", "notes": notes, "action": action, "args": args}
    if cites is not None:
        d["cites"] = cites
    return json.dumps(d)


def chunk(cid, n=40):
    return {"id": cid, "text": f"Title {cid}: " + "word " * n}


class Tools(dict):
    """search_chunks/read_chunks fakes that record the k of every search."""

    def __init__(self, results):
        self.ks, self._results = [], list(results)
        super().__init__(
            search_chunks=self._search, read_chunks=lambda d: [chunk(f"{d}#0")],
            search=lambda q, k: "obs", read=lambda d: "doc")

    def _search(self, query, k):
        self.ks.append(k)
        return self._results.pop(0)


def fmt(chunks):
    return "\n".join(f'[{c["id"]}] {c["text"]}' for c in chunks)


def act(action, query):
    return json.dumps({"action": action, "args": {"query": query}}, separators=(",", ":"))


# ------------------------------------------------------------ react / notes_reset / window2

@pytest.mark.parametrize("arm", list(ARM_POLICY))
def test_baseline_arms_are_the_v1_code_path_byte_for_byte(arm):
    script = [reply("search", {"query": f"q{i}"}) for i in range(4)] + [
        reply("finish", {"answer": "x"})]
    tools = {"search": lambda q, k: f"hits for {q}", "read": lambda d: "doc"}
    v1, v2 = Scripted(script), Scripted(script)
    r1 = run_agent("Q?", v1, ARM_POLICY[arm], 8, tools, SYSTEM_PROMPT)
    r2 = run_arm(arm, "Q?", v2, 8, tools)
    assert v1.calls == v2.calls
    assert v1.calls[0][0]["content"] == SYSTEM_PROMPT
    assert (r1.turns, r1.answer) == (r2.turns, r2.answer) == (5, "x")
    assert all(s["arm"] == arm and s["role"] == "agent" for s in r2.steps)


# ------------------------------------------------------------ pbc

def pbc_run(replies, results, max_steps=10, k0=2, k_hi=8, e_max=6000):
    prov, tools = Scripted(replies), Tools(results)
    res = run_pbc("Q?", prov, k0, tools, max_steps=max_steps, k_hi=k_hi, e_max=e_max)
    return res, prov, tools


def test_pbc_keep_drop_layout_and_prefix():
    s1, s2, s3 = ([chunk("a#0"), chunk("a#1"), chunk("b#0")], [chunk("c#0"), chunk("c#1")],
                  [chunk("d#0")])
    res, prov, _ = pbc_run([
        reply("search", {"query": "x"}, notes="n0", cites=[]),
        reply("search", {"query": "y"}, notes="n1", cites=["a#0", "b#0"]),
        reply("search", {"query": "z"}, notes="n2", cites=["c#1"]),
        reply("finish", {"answer": "ans"}, notes="n3", cites=["c#1"]),
    ], [s1, s2, s3])
    c = prov.calls
    assert c[0] == [{"role": "system", "content": PBC_SYSTEM_PROMPT},
                    {"role": "user", "content": "Question: Q?"}]
    # step 1: nothing cited yet -> no E; layout S, Q, notes, last action, last observation
    assert c[1] == c[0] + [
        {"role": "user", "content": "Notes: n0"},
        {"role": "assistant", "content": act("search", "x")},
        {"role": "user", "content": "Observation: " + fmt(s1)}]
    # step 2: a#0 and b#0 kept verbatim, a#1 dropped; the old observation is gone
    ev = "Evidence:\n" + fmt([s1[0], s1[2]])
    assert c[2] == c[0] + [
        {"role": "user", "content": ev},
        {"role": "user", "content": "Notes: n1"},
        {"role": "assistant", "content": act("search", "y")},
        {"role": "user", "content": "Observation: " + fmt(s2)}]
    assert all(s1[1]["text"] not in m["content"] for m in c[2])
    # step 3: E only grew by appending (same prefix); c#1 added, c#0 dropped
    assert c[3][2]["content"] == ev + "\n" + fmt([s2[1]])
    assert c[3][2]["content"].startswith(c[2][2]["content"]) and c[3][:2] == c[2][:2]
    assert all(s2[0]["text"] not in m["content"] for m in c[3])
    assert res.finished and res.answer == "ans" and res.grounded is True and res.turns == 4


def test_pbc_evict_takes_oldest_uncited_and_never_a_cited_chunk():
    ev = [("a#0", "x" * 100), ("b#0", "y" * 100), ("c#0", "z" * 100)]
    gone = evict(ev, {"a#0"}, e_max=250)  # 3 x ~108 > 250: oldest NOT cited (b#0) goes
    assert gone == ["b#0"] and [i for i, _ in ev] == ["a#0", "c#0"]
    ev = [("a#0", "x" * 100), ("b#0", "y" * 100)]
    assert evict(ev, {"a#0", "b#0"}, e_max=10) == [] and len(ev) == 2  # all cited


def test_pbc_e_max_in_loop_evicts_uncited_but_not_the_current_citation():
    # each chunk line is ~330 chars; e_max 700 holds two
    s1, s2, s3 = [chunk("a#0", 60)], [chunk("b#0", 60)], [chunk("c#0", 60)]
    res, prov, _ = pbc_run([
        reply("search", {"query": "1"}),
        reply("search", {"query": "2"}, cites=["a#0"]),   # E = a
        reply("search", {"query": "3"}, cites=["b#0"]),   # E = a, b (fits)
        reply("search", {"query": "4"}, cites=["c#0"]),   # E = a, b, c over cap: a evicted
        reply("finish", {"answer": "x"}, cites=["b#0", "c#0"]),
    ], [s1, s2, s3, [chunk("d#0", 5)]], e_max=700)
    e = lambda i: next(m["content"] for m in prov.calls[i]
                       if m["content"].startswith("Evidence:"))
    assert "[a#0]" in e(3) and "[b#0]" in e(3) and "[c#0]" not in e(3)
    assert "[a#0]" not in e(4) and "[b#0]" in e(4) and "[c#0]" in e(4)
    assert [s["n_evicted"] for s in res.steps] == [0, 0, 0, 1, 0]


def test_pbc_protects_a_cited_chunk_even_when_over_the_cap():
    s1 = [chunk("a#0", 60)]
    res, prov, _ = pbc_run([
        reply("search", {"query": "1"}),
        reply("search", {"query": "2"}, cites=["a#0"]),
        reply("finish", {"answer": "x"}, cites=["a#0"]),
    ], [s1, [chunk("b#0", 5)]], e_max=10)  # cap far below one chunk
    assert any(m["content"].startswith("Evidence:\n[a#0]") for m in prov.calls[2])
    assert res.steps[1]["n_evicted"] == 0 and res.grounded is True


def test_pbc_escalates_k_after_a_miss_then_resets():
    r = [[chunk(f"s{i}#0")] for i in range(5)]
    res, prov, tools = pbc_run([
        reply("search", {"query": "1"}, cites=[]),
        reply("search", {"query": "2"}, cites=[]),          # s0 uncited -> miss -> next k_hi
        reply("search", {"query": "3"}, cites=[]),          # s1 (k_hi) -> back to k0
        reply("search", {"query": "4"}, cites=["s2#0"]),    # s2 cited -> no miss -> k0
        reply("search", {"query": "5"}, cites=[]),
        reply("finish", {"answer": "x"}, cites=["s2#0"]),
    ], r, k0=2, k_hi=8)
    # step 4 sees s3 (a k0 search) uncited -> it is a miss again, so its search widens
    assert tools.ks == [2, 8, 2, 2, 8]
    assert [s["search_k"] for s in res.steps if s["action"] == "search"] == [2, 8, 2, 2, 8]
    assert [s["miss"] for s in res.steps] == [False, True, False, False, True, False]
    assert all(s["k"] == 2 for s in res.steps)  # the cell k stays k0


def test_pbc_finish_without_grounding_is_rejected_once_then_flagged():
    res, prov, _ = pbc_run([
        reply("search", {"query": "x"}, cites=[]),
        reply("finish", {"answer": "first"}, cites=[]),
        reply("finish", {"answer": "second"}, cites=["not#retrieved"]),
    ], [[chunk("a#0")]])
    assert res.finished and res.answer == "second" and res.grounded is False
    assert res.turns == 3
    assert prov.calls[2][-1] == {"role": "user", "content": "Observation: " + PBC_REJECT_MESSAGE}


def test_pbc_second_finish_with_a_retrieved_citation_is_grounded():
    res, _, _ = pbc_run([
        reply("search", {"query": "x"}),
        reply("finish", {"answer": "a"}, cites=[]),
        reply("finish", {"answer": "b"}, cites=["a#0"]),
    ], [[chunk("a#0")]])
    assert res.answer == "b" and res.grounded is True


def test_pbc_parse_error_costs_a_step():
    fin = reply("finish", {"answer": "x"}, cites=[])
    res, prov, _ = pbc_run(["not json", fin, fin], [])  # nothing retrieved: finish is rejected once
    assert res.parse_errors == 1 and res.turns == 3
    assert prov.calls[1][-2]["content"] == "not json"


# ------------------------------------------------------------ memgpt

def summariser_or(agent_replies):
    it = iter(agent_replies)

    def f(messages):
        if messages[0]["content"] == SUMMARIZER_PROMPT:
            return json.dumps({"summary": "SUM"})
        return next(it)
    return f


def test_memgpt_summariser_triggers_at_threshold_and_is_not_a_turn():
    big = "o" * 2500
    tools = {"search": lambda q, k: big, "read": lambda d: big}
    f = summariser_or([reply("search", {"query": str(i)}) for i in range(10)])
    prov = Scripted([f] * 7)
    res = run_memgpt("Q?", prov, 8, tools, max_steps=5, threshold=6000)
    sums = [i for i, c in enumerate(prov.calls) if c[0]["content"] == SUMMARIZER_PROMPT]
    # pairs are ~2600 chars: 2 pairs = 5.2k < 6k, 3 pairs = 7.8k > 6k -> fold the oldest 1
    assert sums[0] == 3  # agent steps 0..2, then the summariser, then step 3
    assert big in prov.calls[3][1]["content"]
    nxt = prov.calls[4]
    assert nxt[0]["content"] == SYSTEM_PROMPT and nxt[1]["content"] == "Question: Q?"
    assert nxt[2] == {"role": "user", "content": "Summary of earlier steps: SUM"}
    assert [m["role"] for m in nxt[3:]] == ["assistant", "user", "assistant", "user"]
    # the turn cap counts agent steps only
    assert res.turns == 5 and len(res.steps) == 5 + len(sums)
    roles = [s["role"] for s in res.steps]
    assert roles.count("summarizer") == len(sums) and roles.count("agent") == 5
    assert sum(s["input_tokens"] for s in res.steps) == res.total_input_tokens


def test_memgpt_below_threshold_never_summarises():
    tools = {"search": lambda q, k: "short", "read": lambda d: "doc"}
    prov = Scripted([reply("search", {"query": "a"}), reply("search", {"query": "b"}),
                     reply("finish", {"answer": "x"})])
    res = run_memgpt("Q?", prov, 8, tools, threshold=6000)
    assert len(prov.calls) == 3 and res.turns == 3 and res.finished
    assert all(s["role"] == "agent" for s in res.steps)
    assert prov.calls[2][:2] == [{"role": "system", "content": SYSTEM_PROMPT},
                                 {"role": "user", "content": "Question: Q?"}]
    assert len(prov.calls[2]) == 2 + 4  # carry-all until the threshold


def test_memgpt_summariser_requests_do_not_consume_the_turn_cap():
    tools = {"search": lambda q, k: "x" * 50, "read": lambda d: "doc"}
    f = summariser_or([reply("search", {"query": str(i)}) for i in range(10)])
    prov = Scripted([f] * 20)
    res = run_memgpt("Q?", prov, 8, tools, max_steps=3, threshold=10)
    assert res.turns == 3 and not res.finished
    assert len(res.steps) > 3
    assert [s["step"] for s in res.steps] == list(range(len(res.steps)))


# ------------------------------------------------------------ agentless

def test_agentless_is_exactly_two_requests_with_fixed_messages():
    first, second = [chunk("a#0"), chunk("b#0")], [chunk("b#0"), chunk("c#0")]
    tools = Tools([first, second])
    prov = Scripted([json.dumps({"query": "follow up"}),
                     json.dumps({"answer": "Paris", "cites": ["c#0"]})])
    res = run_agentless("Who?", prov, 8, tools)
    assert len(prov.calls) == 2 and res.turns == 2 and len(res.steps) == 2
    assert tools.ks == [8, 8]
    assert prov.calls[0] == [
        {"role": "system", "content": AGENTLESS_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: Who?\n\nPassages:\n{fmt(first)}\n\n"
                                    f"{AGENTLESS_QUERY_INSTRUCTION}"}]
    assert prov.calls[1] == [
        {"role": "system", "content": AGENTLESS_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: Who?\n\nPassages (question search):\n"
                                    f"{fmt(first)}\n\nPassages (follow-up search):\n"
                                    f"{fmt(second[1:])}\n\n{AGENTLESS_ANSWER_INSTRUCTION}"}]
    assert res.finished and res.answer == "Paris" and res.grounded is True


def test_agentless_ungrounded_and_unparseable_still_two_requests():
    tools = Tools([[chunk("a#0")], [chunk("a#0")]])
    prov = Scripted(["garbage", json.dumps({"answer": "x", "cites": ["zzz#9"]})])
    res = run_agentless("Q?", prov, 8, tools)
    assert len(prov.calls) == 2 and res.parse_errors == 1
    assert res.finished and res.grounded is False


# ------------------------------------------------------------ config cells

def test_cells_from_config_follow_the_prereg_table():
    with open("experiments/configs/h2h.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    assert cells_from_config(cfg) == [
        ("react", 2), ("react", 8), ("notes_reset", 2), ("notes_reset", 8),
        ("window2", 8), ("memgpt", 8), ("agentless", 8), ("pbc", 3)]
    assert cfg["pbc"] == {"k0": 3, "k_hi": 8, "e_max": 6000}
    assert cfg["agent"]["cell_order_seed"] == 20261117
    # PREREG_v2 Amendment 2 (B1, B3): provider/model changed before any call
    assert cfg["models"] == [{"provider": "openai", "model": "gpt-6-luna",
                              "reasoning_effort": "none", "max_output_tokens": 4096}]
