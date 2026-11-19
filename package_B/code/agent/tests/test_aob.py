"""PREREG_v3 section 4: AoB (Answer-or-Batch), its ablations, the references and the v3 baseline
prompts. Scripted provider, no network."""
import json

import pytest

from agent import arms
from agent.arms import (AOB_ABLATIONS, aob_first_order, aob_next_order, aob_pack,
                        aob_prefetch, aob_supported, run_arm)
from agent.data import build_kb
from agent.prompts import ANSWER_CONTRACT, AOB_PROMPT, AOB_RETRY_LINE
from agent.providers import LLMResponse, Usage
from agent.retrieval import Retriever


class Scripted:
    """Replies from a list (the last one repeats); records every request's messages."""
    name, model = "mock", "scripted"

    def __init__(self, replies):
        self.replies, self.sent = list(replies), []

    def chat(self, messages, **opts):
        self.sent.append([dict(m) for m in messages])
        text = self.replies[min(len(self.sent) - 1, len(self.replies) - 1)]
        text = text if isinstance(text, str) else json.dumps(text)
        n = sum(len(str(m["content"]).split()) for m in messages)
        return LLMResponse(text=text, usage=Usage(input_tokens=n, cached_input_tokens=0,
                                                  output_tokens=5, latency_s=0.0, raw={}),
                           thinking=None)


def _kb(n_docs=30, sent_len=1):
    titles, sents = [], []
    for i in range(n_docs):
        titles.append(f"Doc{i}")
        sents.append([f"Doc{i} fact alpha{i} beta{i % 5} " + "filler words here " * sent_len])
    titles += ["Gamma City", "Alpha River", "Mount Beta"]
    sents += [["Gamma City is the capital of Testland.", "It lies on the Alpha River."],
              ["Alpha River flows through Testland.", "Its source is Mount Beta."],
              ["Mount Beta is a peak in Testland."]]
    return Retriever(build_kb([{"id": "x", "context": {"title": titles, "sentences": sents}}]))


def _tools(r):
    return {"search_chunks": r.search_chunks, "read_chunks": r.read_chunks,
            "search": r.search_observation, "read": r.read_observation}


Q = "Which mountain is the source of the river that flows through the capital of Testland?"


# ---------------------------------------------------------------- the gate

@pytest.mark.parametrize("answer,text,expected", [
    ("Mount Beta", "Its source is Mount Beta.", "span"),
    ("the Mount Beta.", "its source is mount beta", "span"),  # official normalisation
    ("Yes", "", "yesno"), ("no.", "anything", "yesno"),
    ("Mount", "Mountain view", None),  # token containment, not substring
    ("Beta Mount", "Mount Beta", None),  # contiguous and ordered
    ("", "text", None), ("   ", "text", None),
])
def test_gate_is_normalised_contiguous_token_containment(answer, text, expected):
    assert aob_supported(answer, text) == expected


# ---------------------------------------------------------------- prefetch, order, packing

def test_prefetch_hop_adds_new_documents_only():
    r = _kb()
    e1, e2 = aob_prefetch(Q, r.search_chunks, 6, 4, True)
    assert len(e1) == 6 and len(e2) == 4
    assert not {c["id"].split("#")[0] for c in e2} & {c["id"].split("#")[0] for c in e1}
    assert e1 == r.search_chunks(Q, 6)
    prf = r.search_chunks(Q + " " + e1[0]["text"], 16)
    assert e2 == [c for c in prf if c["id"].split("#")[0]
                  not in {x["id"].split("#")[0] for x in e1}][:4]


def test_prefetch_without_hop_is_question_only_at_the_same_depth():
    r = _kb()
    e1, e2 = aob_prefetch(Q, r.search_chunks, 6, 4, False)
    assert e2 == [] and e1 == r.search_chunks(Q, 10)


def test_first_order_and_next_order():
    e1 = [{"id": f"a{i}"} for i in range(6)]
    e2 = [{"id": f"b{i}"} for i in range(4)]
    assert [c["id"] for c in aob_first_order(e1, e2)] == [
        "a0", "a1", "b0", "b1", "a2", "b2", "a3", "b3", "a4", "a5"]
    n1, n2 = [{"id": "n0"}, {"id": "n1"}], [{"id": "m0"}, {"id": "m1"}, {"id": "m2"}]
    prev = [{"id": f"p{i}"} for i in range(5)]
    assert [c["id"] for c in aob_next_order([n1, n2], prev)] == [
        "n0", "m0", "p0", "p1", "p2", "n1", "m1", "m2", "p3", "p4"]


def test_pack_skips_what_does_not_fit_and_keeps_trying():
    cands = [{"id": "a", "text": "x" * 50}, {"id": "b", "text": "y" * 500},
             {"id": "c", "text": "z" * 30}, {"id": "a", "text": "x" * 50}]
    assert [c["id"] for c in aob_pack(cands, 100, 200)] == ["a", "c"]
    assert [c["id"] for c in aob_pack(cands, 100, None)] == ["a", "b", "c"]
    assert aob_pack(cands, 199, 200) == []


# ---------------------------------------------------------------- the controller

def _run(replies, arm="aob", r=None, **params):
    p = Scripted(replies)
    res = run_arm(arm, Q, p, 6, _tools(r or _kb()), harness="v3", params=params)
    return res, p


def test_supported_first_answer_commits_after_one_request():
    res, p = _run([{"answer": "Mount Beta", "next": ["Alpha River source"]}])
    assert len(p.sent) == 1 and res.finished and res.answer == "Mount Beta"
    assert res.grounded is True and res.steps[0]["gate"] == "span"
    msgs = p.sent[0]
    assert msgs[0] == {"role": "system", "content": AOB_PROMPT}
    assert msgs[1]["content"].startswith(f"Question: {Q}\n\nPassages:\n")
    assert "[" not in msgs[1]["content"].split("Passages:\n", 1)[1][:1]  # no ids


def test_unsupported_answer_runs_one_batched_follow_up_with_the_retry_line():
    r = _kb()
    res, p = _run([{"answer": "Mount Zeta", "next": ["Doc7 alpha7", "Doc9 alpha9"]},
                   {"answer": "Mount Beta", "next": []}], r=r)
    assert len(p.sent) == 2 and res.answer == "Mount Beta" and res.grounded is True
    second = p.sent[1][1]["content"]
    assert second.endswith(AOB_RETRY_LINE.format(answer="Mount Zeta"))
    passages = second.split("Passages:\n", 1)[1].split("\n\n")[0].splitlines()
    first_new = [c["text"] for c in r.search_chunks("Doc7 alpha7", 16)
                 if c["text"] not in p.sent[0][1]["content"]][0]
    assert passages[0] == first_new  # the new chunk of query 1 comes first
    q1_new = [c["text"] for c in r.search_chunks("Doc7 alpha7", 16)
              if c["text"] not in p.sent[0][1]["content"]][:4]
    second_new = [c["text"] for c in r.search_chunks("Doc9 alpha9", 16)
                  if c["text"] not in p.sent[0][1]["content"] and c["text"] not in q1_new][0]
    assert passages[1] == second_new  # then query 2's: both queries run in one batch
    assert res.steps[0]["n_new_chunks"] == 8  # m3 = 4 new chunks per query
    assert [s["committed"] for s in res.steps] == [False, True]


def test_never_supported_stops_at_r_max_flagged_unsupported():
    replies = [{"answer": f"Nowhere{i}", "next": [f"Doc{i} alpha{i}", f"Doc{i + 10}"]}
               for i in range(5)]
    res, p = _run(replies)
    assert len(p.sent) == 3 and res.finished and res.grounded is False
    assert res.answer == "Nowhere2" and res.steps[-1]["committed"] is True


def test_no_new_chunk_commits_unsupported():
    res, p = _run([{"answer": "Nowhere", "next": []}])
    assert len(p.sent) == 1 and res.grounded is False and res.answer == "Nowhere"


def test_unparseable_reply_counts_as_an_empty_answer():
    res, p = _run(["not json at all"])
    assert len(p.sent) == 1 and res.answer == "" and res.parse_errors == 1


def test_every_request_stays_within_the_character_budget():
    r = _kb(n_docs=60, sent_len=25)  # long passages: the budget must bind
    replies = [{"answer": "Nowhere", "next": ["Doc3 alpha3", "Doc40 alpha40"]}] * 3
    res, p = _run(replies, r=r)
    assert len(p.sent) == 3
    for msgs in p.sent:
        chars = sum(len(m["content"]) for m in msgs)
        assert chars <= 3500, chars
    assert all(s["prompt_chars"] <= 3500 for s in res.steps)
    # the no-pack ablation shows everything and goes over
    res2, p2 = _run(replies[:1], arm="aob_nopack", r=r)
    assert sum(len(m["content"]) for m in p2.sent[0]) > 3500


def test_nogate_ablation_always_commits_after_one_request():
    res, p = _run([{"answer": "Nowhere", "next": ["Doc3 alpha3"]}], arm="aob_nogate")
    assert len(p.sent) == 1 and res.grounded is False


def test_llmq_ablation_restores_the_llm_written_first_query():
    r = _kb()
    res, p = _run([{"query": "Doc12 alpha12"}, {"answer": "Doc12", "next": []}],
                  arm="aob_llmq", r=r)
    assert len(p.sent) == 2 and res.turns == 2
    assert "search query" in p.sent[0][1]["content"] and "Passages" not in p.sent[0][1]["content"]
    e1, _ = aob_prefetch("Doc12 alpha12", r.search_chunks, 6, 4, True)
    assert p.sent[1][1]["content"].split("Passages:\n", 1)[1].startswith(e1[0]["text"])


def test_each_ablation_changes_exactly_one_parameter():
    assert all(len(v) == 1 for v in AOB_ABLATIONS.values())
    cfg = {"aob": {"budget": 3200}}
    for arm, over in AOB_ABLATIONS.items():
        p = arms.arm_params(cfg, arm)
        base = arms.arm_params(cfg, "aob")
        assert {k for k in p if p[k] != base[k]} == set(over)


# ---------------------------------------------------------------- references and baselines

def test_rag1_is_one_request_over_the_question_top_k():
    r = _kb()
    p = Scripted([{"answer": "Mount Beta"}])
    res = run_arm("rag1", Q, p, 8, _tools(r), harness="v3")
    assert len(p.sent) == 1 and res.answer == "Mount Beta"
    body = p.sent[0][1]["content"]
    assert all(c["text"] in body for c in r.search_chunks(Q, 8))


def test_closed_book_sends_no_passages():
    p = Scripted([{"answer": "Mount Beta"}])
    res = run_arm("closed_book", Q, p, 0, _tools(_kb()), harness="v3")
    assert len(p.sent) == 1 and res.answer == "Mount Beta"
    assert "Passages" not in p.sent[0][1]["content"] and "Testland." not in p.sent[0][1]["content"]


@pytest.mark.parametrize("arm,k", [("react", 8), ("notes_reset", 8), ("window2", 8),
                                   ("memgpt", 8), ("agentless", 8), ("pbc", 3), ("rag1", 8),
                                   ("closed_book", 0), ("aob", 6)])
def test_every_v3_arm_carries_the_same_answer_contract(arm, k):
    p = Scripted([{"answer": "Mount Beta", "action": "finish", "args": {"answer": "x"},
                   "query": "q", "cites": []}])
    run_arm(arm, Q, p, k, _tools(_kb()), harness="v3")
    final = " ".join(str(m["content"]) for m in p.sent[-1])
    assert ANSWER_CONTRACT in final


@pytest.mark.parametrize("arm", ["react", "window2", "memgpt"])
def test_react_family_v3_prompts_have_no_notes_field(arm):
    p = Scripted([{"thought": "t", "action": "finish", "args": {"answer": "x"}}])
    run_arm(arm, Q, p, 8, _tools(_kb()), harness="v3")
    assert '"notes"' not in p.sent[0][0]["content"] and '"thought"' in p.sent[0][0]["content"]


def test_v2_harness_is_unchanged_for_the_old_arms():
    from agent.prompts import SYSTEM_PROMPT
    p = Scripted([{"thought": "t", "notes": "n", "action": "finish", "args": {"answer": "x"}}])
    run_arm("react", Q, p, 8, _tools(_kb()))
    assert p.sent[0][0]["content"] == SYSTEM_PROMPT
    with pytest.raises(ValueError):
        run_arm("aob", Q, p, 6, _tools(_kb()))
