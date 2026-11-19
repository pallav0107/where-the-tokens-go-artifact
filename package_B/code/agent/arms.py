"""PREREG_v2 section 4 arms: react, notes_reset, window2 (the v1 code paths), memgpt,
agentless and pbc (the prefix-budget controller).

Every arm returns an agent.agent.AgentResult and writes one request record per LLM request
(helper requests included, role "summarizer"); only role "agent" requests count as turns.
"""
from __future__ import annotations

import json

from agent.agent import (AgentResult, ERROR_OBSERVATION, KNOWN_ACTIONS, _extract_json,
                         make_record, run_agent)
from agent.metrics import normalize_answer
from agent.prompts import (AGENTLESS_ANSWER_INSTRUCTION, AGENTLESS_QUERY_INSTRUCTION,
                           AGENTLESS_SYSTEM_PROMPT, ANSWER_CONTRACT, AOB_PROMPT,
                           AOB_RETRY_LINE, AOB_THOUGHT_PROMPT, CLOSED_ANSWER_INSTRUCTION,
                           CLOSED_SYSTEM_PROMPT, NOTES_V3_PROMPT, PBC_REJECT_MESSAGE,
                           PBC_SYSTEM_PROMPT, QUERY_ONLY_INSTRUCTION, REACT_V3_ERROR,
                           REACT_V3_PROMPT, READER_ANSWER_INSTRUCTION, READER_SYSTEM_PROMPT,
                           SUMMARIZER_PROMPT, SYSTEM_PROMPT)
from agent.retrieval import UNKNOWN_DOC_MESSAGE

# The three baselines that reuse v1 code and the v1 prompt unchanged.
ARM_POLICY = {"react": "carry_all", "notes_reset": "reset_summary",
              "window2": "sliding_window:2"}
# PREREG_v3 section 4: AoB and its one-variable ablations (each = the frozen aob parameters with
# exactly one override), and the two reference points.
AOB_DEFAULTS = {"k1": 6, "m2": 4, "budget": 3500, "r_max": 3, "n_next": 2, "m3": 4,
                "hop": True, "llm_query": False, "thought": False}
AOB_ABLATIONS = {"aob_nogate": {"r_max": 1}, "aob_nohop": {"hop": False},
                 "aob_nopack": {"budget": None}, "aob_llmq": {"llm_query": True}}
AOB_FAMILY = ("aob", *AOB_ABLATIONS)
REFERENCES = ("rag1", "closed_book")
ARMS = (*ARM_POLICY, "memgpt", "agentless", "pbc", *REFERENCES, *AOB_FAMILY)
DEFAULT_PARAMS = {"memgpt": {"threshold": 6000},
                  "pbc": {"k0": 3, "k_hi": 8, "e_max": 6000}}


def cells_from_config(config: dict) -> list[tuple[str, int]]:
    """(arm, k) cells of an h2h config. An arm without `k` is pbc (k = pbc.k0) or an AoB-family
    arm (k = k1)."""
    cells = []
    for spec in config["arms"]:
        name = spec["name"]
        if name not in ARMS:
            raise ValueError(f"unknown arm: {name!r}")
        ks = spec.get("k")
        if ks is None:
            p = arm_params(config, name)
            if "k0" in p:
                ks = [p["k0"]]
            elif "k1" in p:
                ks = [p["k1"]]
            else:
                raise ValueError(f"arm {name!r} needs k")
        cells += [(name, int(k)) for k in (ks if isinstance(ks, list) else [ks])]
    return cells


def arm_params(config: dict, arm: str) -> dict:
    if arm in AOB_FAMILY:  # ablations inherit the (frozen) aob block and override one thing
        return {**AOB_DEFAULTS, **(config.get("aob") or {}), **AOB_ABLATIONS.get(arm, {})}
    return {**DEFAULT_PARAMS.get(arm, {}), **(config.get(arm) or {})}


class _Tag:
    """Logger wrapper that stamps arm/role on records of the unchanged v1 loop."""

    def __init__(self, logger, **fields):
        self.logger, self.fields = logger, fields

    def log(self, record):
        record.update(self.fields)
        if self.logger is not None:
            self.logger.log(record)


class _Session:
    """Token accounting and request logging shared by the new arms."""

    def __init__(self, arm, provider, k, logger, run_meta):
        self.arm, self.provider, self.k = arm, provider, k
        self.logger, self.run_meta = logger, run_meta or {}
        self.res = AgentResult()
        self._cached: list = []

    def ask(self, messages, role="agent"):
        resp = self.provider.chat(messages)
        u, res = resp.usage, self.res
        res.turns += role == "agent"
        res.total_input_tokens += u.input_tokens
        res.total_output_tokens += u.output_tokens
        res.total_latency_s += u.latency_s
        self._cached.append(u.cached_input_tokens)
        return resp

    def record(self, resp, messages, role="agent", action=None, parse_error=False,
               obs_chars=0, **extra):
        rec = make_record(self.provider, resp, messages, len(self.res.steps), self.arm,
                          self.k, self.run_meta, action, parse_error, obs_chars)
        rec.update(arm=self.arm, role=role, **extra)
        self.res.steps.append(rec)
        if self.logger is not None:
            self.logger.log(rec)

    def done(self):
        c = self._cached
        self.res.total_cached_tokens = sum(c) if all(x is not None for x in c) else None
        return self.res


def _parse_action(text):
    """(obj, action, args); action is None when the reply is not a valid v1 action."""
    obj = _extract_json(text)
    args = obj.get("args") if obj else None
    if obj is not None and obj.get("action") in KNOWN_ACTIONS and (
            args is None or isinstance(args, dict)):
        return obj, obj["action"], args or {}
    return obj, None, {}


def _cites(obj) -> list[str]:
    raw = (obj or {}).get("cites")
    if isinstance(raw, str):
        raw = [raw]
    return [str(c) for c in raw] if isinstance(raw, list) else []


def _fmt(chunks) -> str:
    return "\n".join(f'[{c["id"]}] {c["text"]}' for c in chunks)


# ---------------------------------------------------------------------------------------
# memgpt: summarise-and-evict
# ---------------------------------------------------------------------------------------

def _summarise(s: _Session, question, summary, pairs) -> str:
    steps = "\n\n".join(f"Action: {a}\nObservation: {o}" for a, o in pairs)
    messages = [
        {"role": "system", "content": SUMMARIZER_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nSummary so far: "
                                    f"{summary or '(none)'}\n\nSteps to fold in:\n{steps}"},
    ]
    resp = s.ask(messages, role="summarizer")
    s.record(resp, messages, role="summarizer")
    obj = _extract_json(resp.text)
    new = obj.get("summary") if obj else None
    return new if isinstance(new, str) else resp.text.strip()


def memgpt_messages(question, summary, pairs, system=SYSTEM_PROMPT):
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": f"Question: {question}"}]
    if summary:
        msgs.append({"role": "user", "content": f"Summary of earlier steps: {summary}"})
    for action_text, observation in pairs:
        msgs.append({"role": "assistant", "content": action_text})
        msgs.append({"role": "user", "content": f"Observation: {observation}"})
    return msgs


def run_memgpt(question, provider, k, tools, max_steps=10, logger=None, run_meta=None,
               threshold=6000, system_prompt=SYSTEM_PROMPT,
               error_observation=ERROR_OBSERVATION) -> AgentResult:
    """When the history after S and Q (summary + pairs) exceeds `threshold` characters, the
    oldest half of the action/observation pairs is folded into the summary by a separate
    summariser request (not a turn). The summary is sent after Q."""
    s = _Session("memgpt", provider, k, logger, run_meta)
    res, pairs, summary = s.res, [], ""
    for _ in range(max_steps):
        size = len(summary) + sum(len(a) + len(o) for a, o in pairs)
        if size > threshold and pairs:
            n = max(1, len(pairs) // 2)
            summary = _summarise(s, question, summary, pairs[:n])
            pairs = pairs[n:]
        messages = memgpt_messages(question, summary, pairs, system_prompt)
        resp = s.ask(messages)
        _, action, args = _parse_action(resp.text)
        observation = error_observation
        if action == "search":
            observation = tools["search"](str(args.get("query", "")), k)
        elif action == "read":
            observation = tools["read"](str(args.get("doc_id", "")))
        elif action == "finish":
            res.finished, res.answer, observation = True, str(args.get("answer", "")), ""
        else:
            res.parse_errors += 1
        pairs.append((resp.text, observation))
        s.record(resp, messages, action=action, parse_error=action is None,
                 obs_chars=len(observation), summary_chars=len(summary))
        if res.finished:
            break
    return s.done()


# ---------------------------------------------------------------------------------------
# agentless: fixed two-request pipeline
# ---------------------------------------------------------------------------------------

def run_agentless(question, provider, k, tools, logger=None, run_meta=None) -> AgentResult:
    s = _Session("agentless", provider, k, logger, run_meta)
    first = tools["search_chunks"](question, k)
    retrieved = {c["id"] for c in first}
    m1 = [{"role": "system", "content": AGENTLESS_SYSTEM_PROMPT},
          {"role": "user", "content": f"Question: {question}\n\nPassages:\n{_fmt(first)}\n\n"
                                      f"{AGENTLESS_QUERY_INSTRUCTION}"}]
    r1 = s.ask(m1)
    obj1 = _extract_json(r1.text)
    query = str((obj1 or {}).get("query") or "").strip()
    if not query:  # unparseable: fall back to re-using the question, and count the error
        s.res.parse_errors += 1
        query = question
    s.record(r1, m1, action="query", parse_error=not (obj1 or {}).get("query"))
    # The second set shows only passages not already in the first set.
    second = [c for c in tools["search_chunks"](query, k) if c["id"] not in retrieved]
    retrieved |= {c["id"] for c in second}
    m2 = [{"role": "system", "content": AGENTLESS_SYSTEM_PROMPT},
          {"role": "user", "content": f"Question: {question}\n\nPassages (question search):\n"
                                      f"{_fmt(first)}\n\nPassages (follow-up search):\n"
                                      f"{_fmt(second)}\n\n{AGENTLESS_ANSWER_INSTRUCTION}"}]
    r2 = s.ask(m2)
    obj2 = _extract_json(r2.text)
    ok = isinstance(obj2, dict) and "answer" in obj2
    if ok:
        s.res.finished, s.res.answer = True, str(obj2["answer"])
        s.res.grounded = any(c in retrieved for c in _cites(obj2))
    else:
        s.res.parse_errors += 1
    s.record(r2, m2, action="answer", parse_error=not ok)
    return s.done()


# ---------------------------------------------------------------------------------------
# pbc: prefix-budget controller
# ---------------------------------------------------------------------------------------

def _evidence_line(chunk_id, text):
    return f"[{chunk_id}] {text}"


def evict(evidence: list, protected: set, e_max: int) -> list:
    """Drop the oldest chunks not in `protected` until E fits e_max characters (item 2).
    evidence is a list of (chunk_id, text) in append order; returns the evicted ids."""
    def size():
        return sum(len(_evidence_line(i, t)) + 1 for i, t in evidence)

    evicted = []
    while size() > e_max:
        idx = next((j for j, (i, _) in enumerate(evidence) if i not in protected), None)
        if idx is None:
            break  # everything left is cited by the current step
        evicted.append(evidence.pop(idx)[0])
    return evicted


def pbc_messages(question, evidence, notes, last_action, last_observation,
                 system=PBC_SYSTEM_PROMPT):
    """Layout [S][Q][E][notes][last action][last observation]; S, Q and E only grow by
    appending (E is one message extended line by line)."""
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": f"Question: {question}"}]
    if evidence:
        lines = "\n".join(_evidence_line(i, t) for i, t in evidence)
        msgs.append({"role": "user", "content": f"Evidence:\n{lines}"})
    if notes:
        msgs.append({"role": "user", "content": f"Notes: {notes}"})
    if last_action is not None:
        msgs.append({"role": "assistant", "content": last_action})
        msgs.append({"role": "user", "content": f"Observation: {last_observation}"})
    return msgs


def run_pbc(question, provider, k, tools, max_steps=10, logger=None, run_meta=None,
            k_hi=8, e_max=6000, system_prompt=PBC_SYSTEM_PROMPT) -> AgentResult:
    """k is k0, the width of a normal search; k_hi is used once after a missed k0 search."""
    s = _Session("pbc", provider, k, logger, run_meta)
    res = s.res
    evidence: list = []  # (chunk_id, text), append order
    notes, last_action, last_obs = "", None, ""
    last_chunks: list = []  # chunks of the observation the model is looking at
    last_search_k = None  # width of the search that produced last_chunks (None: not a search)
    retrieved: set = set()
    rejected = False
    for _ in range(max_steps):
        messages = pbc_messages(question, evidence, notes, last_action, last_obs,
                                system_prompt)
        resp = s.ask(messages)
        obj, action, args = _parse_action(resp.text)
        cites = set(_cites(obj))
        # 1 + 2: keep cited chunks of the last observation, evict if over e_max
        have = {i for i, _ in evidence}
        for c in last_chunks:
            if c["id"] in cites and c["id"] not in have:
                evidence.append((c["id"], c["text"]))
        evicted = evict(evidence, cites, e_max)
        # 4: a k0 search none of whose chunks were cited is a miss -> next search uses k_hi
        miss = (last_search_k == k and bool(last_chunks)
                and not any(c["id"] in cites for c in last_chunks))
        search_k = k_hi if miss else k
        chunks, last_search_k, parse_error = [], None, action is None
        if action == "search":
            chunks = tools["search_chunks"](str(args.get("query", "")), search_k)
            last_search_k = search_k
            obs = _fmt(chunks)
        elif action == "read":
            doc_id = str(args.get("doc_id", ""))
            chunks = tools["read_chunks"](doc_id)
            obs = _fmt(chunks) if chunks else UNKNOWN_DOC_MESSAGE.format(doc_id=doc_id)
        elif action == "finish":
            obs = ""
            grounded = bool(cites & retrieved)  # 5: must cite a chunk retrieved in this run
            if not grounded and not rejected:
                rejected, obs = True, PBC_REJECT_MESSAGE
            else:
                res.finished, res.answer, res.grounded = True, str(args.get("answer", "")), grounded
        else:
            obs = ERROR_OBSERVATION
            res.parse_errors += 1
        retrieved |= {c["id"] for c in chunks}
        if isinstance(obj, dict) and isinstance(obj.get("notes"), str):
            notes = obj["notes"]
        last_action = resp.text if parse_error else json.dumps(
            {"action": action, "args": args}, separators=(",", ":"))
        last_obs, last_chunks = obs, chunks
        s.record(resp, messages, action=action, parse_error=parse_error, obs_chars=len(obs),
                 search_k=search_k if action == "search" else None, miss=miss,
                 evidence_chars=sum(len(t) for _, t in evidence), n_evicted=len(evicted))
        if res.finished:
            break
    return s.done()


# ---------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------
# PREREG_v3 (experiments/PREREG_v3.md, section 4)
# ---------------------------------------------------------------------------------------

def react_v3_prompt(max_steps: int) -> str:
    return REACT_V3_PROMPT.replace("{max_steps}", str(int(max_steps)))


def notes_v3_prompt(max_steps: int) -> str:
    return NOTES_V3_PROMPT.replace("{max_steps}", str(int(max_steps)))


# pbc (superseded v2 controller): its v2 prompt with the shared answer contract (H3).
PBC_V3_PROMPT = PBC_SYSTEM_PROMPT.replace(
    "- When you have enough information, call finish with the shortest correct answer.\n",
    "- " + ANSWER_CONTRACT + "\n- When you have enough information, call finish.\n")
assert PBC_V3_PROMPT != PBC_SYSTEM_PROMPT


def _plain(chunks) -> str:
    """Passages as their "Title: text" lines, without ids (no v3 arm cites)."""
    return "\n".join(c["text"] for c in chunks)


def _answer_of(obj):
    if isinstance(obj, dict) and "answer" in obj and not isinstance(obj["answer"], (dict, list)):
        return str(obj["answer"])
    return None


def run_agentless_v3(question, provider, k, tools, logger=None, run_meta=None) -> AgentResult:
    """Agentless fixed phases (Xia et al. 2024): retrieve for the question; one request writes a
    follow-up query; retrieve for it; one request answers from both passage sets. No cites."""
    s = _Session("agentless", provider, k, logger, run_meta)
    first = tools["search_chunks"](question, k)
    shown = {c["id"] for c in first}
    m1 = [{"role": "system", "content": READER_SYSTEM_PROMPT},
          {"role": "user", "content": f"Question: {question}\n\nPassages:\n{_plain(first)}\n\n"
                                      f"{AGENTLESS_QUERY_INSTRUCTION}"}]
    r1 = s.ask(m1)
    obj1 = _extract_json(r1.text)
    query = str((obj1 or {}).get("query") or "").strip()
    if not query:
        s.res.parse_errors += 1
        query = question
    s.record(r1, m1, action="query", parse_error=not (obj1 or {}).get("query"))
    second = [c for c in tools["search_chunks"](query, k) if c["id"] not in shown]
    m2 = [{"role": "system", "content": READER_SYSTEM_PROMPT},
          {"role": "user", "content": f"Question: {question}\n\nPassages (question search):\n"
                                      f"{_plain(first)}\n\nPassages (follow-up search):\n"
                                      f"{_plain(second)}\n\n{READER_ANSWER_INSTRUCTION}"}]
    r2 = s.ask(m2)
    ans = _answer_of(_extract_json(r2.text))
    if ans is not None:
        s.res.finished, s.res.answer = True, ans
    else:
        s.res.parse_errors += 1
    s.record(r2, m2, action="answer", parse_error=ans is None)
    return s.done()


def run_rag1(question, provider, k, tools, logger=None, run_meta=None) -> AgentResult:
    """Reference point (not a published agent): one retrieve-then-read request over the
    question's top-k chunks, with agentless's answer request format."""
    s = _Session("rag1", provider, k, logger, run_meta)
    chunks = tools["search_chunks"](question, k)
    m = [{"role": "system", "content": READER_SYSTEM_PROMPT},
         {"role": "user", "content": f"Question: {question}\n\nPassages:\n{_plain(chunks)}\n\n"
                                     f"{READER_ANSWER_INSTRUCTION}"}]
    r = s.ask(m)
    ans = _answer_of(_extract_json(r.text))
    if ans is not None:
        s.res.finished, s.res.answer = True, ans
    else:
        s.res.parse_errors += 1
    s.record(r, m, action="answer", parse_error=ans is None)
    return s.done()


def run_closed_book(question, provider, k=0, logger=None, run_meta=None) -> AgentResult:
    """Reference point: one request with no passages (the model's own knowledge)."""
    s = _Session("closed_book", provider, k, logger, run_meta)
    m = [{"role": "system", "content": CLOSED_SYSTEM_PROMPT},
         {"role": "user", "content": f"Question: {question}\n\n{CLOSED_ANSWER_INSTRUCTION}"}]
    r = s.ask(m)
    ans = _answer_of(_extract_json(r.text))
    if ans is not None:
        s.res.finished, s.res.answer = True, ans
    else:
        s.res.parse_errors += 1
    s.record(r, m, action="answer", parse_error=ans is None)
    return s.done()


# ---- AoB (Answer-or-Batch) -------------------------------------------------------------

def _doc(chunk) -> str:
    return chunk["id"].split("#", 1)[0]


def aob_supported(answer: str, passages_text: str) -> str | None:
    """The free span-support gate. Returns "yesno" or "span" when the answer is supported (its
    official normalisation is yes/no, or its token sequence occurs contiguously in the
    normalised passages), else None."""
    a = normalize_answer(answer)
    if a in ("yes", "no"):
        return "yesno"
    if a and f" {a} " in f" {normalize_answer(passages_text)} ":
        return "span"
    return None


def aob_prefetch(query: str, search, k1: int, m2: int, hop: bool):
    """Deterministic prefetch, no LLM call: E1 = top-k1 chunks for the query; with the hop, E2 =
    the first m2 chunks of a pseudo-relevance-feedback search (query + E1[0] text, depth
    k1+m2+6) whose document is not in E1. Without the hop, E1 = top-(k1+m2) chunks, E2 = []."""
    if not hop:
        return search(query, k1 + m2), []
    e1 = search(query, k1)
    e2 = []
    if m2 > 0 and e1:
        docs = {_doc(c) for c in e1}
        for c in search(query + " " + e1[0]["text"], k1 + m2 + 6):
            if _doc(c) not in docs and len(e2) < m2:
                e2.append(c)
    return e1, e2


def aob_first_order(e1: list, e2: list) -> list:
    """Priority for request 1: E1[0], E1[1], E2[0], then E2[i] and E1[j] alternating."""
    order = list(e1[:2]) + list(e2[:1])
    a, b = list(e2[1:]), list(e1[2:])
    while a or b:
        if a:
            order.append(a.pop(0))
        if b:
            order.append(b.pop(0))
    return order


def aob_next_order(new_lists: list, prev_pack: list) -> list:
    """Priority after a batch: the first new chunk of each query, the first three chunks of the
    previous pack, the remaining new chunks round-robin, then the rest of the previous pack."""
    order = [lst[0] for lst in new_lists if lst]
    order += list(prev_pack[:3])
    rest = [list(lst[1:]) for lst in new_lists]
    while any(rest):
        for lst in rest:
            if lst:
                order.append(lst.pop(0))
    return order + list(prev_pack[3:])


def aob_pack(candidates: list, fixed_chars: int, budget) -> list:
    """Append each candidate (deduplicated by id) only if the request's running character total
    stays within `budget`; otherwise skip it and try the next (deterministic). budget None =
    no limit (the aob_nopack ablation)."""
    out, used, seen = [], fixed_chars, set()
    for c in candidates:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        add = len(c["text"]) + 1  # the line plus its newline
        if budget is None or used + add <= budget:
            out.append(c)
            used += add
    return out


def _next_queries(obj, n_next: int) -> list:
    raw = (obj or {}).get("next") if isinstance(obj, dict) else None
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [str(q).strip() for q in raw if isinstance(q, (str, int, float)) and str(q).strip()][
        :n_next]


def aob_messages(question, pack, prev_answer=None, thought=False):
    user = f"Question: {question}\n\nPassages:\n{_plain(pack)}"
    if prev_answer is not None:
        user += "\n\n" + AOB_RETRY_LINE.format(answer=prev_answer)
    return [{"role": "system", "content": AOB_THOUGHT_PROMPT if thought else AOB_PROMPT},
            {"role": "user", "content": user}]


def _fixed_chars(question, prev_answer, thought):
    return sum(len(m["content"]) for m in aob_messages(question, [], prev_answer, thought))


def run_aob(question, provider, tools, logger=None, run_meta=None, arm="aob", k1=6, m2=4,
            budget=3500, r_max=3, n_next=2, m3=4, hop=True, llm_query=False,
            thought=False) -> AgentResult:
    """AoB: prefetch (question + one zero-LLM PRF hop) before request 1; each request returns
    {answer, next}; the span-support gate commits the answer or runs ONE batched follow-up
    retrieval for all `next` queries; at most r_max answer requests. Every request's characters
    (system prompt, question, passages, retry line) stay within `budget`, which keeps it below
    the provider's 1,024-token cache minimum. aob_llmq adds one LLM request that writes the
    first query (the step AoB removes)."""
    s = _Session(arm, provider, k1, logger, run_meta)
    search = tools["search_chunks"]
    res = s.res
    query = question
    if llm_query:  # ablation: restore the LLM-written first query (one extra request)
        mq = [{"role": "system", "content": CLOSED_SYSTEM_PROMPT},
              {"role": "user", "content": f"Question: {question}\n\n{QUERY_ONLY_INSTRUCTION}"}]
        rq = s.ask(mq)
        objq = _extract_json(rq.text)
        q_llm = str((objq or {}).get("query") or "").strip() if isinstance(objq, dict) else ""
        if not q_llm:
            res.parse_errors += 1
        query = q_llm or question
        s.record(rq, mq, action="query", parse_error=not q_llm)
    e1, e2 = aob_prefetch(query, search, k1, m2, hop)
    pack = aob_pack(aob_first_order(e1, e2), _fixed_chars(question, None, thought), budget)
    shown = {c["id"] for c in pack}
    prev_answer = None
    for r in range(1, r_max + 1):
        messages = aob_messages(question, pack, prev_answer, thought)
        resp = s.ask(messages)
        obj = _extract_json(resp.text)
        answer = _answer_of(obj)
        parse_error = answer is None
        if parse_error:
            res.parse_errors += 1
            answer = ""
        gate = aob_supported(answer, _plain(pack))
        queries = _next_queries(obj, n_next)
        new_lists: list = []
        if gate is None and r < r_max:
            taken = set(shown)
            for q in queries:
                lst = []
                for c in search(q, k1 + m3 + 6):
                    if c["id"] not in taken and len(lst) < m3:
                        lst.append(c)
                        taken.add(c["id"])
                new_lists.append(lst)
        escalate = gate is None and r < r_max and any(new_lists)
        s.record(resp, messages, action="answer", parse_error=parse_error, aob_request=r,
                 gate=gate or "fail", committed=not escalate, n_passages=len(pack),
                 prompt_chars=sum(len(m["content"]) for m in messages),
                 next_queries=queries, n_new_chunks=sum(len(x) for x in new_lists))
        if not escalate:
            res.finished, res.answer, res.grounded = True, answer, gate is not None
            break
        prev_answer = answer
        pack = aob_pack(aob_next_order(new_lists, pack),
                        _fixed_chars(question, prev_answer, thought), budget)
        shown |= {c["id"] for c in pack}
    return s.done()


# ---------------------------------------------------------------------------------------

def run_arm(arm, question, provider, k, tools, max_steps=10, logger=None, run_meta=None,
            params=None, harness=None) -> AgentResult:
    """Run one arm. `params` overrides DEFAULT_PARAMS[arm]. harness "v3" selects the PREREG_v3
    settings (faithful baseline prompts, the shared answer contract, the new arms)."""
    if harness == "v3":
        return run_arm_v3(arm, question, provider, k, tools, max_steps, logger, run_meta, params)
    params = {**DEFAULT_PARAMS.get(arm, {}), **(params or {})}
    if arm in ARM_POLICY:  # unchanged v1 code path and prompt
        tagged = _Tag(logger, arm=arm, role="agent")
        return run_agent(question=question, provider=provider, policy=ARM_POLICY[arm], k=k,
                         tools=tools, system_prompt=SYSTEM_PROMPT, max_steps=max_steps,
                         logger=tagged, run_meta=run_meta)
    if arm == "memgpt":
        return run_memgpt(question, provider, k, tools, max_steps, logger, run_meta,
                          threshold=int(params["threshold"]))
    if arm == "agentless":
        return run_agentless(question, provider, k, tools, logger, run_meta)
    if arm == "pbc":
        return run_pbc(question, provider, k, tools, max_steps, logger, run_meta,
                       k_hi=int(params["k_hi"]), e_max=int(params["e_max"]))
    raise ValueError(f"unknown arm for the v1/v2 harness: {arm!r}")


def run_arm_v3(arm, question, provider, k, tools, max_steps=10, logger=None, run_meta=None,
               params=None) -> AgentResult:
    """PREREG_v3 section 4. max_steps is the harness cap; an arm's `max_steps` parameter
    (react: 7, the ReAct HotpotQA setting) overrides it."""
    if arm in AOB_FAMILY:
        p = {**AOB_DEFAULTS, **(params or {}), **AOB_ABLATIONS.get(arm, {})}
        return run_aob(question, provider, tools, logger, run_meta, arm=arm,
                       **{n: p[n] for n in AOB_DEFAULTS})
    params = {**DEFAULT_PARAMS.get(arm, {}), **(params or {})}
    steps = int(params.get("max_steps", max_steps))
    if arm in ("react", "window2"):
        tagged = _Tag(logger, arm=arm, role="agent")
        return run_agent(question=question, provider=provider, policy=ARM_POLICY[arm], k=k,
                         tools=tools, system_prompt=react_v3_prompt(steps), max_steps=steps,
                         logger=tagged, run_meta=run_meta, error_observation=REACT_V3_ERROR)
    if arm == "notes_reset":
        tagged = _Tag(logger, arm=arm, role="agent")
        return run_agent(question=question, provider=provider, policy=ARM_POLICY[arm], k=k,
                         tools=tools, system_prompt=notes_v3_prompt(steps), max_steps=steps,
                         logger=tagged, run_meta=run_meta)
    if arm == "memgpt":
        return run_memgpt(question, provider, k, tools, steps, logger, run_meta,
                          threshold=int(params["threshold"]),
                          system_prompt=react_v3_prompt(steps), error_observation=REACT_V3_ERROR)
    if arm == "agentless":
        return run_agentless_v3(question, provider, k, tools, logger, run_meta)
    if arm == "pbc":
        return run_pbc(question, provider, k, tools, steps, logger, run_meta,
                       k_hi=int(params["k_hi"]), e_max=int(params["e_max"]),
                       system_prompt=PBC_V3_PROMPT)
    if arm == "rag1":
        return run_rag1(question, provider, k, tools, logger, run_meta)
    if arm == "closed_book":
        return run_closed_book(question, provider, k, logger, run_meta)
    raise ValueError(f"unknown arm: {arm!r}")
