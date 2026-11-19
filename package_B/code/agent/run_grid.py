"""Run the experiment grid (python -m agent.run_grid), PREREG section 3.

Models are processed in config order (local first, then cloud priority); all
questions of a model run before the next model, questions in sampled order, the
model's cells per question in a fixed seeded shuffle. Each cloud model is
budgeted as one block before its first run (n_questions x n_cells estimate at
the full uncached input rate). Resumable: run_ids already "ok" in runs.jsonl
are skipped. Cloud cells without a key, with an unresolvable model, refused by
the block budget or cut off when actual spend reaches MAX_USD are recorded
once per cell in not_run.jsonl, never silently skipped.

A run's request records are buffered and appended to requests.jsonl only when
the run finishes with status ok; records of failed attempts go to
requests_failed.jsonl. Every record carries its 1-based attempt number.
Errored runs are retried once at the end (PREREG section 9); --smoke runs are
written with control "smoke".
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import random
import secrets
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

import tokenledger

from agent.agent import run_agent
from agent.arms import arm_params, cells_from_config, run_arm
from agent.budget import (BudgetExceeded, charged_usd, check_budget, collect_prior_spend,
                          ensure_price_verified, estimate_usd)
from agent.data import load_dataset
from agent.ledger import RequestLogger, to_tokenledger_rows
from agent.metrics import exact_match, f1_score
from agent.providers import (CLOUD_PROVIDERS, ProviderInsufficientQuota,
                             ProviderQuotaExhausted, ProviderUnavailable, make_provider)
from agent.prompts import SYSTEM_PROMPT
from agent.retrieval import Retriever

EXIT_HALTED = 3  # insufficient_quota stopped the whole run (PREREG_v2 Amendment 2, B7)
ROOT = Path(__file__).resolve().parent.parent


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: str, records: list) -> None:
    """Rewrite a JSONL file atomically (temp file + replace), so an interrupted rewrite never
    truncates records that were already on disk."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _append_jsonl(path: str, rec: dict) -> None:
    """Append one record and flush it to disk at once (PREREG_v3 Amendment 4: a run that has
    finished is on disk even if the job is killed before the end of the grid)."""
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _dedupe_runs(records: list) -> list:
    """The latest record per run_id (an appended runs.jsonl may hold an error record and, after
    the retry, the ok record of the same run), in first-seen order."""
    last: dict = {}
    for rec in records:
        last[rec["run_id"]] = rec
    return list(last.values())


def _write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


def _csv(value) -> list:
    if value is None:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


class SaltedProvider:
    """PREREG_v3 H1 (cache isolation): every request of a cell starts with the system message
    "session <salt>", so no request can share a cacheable prefix with another cell's requests.
    Everything else (name, model, guard, ...) is the wrapped provider's. With a halt file
    (Amendment 1), no request is sent once the file exists: a sibling shard (or an earlier
    dispatch) stopped on insufficient_quota or a billing error, and only the author may clear it."""

    def __init__(self, inner, salt, halt_file=None):
        self._inner, self.salt, self.halt_file = inner, salt, halt_file

    def chat(self, messages, **opts):
        if self.halt_file and os.path.exists(self.halt_file):
            raise ProviderInsufficientQuota(
                f"halt file {self.halt_file} exists (a billing stop halted this chain); no "
                "request is sent until the author confirms and removes it")
        salted = [{"role": "system", "content": f"session {self.salt}"}, *messages]
        return self._inner.chat(salted, **opts)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def cell_salt(nonce: str, arm: str, k: int, repeat: int | None = None,
              attempt: int = 1) -> str:
    """16 hex characters, constant within one (cell, repeat) of one grid invocation and
    different between cells and between repeats; a new invocation (new nonce) never reuses an
    earlier invocation's salts, and a retry (attempt > 1) never reuses its failed attempt's."""
    key = f"{nonce}|{arm}|{int(k)}" + ("" if repeat is None else f"|{int(repeat)}")
    if int(attempt) > 1:
        key += f"|a{int(attempt)}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def cell_order(cells, cell_seed, question_id, v3, repeats=1):
    """Order of the runs for one question, as (arm, k, repeat). v1/v2: one fixed seeded cell
    order for every question (as registered), repeat None. v3 (H2): every (cell, repeat) pair,
    re-shuffled per question, seeded by (cell_seed, question id)."""
    if v3:
        order = [(p, k, r) for r in range(1, int(repeats) + 1) for p, k in cells]
        random.Random(f"{cell_seed}:{question_id}").shuffle(order)
        return order
    order = list(cells)
    random.Random(cell_seed).shuffle(order)
    return [(p, k, None) for p, k in order]


def remaining_runs(question_ids, cells, repeats, done_ids, run_id_of) -> int:
    """v3 block estimate: the (cell, question, repeat) runs not already ok (resumed runs and
    finished repeats cost nothing more)."""
    return sum(1 for qid in question_ids for p, k in cells for r in range(1, int(repeats) + 1)
               if run_id_of(p, k, qid, r) not in done_ids)


class _BufferLogger:
    """Accumulates one run attempt's request records in memory; the grid runner
    flushes them to requests.jsonl (run finished ok) or requests_failed.jsonl
    (attempt raised), stamped with the 1-based attempt number."""

    def __init__(self):
        self.records = []

    def log(self, record):
        self.records.append(record)

    def close(self):
        pass


def _dump_records(records, logger, attempt_no):
    for rec in records:
        rec["attempt"] = attempt_no
        logger.log(rec)


def _git(args) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True,
                           text=True, timeout=30)
        return r.stdout.strip() or None
    except Exception:  # noqa: BLE001 - provenance must never break the grid
        return None


def _prereg_sha() -> str | None:
    out = _git(["log", "--diff-filter=A", "--format=%H", "--", "experiments/PREREG.md"])
    lines = [l for l in (out or "").splitlines() if l.strip()]
    return lines[-1] if lines else None  # git log is newest-first; last = first commit


def _github_run_url() -> str | None:
    rid = os.environ.get("GITHUB_RUN_ID")
    if not rid:
        return None
    repo = os.environ.get("GITHUB_REPOSITORY")
    return f"https://github.com/{repo}/actions/runs/{rid}" if repo else None


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="agent.run_grid")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--models", default=None, help="comma-separated model filter")
    p.add_argument("--policies", default=None, help="comma-separated policy (or arm) filter")
    p.add_argument("--ks", default=None, help="comma-separated k filter")
    p.add_argument("--questions", type=int, default=None, help="first N of the sample")
    p.add_argument("--split", choices=["dev", "test", "smoke"], default=None,
                   help="h2h configs (with an `arms` list) only: dev or test split (PREREG_v2), "
                        "or a PREREG_v3 config's dataset.blocks split (dev, test, smoke)")
    p.add_argument("--repeats", type=int, default=None,
                   help="v3 configs: independent repeats of every (cell, question); default "
                        "config agent.repeats[split], else 1")
    p.add_argument("--shard", default=None, metavar="I/N",
                   help="run only questions whose index in the split %% N == I (two shards "
                        "run as two processes = concurrency 2, each with half the budget)")
    p.add_argument("--halt-file", default=None,
                   help="v3: written when insufficient_quota or a billing stop halts the run; "
                        "while it exists no request is sent (shared by concurrent shards)")
    p.add_argument("--salt-nonce", default=None,
                   help="v3 configs: fixed nonce for the per-cell cache salts (tests and "
                        "replication); default: a fresh random nonce per invocation")
    p.add_argument("--smoke", action="store_true", help="use the smoke questions instead")
    p.add_argument("--control", default=None, choices=["broken_reset"])
    p.add_argument("--max-usd", type=float, default=None)
    p.add_argument("--prior-spend-usd", type=float, default=None)
    p.add_argument("--spend-from", action="append", default=[], metavar="DIR",
                   help="repeatable: prior spend = usd of every requests*.jsonl and est_usd of "
                        "every embedding_log*.json found recursively under DIR")
    p.add_argument("--est-from", action="append", default=[], metavar="DIR",
                   help="repeatable: use mean total_input/total_output of each model's ok runs "
                        "in DIR/runs.jsonl as the block budget estimate")
    return p.parse_args(argv)


def make_run_id(control, provider, model, policy, k, question_id, repeat=None) -> str:
    prefix = f"{control}|" if control else ""
    suffix = "" if repeat is None else f"|rep{int(repeat)}"
    return f"{prefix}{provider}:{model}|{policy}|k={k}|{question_id}{suffix}"


def _run_record(run_id, provider_name, model, policy, k, control, q, res,
                usd, status="ok", reason=None, num_ctx=None):
    steps = res.steps if res else []
    rec = {
        "run_id": run_id,
        "provider": provider_name,
        "model": model,
        "policy": policy,
        "k": k,
        "control": control,
        "question_id": q.id,
        "position": q.position,
        "split": q.split,
        "status": status,
        "turns": res.turns if res else 0,
        "total_input": res.total_input_tokens if res else 0,
        "total_cached": res.total_cached_tokens if res else None,
        "total_output": res.total_output_tokens if res else 0,
        "P": steps[0]["input_tokens"] if steps else 0,
        "latency_s": round(res.total_latency_s, 3) if res else 0.0,
        "finished": res.finished if res else False,
        "answer": res.answer if res else "",
        "gold": q.answer,
        "em": None,
        "f1": None,
        "parse_errors": res.parse_errors if res else 0,
        "usd": round(usd, 6) if usd is not None else None,
        "truncated": bool(num_ctx and any(s["input_tokens"] >= num_ctx - 64 for s in steps)),
        "reason": reason,
        "ts": _utcnow(),
    }
    if res is not None and status == "ok":
        rec["em"] = exact_match(res.answer, q.answer)
        rec["f1"] = round(f1_score(res.answer, q.answer), 6)
    return rec


def _sum_or_none(values):
    return None if (not values or None in values) else sum(values)


def _h2h_fields(arm, k, split, res) -> dict:
    """PREREG_v2 per-run fields: arm/split/k, grounded, requests incl. helpers, USD (usd_list:
    list price; with OpenAI the billed amount, Amendment 2 B2), and the descriptive OpenAI
    reasoning and cache-write token totals (None when the provider does not report them)."""
    steps = res.steps if res else []
    usds = [s.get("usd") for s in steps]
    return {"arm": arm, "split": split, "k": k,
            "grounded": res.grounded if res else None,
            "n_requests": len(usds),
            "usd_list": None if (not usds or None in usds) else round(sum(usds), 6),
            "total_reasoning": _sum_or_none([s.get("reasoning_tokens") for s in steps]),
            "total_cache_write": _sum_or_none([s.get("cache_write_tokens") for s in steps])}


def _load_estimates(dirs) -> dict:
    """{model_id: {source, est_input, est_output, n_runs}} from DIR/runs.jsonl (status ok)."""
    sums: dict = {}
    for d in dirs:
        for r in _read_jsonl(os.path.join(d, "runs.jsonl")):
            if r.get("status") != "ok":
                continue
            e = sums.setdefault(r["model"], {"sources": [], "in": 0.0, "out": 0.0, "n": 0})
            if d not in e["sources"]:
                e["sources"].append(d)
            e["in"] += r.get("total_input") or 0
            e["out"] += r.get("total_output") or 0
            e["n"] += 1
    return {m: {"source": ",".join(e["sources"]), "est_input": e["in"] / e["n"],
                "est_output": e["out"] / e["n"], "n_runs": e["n"]}
            for m, e in sums.items()}


def main(argv=None) -> int:
    args = parse_args(argv)
    started = _utcnow()
    with open(args.config, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    os.makedirs(args.out, exist_ok=True)

    h2h = "arms" in config  # PREREG_v2 head-to-head: cells are (arm, k) pairs
    v3 = config.get("harness") == "v3"  # PREREG_v3: blocks, cache salts, per-question order
    if v3 and not args.split:
        raise SystemExit("a v3 config needs --split (a dataset.blocks split)")
    if h2h and not (args.split or args.smoke):
        raise SystemExit("an h2h config (arms list) needs --split dev|test")
    if args.split and not h2h:
        raise SystemExit("--split is only for h2h configs (arms list)")
    if args.split == "smoke" and not v3:
        raise SystemExit("--split smoke is only for v3 configs (v1/v2 use --smoke)")
    ds = load_dataset(config["dataset"], blocks_split=args.split if v3 else None)

    def _tools(r):
        return {"search": r.search_observation, "read": r.read_observation,
                "search_chunks": r.search_chunks, "read_chunks": r.read_chunks}

    tools_by_q: dict = {}  # v3: each question searches its own block's KB
    on_dev = args.split == "dev"
    if v3:
        tools = None
        for qs, kb in ds.blocks:
            block_tools = _tools(Retriever(kb))
            tools_by_q.update({q.id: block_tools for q in qs})
    else:
        if on_dev and ds.dev_kb is None:
            raise SystemExit("--split dev needs dataset.n_dev > 0 in the config")
        tools = _tools(Retriever(ds.dev_kb if on_dev else ds.kb))
    agent_cfg = config.get("agent") or {}
    max_steps = int(agent_cfg.get("max_steps", 10))
    cell_seed = int(agent_cfg.get("cell_order_seed", 20261116))

    control = args.control or ("smoke" if (args.smoke or args.split == "smoke") else None)
    if h2h:
        cells = cells_from_config(config)
        keep_p, keep_k = set(_csv(args.policies)), {int(x) for x in _csv(args.ks)}
        cells = [(p, k) for p, k in cells
                 if (not keep_p or p in keep_p) and (not keep_k or k in keep_k)]
    else:
        policies = _csv(args.policies) or list(config["policies"])
        ks = [int(x) for x in (_csv(args.ks) or config["ks"])]
        if args.control:
            policies = ["reset_summary_broken"]
        cells = [(p, k) for p in policies for k in ks]
    split_name = args.split or ("smoke" if args.smoke else None)
    model_specs = list(config["models"])
    if args.models:
        keep = set(_csv(args.models))
        model_specs = [s for s in model_specs if s.get("model") in keep]
    n_cells_per_model = len(cells)

    if v3:
        questions = [q for qs, _ in ds.blocks for q in qs]
    else:
        questions = ds.smoke if args.smoke else (ds.dev if on_dev else ds.sample)
    if args.questions:  # a total over the split, applied before sharding (Amendment 1)
        questions = questions[: args.questions]
    shard = None
    if args.shard:
        si, sn = (int(x) for x in args.shard.split("/"))
        if not 0 <= si < sn:
            raise SystemExit(f"--shard {args.shard}: need 0 <= I < N")
        shard = [si, sn]
        questions = [q for i, q in enumerate(questions) if i % sn == si]
    q_block = {q.id: q.block for q in questions}
    repeats = 1
    if v3:
        cfg_rep = agent_cfg.get("repeats", 1)
        repeats = int(args.repeats or (cfg_rep.get(args.split, 1) if isinstance(cfg_rep, dict)
                                       else cfg_rep))
    salt_nonce = (args.salt_nonce or secrets.token_hex(16)) if v3 else None
    salts = ({(p, k, r): cell_salt(salt_nonce, p, k, r) for p, k in cells
              for r in range(1, repeats + 1)} if v3 else {})

    budget_cfg = config.get("budget") or {}
    est_in = int(budget_cfg.get("est_input_tokens_per_run", 100000))
    est_out = int(budget_cfg.get("est_output_tokens_per_run", 5000))
    max_usd = float(args.max_usd or os.environ.get("MAX_USD") or budget_cfg.get("max_usd", 0))
    if v3 and budget_cfg.get("max_usd") is not None:  # the registered v3 spend stop always binds
        max_usd = min(max_usd, float(budget_cfg["max_usd"]))
    # Prior spend (Amendment A4): every request record already on disk under the --spend-from
    # dirs and the current --out dir (a resumed job), embedding logs, and the manual flag.
    prior_info = collect_prior_spend(list(args.spend_from) + [args.out],
                                     args.prior_spend_usd or 0.0)
    prior = prior_info["total"]
    spent = prior
    est_from = _load_estimates(args.est_from)
    budget_estimates: dict = {}

    requests_path = os.path.join(args.out, "requests.jsonl")
    failed_path = os.path.join(args.out, "requests_failed.jsonl")
    runs_path = os.path.join(args.out, "runs.jsonl")
    notrun_path = os.path.join(args.out, "not_run.jsonl")
    history_path = os.path.join(args.out, "not_run_history.jsonl")
    # not_run.jsonl describes THIS invocation's final state: older lines move to history and
    # nothing is pre-dead, so budget/quota/price are re-evaluated on every start.
    old_not_run = _read_jsonl(notrun_path)
    if old_not_run:
        with open(history_path, "a", encoding="utf-8") as fh:
            for rec in old_not_run:
                fh.write(json.dumps(rec) + "\n")
    logger = RequestLogger(requests_path)
    failed_logger = RequestLogger(failed_path)
    runs = _dedupe_runs(_read_jsonl(runs_path))
    ok_ids = {r["run_id"] for r in runs if r.get("status") == "ok"}
    not_run: list = []
    dead: set = set()
    errored_runs: dict = {}  # (provider, model, policy, k, question_id) -> (spec, question)
    err_stopped: set = set()  # cells stopped by 3 consecutive errors (still retried at the end)
    consec_errors: dict = {}  # cell key -> consecutive errored runs
    block_providers: dict = {}  # (provider, model) -> instance, resolved once per cloud block

    turn = 0
    attempts: dict = {}  # run_id -> attempts so far; records carry the 1-based number
    halted: list = []  # insufficient_quota reason: the whole run stops (Amendment 2, B7)

    def next_attempt(run_id):
        attempts[run_id] = attempts.get(run_id, 0) + 1
        return attempts[run_id]

    def salt_for(policy, k, rep, attempt_no):
        if attempt_no <= 1:
            return salts[(policy, k, rep)]
        return cell_salt(salt_nonce, policy, k, rep, attempt_no)

    def attempt(provider, policy, k, q, run_id, turn_no, buf, rep=None, attempt_no=1):
        if v3:
            return run_arm(policy, q.question,
                           SaltedProvider(provider, salt_for(policy, k, rep, attempt_no),
                                          args.halt_file), k,
                           tools_by_q[q.id], max_steps=max_steps, logger=buf,
                           params=arm_params(config, policy), harness="v3",
                           run_meta={"run_id": run_id, "question_id": q.id, "turn": turn_no,
                                     "control": control})
        if h2h:
            return run_arm(policy, q.question, provider, k, tools, max_steps=max_steps,
                           logger=buf, params=arm_params(config, policy),
                           run_meta={"run_id": run_id, "question_id": q.id, "turn": turn_no,
                                     "control": control})
        return run_agent(
            question=q.question, provider=provider, policy=policy, k=k, tools=tools,
            system_prompt=SYSTEM_PROMPT, max_steps=max_steps, logger=buf,
            run_meta={"run_id": run_id, "question_id": q.id, "turn": turn_no,
                      "control": control},
        )

    def put_run(rec, res=None, rep=None):
        nonlocal runs
        if h2h:
            rec.update(_h2h_fields(rec["policy"], rec["k"], split_name, res))
        if v3:
            rec.update(block=q_block.get(rec["question_id"]), repeat=rep,
                       salt=salts.get((rec["policy"], rec["k"], rep)), harness="v3")
        runs = [r for r in runs if r["run_id"] != rec["run_id"]]
        runs.append(rec)
        _append_jsonl(runs_path, rec)  # on disk now, not only at the end (Amendment 4)

    def stop_block(provider_name, model, resolved, reason, with_n_done=False):
        """Record every unfinished cell of this model block as not run and mark it dead."""
        done = {(r["policy"], r["k"], r["question_id"], r.get("repeat")) for r in runs
                if r.get("status") == "ok" and r["provider"] == provider_name
                and r["model"] == resolved}
        reps = list(range(1, repeats + 1)) if v3 else [None]
        for p, kk in cells:
            if sum(1 for qq in questions for rr in reps
                   if (p, kk, qq.id, rr) in done) < len(questions) * len(reps):
                rec = {"provider": provider_name, "model": resolved, "policy": p,
                       "k": kk, "reason": reason(len(done)) if callable(reason) else reason}
                if with_n_done:
                    rec["n_done"] = len(done)
                not_run.append(rec)
        for m in {model, resolved}:
            dead.update((provider_name, m, p, kk) for p, kk in cells)

    def execute(provider, spec, policy, k, q, run_id, retried=False, rep=None):
        """One run attempt. Spend from the buffered records is added on EVERY path.
        Returns "ok" | "error" | "quota" | "unpriced" (+ the exception, if any)."""
        nonlocal turn, spent
        turn += 1
        buf = _BufferLogger()
        attempt_no = next_attempt(run_id)
        res = exc = None
        guard0 = getattr(provider, "guard", None)
        spent0 = guard0.spent if guard0 is not None else None
        try:
            res = attempt(provider, policy, k, q, run_id, turn, buf, rep, attempt_no)
        except Exception as e:  # noqa: BLE001 - quota or persistent provider error
            exc = e
        if guard0 is not None:  # Amendment 4: a charge with no request record gets one
            gap = guard0.spent - spent0 - sum((charged_usd(r) or 0.0) for r in buf.records)
            if gap > 1e-9:
                buf.records.append({
                    "schema": "agent-request/v1", "run_id": run_id, "question_id": q.id,
                    "turn": turn, "step": len(buf.records), "provider": provider.name,
                    "model": provider.model, "policy": policy, "arm": policy, "k": k,
                    "role": "unrecorded_charge", "control": control,
                    "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0,
                    "usd": round(gap, 8), "usd_worst": None,
                    "reason": "spend-guard charge without a request record (an unconfirmed "
                              f"or failed attempt; {type(exc).__name__ if exc else 'no error'})",
                    "ts": _utcnow()})
        if v3:
            for rec in buf.records:
                rec["salt"], rec["repeat"] = salt_for(policy, k, rep, attempt_no), rep
        cloud = provider.name in CLOUD_PROVIDERS
        usd, unpriced = None, False
        if cloud:
            usds = [r.get("usd") for r in buf.records]
            unpriced = any(u is None for u in usds)  # never count None as 0 for cloud
            usd = sum(u or 0.0 for u in usds)
            spent += usd
            guard = getattr(provider, "guard", None)
            if guard is not None:  # the guard also holds worst-case charges (unpriced/unconfirmed)
                spent = max(spent, guard.spent)
        _dump_records(buf.records, logger if exc is None else failed_logger, attempt_no)
        if isinstance(exc, ProviderInsufficientQuota):
            return "insufficient_quota", exc
        if isinstance(exc, ProviderQuotaExhausted):
            return "quota", exc
        if isinstance(exc, BudgetExceeded):  # per-request spend guard refused (B7)
            return "budget", exc
        if exc is not None:
            reason = f"{exc} (retried once)" if retried else str(exc)
            put_run(_run_record(run_id, provider.name, provider.model, policy, k, control,
                                q, None, usd, status="error", reason=reason,
                                num_ctx=spec.get("num_ctx")), None, rep)
            return "error", exc
        put_run(_run_record(run_id, provider.name, provider.model, policy, k, control, q, res,
                            None if unpriced else usd, num_ctx=spec.get("num_ctx")), res, rep)
        ok_ids.add(run_id)
        errored_runs.pop((spec["provider"], spec["model"], policy, k, q.id, rep), None)
        return ("unpriced" if unpriced else "ok"), None

    def after_run(status, exc, provider, spec):
        """Block-level consequences shared by the main loop and the end-of-grid retry.
        Returns True when the whole model block must stop."""
        pn, model = spec["provider"], spec["model"]
        if status == "insufficient_quota":
            halted.append(str(exc))
            if args.halt_file and not os.path.exists(args.halt_file):
                with open(args.halt_file, "w", encoding="utf-8") as fh:
                    fh.write(f"{_utcnow()} {exc}\n")
            stop_block(pn, model, provider.model, f"insufficient_quota: {exc}",
                       with_n_done=True)
            return True
        if status == "quota":
            stop_block(pn, model, provider.model, f"quota: {exc}", with_n_done=True)
            return True
        if status == "budget":
            stop_block(pn, model, provider.model, f"budget: {exc}", with_n_done=True)
            return True
        if status == "unpriced":
            stop_block(pn, model, provider.model,
                       f"price: unpriced model id {provider.model}")
            return True
        if provider.name in CLOUD_PROVIDERS and spent >= max_usd:
            stop_block(pn, model, provider.model,
                       lambda n: f"budget exhausted after {n} runs")
            return True
        return False

    budget_cleared: set = set()  # (provider, model) cloud blocks already checked

    for spec in model_specs:  # config order: local first, then cloud priority
        provider_name, model = spec["provider"], spec["model"]
        cell_keys = [(provider_name, model, p, k) for p, k in cells]
        is_cloud = provider_name in CLOUD_PROVIDERS
        if halted:  # insufficient_quota: no further block may start
            for p, k in cells:
                not_run.append({"provider": provider_name, "model": model, "policy": p,
                                "k": k, "reason": f"insufficient_quota: {halted[0]}"})
            dead.update(cell_keys)
            continue

        if is_cloud and (provider_name, model) not in budget_cleared:
            budget_cleared.add((provider_name, model))

            def refuse(reason):
                for p, k in cells:
                    not_run.append({"provider": provider_name, "model": model,
                                    "policy": p, "k": k, "reason": reason})
                dead.update(cell_keys)

            try:  # resolve ONCE per block (specs like auto-flash resolve to a concrete id)
                block_provider = make_provider(spec)
            except ProviderQuotaExhausted as exc:
                refuse(f"quota: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001 - keyless, network, no such model...
                refuse(f"resolution: {exc}")
                continue
            block_providers[(provider_name, model)] = block_provider
            resolved_id = block_provider.model
            try:
                ensure_price_verified(provider_name, resolved_id)
            except BudgetExceeded as exc:  # unverified or missing price
                refuse(f"price: {exc}")
                continue
            # Block estimate: whole model block (all its cells x all questions),
            # priced at the full uncached input rate (PREREG section 3).
            blk_in, blk_out = est_in, est_out
            if resolved_id in est_from:
                e = est_from[resolved_id]
                blk_in, blk_out = e["est_input"], e["est_output"]
                budget_estimates[resolved_id] = dict(e)
            n_runs = len(questions) * n_cells_per_model
            if v3:  # every (cell, question, repeat) still to run in this invocation
                n_runs = remaining_runs(
                    [qq.id for qq in questions], cells, repeats, ok_ids,
                    lambda pp, kk, qid, rr: make_run_id(control, provider_name, resolved_id,
                                                        pp, kk, qid, rr))
            est = estimate_usd(provider_name, resolved_id, blk_in * n_runs, blk_out * n_runs)
            try:
                check_budget(est, max_usd, spent)
            except BudgetExceeded as exc:
                refuse(f"budget: {exc}")
                continue
            attach = getattr(block_provider, "attach_budget", None)
            if attach is not None:  # per-request guard at the current spend (Amendment 2 B7)
                attach(max_usd, spent)

        exhausted = False
        for qi, q in enumerate(questions):
            if exhausted:
                break
            order = cell_order(cells, cell_seed, q.id, v3, repeats)
            for policy, k, rep in order:
                key = (provider_name, model, policy, k)
                if key in dead or key in err_stopped:
                    continue
                run_id = make_run_id(control, provider_name, model, policy, k, q.id, rep)
                if run_id in ok_ids:
                    continue
                if is_cloud:
                    provider = block_providers[(provider_name, model)]
                else:
                    try:  # fresh instance per run (scripted/local providers carry state)
                        provider = make_provider(spec)
                    except ProviderUnavailable as exc:
                        not_run.append({"provider": key[0], "model": key[1], "policy": policy,
                                        "k": k, "reason": str(exc)})
                        dead.add(key)  # unavailable: one record per cell
                        continue
                    except Exception as exc:  # noqa: BLE001
                        put_run(_run_record(run_id, provider_name, model, policy, k, control,
                                            q, None, None, status="error", reason=str(exc),
                                            num_ctx=spec.get("num_ctx")))
                        errored_runs[(provider_name, model, policy, k, q.id, rep)] = (spec, q)
                        continue
                status, exc = execute(provider, spec, policy, k, q, run_id, rep=rep)
                if status == "error":
                    errored_runs[(provider_name, model, policy, k, q.id, rep)] = (spec, q)
                if after_run(status, exc, provider, spec):
                    exhausted = True
                    break
                if status == "error":
                    consec_errors[key] = consec_errors.get(key, 0) + 1
                    if consec_errors[key] >= 3:  # stop this cell only (every repeat)
                        reps_left = range(1, repeats + 1) if v3 else [None]
                        for qq in questions[qi:]:
                            for rr in reps_left:
                                rid = make_run_id(control, provider_name, model, policy, k,
                                                  qq.id, rr)
                                if rid not in ok_ids and (provider_name, model, policy, k,
                                                          qq.id, rr) not in errored_runs:
                                    not_run.append({"provider": provider_name,
                                                    "model": provider.model, "policy": policy,
                                                    "k": k, "question_id": qq.id,
                                                    "repeat": rr,
                                                    "reason": "errors: 3 consecutive"})
                        err_stopped.add(key)
                else:
                    consec_errors[key] = 0

    # Persistent provider errors are retried once at the end (PREREG section 9): every
    # errored (run, question) gets exactly one retry, under the same dead-cell, budget and
    # price checks as the main loop, on the block's own provider instance.
    for rkey in sorted(errored_runs):
        if halted:
            break
        provider_name, model, policy, k, qid, rep = rkey
        spec, q = errored_runs[rkey]
        key = (provider_name, model, policy, k)
        if key in dead:
            continue
        run_id = make_run_id(control, provider_name, model, policy, k, q.id, rep)
        if provider_name in CLOUD_PROVIDERS:
            provider = block_providers.get((provider_name, model))
            if provider is None:
                continue
            if spent >= max_usd:
                stop_block(provider_name, model, provider.model,
                           lambda n: f"budget exhausted after {n} runs")
                continue
        else:
            try:
                provider = make_provider(spec)
            except Exception:  # noqa: BLE001 - stays an error record
                continue
        status, exc = execute(provider, spec, policy, k, q, run_id, retried=True, rep=rep)
        after_run(status, exc, provider, spec)

    logger.close()
    failed_logger.close()

    _write_jsonl(runs_path, runs)
    _write_jsonl(notrun_path, not_run)

    all_reqs = _read_jsonl(requests_path)
    tl_rows = to_tokenledger_rows(all_reqs)
    tl_path = os.path.join(args.out, "tokenledger.jsonl")
    _write_jsonl(tl_path, tl_rows)
    try:
        metrics = dataclasses.asdict(tokenledger.account(tokenledger.load_requests(tl_path)))
    except Exception:  # noqa: BLE001 - empty grid: report zeros
        metrics = {"total_input": 0, "n_requests": 0, "n_turns": 0}
    _write_json(os.path.join(args.out, "tokenledger_report.json"), metrics)

    provenance = {
        # CI containers have no git: the workflow passes the SHAs it computed outside.
        "git_sha": _git(["rev-parse", "HEAD"]) or os.environ.get("GIT_SHA") or None,
        "prereg_sha": _prereg_sha() or os.environ.get("PREREG_SHA") or None,
        "config_sha256": _sha256_file(args.config),
        "dataset_sha256": ds.dataset_sha256,
        "python": platform.python_version(),
        "tokenledger": tokenledger.__version__,
        "github_run_id": os.environ.get("GITHUB_RUN_ID") or None,
        "github_run_url": _github_run_url(),
        "hostname": socket.gethostname(),
        "started": started,
        "ended": _utcnow(),
        "cloud_spend_total": round(spent, 6),
        "cloud_spend_this_run": round(spent - prior, 6),
        "prior_spend_usd": prior,
        "prior_spend": prior_info,
        "max_usd": max_usd,
        "budget_estimates": budget_estimates,
        "halted": halted[0] if halted else None,
    }
    if v3:
        v3_sha = _git(["log", "--diff-filter=A", "--format=%H", "--", "experiments/PREREG_v3.md"])
        provenance.update(
            harness="v3", split=args.split,
            prereg_v3_sha=((v3_sha or "").splitlines() or [None])[-1]
            or os.environ.get("PREREG_V3_SHA") or None,
            salt_nonce=salt_nonce,
            salts={f"{p}|k={k}|rep{r}": v for (p, k, r), v in salts.items()},
            repeats=repeats, shard=shard,
            blocks=[[qs[0].position, qs[-1].position] for qs, _ in ds.blocks])
    _write_json(os.path.join(args.out, "provenance.json"), provenance)

    n_ok = sum(1 for r in runs if r.get("status") == "ok")
    print(f"runs ok={n_ok} error={sum(1 for r in runs if r.get('status') == 'error')} "
          f"not_run={len(not_run)} cloud_spend={spent - prior:.4f} USD -> {args.out}")
    if halted:
        print(f"HALTED: insufficient_quota - the whole run stopped; no request will be retried. "
              f"{halted[0]}", file=sys.stderr)
        return EXIT_HALTED
    return 0


if __name__ == "__main__":
    sys.exit(main())
