"""One OpenAI smoke request before any data (PREREG_v2 Amendment 2, B6).

    python -m agent.smoke_openai            # from the repository root, on a local workstation

Synthetic text only: the v1 system prompt and a carry_all-shaped history built from the
invented fixture agent/tests/fixtures/hotpot_tiny.jsonl (no HotpotQA question of any split),
long enough (> 1,024 tokens) to show the provider's cache-write behaviour. Same provider
spec as experiments/configs/h2h.yaml, so it settles the B3 parameter rule for the runs.

The key is read at run time from the git-ignored .env.local (OPENAI_API_KEY) and is never
printed or written. The request is admitted by the spend guard at the prior spend of the
committed off-chain ledger plus --prior-spend-usd; its cost is appended to that ledger
(labelled "smoke", no reply text), so every later CI job counts it. The full record,
including the reply, goes to the git-ignored evidence/raw/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from agent.agent import make_record
from agent.budget import collect_prior_spend
from agent.data import load_dataset
from agent.policies import build_messages
from agent.prompts import SYSTEM_PROMPT
from agent.providers import make_provider
from agent.retrieval import Retriever

ROOT = Path(__file__).resolve().parent.parent
LEDGER_DIR = ROOT / "experiments" / "spend"
LEDGER = LEDGER_DIR / "spend_ledger.jsonl"
FIXTURE = ROOT / "agent" / "tests" / "fixtures" / "hotpot_tiny.jsonl"
H2H = ROOT / "experiments" / "configs" / "h2h.yaml"
RAW_DIR = ROOT / "evidence" / "raw"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_key(path: Path) -> bool:
    """Put OPENAI_API_KEY from a KEY=VALUE file into this process's environment only.
    Returns whether a key is available; never prints it."""
    if os.environ.get("OPENAI_API_KEY"):
        return True
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == "OPENAI_API_KEY":
            value = value.strip().strip('"').strip("'")
            if value:
                os.environ["OPENAI_API_KEY"] = value
                return True
    return False


def synthetic_messages() -> list[dict]:
    """carry_all-shaped request over the invented fixture: every fixture question searched
    (k=8) and every fixture paragraph read, carried as history."""
    ds = load_dataset({"parquet_url": None, "fixture": str(FIXTURE), "sample_seed": 20261115,
                       "n_questions": 4, "n_smoke": 0})
    r = Retriever(ds.kb)
    history = []
    for q in ds.sample:
        reply = json.dumps({"thought": "search", "notes": "searching the corpus",
                            "action": "search", "args": {"query": q.question}})
        history.append((reply, r.search_observation(q.question, 8)))
    for doc in ds.kb.docs:
        reply = json.dumps({"thought": "read", "notes": f"reading {doc['title']}",
                            "action": "read", "args": {"doc_id": doc["doc_id"]}})
        history.append((reply, r.read_observation(doc["doc_id"])))
    return build_messages("carry_all", SYSTEM_PROMPT, ds.sample[0].question, history)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="agent.smoke_openai")
    ap.add_argument("--max-usd", type=float, default=5.0)
    ap.add_argument("--prior-spend-usd", type=float, default=0.0)
    ap.add_argument("--env-file", default=str(ROOT / ".env.local"))
    ap.add_argument("--again", action="store_true",
                    help="allow a further smoke request when the ledger already has one")
    a = ap.parse_args(argv)

    existing = [json.loads(x) for x in LEDGER.read_text(encoding="utf-8").splitlines()
                if x.strip()] if LEDGER.exists() else []
    if existing and not a.again:
        print(f"REFUSING: {LEDGER} already records {len(existing)} smoke request(s); "
              "pass --again to make another", file=sys.stderr)
        return 2
    if not load_key(Path(a.env_file)):
        print("NOT RUN: OPENAI_API_KEY not found in the environment or the env file",
              file=sys.stderr)
        return 2

    with open(H2H, encoding="utf-8") as fh:
        spec = dict(yaml.safe_load(fh)["models"][0])
    prior = collect_prior_spend([LEDGER_DIR], a.prior_spend_usd)["total"]
    provider = make_provider(spec)
    provider.attach_budget(a.max_usd, prior)
    messages = synthetic_messages()

    ts_sent = _utcnow()
    resp = provider.chat(messages)
    ts_received = _utcnow()
    rec = make_record(provider, resp, messages, 0, "carry_all", 8,
                      {"run_id": "smoke|openai-amendment2", "question_id": "fixture:q_mock",
                       "turn": 0, "control": "smoke"}, None, False, 0)
    reply_json = None
    try:
        reply_json = json.loads(resp.text)
    except ValueError:
        pass
    usage = rec["usage_metadata"] or {}
    row = {
        "label": "smoke", "ts_sent": ts_sent, "ts_received": ts_received,
        "provider": rec["provider"], "model": rec["model"],
        "response_model": rec["response_model"], "service_tier": rec["service_tier"],
        "system_fingerprint": rec["system_fingerprint"], "finish_reason": rec["finish_reason"],
        "params_sent": rec["thinking_config"],
        "input_tokens": rec["input_tokens"], "cached_input_tokens": rec["cached_input_tokens"],
        "cache_write_tokens": rec["cache_write_tokens"], "output_tokens": rec["output_tokens"],
        "reasoning_tokens": rec["reasoning_tokens"], "usage": usage,
        "input_upper_bound": rec["input_upper_bound"], "n_messages": rec["n_messages"],
        "latency_s": rec["latency_s"], "reply_chars": len(resp.text),
        "reply_is_json_action": isinstance(reply_json, dict) and "action" in reply_json,
        # fail closed: an unpriced response is ledgered at what the guard charged for it
        "usd": rec["usd"] if rec["usd"] is not None else round(rec["cum_usd"] - prior, 6),
        "usd_basis": "actual" if rec["usd"] is not None else "worst_case_unpriced",
        "usd_worst": rec["usd_worst"], "unconfirmed_attempts": rec["unconfirmed_attempts"],
        "cum_usd": rec["cum_usd"], "max_usd": a.max_usd,
    }
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row) + "\n")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW_DIR / "smoke_openai.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({**rec, "ts_sent": ts_sent, "ts_received": ts_received}) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k != "usage"}, indent=2))
    print("usage:", json.dumps(usage))
    return 0


if __name__ == "__main__":
    sys.exit(main())
