#!/usr/bin/env python3
"""Select the pause experiment's tasks from SWE-bench Verified (PREREG_pause.md, Tasks).

Pool: every instance annotated "1-4 hours" or ">4 hours" (the two hardest difficulty bins).
The pool is sorted by instance_id, shuffled with random.Random(2027), and the first 24 are the
candidates; the runner uses them in that order and takes the first 20 whose image builds (the
rest are spares). The 25th is the pilot task (excluded from the analysis). Input: the dataset rows saved from the Hugging Face datasets server
(princeton-nlp/SWE-bench_Verified, split test), passed as files; their sha256 is recorded.

    python3 select_tasks.py --rows data/rows_*.json --out tasks.json --prompts data/prompts.json

tasks.json (committed) holds ids, difficulty, repo, image name and the sha256 of each problem
statement; prompts.json (work dir only) holds the problem statements the runner sends.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

SEED = 2027
N_CANDIDATES = 24
N_USE = 20
HARD = ("1-4 hours", ">4 hours")
DATASET = "princeton-nlp/SWE-bench_Verified"


def image_name(instance_id: str) -> str:
    """Official prebuilt SWE-bench instance image (swebench harness naming: '__' -> '_1776_')."""
    return f"swebench/sweb.eval.x86_64.{instance_id.replace('__', '_1776_').lower()}:latest"


def select(rows: list[dict], seed: int = SEED, n: int = N_CANDIDATES) -> tuple[list[dict], dict]:
    """The first n of the seeded shuffle of the hard pool, and the next one as the pilot."""
    pool = sorted((r for r in rows if r.get("difficulty") in HARD), key=lambda r: r["instance_id"])
    random.Random(seed).shuffle(pool)
    return pool[:n], pool[n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    args = ap.parse_args()
    rows, sources = [], []
    for p in sorted(args.rows):
        raw = p.read_bytes()
        sources.append({"file": p.name, "sha256": hashlib.sha256(raw).hexdigest()})
        rows += [r["row"] for r in json.loads(raw)["rows"]]
    if len(rows) != 500 or len({r["instance_id"] for r in rows}) != 500:
        raise SystemExit(f"BLOCKED: expected the 500 Verified instances, got {len(rows)}")
    pool_n = sum(r.get("difficulty") in HARD for r in rows)
    chosen, pilot = select(rows)

    def entry(r: dict) -> dict:
        return {"instance_id": r["instance_id"], "repo": r["repo"], "difficulty": r["difficulty"],
                "image": image_name(r["instance_id"]),
                "problem_sha256": hashlib.sha256(r["problem_statement"].encode("utf-8")).hexdigest()}
    tasks = [{"rank": i + 1, **entry(r), "use": i < N_USE} for i, r in enumerate(chosen)]
    args.out.write_text(json.dumps({"dataset": DATASET, "split": "test", "seed": SEED, "difficulty_bins": HARD,
                                    "pool_size": pool_n, "n_candidates": N_CANDIDATES, "n_use": N_USE,
                                    "sources": sources, "pilot": entry(pilot), "tasks": tasks}, indent=1) + "\n", encoding="utf-8")
    args.prompts.write_text(json.dumps({r["instance_id"]: r["problem_statement"] for r in chosen + [pilot]}, indent=1),
                            encoding="utf-8")
    print(f"pool {pool_n}; candidates {len(tasks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
