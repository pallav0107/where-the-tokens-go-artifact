#!/usr/bin/env python3
"""Chained sessions for the pause experiment (PREREG_pause.md, Amendment 2).

A session is one container and one continuing Claude Code conversation that works through 4
SWE-bench Verified tasks of the same repository AND version, so one prebuilt environment serves all
four (the repo is reset to each task's base commit, which is what the instance images do).

Rule (seed 2027):
  * groups = (repo, version) over the 500 Verified instances, excluding repositories whose compiled
    extensions would need a rebuild after a checkout (EXCLUDED_REPOS) and the earlier single-task
    pilot instance (EXCLUDED_IDS);
  * eligible: at least one hard task ("1-4 hours" or ">4 hours") and at least 4 tasks that are hard
    or "15 min - 1 hour";
  * groups sorted by id, shuffled with random.Random(2027), then stably sorted by hard-task count,
    descending; the first 10 are the sessions and the 11th is the chained pilot (excluded);
  * per session: its hard tasks in seeded order (at most 4), filled with "15 min - 1 hour" tasks in
    seeded order up to 4; the task order is that list;
  * image: the instance image of the session's most recent task (latest created_at); every task's base
    commit must be reachable in it (checked when the image is built);
  * session order (dispatch, and C on the first N_C) = the selected sessions shuffled with
    random.Random(2027 + 1).

    python3 select_sessions.py --rows data/rows_*.json --out sessions.json --prompts data/session_prompts.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

SEED = 2027
N_SESSIONS, TASKS_PER_SESSION, N_C = 10, 4, 5
HARD = ("1-4 hours", ">4 hours")
FILL = "15 min - 1 hour"
EXCLUDED_REPOS = ("astropy/astropy", "scikit-learn/scikit-learn", "matplotlib/matplotlib")
EXCLUDED_IDS = ("django__django-16631",)   # the single-task pilot of 19 Nov 2026 (excluded)


def image_name(instance_id: str) -> str:
    return f"swebench/sweb.eval.x86_64.{instance_id.replace('__', '_1776_').lower()}:latest"


def form_sessions(rows: list[dict], seed: int = SEED) -> tuple[list[dict], dict | None]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        if r["repo"] in EXCLUDED_REPOS or r["instance_id"] in EXCLUDED_IDS:
            continue
        groups.setdefault((r["repo"], r["version"]), []).append(r)
    elig = []
    for key, rs in sorted(groups.items()):
        hard = sorted((r for r in rs if r["difficulty"] in HARD), key=lambda r: r["instance_id"])
        fill = sorted((r for r in rs if r["difficulty"] == FILL), key=lambda r: r["instance_id"])
        if hard and len(hard) + len(fill) >= TASKS_PER_SESSION:
            elig.append((key, hard, fill))
    rng = random.Random(seed)
    rng.shuffle(elig)
    elig.sort(key=lambda g: -len(g[1]))          # stable: ties keep the seeded order
    out = []
    for key, hard, fill in elig[: N_SESSIONS + 1]:
        r2 = random.Random(f"{seed}:{key[0]}:{key[1]}")
        hard, fill = hard[:], fill[:]
        r2.shuffle(hard)
        r2.shuffle(fill)
        tasks = (hard[:TASKS_PER_SESSION] + fill)[:TASKS_PER_SESSION]
        # the instance images keep only the history up to their own base commit, so the session runs on the
        # image of its most recent task (latest created_at), whose history holds the other base commits
        newest = max(tasks, key=lambda t: (t["created_at"], t["instance_id"]))
        out.append({"session_id": f"{key[0].split('/')[1]}-{key[1]}", "repo": key[0], "version": key[1],
                    "image": image_name(newest["instance_id"]), "image_task": newest["instance_id"],
                    "tasks": [{"instance_id": t["instance_id"], "difficulty": t["difficulty"],
                               "base_commit": t["base_commit"],
                               "problem_sha256": hashlib.sha256(t["problem_statement"].encode("utf-8")).hexdigest()}
                              for t in tasks]})
    sessions, pilot = out[:N_SESSIONS], (out[N_SESSIONS] if len(out) > N_SESSIONS else None)
    random.Random(seed + 1).shuffle(sessions)
    for i, s in enumerate(sessions):
        s["order"] = i + 1
        s["condition_c"] = i < N_C
    return sessions, pilot


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
    if len(rows) != 500:
        raise SystemExit(f"BLOCKED: expected the 500 Verified instances, got {len(rows)}")
    sessions, pilot = form_sessions(rows)
    by_id = {r["instance_id"]: r for r in rows}
    n_hard = sum(t["difficulty"] in HARD for s in sessions for t in s["tasks"])
    args.out.write_text(json.dumps({
        "dataset": "princeton-nlp/SWE-bench_Verified", "split": "test", "seed": SEED, "sources": sources,
        "tasks_per_session": TASKS_PER_SESSION, "n_sessions": len(sessions), "n_c": N_C,
        "excluded_repos": EXCLUDED_REPOS, "excluded_ids": EXCLUDED_IDS,
        "hard_tasks": n_hard, "fill_tasks": len(sessions) * TASKS_PER_SESSION - n_hard,
        "pilot": pilot, "sessions": sessions}, indent=1) + "\n", encoding="utf-8")
    ids = [t["instance_id"] for s in sessions + ([pilot] if pilot else []) for t in s["tasks"]]
    args.prompts.write_text(json.dumps({i: by_id[i]["problem_statement"] for i in ids}, indent=1), encoding="utf-8")
    print(f"{len(sessions)} sessions, {n_hard} hard + {len(sessions) * TASKS_PER_SESSION - n_hard} fill tasks; "
          f"pilot {pilot['session_id'] if pilot else None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
