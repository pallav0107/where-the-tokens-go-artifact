#!/usr/bin/env python3
"""Paper figures derived from the pre-registered pause experiment (experiments/PREREG_pause.md).

Combines committed files only -- no new measurement:

  * evidence/pause_experiment.json   aggregates written by tools/pause_experiment.py (CIs, rule
                                     confusion counts, direct expiry check, exclusions);
  * experiments/pause/sessions.json  the registered sessions (tasks per session, difficulty split);
  * experiments/PREREG_pause.md      the registered client version and model.

It regroups and restates; every value is checked against a second field where one exists (task
counts against the declared split, sessions against the paired count), or it is BLOCKED.
Writes evidence/pause_figures.json (aggregates only).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_lib import EVIDENCE, REPO, figure, sha256_bytes, utc_now, write_json  # noqa: E402

GENERATOR = "tools/pause_figures.py"
SESSIONS = REPO / "experiments" / "pause" / "sessions.json"
PREREG = REPO / "experiments" / "PREREG_pause.md"
HARD = {"1-4 hours", ">4 hours"}
MEDIUM = {"15 min - 1 hour"}


def blocked(msg: str) -> None:
    sys.exit(f"BLOCKED: {msg}")


def _load(path: Path) -> tuple[Any, bytes]:
    raw = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return raw.decode("utf-8"), raw


def derive(a: dict[str, Any], sessions: dict[str, Any], prereg: str) -> dict[str, Any]:
    tasks = [t for s in sessions["sessions"] for t in s["tasks"]]
    if len(sessions["sessions"]) != sessions["n_sessions"]:
        blocked("sessions.json lists a different number of sessions than n_sessions")
    if len(tasks) != sessions["n_sessions"] * sessions["tasks_per_session"]:
        blocked("task count differs from sessions x tasks per session")
    hard = sum(t["difficulty"] in HARD for t in tasks)
    medium = sum(t["difficulty"] in MEDIUM for t in tasks)
    if hard != sessions["hard_tasks"] or medium != sessions["fill_tasks"] or hard + medium != len(tasks):
        blocked(f"difficulty split {hard}/{medium} differs from the declared {sessions['hard_tasks']}/"
                f"{sessions['fill_tasks']}")
    if a["paired_tasks"] > sessions["n_sessions"]:
        blocked("more paired sessions than registered sessions")
    if a["paired_tasks"] != sessions["n_sessions"] and not a["excluded"].get("unpaired_tasks"):
        blocked("paired count differs from the registered sessions with no exclusion listed")
    flat = " ".join(prereg.split())
    ver = re.search(r"Claude Code (\d+\.\d+\.\d+)", flat)
    model = re.search(r"model `([^`]+)`", flat)
    if not ver or not model:
        blocked("client version or model not found in the pre-registration")
    b, c = a["direct_expiry"], a["condition_c"]["direct_expiry"]
    shared = set(b.get("exploratory_read_after_values") or [])
    if len(shared) != 1:
        blocked(f"read-back after the pauses is not one constant value: {sorted(shared)}")
    ci = lambda k: (a["composition"][k]["read_share_ci"]["lo"] * 100,  # noqa: E731
                    a["composition"][k]["read_share_ci"]["hi"] * 100)
    return {
        "sessions_paired": a["paired_tasks"], "sessions_c": a["condition_c"]["runs"],
        "sessions_valid": 2 * a["paired_tasks"] + a["condition_c"]["runs"],
        "sessions_failed": a["excluded"]["failed_runs"], "pilot_runs": a["excluded"]["pilot_runs"],
        "tasks_per_session": sessions["tasks_per_session"], "tasks_total": len(tasks),
        "tasks_hard": hard, "tasks_medium": medium,
        "client_version": ver.group(1), "model": model.group(1),
        "read_ci_a": ci("A"), "read_ci_b": ci("B"),
        "precision_lo_pct": a["rule"]["precision_ci"]["lo"] * 100,
        "precision_hi_pct": a["rule"]["precision_ci"]["hi"] * 100,
        "false_positives": a["rule"]["fp"], "natural_long_gaps": a["natural_long_gaps"],
        "precision_any_gap_pct": a["rule"]["precision_any_gap"] * 100,
        "recall_any_gap_pct": a["rule"]["recall_any_gap"] * 100,
        "direct_pauses": b["n"] + c["n"],
        "full_rewrite_n": round(b["full_rewrite_share"] * b["n"] + c["full_rewrite_share"] * c["n"]),
        "rest_rewritten_n": round(b["exploratory_rest_rewritten_share"] * b["n"]
                                  + c["exploratory_rest_rewritten_share"] * c["n"]),
        "shared_prefix_tokens": shared.pop(),
    }


def build() -> dict[str, Any]:
    ev_text, ev_raw = _load(EVIDENCE / "pause_experiment.json")
    ss_text, ss_raw = _load(SESSIONS)
    pr_text, pr_raw = _load(PREREG)
    f = derive(json.loads(ev_text)["aggregates"], json.loads(ss_text), pr_text)
    src = "pause_experiment"
    figs = {
        "PauseSessionsPaired": figure(f["sessions_paired"], "int", src, "sessions run in both A and B"),
        "PauseSessionsC": figure(f["sessions_c"], "word", src, "one-hour-caching sessions (C)"),
        "PauseSessionsValid": figure(f["sessions_valid"], "int", src, "valid analysed sessions, A + B + C"),
        "PauseSessionsFailed": figure(f["sessions_failed"], "word", src, "analysed sessions that failed"),
        "PausePilotRuns": figure(f["pilot_runs"], "word", src, "excluded pilot runs"),
        "PauseTasksPerSession": figure(f["tasks_per_session"], "word", "sessions", "tasks chained per session"),
        "PauseTasksTotal": figure(f["tasks_total"], "int", "sessions", "SWE-bench Verified tasks in the sessions"),
        "PauseTasksHard": figure(f["tasks_hard"], "int", "sessions", "tasks annotated 1-4 hours or more"),
        "PauseTasksMedium": figure(f["tasks_medium"], "int", "sessions", "tasks annotated 15 min - 1 hour"),
        "PauseClientVersion": figure(f["client_version"], "text", "prereg", "registered Claude Code version"),
        "PauseModel": figure(f["model"], "text", "prereg", "registered model"),
        "PauseReadCiLoA": figure(f["read_ci_a"][0], "pct1", src, "A: cache-read share, 95% bootstrap low"),
        "PauseReadCiHiA": figure(f["read_ci_a"][1], "pct1", src, "A: cache-read share, 95% bootstrap high"),
        "PauseReadCiLoB": figure(f["read_ci_b"][0], "pct1", src, "B: cache-read share, 95% bootstrap low"),
        "PauseReadCiHiB": figure(f["read_ci_b"][1], "pct1", src, "B: cache-read share, 95% bootstrap high"),
        "PausePrecisionLoPct": figure(f["precision_lo_pct"], "pct0", src, "H3 precision, 95% bootstrap low"),
        "PausePrecisionHiPct": figure(f["precision_hi_pct"], "pct0", src, "H3 precision, 95% bootstrap high"),
        "PauseFalsePositives": figure(f["false_positives"], "word", src, "H3 events labelled idle without an injected pause"),
        "PauseNaturalGaps": figure(f["natural_long_gaps"], "word", src, "natural gaps longer than the lifetime"),
        "PausePrecisionAnyGapPct": figure(f["precision_any_gap_pct"], "pct0", src,
                                          "precision with any gap beyond the lifetime as truth"),
        "PauseRecallAnyGapPct": figure(f["recall_any_gap_pct"], "pct0", src,
                                       "recall with any gap beyond the lifetime as truth"),
        "PauseDirectPauses": figure(f["direct_pauses"], "int", src, "injected pauses in the direct check, B + C"),
        "PauseFullRewriteN": figure(f["full_rewrite_n"], "int", src, "pauses followed by a full re-write"),
        "PauseRestRewrittenN": figure(f["rest_rewritten_n"], "int", src,
                                      "pauses after which everything not read back was re-written (exploratory)"),
        "PauseSharedPrefixTokens": figure(f["shared_prefix_tokens"], "int", src,
                                          "tokens read back after every five-minute pause (shared head)"),
    }
    return {
        "schema_version": 1, "kind": "pause_figures",
        "provenance": {"generated_at": utc_now(), "generator": GENERATOR, "sources": [
            {"id": "pause_experiment", "grade": "measured", "sha256": sha256_bytes(ev_raw),
             "command": "evidence/pause_experiment.json", "description": "pause experiment aggregates"},
            {"id": "sessions", "grade": "documented-record", "sha256": sha256_bytes(ss_raw),
             "command": "experiments/pause/sessions.json", "description": "registered sessions and tasks"},
            {"id": "prereg", "grade": "documented-record", "sha256": sha256_bytes(pr_raw),
             "command": "experiments/PREREG_pause.md", "description": "registered client version and model"}],
            "notes": ["Restates and regroups committed pause-experiment evidence; no new measurement."]},
        "aggregates": f,
        "figures": figs,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=EVIDENCE / "pause_figures.json")
    args = ap.parse_args()
    write_json(args.out, build())
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
