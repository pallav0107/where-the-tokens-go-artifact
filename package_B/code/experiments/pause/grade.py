#!/usr/bin/env python3
"""Grade the pause experiment's final diffs with the official SWE-bench harness (secondary outcome).

Collects out/patch.diff of every valid non-pilot run (latest attempt per task and condition), writes
one predictions file per condition, and runs swebench.harness.run_evaluation on each against the
same prebuilt instance images the agent used. Needs the harness in a venv (pip install swebench).

    python3 grade.py --work ~/work/cache-pause --python venv-swebench/bin/python --workers 4
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def predictions(runs: Path) -> dict[str, list[dict]]:
    best: dict[tuple[str, str], tuple[int, Path]] = {}
    for meta_path in sorted(runs.glob("*/meta.json")):
        m = json.loads(meta_path.read_text())
        if m.get("pilot") or m.get("status") != "ok":
            continue
        key = (m["instance_id"], m["condition"])
        if key not in best or m["attempt"] > best[key][0]:
            best[key] = (m["attempt"], meta_path.parent)
    out: dict[str, list[dict]] = {"A": [], "B": []}
    for (iid, cond), (_, d) in sorted(best.items()):
        patch = d / "out" / "patch.diff"
        out[cond].append({"instance_id": iid, "model_name_or_path": f"claude-code-pause-{cond}",
                          "model_patch": patch.read_text(encoding="utf-8", errors="replace") if patch.exists() else ""})
    return out


def predictions_sessions(runs: Path) -> dict[str, list[dict]]:
    """Amendment 2 chained sessions: one prediction per task, the final diff of that task (out/task_K/patch.diff)."""
    best: dict[tuple[str, str], tuple[int, Path, dict]] = {}
    for meta_path in sorted(runs.glob("*/meta.json")):
        m = json.loads(meta_path.read_text())
        if m.get("pilot") or m.get("status") != "ok":
            continue
        key = (m["instance_id"], m["condition"])
        if key not in best or m["attempt"] > best[key][0]:
            best[key] = (m["attempt"], meta_path.parent, m)
    out: dict[str, list[dict]] = {}
    for (_, cond), (_, d, m) in sorted(best.items()):
        for t in m.get("tasks", []):
            patch = d / "out" / f"task_{t['k']}" / "patch.diff"
            out.setdefault(cond, []).append({
                "instance_id": t["instance_id"], "model_name_or_path": f"claude-code-pause-{cond}",
                "model_patch": patch.read_text(encoding="utf-8", errors="replace") if patch.exists() else ""})
    return out


def summarise(gdir: Path, preds: dict[str, list[dict]]) -> dict:
    """Resolved counts per condition from the harness reports (<model>.<run_id>.json)."""
    out = {}
    for cond, ps in sorted(preds.items()):
        rep = gdir / f"claude-code-pause-{cond}.cachepause_{cond}.json"
        r = json.loads(rep.read_text()) if rep.exists() else {}
        out[cond] = {"submitted": len(ps), "empty_patches": sum(1 for p in ps if not p["model_patch"].strip()),
                     "resolved": len(r.get("resolved_ids", [])) if r else None,
                     "error": len(r.get("error_ids", [])) if r else None,
                     "completed": len(r.get("completed_ids", [])) if r else None, "report": rep.name if r else None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", type=Path, default=Path.home() / "work" / "cache-pause")
    ap.add_argument("--python", default="venv-swebench/bin/python")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sessions", action="store_true", help="Amendment 2 chained sessions: grade every task's diff")
    ap.add_argument("--summary-only", action="store_true", help="only write grading/summary.json from the reports")
    args = ap.parse_args()
    gdir = args.work / ("grading-sessions" if args.sessions else "grading")
    gdir.mkdir(exist_ok=True)
    allp = (predictions_sessions if args.sessions else predictions)(args.work / "runs")
    if args.summary_only:
        (gdir / "summary.json").write_text(json.dumps(summarise(gdir, allp), indent=1, sort_keys=True))
        return 0
    for cond, preds in allp.items():
        if not preds:
            continue
        pf = gdir / f"predictions_{cond}.jsonl"
        pf.write_text("".join(json.dumps(p) + "\n" for p in preds))
        subprocess.run(["nice", "-n", "10", args.python, "-m", "swebench.harness.run_evaluation",
                        "--dataset_name", "princeton-nlp/SWE-bench_Verified", "--predictions_path", str(pf),
                        "--instance_ids", *[p["instance_id"] for p in preds], "--max_workers", str(args.workers),
                        "--namespace", "swebench", "--run_id", f"cachepause_{cond}"], cwd=gdir, check=False)
    (gdir / "summary.json").write_text(json.dumps(summarise(gdir, allp), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
