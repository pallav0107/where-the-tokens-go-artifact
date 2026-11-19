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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", type=Path, default=Path.home() / "work" / "cache-pause")
    ap.add_argument("--python", default="venv-swebench/bin/python")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    gdir = args.work / "grading"
    gdir.mkdir(exist_ok=True)
    for cond, preds in predictions(args.work / "runs").items():
        if not preds:
            continue
        pf = gdir / f"predictions_{cond}.jsonl"
        pf.write_text("".join(json.dumps(p) + "\n" for p in preds))
        subprocess.run(["nice", "-n", "10", args.python, "-m", "swebench.harness.run_evaluation",
                        "--dataset_name", "princeton-nlp/SWE-bench_Verified", "--predictions_path", str(pf),
                        "--instance_ids", *[p["instance_id"] for p in preds], "--max_workers", str(args.workers),
                        "--namespace", "swebench", "--run_id", f"cachepause_{cond}"], cwd=gdir, check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
