"""Existing-agents arm (PREREG 7b, Amendment 1 A7): Gemini CLI headless, run unchanged.

Subcommands: prepare (task dirs), run (resumable headless runs), analyze (H6).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent import data as agent_data  # noqa: E402
from agent import metrics  # noqa: E402

N_TASKS = 16
AGENT = "gemini-cli"
PROMPT = ("Answer the question using only the files in ./kb. Reply with only the short "
          "answer (a name, date, number, or yes/no). Question: {question}")
CACHED_NOTE = "OAuth path: no context caching; cached=0 is an auth-path property"
DEFAULT_CONFIG = os.path.join(_ROOT, "experiments", "configs", "grid.yaml")
_TOKEN_KEYS = ("prompt", "candidates", "cached", "thoughts", "tool", "total")
_QUOTA_RE = re.compile(r"\b429\b|quota|RESOURCE_EXHAUSTED|rate.?limit", re.I)
_AUTH_RE = re.compile(r"unauthenticated|not authenticated|authentication|login|"
                      r"sign.?in|\b401\b|invalid.*credential|api key", re.I)


def _load_cfg(path: str) -> dict:
    import yaml
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)["dataset"]


def prepare(out: str, config: str = DEFAULT_CONFIG, parquet: str | None = None) -> list:
    cfg = dict(_load_cfg(config))
    tmp = None
    if parquet:  # use the given parquet as-is: never download
        import pyarrow.parquet as pq
        rows = pq.read_table(parquet).to_pylist()
        fd, tmp = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, default=str) + "\n")
        cfg.update(parquet_url=None, fixture=tmp)
    try:
        ds = agent_data.load_dataset(cfg)
    finally:
        if tmp:
            os.remove(tmp)
    os.makedirs(out, exist_ok=True)
    made = []
    for q in ds.sample[:N_TASKS]:
        tdir = os.path.join(out, q.id)
        kbdir = os.path.join(tdir, "kb")
        os.makedirs(kbdir, exist_ok=True)
        for d in ds.kb.docs:
            with open(os.path.join(kbdir, f"{d['doc_id']}.txt"), "w", encoding="utf-8") as fh:
                fh.write(f"{d['title']}: {d['paragraph']}")
        with open(os.path.join(tdir, "task.json"), "w", encoding="utf-8") as fh:
            json.dump({"qid": q.id, "question": q.question, "gold": q.answer,
                       "position": q.position}, fh)
        made.append(q.id)
    return made


def _cmd_list(gemini_cmd: str) -> list:
    parts = [p.strip('"') for p in shlex.split(gemini_cmd, posix=(os.name != "nt"))]
    exe = shutil.which(parts[0])
    if exe:
        parts[0] = exe
    return parts


def parse_output(stdout: str) -> dict:
    """Defensive parse of `gemini -o json`: missing fields -> None."""
    out = {"response": None, "tokens": None, "requests": None, "stats": None, "error": None}
    try:
        obj = json.loads(stdout)
    except (ValueError, TypeError):
        return out
    if not isinstance(obj, dict):
        return out
    resp = obj.get("response")
    out["response"] = resp if isinstance(resp, str) else None
    if obj.get("error"):
        err = obj["error"]
        out["error"] = err if isinstance(err, str) else json.dumps(err)
    stats = obj.get("stats")
    out["stats"] = stats
    models = stats.get("models") if isinstance(stats, dict) else None
    if isinstance(models, dict) and models:
        tot = {k: None for k in _TOKEN_KEYS}
        reqs = None
        for m in models.values():
            if not isinstance(m, dict):
                continue
            toks = m.get("tokens") if isinstance(m.get("tokens"), dict) else {}
            for k in _TOKEN_KEYS:
                v = toks.get(k)
                if isinstance(v, (int, float)):
                    tot[k] = (tot[k] or 0) + v
            api = m.get("api") if isinstance(m.get("api"), dict) else {}
            r = api.get("totalRequests")
            if isinstance(r, (int, float)):
                reqs = (reqs or 0) + r
        out["tokens"] = tot if any(v is not None for v in tot.values()) else None
        out["requests"] = reqs
    return out


def _task_dirs(tasks: str) -> list:
    items = []
    for name in os.listdir(tasks):
        p = os.path.join(tasks, name, "task.json")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as fh:
                items.append((json.load(fh), os.path.join(tasks, name)))
    items.sort(key=lambda t: t[0]["position"])
    return items


def _done_qids(results: str) -> set:
    done = set()
    if os.path.isfile(results):
        with open(results, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    if rec.get("qid"):
                        done.add(rec["qid"])
    return done


def _append(results: str, rec: dict) -> None:
    with open(results, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def _classify(code, parsed, stderr) -> str:
    blob = f"{stderr}\n{parsed.get('error') or ''}"
    if code == 41:
        return "not run: auth"
    if _QUOTA_RE.search(blob):
        return "not run: quota"
    if code != 0 and _AUTH_RE.search(blob):
        return "not run: auth"
    if code != 0 or parsed["response"] is None:
        return "error"
    return "ok"


def run(tasks: str, results: str, gemini_cmd: str = "gemini", timeout: float = 600,
        sleep: float = 20) -> None:
    base = _cmd_list(gemini_cmd)
    if not (os.path.isfile(results) and os.path.getsize(results) > 0):
        try:
            v = subprocess.run(base + ["--version"], capture_output=True, text=True,
                               timeout=60, encoding="utf-8", errors="replace")
            ver = (v.stdout or v.stderr).strip()
        except Exception as exc:  # noqa: BLE001
            ver = f"unavailable: {exc}"
        _append(results, {"provenance": True, "agent": AGENT, "gemini_version": ver,
                          "cmd": gemini_cmd})
    done = _done_qids(results)
    stop = None
    first = True
    for task, tdir in _task_dirs(tasks):
        if task["qid"] in done:
            continue
        rec = {"agent": AGENT, "qid": task["qid"], "position": task["position"],
               "gold": task["gold"]}
        if stop:
            rec.update(status=stop, response=None)
            _append(results, rec)
            continue
        if not first and sleep:
            time.sleep(sleep)
        first = False
        prompt = PROMPT.format(question=task["question"])
        t0 = time.time()
        try:
            p = subprocess.run(base + ["-p", prompt, "-o", "json"], cwd=tdir,
                               capture_output=True, text=True, timeout=timeout,
                               encoding="utf-8", errors="replace")
            code, so, se = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired:
            code, so, se = None, "", "timeout"
        wall = time.time() - t0
        parsed = parse_output(so)
        status = "error" if code is None else _classify(code, parsed, se)
        resp = parsed["response"]
        tok = parsed["tokens"] or {}
        rec.update(status=status, exit_code=code, wall_s=round(wall, 3), response=resp,
                   em=metrics.exact_match(resp, task["gold"]) if resp is not None else None,
                   f1=metrics.f1_score(resp, task["gold"]) if resp is not None else None,
                   requests=parsed["requests"], tokens=parsed["tokens"],
                   prompt_tokens=tok.get("prompt"), cached_tokens=tok.get("cached"),
                   output_tokens=tok.get("candidates"), stats=parsed["stats"],
                   stderr=se[-2000:] if status != "ok" else None)
        _append(results, rec)
        if status.startswith("not run"):
            stop = status


def _slope(x, y):
    xc = x - x.mean()
    d = float((xc ** 2).sum())
    return float((xc * (y - y.mean())).sum() / d) if d > 0 else float("nan")


def analyze(results: str, out: str) -> dict:
    with open(results, encoding="utf-8") as fh:
        recs = [json.loads(line) for line in fh if line.strip()]
    recs = [r for r in recs if not r.get("provenance")]
    report = {}
    for agent in sorted({r.get("agent", AGENT) for r in recs}):
        rs = [r for r in recs if r.get("agent", AGENT) == agent]
        ok = [r for r in rs if r.get("status") == "ok"]
        use = [r for r in ok
               if (r.get("requests") or 0) >= 1 and (r.get("prompt_tokens") or 0) > 0]
        res = {"n_tasks": len(rs), "n_ok": len(ok),
               "n_not_run": sum(str(r.get("status", "")).startswith("not run") for r in rs),
               "n_usable": len(use), "slope": None, "ci": None,
               "mean_prompt_tokens": (float(np.mean([r["prompt_tokens"] for r in use]))
                                      if use else None),
               "mean_requests": (float(np.mean([r["requests"] for r in use]))
                                 if use else None),
               "cached_note": CACHED_NOTE}
        x = np.log([r["requests"] for r in use]) if use else np.array([])
        y = np.log([r["prompt_tokens"] for r in use]) if use else np.array([])
        if len(use) < 5 or np.ptp(x) == 0:
            res["decision"] = "inconclusive"
        else:
            res["slope"] = _slope(x, y)
            rng = np.random.default_rng(7)
            n = len(use)
            boots = []
            for _ in range(2000):
                idx = rng.integers(0, n, n)
                s = _slope(x[idx], y[idx])
                if not math.isnan(s):
                    boots.append(s)
            if len(boots) < 100:
                res["decision"] = "inconclusive"
            else:
                lo, hi = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
                res["ci"] = [lo, hi]
                res["decision"] = "supported" if lo > 1 else "not supported"
        report[agent] = res
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", required=True)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--parquet")
    r = sub.add_parser("run")
    r.add_argument("--tasks", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--gemini-cmd", default="gemini")
    r.add_argument("--timeout", type=float, default=600)
    r.add_argument("--sleep", type=float, default=20)
    a = sub.add_parser("analyze")
    a.add_argument("--results", required=True)
    a.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "prepare":
        print(f"prepared {len(prepare(args.out, args.config, args.parquet))} tasks")
    elif args.cmd == "run":
        run(args.tasks, args.out, args.gemini_cmd, args.timeout, args.sleep)
    else:
        print(json.dumps(analyze(args.results, args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
