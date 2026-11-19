"""Tests for the existing-agents arm; a fake gemini script stands in, never the real CLI."""
import json
import os
import sys

import pytest
import yaml

from experiments import existing_agents as ea

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "..", "agent", "tests", "fixtures",
                       "hotpot_tiny.jsonl")

FAKE = r'''
import json, os, sys
here = os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:]
if "--version" in args:
    print("0.0.fake")
    sys.exit(0)
mode = open(os.path.join(here, "mode.txt")).read().strip()
cwd_ok = os.path.isdir("kb")
with open(os.path.join(here, "calls.log"), "a") as f:
    f.write(os.path.basename(os.getcwd()) + "\n")
if mode == "auth":
    sys.stderr.write("auth failed")
    sys.exit(41)
if mode == "quota":
    sys.stderr.write("RESOURCE_EXHAUSTED: quota")
    sys.exit(1)
m1 = {"api": {"totalRequests": 2}, "tokens": {"prompt": 1000, "candidates": 5,
      "cached": 0, "thoughts": 3, "tool": 0, "total": 1008}}
m2 = {"api": {"totalRequests": 1}, "tokens": {"prompt": 200, "candidates": 1,
      "cached": 0, "thoughts": 0, "tool": 0, "total": 201}}
print(json.dumps({"response": "Mount Beta" if cwd_ok else "x",
                  "stats": {"models": {"m-a": m1, "m-b": m2}}}))
'''


@pytest.fixture
def tasks(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"dataset": {
        "fixture": os.path.abspath(FIXTURE), "parquet_url": None, "sample_seed": 1,
        "n_questions": 4, "n_smoke": 0}}))
    out = tmp_path / "tasks"
    ea.main(["prepare", "--out", str(out), "--config", str(cfg)])
    return out


@pytest.fixture
def fake(tmp_path):
    (tmp_path / "fake_gemini.py").write_text(FAKE)
    (tmp_path / "mode.txt").write_text("ok")
    cmd = f'"{sys.executable}" "{tmp_path / "fake_gemini.py"}"'
    return cmd, tmp_path


def _lines(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _ncalls(d):
    return len((d / "calls.log").read_text().split())


def _run(tasks, res, cmd):
    ea.main(["run", "--tasks", str(tasks), "--out", str(res), "--gemini-cmd", cmd,
             "--sleep", "0"])


def test_prepare_layout(tasks):
    dirs = sorted(os.listdir(tasks))
    assert len(dirs) >= 1
    with open(tasks / dirs[0] / "task.json", encoding="utf-8") as fh:
        t = json.load(fh)
    assert set(t) == {"qid", "question", "gold", "position"}
    kb = os.listdir(tasks / dirs[0] / "kb")
    assert kb and all(f.endswith(".txt") for f in kb)
    assert sorted(kb) == sorted(os.listdir(tasks / dirs[-1] / "kb"))
    txt = (tasks / dirs[0] / "kb" / kb[0]).read_text(encoding="utf-8")
    assert ": " in txt


def test_run_parses_and_resumes(tasks, fake, tmp_path):
    cmd, d = fake
    res = tmp_path / "r.jsonl"
    _run(tasks, res, cmd)
    lines = _lines(res)
    assert lines[0]["provenance"] and "0.0.fake" in lines[0]["gemini_version"]
    recs = lines[1:]
    assert len(recs) == len(os.listdir(tasks))
    r = recs[0]
    assert r["status"] == "ok" and r["response"] == "Mount Beta"
    assert r["prompt_tokens"] == 1200 and r["requests"] == 3 and r["output_tokens"] == 6
    assert r["cached_tokens"] == 0 and r["stats"]["models"]
    assert [x["position"] for x in recs] == sorted(x["position"] for x in recs)
    n_calls = _ncalls(d)
    _run(tasks, res, cmd)
    assert _ncalls(d) == n_calls
    assert len(_lines(res)) == len(lines)


def test_parse_defensive():
    p = ea.parse_output(json.dumps({"response": "a", "stats": {"models": {"m": {}}}}))
    assert p["response"] == "a" and p["tokens"] is None and p["requests"] is None
    assert ea.parse_output("not json")["response"] is None


@pytest.mark.parametrize("mode,status", [("auth", "not run: auth"),
                                         ("quota", "not run: quota")])
def test_stop_statuses(tasks, fake, tmp_path, mode, status):
    cmd, d = fake
    (d / "mode.txt").write_text(mode)
    res = tmp_path / "r.jsonl"
    _run(tasks, res, cmd)
    recs = _lines(res)[1:]
    assert len(recs) == len(os.listdir(tasks))
    assert all(r["status"] == status for r in recs)
    assert _ncalls(d) == 1  # stopped after the first call


def _synth(path, pairs):
    with open(path, "w") as fh:
        fh.write(json.dumps({"provenance": True}) + "\n")
        for i, (req, pt) in enumerate(pairs):
            fh.write(json.dumps({"agent": "gemini-cli", "qid": str(i), "status": "ok",
                                 "requests": req, "prompt_tokens": pt}) + "\n")


def _decide(tmp_path, name, pairs):
    p = tmp_path / name
    _synth(p, pairs)
    return ea.analyze(str(p), str(tmp_path / "o.json"))["gemini-cli"]


def test_analyze_decisions(tmp_path):
    reqs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2
    rep = _decide(tmp_path, "a.jsonl", [(r, 1000 * r ** 2) for r in reqs])
    assert rep["decision"] == "supported" and rep["ci"][0] > 1
    with open(tmp_path / "o.json", encoding="utf-8") as fh:
        assert json.load(fh)["gemini-cli"]["cached_note"].startswith("OAuth")
    assert _decide(tmp_path, "b.jsonl", [(r, 1000 * r) for r in reqs])["decision"] == "not supported"
    assert _decide(tmp_path, "c.jsonl", [(1, 10), (2, 40), (3, 90)])["decision"] == "inconclusive"
    assert _decide(tmp_path, "d.jsonl",
                   [(2, 10 + i) for i in range(8)])["decision"] == "inconclusive"
