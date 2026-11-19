"""Acceptance check C4 (written before the implementation): mock end-to-end grid run."""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out = tempfile.mkdtemp(prefix="e2e_mock_")
env = {k: v for k, v in os.environ.items() if k not in ("GEMINI_API_KEY", "DASHSCOPE_API_KEY")}
r = subprocess.run([sys.executable, "-m", "agent.run_grid", "--config", "agent/tests/fixtures/mock_grid.yaml",
                    "--out", out], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
print(r.stdout[-2000:], r.stderr[-2000:])
assert r.returncode == 0, "run_grid failed"


def load(name):
    with open(os.path.join(out, name), encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


runs, reqs, not_run = load("runs.jsonl"), load("requests.jsonl"), load("not_run.jsonl")
ok = [x for x in runs if x["status"] == "ok"]
assert len(ok) == 3 * 2 * 2, f"expected 12 mock runs, got {len(ok)}"
assert {(x["policy"], x["k"]) for x in ok} == {("carry_all", 2), ("carry_all", 8), ("reset_summary", 2), ("reset_summary", 8)}
for x in ok:
    mine = [q for q in reqs if q["run_id"] == x["run_id"]]
    assert len(mine) == x["turns"] >= 1
    assert sum(q["input_tokens"] for q in mine) == x["total_input"]
    assert x["P"] == mine[0]["input_tokens"]
    assert x["split"] in ("train", "test") and "em" in x and "f1" in x
gem = [n for n in not_run if n["model"] == "gemini-2.5-flash"]
assert len(gem) == 4 and all("GEMINI_API_KEY" in n["reason"] for n in gem), not_run
assert not any(x["model"] == "gemini-2.5-flash" for x in runs)
rep = json.load(open(os.path.join(out, "tokenledger_report.json"), encoding="utf-8"))
total = rep.get("total_input", rep.get("metrics", {}).get("total_input"))
assert total == sum(x["total_input"] for x in ok), (total, sum(x["total_input"] for x in ok))
prov = json.load(open(os.path.join(out, "provenance.json"), encoding="utf-8"))
for key in ("git_sha", "dataset_sha256", "started", "ended", "config_sha256"):
    assert prov.get(key), key
# resumability: a second invocation must not re-run completed runs
r2 = subprocess.run([sys.executable, "-m", "agent.run_grid", "--config", "agent/tests/fixtures/mock_grid.yaml",
                     "--out", out], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
assert r2.returncode == 0
assert len([x for x in load("runs.jsonl") if x["status"] == "ok"]) == 12, "resume re-ran completed runs"
print("E2E OK", out)
