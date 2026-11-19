"""Acceptance A3 (written before the implementation): analyze.py CLI on a mock grid output."""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
d = tempfile.mkdtemp(prefix="anz_")
env = {k: v for k, v in os.environ.items() if k not in ("GEMINI_API_KEY", "DASHSCOPE_API_KEY")}
r = subprocess.run([sys.executable, "-m", "agent.run_grid", "--config", "agent/tests/fixtures/mock_grid.yaml",
                    "--out", os.path.join(d, "grid")], cwd=ROOT, env=env, capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-2000:]
out = os.path.join(d, "agent_runs.json")
r = subprocess.run([sys.executable, "experiments/analyze.py", "--input", os.path.join(d, "grid"), "--out", out,
                    "--n-boot", "50", "--n-perm", "10"], cwd=ROOT, capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-1500:])
assert r.returncode == 0
ev = json.load(open(out, encoding="utf-8"))
for key in ("provenance", "cells", "not_run", "hypotheses", "controls", "tokenledger", "exclusions"):
    assert key in ev["aggregates"], key
for h in ("H1", "H2", "H3", "H4", "H5"):
    assert h in ev["aggregates"]["hypotheses"], h
assert ev["aggregates"]["hypotheses"]["H5"]["status"] == "not run"  # no cloud cell ran
assert any(n["model"] == "gemini-2.5-flash" for n in ev["aggregates"]["not_run"])
assert ev["aggregates"]["provenance"]["inputs"][0]["dataset_sha256"]
assert ev["aggregates"]["controls"]["C2_broken_policy"]["status"] in ("not run", "passed", "failed")
print("ANALYZE CLI OK")
