"""Acceptance A4: the full pre-registered analysis must run in under 600 s on 384 synthetic runs."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import experiments.analyze as az  # noqa: E402
from experiments.tests.test_analyze_contract import synth  # noqa: E402

runs, reqs = synth()
assert len(runs) == 384
t = time.time()
az.h1(runs, n_boot=2000, seed=7)
az.h2(runs, reqs, n_boot=2000, seed=7)
az.h3(runs, n_boot=2000, seed=7)
az.h4(runs, n_boot=2000, seed=7)
az.control3(runs, n_perm=200, seed=11, n_boot=2000)
dt = time.time() - t
print(f"full analysis {dt:.1f} s")
sys.exit(0 if dt < 600 else 1)
