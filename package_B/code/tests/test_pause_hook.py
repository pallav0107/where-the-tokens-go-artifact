"""Tests for experiments/pause/image/pause_hook.sh (the PostToolUse hook that injects pauses).

Runs the real script with bash; skipped where bash or flock is missing.
Run: python -m unittest discover -s tests -v
"""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

HOOK = Path(__file__).resolve().parent.parent / "experiments" / "pause" / "image" / "pause_hook.sh"
BASH = shutil.which("bash")


def have_flock() -> bool:
    return bool(BASH) and subprocess.run([BASH, "-c", "command -v flock"], capture_output=True).returncode == 0


@unittest.skipUnless(have_flock(), "needs bash and flock")
class Hook(unittest.TestCase):
    def call(self, d, secs, at="2,4"):
        env = {**os.environ, "PAUSE_STATE_DIR": d, "PAUSE_SECONDS": str(secs), "PAUSE_AT": at}
        r = subprocess.run([BASH, str(HOOK)], input='{"tool_name":"Bash"}', env=env, capture_output=True, text=True)
        self.assertEqual((r.returncode, r.stdout), (0, ""))   # never changes what the agent sees

    def pauses(self, d):
        p = Path(d) / "pauses.jsonl"
        return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []

    def test_condition_a_logs_the_planned_points_with_zero_length(self):
        with tempfile.TemporaryDirectory() as d:
            for _ in range(5):
                self.call(d, 0)
            self.assertEqual([p["call"] for p in self.pauses(d)], [2, 4])
            self.assertEqual({p["seconds"] for p in self.pauses(d)}, {0})
            self.assertEqual(len((Path(d) / "calls.jsonl").read_text().splitlines()), 5)

    def test_condition_b_sleeps_at_the_planned_points_only(self):
        with tempfile.TemporaryDirectory() as d:
            for _ in range(3):
                self.call(d, 1)
            (p,) = self.pauses(d)
            self.assertEqual((p["call"], p["seconds"]), (2, 1))
            t = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))  # noqa: E731
            self.assertGreaterEqual((t(p["end"]) - t(p["start"])).total_seconds(), 0.9)


if __name__ == "__main__":
    unittest.main()
