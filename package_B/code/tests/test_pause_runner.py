"""Tests for experiments/pause/runner.py helpers: key-file parsing and run classification.

No docker, no API. Run: python -m unittest discover -s tests -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "pause"))
import runner  # noqa: E402


class KeyFile(unittest.TestCase):
    def read(self, text):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / ".env"
            p.write_text(text, encoding="utf-8")
            return runner.read_key(p)

    def test_only_the_anthropic_line_is_taken(self):
        self.assertEqual(self.read("OTHER_SECRET=nope\nANTHROPIC_API_KEY=abc123\nX=1\n"), "abc123")

    def test_quotes_and_export_are_stripped(self):
        self.assertEqual(self.read('export ANTHROPIC_API_KEY="q-1"\n'), "q-1")
        self.assertEqual(self.read("ANTHROPIC_API_KEY = 'q-2'  \n"), "q-2")

    def test_missing_key_blocks_without_printing_the_file(self):
        with self.assertRaises(SystemExit) as cm:
            self.read("OTHER_SECRET=do-not-print\n")
        self.assertNotIn("do-not-print", str(cm.exception))


class Classify(unittest.TestCase):
    def out(self, result, rc=0):
        d = Path(self.tmp.name)
        (d / "exit.json").write_text(json.dumps({"rc": rc}))
        (d / "result.json").write_text(json.dumps(result))
        return d

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_request_is_an_infra_failure(self):
        self.assertEqual(runner.classify(self.out({"subtype": "success"}), {"requests": 0})[0], "infra_fail")

    def test_api_error_is_an_infra_failure(self):
        res = {"is_error": True, "subtype": "success", "result": "API Error: 529 Overloaded"}
        self.assertEqual(runner.classify(self.out(res), {"requests": 3})[0], "infra_fail")

    def test_agent_outcomes_are_valid_runs(self):
        self.assertEqual(runner.classify(self.out({"subtype": "error_max_turns", "is_error": True}), {"requests": 60}),
                         ("ok", "error_max_turns"))
        self.assertEqual(runner.classify(self.out({}, rc=124), {"requests": 40}), ("ok", "wall-clock timeout"))


class Plans(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        w = Path(self.tmp.name)
        (w / "data").mkdir()
        tasks = [{"rank": i + 1, "instance_id": f"t{i + 1:02d}", "image": "x"} for i in range(24)]
        (w / "tasks.json").write_text(json.dumps({"n_use": 20, "tasks": tasks, "pilot": {"instance_id": "p", "image": "x"}}))
        (w / "data" / "prompts.json").write_text(json.dumps({t["instance_id"]: "fix" for t in tasks} | {"p": "fix"}))
        builds = {t["instance_id"]: {"ok": t["rank"] != 3} for t in tasks}     # rank 3 failed to build
        (w / "builds.json").write_text(json.dumps(builds))
        self.w = w

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, plan, conditions=None):
        import argparse
        return argparse.Namespace(work=self.w, plan=plan, conditions=conditions, pause_seconds=360,
                                  c_pause_seconds=3660, image=None)

    def test_full_plan_is_20_pairs_then_c_on_the_first_10_with_the_registered_settings(self):
        specs = runner.plan_specs(self.args("full"))
        ab = [s for s in specs if s["condition"] in "AB"]
        cs = [s for s in specs if s["condition"] == "C"]
        self.assertEqual(len(ab), 40)
        self.assertNotIn("t03", {s["instance_id"] for s in specs})              # build failure -> spare t21
        self.assertIn("t21", {s["instance_id"] for s in ab})
        self.assertEqual([s["instance_id"] for s in cs], ["t01", "t02", "t04", "t05", "t06", "t07", "t08", "t09", "t10", "t11"])
        a, b, c = (next(s for s in specs if s["condition"] == k) for k in "ABC")
        self.assertEqual((a["pause_at"], a["pause_seconds"], a["cache_1h"]), ("5,15,25", 0, False))
        self.assertEqual((b["pause_at"], b["pause_seconds"], b["cache_1h"]), ("5,15,25", 360, False))
        self.assertEqual((c["pause_at"], c["pause_seconds"], c["cache_1h"]), ("10", 3660, True))
        # a task's A and B are adjacent (they start together)
        for i, s in enumerate(specs):
            if s["condition"] == "A":
                self.assertEqual((specs[i + 1]["instance_id"], specs[i + 1]["condition"]), (s["instance_id"], "B"))

    def test_pilot_is_a_and_b_on_the_pilot_task(self):
        specs = runner.plan_specs(self.args("pilot"))
        self.assertEqual([(s["instance_id"], s["condition"]) for s in specs], [("p", "A"), ("p", "B")])

    def test_missing_build_attempt_blocks_the_full_plan(self):
        builds = json.loads((self.w / "builds.json").read_text())
        del builds["t05"]
        (self.w / "builds.json").write_text(json.dumps(builds))
        with self.assertRaises(SystemExit):
            runner.plan_specs(self.args("full"))

    def test_resume_skips_ok_and_exhausted_runs_only(self):
        runs = self.w / "runs"
        spec = {"run_id": "t01__A__r1"}
        self.assertFalse(runner.finished(runs, spec))
        (runs / "t01__A__r1").mkdir(parents=True)
        (runs / "t01__A__r1" / "meta.json").write_text(json.dumps({"status": "infra_fail"}))
        self.assertFalse(runner.finished(runs, spec))
        (runs / "t01__A__r2").mkdir()
        (runs / "t01__A__r2" / "meta.json").write_text(json.dumps({"status": "infra_fail"}))
        self.assertTrue(runner.finished(runs, spec))


if __name__ == "__main__":
    unittest.main()


class SessionPlans(unittest.TestCase):
    """Amendment 2: chained sessions."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        w = Path(self.tmp.name)
        (w / "data").mkdir()
        sessions = [{"session_id": f"s{i}", "order": i + 1, "condition_c": i < 5, "image": "x",
                     "tasks": [{"instance_id": f"s{i}-t{k}", "base_commit": f"c{k}"} for k in range(4)]} for i in range(10)]
        pilot = {"session_id": "pilot", "image": "x", "tasks": [{"instance_id": f"p-t{k}", "base_commit": f"c{k}"} for k in range(4)]}
        (w / "sessions.json").write_text(json.dumps({"sessions": sessions, "pilot": pilot}))
        ids = [t["instance_id"] for s in sessions + [pilot] for t in s["tasks"]]
        (w / "data" / "session_prompts.json").write_text(json.dumps({i: "fix " + i for i in ids}))
        (w / "builds_sessions.json").write_text(json.dumps({s["session_id"]: {"ok": True} for s in sessions}))
        self.w = w

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, plan, **kw):
        import argparse
        return argparse.Namespace(work=self.w, plan=plan, conditions=kw.get("conditions"), pause_seconds=360,
                                  c_pause_seconds=3660, image=None, session=kw.get("session"))

    def test_pauses_between_tasks_per_condition(self):
        a = self.args("full")
        self.assertEqual(runner.session_pauses("A", a), ({"1": 0, "2": 0, "3": 0}, False))
        self.assertEqual(runner.session_pauses("B", a), ({"1": 360, "2": 360, "3": 360}, False))
        self.assertEqual(runner.session_pauses("C", a), ({"2": 3660}, True))

    def test_full_plan_is_10_ab_pairs_and_c_on_the_five_marked_sessions(self):
        specs = runner.session_specs(self.args("full"))
        self.assertEqual(sum(s["condition"] == "A" for s in specs), 10)
        self.assertEqual(sum(s["condition"] == "B" for s in specs), 10)
        self.assertEqual([s["session_id"] for s in specs if s["condition"] == "C"], ["s0", "s1", "s2", "s3", "s4"])
        for i, s in enumerate(specs):
            if s["condition"] == "A":
                self.assertEqual((specs[i + 1]["session_id"], specs[i + 1]["condition"]), (s["session_id"], "B"))
            self.assertEqual([t["problem"] for t in s["tasks"]], ["fix " + t["instance_id"] for t in s["tasks"]])
            self.assertEqual(s["instance_id"], s["session_id"])      # the analysis pairs sessions by this key

    def test_pilot_is_the_pilot_session_in_a_only(self):
        specs = runner.session_specs(self.args("pilot"))
        self.assertEqual([(s["session_id"], s["condition"]) for s in specs], [("pilot", "A")])

    def test_unbuilt_session_blocks_the_full_plan(self):
        (self.w / "builds_sessions.json").write_text(json.dumps({"s0": {"ok": True}}))
        with self.assertRaises(SystemExit):
            runner.session_specs(self.args("full"))

    def test_wait_images_keeps_unbuilt_sessions_in_the_plan(self):
        (self.w / "builds_sessions.json").write_text(json.dumps({"s0": {"ok": True}}))
        a = self.args("full")
        a.wait_images = True
        specs = runner.session_specs(a)
        self.assertEqual(len({s["session_id"] for s in specs}), 10)

    def test_api_error_detection(self):
        self.assertTrue(runner.task_api_error({"is_error": True, "result": "API Error: 529 overloaded"}))
        self.assertFalse(runner.task_api_error({"is_error": True, "subtype": "error_max_turns"}))


class Amendment3Settings(unittest.TestCase):
    """Amendment 3: injected configuration stays in force, so each change alters the request only at its boundary."""

    def test_settings_are_cumulative_and_start_after_their_boundary(self):
        s1 = {b: list(v) for b, v in runner.A3_TEMPLATES["S1"].items()}
        self.assertEqual(runner.a3_task_settings(s1, 1, "claude-sonnet-5-5"), ("claude-sonnet-5-5", []))
        self.assertEqual(runner.a3_task_settings(s1, 2, "claude-sonnet-5-5"),
                         ("claude-sonnet-5-5", ["--mcp-config", runner.A3_MCP_CONFIG]))
        self.assertEqual(runner.a3_task_settings(s1, 3, "claude-sonnet-5-5"),
                         ("claude-sonnet-5-5", ["--mcp-config", runner.A3_MCP_CONFIG, "--disallowedTools", "Edit"]))
        self.assertEqual(runner.a3_task_settings(s1, 4, "claude-sonnet-5-5")[0], runner.A3_SWITCH_MODEL)

    def test_pauses_only_where_injected(self):
        self.assertEqual(runner.a3_pauses(runner.A3_TEMPLATES["S1"], 360), {"1": 0, "2": 360, "3": 0})
        self.assertEqual(runner.a3_pauses(runner.A3_TEMPLATES["S2"], 360), {"1": 0, "2": 360, "3": 360})

    def test_every_boundary_type_appears_once_per_template_pair(self):
        seen = sorted("+".join(sorted(v)) for t in runner.A3_TEMPLATES.values() for v in t.values())
        self.assertEqual(seen, sorted(["mcp", "pause+tools", "model", "tools", "mcp+pause", "model+pause"]))
