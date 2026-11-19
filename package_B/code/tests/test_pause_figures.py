"""Tests for tools/pause_figures.py: paper figures derived from the pause experiment's evidence.

Synthetic aggregates with hand-computed answers, plus consistency checks on the committed evidence.
Run: python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import pause_figures as pf  # noqa: E402


def agg():
    ci = lambda v, lo, hi: {"value": v, "lo": lo, "hi": hi}  # noqa: E731
    return {
        "composition": {"A": {"read_share_ci": ci(0.95, 0.94, 0.96)}, "B": {"read_share_ci": ci(0.89, 0.87, 0.91)}},
        "paired_tasks": 2,
        "excluded": {"failed_runs": 0, "pilot_runs": 3},
        "natural_long_gaps": 3,
        "rule": {"fp": 3, "precision_ci": ci(0.909, 0.769, 1.0), "precision_any_gap": 1.0, "recall_any_gap": 1.0},
        "direct_expiry": {"n": 30, "full_rewrite_share": 0.0, "exploratory_rest_rewritten_share": 1.0,
                          "exploratory_read_after_values": [12784]},
        "condition_c": {"runs": 5, "direct_expiry": {"n": 5, "full_rewrite_share": 0.0,
                                                     "exploratory_rest_rewritten_share": 1.0,
                                                     "exploratory_read_after_values": [12659]}},
    }


SESSIONS = {"tasks_per_session": 4, "n_sessions": 2, "hard_tasks": 5, "fill_tasks": 3,
            "sessions": [{"tasks": [{"difficulty": "1-4 hours"}, {"difficulty": ">4 hours"},
                                    {"difficulty": "1-4 hours"}, {"difficulty": "15 min - 1 hour"}]},
                         {"tasks": [{"difficulty": "1-4 hours"}, {"difficulty": "1-4 hours"},
                                    {"difficulty": "15 min - 1 hour"}, {"difficulty": "15 min - 1 hour"}]}]}
PREREG = "Agent: Claude\nCode 2.1.326, headless (`claude -p`), model `claude-sonnet-5-5`, at most 60 turns"


class Derive(unittest.TestCase):
    def test_counts_and_intervals(self):
        f = pf.derive(agg(), SESSIONS, PREREG)
        self.assertEqual(f["sessions_valid"], 9)               # 2 A + 2 B + 5 C
        self.assertEqual(f["tasks_total"], 8)
        self.assertEqual((f["tasks_hard"], f["tasks_medium"]), (5, 3))
        self.assertEqual(f["direct_pauses"], 35)
        self.assertEqual(f["rest_rewritten_n"], 35)
        self.assertEqual(f["full_rewrite_n"], 0)
        self.assertEqual(f["shared_prefix_tokens"], 12784)
        self.assertEqual((f["client_version"], f["model"]), ("2.1.326", "claude-sonnet-5-5"))
        self.assertAlmostEqual(f["read_ci_b"][0], 87.0)
        self.assertAlmostEqual(f["precision_hi_pct"], 100.0)

    def test_task_counts_must_match_the_declared_split(self):
        bad = dict(SESSIONS, hard_tasks=6)
        with self.assertRaises(SystemExit):
            pf.derive(agg(), bad, PREREG)

    def test_session_count_must_match_the_paired_count(self):
        bad = agg()
        bad["paired_tasks"] = 3
        with self.assertRaises(SystemExit):
            pf.derive(bad, SESSIONS, PREREG)

    def test_missing_client_version_is_blocked(self):
        with self.assertRaises(SystemExit):
            pf.derive(agg(), SESSIONS, "no version here")


class OnCommittedEvidence(unittest.TestCase):
    def test_committed_evidence_is_consistent(self):
        out = pf.build()
        figs = out["figures"]
        for c in ("A", "B"):
            lo, hi = figs[f"PauseReadCiLo{c}"]["value"], figs[f"PauseReadCiHi{c}"]["value"]
            self.assertLessEqual(lo, hi)
        self.assertEqual(figs["PauseTasksTotal"]["value"], 40)
        self.assertEqual(figs["PauseTasksHard"]["value"] + figs["PauseTasksMedium"]["value"], 40)


if __name__ == "__main__":
    unittest.main()
