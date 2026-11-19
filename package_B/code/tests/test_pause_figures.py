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
        "rule": {"tp": 30, "fp": 3, "fn": 0, "precision": 30 / 33, "precision_ci": ci(0.909, 0.769, 1.0),
                 "precision_any_gap": 1.0, "recall_any_gap": 1.0, "false_idle_share_a": 1.0},
        "direct_expiry": {"n": 3, "full_rewrite_share": 0.0, "exploratory_rest_rewritten_share": 1.0,
                          "exploratory_read_after_values": [12784], "pauses": PAUSES_B},
        "pauses": {"injected": 3, "expired_after_pause": 1},
        "paired": {"reconciliation": {"per_session": [
            {"b_point_rewrite": 7216 + 27216, "a_point_rewrite": 0}, {"b_point_rewrite": 6000, "a_point_rewrite": 0}]}},
        "condition_c": {"runs": 2, "injected_pauses": 2,
                        "rule": {"events": 2, "tp": 2, "fp": 0, "fn": 0},
                        "direct_expiry": {"n": 2, "full_rewrite_share": 0.0,
                                          "exploratory_rest_rewritten_share": 1.0,
                                          "exploratory_read_after_values": [12659], "pauses": PAUSES_C}},
    }


# Hand-computed re-writes: re-write = max(0, min(C, read + write) - read), C = prefix before the pause.
#   B1: C=20000, r=12784, w=8000  -> min(20000, 20784) - 12784 = 7216  (< 10000: not half)
#   B2: C=40000, r=12784, w=30000 -> 40000 - 12784 = 27216 (>= 20000: half)
#   B3: C=19440, r=12784, w=6000  -> min(19440, 18784) - 12784 = 6000 (< 9720: not half)
#   mean B re-write per pause = (7216 + 27216 + 6000) / 3 = 40432 / 3
PAUSES_B = [{"prefix_before": 20000, "read_after": 12784, "write_after": 8000},
            {"prefix_before": 40000, "read_after": 12784, "write_after": 30000},
            {"prefix_before": 19440, "read_after": 12784, "write_after": 6000}]
#   C1: C=25304, r=12659, w=14062 -> 25304 - 12659 = 12645 < 12652 (just under half)
#   C2: C=36162, r=12659, w=25576 -> 36162 - 12659 = 23503 >= 18081 (half)
PAUSES_C = [{"prefix_before": 25304, "read_after": 12659, "write_after": 14062},
            {"prefix_before": 36162, "read_after": 12659, "write_after": 25576}]


SESSIONS = {"tasks_per_session": 4, "n_sessions": 2, "hard_tasks": 5, "fill_tasks": 3,
            "sessions": [{"tasks": [{"difficulty": "1-4 hours"}, {"difficulty": ">4 hours"},
                                    {"difficulty": "1-4 hours"}, {"difficulty": "15 min - 1 hour"}]},
                         {"tasks": [{"difficulty": "1-4 hours"}, {"difficulty": "1-4 hours"},
                                    {"difficulty": "15 min - 1 hour"}, {"difficulty": "15 min - 1 hour"}]}]}
PREREG = ("Agent: Claude\nCode 2.1.326, headless (`claude -p`), model `claude-sonnet-5-5`, at most 60 turns.\n"
          "- **H1**: pooled cache-read share of input has a 95% bootstrap lower bound\n  above 80% in both conditions.\n"
          "- **H3**: precision >= 0.90 and recall >= 0.90 (point\n  estimates; CIs reported). Also: injected pauses "
          "followed by a re-write of at least half\n  the cached prefix.")


class Derive(unittest.TestCase):
    def test_counts_and_intervals(self):
        f = pf.derive(agg(), SESSIONS, PREREG)
        self.assertEqual(f["sessions_valid"], 6)               # 2 A + 2 B + 2 C
        self.assertEqual(f["tasks_total"], 8)
        self.assertEqual((f["tasks_hard"], f["tasks_medium"]), (5, 3))
        self.assertEqual(f["direct_pauses"], 5)
        self.assertEqual(f["rest_rewritten_n"], 5)
        self.assertEqual(f["full_rewrite_n"], 0)
        self.assertEqual(f["shared_prefix_tokens"], 12784)
        self.assertEqual((f["client_version"], f["model"]), ("2.1.326", "claude-sonnet-5-5"))
        self.assertAlmostEqual(f["read_ci_b"][0], 87.0)
        self.assertAlmostEqual(f["precision_hi_pct"], 100.0)

    def test_registered_thresholds_come_from_the_preregistration(self):
        f = pf.derive(agg(), SESSIONS, PREREG)
        self.assertEqual((f["h1_lower_bound_pct"], f["h3_min_pct"], f["ci_level_pct"]), (80, 90, 95))

    def test_threshold_differing_from_the_analysis_code_is_blocked(self):
        with self.assertRaises(SystemExit):
            pf.derive(agg(), SESSIONS, PREREG.replace("above 80%", "above 75%"))

    def test_rule_event_counts(self):
        f = pf.derive(agg(), SESSIONS, PREREG)
        self.assertEqual((f["rule_tp"], f["rule_fp"], f["rule_fn"]), (30, 3, 0))
        self.assertEqual((f["c_events"], f["c_tp"], f["c_fp"], f["c_fn"]), (2, 2, 0, 0))
        self.assertAlmostEqual(f["precision_pct"], 3000 / 33)
        self.assertAlmostEqual(f["idle_share_a_pct"], 100.0)

    def test_half_prefix_rewrites_per_condition(self):
        f = pf.derive(agg(), SESSIONS, PREREG)
        self.assertEqual((f["half_rewrite_b"], f["pauses_b"]), (1, 3))
        self.assertEqual((f["half_rewrite_c"], f["pauses_c"]), (1, 2))

    def test_half_prefix_count_must_match_the_analysis(self):
        bad = agg()
        bad["pauses"]["expired_after_pause"] = 3
        with self.assertRaises(SystemExit):
            pf.derive(bad, SESSIONS, PREREG)

    def test_rewrite_per_pause_is_checked_against_the_point_rewrites(self):
        f = pf.derive(agg(), SESSIONS, PREREG)
        self.assertAlmostEqual(f["rewrite_per_pause_b"], 40432 / 3)
        bad = agg()
        bad["paired"]["reconciliation"]["per_session"][1]["b_point_rewrite"] = 9000
        with self.assertRaises(SystemExit):
            pf.derive(bad, SESSIONS, PREREG)

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
        self.assertEqual(figs["PauseHalfRewriteB"]["value"], 21)       # = pauses.expired_after_pause
        self.assertEqual(figs["PauseHalfRewriteC"]["value"], 4)
        self.assertEqual((figs["PauseRuleTP"]["value"], figs["PauseRuleFN"]["value"]), (30, 0))
        self.assertAlmostEqual(figs["PauseRewritePerPauseB"]["value"], 54957.7 / 3, delta=0.1)


if __name__ == "__main__":
    unittest.main()
