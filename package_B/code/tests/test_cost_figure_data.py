"""Tests for tools/cost_figure_data.py: the stacked cost-share figure's data (Fig. 1).

Synthetic cost_table aggregates with hand-computed answers. Run: python -m unittest discover -s tests -v
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import cost_figure_data as cfd  # noqa: E402

CAUSES = ("gap_gt_60min", "gap_5_to_60min", "within_lifetime", "config_change", "model_change", "compaction")


def agg():
    rw_tok = dict(zip(CAUSES, (0.010, 0.002, 0.001, 0.001, 0.0005, 0.0005)))
    rw_cost = dict(zip(CAUSES, (0.16, 0.02, 0.02, 0.03, 0.01, 0.01)))
    return {
        "token_share": {"read": 0.95, "write1h": 0.03, "write5m": 0.01, "fresh": 0.004, "output": 0.006},
        "cost_share": {"read": 0.50, "write1h": 0.25, "write5m": 0.07, "fresh": 0.01, "output": 0.17},
        "rewrite_by_cause": {c: {"token_share": rw_tok[c], "cost_share": rw_cost[c]} for c in CAUSES},
        "rewrite_all": {"token_share": sum(rw_tok.values()), "cost_share": sum(rw_cost.values())},
    }


class Rows(unittest.TestCase):
    def test_segments_split_writes_into_new_and_rewrites_by_cause(self):
        rows = cfd.rows(agg())
        cost = rows["cost"]
        self.assertAlmostEqual(cost["read"], 50.0)
        self.assertAlmostEqual(cost["rwidle"], 16.0)
        self.assertAlmostEqual(cost["rwmid"], 2.0)
        self.assertAlmostEqual(cost["rwother"], 7.0)          # within + config + model + compaction
        self.assertAlmostEqual(cost["newwrite"], 32.0 - 25.0)  # all writes minus all re-writes
        self.assertAlmostEqual(cost["fresh"], 1.0)
        self.assertAlmostEqual(cost["output"], 17.0)
        self.assertAlmostEqual(sum(cost.values()), 100.0)
        self.assertAlmostEqual(rows["tokens"]["newwrite"], 4.0 - 1.5)

    def test_shares_that_do_not_add_up_are_blocked(self):
        bad = agg()
        bad["cost_share"]["output"] = 0.30
        with self.assertRaises(SystemExit):
            cfd.rows(bad)

    def test_rewrites_larger_than_writes_are_blocked(self):
        bad = agg()
        bad["rewrite_all"]["cost_share"] = 0.40
        with self.assertRaises(SystemExit):
            cfd.rows(bad)


class Dat(unittest.TestCase):
    def test_render_has_one_row_per_bar_and_check_detects_staleness(self):
        text = cfd.render_dat({"aggregates": agg()})
        data = [ln for ln in text.splitlines() if not ln.startswith("#")]
        self.assertEqual(data[0].split(), ["bar", "y", *cfd.SEGMENTS])
        self.assertEqual([ln.split()[0] for ln in data[1:]], ["Tokens", "Cost"])   # also the y tick labels
        with tempfile.TemporaryDirectory() as d:
            ev = Path(d) / "ev.json"
            ev.write_text('{"aggregates": %s}' % __import__("json").dumps(agg()), encoding="utf-8")
            dat = Path(d) / "out.dat"
            self.assertEqual(cfd.dat_command(ev, dat, check=True), 1)    # missing
            self.assertEqual(cfd.dat_command(ev, dat, check=False), 0)
            self.assertEqual(cfd.dat_command(ev, dat, check=True), 0)
            dat.write_text(text.replace("50.000", "51.000"), encoding="utf-8")
            self.assertEqual(cfd.dat_command(ev, dat, check=True), 1)    # stale

    def test_committed_evidence_renders_and_matches_table_one(self):
        rows = cfd.rows(cfd.load(cfd.EVIDENCE_FILE)["aggregates"])
        self.assertAlmostEqual(rows["cost"]["read"], 54.9867, places=3)   # TabReadCostPct 55.0%
        self.assertAlmostEqual(sum(rows["tokens"].values()), 100.0, places=6)


if __name__ == "__main__":
    unittest.main()
