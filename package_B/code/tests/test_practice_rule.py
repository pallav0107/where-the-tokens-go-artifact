"""Tests for tools/practice_rule.py on synthetic evidence aggregates with hand-computed answers.

Run: python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import practice_rule as pr  # noqa: E402

PRICES = {"a": {"input": 5.0, "cache_write_5m": 6.25, "cache_write_1h": 10.0, "cache_hit": 0.5, "output": 25.0},
          "b": {"input": 4.0, "cache_write_5m": 5.0, "cache_write_1h": 8.0, "cache_hit": 0.2, "output": 20.0}}
MONTHS = {"m1": {"rewrite": 100.0, "causes": {"gap_gt_60min": 80.0}},
          "m2": {"rewrite": 300.0, "causes": {"gap_gt_60min": 150.0}},
          "m3": {"rewrite": 100.0, "causes": {"gap_gt_60min": 20.0}}}


class Horizons(unittest.TestCase):
    def test_break_even_horizon_is_w_over_rho_minus_one_lifetimes(self):
        h = pr.horizons(PRICES)
        # w(1h) = 2, w(5m) = 1.25 for both models; rho = 0.1 (a) and 0.05 (b)
        self.assertAlmostEqual(h["w_1h"], 2.0)
        self.assertAlmostEqual(h["w_5m"], 1.25)
        self.assertAlmostEqual(h["rho_max"], 0.1)
        self.assertAlmostEqual(h["rho_min"], 0.05)
        self.assertAlmostEqual(h["long_hours_at_rho_max"], 19.0)       # (2 / 0.1 - 1) x 1 h
        self.assertAlmostEqual(h["long_hours_at_rho_min"], 39.0)       # (2 / 0.05 - 1) x 1 h
        self.assertAlmostEqual(h["short_minutes_at_rho_max"], 57.5)    # (1.25 / 0.1 - 1) x 5 min
        self.assertAlmostEqual(h["short_minutes_at_rho_min"], 120.0)   # (1.25 / 0.05 - 1) x 5 min
        # per-tariff one-hour break-even reads (w - 1) / (1 - rho)
        self.assertAlmostEqual(h["long_breakeven_at_rho_max"], 1 / 0.9)
        self.assertAlmostEqual(h["long_breakeven_at_rho_min"], 1 / 0.95)

    def test_mixed_write_multipliers_are_refused(self):
        bad = dict(PRICES, c={"input": 1.0, "cache_write_5m": 1.3, "cache_write_1h": 2.0, "cache_hit": 0.1,
                              "output": 5.0})
        with self.assertRaises(SystemExit):
            pr.horizons(bad)


class LeaveOneMonthOut(unittest.TestCase):
    def test_expiry_share_without_each_month(self):
        lomo = pr.leave_one_month_out(MONTHS, total_rewrite=500.0, total_expiry=250.0)
        # without m1: 170/400; without m2: 100/200; without m3: 230/400
        self.assertAlmostEqual(lomo["by_month_left_out"]["m1"], 42.5)
        self.assertAlmostEqual(lomo["by_month_left_out"]["m2"], 50.0)
        self.assertAlmostEqual(lomo["by_month_left_out"]["m3"], 57.5)
        self.assertAlmostEqual(lomo["min_pct"], 42.5)
        self.assertAlmostEqual(lomo["max_pct"], 57.5)
        self.assertAlmostEqual(lomo["full_pct"], 50.0)

    def test_months_must_add_up_to_the_reported_totals(self):
        with self.assertRaises(SystemExit):
            pr.leave_one_month_out(MONTHS, total_rewrite=501.0, total_expiry=250.0)
        with self.assertRaises(SystemExit):
            pr.leave_one_month_out(MONTHS, total_rewrite=500.0, total_expiry=251.0)


class Plain(unittest.TestCase):
    def test_mean_expiry_rewrite_and_payback(self):
        ka = {"events": 4, "rewrite_tokens": 2_000_000, "rewrite_cost_usd": 20.0}
        one_hour = {"cost_actual_usd": 100.0, "cost_5m_counterfactual_usd": 153.7}
        p = pr.plain(ka, one_hour)
        self.assertAlmostEqual(p["mean_rewrite_tokens"], 500_000)
        self.assertAlmostEqual(p["mean_rewrite_usd"], 5.0)
        self.assertAlmostEqual(p["five_minute_more_pct"], 53.7)

    def test_no_events_is_blocked(self):
        with self.assertRaises(SystemExit):
            pr.plain({"events": 0, "rewrite_tokens": 0, "rewrite_cost_usd": 0.0},
                     {"cost_actual_usd": 1.0, "cost_5m_counterfactual_usd": 1.0})


class OnCommittedEvidence(unittest.TestCase):
    """The producer's figures on the committed evidence equal the reported ones they restate."""

    def test_restated_figures_match_their_sources(self):
        figs = pr.build()["figures"]
        self.assertEqual(round(figs["PlainFiveMinMorePct"]["value"], 1), 53.7)
        self.assertEqual(round(figs["PlainCostRatio"]["value"], 3), 0.487)
        self.assertEqual(round(figs["RuleHorizonLongHours"]["value"]), 19)   # = KaPolicyPings
        lo, hi = figs["RbLomoExpiryMinPct"]["value"], figs["RbLomoExpiryMaxPct"]["value"]
        self.assertTrue(lo < 63.34 < hi)
        self.assertAlmostEqual(figs["TotalListCostUsd"]["value"], 21264.880151150082)   # cache_writes total
        self.assertEqual(round(figs["RuleBreakevenLongMin"]["value"], 3), 1.026)        # rho = 0.025
        self.assertEqual(round(figs["RuleBreakevenLongMax"]["value"], 3), 1.111)        # rho = 0.1
        self.assertEqual(round(figs["RuleHorizonShortMaxMin"]["value"]), 245)          # (1.25/0.025 - 1) x 5
        # round 4: pings per re-write w/rho across models; Khailo's 4-minute ping interval in the same formula
        self.assertAlmostEqual(figs["RulePingsPerRewriteMin"]["value"], 20.0)
        self.assertAlmostEqual(figs["RulePingsPerRewriteMax"]["value"], 80.0)
        self.assertEqual(figs["KhailoPingMin"]["value"], 4)
        self.assertAlmostEqual(figs["RuleHorizonKhailoMin"]["value"], 46.0)            # 4 x (1.25/0.1 - 1)
        # netting the base rate lowers the idle share, by (348098183 - 347684034.68) / 549571289
        self.assertAlmostEqual(figs["IdleNetDropPts"]["value"], 0.07536, places=4)
        # five-minute counterfactual: extra USD over the one-hour blocks' actual cost, and over the total
        self.assertAlmostEqual(figs["PlainFiveMinExtraUsd"]["value"], 24437.680210846906 - 15903.452288396928)
        self.assertAlmostEqual(figs["PlainFiveMinExtraOfTotalPct"]["value"],
                               (24437.680210846906 - 15903.452288396928) / 21264.880151150082 * 100)


if __name__ == "__main__":
    unittest.main()
