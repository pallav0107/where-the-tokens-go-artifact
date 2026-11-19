"""Tests for experiments/pause/ledger.py: cost from transcript usage and the budget stop.

Synthetic usage only (no API, no real transcripts). Run: python -m unittest discover -s tests -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "pause"))
import ledger  # noqa: E402

PRICES = json.loads((ROOT / "evidence" / "params" / "anthropic_list_prices.json").read_text(encoding="utf-8"))["models"]


def rec(model="claude-sonnet-5-5", fresh=1000, read=10000, w5=2000, w1=1000, out=500):
    return {"model": model, "usage": {"input_tokens": fresh, "cache_read_input_tokens": read,
                                      "cache_creation_input_tokens": w5 + w1, "output_tokens": out,
                                      "cache_creation": {"ephemeral_5m_input_tokens": w5,
                                                         "ephemeral_1h_input_tokens": w1}}}


def line(mid, ts, **kw):
    r = rec(**kw)
    return json.dumps({"type": "assistant", "timestamp": ts, "uuid": mid + "-u",
                       "message": {"id": mid, "model": r["model"], "usage": r["usage"]}})


class Cost(unittest.TestCase):
    def test_sonnet_55_list_price_by_token_kind(self):
        # 1000x2 + 10000x0.1 + 2000x2.5 + 1000x4 + 500x10 = 17000 USD per million
        c = ledger.usage_cost([rec()], PRICES)
        self.assertAlmostEqual(c["usd"], 0.017)
        self.assertEqual(c["tokens"], {"fresh": 1000, "cache_read": 10000, "cache_write_5m": 2000,
                                       "cache_write_1h": 1000, "output": 500})
        self.assertEqual((c["requests"], c["unpriced_requests"]), (1, 0))

    def test_dated_model_id_is_priced(self):
        c = ledger.usage_cost([rec(model="claude-sonnet-5-5-20261101")], PRICES)
        self.assertAlmostEqual(c["usd"], 0.017)

    def test_unpriced_model_is_charged_at_the_dearest_rates_not_zero(self):
        c = ledger.usage_cost([rec(model="claude-unknown-9")], PRICES)
        self.assertEqual(c["unpriced_requests"], 1)
        self.assertGreater(c["usd"], 0.017)

    def test_cost_from_a_transcript_tree_deduplicates_repeated_lines(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "proj" / "s.jsonl"
            p.parent.mkdir(parents=True)
            p.write_text("\n".join([line("m1", "2026-11-20T00:00:00.000Z"), line("m1", "2026-11-20T00:00:00.100Z"),
                                    line("m2", "2026-11-20T00:00:05.000Z")]) + "\n", encoding="utf-8")
            c = ledger.transcript_cost(Path(d), PRICES)
        self.assertEqual(c["requests"], 2)
        self.assertAlmostEqual(c["usd"], 0.034)


class Billing(unittest.TestCase):
    """Billing ground truth: request ids and list-price cost per UTC hour and day, to reconcile with the console."""

    def test_request_ids_and_hourly_daily_buckets(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "proj" / "s.jsonl"
            p.parent.mkdir(parents=True)
            p.write_text("\n".join([line("m1", "2026-11-20T23:59:59.000Z"), line("m2", "2026-11-21T00:00:01.000Z"),
                                    line("m3", "2026-11-21T00:30:00.000Z")]) + "\n", encoding="utf-8")
            c = ledger.transcript_cost(Path(d), PRICES)
        self.assertEqual(c["request_ids"], ["m1", "m2", "m3"])
        self.assertEqual((c["first_request"], c["last_request"]), ("2026-11-20T23:59:59.000Z", "2026-11-21T00:30:00.000Z"))
        self.assertEqual(sorted(c["usd_by_hour"]), ["2026-11-20T23", "2026-11-21T00"])
        self.assertAlmostEqual(c["usd_by_hour"]["2026-11-20T23"], 0.017)
        self.assertAlmostEqual(c["usd_by_hour"]["2026-11-21T00"], 0.034)
        self.assertAlmostEqual(c["usd_by_day"]["2026-11-21"], 0.034)

    def test_ledger_sums_periods_over_runs(self):
        with tempfile.TemporaryDirectory() as d:
            lg = ledger.Ledger(Path(d) / "ledger.jsonl")
            lg.append({"run_id": "a", "usd": 1.0, "usd_by_hour": {"2026-11-21T00": 0.4, "2026-11-21T01": 0.6},
                       "usd_by_day": {"2026-11-21": 1.0}})
            lg.append({"run_id": "b", "usd": 2.0, "usd_by_hour": {"2026-11-21T01": 2.0}, "usd_by_day": {"2026-11-21": 2.0}})
            per = lg.by_period()
        self.assertEqual(per["hour"], {"2026-11-21T00": 0.4, "2026-11-21T01": 2.6})
        self.assertEqual(per["day"], {"2026-11-21": 3.0})


class Budget(unittest.TestCase):
    def test_stop_is_spent_plus_max_run_cost_times_concurrency(self):
        self.assertTrue(ledger.may_start(spent=100.0, max_run_cost=5.0, concurrency=8, stop=140.0))
        self.assertFalse(ledger.may_start(spent=100.01, max_run_cost=5.0, concurrency=8, stop=140.0))
        self.assertFalse(ledger.may_start(spent=0.0, max_run_cost=20.0, concurrency=8, stop=140.0))

    def test_ledger_spent_takes_the_larger_of_transcript_and_cli_cost(self):
        with tempfile.TemporaryDirectory() as d:
            lg = ledger.Ledger(Path(d) / "ledger.jsonl")
            lg.append({"run_id": "a", "usd": 1.0, "cli_usd": 1.5})
            lg.append({"run_id": "b", "usd": 2.0, "cli_usd": None})
            self.assertAlmostEqual(lg.spent(), 3.5)
            self.assertAlmostEqual(lg.max_run_cost(prior=0.5), 2.0)
            again = ledger.Ledger(Path(d) / "ledger.jsonl")   # persisted
            self.assertAlmostEqual(again.spent(), 3.5)

    def test_prior_is_used_until_a_run_is_observed(self):
        with tempfile.TemporaryDirectory() as d:
            lg = ledger.Ledger(Path(d) / "ledger.jsonl")
            self.assertEqual(lg.max_run_cost(prior=6.0), 6.0)
            lg.append({"run_id": "a", "usd": 1.0, "cli_usd": None})
            self.assertEqual(lg.max_run_cost(prior=6.0), 6.0)    # prior is a floor, never lowered by luck


if __name__ == "__main__":
    unittest.main()
