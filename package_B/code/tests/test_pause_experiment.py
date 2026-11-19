"""Tests for tools/pause_experiment.py on synthetic container runs with hand-computed answers.

Each synthetic run is a directory as experiments/pause/runner.py leaves it: meta.json, the
Claude Code transcript tree under projects/, and the pause hook's log under out/pause/.
Run: python -m unittest discover -s tests -v
"""
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pause_experiment as pe  # noqa: E402

PRICES = json.loads((ROOT / "evidence" / "params" / "anthropic_list_prices.json").read_text(encoding="utf-8"))["models"]
T0 = datetime(2026, 11, 20, 1, 0, 0, tzinfo=timezone.utc)
FRESH = 5


def iso(secs):
    return (T0 + timedelta(seconds=secs)).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def transcript(reqs, ttl="5m", model="claude-sonnet-5-5"):
    """reqs: (secs, read, write). Lines in Claude Code's format with an unbroken parentUuid chain."""
    lines, parent = [], None
    for i, (secs, rd, wr) in enumerate(reqs):
        u_user, u_asst = f"user-{i}", f"asst-{i}"
        lines.append(json.dumps({"type": "user", "uuid": u_user, "parentUuid": parent, "timestamp": iso(secs - 1),
                                 "message": {"role": "user", "content": "x"}}))
        usage = {"input_tokens": FRESH, "cache_read_input_tokens": rd, "cache_creation_input_tokens": wr,
                 "output_tokens": 10,
                 "cache_creation": {"ephemeral_5m_input_tokens": wr if ttl == "5m" else 0,
                                    "ephemeral_1h_input_tokens": wr if ttl == "1h" else 0}}
        lines.append(json.dumps({"type": "assistant", "uuid": u_asst, "parentUuid": u_user, "timestamp": iso(secs),
                                 "version": "2.1.326", "message": {"id": f"msg_{i}", "model": model, "usage": usage}}))
        parent = u_asst
    return "\n".join(lines) + "\n"


def write_run(root, run_id, instance, cond, reqs, pauses, ttl="5m", status="ok", attempt=1, pilot=False):
    d = root / run_id
    (d / "projects" / "-testbed").mkdir(parents=True)
    (d / "out" / "pause").mkdir(parents=True)
    (d / "projects" / "-testbed" / "session.jsonl").write_text(transcript(reqs, ttl), encoding="utf-8")
    (d / "out" / "pause" / "pauses.jsonl").write_text(
        "".join(json.dumps({"call": c, "start": iso(s), "end": iso(e), "seconds": int(e - s)}) + "\n"
                for c, s, e in pauses), encoding="utf-8")
    (d / "meta.json").write_text(json.dumps({"run_id": run_id, "instance_id": instance, "condition": cond,
                                             "status": status, "attempt": attempt, "pilot": pilot}), encoding="utf-8")
    return d


# Condition B: a 360 s pause between requests 1 and 2 expires the five-minute prefix (11000 tokens),
# so request 2 re-writes 11000 after a 375 s gap: the rule's five-minute expiry, i.e. idle.
B_REQS = [(0, 0, 10000), (10, 10000, 1000), (385, 0, 11500), (390, 11500, 500)]
B_PAUSES = [(5, 20, 380)]
# Condition A: the same planned point logged with 0 s; a within-lifetime re-write of 6500 tokens
# (cached 11500, read 5000, write 7000) five seconds later: not idle.
A_REQS = [(0, 0, 10000), (10, 10000, 1000), (25, 11000, 500), (30, 5000, 7000)]
A_PAUSES = [(5, 20, 20)]


class GroundTruth(unittest.TestCase):
    def test_gap_contains_only_pauses_inside_it(self):
        p = [{"start": pe.parse_ts(iso(20)), "end": pe.parse_ts(iso(380)), "seconds": 360}]
        self.assertIsNotNone(pe.pause_in_gap(pe.parse_ts(iso(10)), pe.parse_ts(iso(385)), p))
        self.assertIsNone(pe.pause_in_gap(pe.parse_ts(iso(385)), pe.parse_ts(iso(390)), p))
        self.assertIsNone(pe.pause_in_gap(pe.parse_ts(iso(30)), pe.parse_ts(iso(385)), p))   # started before the gap

    def test_idle_share_counts_both_expiry_causes_only(self):
        self.assertEqual(pe.idle_share({"gap_5_to_60min": 1.0}), 1.0)
        self.assertEqual(pe.idle_share({"gap_gt_60min": 1.0}), 1.0)
        self.assertEqual(pe.idle_share({"within_lifetime": 1.0}), 0.0)
        self.assertEqual(pe.idle_share({"config_change": 1.0}), 0.0)

    def test_lifetime_from_write_tiers(self):
        self.assertEqual(pe.lifetime_tier(5000, 0), "5m")
        self.assertEqual(pe.lifetime_tier(0, 5000), "1h")
        self.assertEqual(pe.lifetime_tier(2500, 2500), "mixed")
        self.assertEqual(pe.lifetime_tier(0, 0), "none")


class Analysis(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def pair(self, inst="t1", b_reqs=B_REQS, a_reqs=A_REQS, ttl="5m"):
        write_run(self.root, f"{inst}__B__r1", inst, "B", b_reqs, B_PAUSES, ttl)
        write_run(self.root, f"{inst}__A__r1", inst, "A", a_reqs, A_PAUSES, ttl)

    def test_run_metrics_label_the_pause_rewrite_idle(self):
        self.pair()
        runs = pe.load_runs(self.root)
        b = pe.run_metrics(next(r for r in runs if r["condition"] == "B"), PRICES)
        self.assertEqual((b["write"], b["rewrite"]), (23000, 11000))
        self.assertEqual(b["point_rewrite"], 11000)
        self.assertEqual(b["injected_pauses"], 1)
        self.assertEqual(b["expired_after_pause"], 1)
        ev = [e for e in b["events"] if e["rewrite"]]
        self.assertEqual(len(ev), 1)
        self.assertTrue(ev[0]["after_pause"])
        self.assertAlmostEqual(ev[0]["idle"], 1.0)
        self.assertGreater(ev[0]["gap_s"], 300)
        self.assertEqual(b["lifetime"], "5m")

    def test_direct_expiry_check_per_injected_pause(self):
        # first request after the pause writes 11500 against an 11000-token cached prefix, reading 0
        self.pair()
        agg = pe.evaluate(pe.load_runs(self.root), PRICES, n_boot=200)
        (pw,) = agg["direct_expiry"]["pauses"]
        self.assertEqual((pw["prefix_before"], pw["write_after"], pw["read_after"]), (11000, 11500, 0))
        self.assertAlmostEqual(pw["write_over_prefix"], 11500 / 11000)
        self.assertTrue(pw["full_rewrite"])
        self.assertEqual(agg["direct_expiry"]["full_rewrite_share"], 1.0)

    def test_a_pause_the_cache_survived_is_not_a_full_rewrite(self):
        b_reqs = [(0, 0, 10000), (10, 10000, 1000), (385, 11000, 500), (390, 11500, 500)]
        self.pair(b_reqs=b_reqs)
        agg = pe.evaluate(pe.load_runs(self.root), PRICES, n_boot=200)
        (pw,) = agg["direct_expiry"]["pauses"]
        self.assertEqual((pw["read_after"], pw["write_after"]), (11000, 500))
        self.assertFalse(pw["full_rewrite"])
        self.assertEqual(agg["direct_expiry"]["full_rewrite_share"], 0.0)

    def test_shared_static_prefix_read_back_rest_rewritten_is_exploratory_only(self):
        # 12784 tokens of a shared static prefix stay warm; the rest (6656) is re-written: not a registered full
        # re-write, but the exploratory rest-rewritten flag is set
        row = pe.direct_expiry_row({"prefix_before": 19440, "write_after": 8522, "read_after": 12784, "gap_s": 366})
        self.assertEqual((row["full_rewrite"], row["rest_rewritten"]), (False, True))
        row = pe.direct_expiry_row({"prefix_before": 19440, "write_after": 3000, "read_after": 12784, "gap_s": 366})
        self.assertFalse(row["rest_rewritten"])

    def test_a_large_write_that_also_read_the_prefix_back_is_not_a_full_rewrite(self):
        row = pe.direct_expiry_row({"prefix_before": 11000, "write_after": 12000, "read_after": 11000, "gap_s": 400})
        self.assertFalse(row["full_rewrite"])

    def test_evaluate_precision_recall_and_paired_differences(self):
        self.pair()
        agg = pe.evaluate(pe.load_runs(self.root), PRICES, n_boot=200)
        self.assertEqual(agg["paired_tasks"], 1)
        r = agg["rule"]
        self.assertEqual((r["tp"], r["fp"], r["fn"], r["tn"]), (1, 0, 0, 1))
        self.assertEqual((r["precision"], r["recall"]), (1.0, 1.0))
        self.assertEqual(r["false_idle_share_a"], 0.0)
        d = agg["paired"]
        self.assertEqual(d["rewrite_diff_tokens"]["mean"], 11000 - 6500)
        self.assertEqual(d["point_rewrite_diff_tokens"]["mean"], 11000)
        ca = agg["composition"]["A"]
        self.assertEqual((ca["fresh"], ca["cache_read"], ca["cache_write"]), (4 * FRESH, 26000, 18500))
        self.assertAlmostEqual(ca["read_share"], 26000 / (20 + 26000 + 18500))
        self.assertAlmostEqual(ca["rewrite_share"], 6500 / 18500)

    def test_natural_long_gap_without_a_pause_is_a_false_positive(self):
        # B run with an extra 400 s tool-runtime gap (no injected pause) before a re-write
        reqs = B_REQS + [(790, 0, 12500)]
        self.pair(b_reqs=reqs)
        agg = pe.evaluate(pe.load_runs(self.root), PRICES, n_boot=200)
        r = agg["rule"]
        self.assertEqual((r["tp"], r["fp"]), (1, 1))
        self.assertAlmostEqual(r["precision"], 0.5)
        self.assertEqual(agg["natural_long_gaps"], 1)
        self.assertEqual((r["precision_any_gap"], r["recall_any_gap"]), (1.0, 1.0))   # secondary truth

    def test_one_hour_entries_survive_a_six_minute_pause_under_the_rule(self):
        # with one-hour writes the rule calls a 6-minute-gap re-write within lifetime: a miss
        self.pair(ttl="1h")
        agg = pe.evaluate(pe.load_runs(self.root), PRICES, n_boot=200)
        self.assertEqual((agg["rule"]["tp"], agg["rule"]["fn"]), (0, 1))
        self.assertEqual(agg["lifetime"], "1h")

    def test_zero_length_planned_point_is_not_ground_truth(self):
        # condition A re-writes 6500 tokens in the gap holding its 0 s planned point: a true negative
        a_reqs = [(0, 0, 10000), (10, 10000, 1000), (25, 4500, 7000)]
        self.pair(a_reqs=a_reqs)
        agg = pe.evaluate(pe.load_runs(self.root), PRICES, n_boot=200)
        self.assertEqual((agg["rule"]["tp"], agg["rule"]["fn"], agg["rule"]["tn"]), (1, 0, 1))
        self.assertEqual(agg["paired"]["point_rewrite_diff_tokens"]["mean"], 11000 - 6500)

    def test_partial_idle_share_is_a_positive_label_at_one_half(self):
        evs = [{"idle": 0.6, "after_pause": True, "gap_s": 400, "rewrite": 1},
               {"idle": 0.4, "after_pause": False, "gap_s": 400, "rewrite": 1}]
        self.assertEqual(pe._confusion(evs), {"tp": 1, "fp": 0, "fn": 0, "tn": 1})

    def test_pilot_and_failed_runs_are_excluded_and_a_rerun_replaces_a_failure(self):
        self.pair("t1")
        write_run(self.root, "p__A__r1", "p", "A", A_REQS, A_PAUSES, pilot=True)
        write_run(self.root, "p__B__r1", "p", "B", B_REQS, B_PAUSES, pilot=True)
        write_run(self.root, "t2__A__r1", "t2", "A", A_REQS, A_PAUSES, status="infra_fail")
        write_run(self.root, "t2__A__r2", "t2", "A", A_REQS, A_PAUSES, attempt=2)
        write_run(self.root, "t2__B__r1", "t2", "B", B_REQS, B_PAUSES)
        write_run(self.root, "t3__A__r1", "t3", "A", A_REQS, A_PAUSES)           # unpaired
        agg = pe.evaluate(pe.load_runs(self.root), PRICES, n_boot=200)
        self.assertEqual(agg["paired_tasks"], 2)
        self.assertEqual(agg["excluded"]["pilot_runs"], 2)
        self.assertEqual(agg["excluded"]["failed_runs"], 1)
        self.assertEqual(agg["excluded"]["unpaired_tasks"], ["t3"])

    def test_include_pilot_is_for_checks_only(self):
        write_run(self.root, "p__A__r1", "p", "A", A_REQS, A_PAUSES, pilot=True)
        write_run(self.root, "p__B__r1", "p", "B", B_REQS, B_PAUSES, pilot=True)
        self.assertEqual(pe.evaluate(pe.load_runs(self.root), PRICES, n_boot=50)["paired_tasks"], 0)
        self.assertEqual(pe.evaluate(pe.load_runs(self.root), PRICES, n_boot=50, include_pilot=True)["paired_tasks"], 1)
        with self.assertRaises(SystemExit):
            pe.main(["--runs", str(self.root), "--ledger", str(self.root / "none.jsonl"), "--include-pilot"])

    # Condition C (Amendment 1): one-hour caching, one 3660 s pause after tool call 10.
    C_REQS = [(0, 0, 10000), (10, 10000, 1000), (3685, 0, 11500), (3690, 11500, 500)]
    C_PAUSES = [(10, 20, 3680)]

    def test_condition_c_long_pause_is_labelled_expiry_after_an_hour(self):
        self.pair()
        write_run(self.root, "t1__C__r1", "t1", "C", self.C_REQS, self.C_PAUSES, ttl="1h")
        agg = pe.evaluate(pe.load_runs(self.root), PRICES, n_boot=200)
        c = agg["condition_c"]
        self.assertEqual(c["runs"], 1)
        self.assertEqual(c["lifetime"], "1h")
        self.assertEqual((c["rule"]["tp"], c["rule"]["fp"], c["rule"]["fn"]), (1, 0, 0))
        self.assertEqual((c["rule"]["precision"], c["rule"]["recall"]), (1.0, 1.0))
        (pw,) = c["direct_expiry"]["pauses"]
        self.assertEqual((pw["prefix_before"], pw["write_after"], pw["read_after"]), (11000, 11500, 0))
        self.assertEqual(c["direct_expiry"]["full_rewrite_share"], 1.0)
        self.assertAlmostEqual(c["composition"]["read_share"], 21500 / (20 + 21500 + 23000))
        # C never enters the paired A/B analysis or its H3
        self.assertEqual(agg["paired_tasks"], 1)
        self.assertEqual(agg["rule"]["events"], 2)
        self.assertEqual(pe.decisions(agg)["H3C"], "supported")

    def test_condition_c_needs_the_over_an_hour_label_not_any_idle_label(self):
        # a 40-minute pause on one-hour entries is within their lifetime: not a positive C label
        reqs = [(0, 0, 10000), (10, 10000, 1000), (2425, 0, 11500)]
        write_run(self.root, "t1__C__r1", "t1", "C", reqs, [(10, 20, 2420)], ttl="1h")
        c = pe.evaluate(pe.load_runs(self.root), PRICES, n_boot=50)["condition_c"]
        self.assertEqual((c["rule"]["tp"], c["rule"]["fn"]), (0, 1))
        # five-minute entries after a 40-minute pause are idle (five-minute expiry) but not a C positive
        write_run(self.root, "t2__C__r1", "t2", "C", reqs, [(10, 20, 2420)], ttl="5m")
        c = pe.evaluate(pe.load_runs(self.root), PRICES, n_boot=50)["condition_c"]
        self.assertEqual((c["rule"]["tp"], c["rule"]["fn"]), (0, 2))
        # the C label is the gap_gt_60min share alone, at one half
        self.assertEqual(pe._confusion_long([{"long": 0.4, "after_pause": True}]), {"tp": 0, "fp": 0, "fn": 1, "tn": 0})

    def test_pilot_and_run_counts_come_from_the_ledger_including_pilots_without_a_run_directory(self):
        self.pair()
        write_run(self.root, "p__A__r1", "p", "A", A_REQS, A_PAUSES, pilot=True)
        ledger = self.root / "ledger.jsonl"
        rows = [{"run_id": "t1__A__r1", "plan": "full", "usd": 0.1, "cli_usd": None},
                {"run_id": "t1__B__r1", "plan": "full", "usd": 0.2, "cli_usd": None},
                {"run_id": "p__A__r1", "plan": "pilot", "usd": 0.05, "cli_usd": None},
                {"run_id": "chain__A__r1", "plan": "pilot", "usd": 0.23, "cli_usd": 0.25}]   # no run directory
        ledger.write_text("".join(json.dumps(r) + chr(10) for r in rows), encoding="utf-8")
        out = self.root / "e.json"
        pe.main(["--runs", str(self.root), "--ledger", str(ledger), "--out", str(out), "--n-boot", "50"])
        a = json.loads(out.read_text(encoding="utf-8"))["aggregates"]
        self.assertEqual(a["excluded"]["pilot_runs"], 2)
        self.assertEqual(a["ledger"]["rows"], 4)
        self.assertEqual(a["ledger"]["pilot_rows"], 2)
        self.assertAlmostEqual(a["ledger"]["usd_total"], 0.1 + 0.2 + 0.05 + 0.25)
        self.assertAlmostEqual(a["ledger"]["usd_pilots"], 0.05 + 0.25)
        self.assertEqual(a["runs_total"], 4)
        self.assertEqual(a["run_directories"], 3)

    def test_reconciliation_separates_new_suffix_and_unpaired_rewrites(self):
        # B's post-pause request writes 11500: 11000 re-written positions + 500 new suffix; A's planned point
        # gap has no write above the cache; A's later in-lifetime re-write (6500, total write 7000) is not at a point
        self.pair()
        agg = pe.evaluate(pe.load_runs(self.root), PRICES, n_boot=50)
        rec = agg["paired"]["reconciliation"]
        (row,) = rec["per_session"]
        self.assertEqual((row["b_point_rewrite"], row["a_point_rewrite"]), (11000, 0))
        self.assertEqual((row["b_point_write"], row["a_point_write"]), (11500, 500))
        self.assertEqual((row["b_rewrite_request_write"], row["a_rewrite_request_write"]), (11500, 7000))
        self.assertEqual(rec["registered"], "point_rewrite_diff_tokens")
        self.assertEqual(rec["mean_registered"], 11000)
        self.assertEqual(rec["mean_rewrite_request_write"], 11500 - 7000)
        self.assertEqual((rec["mean_b_new_suffix_at_points"], rec["mean_off_point_term"]), (500, -7000))
        self.assertEqual(rec["mean_registered"] + rec["mean_b_new_suffix_at_points"] + rec["mean_off_point_term"],
                         rec["mean_rewrite_request_write"])

    def test_bootstrap_is_seeded_and_brackets_the_mean(self):
        vals = [1.0, 2.0, 3.0, 10.0, -4.0]
        a = pe.bootstrap_mean_ci(vals, n_boot=500, seed=2027)
        self.assertEqual(a, pe.bootstrap_mean_ci(vals, n_boot=500, seed=2027))
        self.assertLessEqual(a["lo"], a["mean"])
        self.assertGreaterEqual(a["hi"], a["mean"])
        self.assertAlmostEqual(a["mean"], 2.4)

    def test_main_writes_aggregates_with_provenance(self):
        self.pair()
        out = self.root / "pause_experiment.json"
        ledger = self.root / "ledger.jsonl"
        ledger.write_text(json.dumps({"run_id": "t1__A__r1", "usd": 0.1, "cli_usd": None}) + "\n", encoding="utf-8")
        rc = pe.main(["--runs", str(self.root), "--ledger", str(ledger), "--out", str(out), "--n-boot", "100"])
        self.assertEqual(rc, 0)
        d = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(d["kind"], "pause_experiment")
        self.assertTrue(d["provenance"]["generated_at"])
        for s in d["provenance"]["sources"]:
            self.assertRegex(s["sha256"], r"^[0-9a-f]{64}$")
        self.assertIn("PauseRulePrecisionPct", d["figures"])
        self.assertEqual(d["figures"]["PauseRulePrecisionPct"]["value"], 100.0)
        text = out.read_text(encoding="utf-8")
        self.assertNotIn("msg_", text)          # aggregates only: no message ids
        self.assertNotIn(str(self.root), text)  # no local paths


if __name__ == "__main__":
    unittest.main()
