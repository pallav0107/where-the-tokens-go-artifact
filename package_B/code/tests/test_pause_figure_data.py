"""Tests for tools/pause_figure_data.py: the per-pause figure's data (Fig. 1).

Synthetic direct-expiry rows with hand-computed answers. Run: python -m unittest discover -s tests -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import pause_figure_data as pfd  # noqa: E402


def agg():
    # re-write = max(0, min(C, read + write) - read); new suffix = write - re-write
    b = [{"prefix_before": 40000, "read_after": 12784, "write_after": 30000},   # rw 27216, new 2784
         {"prefix_before": 20000, "read_after": 12784, "write_after": 8000}]    # rw 7216, new 784
    c = [{"prefix_before": 25304, "read_after": 12659, "write_after": 14062}]   # rw 12645, new 1417
    return {"direct_expiry": {"n": 2, "pauses": b},
            "condition_c": {"direct_expiry": {"n": 1, "pauses": c}}}


class Rows(unittest.TestCase):
    def test_segments_and_order(self):
        rows = pfd.rows(agg())
        # B first, each condition sorted by cached prefix; x runs on with a gap between conditions
        self.assertEqual([r["cond"] for r in rows], ["B", "B", "C"])
        self.assertEqual([r["prefix"] for r in rows], [20.0, 40.0, 25.304])
        self.assertEqual([r["x"] for r in rows], [1, 2, 4])
        first = rows[0]
        self.assertAlmostEqual(first["head"], 12.784)
        self.assertAlmostEqual(first["rewrite"], 7.216)
        self.assertAlmostEqual(first["newsuffix"], 0.784)
        self.assertAlmostEqual(rows[2]["rewrite"], 12.645)

    def test_head_plus_rewrite_beyond_the_prefix_is_blocked(self):
        bad = agg()
        bad["direct_expiry"]["pauses"][0]["read_after"] = 41000
        with self.assertRaises(SystemExit):
            pfd.rows(bad)

    def test_row_count_must_match(self):
        bad = agg()
        bad["direct_expiry"]["n"] = 3
        with self.assertRaises(SystemExit):
            pfd.rows(bad)


class Dat(unittest.TestCase):
    def test_render_and_check_detects_staleness(self):
        text = pfd.render_dat({"aggregates": agg()})
        data = [ln for ln in text.splitlines() if not ln.startswith("#")]
        self.assertEqual(data[0].split(), ["x", "cond", *pfd.COLUMNS])
        self.assertEqual(len(data), 4)
        with tempfile.TemporaryDirectory() as d:
            ev = Path(d) / "ev.json"
            ev.write_text(json.dumps({"aggregates": agg()}), encoding="utf-8")
            dat = Path(d) / "out.dat"
            self.assertEqual(pfd.dat_command(ev, dat, check=True), 1)    # missing
            self.assertEqual(pfd.dat_command(ev, dat, check=False), 0)
            self.assertEqual(pfd.dat_command(ev, dat, check=True), 0)
            dat.write_text(text.replace("12.784", "12.000", 1), encoding="utf-8")
            self.assertEqual(pfd.dat_command(ev, dat, check=True), 1)    # stale

    def test_committed_evidence_renders(self):
        rows = pfd.rows(pfd.load(pfd.EVIDENCE_FILE)["aggregates"])
        self.assertEqual(sum(r["cond"] == "B" for r in rows), 30)
        self.assertEqual(sum(r["cond"] == "C" for r in rows), 5)
        self.assertEqual({round(r["head"] * 1000) for r in rows if r["cond"] == "B"}, {12784})
        mean_b = sum(r["rewrite"] for r in rows if r["cond"] == "B") / 30
        self.assertAlmostEqual(mean_b * 1000, 54957.7 / 3, delta=0.5)   # = H2 point re-write per pause


if __name__ == "__main__":
    unittest.main()
