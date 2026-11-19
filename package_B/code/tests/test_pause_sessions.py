"""Tests for experiments/pause/select_sessions.py (Amendment 2: chained sessions).

Synthetic rows only. Run: python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "pause"))
import select_sessions as ss  # noqa: E402


def row(i, repo, version, diff):
    return {"instance_id": f"{repo.split('/')[1]}__{repo.split('/')[1]}-{i}", "repo": repo, "version": version,
            "difficulty": diff, "base_commit": f"c{i}", "problem_statement": f"p{i}",
            "created_at": f"2020-01-01T00:00:{(i * 7) % 60:02d}"}


def rows():
    out, i = [], 0
    spec = [("django/django", "3.2", 5, 9), ("django/django", "4.0", 3, 5), ("sympy/sympy", "1.1", 2, 4),
            ("pydata/xarray", "0.12", 1, 3), ("astropy/astropy", "5.0", 4, 4),      # excluded repo
            ("pytest-dev/pytest", "5.4", 0, 9),                                         # no hard task
            ("sphinx-doc/sphinx", "3.5", 1, 2)]                                         # too few tasks
    for repo, ver, h, m in spec:
        for _ in range(h):
            i += 1
            out.append(row(i, repo, ver, "1-4 hours"))
        for _ in range(m):
            i += 1
            out.append(row(i, repo, ver, "15 min - 1 hour"))
        i += 1
        out.append(row(i, repo, ver, "<15 min fix"))                                   # never used
    return out


class Sessions(unittest.TestCase):
    def setUp(self):
        self.sessions, self.pilot = ss.form_sessions(rows())

    def test_eligible_groups_only_and_most_hard_first(self):
        ids = {s["session_id"] for s in self.sessions} | ({self.pilot["session_id"]} if self.pilot else set())
        self.assertEqual(ids, {"django-3.2", "django-4.0", "sympy-1.1", "xarray-0.12"})
        self.assertNotIn("astropy-5.0", ids)

    def test_four_tasks_of_one_repo_and_version_hard_first(self):
        for s in self.sessions:
            self.assertEqual(len(s["tasks"]), 4)
            self.assertEqual(len({t["instance_id"].split("-")[0] for t in s["tasks"]}), 1)
            diffs = [t["difficulty"] for t in s["tasks"]]
            self.assertNotIn("<15 min fix", diffs)
            n_hard = sum(d == "1-4 hours" for d in diffs)
            self.assertEqual(diffs, ["1-4 hours"] * n_hard + ["15 min - 1 hour"] * (4 - n_hard))
        d32 = next(s for s in self.sessions if s["session_id"] == "django-3.2")
        self.assertEqual([t["difficulty"] for t in d32["tasks"]], ["1-4 hours"] * 4)

    def test_deterministic_and_c_on_the_first_n_c_in_session_order(self):
        again, _ = ss.form_sessions(rows())
        self.assertEqual(again, self.sessions)
        self.assertEqual([s["order"] for s in self.sessions], list(range(1, len(self.sessions) + 1)))
        self.assertEqual([s["condition_c"] for s in self.sessions],
                         [i < ss.N_C for i in range(len(self.sessions))])

    def test_most_hard_tasks_win_when_the_session_count_binds(self):
        old = ss.N_SESSIONS
        try:
            ss.N_SESSIONS = 2
            sessions, pilot = ss.form_sessions(rows())
        finally:
            ss.N_SESSIONS = old
        self.assertEqual({s["session_id"] for s in sessions}, {"django-3.2", "django-4.0"})
        self.assertEqual(pilot["session_id"], "sympy-1.1")

    def test_image_is_the_most_recent_tasks_instance_image(self):
        by_id = {r["instance_id"]: r for r in rows()}
        for s in self.sessions:
            newest = max(s["tasks"], key=lambda t: by_id[t["instance_id"]]["created_at"])
            self.assertEqual(s["image"], ss.image_name(newest["instance_id"]))

    def test_the_earlier_single_task_pilot_is_never_selected(self):
        extra = [{"instance_id": "django__django-16631", "repo": "django/django", "version": "9.9",
                  "difficulty": "1-4 hours", "base_commit": "x", "problem_statement": "p",
                  "created_at": "2021-01-01T00:00:00"}]
        extra += [row(900 + k, "django/django", "9.9", "15 min - 1 hour") for k in range(3)]
        r = rows() + extra
        sessions, pilot = ss.form_sessions(r)
        ids = {t["instance_id"] for s in sessions + [pilot] if s for t in s["tasks"]}
        self.assertNotIn("django__django-16631", ids)


if __name__ == "__main__":
    unittest.main()
