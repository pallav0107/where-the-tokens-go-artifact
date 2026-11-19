"""PREREG_v3 shared harness (section 3): per-cell cache salts, per-question cell order, block
KBs. Written fail-first against the v2 grid runner, which re-created random.Random(cell_seed)
for every question (one fixed cell order, so arms with identical prefixes could read each
other's implicit-cache writes) and sent no salt."""
import json
import os
import re

import pytest
import yaml

from agent import run_grid
from agent.providers import MockProvider

SALT_RE = re.compile(r"^session [0-9a-f]{16}$")
OLD_ARM_CELLS = [{"name": "react", "k": [2, 8]}, {"name": "notes_reset", "k": [2, 8]},
                 {"name": "window2", "k": [8]}, {"name": "agentless", "k": [8]}]


def _rows(n):
    rows = []
    for i in range(n):
        rows.append({"id": f"id{i:03d}", "question": f"Which river flows past town{i} hill{i}?",
                     "answer": f"river{i}",
                     "context": {"title": [f"Town{i}", f"River{i}"],
                                 "sentences": [[f"Town{i} lies on river{i}.", "It is small."],
                                               [f"River{i} rises at hill{i}."]]}})
    return rows


def _config(tmp_path, arms=None, blocks=None):
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in _rows(12)), encoding="utf-8")
    cfg = {
        "experiment": "h2h_v3_test", "harness": "v3",
        # n_dev keeps the config runnable by the v2 runner (dev = positions 2-7), so the
        # fail-first run exercises the v2 ordering rather than a config error.
        "dataset": {"fixture": str(path), "parquet_url": None, "sha256": None,
                    "cache_dir": None, "sample_seed": 20261115, "n_questions": 2,
                    "n_smoke": 0, "n_dev": 6,
                    "blocks": blocks or {"dev": [[2, 4], [5, 7]], "test": [[8, 10]]}},
        "agent": {"max_steps": 10, "cell_order_seed": 20261117},
        "arms": arms or OLD_ARM_CELLS,
        "models": [{"provider": "mock", "model": "mock-scripted"}],
        "budget": {"max_usd": 1.0, "est_input_tokens_per_run": 100,
                   "est_output_tokens_per_run": 10},
    }
    p = tmp_path / "h2h_v3_test.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(p)


class _Recorder(MockProvider):
    sent: list = []

    def chat(self, messages, **opts):
        _Recorder.sent.append([dict(m) for m in messages])
        return super().chat(messages, **opts)


@pytest.fixture
def recorded(monkeypatch):
    _Recorder.sent = []
    monkeypatch.setattr(run_grid, "make_provider", lambda spec: _Recorder(spec))
    return _Recorder.sent


def _run(tmp_path, cfg, split="dev", extra=()):
    out = tmp_path / f"out_{split}"
    rc = run_grid.main(["--config", cfg, "--out", str(out), "--split", split, *extra])
    assert rc == 0
    reqs = [json.loads(l) for l in (out / "requests.jsonl").read_text().splitlines() if l]
    runs = [json.loads(l) for l in (out / "runs.jsonl").read_text().splitlines() if l]
    return out, reqs, runs


def _cell(rec):
    return (rec.get("arm") or rec["policy"], int(rec["k"]))


def test_cell_order_is_reshuffled_per_question(tmp_path, recorded):
    _, reqs, _ = _run(tmp_path, _config(tmp_path))
    orders: dict = {}
    for r in reqs:
        o = orders.setdefault(r["question_id"], [])
        if not o or o[-1] != _cell(r):
            o.append(_cell(r))
    assert len(orders) == 6 and all(len(o) == 6 for o in orders.values())
    # v2 bug: every question ran the cells in one identical order
    assert len({tuple(o) for o in orders.values()}) > 1


def test_every_request_starts_with_its_cells_salt_and_cells_never_share_one(tmp_path, recorded):
    _, reqs, _ = _run(tmp_path, _config(tmp_path))
    assert len(recorded) == len(reqs) > 0
    salt_of: dict = {}
    for rec, msgs in zip(reqs, recorded):
        first = msgs[0]
        assert first["role"] == "system" and SALT_RE.match(first["content"]), first
        salt_of.setdefault(_cell(rec), set()).add(first["content"])
        assert rec.get("salt") and first["content"] == f"session {rec['salt']}"
    assert all(len(v) == 1 for v in salt_of.values())  # constant within a cell
    flat = [next(iter(v)) for v in salt_of.values()]
    assert len(set(flat)) == len(flat) == 6  # different between cells


def test_no_two_cells_share_a_request_prefix(tmp_path, recorded):
    """The cross-arm cache read needs a shared prefix; with the salt the first message of any
    two cells' requests already differs."""
    _, reqs, _ = _run(tmp_path, _config(tmp_path))
    firsts = {}
    for rec, msgs in zip(reqs, recorded):
        firsts.setdefault(_cell(rec), msgs[0]["content"])
    cells = list(firsts)
    for i, a in enumerate(cells):
        for b in cells[i + 1:]:
            assert firsts[a] != firsts[b]


def test_salts_change_with_the_invocation_nonce_and_are_recorded(tmp_path, recorded):
    cfg = _config(tmp_path)
    out1, reqs1, runs1 = _run(tmp_path, cfg, extra=("--salt-nonce", "a" * 32))
    prov = json.loads((out1 / "provenance.json").read_text())
    assert prov["harness"] == "v3" and prov["salt_nonce"] == "a" * 32
    assert set(prov["salts"]) == {f"{c}|rep1" for c in (
        "react|k=2", "react|k=8", "notes_reset|k=2", "notes_reset|k=8", "window2|k=8",
        "agentless|k=8")}
    assert all(r["salt"] == prov["salts"][f"{r['arm']}|k={r['k']}|rep{r['repeat']}"]
               for r in runs1)
    assert run_grid.cell_salt("a" * 32, "react", 8) != run_grid.cell_salt("b" * 32, "react", 8)
    assert run_grid.cell_salt("a" * 32, "react", 8) != run_grid.cell_salt("a" * 32, "react", 2)


def test_each_block_has_its_own_kb_and_questions_carry_their_block(tmp_path, recorded):
    _, reqs, runs = _run(tmp_path, _config(tmp_path))
    assert sorted({(r["position"], r["block"]) for r in runs}) == [
        (2, 0), (3, 0), (4, 0), (5, 1), (6, 1), (7, 1)]
    # a search can only return documents of the question's own block
    rows_of: dict = {}
    for r in runs:
        rows_of.setdefault(r["block"], set()).add(int(r["question_id"][2:]))
    blk = {r["question_id"]: r["block"] for r in runs}
    seen = 0
    for rec, msgs in zip(reqs, recorded):
        text = " ".join(str(m["content"]) for m in msgs)
        found = {int(x) for x in re.findall(r"(?:Town|River)(\d+):", text)}
        seen += bool(found)
        assert found <= rows_of[blk[rec["question_id"]]], (found, rec["question_id"])
    assert seen > 0


def test_react_v3_runs_under_its_source_step_cap_and_without_notes(tmp_path, recorded):
    arms = [{"name": "react", "k": [8]}]
    cfg = _config(tmp_path, arms=arms)
    data = yaml.safe_load(open(cfg, encoding="utf-8"))
    data["react"] = {"max_steps": 7}
    open(cfg, "w", encoding="utf-8").write(yaml.safe_dump(data))
    _run(tmp_path, cfg)
    system = recorded[0][1]["content"]
    assert "Take at most 7 steps." in system and '"notes"' not in system


def test_repeats_are_independent_runs_with_their_own_salts(tmp_path, recorded):
    cfg = _config(tmp_path, arms=[{"name": "react", "k": [8]}, {"name": "agentless", "k": [8]}])
    _, reqs, runs = _run(tmp_path, cfg, extra=("--repeats", "3"))
    assert len(runs) == 6 * 2 * 3 and len({r["run_id"] for r in runs}) == len(runs)
    assert sorted({r["repeat"] for r in runs}) == [1, 2, 3]
    salts = {(r["arm"], r["repeat"]): r["salt"] for r in runs}
    assert len(set(salts.values())) == 6  # a different salt for every (cell, repeat)
    for rec, msgs in zip(reqs, recorded):
        assert msgs[0]["content"] == f"session {salts[(rec['arm'], rec['repeat'])]}"


def test_two_shards_partition_the_questions(tmp_path, recorded):
    cfg = _config(tmp_path, arms=[{"name": "react", "k": [8]}])
    out0 = tmp_path / "s0"
    out1 = tmp_path / "s1"
    assert run_grid.main(["--config", cfg, "--out", str(out0), "--split", "dev",
                          "--shard", "0/2"]) == 0
    assert run_grid.main(["--config", cfg, "--out", str(out1), "--split", "dev",
                          "--shard", "1/2"]) == 0
    q0 = {json.loads(l)["position"] for l in (out0 / "runs.jsonl").read_text().splitlines()}
    q1 = {json.loads(l)["position"] for l in (out1 / "runs.jsonl").read_text().splitlines()}
    assert q0 == {2, 4, 6} and q1 == {3, 5, 7}
    assert json.loads((out0 / "provenance.json").read_text())["shard"] == [0, 2]


def test_v3_block_estimate_counts_repeats_and_skips_finished_runs():
    cells = [("react", 8), ("aob", 6)]
    rid = lambda p, k, q, r: run_grid.make_run_id(None, "openai", "m", p, k, q, r)
    assert run_grid.remaining_runs(["q1", "q2", "q3"], cells, 3, set(), rid) == 18
    done = {rid("react", 8, "q1", 1), rid("react", 8, "q1", 2), rid("aob", 6, "q3", 3)}
    assert run_grid.remaining_runs(["q1", "q2", "q3"], cells, 3, done, rid) == 15
    assert rid("react", 8, "q1", 2).endswith("|q1|rep2")


# ---- PREREG_v3 Amendment 1 (review findings, before any v3 request) ----------------------

def test_questions_is_a_total_applied_before_sharding(tmp_path, recorded):
    cfg = _config(tmp_path, arms=[{"name": "react", "k": [8]}])
    pos = []
    for sh in ("0/2", "1/2"):
        out = tmp_path / f"q{sh[0]}"
        assert run_grid.main(["--config", cfg, "--out", str(out), "--split", "dev",
                              "--questions", "4", "--shard", sh]) == 0
        pos.append({json.loads(l)["position"] for l in (out / "runs.jsonl").read_text()
                    .splitlines()})
    assert pos == [{2, 4}, {3, 5}]


class _QuotaAfter(_Recorder):
    calls = 0

    def chat(self, messages, **opts):
        _QuotaAfter.calls += 1
        if _QuotaAfter.calls > 3:
            from agent.providers import ProviderInsufficientQuota
            raise ProviderInsufficientQuota("insufficient_quota (test)")
        return super().chat(messages, **opts)


def test_a_billing_stop_writes_the_halt_file_and_nothing_is_sent_while_it_exists(
        tmp_path, monkeypatch):
    _Recorder.sent, _QuotaAfter.calls = [], 0
    monkeypatch.setattr(run_grid, "make_provider", lambda spec: _QuotaAfter(spec))
    cfg = _config(tmp_path, arms=[{"name": "react", "k": [8]}])
    halt = tmp_path / "HALT"
    rc = run_grid.main(["--config", cfg, "--out", str(tmp_path / "a"), "--split", "dev",
                        "--halt-file", str(halt)])
    assert rc == run_grid.EXIT_HALTED and halt.exists()
    sent_before = len(_Recorder.sent)
    # a sibling shard or a later dispatch: no request at all while the file exists
    monkeypatch.setattr(run_grid, "make_provider", lambda spec: _Recorder(spec))
    rc = run_grid.main(["--config", cfg, "--out", str(tmp_path / "b"), "--split", "dev",
                        "--halt-file", str(halt)])
    assert rc == run_grid.EXIT_HALTED and len(_Recorder.sent) == sent_before


class _FailFirst(_Recorder):
    failed = False

    def chat(self, messages, **opts):
        if not _FailFirst.failed:
            _FailFirst.failed = True
            raise RuntimeError("transient (test)")
        return super().chat(messages, **opts)


def test_a_retried_run_gets_a_new_salt(tmp_path, monkeypatch):
    _Recorder.sent, _FailFirst.failed = [], False
    monkeypatch.setattr(run_grid, "make_provider", lambda spec: _FailFirst(spec))
    cfg = _config(tmp_path, arms=[{"name": "react", "k": [8]}])
    out = tmp_path / "r"
    assert run_grid.main(["--config", cfg, "--out", str(out), "--split", "dev",
                          "--salt-nonce", "c" * 32]) == 0
    runs = [json.loads(l) for l in (out / "runs.jsonl").read_text().splitlines()]
    assert all(r["status"] == "ok" for r in runs) and len(runs) == 6
    retried = [json.loads(l) for l in (out / "requests.jsonl").read_text().splitlines()
               if json.loads(l)["attempt"] == 2]
    assert retried
    first_salt = run_grid.cell_salt("c" * 32, "react", 8, 1)
    assert all(r["salt"] == run_grid.cell_salt("c" * 32, "react", 8, 1, 2) != first_salt
               for r in retried)
