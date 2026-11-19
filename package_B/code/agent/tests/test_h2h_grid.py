"""PREREG_v2 data split, chunk ids and the h2h grid runner (mock provider, no network)."""
import json
import os
import random
import types

import numpy as np
import pytest
import yaml

from agent import run_grid
from agent.data import build_kb, load_dataset
from agent.retrieval import Retriever

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "hotpot_tiny.jsonl")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _ds(n_q, n_sm, n_dev, path=FIXTURE):
    return load_dataset({"fixture": path, "parquet_url": None, "sha256": None,
                         "cache_dir": None, "sample_seed": 20261115,
                         "n_questions": n_q, "n_smoke": n_sm, "n_dev": n_dev})


def test_dev_is_disjoint_from_test_and_smoke_and_leaves_v1_sampling_alone():
    d = _ds(1, 1, 2)
    ids = lambda qs: {q.id for q in qs}
    assert len(d.dev) == 2
    assert ids(d.dev).isdisjoint(ids(d.sample)) and ids(d.dev).isdisjoint(ids(d.smoke))
    assert ids(d.sample).isdisjoint(ids(d.smoke))
    assert [q.position for q in d.dev] == [2, 3] and all(q.split == "dev" for q in d.dev)
    v1 = _ds(1, 1, 0)
    assert v1.dev == [] and v1.dev_kb is None
    assert [q.id for q in v1.sample] == [q.id for q in d.sample]
    assert [q.id for q in v1.smoke] == [q.id for q in d.smoke]
    assert [c["text"] for c in v1.kb.chunks] == [c["text"] for c in d.kb.chunks]


def test_dev_is_positions_54_to_77_of_the_v1_permutation(tmp_path):
    rows = [{"id": f"id{i:04d}", "question": f"q{i}", "answer": "a",
             "context": {"title": [f"T{i}"], "sentences": [[f"S{i}."]]}} for i in range(300)]
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    d = _ds(48, 6, 24, str(path))
    perm = np.random.default_rng(20261115).permutation(300)
    assert [q.id for q in d.sample] == [f"id{i:04d}" for i in perm[:48]]
    assert [q.id for q in d.smoke] == [f"id{i:04d}" for i in perm[48:54]]
    assert [q.id for q in d.dev] == [f"id{i:04d}" for i in perm[54:78]]
    assert [q.position for q in d.dev] == list(range(54, 78))
    # the dev KB is built from the dev questions only
    assert {doc["title"] for doc in d.dev_kb.docs} == {f"T{i}" for i in perm[54:78]}
    assert {doc["title"] for doc in d.kb.docs} == {f"T{i}" for i in perm[:48]}


def test_chunk_ids_are_stable_doc_id_hash_index():
    kb = build_kb([{"id": "a", "context": {"title": ["Big", "Small"], "sentences": [
        ["s1.", "s2.", "s3.", "s4.", "s5."], ["t1."]]}}])
    r = Retriever(kb)
    big = kb.docs[0]["doc_id"]
    assert r.chunk_ids == [f"{big}#0", f"{big}#1", f"{kb.docs[1]['doc_id']}#0"]
    hit = r.search_chunks("s4 s5", 1)[0]
    assert hit["id"] == f"{big}#1" and hit["text"].startswith("Big: s4")
    assert [c["id"] for c in r.read_chunks(big)] == [f"{big}#0", f"{big}#1"]
    assert r.read_chunks(f"{big}#1") == r.read_chunks(big) and r.read_chunks("nope") == []
    # v1 observation format is unchanged
    assert r.search_observation("s4 s5", 1).startswith(f"[{big}] Big: s4")


# ------------------------------------------------------------ grid runner

def _config(tmp_path, provider="mock", n_dev=1):
    with open(os.path.join(REPO, "experiments", "configs", "h2h.yaml"), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["dataset"].update(parquet_url=None, sha256=None, cache_dir=None,
                          fixture=FIXTURE, n_questions=2, n_smoke=1, n_dev=n_dev)
    if provider == "mock":
        cfg["models"] = [{"provider": "mock", "model": "mock-scripted"}]
    path = tmp_path / "h2h_mock.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


def _load(out, name):
    with open(os.path.join(out, name), encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def test_h2h_grid_runs_every_cell_question_major_with_seeded_arm_order(tmp_path):
    out = str(tmp_path / "out")
    argv = ["--config", _config(tmp_path), "--split", "test", "--out", out]
    assert run_grid.main(argv) == 0
    runs = _load(out, "runs.jsonl")
    assert len(runs) == 2 * 8 and all(r["status"] == "ok" for r in runs)
    for r in runs:
        assert r["split"] == "test" and r["arm"] == r["policy"]
        assert {"k", "grounded", "n_requests", "usd_list"} <= set(r)
    assert {(r["arm"], r["k"]) for r in runs} == {
        ("react", 2), ("react", 8), ("notes_reset", 2), ("notes_reset", 8),
        ("window2", 8), ("memgpt", 8), ("agentless", 8), ("pbc", 3)}
    by_run = {r["run_id"]: r for r in runs}
    # n_requests = every request of the run, agentless is exactly 2
    reqs = _load(out, "requests.jsonl")
    for r in runs:
        assert r["n_requests"] == sum(1 for q in reqs if q["run_id"] == r["run_id"])
    assert all(r["n_requests"] == 2 for r in runs if r["arm"] == "agentless")
    # question-major: the cell order of each question is the seeded shuffle
    cells = [(a, k) for a, k in [("react", 2), ("react", 8), ("notes_reset", 2),
                                 ("notes_reset", 8), ("window2", 8), ("memgpt", 8),
                                 ("agentless", 8), ("pbc", 3)]]
    random.Random(20261117).shuffle(cells)
    seen = []
    for q in reqs:
        key = (by_run[q["run_id"]]["question_id"], by_run[q["run_id"]]["arm"],
               by_run[q["run_id"]]["k"])
        if key not in seen:
            seen.append(key)
    qids = list(dict.fromkeys(k[0] for k in seen))
    assert len(qids) == 2
    for qid in qids:
        assert [(a, k) for q, a, k in seen if q == qid] == cells
    # resume: nothing is re-run
    assert run_grid.main(argv) == 0
    assert len(_load(out, "runs.jsonl")) == 16


def test_h2h_dev_split_runs_dev_questions_on_the_dev_kb(tmp_path):
    out = str(tmp_path / "out")
    assert run_grid.main(["--config", _config(tmp_path), "--split", "dev", "--out", out]) == 0
    runs = _load(out, "runs.jsonl")
    ds = _ds(2, 1, 1)
    assert {r["question_id"] for r in runs} == {q.id for q in ds.dev}
    assert {r["split"] for r in runs} == {"dev"}


def test_h2h_requires_a_split(tmp_path):
    with pytest.raises(SystemExit):
        run_grid.main(["--config", _config(tmp_path), "--out", str(tmp_path / "o")])


def test_h2h_cloud_cells_obey_the_existing_key_guard(tmp_path):
    out = str(tmp_path / "out")
    run_grid.main(["--config", _config(tmp_path, provider="cloud"), "--split", "test",
                   "--out", out])
    not_run = _load(out, "not_run.jsonl")
    # PREREG_v2 Amendment 2: the configured cloud model is OpenAI gpt-6-luna
    assert len(not_run) == 8 and all("OPENAI_API_KEY" in r["reason"] for r in not_run)
    assert _load(out, "runs.jsonl") == []


def test_h2h_fields_sum_list_price_over_every_request():
    steps = [{"usd": 0.001, "role": "agent"}, {"usd": 0.0005, "role": "summarizer"}]
    f = run_grid._h2h_fields("memgpt", 8, "test", types.SimpleNamespace(steps=steps, grounded=None))
    assert f["n_requests"] == 2 and f["usd_list"] == pytest.approx(0.0015)
    f = run_grid._h2h_fields("memgpt", 8, "test", types.SimpleNamespace(
        steps=steps + [{"usd": None}], grounded=True))
    assert f["usd_list"] is None and f["grounded"] is True
