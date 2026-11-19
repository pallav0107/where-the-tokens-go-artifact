"""PREREG_v3 section 2: the test split is disjoint from every position any earlier computation
touched (experiments/offline/TOUCHED_POSITIONS.json) and from dev; blocks are 48 questions."""
import json
import os

import yaml

from agent.data import block_positions, split_positions

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG = os.path.join(REPO, "experiments", "configs", "h2h_v3.yaml")
TOUCHED = os.path.join(REPO, "experiments", "offline", "TOUCHED_POSITIONS.json")


def _cfg():
    with open(CONFIG, encoding="utf-8") as fh:
        return yaml.safe_load(fh)["dataset"]


def _touched():
    with open(TOUCHED, encoding="utf-8") as fh:
        t = json.load(fh)
    return t, {p for r in t["ranges"] for p in range(r["start"], r["end"] + 1)}


def test_test_split_touches_no_earlier_position():
    t, touched = _touched()
    test = set(split_positions(_cfg(), "test"))
    assert test and not (test & touched), sorted(test & touched)[:10]
    assert min(test) > t["max_touched"] == max(touched)


def test_dev_and_test_are_disjoint_and_dev_is_inside_the_touched_set():
    _, touched = _touched()
    dev, test = set(split_positions(_cfg(), "dev")), set(split_positions(_cfg(), "test"))
    assert not dev & test
    assert dev <= touched  # dev may reuse design positions; test may not
    assert sorted(dev) == list(range(54, 150))


def test_blocks_are_48_questions_and_do_not_overlap():
    for split, n_blocks in (("dev", 2), ("test", 5)):
        blocks = block_positions(_cfg()["blocks"][split])
        assert len(blocks) == n_blocks and all(len(b) == 48 for b in blocks)
        flat = [p for b in blocks for p in b]
        assert len(flat) == len(set(flat))


def test_test_positions_exist_in_the_dataset():
    assert max(split_positions(_cfg(), "test")) < 7405
