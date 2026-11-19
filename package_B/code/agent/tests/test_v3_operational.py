"""PREREG_v3 Amendment 4 (operational, no design change): finished runs on disk at once,
outputs copied out of the container on any exit, no resume from a hollow artifact, and
unrecorded spend-guard charges recorded and counted per arm. Each test failed on e274b76."""
import json
import os

import pytest
import yaml

from agent import run_grid
from agent.budget import SpendGuard, collect_prior_spend
from agent.providers import MockProvider, ProviderUnavailable
from agent.tests.test_v3_harness import _config

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _lines(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# ---- (a) runs.jsonl grows run by run ----------------------------------------------------

class _KilledAfter(MockProvider):
    calls = 0

    def chat(self, messages, **opts):
        _KilledAfter.calls += 1
        if _KilledAfter.calls > 7:  # MockProvider runs take 3 requests: 2 runs finish
            raise KeyboardInterrupt("job killed (test)")
        return super().chat(messages, **opts)


def test_finished_runs_are_on_disk_when_the_job_is_killed(tmp_path, monkeypatch):
    _KilledAfter.calls = 0
    monkeypatch.setattr(run_grid, "make_provider", lambda spec: _KilledAfter(spec))
    cfg = _config(tmp_path, arms=[{"name": "react", "k": [8]}])
    out = tmp_path / "killed"
    with pytest.raises(KeyboardInterrupt):
        run_grid.main(["--config", cfg, "--out", str(out), "--split", "dev"])
    runs = _lines(out / "runs.jsonl")
    assert [r["status"] for r in runs] == ["ok", "ok"]
    # resuming skips them and ends with one deduplicated record per run
    monkeypatch.setattr(run_grid, "make_provider", lambda spec: MockProvider(spec))
    assert run_grid.main(["--config", cfg, "--out", str(out), "--split", "dev"]) == 0
    final = _lines(out / "runs.jsonl")
    assert len(final) == len({r["run_id"] for r in final}) == 6


# ---- (b) outputs leave the container on every exit ---------------------------------------

def test_outputs_are_copied_out_of_the_container_before_upload_and_cleanup():
    wf = yaml.safe_load(open(os.path.join(REPO, ".github", "workflows", "experiment.yml"),
                             encoding="utf-8"))
    job = wf["jobs"]["run"]
    names = [s.get("name") or "" for s in job["steps"]]
    run_i = names.index("run job in container")
    copy_i = next(i for i, n in enumerate(names) if n.startswith("stop container and copy"))
    assert run_i < copy_i < names.index("upload outputs") < names.index(
        "remove leftover container")
    copy = job["steps"][copy_i]
    assert "always()" in copy["if"] and "docker stop" in copy["run"] and "docker cp" in copy["run"]
    assert int(job["steps"][run_i]["timeout-minutes"]) < int(job["timeout-minutes"])
    sh = open(os.path.join(REPO, "experiments", "ci", "run_in_container.sh"),
              encoding="utf-8").read()
    assert "trap 'rm -f \"$envfile\"; copy_out' EXIT" in sh and "trap 'exit 143' TERM INT" in sh
    body = sh.split("copy_out() {", 1)[1].split("}", 1)[0]
    assert "docker stop" in body and 'docker cp "$name:/w/out/." out/' in body


# ---- (c) no resume from a hollow artifact ------------------------------------------------

def _prev(tmp_path, n_ok, split="test", questions="24", policies="", run_id="111",
          other_ok=0, job="h2h3"):
    prev = tmp_path / f"prev_{n_ok}_{split}_{policies or 'all'}"
    prev.mkdir()
    (prev / "ci_context.txt").write_text(
        f"job={job}\ngit_sha=abc\nrun_url=https://github.com/o/r/actions/runs/{run_id}\n"
        f"inputs: split={split} models= policies={policies} repeats= questions={questions} "
        f"max_usd=5 prior_spend_usd=0 first_in_chain=false resume=1\n", encoding="utf-8")
    for i, (rid, n) in enumerate(((run_id, n_ok), ("999", other_ok))):
        for s in (0, 1):
            d = prev / f"h2h3_{split}_r{i}_s{s}"
            d.mkdir()
            (d / "provenance.json").write_text(json.dumps({"github_run_id": rid}))
            half = n // 2 + (n % 2 if s == 0 else 0)
            recs = [{"run_id": f"{rid}-{s}-{j}", "status": "ok"} for j in range(half)]
            if recs:  # an error record superseded by its ok retry is counted once
                recs.insert(0, {"run_id": recs[0]["run_id"], "status": "error"})
            (d / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    return str(prev)


def test_resume_check_refuses_a_hollow_artifact(tmp_path):
    from experiments.ci import check_prev_v3 as cp
    assert cp.check(_prev(tmp_path, 1080))[0] is True  # 24 questions x 15 cells x 3 repeats
    ok, msg = cp.check(_prev(tmp_path, 1079, other_ok=5000))
    assert ok is False and "1079 ok runs on disk, 1080 expected" in msg  # other runs ignored
    assert cp.check(_prev(tmp_path, 288, split="dev", questions="96", policies="aob"))[0]
    assert not cp.check(_prev(tmp_path, 287, split="dev", questions="96", policies="aob"))[0]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert cp.check(str(empty))[0] is False  # no ci_context.txt
    assert cp.check(_prev(tmp_path, 0, job="h2h"))[0] is True  # not an h2h3 chain
    assert cp.main([_prev(tmp_path, 3, questions="48")]) == 2


def test_job_runs_the_resume_check_before_any_request():
    sh = open(os.path.join(REPO, "experiments", "ci", "job.sh"), encoding="utf-8").read()
    block = sh.split('if [ "$JOB" = "h2h3" ]; then', 1)[1]
    check = block.index("python experiments/ci/check_prev_v3.py prev || exit 2")
    assert check < block.index("python -m agent.run_grid")


# ---- (d) unrecorded charges recorded, counted in spend and per arm -----------------------

class _FailsAfterCharging:
    name = "openai"

    def __init__(self, spec):
        self.model, self.guard = spec["model"], None

    def attach_budget(self, max_usd, spent):
        self.guard = SpendGuard("openai", self.model, max_usd, spent)
        return self.guard

    def chat(self, messages, **opts):
        self.guard.charge_unconfirmed(0.002)  # sent, maybe billed, no response
        raise ProviderUnavailable("network error calling test: TimeoutError")


def test_an_unrecorded_charge_is_recorded_counted_in_spend_and_per_arm(tmp_path, monkeypatch):
    monkeypatch.setattr(run_grid, "make_provider", lambda spec: _FailsAfterCharging(spec))
    cfg = _config(tmp_path, arms=[{"name": "react", "k": [8]}])
    data = yaml.safe_load(open(cfg, encoding="utf-8"))
    data["models"] = [{"provider": "openai", "model": "gpt-6-luna", "max_output_tokens": 64}]
    open(cfg, "w", encoding="utf-8").write(yaml.safe_dump(data))
    out = tmp_path / "unrec"
    run_grid.main(["--config", cfg, "--out", str(out), "--split", "dev"])
    failed = _lines(out / "requests_failed.jsonl")
    unrec = [r for r in failed if r.get("role") == "unrecorded_charge"]
    assert unrec and all(r["usd"] == pytest.approx(0.002) and r["arm"] == "react"
                         for r in unrec)
    assert collect_prior_spend([str(out)])["total"] == pytest.approx(0.002 * len(unrec))
    from experiments.attempts_v3 import summarise
    s = summarise([str(out)])
    assert s["per_arm"]["react"]["unrecorded_charges"] == len(unrec)
    assert s["per_arm"]["react"]["charged_usd_failed"] == pytest.approx(0.002 * len(unrec))
    assert s["per_arm"]["react"]["error_runs"] >= 1
