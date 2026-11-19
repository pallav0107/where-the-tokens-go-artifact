# AGENT 2027 paper: open-science material (redacted)

The data, code and aggregate evidence behind the paper, with its pre-registrations. Personal and
infrastructure details are redacted; every redaction is logged in `REDACTION_LOG.md`.

Paper: *Where the Tokens Go: How Prompt Caching Shapes the Cost of an LLM Coding Agent*.
AGENT 2027 workshop at ICSE 2027, short paper.
Author: Pallav Pokhriyal (Kfintech).

Repository: https://github.com/pallav0107/where-the-tokens-go-artifact.
Pre-registrations, with every registered version and its commit history:
https://github.com/pallav0107/where-the-tokens-go-preregistrations.

Release note: this material is published only after the author, who owns the source repository,
has reviewed and approved it.

Source: private repository; available to the programme chairs on request. Files come from commits
`de84195`, `9e86a6b`, `1c815e9` and `3bd6629`; each file's source commit and git blob hash are in `MANIFEST.json`.

## Layout

| Path | What it is |
|---|---|
| `package_A/` | The pre-registrations. Publishable soon. |
| `package_A/PREREG_heldout.md` | Held-out test of context x turns against prompt size (registration, never amended). |
| `package_A/PREREG.md` | v1, with Amendment 1. Never run. |
| `package_A/PREREG_v2.md` | v2, with Amendment 2. Superseded after dev round 0. |
| `package_A/PREREG_v3.md` | v3, with Amendments 1-5. This is the comparison the paper reports. |
| `package_A/versions/` | Each earlier registered version, exactly as it stood at its registration or amendment commit (file name = commit). |
| `package_A/PREREG_pause.md` | The controlled pause experiment, with Amendments 1 and 2 (byte-identical to the copy in the public pre-registration repository, where its earlier versions are). |
| `package_A/DECISIONS_excerpt.md` | The decision-log entries that the pre-registrations cite, plus the entries that record each registration. Line numbers match the original. |
| `package_B/` | The data, code and aggregate evidence. |
| `package_B/data/` | The controlled experiment's per-request and per-run records: every v3 shard directory (test, dev rounds 0 and 1, smoke, failed smoke) and the superseded v2 dev round 0. Also `provenance_by_dispatch/`, which holds the provenance of each of the 15 CI dispatches, including every salt nonce. |
| `package_B/code/` | `agent/` (the agent, arms, harness and grid runner, with tests), the analysis scripts in `experiments/` with their configs, tests and offline inputs, the evidence producers in `tools/`, and `contrib/`, a zero-dependency contributor script that computes the paper's figures from anyone's own Claude Code logs and writes aggregates only. |
| `package_B/code/experiments/pause/` | The controlled pause experiment (`PREREG_pause.md`): the dispatcher `runner.py`, the allowlist egress proxy, the API stub for dry runs, the cost ledger and budget stop, task and session selection with the selected `tasks.json` and `sessions.json`, SWE-bench grading, and `image/` (the two Dockerfiles, the container entry and session scripts, and the pause hook). |
| `package_B/code/tools/` | The evidence producers, among them `pause_experiment.py` (the pause analysis; it needs the run directories, which are not released), `pause_figures.py`, `pause_figure_data.py`, `practice_rule.py` and `cost_figure_data.py`, which run on the released evidence alone. |
| `package_B/code/tests/` | Tests of the pause experiment's code, of its analysis and figures (including `pause_figure_data.py`), and of `practice_rule.py` and `cost_figure_data.py` (standard library only). |
| `package_B/evidence/` | The aggregate evidence files behind the paper's numbers, including `pause_experiment.json` and `pause_figures.json` (the pause experiment), `practice_rule.json`, and `cost_table.json`, `cache_writes.json` and `share_robustness.json`, which those producers read. `generated/cost_shares.dat` is the data of the cost-share figure and `generated/pause_points.dat` that of the per-pause figure (Fig. 1). |
| `package_B/verify/compare_evidence.py` | Compares a re-computed evidence file with a published one. |
| `MANIFEST.json` | One record per registration or amendment: the commit, the author date, GitHub's server-side first-push time, the original file's git blob hash and the released file's sha256. It also lists every file's sha256 and source, the Actions run times and the hashes of the archived CI artifacts. |
| `REDACTION_LOG.md` | Each changed line, with its category and the reason. The removed text is never shown. |
| `VERIFICATION.md` | Registration timing: what an independent audit verified, and what GitHub's server timestamps show. |
| `make_archive.py` | Builds the upload archive without local metadata (see below). |

## Redaction

The markers `[REDACTED: personal]`, `[REDACTED: infrastructure]` and `[REDACTED: unrelated]`
replace personal data, machine and account details, and material about other projects.
Redaction keeps the line structure, so line numbers match the originals. No hypothesis, rule,
number or procedure step was changed. Every timestamp keeps its instant; those that carried a
local UTC offset are written in UTC.

- **Package B records:** no field was dropped from any JSONL record. Each record was checked
  against an allowlist of scientific fields. The JSONL files are byte-identical to the archived
  CI artifacts, so their sha256 equals the original's. Two fields were replaced in each
  `provenance.json`: `hostname` and `github_run_url`.
- **Creator:** the author's name and affiliation appear only in the creator line above and in
  the `creator` record of `MANIFEST.json`. No contact address is included.
- **Four redaction rounds:** a second, independent pass neutralised the remaining personal
  details (an author line in the v1 registration, references to a personal machine,
  subscription, key or tooling, gendered pronouns and local time-zone offsets). Its rows are
  under "Round 2" in `REDACTION_LOG.md`. A third pass looked for indirect identifiers: dates
  that followed the local calendar are now UTC dates, role mentions of the owner read "the
  author", and the names of private harness scripts and files are markers. Its rows are under
  "Round 3". Round 4 applied the author's decisions on what round 3 left open: three harness
  request timestamps are cut to dates, the v1 local model alias is described neutrally, and
  three evidence files about the private harness are removed (see "Left out"). Rounds 5 and 6
  added the files behind later revisions of the paper (round 6: the pause experiment and the
  practitioner-rule and cost-figure producers), with their own rows (4 and 2).
- The cited authors of published work (for example "Yao et al. 2023") are kept as bibliographic
  references.

## Checking the hashes

- **Released files:** `sha256sum <file>` should equal the `sha256` given in `MANIFEST.json`.
- **Originals:** holders of the private repository can check each original with
  `git rev-parse <commit>:<path>`, which should equal `original_file_blob`. The released
  pre-registrations differ from the originals only on the lines listed in `REDACTION_LOG.md`.

## Building an archive

`python make_archive.py --dry-run` checks every sha256 in `MANIFEST.json` and lists the files.
`python make_archive.py --out ../agent2027_release.zip` then writes the zip with sorted entries,
one fixed timestamp, fixed permissions and no owner, user, group or host fields, under the
published folder names, so both copies of the package give the same archive. It leaves out
`README_REVIEW.md` and `.gitattributes`. The script's docstring gives an equivalent `tar`
command (`--mtime`, `--owner=0 --group=0 --numeric-owner`, `gzip -n`).

## Reproducing the registered analysis from the released records

```sh
cd package_B/code
python -m venv .venv && . .venv/bin/activate
pip install numpy==2.2.6 PyYAML==6.0.3 pyarrow==24.0.0 tokenledger-llm==0.2.0
python experiments/analyze_v3.py --input ../data/h2h3_test_r1_s0 --input ../data/h2h3_test_r1_s1 \
    --split test --out /tmp/agent_h2h_v3.json
python experiments/report_v3.py --input ../data/h2h3_test_r1_s0 --input ../data/h2h3_test_r1_s1 \
    --checks experiments/offline/posttest_v3_checks.json --out /tmp/agent_h2h_v3_report.json
python ../verify/compare_evidence.py ../evidence/agent_h2h_v3.json /tmp/agent_h2h_v3.json
python ../verify/compare_evidence.py ../evidence/agent_h2h_v3_report.json /tmp/agent_h2h_v3_report.json
```

On the redacted records and redacted code, both comparisons report `aggregates: IDENTICAL` and
`figures: IDENTICAL` (re-run after round 3 on a Linux CPU machine with the versions above, and after round 5 with the
refreshed `report_v3.py`):

- `agent_h2h_v3.json` has 6 aggregate sections, 10,800 runs and 99 figures. It gives the
  registered H8 decision: not supported.
- `agent_h2h_v3_report.json` has 333 figures (291 before the cache-neutral ratios were added).

The data-source hashes also match the published ones. Only provenance strings differ: the time
of generation, the run-URL text, the hash of `PREREG_v3.md`, which sits in `package_A/`
rather than next to the code, and, in the report, the hash of `posttest_v3_checks.json`: the
published value is the sha256 of the same file with CRLF line endings (a Windows checkout).

Every figure value and description in every released evidence file equals the original's, so
every number in the paper is unchanged. Three `aggregates` sections differ, in non-scientific
values only: one run URL in `agent_runs.json`, one subagent role name in
`orchestrator_snapshot.json`, and, in `heldout.json`, the registration time
(`aggregates.prereg.committed_at`, now written in UTC, the same instant) and three harness
request timestamps (`train_first_ts`, `test_first_ts`, `test_last_ts`), which round 4 cut to
their UTC dates. `heldout.json` also records the sha256 of the registered `PREREG_heldout.md`. The
released copy differs from that original on lines 28-29 (Round 2), so the recorded hash matches
the original in the private repository (`original_file_blob` in `MANIFEST.json`), not the
released file.

## Re-running the pause experiment's figures and the rev7 producers from the released aggregates

These scripts need only the Python 3.12 standard library. The producers expect the repository
layout, so link the evidence folder next to the code and copy the registration in first:

```sh
cd package_B/code
python3 -m venv .venv && . .venv/bin/activate
ln -s ../evidence evidence
cp ../../package_A/PREREG_pause.md experiments/PREREG_pause.md
python -m unittest discover -s tests
python tools/pause_figures.py --out /tmp/pause_figures.json
python tools/practice_rule.py --out /tmp/practice_rule.json
python ../verify/compare_evidence.py ../evidence/pause_figures.json /tmp/pause_figures.json
python ../verify/compare_evidence.py ../evidence/practice_rule.json /tmp/practice_rule.json
python -c "import sys; from pathlib import Path; sys.path.insert(0, 'tools'); import cost_figure_data as c; \
sys.exit(c.dat_command(Path('../evidence/cost_table.json'), Path('../evidence/generated/cost_shares.dat'), True))"
python -c "import sys; from pathlib import Path; sys.path.insert(0, 'tools'); import pause_figure_data as p; \nsys.exit(p.dat_command(Path('../evidence/pause_experiment.json'), Path('../evidence/generated/pause_points.dat'), True))"
```

Result on a Linux CPU machine (Python 3.12.3, clean virtual environment, 19 Nov 2026, round 7): 82 tests
pass, none skipped (the pause-hook tests need `bash` and `flock`). Both comparisons report
`aggregates: IDENTICAL` and `figures: IDENTICAL` (41 figures in `pause_figures.json`, 20 in
`practice_rule.json`), and the last two commands print `PASS: cost_shares.dat matches cost_table.json`
and `PASS: pause_points.dat matches pause_experiment.json`.
Two source hashes in the provenance differ, as expected: `prereg` in `pause_figures.json` (the
recorded hash is that of the unredacted registration; the released copy differs on line 48) and
`h2h3_test_r1_s0` in `practice_rule.json` (the hash of the unredacted `agent_h2h_v3.json`; see the
round-1 rows of that file).

`pause_experiment.json` itself is computed from the container run directories (Claude Code
transcripts and pause-hook logs) and the run ledger, which are not released (see "Left out"). Its
tests in `tests/test_pause_experiment.py` run the same analysis on synthetic runs. To run the
experiment yourself: `experiments/pause/runner.py` builds the images and dispatches the runs (it
needs Docker, the official SWE-bench instance images and an Anthropic API key), then
`python tools/pause_experiment.py --runs <runs dir> --ledger <ledger.jsonl>`.

## Release checks run before review

- **Privacy scanner:** the repository's own scanner (`tools/privacy_scan.py`) found 0 findings
  in every release file, data included.
- **Wider deny-list scan:** 0 findings. It covers personal identifiers, hostnames, home paths,
  IPs, runner, device and secret names, the account URL, the names of other projects and topic
  words. Allowed matches:
  - the creator line;
  - the generic dataset cache directory in three configs, which are hash-locked by the
    registration;
  - ordinary topic words inside public HotpotQA text and model answers.
- **Fail-first:** a fake email address and a fake Windows home-directory path were planted in a
  temporary copy. Both scanners reported them. The copy was then deleted.
- **Round 2 (second, independent pass):** after the round-2 edits, the repository scanner again
  found 0 findings in all 195 files, and a round-2 deny-list scan (name forms, other people,
  emails, phones, addresses, birth dates, identifiers, company names, machine and account
  traces, time-zone offsets and names, and possessive references to a personal machine,
  subscription, key or tooling; JSONL records checked field by field) found 0 findings outside
  the creator line. Fail-first: a fake email address (in a Markdown file) and a fake phone number
  (in a JSONL free-text field) were planted in a temporary copy. The deny-list scan reported
  both; the repository scanner, which has no phone rule, reported the email. The copy was then
  deleted. Every changed line was checked against the original git blobs: each one is logged,
  and each logged line differs.
- **Round 3 (indirect identifiers):** 102 log rows on 100 lines in 18 files. After the edits, the
  repository scanner found 0 findings in all 195 files. A round-3 deny-list scan (the round-2
  terms plus local time-of-day wording, role wording, private harness file names, self-hosting
  wording, plan and usage-limit terms, and provider project or key names; JSONL records checked
  field by field) found matches only in the creator line, the redaction notes themselves,
  public HotpotQA answers and model replies, test placeholders, digit runs inside numbers, and
  the items left for the author's decision. Fail-first: a fake email, a fake Windows home path, a
  harness script name with self-hosting wording, and role and time-of-day wording in a JSONL
  free-text field were planted in a temporary copy. The deny-list scan reported all of them;
  the repository scanner reported the two it has rules for (email and home path). The copy was
  then deleted. The package tests give the same result before and after the round-3 edits
  (222 pass; the same 17 fail because `experiments/ci/`, the workflow file and the repo-level
  schema are left out).
- **Round 4 (the author's decisions):** 251 log rows on 251 lines in 5 files, of which 242 are
  the marked stubs of three producer functions. Three evidence files were removed. After the
  edits, the repository scanner found 0 findings, and the deny-list scan found no new match;
  the matches the round-4 items covered are gone. Fail-first: the same kinds of items were
  planted in a temporary copy, and both scanners reported the items they have rules for. The
  copy was then deleted. The package tests give the same result as before (222 pass, the same
  17 fail).
- **Round 6 (pause experiment and rev7 producers):** 34 files added from `1c815e9`, 2 log rows
  on 2 lines in 2 files. The repository scanner found 0 findings in all package files. Fail-first:
  a fake email and a fake Windows home path planted in a temporary copy of a new file were both
  reported; the copy was then deleted. Every released evidence file's `aggregates` and `figures`
  equal the paper's evidence at `1c815e9`, except the three `aggregates` sections described above.
  The registered v3 analysis was re-run from the released records (both files `IDENTICAL`).
- **Reproduction:** see above.

## Left out, and why

- **Harness transcripts and `evidence/raw/`:** private raw logs. Excluded by rule.
- **Pause experiment run directories and ledger:** the run directories hold the containers'
  Claude Code transcripts, pause-hook logs and run metadata (raw logs). The ledger lists every
  provider request id of the paid runs; these ids can be matched to the API account's billing
  records, so they are treated as account identifiers. The aggregates the paper uses (runs,
  pilot runs, list-price cost) are in `pause_experiment.json`, which records the sha256 of both.
- **`tools/cost_table.py`, `tools/share_robustness.py` and `evidence/usage_audit.json`:** the
  producers of `cost_table.json` and `share_robustness.json` and an input only they read. Those two
  evidence files are released as inputs of `practice_rule.py` and `cost_figure_data.py`; they are
  measured on the private transcripts.
- **`evidence/failfirst_cases.json` and `tools/pull_cases.py`:** evidence for a different
  article, about other projects.
- **`evidence/failfirst_incidents.json`:** not used by this paper. Its records come from another
  project and another article. Its producer function in `tools/pull_evidence.py` is replaced by a
  stub that keeps the line count. Every replaced line is marked and listed in
  `REDACTION_LOG.md`.
- **`evidence/harness_costs.json`, `evidence/index_ab.json` and `evidence/baseline_sample.json`:**
  evidence about the private harness (executor usage, an internal A/B and a baseline table). No
  figure in the paper comes from them. Removed in round 4; their producer functions in
  `tools/pull_evidence.py` are marked stubs that keep the line count.
- **`README_REVIEW.md`:** the guide that accompanies the review copy. It is not part of the
  release, is not in the repository, and `make_archive.py` leaves it out.
- **`tools/privacy_scan.py`:** it encodes infrastructure names to search for them.
- **`experiments/ci/`, `tools/runner/` and the CI workflows:** runner and host configuration.
- **`experiments/offline/design_scripts/`:** design-stage scripts. Their hashes are in
  `package_B/code/experiments/offline/MANIFEST.json`.
- **CI job logs and `ci_context.txt`:** they carry run URLs and runner details. The JSONL records
  and the redacted provenance hold the same scientific content.

The dataset is HotpotQA (distractor validation, CC BY-SA 4.0). Gold answers and model outputs in
`package_B/data/` are derived from it.

## Licence

Code (`package_B/code/`, `package_B/verify/`, `make_archive.py`): MIT, see `LICENSE`.
Data and documents (`package_A/`, `package_B/data/`, `package_B/evidence/`, the manifest, logs and this README): CC BY 4.0, see `LICENSE-DATA`.
