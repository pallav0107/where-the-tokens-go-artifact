#!/usr/bin/env python3
"""LOCAL CHECK (local workstation only): the contributor kit reproduces the paper's figures on the author's own logs.

Runs contrib/tokens_contrib.py (imported, not re-implemented) on the Claude Code transcripts with its
--until filter at the paper's freeze: requests before 2026-11-15T00:00:00Z, the selection
tools/pull_evidence.py and tools/cache_writes.py use for the orchestrator snapshot. Then:

  1. the per-request log rebuilt from the kit's records must hash to the snapshot's tokenledger_log
     sha256 (the same requests, fields and order the paper was measured on: the snapshot's hash check);
  2. every compared paper figure must match: counts and token totals exactly, shares within 5e-5 points
     of the evidence value (the kit rounds to four decimals) and identical at the paper's printed rounding.

Writes evidence/contrib_equivalence.json (the comparison table and the kit's own result for that window,
aggregates only) only when everything matches; otherwise prints the table and exits 1. CI cannot run
this (no transcripts there); the evidence file and its hashes are its record.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cache_writes  # noqa: E402
import claude_usage  # noqa: E402
from evidence_lib import EVIDENCE, REPO, figure, format_value, sha256_bytes, utc_now, write_json  # noqa: E402

GENERATOR = "tools/contrib_equivalence.py"
KIT = REPO / "contrib" / "tokens_contrib.py"
OUT = EVIDENCE / "contrib_equivalence.json"
SHARE_TOL = 5e-5 + 1e-9


def load_kit():
    spec = importlib.util.spec_from_file_location("tokens_contrib", KIT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ev(name: str) -> tuple[dict[str, Any], str]:
    raw = (EVIDENCE / name).read_bytes().replace(b"\r\n", b"\n")
    return json.loads(raw.decode("utf-8")), sha256_bytes(raw)


def comparisons(res: dict[str, Any]) -> list[tuple[str, str, Callable[[dict[str, Any]], Any]]]:
    """(evidence file, figure name, kit value getter) for every paper figure the kit reproduces."""
    rw = res["rewrites"]
    return [
        ("orchestrator_snapshot.json", "RequestCount", lambda r: r["counts"]["requests"]),
        ("orchestrator_snapshot.json", "TranscriptCount", lambda r: r["counts"]["sessions"] + r["counts"]["subagent_runs"]),
        ("orchestrator_snapshot.json", "OrchMainTranscripts", lambda r: r["counts"]["sessions"]),
        ("orchestrator_snapshot.json", "OrchSubagentTranscripts", lambda r: r["counts"]["subagent_runs"]),
        ("orchestrator_snapshot.json", "OrchModelCount", lambda r: r["counts"]["model_count"]),
        ("share_robustness.json", "RbMonthAll", lambda r: r["counts"]["months_covered"]),
        ("orchestrator_snapshot.json", "TotalInputTokens", lambda r: r["tokens"]["input_total"]),
        ("orchestrator_snapshot.json", "CacheReadSharePct", lambda r: r["input_shares_pct"]["cache_read"]),
        ("orchestrator_snapshot.json", "CacheWriteSharePct", lambda r: r["input_shares_pct"]["cache_write"]),
        ("orchestrator_snapshot.json", "FreshInputSharePct", lambda r: r["input_shares_pct"]["uncached"]),
        ("orchestrator_snapshot.json", "CacheWriteLongSharePct", lambda r: r["cache_write_1h_share_pct"]),
        ("orchestrator_snapshot.json", "CostShareCacheReadPct", lambda r: r["list_price_cost_shares_pct"]["cache_read"]),
        ("orchestrator_snapshot.json", "CostShareCacheWritePct", lambda r: r["list_price_cost_shares_pct"]["cache_write"]),
        ("orchestrator_snapshot.json", "CostShareOutputPct", lambda r: r["list_price_cost_shares_pct"]["output"]),
        ("orchestrator_snapshot.json", "CostShareFreshPct", lambda r: r["list_price_cost_shares_pct"]["uncached_input"]),
        ("orchestrator_snapshot.json", "ReconLineFactor", lambda r: r["line_factor"]["lines_per_request"]),
        ("cache_writes.json", "CwNewSharePct", lambda r: round(rw["new_suffix_tokens"] / rw["written_tokens"] * 100, 4)),
        ("cache_writes.json", "CwRewriteSharePct", lambda r: rw["rewrite_share_of_written_pct"]),
        ("cache_writes.json", "CwRewriteCostSharePct", lambda r: rw["rewrite_share_of_write_cost_pct"]),
        ("cache_writes.json", "CwRewriteTotalCostPct", lambda r: rw["rewrite_share_of_total_cost_pct"]),
        ("cache_writes.json", "CwRewriteShortGapPct", lambda r: rw["by_gap_pct"]["le_5min"]),
        ("cache_writes.json", "CwRewriteMidGapPct", lambda r: rw["by_gap_pct"]["5_to_60min"]),
        ("cache_writes.json", "CwRewriteLongGapPct", lambda r: rw["by_gap_pct"]["gt_60min"]),
        ("cache_writes.json", "CwCauseModelPct", lambda r: rw["by_cause_pct"]["model_change"]),
        ("cache_writes.json", "CwCauseCompactPct", lambda r: rw["by_cause_pct"]["compaction"]),
        ("cache_writes.json", "CwCauseConfigPct", lambda r: rw["by_cause_pct"]["config_change"]),
        ("cache_writes.json", "CwCauseLongGapPct", lambda r: rw["by_cause_pct"]["gap_gt_60min"]),
        ("cache_writes.json", "CwCauseMidGapPct", lambda r: rw["by_cause_pct"]["gap_5_to_60min"]),
        ("cache_writes.json", "CwCauseWithinPct", lambda r: rw["by_cause_pct"]["within_lifetime"]),
        ("cache_writes.json", "CwCauseLongGapNetPct", lambda r: rw["expiry_gt_60min_net_of_base_rate_pct"]),
    ]


def count_checks(res: dict[str, Any], snap: dict[str, Any], cw: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    """(aggregate, paper value, kit value): integer aggregates that must be equal."""
    sa, ca = snap["aggregates"], cw["aggregates"]
    paper_split = sa["cost_split_sensitivity"]["paper"]
    return [
        ("orchestrator_snapshot.input_tokens.fresh", sa["input_tokens"]["fresh"], res["tokens"]["uncached_input"]),
        ("orchestrator_snapshot.input_tokens.cache_read", sa["input_tokens"]["cache_read"], res["tokens"]["cache_read"]),
        ("orchestrator_snapshot.input_tokens.cache_write", sa["input_tokens"]["cache_write"],
         res["tokens"]["cache_write_5m"] + res["tokens"]["cache_write_1h"]),
        ("orchestrator_snapshot.cache_write_tokens_1h", sa["cache_write_tokens_1h"], res["tokens"]["cache_write_1h"]),
        ("orchestrator_snapshot.output_tokens", sa["output_tokens"], res["tokens"]["output"]),
        ("orchestrator_snapshot.reconciliation.line_level_requests_frozen",
         sa["reconciliation"]["line_level_requests_frozen"], res["line_factor"]["usage_lines"]),
        ("orchestrator_snapshot.extraction.cross_file_duplicates", sa["extraction"]["cross_file_duplicates"],
         res["checks"]["cross_file_duplicates"]),
        ("orchestrator_snapshot.extraction.write_split_mismatch", sa["extraction"]["write_split_mismatch"],
         res["checks"]["write_split_mismatch"]),
        ("orchestrator_snapshot.unpriced_requests", sa["unpriced_requests"], res["unpriced"]["requests"]),
        ("orchestrator_snapshot: cache reads the largest cost component (paper prices)",
         paper_split["cacheRead"] == max(paper_split.values()), res["cache_read_largest_cost"]),
        ("cache_writes.write_tokens", ca["write_tokens"], res["rewrites"]["written_tokens"]),
        ("cache_writes.new_suffix", ca["new_suffix"], res["rewrites"]["new_suffix_tokens"]),
        ("cache_writes.rewrite", ca["rewrite"], res["rewrites"]["rewrite_tokens"]),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--projects", type=Path, default=claude_usage.DEFAULT_PROJECTS)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    kit = load_kit()
    until = kit.parse_ts(cache_writes.FREEZE)
    snap, snap_sha = _ev("orchestrator_snapshot.json")
    cw, cw_sha = _ev("cache_writes.json")
    rb, rb_sha = _ev("share_robustness.json")
    evid = {"orchestrator_snapshot.json": snap, "cache_writes.json": cw, "share_robustness.json": rb}

    def progress(i: int, n: int) -> None:
        if i % 250 == 0:
            print(f"  kit reading file {i}/{n}", file=sys.stderr)

    res, internals = kit.compute(args.projects, until=until, progress=progress)
    raw = "".join(json.dumps(r, separators=(",", ":"), sort_keys=True) + "\n"
                  for r in kit.records(internals["transcripts"])).encode("utf-8")
    log_sha = sha256_bytes(raw)
    want = next(s["sha256"] for s in snap["provenance"]["sources"] if s["id"] == "tokenledger_log")
    print(f"per-request log sha256: kit {log_sha[:16]}  snapshot {want[:16]}  "
          f"{'IDENTICAL' if log_sha == want else 'DIFFERENT'}")

    rows = []
    for fname, name, get in comparisons(res):
        fig = evid[fname]["figures"][name]
        pv, kv, fmt = fig["value"], get(res), fig["format"]
        if isinstance(pv, int) and not isinstance(pv, bool) and fmt in ("int", "billions1"):
            value_ok = kv == pv
        else:
            value_ok = kv is not None and abs(float(kv) - float(pv)) <= SHARE_TOL
        pp, kp = format_value(pv, fmt), (format_value(kv, fmt) if kv is not None else "null")
        rows.append({"paper_macro": name, "evidence_file": fname, "format": fmt, "paper_value": pv, "kit_value": kv,
                     "paper_printed": pp, "kit_printed": kp, "printed_match": pp == kp, "value_match": value_ok})
    counts = [{"aggregate": a, "paper": p, "kit": k, "match": p == k} for a, p, k in count_checks(res, snap, cw)]

    print(f"\n{'paper macro':26} {'paper (printed)':>16} {'kit (printed)':>16} {'paper value':>22} {'kit value':>18}  ok")
    for r in rows:
        ok = r["printed_match"] and r["value_match"]
        print(f"{r['paper_macro']:26} {r['paper_printed']:>16} {r['kit_printed']:>16} {str(r['paper_value']):>22} "
              f"{str(r['kit_value']):>18}  {'yes' if ok else 'NO'}")
    for c in counts:
        print(f"  {c['aggregate']}: paper {c['paper']} kit {c['kit']} {'yes' if c['match'] else 'NO'}")
    matched = sum(r["printed_match"] and r["value_match"] for r in rows)
    cmatched = sum(c["match"] for c in counts)
    print(f"\nfigures matched {matched}/{len(rows)}; exact aggregates matched {cmatched}/{len(counts)}; "
          f"log {'identical' if log_sha == want else 'DIFFERENT'}")
    if log_sha != want or matched != len(rows) or cmatched != len(counts):
        print("BLOCKED: the kit does not reproduce the paper on these logs; nothing written")
        return 1

    kit_sha = kit.script_sha256()
    result_text = kit.render(res)
    write_json(args.out, {
        "schema_version": 1, "kind": "contrib_equivalence",
        "provenance": {"generated_at": utc_now(), "generator": GENERATOR, "sources": [
            {"id": "tokenledger_log", "grade": "measured", "sha256": log_sha,
             "command": "python tools/contrib_equivalence.py",
             "description": "per-request log rebuilt from the kit's records with --until 2026-11-15; identical to the "
                            "orchestrator snapshot's tokenledger_log (same sha256)"},
            {"id": "contrib_kit", "grade": "documented-record", "sha256": kit_sha,
             "command": "contrib/tokens_contrib.py",
             "description": "the contributor kit under test (the released version, contrib/KIT_SHA256.txt)"},
            {"id": "kit_result", "grade": "measured", "sha256": sha256_bytes(result_text.encode("utf-8")),
             "command": "python contrib/tokens_contrib.py --until 2026-11-15",
             "description": "the kit's tokens_contrib_result.json for the paper's window (aggregates.kit_result)"},
            {"id": "orchestrator_snapshot", "grade": "documented-record", "sha256": snap_sha,
             "command": "evidence/orchestrator_snapshot.json", "description": "paper figures compared (snapshot)"},
            {"id": "cache_writes", "grade": "documented-record", "sha256": cw_sha,
             "command": "evidence/cache_writes.json", "description": "paper figures compared (cache writes, causes)"},
            {"id": "share_robustness", "grade": "documented-record", "sha256": rb_sha,
             "command": "evidence/share_robustness.json", "description": "paper figure compared (months with requests)"},
        ], "notes": [
            "Local check (18 Nov 2026), not pre-registered: CI has no transcripts, so this file is its record. "
            "tests/test_contrib_kit.py runs the same comparison against the repo tools on synthetic logs in CI.",
            "Selection: the kit's --until filter at the paper's freeze (requests before 2026-11-15T00:00:00Z), "
            "applied per request before cross-file de-duplication, as tools/claude_usage.py applies the freeze.",
            "Shares: the kit rounds to four decimals; a figure matches when it is within 5e-5 points of the evidence "
            "value and prints identically at the paper's rounding. Counts and token totals must be equal.",
        ]},
        "aggregates": {"log_sha256_identical": True, "figures": rows, "exact_aggregates": counts,
                       "kit_result": json.loads(result_text)},
        "figures": {
            "ContribEquivFigures": figure(len(rows), "int", "orchestrator_snapshot",
                                          "paper figures the contributor kit was checked against on the author's logs"),
            "ContribEquivMatched": figure(matched, "int", "kit_result",
                                          "of those, figures the kit reproduced (printed rounding and within 5e-5 points)"),
            "ContribEquivCounts": figure(len(counts), "int", "kit_result",
                                         "token totals and counts compared exactly (all equal)"),
        },
    })
    print(f"wrote {args.out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
