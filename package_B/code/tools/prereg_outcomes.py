#!/usr/bin/env python3
"""Counts restated by the paper's registered-outcomes table (AGENT 2027 paper, Table II).

Combines committed files only -- no new measurement:

  * evidence/gateway_live.json   the pre-registered live check's hypotheses and their verdicts
                                 (algorithm/benchmarks/gateway_live/analyze.py).

It counts the live check's hypotheses and how many were supported, so the table can state the
outcome in one cell instead of five; each verdict must be present, or it is BLOCKED.
Writes evidence/prereg_outcomes.json (aggregates only).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_lib import EVIDENCE, figure, sha256_bytes, utc_now, write_json  # noqa: E402

GENERATOR = "tools/prereg_outcomes.py"


def blocked(msg: str) -> None:
    sys.exit(f"BLOCKED: {msg}")


def derive(live: dict[str, Any]) -> dict[str, Any]:
    hyps = live.get("hypotheses") or {}
    if not hyps:
        blocked("gateway_live.json has no hypotheses")
    verdicts = {}
    for name, h in sorted(hyps.items()):
        v = h.get("verdict")
        if not isinstance(v, str) or not v.strip():
            blocked(f"live-check hypothesis {name} has no verdict")
        verdicts[name] = v.strip().upper()
    return {"live_verdicts": verdicts, "live_n": len(verdicts),
            "live_supported": sum(v == "SUPPORTED" for v in verdicts.values())}


def build() -> dict[str, Any]:
    raw = (EVIDENCE / "gateway_live.json").read_bytes().replace(b"\r\n", b"\n")
    f = derive(json.loads(raw.decode("utf-8"))["aggregates"])
    src = "gateway_live"
    return {
        "schema_version": 1, "kind": "prereg_outcomes",
        "provenance": {"generated_at": utc_now(), "generator": GENERATOR, "sources": [
            {"id": src, "grade": "measured", "sha256": sha256_bytes(raw),
             "command": "evidence/gateway_live.json", "description": "pre-registered live check of the second harness"}],
            "notes": ["Restates committed live-check verdicts as counts; no new measurement."]},
        "aggregates": f,
        "figures": {
            "PregLiveN": figure(f["live_n"], "word", src, "pre-registered live-check hypotheses"),
            "PregLiveSupportedN": figure(f["live_supported"], "word", src, "live-check hypotheses supported"),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=EVIDENCE / "prereg_outcomes.json")
    args = ap.parse_args()
    write_json(args.out, build())
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
