"""Per-arm count of failed and unconfirmed attempts and of what they cost (PREREG_v3 Amendment 4).

    python experiments/attempts_v3.py --input DIR [--input DIR ...] --out attempts.json

Reads requests.jsonl, requests_failed.jsonl and runs.jsonl of v3 grid dirs. Charged USD uses the
budget's own rule (agent.budget.charged_usd: a record's usd, plus the worst case of each attempt
that may have been billed without a response, and 2x the worst case when unpriced), so the totals
here equal what the spend chain counts. Records with role "unrecorded_charge" are spend-guard
charges that had no request record (a request that failed after possibly billed attempts).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.budget import charged_usd  # noqa: E402


def _read(path):
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def summarise(dirs) -> dict:
    per = defaultdict(lambda: {"ok_requests": 0, "failed_request_records": 0,
                               "unrecorded_charges": 0, "unconfirmed_attempts": 0,
                               "error_runs": 0, "charged_usd_ok": 0.0,
                               "charged_usd_failed": 0.0, "unpriced_records": 0})
    for d in dirs:
        for kind in ("requests.jsonl", "requests_failed.jsonl"):
            for rec in _read(os.path.join(d, kind)):
                a = per[rec.get("arm") or rec.get("policy")]
                usd = charged_usd(rec)
                a["unconfirmed_attempts"] += int(rec.get("unconfirmed_attempts") or 0)
                a["unpriced_records"] += usd is None
                if kind == "requests.jsonl":
                    a["ok_requests"] += rec.get("role") != "unrecorded_charge"
                    a["charged_usd_ok"] += usd or 0.0
                else:
                    if rec.get("role") == "unrecorded_charge":
                        a["unrecorded_charges"] += 1
                    else:
                        a["failed_request_records"] += 1
                    a["charged_usd_failed"] += usd or 0.0
        last: dict = {}
        for r in _read(os.path.join(d, "runs.jsonl")):
            last[r["run_id"]] = r
        for r in last.values():
            per[r.get("arm") or r.get("policy")]["error_runs"] += r.get("status") == "error"
    per = {k: {**v, "charged_usd_ok": round(v["charged_usd_ok"], 8),
               "charged_usd_failed": round(v["charged_usd_failed"], 8)}
           for k, v in sorted(per.items())}
    tot = {k: (round(sum(v[k] for v in per.values()), 8) if k.startswith("charged")
               else sum(v[k] for v in per.values()))
           for k in next(iter(per.values()), {})}
    return {"per_arm": per, "total": tot}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="experiments/attempts_v3.py")
    p.add_argument("--input", action="append", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = summarise(a.input)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2)
    t = out["total"]
    print(f"attempts: failed request records {t.get('failed_request_records', 0)}, unrecorded "
          f"charges {t.get('unrecorded_charges', 0)}, unconfirmed attempts "
          f"{t.get('unconfirmed_attempts', 0)}, error runs {t.get('error_runs', 0)}; charged USD "
          f"ok {t.get('charged_usd_ok', 0)} failed {t.get('charged_usd_failed', 0)} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
