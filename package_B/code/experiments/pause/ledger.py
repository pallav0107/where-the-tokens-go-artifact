"""Run ledger and budget stop for the pause experiment (PREREG_pause.md, Budget).

Cost of a run = its transcripts' per-request usage (tools/claude_usage.py: one record per API
request) at the repo's list prices (evidence/params/anthropic_list_prices.json). A model missing
from the sheet is charged at the dearest rate of each token kind, never at zero. The ledger keeps
the CLI's own reported cost beside it (it also sees requests the transcripts omit) and the budget
counts the larger of the two.

Billing ground truth: each run's provider request ids, first/last request time and list-price
cost per UTC hour and day are kept, so the ledger can be reconciled with the console's billed usage.

Budget stop: never start a run if spent + max per-run cost x concurrency would exceed the stop
(USD 140 under a USD 150 cap). The max per-run cost is the larger of the observed maximum and a
prior (a floor set from the pilot), so one cheap early run cannot loosen the stop.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import claude_usage  # noqa: E402
from cache_writes import price_for  # noqa: E402

STOP_USD = 140.0
KINDS = ("input", "cache_hit", "cache_write_5m", "cache_write_1h", "output")


def dearest(prices: dict[str, dict[str, float]]) -> dict[str, float]:
    return {k: max(p[k] for p in prices.values()) for k in KINDS}


def usage_cost(records: Iterable[dict[str, Any]], prices: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Tokens by kind and list-price USD over request records ({"model", "usage"})."""
    tok = {"fresh": 0, "cache_read": 0, "cache_write_5m": 0, "cache_write_1h": 0, "output": 0}
    usd, n, unpriced = 0.0, 0, 0
    by_hour: dict[str, float] = {}
    by_day: dict[str, float] = {}
    worst = dearest(prices)
    for r in records:
        u = r["usage"]
        cc = u.get("cache_creation") or {}
        row = {"fresh": u.get("input_tokens", 0), "cache_read": u.get("cache_read_input_tokens", 0),
               "cache_write_5m": cc.get("ephemeral_5m_input_tokens", 0),
               "cache_write_1h": cc.get("ephemeral_1h_input_tokens", 0), "output": u.get("output_tokens", 0)}
        p = price_for(r["model"], prices)
        if p is None:
            unpriced += 1
            p = worst
        c = (row["fresh"] * p["input"] + row["cache_read"] * p["cache_hit"] + row["cache_write_5m"] * p["cache_write_5m"]
             + row["cache_write_1h"] * p["cache_write_1h"] + row["output"] * p["output"]) / 1e6
        usd += c
        ts = r.get("ts")
        if isinstance(ts, str) and len(ts) >= 13:     # UTC hour and day of the request (billing reconciliation)
            by_hour[ts[:13]] = by_hour.get(ts[:13], 0.0) + c
            by_day[ts[:10]] = by_day.get(ts[:10], 0.0) + c
        for k, v in row.items():
            tok[k] += v
        n += 1
    return {"requests": n, "tokens": tok, "usd": usd, "unpriced_requests": unpriced,
            "usd_by_hour": by_hour, "usd_by_day": by_day}


def transcript_cost(projects: Path, prices: dict[str, dict[str, float]]) -> dict[str, Any]:
    """usage_cost over every transcript under a run's projects/ tree (de-duplicated requests)."""
    transcripts, _ = claude_usage.extract(projects)
    out = usage_cost(claude_usage.records(transcripts), prices)
    reqs = sorted((r for t in transcripts for r in t.requests), key=lambda r: r.ts)
    out["models"] = sorted({r.model for r in reqs})
    out["request_ids"] = [r.msg_id for r in reqs]      # provider message ids, for billing reconciliation
    out["first_request"] = reqs[0].ts if reqs else None
    out["last_request"] = reqs[-1].ts if reqs else None
    out["transcripts"] = len(transcripts)
    return out


def may_start(spent: float, max_run_cost: float, concurrency: int, stop: float = STOP_USD) -> bool:
    """The dispatcher's gate: spent + max per-run cost x concurrency must stay within the stop."""
    return spent + max_run_cost * concurrency <= stop + 1e-9


class Ledger:
    """Append-only JSONL ledger, one row per finished run (run id, task, condition, times, tokens, USD)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(x) for x in self.path.read_text(encoding="utf-8").splitlines() if x.strip()]

    def append(self, row: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    @staticmethod
    def charged(row: dict[str, Any]) -> float:
        return max(float(row.get("usd") or 0.0), float(row.get("cli_usd") or 0.0))

    def spent(self) -> float:
        return sum(self.charged(r) for r in self.rows())

    def by_period(self) -> dict[str, dict[str, float]]:
        """List-price USD per UTC hour and per UTC day, summed over runs (to reconcile with billed usage)."""
        out: dict[str, dict[str, float]] = {"hour": {}, "day": {}}
        for r in self.rows():
            for k, key in (("hour", "usd_by_hour"), ("day", "usd_by_day")):
                for period, v in (r.get(key) or {}).items():
                    out[k][period] = round(out[k].get(period, 0.0) + v, 9)
        return {k: dict(sorted(v.items())) for k, v in out.items()}

    def max_run_cost(self, prior: float) -> float:
        return max([prior] + [self.charged(r) for r in self.rows()])
