#!/usr/bin/env python3
"""Practitioner rule, leave-one-month-out robustness and plain-wording figures (rev7).

Not pre-registered (19 Nov 2026). Combines committed evidence only -- no new measurement, no raw
data, so it runs anywhere:

  * evidence/params/anthropic_list_prices.json  write multipliers w and read multipliers rho;
  * evidence/cache_writes.json                  the keepalive replay's expiry re-writes and the
                                                one-hour payback (five-minute counterfactual);
  * evidence/share_robustness.json              re-written tokens and expiry re-writes by month;
  * evidence/agent_h2h_v3.json                  the registered H8 cost ratio (restated, rounded).

1. Rule. A cached prefix of P tokens re-written after expiry costs w*P (base-input units); a ping
   re-reads it for rho*P and restarts its lifetime L. Pinging every L until the return, plus the
   returning request's read, costs no more than one re-write while the gap is at most
   (w/rho - 1) lifetimes: the break-even horizon, independent of P. Beyond it, ending the session or
   compacting it first (which shrinks P) is cheaper. P sets only the stakes: the mean expiry
   re-write of an unchanged prefix (the keepalive replay's events) is reported in tokens and USD.
   Derived from this audit's prices and gaps; it is not validated, and it ignores a ping's own
   uncached and output tokens, as the replay does.
2. Leave-one-month-out. The expiry share of re-written tokens (CwCauseLongGapPct) recomputed with
   each calendar month dropped; the months must add up to the reported totals.
3. Plain wording. The one-hour payback as "would have cost N% more" and the H8 cost ratio at two
   decimals, restated from their sources for the abstract.

Writes evidence/practice_rule.json (aggregates only).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_lib import EVIDENCE, figure, sha256_bytes, utc_now, write_json  # noqa: E402

GENERATOR = "tools/practice_rule.py"
LONG_LIFETIME_H, SHORT_LIFETIME_MIN = 1.0, 5.0
# Khailo, "Keeping the Cache Warm Pays" (arXiv 2607.19214), abstract: the economical ping interval is
# "about 4 minutes at Anthropic's 5-minute TTL" and the break-even idle is "~tau(w/r - 1)".
KHAILO_PING_MIN = 4
KHAILO_QUOTE = ("about 4 minutes at Anthropic's 5-minute TTL rather than the 30-second convention, and the "
                "strategy breaks even against a re-prefill at idle ~tau(w/r - 1)")
TOL = 1e-6


def _load(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    return json.loads(raw.decode("utf-8")), raw


def horizons(prices: dict[str, dict[str, float]]) -> dict[str, float]:
    """Break-even horizons (w/rho - 1) lifetimes; the write multipliers must be one per tier."""
    w1 = {round(p["cache_write_1h"] / p["input"], 9) for p in prices.values()}
    w5 = {round(p["cache_write_5m"] / p["input"], 9) for p in prices.values()}
    if len(w1) != 1 or len(w5) != 1:
        sys.exit(f"BLOCKED: write multipliers differ across models (1h {sorted(w1)}, 5m {sorted(w5)})")
    rhos = [p["cache_hit"] / p["input"] for p in prices.values()]
    w_1h, w_5m, rho_max, rho_min = w1.pop(), w5.pop(), max(rhos), min(rhos)
    return {"w_1h": w_1h, "w_5m": w_5m, "rho_max": rho_max, "rho_min": rho_min,
            "long_hours_at_rho_max": (w_1h / rho_max - 1) * LONG_LIFETIME_H,
            "long_hours_at_rho_min": (w_1h / rho_min - 1) * LONG_LIFETIME_H,
            "short_minutes_at_rho_max": (w_5m / rho_max - 1) * SHORT_LIFETIME_MIN,
            "short_minutes_at_rho_min": (w_5m / rho_min - 1) * SHORT_LIFETIME_MIN,
            "long_breakeven_at_rho_max": (w_1h - 1) / (1 - rho_max),
            "long_breakeven_at_rho_min": (w_1h - 1) / (1 - rho_min)}


def leave_one_month_out(by_month: dict[str, dict[str, Any]], total_rewrite: float,
                        total_expiry: float) -> dict[str, Any]:
    rw = sum(m["rewrite"] for m in by_month.values())
    ex = sum(m["causes"]["gap_gt_60min"] for m in by_month.values())
    if abs(rw - total_rewrite) > TOL * max(1.0, total_rewrite) or abs(ex - total_expiry) > TOL * max(1.0, total_expiry):
        sys.exit(f"BLOCKED: months add up to {rw} re-written / {ex} expiry tokens, "
                 f"reported {total_rewrite} / {total_expiry}")
    left = {k: (ex - m["causes"]["gap_gt_60min"]) / (rw - m["rewrite"]) * 100 for k, m in by_month.items()}
    return {"by_month_left_out": left, "min_pct": min(left.values()), "max_pct": max(left.values()),
            "full_pct": ex / rw * 100}


def plain(ka: dict[str, Any], one_hour: dict[str, Any]) -> dict[str, float]:
    if not ka.get("events"):
        sys.exit("BLOCKED: the keepalive replay has no expiry events")
    return {"mean_rewrite_tokens": ka["rewrite_tokens"] / ka["events"],
            "mean_rewrite_usd": ka["rewrite_cost_usd"] / ka["events"],
            "five_minute_more_pct": (one_hour["cost_5m_counterfactual_usd"] / one_hour["cost_actual_usd"] - 1) * 100,
            "five_minute_extra_usd": one_hour["cost_5m_counterfactual_usd"] - one_hour["cost_actual_usd"]}


def build() -> dict[str, Any]:
    pr_d, pr_raw = _load(EVIDENCE / "params" / "anthropic_list_prices.json")
    cw_d, cw_raw = _load(EVIDENCE / "cache_writes.json")
    sr_d, sr_raw = _load(EVIDENCE / "share_robustness.json")
    h8_d, h8_raw = _load(EVIDENCE / "agent_h2h_v3.json")
    cw = cw_d["aggregates"]
    h = horizons(pr_d["models"])
    lomo = leave_one_month_out(sr_d["aggregates"]["by_month"], cw["rewrite"], cw["rewrite_by_cause"]["gap_gt_60min"])
    p = plain(cw["keepalive"], cw["one_hour"])
    ratio = h8_d["aggregates"]["H8"]["cost_ratio"]
    figs = {
        "RuleHorizonLongHours": figure(h["long_hours_at_rho_max"], "int", "list_prices",
                                       "rule: break-even horizon (hours) for hourly pings vs one re-write, one-hour "
                                       "writes, largest read multiplier: w/rho - 1"),
        "RuleHorizonLongMaxHours": figure(h["long_hours_at_rho_min"], "int", "list_prices",
                                          "rule: the same horizon at the smallest read multiplier (hours)"),
        "RuleHorizonShortMin": figure(h["short_minutes_at_rho_max"], "int", "list_prices",
                                      "rule: break-even horizon (minutes) for five-minute writes, largest read "
                                      "multiplier: (w/rho - 1) x 5 min"),
        "RuleHorizonShortMaxMin": figure(h["short_minutes_at_rho_min"], "int", "list_prices",
                                         "rule: five-minute-write horizon (minutes) at the smallest read multiplier"),
        "RuleBreakevenLongMin": figure(h["long_breakeven_at_rho_min"], "float2", "list_prices",
                                       "one-hour write break-even reads (w-1)/(1-rho) at the smallest read multiplier"),
        "RuleBreakevenLongMax": figure(h["long_breakeven_at_rho_max"], "float2", "list_prices",
                                       "one-hour write break-even reads (w-1)/(1-rho) at the largest read multiplier"),
        "RulePingsPerRewriteMin": figure(h["w_1h"] / h["rho_max"], "int", "list_prices",
                                         "prefix reads costing as much as one one-hour re-write: w/rho, largest rho"),
        "RulePingsPerRewriteMax": figure(h["w_1h"] / h["rho_min"], "int", "list_prices",
                                         "prefix reads costing as much as one one-hour re-write: w/rho, smallest rho"),
        "KhailoPingMin": figure(KHAILO_PING_MIN, "int", "khailo_abstract",
                                "Khailo (arXiv 2607.19214): ping interval at a five-minute TTL, minutes (abstract)"),
        "RuleHorizonKhailoMin": figure(KHAILO_PING_MIN * (h["w_5m"] / h["rho_max"] - 1), "int", "khailo_abstract",
                                       "Khailo's break-even idle tau(w/r - 1) at his ping interval, w and rho as here"),
        "IdleNetDropPts": figure((cw["rewrite_by_cause"]["gap_gt_60min"] - cw["rewrite_gt_60min_net_of_base_rate"])
                                 / cw["rewrite"] * 100, "float2", "cache_writes",
                                 "percentage points the idle share of re-written tokens falls net of the base rate"),
        "PlainFiveMinExtraUsd": figure(p["five_minute_extra_usd"], "int", "cache_writes",
                                       "five-minute counterfactual minus actual cost of the one-hour-written blocks "
                                       "(their writes and same-transcript reads), list-price USD, all transcripts"),
        "PlainFiveMinExtraOfTotalPct": figure(p["five_minute_extra_usd"] / cw["total_cost_usd"] * 100, "pct0",
                                              "cache_writes", "the same extra cost as a share of total list-price cost"),
        "TotalListCostUsd": figure(cw["total_cost_usd"], "int", "cache_writes",
                                   "API-equivalent total cost of the frozen orchestrator requests, USD at list price"),
        "RuleMeanRewriteKTok": figure(p["mean_rewrite_tokens"], "kilo", "cache_writes",
                                      "rule: mean tokens per expiry re-write of an unchanged prefix (keepalive events)"),
        "RuleMeanRewriteUsd": figure(p["mean_rewrite_usd"], "float2", "cache_writes",
                                     "rule: mean list-price USD per expiry re-write of an unchanged prefix"),
        "RbLomoExpiryMinPct": figure(lomo["min_pct"], "pct0", "share_robustness",
                                     "expiry share of re-written tokens, leaving out one month: smallest"),
        "RbLomoExpiryMaxPct": figure(lomo["max_pct"], "pct0", "share_robustness",
                                     "expiry share of re-written tokens, leaving out one month: largest"),
        "PlainFiveMinMorePct": figure(p["five_minute_more_pct"], "pct0", "cache_writes",
                                      "one-hour blocks: five-minute counterfactual cost over actual, percent more "
                                      "(CwLongFiveMinDeltaPct, unsigned, rounded)"),
        "PlainCostRatio": figure(ratio, "float2", "h2h3_test_r1_s0",
                                 "H8 cost(aob) / cost(B*), two decimals (HthreeCostRatio, rounded)"),
    }
    return {
        "schema_version": 1, "kind": "practice_rule",
        "provenance": {"generated_at": utc_now(), "generator": GENERATOR, "sources": [
            {"id": "list_prices", "grade": "model-parameter", "sha256": sha256_bytes(pr_raw),
             "command": "evidence/params/anthropic_list_prices.json", "description": "write and read multipliers"},
            {"id": "cache_writes", "grade": "measured", "sha256": sha256_bytes(cw_raw),
             "command": "evidence/cache_writes.json", "description": "keepalive replay and one-hour payback"},
            {"id": "share_robustness", "grade": "measured", "sha256": sha256_bytes(sr_raw),
             "command": "evidence/share_robustness.json", "description": "re-writes and expiry by calendar month"},
            {"id": "h2h3_test_r1_s0", "grade": "measured", "sha256": sha256_bytes(h8_raw),
             "command": "evidence/agent_h2h_v3.json", "description": "registered H8 cost ratio (restated)"},
            {"id": "khailo_abstract", "grade": "documented-record", "sha256": sha256_bytes(KHAILO_QUOTE.encode("utf-8")),
             "command": "https://arxiv.org/abs/2607.19214 (abstract, fetched 19 Nov 2026)",
             "description": "Khailo's ping interval and break-even formula; sha256 of the quoted sentence"}],
            "notes": ["Not pre-registered; combines committed evidence only (no new measurement).",
                      "The rule is derived from this audit's prices and gaps, not validated."]},
        "aggregates": {"horizons": h, "leave_one_month_out": lomo, "plain": p, "h8_cost_ratio": ratio},
        "figures": figs,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=EVIDENCE / "practice_rule.json")
    args = ap.parse_args()
    write_json(args.out, build())
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
