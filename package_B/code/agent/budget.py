"""Cloud pricing and the budget guard (PREREG sections 3 and 6).

Prices live in agent/prices.yaml (USD per 1M tokens) with source URLs and
retrieval dates. Every entry is verified by hand before its `verified` flag is
flipped; until then every cloud run is refused.
"""
from __future__ import annotations

import fnmatch
import json
import os
import threading

import yaml

_PRICES_PATH = os.path.join(os.path.dirname(__file__), "prices.yaml")
_cache = None


class BudgetExceeded(Exception):
    """Raised when a run is refused: over budget, unverified or unknown price."""


def _prices() -> dict:
    global _cache
    if _cache is None:
        with open(_PRICES_PATH, encoding="utf-8") as fh:
            _cache = yaml.safe_load(fh)
    return _cache


def _entry(provider: str, model: str) -> dict | None:
    prov = (_prices().get("providers") or {}).get(provider) or {}
    return (prov.get("models") or {}).get(model)


def _max_input_rate(e: dict) -> float:
    """Highest per-token input rate of an entry: uncached input or, where the provider bills
    cache writes above the input rate (OpenAI GPT-5.6+, PREREG_v2 Amendment 2 B2), the write."""
    return max(float(e["input"]), float(e.get("cache_write") or 0.0))


def estimate_usd(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    """Cell estimate at the FULL UNCACHED input price (conservative; the cache-write price
    where that is higher)."""
    e = _entry(provider, model)
    if e is None:
        raise BudgetExceeded(f"no price entry for {provider}/{model}")
    return (input_tokens * _max_input_rate(e) + output_tokens * e["output"]) / 1_000_000


def request_usd(provider: str, model: str, usage) -> float | None:
    """Actual cost of one request; None (unpriced, which stops a block) when the price entry
    is unknown, the response was served on a service tier the entry does not price, cache
    writes are reported without a write price, or the token counts are inconsistent."""
    e = _entry(provider, model)
    if e is None:
        return None
    tiers = e.get("service_tiers")
    if tiers is not None and (getattr(usage, "raw", None) or {}).get("service_tier") not in tiers:
        return None
    cached = usage.cached_input_tokens or 0
    written = getattr(usage, "cache_write_tokens", None) or 0
    if written and e.get("cache_write") is None:
        return None
    uncached = usage.input_tokens - cached - written
    if uncached < 0:
        return None
    return (uncached * e["input"]
            + cached * e.get("cached_input", e["input"])
            + written * (e.get("cache_write") or 0.0)
            + usage.output_tokens * e["output"]) / 1_000_000


def input_token_upper_bound(messages) -> int:
    """Upper bound on the prompt tokens of a chat request: byte-level BPE tokens carry at
    least one byte each, so UTF-8 bytes of every message, plus 16 tokens per message for the
    chat template and a fixed 1,024 for any provider-side preamble (PREREG_v2 Amendment 2 B7).
    Each response's prompt_tokens is checked against it (SpendGuard.charge)."""
    n_bytes = sum(len(str(m.get("content") or "").encode("utf-8")) for m in messages)
    return n_bytes + 16 * len(messages) + 1024


# An unpriced response (e.g. served on a non-default tier) is charged at this multiple of
# its standard-tier worst case: the pricing page's Fast tier is 2x standard (Flex is 0.5x).
UNPRICED_FACTOR = 2.0


def charged_usd(rec: dict) -> float | None:
    """What a request record counts against the budget: its `usd`, plus the worst case of any
    attempt that may have been billed without a response (`unconfirmed_attempts`); for an
    unpriced record (usd None) UNPRICED_FACTOR x its worst case (`usd_worst`), when known.
    None only when the record is unpriced and carries no worst case (pre-OpenAI providers)."""
    usd, worst = rec.get("usd"), rec.get("usd_worst")
    extra = (rec.get("unconfirmed_attempts") or 0) * (worst or 0.0)
    if usd is None:
        return None if worst is None else UNPRICED_FACTOR * worst + extra
    return float(usd) + extra


class SpendGuard:
    """Per-request hard cap (PREREG_v2 Amendment 2, B7).

    admit() refuses a request when spend so far (prior + every charged response + the worst
    case of requests in flight) plus its own worst case could exceed max_usd; charge() adds
    the response's actual USD. Fail closed: an unpriced response is charged at its worst case
    and, like a prompt larger than the bound, stops every later request."""

    def __init__(self, provider: str, model: str, max_usd: float, spent: float = 0.0):
        ensure_price_verified(provider, model)
        self.provider, self.model = provider, model
        self.max_usd, self.spent = float(max_usd), float(spent)
        self.reserved = 0.0
        self.stopped: str | None = None
        self._lock = threading.Lock()

    def worst_case_usd(self, input_upper: int, max_output_tokens: int) -> float:
        e = _entry(self.provider, self.model)
        return (input_upper * _max_input_rate(e) + max_output_tokens * e["output"]) / 1_000_000

    def admit(self, input_upper: int, max_output_tokens: int) -> float:
        """Reserve the worst case of one request (returned) or raise BudgetExceeded."""
        e = _entry(self.provider, self.model)
        with self._lock:
            if self.stopped:
                raise BudgetExceeded(f"spend guard stopped: {self.stopped}")
            cap = e.get("max_input_tokens")
            if cap is not None and input_upper > cap:
                raise BudgetExceeded(f"prompt bound {input_upper} tokens exceeds the "
                                     f"{cap}-token short-context price tier")
            worst = self.worst_case_usd(input_upper, max_output_tokens)
            if self.spent + self.reserved + worst > self.max_usd:
                raise BudgetExceeded(
                    f"request refused: spend {self.spent:.6f} + in flight {self.reserved:.6f}"
                    f" + worst case {worst:.6f} USD could exceed the {self.max_usd:.2f} USD "
                    f"budget")
            self.reserved += worst
            return worst

    def release(self, worst: float) -> None:
        """The request got no response (an HTTP error is not billed)."""
        with self._lock:
            self.reserved = max(self.reserved - worst, 0.0)

    def charge(self, usd: float | None, worst: float, prompt_tokens: int | None = None,
               input_upper: int | None = None) -> float:
        """Add one response's actual cost; returns cumulative spend (cum_usd)."""
        with self._lock:
            self.reserved = max(self.reserved - worst, 0.0)
            if usd is None:
                self.spent += UNPRICED_FACTOR * worst
                self.stopped = (f"unpriced response (charged at {UNPRICED_FACTOR:g}x its "
                                f"standard worst case)")
            else:
                self.spent += float(usd)
            if (prompt_tokens is not None and input_upper is not None
                    and prompt_tokens > input_upper):
                self.stopped = (f"prompt_tokens {prompt_tokens} exceeded the worst-case bound "
                                f"{input_upper}")
            return round(self.spent, 6)

    def charge_unconfirmed(self, usd: float) -> None:
        """A request that may have been billed without a response (timeout after sending)."""
        with self._lock:
            self.spent += float(usd)


def check_budget(estimate: float, max_usd: float, spent: float = 0.0) -> None:
    if spent + estimate > max_usd:
        raise BudgetExceeded(
            f"spend {spent + estimate:.4f} USD (including estimate) would exceed the "
            f"{max_usd:.2f} USD budget"
        )


def ensure_price_verified(provider: str, model: str) -> None:
    """Cloud runs are refused until their price entry exists and is verified."""
    e = _entry(provider, model)
    if e is None:
        raise BudgetExceeded(f"no price entry for {provider}/{model}")
    if not e.get("verified"):
        raise BudgetExceeded(f"price entry for {provider}/{model} is not verified")


def _sum_usd(path: str) -> tuple[float, int]:
    """(sum of charged USD, number of unpriced records) over one request JSONL file. A record
    counts as charged_usd(): an unpriced record with a worst case (OpenAI) counts at
    UNPRICED_FACTOR x that worst case, never as 0; only a record with neither stays at 0 and
    is counted as unpriced."""
    total, unpriced = 0.0, 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("usd") is None:
                unpriced += 1
            usd = charged_usd(rec)
            if usd is not None:
                total += usd
    return total, unpriced


def collect_prior_spend(dirs, extra_usd: float = 0.0) -> dict:
    """Spend already incurred: `usd` over every requests.jsonl / requests_failed.jsonl /
    spend_ledger*.jsonl (the committed off-chain ledger, PREREG_v2 Amendment 2 B6) found
    recursively under dirs, plus `est_usd` of every embedding_log*.json, plus extra_usd.
    Each physical file is counted once even when reachable through several dirs."""
    seen: set = set()
    requests: dict = {}
    logs: dict = {}
    unpriced = 0
    for d in dirs:
        for root, _dirs, names in os.walk(str(d)):
            for name in sorted(names):
                is_req = (name in ("requests.jsonl", "requests_failed.jsonl")
                          or fnmatch.fnmatch(name, "spend_ledger*.jsonl"))
                is_log = fnmatch.fnmatch(name, "embedding_log*.json")
                if not (is_req or is_log):
                    continue
                path = os.path.join(root, name)
                real = os.path.realpath(path)
                if real in seen:
                    continue
                seen.add(real)
                label = path.replace(os.sep, "/")
                if is_req:
                    usd, n_none = _sum_usd(path)
                    requests[label] = round(usd, 6)
                    unpriced += n_none
                else:
                    with open(path, encoding="utf-8") as fh:
                        est = json.load(fh).get("est_usd")
                    if est is None:  # fail closed: an unpriced embedding spend is unknown
                        raise BudgetExceeded(f"{path} has no est_usd (unpriced embedding model)")
                    logs[label] = round(float(est), 6)
    total = sum(requests.values()) + sum(logs.values()) + float(extra_usd or 0.0)
    return {"requests": requests, "embedding_logs": logs, "flag_usd": float(extra_usd or 0.0),
            "n_unpriced_requests": unpriced, "total": round(total, 6)}


def _main(argv=None) -> int:
    """python -m agent.budget remaining --max-usd N [--prior-spend-usd X] [--spend-from DIR]...
    prints the remaining budget (never negative)."""
    import argparse
    ap = argparse.ArgumentParser(prog="agent.budget")
    ap.add_argument("cmd", choices=["remaining"])
    ap.add_argument("--max-usd", type=float, required=True)
    ap.add_argument("--prior-spend-usd", type=float, default=0.0)
    ap.add_argument("--spend-from", action="append", default=[])
    a = ap.parse_args(argv)
    info = collect_prior_spend(a.spend_from, a.prior_spend_usd)
    print(f"{max(a.max_usd - info['total'], 0.0):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
