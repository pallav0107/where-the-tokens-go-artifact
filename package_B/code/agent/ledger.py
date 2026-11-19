"""Request logging (JSONL, one line per LLM request) and the tokenledger view."""
from __future__ import annotations

import json


class RequestLogger:
    def __init__(self, path):
        self.path = str(path)
        self._fh = open(self.path, "a", encoding="utf-8")

    def log(self, record):
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()  # flush per line: a crash must not lose requests

    def close(self):
        if not self._fh.closed:
            self._fh.close()


def to_tokenledger_rows(records) -> list[dict]:
    """One tokenledger "step" per request, grouped into one "turn" per run.

    priming_tokens = the run's step-0 input; conversation_tokens = x_t - priming.
    cache_read_tokens appears only when the provider reported it (never 0-for-null).
    """
    by_run: dict = {}
    for rec in records:
        by_run.setdefault(rec["run_id"], []).append(rec)
    rows = []
    for run_id, recs in by_run.items():
        recs = sorted(recs, key=lambda r: r["step"])
        priming = recs[0]["input_tokens"]
        for r in recs:
            row = {
                "turn": r["turn"],
                "step": r["step"],
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "priming_tokens": priming,
                "conversation_tokens": r["input_tokens"] - priming,
                "label": str(run_id),
            }
            if r.get("cached_input_tokens") is not None:
                row["cache_read_tokens"] = r["cached_input_tokens"]
            rows.append(row)
    return rows
