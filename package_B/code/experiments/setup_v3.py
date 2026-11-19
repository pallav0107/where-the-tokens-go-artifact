"""PREREG_v3 test setup figures: the model settings and price sheet every v3 arm ran with.

    python experiments/setup_v3.py   ->  evidence/agent_h2h_v3_setup.json

Reads the registered run configuration (experiments/configs/h2h_v3.yaml) and the price sheet the
spend guard and the cost analysis used (agent/prices.yaml), records the sha256 of each, and emits
figures prefixed "Vthree" so the paper states them from evidence. It reads no result and changes
no registered file.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from evidence_lib import EVIDENCE, figure, sha256_bytes, utc_now, write_json  # noqa: E402

CONFIG = ROOT / "experiments" / "configs" / "h2h_v3.yaml"
PRICES = ROOT / "agent" / "prices.yaml"
H2H = EVIDENCE / "agent_h2h_v3.json"
FREEZE = "e274b76"   # the registered v3 freeze (PREREG_v3 Amendment 4); Amendment 5 changed only the budget stop
GENERATOR = "experiments/setup_v3.py"


def setup(cfg: dict[str, Any], prices: dict[str, Any]) -> dict[str, Any]:
    models = cfg.get("models") or []
    if len(models) != 1:
        sys.exit(f"BLOCKED: expected exactly one registered model, found {len(models)}")
    m = models[0]
    entry = (((prices.get("providers") or {}).get(m["provider"]) or {}).get("models") or {}).get(m["model"])
    if not entry or entry.get("verified") is not True:
        sys.exit(f"BLOCKED: no verified price entry for {m['provider']}/{m['model']}")
    return {"provider": m["provider"], "model": m["model"], "temperature": cfg["agent"]["temperature"],
            "seed": cfg["agent"]["seed"], "reasoning_effort": m["reasoning_effort"],
            "max_output_tokens": m["max_output_tokens"], "price_input": entry["input"],
            "price_cached_input": entry["cached_input"], "price_cache_write": entry["cache_write"],
            "price_output": entry["output"], "price_retrieved": entry["retrieved"]}


def figures(s: dict[str, Any]) -> dict[str, Any]:
    return {
        "VthreeSetupTemperature": figure(s["temperature"], "int", "config", "Sampling temperature of every v3 arm."),
        "VthreeSetupSeed": figure(s["seed"], "int", "config", "Sampling seed of every v3 arm."),
        "VthreeSetupReasoning": figure(s["reasoning_effort"], "text", "config", "Reasoning effort of every v3 arm."),
        "VthreeSetupMaxOutput": figure(s["max_output_tokens"], "int", "config", "Output-token cap per request."),
        "VthreePriceInput": figure(s["price_input"], "float2", "prices", "Input price, USD per million tokens."),
        "VthreePriceCached": figure(s["price_cached_input"], "float2", "prices",
                                    "Cached-input (cache read) price, USD per million tokens."),
        "VthreePriceWrite": figure(s["price_cache_write"], "float3", "prices",
                                   "Cache-write price, USD per million tokens."),
        "VthreePriceOutput": figure(s["price_output"], "float2", "prices", "Output price, USD per million tokens."),
        "VthreePriceRetrieved": figure(s["price_retrieved"], "date", "prices", "Price sheet retrieval date."),
    }


def budget(cfg: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    """Amendment 5 (budget only): the spend stop at the freeze and in the run configuration."""
    reg, amended = float(frozen["budget"]["max_usd"]), float(cfg["budget"]["max_usd"])
    if not amended > reg:
        sys.exit(f"BLOCKED: the budget stop was not raised ({reg} -> {amended})")
    return {"VthreeBudgetRegistered": figure(reg, "float2", "config_frozen",
                                             "v3 spend stop at the freeze, USD project total."),
            "VthreeBudgetAmended": figure(amended, "float2", "config",
                                          "v3 spend stop after Amendment 5, USD project total.")}


def reporting(h2h: dict[str, Any]) -> dict[str, Any]:
    """The registered F1 difference (AoB minus B*) stated as a shortfall, so the text needs no
    double negative ("2.34 points lower"); only formats a number evidence/agent_h2h_v3.json holds."""
    diff = h2h["figures"]["HthreeFOneDiff"]["value"]
    if not diff < 0:
        sys.exit(f"BLOCKED: HthreeFOneDiff is {diff}; the shortfall wording applies only to a negative difference")
    return {"VthreeFOneLowerBy": figure(-diff, "float2", "h2h_v3",
                                        "H8: how many F1 points AoB was below B* (minus HthreeFOneDiff).")}


def main() -> int:
    import json
    import subprocess
    import yaml
    craw, praw = CONFIG.read_bytes().replace(b"\r\n", b"\n"), PRICES.read_bytes().replace(b"\r\n", b"\n")
    hraw = H2H.read_bytes().replace(b"\r\n", b"\n")
    fraw = subprocess.run(["git", "show", f"{FREEZE}:experiments/configs/h2h_v3.yaml"], cwd=ROOT, check=True,
                          capture_output=True).stdout.replace(b"\r\n", b"\n")
    s = setup(yaml.safe_load(craw.decode("utf-8")), yaml.safe_load(praw.decode("utf-8")))
    bud = budget(yaml.safe_load(craw.decode("utf-8")), yaml.safe_load(fraw.decode("utf-8")))
    write_json(EVIDENCE / "agent_h2h_v3_setup.json", {
        "schema_version": 1, "kind": "agent_h2h_v3_setup",
        "provenance": {"generated_at": utc_now(), "generator": GENERATOR, "sources": [
            {"id": "config", "grade": "documented-record", "sha256": sha256_bytes(craw),
             "command": "experiments/configs/h2h_v3.yaml",
             "description": "registered v3 run configuration (PREREG_v3; agent settings and model)"},
            {"id": "prices", "grade": "model-parameter", "sha256": sha256_bytes(praw), "command": "agent/prices.yaml",
             "description": "price sheet used by the spend guard and the billed-cost analysis (provider pricing page)"},
            {"id": "config_frozen", "grade": "documented-record", "sha256": sha256_bytes(fraw),
             "command": f"git show {FREEZE}:experiments/configs/h2h_v3.yaml",
             "description": "the v3 run configuration at the registered freeze (budget stop before Amendment 5)"},
            {"id": "h2h_v3", "grade": "measured", "sha256": sha256_bytes(hraw), "command": "evidence/agent_h2h_v3.json",
             "description": "the registered H8 analysis (figure HthreeFOneDiff), restated as a shortfall"},
        ]},
        "aggregates": {"setup": s},
        "figures": {**figures(s), **bud, **reporting(json.loads(hraw.decode("utf-8")))},
    })
    print(f"wrote agent_h2h_v3_setup.json: {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
