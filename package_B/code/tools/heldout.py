"""Held-out test of context x turns versus prompt size: the analysis, as pre-registered.

The plan is papers/agent2027-cost/PREREG_heldout.md (committed before any per-unit value
was computed). This module is pure: it takes per-transcript units and returns aggregates.
tools/pull_evidence.py feeds it from the transcripts and writes evidence/heldout.json.

  unit     one transcript (main-loop session or one subagent run)
  T        requests;  C  peak per-request context;  P  first-request input
  y        list-price input cost (primary target);  I  total input tokens (secondary)
  split    earliest floor(0.7 n) by first-request time -> train; rest -> test
  models   A: ln y ~ ln(C*T)   B: ln y ~ ln(P)   S1: ln y ~ ln(T)   S2: ln y ~ ln(P*T)
  metrics  RMSE on ln y, MAPE (naive back-transform), R^2 on ln y about the test mean
  CI       paired bootstrap over test units, 2000 resamples, random.Random(0), positions 50/1949
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

MIN_REQUESTS = 3
TRAIN_FRACTION = 0.7
RESAMPLES = 2000
SEED = 0
CI_LOW_POS, CI_HIGH_POS = 50, 1949  # percentile 95% interval of RESAMPLES sorted values

FEATURES: dict[str, Callable[["Unit"], float]] = {
    "A": lambda u: u.C * u.T,
    "B": lambda u: u.P,
    "S1": lambda u: u.T,
    "S2": lambda u: u.P * u.T,
}
TARGETS: dict[str, Callable[["Unit"], float]] = {"cost": lambda u: u.y, "tokens": lambda u: u.I}


@dataclass
class Unit:
    first_ts: str   # ISO-8601 UTC of the first request
    order: int      # tie-breaker (transcript processing order)
    kind: str       # "main" | "subagent"
    T: int
    C: int
    P: int
    I: int
    y: float
    priced: bool = True


def eligible(units: list[Unit]) -> tuple[list[Unit], dict[str, int]]:
    """Exclusions E2-E4 in the pre-registered order (E1 is applied at extraction)."""
    counts = {"E2_fewer_than_3_requests": 0, "E3_unpriced_model": 0, "E4_zero_cost_or_prompt": 0}
    keep = []
    for u in units:
        if u.T < MIN_REQUESTS:
            counts["E2_fewer_than_3_requests"] += 1
        elif not u.priced:
            counts["E3_unpriced_model"] += 1
        elif u.y <= 0 or u.P <= 0 or u.I <= 0:
            counts["E4_zero_cost_or_prompt"] += 1
        else:
            keep.append(u)
    return keep, counts


def temporal_split(units: list[Unit]) -> tuple[list[Unit], list[Unit]]:
    ordered = sorted(units, key=lambda u: (u.first_ts, u.order))
    n_train = math.floor(TRAIN_FRACTION * len(ordered))
    return ordered[:n_train], ordered[n_train:]


def ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Intercept and slope of ys on xs."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        raise ValueError("feature has no variance in the training set")
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def _rmse_mape(log_y: list[float], log_pred: list[float]) -> tuple[float, float]:
    n = len(log_y)
    rmse = math.sqrt(sum((p - y) ** 2 for y, p in zip(log_y, log_pred)) / n)
    mape = 100 * sum(abs(math.exp(p) - math.exp(y)) / math.exp(y) for y, p in zip(log_y, log_pred)) / n
    return rmse, mape


def scores(log_y: list[float], log_pred: list[float]) -> dict[str, float]:
    rmse, mape = _rmse_mape(log_y, log_pred)
    mean = sum(log_y) / len(log_y)
    ss_tot = sum((y - mean) ** 2 for y in log_y)
    ss_res = sum((p - y) ** 2 for y, p in zip(log_y, log_pred))
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return {"rmse_log": rmse, "mape": mape, "r2_log": r2}


def _interval(values: list[float]) -> list[float]:
    s = sorted(values)
    return [s[CI_LOW_POS], s[CI_HIGH_POS]]


def paired_bootstrap(log_y: list[float], pred_a: list[float], pred_b: list[float]) -> dict[str, list[float]]:
    """95% percentile intervals of RMSE_A - RMSE_B and MAPE_A - MAPE_B over test units."""
    rng = random.Random(SEED)
    n = len(log_y)
    d_rmse, d_mape = [], []
    for _ in range(RESAMPLES):
        idx = [rng.randrange(n) for _ in range(n)]
        y = [log_y[i] for i in idx]
        ra, ma = _rmse_mape(y, [pred_a[i] for i in idx])
        rb, mb = _rmse_mape(y, [pred_b[i] for i in idx])
        d_rmse.append(ra - rb)
        d_mape.append(ma - mb)
    return {"d_rmse_log": _interval(d_rmse), "d_mape": _interval(d_mape)}


def decide(d_rmse: float, interval: list[float]) -> str:
    if d_rmse < 0 and interval[1] < 0:
        return "A predicts better"
    if d_rmse > 0 and interval[0] > 0:
        return "B predicts better"
    return "no detectable difference"


def evaluate(train: list[Unit], test: list[Unit], target: str) -> dict:
    """Fit every model on train, score on train and test, bootstrap A versus B on test."""
    get_y = TARGETS[target]
    ly_train = [math.log(get_y(u)) for u in train]
    ly_test = [math.log(get_y(u)) for u in test]
    out: dict = {"models": {}}
    preds: dict[str, list[float]] = {}
    for name, feat in FEATURES.items():
        a, b = ols([math.log(feat(u)) for u in train], ly_train)
        p_train = [a + b * math.log(feat(u)) for u in train]
        p_test = [a + b * math.log(feat(u)) for u in test]
        preds[name] = p_test
        out["models"][name] = {"intercept": a, "slope": b, "train": scores(ly_train, p_train),
                               "test": scores(ly_test, p_test)}
    ci = paired_bootstrap(ly_test, preds["A"], preds["B"])
    d_rmse = out["models"]["A"]["test"]["rmse_log"] - out["models"]["B"]["test"]["rmse_log"]
    d_mape = out["models"]["A"]["test"]["mape"] - out["models"]["B"]["test"]["mape"]
    out["a_minus_b"] = {"d_rmse_log": d_rmse, "d_rmse_log_ci95": ci["d_rmse_log"],
                        "d_mape": d_mape, "d_mape_ci95": ci["d_mape"]}
    out["decision"] = decide(d_rmse, ci["d_rmse_log"])
    out["_test_preds"] = preds  # dropped before writing evidence
    return out


def subset_scores(test: list[Unit], preds: dict[str, list[float]], kind: str, target: str) -> dict:
    """Test metrics for one kind of unit, using predictions from the full-test-set fit."""
    idx = [i for i, u in enumerate(test) if u.kind == kind]
    get_y = TARGETS[target]
    out = {"n": len(idx)}
    if len(idx) >= 2:
        ly = [math.log(get_y(test[i])) for i in idx]
        for name, p in preds.items():
            sub = [p[i] for i in idx]
            out[name] = scores(ly, sub)
    return out


def run(units: list[Unit]) -> dict:
    """The whole pre-registered analysis: exclusions, split, primary and secondary results."""
    keep, excluded = eligible(units)
    train, test = temporal_split(keep)
    primary = evaluate(train, test, "cost")
    secondary = evaluate(train, test, "tokens")
    subsets = {k: subset_scores(test, primary["_test_preds"], k, "cost") for k in ("main", "subagent")}
    for res in (primary, secondary):
        res.pop("_test_preds")
    return {
        "n_units": len(units), "excluded": excluded, "n_eligible": len(keep),
        "n_train": len(train), "n_test": len(test),
        "n_train_main": sum(u.kind == "main" for u in train),
        "n_test_main": sum(u.kind == "main" for u in test),
        "train_first_ts": train[0].first_ts if train else None,
        "test_first_ts": test[0].first_ts if test else None,
        "test_last_ts": test[-1].first_ts if test else None,
        "primary_cost": primary, "secondary_tokens": secondary, "test_subsets_cost": subsets,
    }


# ---------------------------------------------------------------- exploratory (NOT registered)
def exploratory(units: list[Unit]) -> dict:
    """Context for reading the registered result; added after an independent audit.

    None of this was pre-registered and none of it changes the registered decision:
      * const: test scores of predicting the training mean of ln cost (an intercept-only model);
      * ct_over_i: how close model A's feature is to total input tokens (Eq. 1's identity);
      * day-clustered paired bootstrap of RMSE_A - RMSE_B (test units resampled by start day,
        since conversations started close together are not independent).
    """
    keep, _ = eligible(units)
    train, test = temporal_split(keep)
    ly_train = [math.log(u.y) for u in train]
    ly_test = [math.log(u.y) for u in test]
    mean_train = sum(ly_train) / len(ly_train)
    const = scores(ly_test, [mean_train] * len(test))
    ratios = sorted(u.C * u.T / u.I for u in test)
    mid = len(ratios) // 2
    median_ratio = ratios[mid] if len(ratios) % 2 else (ratios[mid - 1] + ratios[mid]) / 2
    xs = [math.log(u.C * u.T) for u in test]
    zs = [math.log(u.I) for u in test]
    mx, mz = sum(xs) / len(xs), sum(zs) / len(zs)
    corr = (sum((x - mx) * (z - mz) for x, z in zip(xs, zs))
            / math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((z - mz) ** 2 for z in zs)))
    preds = {}
    for name in ("A", "B"):
        a, b = ols([math.log(FEATURES[name](u)) for u in train], ly_train)
        preds[name] = [a + b * math.log(FEATURES[name](u)) for u in test]
    days: dict[str, list[int]] = {}
    for i, u in enumerate(test):
        days.setdefault(u.first_ts[:10], []).append(i)
    keys = sorted(days)
    rng = random.Random(SEED)
    diffs = []
    for _ in range(RESAMPLES):
        idx = [i for _ in keys for i in days[keys[rng.randrange(len(keys))]]]
        y = [ly_test[i] for i in idx]
        ra, _ = _rmse_mape(y, [preds["A"][i] for i in idx])
        rb, _ = _rmse_mape(y, [preds["B"][i] for i in idx])
        diffs.append(ra - rb)
    return {"const_test": const, "ct_over_i_median_test": median_ratio, "corr_ln_ct_ln_i_test": corr,
            "test_days": len(keys), "d_rmse_log_day_clustered_ci95": _interval(diffs)}
