"""Pre-registered analysis (PREREG.md sections 6-10) for the agent-runs experiment.

Pure numpy + PyYAML + tokenledger-llm: OLS, fixed-effects demeaning and the
cluster bootstrap are implemented here. The cluster bootstrap resamples
QUESTIONS with replacement (all runs of a question move together) and reports
95% percentile intervals, per PREREG section 7. CIs whose statistic was valid
in fewer than 95% of the requested replicates are marked "unstable" and any
decision resting on them becomes "not supported (unstable CI)".

Records are runs.jsonl / requests.jsonl as written by agent/run_grid.py.
Requests' `step` is 0-based, so PREREG's step t (1-based) is step t-1 here.

Output envelope (evidence pipeline contract):
    {"schema_version": 1, "kind": "agent_runs", "provenance": {...},
     "aggregates": {...}, "figures": {...}}
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tokenledger  # noqa: E402
from agent.ledger import to_tokenledger_rows  # noqa: E402

BOOT_SEED = 7     # PREREG section 7
PERM_SEED = 11    # PREREG section 8, control 3
CONTROL_POLICIES = ("broken_reset", "smoke")
H1_MODELS = ("M_ctx", "M_prompt", "M_cell", "M_carry")
PREREG_PATH = ROOT / "experiments" / "PREREG.md"

# Design constants (PREREG sections 2-4), reported as figures sourced "prereg".
N_QUESTIONS = 48
TURN_CAP = 10
K_LOW, K_HIGH = 2, 8
N_TEST_QUESTIONS = 16  # p mod 3 == 2 over p = 0..47

FIG_FORMATS = frozenset(
    "int pct0 pct1 pct2 pct3 signedpct1 float1 float2 float3 "
    "millions1 billions1 kilo word date text".split())


# ---------------------------------------------------------------------------
# Request-log hygiene (review item 1)
# ---------------------------------------------------------------------------

def final_attempt_requests(requests):
    """Keep only the records of the final attempt per run_id (`attempt`, default 1)."""
    best = {}
    for q in requests:
        rid = q.get("run_id")
        att = q.get("attempt", 1)
        if att is None:
            att = 1
        if rid not in best or att > best[rid]:
            best[rid] = att
    return [q for q in requests
            if q.get("attempt", 1) == best.get(q.get("run_id"))]


def request_log_mismatch_ids(runs, requests):
    """Run ids of analysable runs whose request steps != range(turns)."""
    steps = {}
    for q in final_attempt_requests(requests):
        steps.setdefault(q.get("run_id"), set()).add(int(q["step"]))
    bad = set()
    for r in analysable(runs):
        if steps.get(r["run_id"], set()) != set(range(int(r.get("turns") or 0))):
            bad.add(r["run_id"])
    return bad


# ---------------------------------------------------------------------------
# Data filters (PREREG section 9; review item 7)
# ---------------------------------------------------------------------------

def analysable(runs):
    """Runs that may enter H1-H5: ok, not truncated, not a control run."""
    return [r for r in runs
            if r.get("status") == "ok" and not r.get("truncated")
            and r.get("control") not in CONTROL_POLICIES]


def exclusions(runs, request_mismatch_ids=None):
    """Excluded runs, split into analysed-arm exclusions and control runs.

    Analysed arm: status != 'ok', truncated, or a request-log mismatch (review
    item 1). Control runs (broken_reset, smoke) are reported separately (review
    item 7).
    """
    mismatch = set(request_mismatch_ids or [])
    arm, control = [], []
    for r in runs:
        if r.get("control") in CONTROL_POLICIES:
            control.append(r)
        elif (r.get("status") != "ok" or r.get("truncated")
              or r["run_id"] in mismatch):
            arm.append(r)
    by_cell, by_reason = {}, {}
    for r in arm:
        key = f'{r.get("provider")}:{r.get("model")}|{r.get("policy")}|{r.get("k")}'
        by_cell[key] = by_cell.get(key, 0) + 1
        if r["run_id"] in mismatch:
            reason = "request log mismatch"
        elif r.get("status") != "ok":
            reason = "status"
        else:
            reason = "truncated"
        by_reason[reason] = by_reason.get(reason, 0) + 1
    by_control = {}
    for r in control:
        by_control[r["control"]] = by_control.get(r["control"], 0) + 1
    return {
        "n_excluded": len(arm),
        "n_error_status": sum(1 for r in arm if r.get("status") != "ok"),
        "n_truncated": sum(1 for r in arm if r.get("status") == "ok"
                           and r.get("truncated")),
        "n_request_log_mismatch": len(mismatch),
        "by_reason": dict(sorted(by_reason.items())),
        "by_cell": dict(sorted(by_cell.items())),
        "control_runs": {"n": len(control),
                         "by_control": dict(sorted(by_control.items())),
                         "run_ids": sorted(r["run_id"] for r in control)},
    }


# ---------------------------------------------------------------------------
# Statistics primitives (review item 3)
# ---------------------------------------------------------------------------

class CI(list):
    """A [lo, hi] percentile CI that carries bootstrap validity metadata."""

    def __init__(self, lo=None, hi=None, n_valid_boot=0, n_boot=0):
        super().__init__([lo, hi] if lo is not None else [None, None])
        self.n_valid_boot = int(n_valid_boot)
        self.n_boot = int(n_boot)

    @property
    def unstable(self):
        """True when fewer than 95% of the requested replicates were valid."""
        return self.n_valid_boot < 0.95 * self.n_boot

    def __repr__(self):
        return (f"CI([{self[0]}, {self[1]}], n_valid_boot={self.n_valid_boot}, "
                f"n_boot={self.n_boot}, unstable={self.unstable})")


class VecStat:
    """A statistic that is a function of SUMMED per-item payload vectors.

    `vec(T)` takes a (B, d) array of summed payloads and returns B values (NaN =
    invalid replicate). Called on a plain list of payload vectors it behaves
    like an ordinary `stat(list_of_payloads)`, so it works with any bootstrap
    implementation; `_boot_questions` detects it and draws all replicates at
    once (a (n_boot x n_questions) index matrix) instead of looping in Python.
    """

    def __init__(self, vec):
        self.vec = vec

    def __call__(self, payloads):
        payloads = list(payloads)
        if not payloads:
            return None
        total = np.sum(np.asarray(payloads, dtype=float), axis=0)[None, :]
        with np.errstate(all="ignore"):
            v = float(np.asarray(self.vec(total), dtype=float)[0])
        return v if np.isfinite(v) else None


def _boot_vec(S, n_boot, seed, vec):
    """Vectorised cluster bootstrap on per-question payload sums S (n_q x d).

    Each replicate draws n_q questions with replacement from
    np.random.default_rng(seed); its statistic is vec(sum of drawn rows).
    """
    nq = S.shape[0]
    n_boot = int(n_boot)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nq, size=(n_boot, nq))
    counts = np.bincount((idx + (np.arange(n_boot) * nq)[:, None]).ravel(),
                         minlength=n_boot * nq).reshape(n_boot, nq).astype(float)
    with np.errstate(all="ignore"):
        stats = np.asarray(vec(counts @ S), dtype=float)
    stats = stats[np.isfinite(stats)]
    if stats.size == 0:
        return CI(None, None, 0, n_boot)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return CI(float(lo), float(hi), int(stats.size), n_boot)


def _boot_questions(items, n_boot, seed, stat):
    """Cluster bootstrap over questions.

    items: iterable of (question_id, payload). Questions are resampled with
    replacement (all items of a question move together); stat(list_of_payloads)
    -> float | None. Returns a CI with n_valid_boot metadata, or None when
    there are no questions or n_boot <= 0. A `VecStat` takes the vectorised
    path (same resampling semantics, no per-replicate Python loop).
    """
    items = list(items)
    by_q = {}
    for q, p in items:
        by_q.setdefault(q, []).append(p)
    qs = sorted(by_q)
    if not qs or n_boot <= 0:
        return None
    if isinstance(stat, VecStat):
        S = np.array([np.sum(np.asarray(by_q[q], dtype=float), axis=0) for q in qs],
                     dtype=float)
        return _boot_vec(S, n_boot, seed, stat.vec)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(int(n_boot)):
        draw = rng.integers(0, len(qs), size=len(qs))
        sample = [p for i in draw for p in by_q[qs[i]]]
        v = stat(sample)
        if v is not None and np.isfinite(v):
            stats.append(float(v))
    if not stats:
        return CI(None, None, 0, n_boot)
    lo, hi = np.percentile(np.asarray(stats), [2.5, 97.5])
    return CI(float(lo), float(hi), len(stats), n_boot)


def _ratio_vec(num_col, den_col):
    """vec: sum[num_col] / sum[den_col], invalid when the denominator is <= 0."""
    def vec(T):
        den = T[:, den_col]
        return np.where(den > 0, T[:, num_col] / np.where(den > 0, den, 1.0), np.nan)
    return vec


def _mean_vec(T):
    """vec for payload [sum_value, count]: the sample mean."""
    n = T[:, 1]
    return np.where(n > 0, T[:, 0] / np.where(n > 0, n, 1.0), np.nan)


def _slope_vec(T):
    """vec for payload [n, sx, sy, sxx, sxy]: OLS slope of y on x with intercept."""
    n, sx, sy, sxx, sxy = (T[:, i] for i in range(5))
    nn = np.where(n > 0, n, 1.0)
    den = sxx - sx * sx / nn
    num = sxy - sx * sy / nn
    ok = (n >= 2) & (den > 1e-12 * np.maximum(sxx, 1e-300))
    return np.where(ok, num / np.where(ok, den, 1.0), np.nan)


def _slope_payload(x, y):
    return np.array([1.0, x, y, x * x, x * y])


def _ols(X, y):
    """OLS via least squares; X already includes any intercept column."""
    beta, *_ = np.linalg.lstsq(np.asarray(X, dtype=float), np.asarray(y, dtype=float),
                               rcond=None)
    return np.asarray(beta, dtype=float)


def _slope_with_intercept(xs, ys):
    """OLS slope of ys on xs with an intercept."""
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.size < 2:
        return None
    dx = x - x.mean()
    den = float(dx @ dx)
    if den <= 0:
        return None
    return float((dx @ (y - y.mean())) / den)


def _empty_metrics():
    return {"r2": None, "mae": None, "mape": None, "n_scored": 0, "n_test_total": 0}


def _metrics(ys, yhats):
    """R^2 (1 - SSE/SST around the mean of y), MAE, MAPE on a scored sample."""
    y = np.asarray(ys, dtype=float)
    p = np.asarray(yhats, dtype=float)
    e = y - p
    sse = float(e @ e)
    dy = y - y.mean()
    sst = float(dy @ dy)
    r2 = (1.0 - sse / sst) if sst > 0 else None
    mae = float(np.abs(e).mean())
    pos = y > 0
    mape = float((np.abs(e[pos]) / y[pos]).mean()) if pos.any() else None
    return {"r2": r2, "mae": mae, "mape": mape}


def _G(run):
    """Carry-over regressor G(T): T(T-1)/2 under carry_all, T-1 under reset."""
    T = run["turns"]
    return T * (T - 1) / 2.0 if run["policy"] == "carry_all" else float(T - 1)


def _fit_h1_model(train_runs):
    """Fit the four H1 models on one model's training runs."""
    Y = np.array([float(r["total_input"]) for r in train_runs])
    cbar, cell_mean = {}, {}
    for pol, k in sorted({(r["policy"], int(r["k"])) for r in train_runs}):
        rs = [r for r in train_runs if r["policy"] == pol and int(r["k"]) == k]
        sT = float(sum(r["turns"] for r in rs))
        cbar[(pol, k)] = (float(sum(r["total_input"] for r in rs)) / sT) if sT > 0 else None
        cell_mean[(pol, k)] = float(np.mean([r["total_input"] for r in rs]))
    beta_prompt = _ols([[1.0, r["P"]] for r in train_runs], Y)
    beta_carry = _ols([[r["P"] * r["turns"],
                        _G(r) if int(r["k"]) == 2 else 0.0,
                        _G(r) if int(r["k"]) == 8 else 0.0]
                       for r in train_runs], Y)
    return {"cbar": cbar, "cell_mean": cell_mean, "prompt": beta_prompt, "carry": beta_carry}


def _predict_h1(fit, run):
    """Predict total input Y for one run under each H1 model (None = undefined)."""
    T, P, pol, k = run["turns"], float(run["P"]), run["policy"], int(run["k"])
    cb = fit["cbar"].get((pol, k))
    cm = fit["cell_mean"].get((pol, k))
    a, b = fit["prompt"]
    beta = fit["carry"]
    return {
        "M_ctx": cb * T if cb is not None else None,
        "M_prompt": float(a + b * P),
        "M_cell": cm,
        "M_carry": float(beta[0] * P * T + (beta[1] if k == 2 else beta[2]) * _G(run)),
    }


# ---------------------------------------------------------------------------
# H1 (PREREG section 7; review items 3, 4)
# ---------------------------------------------------------------------------

def _mape_delta(rows, n_boot, seed):
    """Point estimate and cluster bootstrap CI of MAPE(M_prompt) - MAPE(M_ctx).

    Fits are fixed; only the test questions are resampled.
    """
    items = []
    for row in rows:
        y = row["y"]
        pc, pp = row["preds"]["M_ctx"], row["preds"]["M_prompt"]
        if y > 0 and pc is not None and pp is not None:
            items.append((row["q"], np.array([abs(y - pp) / y, abs(y - pc) / y, 1.0])))

    stat = VecStat(lambda T: np.where(
        T[:, 2] > 0, (T[:, 0] - T[:, 1]) / np.where(T[:, 2] > 0, T[:, 2], 1.0), np.nan))
    point = stat([p for _, p in items])
    ci = _boot_questions(items, n_boot, seed, stat)
    return point, ci


def h1(runs, n_boot=2000, seed=BOOT_SEED):
    """Fit on split == 'train', score on split == 'test', per model and pooled.

    Every model is scored on the same rows: test rows where all four
    predictions exist (review item 4); n_scored / n_test_total are reported.
    """
    runs = analysable(runs)
    train = [r for r in runs if r.get("split") == "train"]
    test = [r for r in runs if r.get("split") == "test"]
    fits = {}
    for m in sorted({r["model"] for r in train}):
        tr = [r for r in train if r["model"] == m]
        if tr:
            fits[m] = _fit_h1_model(tr)

    rows = []
    for r in test:
        f = fits.get(r["model"])
        if f is None:
            continue
        rows.append({"q": r["question_id"], "model": r["model"],
                     "y": float(r["total_input"]), "preds": _predict_h1(f, r)})

    pooled = {nm: _empty_metrics() for nm in H1_MODELS}
    per_model = {}
    scopes = [("", rows)]
    for m in sorted({row["model"] for row in rows}):
        scopes.append((m, [row for row in rows if row["model"] == m]))
    for scope_name, sc in scopes:
        target = pooled if scope_name == "" else per_model.setdefault(
            scope_name, {nm: _empty_metrics() for nm in H1_MODELS})
        scored = [row for row in sc
                  if all(row["preds"][nm] is not None for nm in H1_MODELS)]
        for nm in H1_MODELS:
            if scored:
                m = _metrics([row["y"] for row in scored],
                             [row["preds"][nm] for row in scored])
            else:
                m = _empty_metrics()
            m["n_scored"] = len(scored)
            m["n_test_total"] = len(sc)
            target[nm] = m

    delta, dci = _mape_delta(rows, n_boot, seed)
    r2_ctx = pooled["M_ctx"]["r2"]
    if dci is None:
        decision = "not supported"
    elif dci.unstable:
        decision = "not supported (unstable CI)"
    else:
        decision = "supported" if (r2_ctx is not None and r2_ctx >= 0.80
                                   and dci[0] > 0.0) else "not supported"
    return {
        "decision": decision,
        "pooled": pooled,
        "per_model": per_model,
        "delta_mape": delta,
        "delta_mape_ci": dci,
        "n_train": len(train),
        "n_test": len(test),
    }


# ---------------------------------------------------------------------------
# H2 (PREREG section 7; review items 2, 3)
# ---------------------------------------------------------------------------

def _h2_curves(runs, xs_by_run):
    """Per-run (log n, log C) demeaned segments for runs with T >= 3.

    C_r(n) = cumulative input of 0-based steps 1..n, n = 1..T-1.
    """
    segs = []
    for r in runs:
        xs = xs_by_run.get(r["run_id"])
        if not xs or len(xs) < 3:
            continue
        x = np.array([v for _, v in sorted(xs)], dtype=float)
        if (x <= 0).any():
            continue
        T = x.size
        n = np.arange(1, T, dtype=float)
        C = np.cumsum(x)[1:] - x[0]
        if (C <= 0).any():
            continue
        lx, ly = np.log(n), np.log(C)
        segs.append((lx - lx.mean(), ly - ly.mean()))
    return segs


def _h2_slope(segs):
    """Pooled within-run (fixed-effect) slope through the origin."""
    if not segs:
        return None
    X = np.concatenate([s[0] for s in segs])
    Y = np.concatenate([s[1] for s in segs])
    den = float(X @ X)
    if den <= 0:
        return None
    return float(X @ Y / den)


def _h2_run_stats(runs, xs_by_run):
    """Per-run within-run (sxy, sxx) of the H2 fit, computed once. Runs with
    T < 3 or non-positive inputs are absent (they contribute nothing)."""
    out = {}
    for r in runs:
        seg = _h2_curves([r], xs_by_run)
        if seg:
            lx, ly = seg[0]
            out[r["run_id"]] = (float(lx @ ly), float(lx @ lx))
    return out


def _h2_n_nonpositive(runs, xs_by_run):
    """Runs considered for H2 (T >= 3) dropped for non-positive inputs."""
    n = 0
    for r in runs:
        xs = xs_by_run.get(r["run_id"])
        if not xs or len(xs) < 3:
            continue
        x = np.array([v for _, v in sorted(xs)], dtype=float)
        if (x <= 0).any():
            n += 1
    return n


def h2(runs, requests, n_boot=2000, seed=BOOT_SEED):
    """log C_r(n) = a_r + b log n with run fixed effects, per policy."""
    runs = analysable(runs)
    ids = {r["run_id"] for r in runs}
    xs_by_run = {}
    for q in final_attempt_requests(requests):
        if q.get("run_id") in ids:
            xs_by_run.setdefault(q["run_id"], []).append(
                (int(q["step"]), float(q["input_tokens"])))

    rstats = _h2_run_stats(runs, xs_by_run)

    def pay(r):
        return np.array(rstats.get(r["run_id"], (0.0, 0.0)))

    def fit_b(sel):
        sxy = sum(rstats[r["run_id"]][0] for r in sel if r["run_id"] in rstats)
        sxx = sum(rstats[r["run_id"]][1] for r in sel if r["run_id"] in rstats)
        return float(sxy / sxx) if sxx > 0 else None

    slope_stat = VecStat(_ratio_vec(0, 1))

    def boot_b(sub):
        return _boot_questions([(r["question_id"], pay(r)) for r in sub],
                               n_boot, seed, slope_stat)

    policies = sorted({r["policy"] for r in runs})
    b, ci = {}, {}
    for pol in policies:
        sub = [r for r in runs if r["policy"] == pol]
        b[pol] = fit_b(sub)
        ci[pol] = boot_b(sub)

    def diff_payload(r):
        sxy, sxx = rstats.get(r["run_id"], (0.0, 0.0))
        c = float(r["policy"] == "carry_all")
        rr = float(r["policy"] == "reset_summary")
        return np.array([sxy * c, sxx * c, sxy * rr, sxx * rr])

    def diff_vec(T):
        return _ratio_vec(0, 1)(T) - _ratio_vec(2, 3)(T)

    bc0 = fit_b([r for r in runs if r["policy"] == "carry_all"])
    br0 = fit_b([r for r in runs if r["policy"] == "reset_summary"])
    diff = None if bc0 is None or br0 is None else bc0 - br0
    diff_ci = _boot_questions([(r["question_id"], diff_payload(r)) for r in runs],
                              n_boot, seed, VecStat(diff_vec))

    per_mk = {}
    for m in sorted({r["model"] for r in runs}):
        for k in sorted({int(r["k"]) for r in runs}):
            entry = {}
            for pol in policies:
                sub = [r for r in runs
                       if r["model"] == m and int(r["k"]) == k and r["policy"] == pol]
                entry[pol] = {"b": fit_b(sub), "ci": boot_b(sub)}
            per_mk[f"{m}|k={k}"] = entry

    # Secondary (descriptive): across-run log Y ~ log T per policy.
    secondary = {}
    for pol in policies:
        items = [(r["question_id"], _slope_payload(float(math.log(r["turns"])),
                                                   float(math.log(r["total_input"]))))
                 for r in runs
                 if r["policy"] == pol and (r.get("turns") or 0) >= 1
                 and (r.get("total_input") or 0) > 0]
        stat = VecStat(_slope_vec)
        secondary[pol] = {"b": stat([p for _, p in items]),
                          "ci": _boot_questions(items, n_boot, seed, stat)}

    unstable = (ci.get("carry_all") is not None and ci["carry_all"].unstable) or \
               (diff_ci is not None and diff_ci.unstable)
    if unstable:
        decision = "not supported (unstable CI)"
    else:
        cond_a = (b.get("carry_all") is not None and ci.get("carry_all") is not None
                  and ci["carry_all"][0] > 1.0)
        cond_b = diff_ci is not None and diff_ci[0] > 0.0
        cond_c = b.get("reset_summary") is not None and 0.90 <= b["reset_summary"] <= 1.10
        if cond_a and cond_b and cond_c:
            decision = "supported"
        elif cond_a and cond_b:
            decision = "partially supported"
        else:
            decision = "not supported"

    return {
        "decision": decision,
        "b": b,
        "ci": ci,
        "diff": diff,
        "diff_ci": diff_ci,
        "per_model_k": per_mk,
        "secondary_logY_logT": secondary,
        "n_runs_used": sum(1 for r in runs
                           if xs_by_run.get(r["run_id"])
                           and len(xs_by_run[r["run_id"]]) >= 3),
        "n_runs_dropped_nonpositive": _h2_n_nonpositive(runs, xs_by_run),
    }


# ---------------------------------------------------------------------------
# H3 (PREREG section 7; review items 3, 6)
# ---------------------------------------------------------------------------

def h3(runs, n_boot=2000, seed=BOOT_SEED):
    """Top-k x policy interaction: I_qm and rho = mean carry diff / mean reset diff."""
    runs = analysable(runs)
    need = {("carry_all", 2), ("carry_all", 8), ("reset_summary", 2), ("reset_summary", 8)}
    by_qm = {}
    for r in runs:
        by_qm.setdefault((r["question_id"], r["model"]), {})[(r["policy"], int(r["k"]))] = r
    pairs = {key: cells for key, cells in by_qm.items() if set(cells) >= need}

    def stats_of(cells):
        yc2 = float(cells[("carry_all", 2)]["total_input"])
        yc8 = float(cells[("carry_all", 8)]["total_input"])
        yr2 = float(cells[("reset_summary", 2)]["total_input"])
        yr8 = float(cells[("reset_summary", 8)]["total_input"])
        return (yc8 - yc2) - (yr8 - yr2), yc8 - yc2, yr8 - yr2

    if not pairs:
        return {"decision": "not supported", "n_pairs": 0, "I_mean": None, "I_ci": None,
                "rho": None, "rho_ci": None, "note": "no (question, model) with all four cells",
                "mean_T": {}, "per_model": {}}

    # items: one (question_id, payload) per (question, model) pair; payload is
    # [I, carry k-effect, reset k-effect, 1] so every statistic is a ratio of sums.
    items = [(q, np.array(stats_of(cells) + (1.0,))) for (q, _m), cells in pairs.items()]
    I_mean = float(np.mean([p[0] for _, p in items]))
    num_mean = float(np.mean([p[1] for _, p in items]))
    den_mean = float(np.mean([p[2] for _, p in items]))
    rho = (num_mean / den_mean) if den_mean != 0 else None

    def vec_ratio(num, den):
        def vec(T):
            d = T[:, den]
            return np.where(d != 0, T[:, num] / np.where(d != 0, d, 1.0), np.nan)
        return vec

    stat_i = VecStat(vec_ratio(0, 3))
    stat_den = VecStat(vec_ratio(2, 3))
    stat_rho = VecStat(vec_ratio(1, 2))

    I_ci = _boot_questions(items, n_boot, seed, stat_i)
    den_ci = _boot_questions(items, n_boot, seed, stat_den)

    if den_ci is None or den_ci.unstable or (den_ci[0] is not None
                                             and den_ci[0] <= 0 <= den_ci[1]):
        rho_out, rho_ci = None, None
        note = ("rho undefined (denominator CI includes 0 or is unstable); "
                "H3 decided on I alone")
    else:
        rho_out, rho_ci = rho, _boot_questions(items, n_boot, seed, stat_rho)
        note = None

    if (I_ci is not None and I_ci.unstable) or (rho_ci is not None and rho_ci.unstable):
        decision = "not supported (unstable CI)"
    else:
        i_ok = I_ci is not None and I_ci[0] > 0.0
        if rho_ci is None:
            decision = "supported" if i_ok else "not supported"
        else:
            decision = "supported" if (i_ok and rho_ci[0] > 1.0) else "not supported"

    # Mean T per cell over the (question, model) pairs actually used above.
    paired_keys = set(pairs)
    mean_t = {}
    for (pol, k) in sorted(need):
        rs = [r for r in runs
              if r["policy"] == pol and int(r["k"]) == k
              and (r["question_id"], r["model"]) in paired_keys]
        mean_t[f"{pol}|k={k}"] = (float(np.mean([r["turns"] for r in rs])) if rs else None)

    # Per model (descriptive): the interaction I within each model's pairs.
    per_model = {}
    for m in sorted({mm for (_q, mm) in pairs}):
        m_items = [(q, np.array(stats_of(cells) + (1.0,)))
                   for (q, mm), cells in pairs.items() if mm == m]
        per_model[m] = {
            "n_pairs": len(m_items),
            "I_mean": float(np.mean([p[0] for _, p in m_items])),
            "I_ci": _boot_questions(m_items, n_boot, seed, stat_i),
        }

    return {
        "decision": decision,
        "n_pairs": len(pairs),
        "I_mean": I_mean,
        "I_ci": I_ci,
        "rho": rho_out,
        "rho_ci": rho_ci,
        "note": note,
        "mean_T": mean_t,
        "per_model": per_model,
    }


# ---------------------------------------------------------------------------
# H4 (PREREG section 7) - descriptive only, no decision
# ---------------------------------------------------------------------------

def h4(runs, n_boot=2000, seed=BOOT_SEED):
    """EM and F1 per cell with CIs, plus paired policy and k differences."""
    runs = [r for r in analysable(runs) if r.get("em") is not None]
    cells = {}
    for r in runs:
        cells.setdefault((r["model"], r["policy"], int(r["k"])), []).append(r)

    out_cells = []
    cell_maps = {}
    for (m, pol, k), rs in sorted(cells.items()):
        cell_maps[(m, pol, k)] = {r["question_id"]: r for r in rs}
        em = float(np.mean([r["em"] for r in rs]))
        f1 = float(np.mean([r["f1"] for r in rs]))

        mean_stat = VecStat(_mean_vec)
        em_items = [(r["question_id"], np.array([float(r["em"]), 1.0])) for r in rs]
        f1_items = [(r["question_id"], np.array([float(r["f1"]), 1.0])) for r in rs]

        out_cells.append({
            "model": m, "policy": pol, "k": k, "n": len(rs),
            "em": em, "em_ci": _boot_questions(em_items, n_boot, seed, mean_stat),
            "f1": f1, "f1_ci": _boot_questions(f1_items, n_boot, seed, mean_stat),
        })

    def paired(map_a, map_b, fields):
        common = sorted(set(map_a) & set(map_b))
        out = {"n_pairs": len(common)}
        for f in fields:
            items = [(q, np.array([float(map_a[q][f] - map_b[q][f]), 1.0]))
                     for q in common]
            out[f + "_diff"] = (float(np.mean([p[0] for _, p in items]))
                                if items else None)
            out[f + "_diff_ci"] = _boot_questions(items, n_boot, seed,
                                                  VecStat(_mean_vec))
        return out

    pol_diffs = []
    for (m, k) in sorted({(m, k) for (m, _, k) in cell_maps}):
        map_c = cell_maps.get((m, "carry_all", k))
        map_r = cell_maps.get((m, "reset_summary", k))
        if map_c and map_r:
            pol_diffs.append({"model": m, "k": k,
                              **paired(map_c, map_r, ("em", "f1"))})
    k_diffs = []
    for (m, pol) in sorted({(m, pol) for (m, pol, _) in cell_maps}):
        map_2 = cell_maps.get((m, pol, 2))
        map_8 = cell_maps.get((m, pol, 8))
        if map_2 and map_8:
            k_diffs.append({"model": m, "policy": pol,
                            **paired(map_2, map_8, ("em", "f1"))})

    return {
        "cells": out_cells,
        "paired_carry_minus_reset": pol_diffs,
        "paired_k8_minus_k2": k_diffs,
    }


# ---------------------------------------------------------------------------
# H5 (PREREG section 7 + Amendment 1; review item 3)
# ---------------------------------------------------------------------------

def _h5_run_cached_stats(reqs):
    """(num, den) of one run's fixed-effect slope of cached share s_t on step t
    (0-based steps >= 1), or None when the run has < 2 usable points."""
    pts = []
    for q in reqs:
        t = int(q["step"])
        if t >= 1:
            pts.append((t, float(q["cached_input_tokens"]) / float(q["input_tokens"])))
    if len(pts) < 2:
        return None
    t = np.array([p[0] for p in pts], dtype=float)
    s_ = np.array([p[1] for p in pts], dtype=float)
    dt = t - t.mean()
    return float(dt @ (s_ - s_.mean())), float(dt @ dt)


def _h5_usable_runs(runs, requests):
    """Amendment 1: H5 uses analysable runs whose cached_input_tokens are non-null
    (every request of the run reports a cached count and a positive input)."""
    by_run = {}
    for q in requests:
        by_run.setdefault(q.get("run_id"), []).append(q)
    out = []
    for r in runs:
        rq = by_run.get(r["run_id"])
        if rq and all(q.get("cached_input_tokens") is not None
                      and (q.get("input_tokens") or 0) > 0 for q in rq):
            out.append((r, rq))
    return out


def _h5_block(usable, n_boot, seed):
    """Slopes and CIs for one set of usable (run, requests) pairs."""
    cached_items, usd_items = [], []
    for r, rq in usable:
        st = _h5_run_cached_stats(rq)
        cached_items.append((r["question_id"], np.array(st if st else (0.0, 0.0))))
        if r.get("usd") is not None and (r.get("total_input") or 0) > 0:
            usd_items.append((r["question_id"], _slope_payload(
                float(r["turns"]), float(r["usd"]) / float(r["total_input"]))))
        else:
            usd_items.append((r["question_id"], np.zeros(5)))
    sa, sb = VecStat(_ratio_vec(0, 1)), VecStat(_slope_vec)
    return {
        "n_runs": len(usable),
        "cached_share_slope": sa([p for _, p in cached_items]),
        "cached_share_slope_ci": _boot_questions(cached_items, n_boot, seed, sa),
        "usd_per_input_slope": sb([p for _, p in usd_items]),
        "usd_per_input_slope_ci": _boot_questions(usd_items, n_boot, seed, sb),
    }


def h5(runs, requests, n_boot=2000, seed=BOOT_SEED):
    """Amendment 1: runs with non-null cached_input_tokens, any provider; also per model."""
    usable = _h5_usable_runs(analysable(runs), final_attempt_requests(requests))
    if not usable:
        return {"status": "not run",
                "reason": "no analysable runs with non-null cached_input_tokens"}
    main = _h5_block(usable, n_boot, seed)
    ci_a, ci_b = main["cached_share_slope_ci"], main["usd_per_input_slope_ci"]
    if (ci_a is not None and ci_a.unstable) or (ci_b is not None and ci_b.unstable):
        decision = "not supported (unstable CI)"
    else:
        supported = (ci_a is not None and ci_a[0] > 0.0
                     and ci_b is not None and ci_b[1] < 0.0)
        decision = "supported" if supported else "not supported"
    per_model = {m: _h5_block([u for u in usable if u[0]["model"] == m], n_boot, seed)
                 for m in sorted({u[0]["model"] for u in usable})}
    return {"status": "evaluated", "decision": decision, **main, "per_model": per_model}


# ---------------------------------------------------------------------------
# Fail-first controls 2 and 3 (PREREG section 8; review item 5)
# ---------------------------------------------------------------------------

def control2(runs, requests, broken_runs, broken_reqs, n_boot=2000, seed=BOOT_SEED):
    """Swap broken reset_summary runs in for the real ones; H2 must NOT be supported.

    Broken runs are filtered to usable ones (ok, not truncated) before the swap
    (review item 5). If H2 cannot be evaluated on the swapped subset (too few
    runs with T >= 3 or an unstable CI), the control is "inconclusive", not
    "passed".
    """
    usable = [r for r in broken_runs
              if r.get("status") == "ok" and not r.get("truncated")]
    keys = {(r["question_id"], r["model"], int(r["k"])) for r in usable}
    swapped = [r for r in analysable(runs)
               if (r["question_id"], r["model"], int(r["k"])) in keys
               and r["policy"] == "carry_all"]
    for r in usable:
        nr = dict(r)
        nr["control"] = None
        nr["policy"] = "reset_summary"  # normalise reset_summary_broken for the fit
        swapped.append(nr)
    ids = {r["run_id"] for r in swapped}
    bids = {r["run_id"] for r in usable}
    reqs = ([q for q in requests if q.get("run_id") in ids and q.get("run_id") not in bids]
            + [q for q in broken_reqs if q.get("run_id") in bids])
    out = h2(swapped, reqs, n_boot=n_boot, seed=seed)
    b, ci, dci = out["b"], out["ci"], out["diff_ci"]
    too_few = (not usable or b.get("carry_all") is None or b.get("reset_summary") is None
               or out["diff"] is None or out["n_runs_used"] == 0)
    unstable = ((ci.get("carry_all") is not None and ci["carry_all"].unstable)
                or (dci is not None and dci.unstable))
    if too_few:
        status, reason = "inconclusive", "too few runs with T >= 3 on the swapped subset"
    elif unstable:
        status, reason = "inconclusive", "unstable bootstrap CI on the swapped subset"
    elif out["decision"] == "supported":
        status, reason = "failed", None
    else:
        status, reason = "passed", None

    if not too_few and not unstable:
        cond_a = (b.get("carry_all") is not None and ci.get("carry_all") is not None
                  and ci["carry_all"][0] > 1.0)
        cond_b = dci is not None and dci[0] > 0.0
        cond_c = b.get("reset_summary") is not None and 0.90 <= b["reset_summary"] <= 1.10
    else:
        cond_a = cond_b = cond_c = None
    rules = {"a": cond_a, "b": cond_b, "c": cond_c}
    return {
        "h2_decision_with_broken": out["decision"],
        "status": status,
        "passed": status == "passed",
        "inconclusive_reason": reason,
        "n_broken_used": len(usable),
        "n_questions": len({k[0] for k in keys}),
        "rules_failed": [nm for nm, ok in rules.items() if ok is False],
        "rules_passed": [nm for nm, ok in rules.items() if ok is True],
        "h2": out,
    }


def _h1_decisions_for(base, Ymat, n_boot, seed=BOOT_SEED):
    """H1 decision rule (PREREG 7) for each row of Ymat, vectorised.

    base: analysable runs; Ymat: (n_sets, len(base)) candidate total_input
    vectors. Only what the rule needs is recomputed per set: the M_ctx and
    M_prompt fits (train split, per model), pooled test R^2 of M_ctx on the rows
    all four models can score, and the question-cluster bootstrap CI of
    MAPE(M_prompt) - MAPE(M_ctx). Equivalent to h1(...)["decision"].
    """
    Ymat = np.atleast_2d(np.asarray(Ymat, dtype=float))
    n_sets = Ymat.shape[0]
    if n_boot <= 0:
        return ["not supported"] * n_sets
    cols = {"y": [], "pc": [], "pp": []}
    ok_all, q_all = [], []
    for m in sorted({r["model"] for r in base if r.get("split") == "train"}):
        tr = [j for j, r in enumerate(base) if r["model"] == m and r.get("split") == "train"]
        te = [j for j, r in enumerate(base) if r["model"] == m and r.get("split") == "test"]
        if not te:
            continue
        cells = sorted({(base[j]["policy"], int(base[j]["k"])) for j in tr})
        code = {c: i for i, c in enumerate(cells)}
        tr_code = np.array([code[(base[j]["policy"], int(base[j]["k"]))] for j in tr])
        sT = np.bincount(tr_code, weights=[float(base[j]["turns"]) for j in tr],
                         minlength=len(cells))
        Y_tr = Ymat[:, tr]
        cell_sum = np.stack([Y_tr[:, tr_code == i].sum(axis=1) for i in range(len(cells))],
                            axis=1)
        with np.errstate(all="ignore"):
            cbar = cell_sum / sT[None, :]
        X = np.array([[1.0, float(base[j]["P"])] for j in tr])
        coef = Y_tr @ np.linalg.pinv(X).T          # (n_sets, 2): intercept, slope
        te_code = np.array([code.get((base[j]["policy"], int(base[j]["k"])), -1)
                            for j in te])
        T_te = np.array([float(base[j]["turns"]) for j in te])
        P_te = np.array([float(base[j]["P"]) for j in te])
        known = te_code >= 0
        ok = known & (sT[np.where(known, te_code, 0)] > 0)
        pc = cbar[:, np.where(known, te_code, 0)] * T_te[None, :]
        cols["y"].append(Ymat[:, te])
        cols["pc"].append(pc)
        cols["pp"].append(coef[:, :1] + coef[:, 1:2] * P_te[None, :])
        ok_all.append(ok)
        q_all.extend(base[j]["question_id"] for j in te)
    if not ok_all:
        return ["not supported"] * n_sets
    y, pc, pp = (np.concatenate(cols[c], axis=1) for c in ("y", "pc", "pp"))
    ok = np.concatenate(ok_all)
    uq, qcode = np.unique(np.array(q_all), return_inverse=True)

    decisions = []
    for i in range(n_sets):
        ys, pcs = y[i, ok], pc[i, ok]
        if ys.size:
            sse = float(((ys - pcs) ** 2).sum())
            sst = float(((ys - ys.mean()) ** 2).sum())
            r2 = (1.0 - sse / sst) if sst > 0 else None
        else:
            r2 = None
        use = ok & (y[i] > 0)
        if not use.any():
            decisions.append("not supported")
            continue
        yu = y[i, use]
        pay = np.stack([np.abs(yu - pp[i, use]) / yu, np.abs(yu - pc[i, use]) / yu,
                        np.ones(yu.size)], axis=1)
        qc = qcode[use]
        S = np.zeros((uq.size, 3))
        np.add.at(S, qc, pay)
        S = S[np.bincount(qc, minlength=uq.size) > 0]
        dci = _boot_vec(S, n_boot, seed, lambda T: np.where(
            T[:, 2] > 0, (T[:, 0] - T[:, 1]) / np.where(T[:, 2] > 0, T[:, 2], 1.0), np.nan))
        if dci.unstable:
            decisions.append("not supported (unstable CI)")
        else:
            good = r2 is not None and r2 >= 0.80 and dci[0] > 0.0
            decisions.append("supported" if good else "not supported")
    return decisions


def control3(runs, n_perm=200, seed=PERM_SEED, n_boot=2000):
    """Permute total_input across analysable runs within model; rerun the H1 rule."""
    base = analysable(runs)
    rng = np.random.default_rng(seed)
    n_perm = int(n_perm)
    Y = np.array([float(r["total_input"]) for r in base])
    midx = {m: np.array([j for j, r in enumerate(base) if r["model"] == m])
            for m in sorted({r["model"] for r in base})}
    Yp = np.empty((n_perm, len(base)))
    for i in range(n_perm):
        for m, idx in midx.items():
            Yp[i, idx] = Y[idx][rng.permutation(idx.size)]
    decisions = _h1_decisions_for(base, Yp, n_boot) if n_perm else []
    n_supported = sum(d == "supported" for d in decisions)
    frac = n_supported / float(n_perm) if n_perm else None
    return {"frac_supported": frac, "passed": (frac is not None and frac <= 0.05),
            "n_perm": n_perm, "n_supported": n_supported}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _git(args):
    try:
        r = subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True,
                           text=True, timeout=30)
        return r.stdout.strip() or None
    except Exception:  # noqa: BLE001 - provenance must never break the analysis
        return None


def _prereg_sha():
    out = _git(["log", "--diff-filter=A", "--format=%H", "--", "experiments/PREREG.md"])
    lines = [l for l in (out or "").splitlines() if l.strip()]
    # CI containers have no git; the workflow computes the SHA outside and passes PREREG_SHA.
    return lines[-1] if lines else (os.environ.get("PREREG_SHA") or None)


def _sha256_file(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def _tokenledger_report(az_reqs):
    """tokenledger.account metrics over analysable requests, overall and per model."""
    if not az_reqs:
        return {"overall": None, "per_model": {},
                "cache_read_reported": False, "version": tokenledger.__version__}
    norm = [dict(q, turn=q.get("turn", 0)) for q in az_reqs]
    reported = any(q.get("cached_input_tokens") is not None for q in norm)

    def account(rows):
        if not rows:
            return None
        fd, path = tempfile.mkstemp(prefix="tl_", suffix=".jsonl")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")
            return dataclasses.asdict(
                tokenledger.account(tokenledger.load_requests(path)))
        finally:
            os.unlink(path)

    rows = to_tokenledger_rows(norm)
    per_model = {}
    by_model = {}
    for q in norm:
        by_model.setdefault(q.get("model"), []).append(q)
    for m, qs in sorted(by_model.items()):
        per_model[str(m)] = account(to_tokenledger_rows(qs))
    return {"overall": account(rows), "per_model": per_model,
            "cache_read_reported": reported, "version": tokenledger.__version__}


def _clean(o):
    """Recursively convert numpy types, CIs and non-finite floats for JSON."""
    if isinstance(o, CI):
        return {"ci": list(o), "n_valid_boot": o.n_valid_boot,
                "n_boot": o.n_boot, "unstable": o.unstable}
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        o = float(o)
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def _san(name):
    """LaTeX-safe figure-name fragment: letters only, capitalised."""
    return re.sub(r"[^A-Za-z]", "", str(name)).capitalize()


_KWORDS = ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")


def _kword(k):
    """Digits spelled out so figure names stay letters-only (KTwo, KEight)."""
    return "".join(_KWORDS[int(d)] for d in str(k))


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _median(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def _sum(xs):
    return float(sum(x for x in xs if x is not None))


# ---------------------------------------------------------------------------
# Figures (review item 8)
# ---------------------------------------------------------------------------

def build_figures(aggregates, n_boot, n_perm, data_source, prereg_sha):
    """Flat dict of LaTeX-safe Agent* figure names -> {value, format, source, desc}."""
    figures = {}

    def fig(name, value, fmt, desc, source=data_source):
        if not re.fullmatch(r"Agent[A-Za-z]+", name) or not (3 <= len(name) <= 60):
            raise ValueError(f"invalid figure name: {name!r}")
        if fmt not in FIG_FORMATS:
            raise ValueError(f"invalid figure format: {fmt!r}")
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            figures[name] = {"value": "not yet run", "format": "text",
                             "source": source, "desc": desc}
        else:
            figures[name] = {"value": value, "format": fmt,
                             "source": source, "desc": desc}

    def fig_ci(name, ci, fmt, desc, source=data_source):
        if ci is None:
            fig(name + "CIlo", None, fmt, desc + " (CI lower bound)", source)
            fig(name + "CIhi", None, fmt, desc + " (CI upper bound)", source)
        else:
            fig(name + "CIlo", ci[0], fmt, desc + " (CI lower bound)", source)
            fig(name + "CIhi", ci[1], fmt, desc + " (CI upper bound)", source)

    # Design constants (source "prereg").
    fig("AgentNQuestions", N_QUESTIONS, "int",
        "Pre-registered number of sampled questions (PREREG section 4).", "prereg")
    fig("AgentTurnCap", TURN_CAP, "int",
        "Per-run turn cap, max_steps = 10 (PREREG section 2).", "prereg")
    fig("AgentKLow", K_LOW, "int",
        "Retrieval top-k, low level (PREREG section 3).", "prereg")
    fig("AgentKHigh", K_HIGH, "int",
        "Retrieval top-k, high level (PREREG section 3).", "prereg")
    fig("AgentNTestQuestions", N_TEST_QUESTIONS, "int",
        "Held-out test questions for H1: p mod 3 == 2 over 48 positions.", "prereg")
    fig("AgentNBoot", n_boot, "int",
        "Cluster bootstrap replicates over questions (PREREG section 7).", "prereg")
    fig("AgentNPerm", n_perm, "int",
        "Label-shuffle permutations for control 3 (PREREG section 8).", "prereg")
    fig("AgentPreregShaShort", prereg_sha[:7] if prereg_sha else None, "text",
        "First 7 characters of the PREREG.md registration commit SHA.", "prereg")

    # Aggregates.
    fig("AgentNRunsAnalysed", aggregates["n_runs_analysable"], "int",
        "Runs entering the hypothesis analyses after exclusions.")
    fig("AgentNRunsExcluded", aggregates["exclusions"]["n_excluded"], "int",
        "Runs excluded from the analysed arms (status, truncation, request-log mismatch).")
    fig("AgentNControlRuns", aggregates["exclusions"]["control_runs"]["n"], "int",
        "Control runs (broken_reset, smoke), reported separately from the analysed arms.")
    fig("AgentNQuestionsAnalysed", aggregates["n_questions"], "int",
        "Distinct questions with analysable runs.")
    fig("AgentCloudSpendUsd", aggregates["cloud_spend_usd"], "float2",
        "Total cloud API spend (USD): sum of cloud_spend_total over the input "
        "provenance files (0 when none is recorded).")
    models = aggregates.get("agent_models", {})
    fig("AgentModelFlashId", models.get("flash") or "not yet run", "text",
        "Exact Gemini model id(s) of the flash-tier cells found in the runs.")
    fig("AgentModelLiteId", models.get("lite") or "not yet run", "text",
        "Exact Gemini model id(s) of the flash-lite-tier cells found in the runs.")

    tl = aggregates["tokenledger"]
    fig("AgentTlTotalInput", tl["overall"]["total_input"] if tl["overall"] else None,
        "kilo", "Total input tokens over analysable requests (tokenledger accounting).")
    fig("AgentTlRequests", tl["overall"]["n_requests"] if tl["overall"] else None,
        "int", "Number of analysable requests (tokenledger accounting).")
    fig("AgentTlCacheReadPct",
        tl["overall"]["cache_read_pct"] if (tl["overall"] and tl["cache_read_reported"])
        else None,
        "pct1", "Cache-read share of input tokens over analysable requests.")

    h = aggregates["hypotheses"]
    if h["H1"]["status"] == "evaluated":
        p = h["H1"]["pooled"]
        fig("AgentHOneDecision", h["H1"]["decision"], "text", "H1 decision.")
        fig("AgentHOneRsqCtx", p["M_ctx"]["r2"], "pct1",
            "H1 pooled test R^2 of the primary model M_ctx (context x turns).")
        fig("AgentHOneMapeCtx", p["M_ctx"]["mape"], "pct1",
            "H1 pooled test MAPE of M_ctx.")
        fig("AgentHOneMapePrompt", p["M_prompt"]["mape"], "pct1",
            "H1 pooled test MAPE of the prompt-size baseline M_prompt.")
        fig("AgentHOneDeltaMape", h["H1"]["delta_mape"], "signedpct1",
            "H1 MAPE(M_prompt) - MAPE(M_ctx) on the pooled test rows.")
        fig_ci("AgentHOneDeltaMape", h["H1"]["delta_mape_ci"], "signedpct1",
               "H1 bootstrap CI of MAPE(M_prompt) - MAPE(M_ctx).")
    else:
        for nm in ("AgentHOneDecision", "AgentHOneRsqCtx", "AgentHOneMapeCtx",
                   "AgentHOneMapePrompt", "AgentHOneDeltaMape",
                   "AgentHOneDeltaMapeCIlo", "AgentHOneDeltaMapeCIhi"):
            fig(nm, None, "text", "H1 not evaluated.")

    if h["H2"]["status"] == "evaluated":
        fig("AgentHTwoDecision", h["H2"]["decision"], "text", "H2 decision.")
        fig("AgentHTwoBCarry", h["H2"]["b"].get("carry_all"), "float2",
            "H2 fixed-effects exponent b under carry_all (log C ~ b log n).")
        fig_ci("AgentHTwoBCarry", h["H2"]["ci"].get("carry_all"), "float2",
               "H2 bootstrap CI of b under carry_all.")
        fig("AgentHTwoBReset", h["H2"]["b"].get("reset_summary"), "float2",
            "H2 fixed-effects exponent b under reset_summary.")
        fig_ci("AgentHTwoBReset", h["H2"]["ci"].get("reset_summary"), "float2",
               "H2 bootstrap CI of b under reset_summary.")
        fig("AgentHTwoDiffBCarryMinusBReset", h["H2"]["diff"], "float2",
            "H2 point estimate of b_carry - b_reset.")
        fig_ci("AgentHTwoDiffBCarryMinusBReset", h["H2"]["diff_ci"], "float2",
               "H2 bootstrap CI of b_carry - b_reset.")
    else:
        for nm in ("AgentHTwoDecision", "AgentHTwoBCarry", "AgentHTwoBCarryCIlo",
                   "AgentHTwoBCarryCIhi", "AgentHTwoBReset", "AgentHTwoBResetCIlo",
                   "AgentHTwoBResetCIhi", "AgentHTwoDiffBCarryMinusBReset",
                   "AgentHTwoDiffBCarryMinusBResetCIlo",
                   "AgentHTwoDiffBCarryMinusBResetCIhi"):
            fig(nm, None, "text", "H2 not evaluated.")

    if h["H3"]["status"] == "evaluated":
        fig("AgentHThreeDecision", h["H3"]["decision"], "text", "H3 decision.")
        fig("AgentHThreeIMean", h["H3"]["I_mean"], "float2",
            "H3 mean interaction I = (Y[c,8]-Y[c,2]) - (Y[r,8]-Y[r,2]) in input tokens.")
        fig_ci("AgentHThreeIMean", h["H3"]["I_ci"], "float2",
               "H3 bootstrap CI of the mean interaction I.")
        fig("AgentHThreeRho", h["H3"]["rho"], "float2",
            "H3 rho = mean carry k-effect / mean reset k-effect (undefined when the "
            "denominator CI includes 0).")
        fig_ci("AgentHThreeRho", h["H3"]["rho_ci"], "float2",
               "H3 bootstrap CI of rho.")
    else:
        for nm in ("AgentHThreeDecision", "AgentHThreeIMean", "AgentHThreeIMeanCIlo",
                   "AgentHThreeIMeanCIhi", "AgentHThreeRho", "AgentHThreeRhoCIlo",
                   "AgentHThreeRhoCIhi"):
            fig(nm, None, "text", "H3 not evaluated.")

    if h["H4"]["status"] == "evaluated":
        for c in h["H4"]["cells"]:
            frag = f"{_san(c['model'])}{_san(c['policy'])}K{_kword(c['k'])}"
            fig(f"AgentHFourEm{frag}", c["em"], "pct1",
                f"H4 exact match, cell {c['model']}/{c['policy']}/k={c['k']}.")
            fig_ci(f"AgentHFourEm{frag}", c["em_ci"], "pct1",
                   f"H4 exact match CI, cell {c['model']}/{c['policy']}/k={c['k']}.")
            fig(f"AgentHFourFone{frag}", c["f1"], "pct1",
                f"H4 token F1, cell {c['model']}/{c['policy']}/k={c['k']}.")
            fig_ci(f"AgentHFourFone{frag}", c["f1_ci"], "pct1",
                   f"H4 token F1 CI, cell {c['model']}/{c['policy']}/k={c['k']}.")
        for d in h["H4"]["paired_carry_minus_reset"]:
            frag = f"{_san(d['model'])}K{_kword(d['k'])}"
            fig(f"AgentHFourEmDiff{frag}", d["em_diff"], "signedpct1",
                f"H4 paired EM difference carry_all - reset_summary, "
                f"{d['model']} k={d['k']}.")
            fig_ci(f"AgentHFourEmDiff{frag}", d["em_diff_ci"], "signedpct1",
                   f"H4 paired EM difference CI, {d['model']} k={d['k']}.")
            fig(f"AgentHFourFoneDiff{frag}", d["f1_diff"], "signedpct1",
                f"H4 paired F1 difference carry_all - reset_summary, "
                f"{d['model']} k={d['k']}.")
            fig_ci(f"AgentHFourFoneDiff{frag}", d["f1_diff_ci"], "signedpct1",
                   f"H4 paired F1 difference CI, {d['model']} k={d['k']}.")
        for d in h["H4"]["paired_k8_minus_k2"]:
            frag = f"{_san(d['model'])}{_san(d['policy'])}"
            fig(f"AgentHFourEmKdiff{frag}", d["em_diff"], "signedpct1",
                f"H4 paired EM difference k=8 - k=2, {d['model']} {d['policy']}.")
            fig_ci(f"AgentHFourEmKdiff{frag}", d["em_diff_ci"], "signedpct1",
                   f"H4 paired EM difference CI, {d['model']} {d['policy']}.")
            fig(f"AgentHFourFoneKdiff{frag}", d["f1_diff"], "signedpct1",
                f"H4 paired F1 difference k=8 - k=2, {d['model']} {d['policy']}.")
            fig_ci(f"AgentHFourFoneKdiff{frag}", d["f1_diff_ci"], "signedpct1",
                   f"H4 paired F1 difference CI, {d['model']} {d['policy']}.")
    else:
        fig("AgentHFourStatus", "not yet run", "text", "H4 not evaluated.")

    if h["H5"]["status"] == "evaluated":
        fig("AgentHFiveDecision", h["H5"]["decision"], "text", "H5 decision.")
        fig_ci("AgentHFiveCachedSlope", h["H5"]["cached_share_slope_ci"], "pct1",
               "H5 CI of the run-fixed-effect slope of cached share on step (t >= 2), "
               "runs with non-null cached tokens.")
        fig_ci("AgentHFiveUsdSlope", h["H5"]["usd_per_input_slope_ci"], "float3",
               "H5 CI of the OLS slope of USD per input token on turns.")
    else:
        for nm in ("AgentHFiveDecision", "AgentHFiveCachedSlopeCIlo",
                   "AgentHFiveCachedSlopeCIhi", "AgentHFiveUsdSlopeCIlo",
                   "AgentHFiveUsdSlopeCIhi"):
            fig(nm, None, "text", "H5 not evaluated.")

    c1 = aggregates["controls"].get("C1_token_accounting") or {}
    c1_status = c1.get("status") or ("passed" if c1.get("passed") is True
                                     else "failed" if c1.get("passed") is False else "not run")
    fig("AgentCOneStatus", c1_status, "text",
        "Control 1 (token accounting reproduces a hand-computed prefix sum): passed/failed/not run.")
    c2 = aggregates["controls"]["C2_broken_policy"]
    fig("AgentCTwoStatus", c2.get("status", "not run"), "text",
        "Control 2 (broken policy) status: passed/failed/inconclusive/not run.")
    c3 = aggregates["controls"]["C3_label_shuffle"]
    if c3.get("status") in ("passed", "failed"):
        fig("AgentCThreeFracSupported", c3["frac_supported"], "float3",
            "Control 3: fraction of label-shuffle permutations supporting H1.")
        fig("AgentCThreePassed", str(c3["passed"]), "text",
            "Control 3 passed: frac_supported <= 0.05.")
    else:
        fig("AgentCThreeFracSupported", None, "text", "Control 3 not run.")
        fig("AgentCThreePassed", None, "text", "Control 3 not run.")

    return figures


# ---------------------------------------------------------------------------
# Provenance sources and evidence assembly (review items 1, 8)
# ---------------------------------------------------------------------------

def _source_id(name):
    """Sanitize a dir name to ^[a-z][a-z0-9_]{2,40}$."""
    s = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    if not s:
        s = "grid"
    if not s[0].isalpha():
        s = "src_" + s
    if len(s) < 3:
        s = s + "_" * (3 - len(s))
    return s[:41]


def _build_sources(input_dirs):
    """One measured source per input dir (sha256 of runs.jsonl) + the prereg source."""
    sources = []
    used = set()
    for d in input_dirs:
        base = os.path.basename(os.path.normpath(d))
        sid = _source_id(base)
        while sid in used:
            sid = (sid + "_x")[:41]
        used.add(sid)
        prov = _read_json(os.path.join(d, "provenance.json")) or {}
        n_runs = len(_read_jsonl(os.path.join(d, "runs.jsonl")))
        n_reqs = len(_read_jsonl(os.path.join(d, "requests.jsonl")))
        desc = (f"Grid output '{base}': {n_runs} run records, "
                f"{n_reqs} request records")
        url = prov.get("github_run_url")
        if url:
            desc += f"; Actions run: {url}"
        sources.append({"id": sid, "grade": "measured",
                        "sha256": _sha256_file(os.path.join(d, "runs.jsonl")),
                        "command": f"python -m agent.run_grid --out {d}",
                        "description": desc})
    sha = _prereg_sha()
    desc = ("Pre-registration experiments/PREREG.md, sections 6-10 specify this "
            "analysis")
    if sha:
        desc += f"; registration commit {sha}"
    sources.append({"id": "prereg", "grade": "documented-record",
                    "sha256": _sha256_file(str(PREREG_PATH)),
                    "command": f"git show {sha}:experiments/PREREG.md" if sha
                    else "experiments/PREREG.md",
                    "description": desc})
    return sources, sha


def _num_or_none(v):
    """A finite number, or None (provenance fields may be missing or malformed)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return float(v)


def _agent_models(az, all_runs):
    """Exact model ids per price tier, by id: '-flash-lite' -> lite, else '-flash'.

    Looks at analysable runs first, then at every run record (smoke/control
    runs carry the same ids); None when no run of that tier exists.
    """
    def ids(runs, tier):
        found = set()
        for r in runs:
            mid = str(r.get("model") or "")
            is_lite = "-flash-lite" in mid
            if (tier == "lite" and is_lite) or (tier == "flash" and not is_lite
                                                and "-flash" in mid):
                found.add(mid)
        return sorted(found)

    out = {}
    for tier in ("flash", "lite"):
        found = ids(az, tier) or ids(all_runs, tier)
        out[tier] = ", ".join(found) if found else None
    return out


def _h1_ready(az):
    splits = {r.get("split") for r in az}
    return "train" in splits and "test" in splits


def _h2_ready(az, requests):
    ids = {r["run_id"] for r in az if (r.get("turns") or 0) >= 3}
    return any(q.get("run_id") in ids for q in requests)


def _h3_ready(az):
    need = {("carry_all", 2), ("carry_all", 8), ("reset_summary", 2), ("reset_summary", 8)}
    by_qm = {}
    for r in az:
        by_qm.setdefault((r["question_id"], r["model"]), set()).add((r["policy"], int(r["k"])))
    return any(cells >= need for cells in by_qm.values())


def _h4_ready(az):
    return any(r.get("em") is not None for r in az)


def run_evidence(input_dirs, n_boot, n_perm, control1_path):
    """Compute the full evidence object (schema_version 1 envelope) from grid dirs."""
    all_runs, all_reqs, not_run, inputs = [], [], [], []
    for d in input_dirs:
        all_runs.extend(_read_jsonl(os.path.join(d, "runs.jsonl")))
        all_reqs.extend(_read_jsonl(os.path.join(d, "requests.jsonl")))
        not_run.extend(_read_jsonl(os.path.join(d, "not_run.jsonl")))
        prov = _read_json(os.path.join(d, "provenance.json")) or {}
        prov["dir"] = os.path.basename(os.path.normpath(d))
        inputs.append(prov)

    # Review item 1: final attempt per run_id; step-set check per analysable run.
    final_reqs = final_attempt_requests(all_reqs)
    mismatch_ids = request_log_mismatch_ids(all_runs, all_reqs)
    az = [r for r in analysable(all_runs) if r["run_id"] not in mismatch_ids]
    az_ids = {r["run_id"] for r in az}
    az_reqs = [q for q in final_reqs if q.get("run_id") in az_ids]

    hypotheses = {}
    if _h1_ready(az):
        hypotheses["H1"] = {"status": "evaluated", **h1(az, n_boot=n_boot, seed=BOOT_SEED)}
    else:
        hypotheses["H1"] = {"status": "not yet run",
                            "reason": "no analysable runs in the train or test split"}

    if _h2_ready(az, final_reqs):
        hypotheses["H2"] = {"status": "evaluated",
                            **h2(az, final_reqs, n_boot=n_boot, seed=BOOT_SEED)}
    else:
        hypotheses["H2"] = {"status": "not yet run",
                            "reason": "no analysable runs with T >= 3 and request records"}

    if _h3_ready(az):
        hypotheses["H3"] = {"status": "evaluated", **h3(az, n_boot=n_boot, seed=BOOT_SEED)}
    else:
        hypotheses["H3"] = {"status": "not yet run",
                            "reason": "no (question, model) with all four policy x k cells"}

    if _h4_ready(az):
        hypotheses["H4"] = {"status": "evaluated", **h4(az, n_boot=n_boot, seed=BOOT_SEED)}
    else:
        hypotheses["H4"] = {"status": "not yet run",
                            "reason": "no analysable runs with accuracy metrics"}

    h5_out = h5(az, final_reqs, n_boot=n_boot, seed=BOOT_SEED)
    if not az:
        # Nothing has run yet: H5 is pending, not inapplicable.
        hypotheses["H5"] = {"status": "not yet run", "reason": "no analysable runs yet"}
    else:
        hypotheses["H5"] = h5_out if h5_out.get("status") == "not run" \
            else {"status": "evaluated", **h5_out}

    controls = {}
    if control1_path:
        c1 = _read_json(control1_path)
        controls["C1_token_accounting"] = c1 if c1 is not None else {"status": "not run"}
    else:
        controls["C1_token_accounting"] = {"status": "not run",
                                           "reason": "no --control1 input provided"}

    broken = [r for r in all_runs if r.get("control") == "broken_reset"]
    if not broken:
        controls["C2_broken_policy"] = {"status": "not run",
                                        "reason": "no broken_reset control runs present"}
    else:
        bids = {r["run_id"] for r in broken}
        c2 = control2(all_runs, final_reqs, broken,
                      [q for q in final_reqs if q.get("run_id") in bids],
                      n_boot=n_boot, seed=BOOT_SEED)
        controls["C2_broken_policy"] = c2

    if hypotheses.get("H1", {}).get("status") == "evaluated":
        c3 = control3(az, n_perm=n_perm, seed=PERM_SEED, n_boot=n_boot)
        controls["C3_label_shuffle"] = {"status": "passed" if c3["passed"] else "failed", **c3}
    else:
        # Without an evaluable H1 the shuffle has nothing to falsify: a vacuous pass is not a pass.
        controls["C3_label_shuffle"] = {"status": "not yet run",
                                        "reason": "H1 not evaluable on the inputs"}

    cells = []
    grouped = {}
    for r in az:
        grouped.setdefault((r["provider"], r["model"], r["policy"], int(r["k"])), []).append(r)
    for (provider, model, pol, k), rs in sorted(grouped.items()):
        cached = [r.get("total_cached") for r in rs]
        has_cached = any(c is not None for c in cached)
        cells.append({
            "provider": provider, "model": model, "policy": pol, "k": k,
            "n_ok": len(rs),
            "mean_turns": _mean([r.get("turns") for r in rs]),
            "mean_total_input": _mean([r.get("total_input") for r in rs]),
            "median_total_input": _median([r.get("total_input") for r in rs]),
            "mean_P": _mean([r.get("P") for r in rs]),
            "mean_output": _mean([r.get("total_output") for r in rs]),
            "mean_latency_s": _mean([r.get("latency_s") for r in rs]),
            "em": _mean([r.get("em") for r in rs]),
            "f1": _mean([r.get("f1") for r in rs]),
            "usd_total": round(_sum([r.get("usd") for r in rs]), 6),
            "cached_share": ((_sum(cached) / _sum([r.get("total_input") for r in rs]))
                             if has_cached else None),
        })

    aggregates = {
        "prereg": {"sha": _prereg_sha()},
        "provenance": {"analysis_git_sha": _git(["rev-parse", "HEAD"]), "inputs": inputs},
        "n_runs_analysable": len(az),
        "n_questions": len({r["question_id"] for r in az}),
        "cells": cells,
        "not_run": not_run,
        "exclusions": exclusions(all_runs, request_mismatch_ids=mismatch_ids),
        "tokenledger": _tokenledger_report(az_reqs),
        "hypotheses": hypotheses,
        "controls": controls,
        "cloud_spend_usd": round(_sum([_num_or_none(p.get("cloud_spend_total"))
                                       for p in inputs]), 6),
        "run_usd_total": round(_sum([r.get("usd") for r in all_runs]), 6),
        "agent_models": _agent_models(az, all_runs),
    }

    sources, prereg_sha = _build_sources(input_dirs)
    notes = [
        "Requests are deduplicated per run_id to the final attempt (max `attempt`).",
        "Analysable runs whose request steps != range(turns) are excluded and "
        "counted under 'request log mismatch'.",
        "Control runs (broken_reset, smoke) never enter H1-H5 and are reported "
        "separately in exclusions.control_runs.",
        "Amendment 1: H1-H4 use all analysable custom-agent runs regardless of "
        "provider; H5 uses analysable runs whose cached_input_tokens are non-null; "
        "per-model results are reported alongside.",
        "All CIs are 95% percentile intervals from a cluster bootstrap over "
        "questions; a CI with <95% valid replicates is marked unstable and "
        "demotes the decision to 'not supported (unstable CI)'.",
    ]
    return {
        "schema_version": 1,
        "kind": "agent_runs",
        "provenance": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "generator": "experiments/analyze.py",
            "notes": notes,
            "sources": sources,
        },
        "aggregates": aggregates,
        "figures": build_figures(aggregates, n_boot, n_perm,
                                 data_source=sources[0]["id"], prereg_sha=prereg_sha),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="experiments/analyze.py",
                                description="Pre-registered analysis (PREREG.md 6-10).")
    p.add_argument("--input", action="append", required=True,
                   help="grid output dir (runs.jsonl, requests.jsonl, ...); repeatable")
    p.add_argument("--out", required=True, help="output evidence JSON path")
    p.add_argument("--n-boot", type=int, default=2000, help="cluster bootstrap replicates")
    p.add_argument("--n-perm", type=int, default=200, help="label-shuffle permutations")
    p.add_argument("--control1", default=None, help="C1 token-accounting result JSON")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    evidence = run_evidence(args.input, n_boot=args.n_boot, n_perm=args.n_perm,
                            control1_path=args.control1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # URLs keep their scheme: tools/validate_evidence.py accepts https:// (f64acc9).
    text = json.dumps(_clean(evidence), indent=2)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text + "\n")
    h = evidence["aggregates"]["hypotheses"]
    print(f"wrote {args.out} "
          f"(runs_analysable={evidence['aggregates']['n_runs_analysable']} "
          f"H1={h['H1'].get('decision', h['H1']['status'])} "
          f"H2={h['H2'].get('decision', h['H2']['status'])} "
          f"H3={h['H3'].get('decision', h['H3']['status'])} "
          f"H5={h['H5'].get('decision', h['H5']['status'])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
