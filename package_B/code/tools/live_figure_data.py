#!/usr/bin/env python3
"""Per-session data behind the live-check figure (pre-registered: PREREG_GATEWAY_LIVE.md).

  python tools/live_figure_data.py produce --run RUN_DIR --run RUN_DIR [--out FILE]
  python tools/live_figure_data.py dat [--check]

produce  (where the raw records are; read-only) reads each run directory's requests.jsonl and
         manifest.json and writes evidence/gateway_live_sessions.json: for every analysed E1
         session of the two Ollama arms, the measured request's prompt-processing seconds and
         that value divided by the SAME arm's median at gap 0. Latency is never compared
         across arms (they run on different hardware), so only the within-arm ratio is plotted.
         Sessions are selected as the registered analyzer selects them (a session record, not
         contaminated, E1, a request with role "measured"; PP = prompt_eval_s for Ollama arms).
         It refuses (BLOCKED) unless each raw file's sha256 equals the analyzer's source hash
         and, per arm and gap, the session count and PP minimum, median and maximum equal the
         analyzer's in evidence/gateway_live.json. The analyzer and its decisions are unchanged.
dat      (CI) renders papers/agent2027-cost/generated/live_sessions.dat for pgfplots from the
         committed evidence; --check exits 1 if the committed file is stale.

The spark-vllm arm is not plotted: it reports no prompt-evaluation time (its stand-in is time to
first token) and tests a different hypothesis (H3).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_lib import REPO, figure, sha256_bytes, utc_now, write_json  # noqa: E402

GENERATOR = "tools/live_figure_data.py"
SESSIONS = REPO / "evidence" / "gateway_live_sessions.json"
ANALYZER = REPO / "evidence" / "gateway_live.json"
DAT = REPO / "papers" / "agent2027-cost" / "generated" / "live_sessions.dat"
ANALYZE_PY = REPO / "algorithm" / "benchmarks" / "gateway_live" / "analyze.py"

PLOTTED_ARMS = ("box-ollama", "spark-ollama")
GAPS = ("0", "2", "4", "6", "10")
# (arm, condition) -> series; anything else in a plotted arm is BLOCKED rather than guessed.
SERIES = {**{("box-ollama", g): "local" for g in GAPS},
          **{("spark-ollama", g): "control" for g in GAPS},
          ("box-ollama", "10+ping"): "ping"}
SERIES_ORDER = ("local", "control", "ping")
# Display only: series sit side by side at each gap and sessions are spread within a series,
# so that coinciding values stay visible. The true gap is kept in the "gap" column.
SERIES_OFFSET = {"local": -0.3, "control": 0.3, "ping": -0.3}  # ping and local never share a y
SESSION_SPREAD = 0.18
TOL = 1e-9


def blocked(msg: str) -> None:
    raise SystemExit(f"BLOCKED: {msg}")


# ------------------------------------------------------------------ selection (as the analyzer)
def sessions_from_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Analysed E1 sessions of the plotted arms: arm, condition, block, pp_s."""
    bundles: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in records:
        if rec.get("arm") not in PLOTTED_ARMS:
            continue
        b = bundles.setdefault((rec["arm"], rec["session"]), {"requests": [], "session": None})
        if rec.get("kind") == "request":
            b["requests"].append(rec)
        elif rec.get("kind") == "session":
            b["session"] = rec
    rows = []
    for (arm, _), b in bundles.items():
        s = b["session"]
        if s is None or s.get("contaminated"):
            continue
        cond = str(s.get("condition"))
        if s.get("experiment") == "E2" or cond == "growth":
            continue
        measured = next((r for r in b["requests"] if r.get("role") == "measured"), None)
        if measured is None:
            continue
        rows.append({"arm": arm, "condition": cond, "block": s.get("block"),
                     "pp_s": float(measured["prompt_eval_s"])})
    return sorted(rows, key=_row_key)


def _row_key(r: dict[str, Any]) -> tuple:
    return (r["arm"], _gap_of(r["condition"]), r["condition"], r["block"] or 0, r["pp_s"])


def _gap_of(cond: str) -> int:
    head = cond.split("+", 1)[0]
    return int(head) if head.isdigit() else -1


def with_relative(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Adds series, gap_min and rel = pp_s / median pp_s of the same arm at gap 0."""
    base = {}
    for arm in sorted({r["arm"] for r in rows}):
        zero = [r["pp_s"] for r in rows if r["arm"] == arm and r["condition"] == "0"]
        if not zero:
            blocked(f"{arm} has no analysed gap-0 session to normalise by")
        base[arm] = statistics.median(zero)
        if base[arm] <= 0:
            blocked(f"{arm} gap-0 median prompt-processing time is not positive")
    out = []
    for r in rows:
        series = SERIES.get((r["arm"], r["condition"]))
        if series is None:
            blocked(f"{r['arm']} condition {r['condition']!r} is not a plotted series")
        out.append({"arm": r["arm"], "condition": r["condition"], "block": r["block"],
                    "pp_s": r["pp_s"], "series": series, "gap_min": _gap_of(r["condition"]),
                    "rel": r["pp_s"] / base[r["arm"]]})
    return sorted(out, key=_row_key)


# ------------------------------------------------------------------ agreement with the analyzer
def cross_check(rows: list[dict[str, Any]], analyzer: dict[str, Any], raw_sha: dict[str, str]) -> None:
    """BLOCKED unless these sessions are the analyzer's: same raw bytes, and per arm and gap the
    same n and PP minimum, median and maximum."""
    src = {s["id"]: s["sha256"] for s in analyzer["provenance"]["sources"]}
    for arm in PLOTTED_ARMS:
        want = src.get(f"raw_{arm.replace('-', '_')}")
        if want is None or raw_sha.get(arm) != want:
            blocked(f"{arm} raw records sha256 {raw_sha.get(arm)} != analyzer source {want}")
        e1 = analyzer["aggregates"]["arms"].get(arm, {}).get("e1", {})
        mine: dict[str, list[float]] = {}
        for r in rows:
            if r["arm"] == arm:
                mine.setdefault(r["condition"], []).append(r["pp_s"])
        theirs = {c for c, v in e1.items() if v.get("n")}
        if set(mine) != theirs:
            blocked(f"{arm} gaps differ: here {sorted(mine)}, analyzer {sorted(theirs)}")
        for cond, vals in mine.items():
            a = e1[cond]
            got = (len(vals), min(vals), statistics.median(vals), max(vals))
            exp = (a["n"], a["pp"]["min"], a["pp"]["median"], a["pp"]["max"])
            if got[0] != exp[0] or any(abs(g - e) > TOL for g, e in zip(got[1:], exp[1:])):
                blocked(f"{arm} gap {cond}: (n, min, median, max) {got} != analyzer {exp}")


# ------------------------------------------------------------------ produce
def _sha_lf(path: Path) -> str:
    """sha256 with CRLF normalised to LF, as the analyzer hashes its raw files."""
    return sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def _load_analyzer_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("gateway_live_analyze", ANALYZE_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _medians(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for series in SERIES_ORDER:
        by_gap: dict[int, list[float]] = {}
        for r in rows:
            if r["series"] == series:
                by_gap.setdefault(r["gap_min"], []).append(r[key])
        if by_gap:
            out[series] = {str(g): statistics.median(v) for g, v in sorted(by_gap.items())}
    return out


def produce(run_dirs: list[Path], analyzer_path: Path, out_path: Path) -> dict[str, Any]:
    ana_mod = _load_analyzer_module()
    records: list[dict[str, Any]] = []
    raw_sha: dict[str, str] = {}
    sources: list[dict[str, Any]] = []
    lifetimes: dict[str, int] = {}
    for d in map(Path, run_dirs):
        manifest = ana_mod.load_manifest(d / "manifest.json")
        arm = manifest["arm"]
        if arm not in PLOTTED_ARMS:
            blocked(f"run {d.name} is arm {arm}, not one of {PLOTTED_ARMS}")
        if arm in raw_sha:
            blocked(f"two runs for arm {arm}")
        recs = ana_mod.load_records([d / "requests.jsonl"])
        if {r["arm"] for r in recs} != {arm}:
            blocked(f"run {d.name}: records are not all arm {arm}")
        records += recs
        raw_sha[arm] = _sha_lf(d / "requests.jsonl")
        lifetimes[arm] = int(round(float(manifest["lifetime_s"]) / 60))
        sid = arm.replace("-", "_")
        sources += [
            {"id": f"raw_{sid}", "grade": "measured", "sha256": raw_sha[arm],
             "command": f"algorithm/benchmarks/gateway_live/run_live.py --arm {arm}",
             "description": f"per-request timings and token counts (numbers only, no content), "
                            f"arm {arm}, runs/{d.name}/requests.jsonl"},
            {"id": f"manifest_{sid}", "grade": "measured", "sha256": _sha_lf(d / "manifest.json"),
             "description": f"run manifest (server, model, keep-alive lifetime), runs/{d.name}/manifest.json"},
        ]
    missing = [a for a in PLOTTED_ARMS if a not in raw_sha]
    if missing:
        blocked(f"no run directory for {missing}")
    analyzer = json.loads(Path(analyzer_path).read_text(encoding="utf-8"))
    rows = sessions_from_records(records)
    cross_check(rows, analyzer, raw_sha)
    rows = with_relative(rows)
    sources.append({"id": "gateway_live", "grade": "measured", "sha256": _sha_lf(Path(analyzer_path)),
                    "command": "algorithm/benchmarks/gateway_live/analyze.py",
                    "description": "registered analyzer output (evidence/gateway_live.json); per arm and "
                                   "gap, n and PP min/median/max checked equal to these sessions"})
    arms = {arm: {"lifetime_min": lifetimes[arm],
                  "gap0_median_pp_s": statistics.median(r["pp_s"] for r in rows
                                                        if r["arm"] == arm and r["condition"] == "0"),
                  "n_sessions": sum(1 for r in rows if r["arm"] == arm)} for arm in PLOTTED_ARMS}
    evidence = {
        "schema_version": 1, "kind": "gateway_live_sessions",
        "provenance": {"generated_at": utc_now(), "generator": GENERATOR, "sources": sources, "notes": [
            "Pre-registered experiment (algorithm/experiments/PREREG_GATEWAY_LIVE.md); this file only "
            "re-expresses the analyzer's sessions for a figure and decides nothing.",
            "pp_s: prompt_eval_s of the session's measured request (role measured); rel: pp_s over the "
            "median pp_s of the same arm at gap 0. Arms differ in hardware, so only rel is plotted.",
            "Series: local = box-ollama gaps 0-10 min; control = spark-ollama gaps 0-10 min (negative "
            "control only); ping = box-ollama 10 min with keepalive pings, relative to box-ollama gap 0.",
        ]},
        "aggregates": {"arms": arms, "sessions": rows,
                       "series_medians": _medians(rows, "rel"),
                       "series_medians_pp_s": _medians(rows, "pp_s")},
        "figures": {
            "LiveFigGaps": figure(",".join(str(g) for g in sorted({r["gap_min"] for r in rows})), "text",
                                  "raw_box_ollama", "idle gaps (min) of the plotted E1 sessions: the "
                                  "live-check figure's x ticks"),
        },
    }
    write_json(Path(out_path), evidence)
    return evidence


# ------------------------------------------------------------------ dat for pgfplots
def render_dat(evidence: dict[str, Any]) -> str:
    rows = sorted(evidence["aggregates"]["sessions"],
                  key=lambda r: (SERIES_ORDER.index(r["series"]), r["gap_min"], r["block"] or 0, r["rel"]))
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["series"], r["gap_min"]), []).append(r)
    lines = [
        "# GENERATED by tools/live_figure_data.py dat from evidence/gateway_live_sessions.json -- DO NOT EDIT.",
        "# One row per analysed E1 session. y = prompt-processing time / same arm's gap-0 median.",
        "# gap = true idle gap (min); x = gap + display offset (series side by side, sessions spread).",
        "series gap block x ylocal ycontrol yping",
    ]
    for (series, gap), members in groups.items():
        n = len(members)
        for i, r in enumerate(members):
            x = gap + SERIES_OFFSET[series] + (i - (n - 1) / 2) * SESSION_SPREAD
            ys = ["nan"] * len(SERIES_ORDER)
            if not math.isfinite(r["rel"]) or r["rel"] <= 0:
                blocked(f"non-positive or non-finite relative value in {series} gap {gap}")
            ys[SERIES_ORDER.index(series)] = f"{r['rel']:.6f}"
            lines.append(f"{series} {gap} {r['block']} {x:.3f} {' '.join(ys)}")
    return "\n".join(lines) + "\n"


def dat_command(evidence_path: Path, dat_path: Path, check: bool) -> int:
    text = render_dat(json.loads(Path(evidence_path).read_text(encoding="utf-8")))
    if check:
        current = dat_path.read_text(encoding="utf-8").replace("\r\n", "\n") if dat_path.is_file() else None
        if current != text:
            print(f"FAIL: {dat_path.name} is stale or missing (run `python tools/live_figure_data.py dat` "
                  "and commit)")
            return 1
        print(f"PASS: {dat_path.name} matches {Path(evidence_path).name}")
        return 0
    dat_path.parent.mkdir(parents=True, exist_ok=True)
    dat_path.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {dat_path.name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("produce", help="raw records -> evidence/gateway_live_sessions.json")
    p.add_argument("--run", action="append", required=True, type=Path, help="run directory (repeat)")
    p.add_argument("--analyzer", type=Path, default=ANALYZER)
    p.add_argument("--out", type=Path, default=SESSIONS)
    d = sub.add_parser("dat", help="evidence -> generated/live_sessions.dat")
    d.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "produce":
        ev = produce(args.run, args.analyzer, args.out)
        print(json.dumps({"arms": ev["aggregates"]["arms"],
                          "series_medians": ev["aggregates"]["series_medians"]}, indent=1))
        return 0
    return dat_command(SESSIONS, DAT, args.check)


if __name__ == "__main__":
    raise SystemExit(main())
