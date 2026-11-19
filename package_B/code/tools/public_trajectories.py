#!/usr/bin/env python3
"""Cache-read share of input in PUBLIC agent trajectories (external-validity check).

Question: does "cache reads are most of an agent's input" hold beyond one developer's Claude Code
logs? This reads publicly released benchmark trajectories that record per-call prompt-cache usage
and computes, per system (scaffold x model), the cache-read share of input:

    input = uncached + cache read + cache write        (the paper's definition)
    share = cache read / input

Scope. Benchmark runs have no human idle pauses, so this speaks to cache-read COMPOSITION only,
never to idle expiry. Scaffolds also differ in how (and whether) they enable caching.

Two stages, run where the data lives (the Linux box; CPU only):

  fetch  --data DIR [--n 50] [--seed 2027]   list each source at a pinned revision, sample N
                                             trajectories per system (seeded, documented),
                                             download them read-only, write DIR/manifest.json
                                             with every file's URL and sha256.
  build  --data DIR [--out evidence/public_trajectories.json]
                                             parse, aggregate, write the evidence file
                                             (aggregates only: no message content).

Field conventions (per-call usage as recorded by each scaffold):
  * litellm messages (mini-swe-agent, live-SWE-agent): prompt_tokens, cached read from
    cache_read_input_tokens or prompt_tokens_details.cached_tokens, write from
    cache_creation_input_tokens. Whether prompt_tokens already includes the write differs by
    litellm version, so it is DETECTED per system: if any call has prompt < read + write, the
    prompt excludes the write (Anthropic through older litellm); otherwise it includes it.
  * Moatless Tools: nodes[].completions.*.attempts[].usage {prompt, cache_read, cache_write};
    same detection.
  * Anthropic-native logs (aide): input_tokens is the uncached part.
  * Harbor ATIF (Terminal-Bench 2): steps[].metrics {prompt_tokens, cached_tokens}; prompt
    includes the read, and the write is not separately recorded (it stays inside prompt_tokens,
    so the share is unaffected; checked against cost_usd for one Anthropic call by hand).
  * Claude Code session JSONL: tools/claude_usage.py, the parser used for the paper's own logs
    (one record per message id).
  A system whose calls carry no read field is "not_reported"; one whose read fields are all zero
  is "not_used". Neither is evidence of a 0% share, and neither enters the min/median/max.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_lib import EVIDENCE, figure, sha256_text, utc_now, write_json  # noqa: E402

GENERATOR = "tools/public_trajectories.py"
OUT = EVIDENCE / "public_trajectories.json"
SEED, N_PER_SYSTEM = 2027, 50
WORKERS = 8
SOURCE_ID = "public_traj"

S3 = "https://swe-bench-submissions.s3.amazonaws.com"
S3_NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
HF_DS = "harborframework/terminal-bench-2-leaderboard"
HF_REV = "572b2614be2c0cb2527e14f5b1e4026f1072e6c1"
GH_EXPERIMENTS_SHA = "40f164d5b8f1d249bf95a6df8b74b577fd8e519d"

SOURCES = {
    "s3": {"description": "SWE-bench leaderboard submission trajectories (swe-bench/experiments "
                          "metadata.yaml assets -> public S3 bucket swe-bench-submissions)",
           "url": "https://github.com/swe-bench/experiments",
           "revision": f"swe-bench/experiments@{GH_EXPERIMENTS_SHA}; S3 objects pinned by sha256",
           "licence": "none declared (no LICENSE file in swe-bench/experiments; public bucket, "
                      "read-only; only aggregates are derived and nothing is redistributed)"},
    "hf": {"description": "Terminal-Bench 2.0 leaderboard submissions (Harbor job folders with ATIF "
                          "trajectories and agent session logs)",
           "url": f"https://huggingface.co/datasets/{HF_DS}",
           "revision": f"{HF_DS}@{HF_REV}",
           "licence": "apache-2.0 (dataset card)"},
}

# One row per system = scaffold x model, as submitted. expect is what the probe saw (documented
# before the bulk run); the measured status is what build() reports.
SYSTEMS: list[dict[str, Any]] = [
    {"id": "mini_opus45", "scaffold": "mini-swe-agent", "version": "1.16.0",
     "model": "claude-opus-4-5-20251101", "benchmark": "SWE-bench Verified", "source": "s3",
     "prefix": "bash-only/20251124_mini-v1.16.0_claude-opus-4-5-20251101/trajs/",
     "suffix": ".traj.json", "format": "litellm_messages"},
    {"id": "mini_sonnet45", "scaffold": "mini-swe-agent", "version": "1.13.3",
     "model": "claude-sonnet-4-5-20250929", "benchmark": "SWE-bench Verified", "source": "s3",
     "prefix": "bash-only/20250929_mini-v1.13.3_sonnet-4-5-20250929/trajs/",
     "suffix": ".traj.json", "format": "litellm_messages"},
    {"id": "mini_gpt52", "scaffold": "mini-swe-agent", "version": "1.17.2",
     "model": "gpt-5.2-2025-12-11", "benchmark": "SWE-bench Verified", "source": "s3",
     "prefix": "bash-only/20251211_mini-v1.17.2_gpt-5.2-2025-12-11/trajs/", "suffix": ".traj.json",
     "format": "litellm_messages"},
    {"id": "livesweagent_opus45", "scaffold": "live-SWE-agent", "version": "",
     "model": "claude-opus-4-5", "benchmark": "SWE-bench Verified", "source": "s3",
     "prefix": "verified/20251215_livesweagent_claude-opus-4-5/trajs/", "suffix": ".traj.json",
     "format": "litellm_messages"},
    {"id": "moatless_sonnet4", "scaffold": "Moatless Tools", "version": "",
     "model": "claude-sonnet-4-20250514", "benchmark": "SWE-bench Verified", "source": "s3",
     "prefix": "verified/20250611_moatless_claude-4-sonnet-20250514/trajs/",
     "suffix": "/trajectory.json", "format": "moatless"},
    {"id": "aide_sonnet45", "scaffold": "aide-platform", "version": "",
     "model": "claude-sonnet-4-5", "benchmark": "SWE-bench Verified", "source": "s3",
     "prefix": "verified/20251102_aide-platform_claude4.5sonnet/trajs/", "suffix": ".json",
     "format": "aide"},
    {"id": "terminus2_opus46", "scaffold": "Terminus 2", "version": "2.0.0",
     "model": "claude-opus-4-6", "benchmark": "Terminal-Bench 2.0", "source": "hf",
     "submission": "Terminus2__Claude-Opus-4.6", "format": "atif"},
    {"id": "terminus2_deepseek32", "scaffold": "Terminus 2", "version": "2.0.0",
     "model": "deepseek-v3.2", "benchmark": "Terminal-Bench 2.0", "source": "hf",
     "submission": "Terminus2__DeepSeek-V3.2", "format": "atif"},
    {"id": "terminus2_kimik25", "scaffold": "Terminus 2", "version": "2.0.0", "model": "kimi-k2.5",
     "benchmark": "Terminal-Bench 2.0", "source": "hf", "submission": "Terminus2__Kimi-k2.5",
     "format": "atif"},
    {"id": "terminuskira_opus46", "scaffold": "Terminus-KIRA", "version": "3.3.0",
     "model": "claude-opus-4-6", "benchmark": "Terminal-Bench 2.0", "source": "hf",
     "submission": "Terminus-KIRA__Claude-Opus-4.6", "format": "atif"},
    {"id": "geminicli_gemini31pro", "scaffold": "Gemini CLI", "version": "0.34.0",
     "model": "gemini-3.1-pro-preview", "benchmark": "Terminal-Bench 2.0", "source": "hf",
     "submission": "Gemini_CLI__Gemini-3.1-Pro-Preview", "format": "atif"},
    {"id": "judy_opus46", "scaffold": "Judy", "version": "0.7.0", "model": "claude-opus-4-6",
     "benchmark": "Terminal-Bench 2.0", "source": "hf", "submission": "Judy__Claude-Opus-4.6",
     "format": "atif"},
    {"id": "claudecode_opus46", "scaffold": "Claude Code", "version": "2.1.86 (WozCode plugin)",
     "model": "claude-opus-4-6", "benchmark": "Terminal-Bench 2.0", "source": "hf",
     "submission": "WozCode__Claude-Opus-4.6", "format": "claude_code"},
]


# --------------------------------------------------------------------------- parsing
def _int(v: Any) -> int | None:
    """A token count, or None when the field is absent/null (absent is not zero)."""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return int(v)
    return None


def _first(*vals: Any) -> int | None:
    for v in vals:
        iv = _int(v)
        if iv is not None:
            return iv
    return None


def _prompt_call(u: dict[str, Any], model: Any) -> dict[str, Any] | None:
    """An OpenAI/litellm-shaped usage dict -> raw call (prompt basis)."""
    if not isinstance(u, dict) or _int(u.get("prompt_tokens")) is None:
        return None
    det = u.get("prompt_tokens_details") if isinstance(u.get("prompt_tokens_details"), dict) else {}
    read = _first(u.get("cache_read_input_tokens"), u.get("cache_read_tokens"), det.get("cached_tokens"))
    write = _first(u.get("cache_creation_input_tokens"), u.get("cache_write_tokens"),
                   det.get("cache_creation_tokens"), det.get("cache_write_tokens"))
    return {"basis": "prompt", "p": _int(u["prompt_tokens"]), "r": read, "w": write,
            "model": str(model) if model else "?"}


def calls_litellm_messages(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """mini-swe-agent / live-SWE-agent: one call per assistant message carrying a response usage."""
    out = []
    for m in doc.get("messages") or []:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        resp = (m.get("extra") or {}).get("response") if isinstance(m.get("extra"), dict) else None
        if not isinstance(resp, dict):
            continue
        c = _prompt_call(resp.get("usage"), resp.get("model"))
        if c:
            out.append(c)
    return out


def calls_moatless(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Moatless Tools: every attempt of every completion of every node."""
    out = []
    for node in doc.get("nodes") or []:
        comps = node.get("completions") if isinstance(node, dict) else None
        if not isinstance(comps, dict):
            continue
        for comp in comps.values():
            if not isinstance(comp, dict):
                continue
            for att in comp.get("attempts") or []:
                if isinstance(att, dict):
                    c = _prompt_call(att.get("usage"), comp.get("model"))
                    if c:
                        out.append(c)
    return out


def calls_aide(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Anthropic-native usage per interaction: input_tokens is the uncached part."""
    out = []
    for it in doc.get("llm_interactions") or []:
        resp = it.get("response") if isinstance(it, dict) else None
        u = resp.get("usage") if isinstance(resp, dict) else None
        if not isinstance(u, dict) or _int(u.get("input_tokens")) is None:
            continue
        out.append({"basis": "anthropic", "p": _int(u["input_tokens"]),
                    "r": _int(u.get("cache_read_input_tokens")),
                    "w": _int(u.get("cache_creation_input_tokens")), "model": str(resp.get("model") or "?")})
    return out


def calls_atif(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Harbor ATIF: agent steps with metrics. prompt_tokens includes the read; no separate write."""
    model = (doc.get("agent") or {}).get("model_name") if isinstance(doc.get("agent"), dict) else None
    out = []
    for st in doc.get("steps") or []:
        if not isinstance(st, dict) or st.get("source") != "agent":
            continue
        m = st.get("metrics")
        if not isinstance(m, dict) or _int(m.get("prompt_tokens")) is None:
            continue
        out.append({"basis": "prompt", "p": _int(m["prompt_tokens"]), "r": _int(m.get("cached_tokens")),
                    "w": None, "model": str(model or "?")})
    return out


def calls_claude_code(projects: Path) -> list[dict[str, Any]]:
    """Claude Code session JSONL under <trial>/agent/sessions/projects, via the paper's own parser."""
    import claude_usage  # local import: only this format needs it
    transcripts, _stats = claude_usage.extract(projects=Path(projects))
    out = []
    for t in transcripts:
        for r in t.requests:
            u = r.usage
            out.append({"basis": "anthropic", "p": u["input_tokens"], "r": u["cache_read_input_tokens"],
                        "w": u["cache_creation_input_tokens"], "model": r.model,
                        "subagent": t.kind == "subagent"})
    return out


# --------------------------------------------------------------------------- aggregation
def resolve_convention(calls: list[dict[str, Any]]) -> str:
    """For prompt-basis calls with a write field: does prompt_tokens already include the write?"""
    pw = [c for c in calls if c["basis"] == "prompt" and (c["w"] or 0) > 0]
    if not pw:
        return "n/a"
    if any(c["p"] < (c["r"] or 0) + c["w"] for c in pw):
        return "prompt_excludes_write"
    return "prompt_includes_write"


def _tokens(c: dict[str, Any], conv: str) -> tuple[int, int, int | None, bool]:
    """(uncached, read, write or None if unreported, anomalous) for one call."""
    r = c["r"] or 0
    if c["basis"] == "anthropic":
        return c["p"], r, (c["w"] or 0), False
    if c["w"] is None:                      # write not reported: it stays inside prompt_tokens
        unc = c["p"] - r
        return max(unc, 0), r, None, unc < 0
    w = c["w"]
    unc = c["p"] - r - (w if conv == "prompt_includes_write" else 0)
    return max(unc, 0), r, w, unc < 0


def _pct(num: float, den: float) -> float | None:
    return 100.0 * num / den if den > 0 else None


def summarise(per_traj: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Aggregate one system's sampled trajectories (each a list of raw calls)."""
    allc = [c for t in per_traj for c in t]
    conv = resolve_convention(allc)
    unc = read = write = anomalies = 0
    write_seen = False
    shares = []
    for t in per_traj:
        tu = tr = tw = 0
        for c in t:
            u, r, w, bad = _tokens(c, conv)
            anomalies += bad
            tu, tr = tu + u, tr + r
            if w is not None:
                write_seen = True
                tw += w
        unc, read, write = unc + tu, read + tr, write + tw
        s = _pct(tr, tu + tr + tw)
        if t and s is not None:
            shares.append(s)
    read_reported = any(c["r"] is not None for c in allc)
    total = unc + read + write
    if not read_reported:
        status = "not_reported"
    elif read == 0:
        status = "not_used"
    else:
        status = "enabled"
    out = {
        "n_trajectories": sum(1 for t in per_traj if t),
        "n_empty_trajectories": sum(1 for t in per_traj if not t),
        "n_calls": len(allc),
        "n_subagent_calls": sum(1 for c in allc if c.get("subagent")),
        "uncached": unc, "read": read, "write": write if write_seen else None, "input": total,
        "convention": ("write_not_reported" if not write_seen else
                       "anthropic_native" if allc and all(c["basis"] == "anthropic" for c in allc) else conv),
        "status": status,
        "cache_read_share_pct": _pct(read, total) if status == "enabled" else None,
        "median_trajectory_share_pct": statistics.median(shares) if status == "enabled" and shares else None,
        "anomalous_calls": anomalies,
        "models": sorted({c["model"] for c in allc}),
    }
    if not write_seen:
        out["uncached_plus_write"] = unc
        out["uncached"] = None
    return out


def figures_from(rows: list[dict[str, Any]], source: str) -> dict[str, Any]:
    en = [r for r in rows if r["status"] == "enabled"]
    shares = sorted(r["cache_read_share_pct"] for r in en)
    if not shares:
        raise ValueError("no caching-enabled system: nothing to report")
    f = {
        "PubSystemsCachingN": figure(len(en), "int", source, "public systems (scaffold x model) with caching in use"),
        "PubScaffoldsCachingN": figure(len({r["scaffold"] for r in en}), "int", source,
                                       "distinct scaffolds among the caching systems"),
        "PubSystemsNoCacheN": figure(len(rows) - len(en), "int", source,
                                     "public systems with no cache reads recorded (not used or not reported)"),
        "PubShareMinPct": figure(shares[0], "pct1", source, "lowest cache-read share of input, caching systems"),
        "PubShareMedianPct": figure(statistics.median(shares), "pct1", source,
                                    "median cache-read share of input across caching systems"),
        "PubShareMaxPct": figure(shares[-1], "pct1", source, "highest cache-read share of input, caching systems"),
        "PubTrajectoriesN": figure(sum(r["n_trajectories"] for r in rows), "int", source,
                                   "sampled public trajectories with at least one call"),
        "PubCallsN": figure(sum(r["n_calls"] for r in rows), "int", source, "LLM calls in the sampled trajectories"),
    }
    cc = [r for r in en if r["scaffold"] == "Claude Code"]
    if cc:
        f["PubClaudeCodeSharePct"] = figure(cc[0]["cache_read_share_pct"], "pct1", source,
                                            "cache-read share, Claude Code on Terminal-Bench 2.0 (public)")
        f["PubClaudeCodeTrajN"] = figure(cc[0]["n_trajectories"], "int", source,
                                         "sampled Claude Code trajectories (Terminal-Bench 2.0)")
    return f


# --------------------------------------------------------------------------- reading the sample
def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def system_calls(root: Path, sysdef: dict[str, Any], units: list[str]) -> list[list[dict[str, Any]]]:
    fmt = sysdef["format"]
    per = []
    for u in units:
        d = root / sysdef["id"] / u
        if fmt == "claude_code":
            proj = d / "agent" / "sessions" / "projects"
            per.append(calls_claude_code(proj) if proj.is_dir() else [])
            continue
        files = sorted(p for p in d.rglob("*.json") if p.is_file()) if d.is_dir() else []
        calls: list[dict[str, Any]] = []
        for p in files:
            doc = _load(p)
            calls += {"litellm_messages": calls_litellm_messages, "moatless": calls_moatless,
                      "aide": calls_aide, "atif": calls_atif}[fmt](doc)
        per.append(calls)
    return per


def _manifest_hash(files: list[dict[str, Any]]) -> str:
    return sha256_text("".join(f"{f['path']}\t{f['sha256']}\n" for f in sorted(files, key=lambda f: f["path"])))


def build(root: Path, out: Path = OUT, systems: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    systems = systems or SYSTEMS
    man = _load(root / "manifest.json")
    rows = []
    for s in systems:
        ms = man["systems"][s["id"]]
        row = {"id": s["id"], "scaffold": s["scaffold"], "version": s.get("version", ""),
               "declared_model": s.get("model", ""),
               "benchmark": s.get("benchmark", ""), "source": s["source"],
               "submission": s.get("submission") or s.get("prefix", ""), "format": s["format"],
               "n_sampled": len(ms["units"]), "n_missing": len(ms["missing"]), "n_files": len(ms["files"]),
               "files_sha256": _manifest_hash(ms["files"]), "sampled_units": ms["units"]}
        row.update(summarise(system_calls(root, s, ms["units"])))
        rows.append(row)
    all_files = [f for s in systems for f in man["systems"][s["id"]]["files"]]
    used = sorted({s["source"] for s in systems})
    src_text = json.dumps({k: man["sources"][k] for k in used}, sort_keys=True)
    sources = [{"id": SOURCE_ID, "grade": "documented-record",
                "sha256": _manifest_hash(all_files) if all_files else sha256_text(src_text),
                "command": f"python {GENERATOR} fetch --data <dir> --n {man['n_per_system']} --seed {man['seed']}"
                           f" && python {GENERATOR} build --data <dir>",
                "description": "Per-call usage recorded in publicly released agent trajectories: "
                               + "; ".join(f"{man['sources'][k]['description']} ({man['sources'][k]['url']}, "
                                           f"{man['sources'][k]['revision']}, licence: {man['sources'][k]['licence']})"
                                           for k in used)
                               + ". sha256 is over the sorted '<path> TAB <sha256>' list of every sampled file."}]
    data = {
        "schema_version": 1,
        "kind": "public_trajectories",
        "provenance": {
            "generated_at": utc_now(),
            "generator": GENERATOR,
            "notes": [
                "Composition only: benchmark runs have no human idle pauses, so nothing here bears on "
                "idle expiry of cache entries.",
                "input = uncached + cache read + cache write (as in the paper); share = read / input. Where a "
                "scaffold records no separate write (ATIF, OpenAI-style usage) the write stays inside "
                "prompt_tokens, so input and the share are unaffected but uncached and write are merged.",
                "Scaffolds differ in caching configuration (explicit cache_control placement, provider "
                "automatic caching, proxies); systems with no read field are 'not_reported' and systems "
                "whose reads are all zero are 'not_used'; neither is a 0% data point.",
                f"Sampling: {man['n_per_system']} units per system drawn with random.Random('<seed>:<system id>')"
                f".sample over the sorted unit list (seed {man['seed']}); a unit is one SWE-bench instance "
                "trajectory or one Terminal-Bench trial (task x attempt).",
                "System = one leaderboard submission (scaffold x model); several share a scaffold. "
                "declared_model is the submission's stated model; models lists the model field of the recorded "
                "responses (they can differ, e.g. a litellm alias, and Claude Code also calls a small model).",
            ],
            "sources": sources,
        },
        "aggregates": {"systems": rows, "seed": man["seed"], "n_per_system": man["n_per_system"],
                       "sources": {k: man["sources"][k] for k in used}},
        "figures": figures_from(rows, SOURCE_ID),
    }
    write_json(out, data)
    return data


# --------------------------------------------------------------------------- fetching (network)
def _get(url: str, tries: int = 4) -> bytes:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "public-trajectories-research/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            if i == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if i == tries - 1:
                raise
        time.sleep(2 * (i + 1))
    raise RuntimeError("unreachable")


def _s3_keys(prefix: str) -> list[str]:
    keys, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        root = ET.fromstring(_get(f"{S3}/?{urllib.parse.urlencode(q)}"))
        keys += [c.find("s:Key", S3_NS).text for c in root.findall("s:Contents", S3_NS)]
        t = root.find("s:NextContinuationToken", S3_NS)
        if t is None or not t.text:
            return keys
        token = t.text


def _hf_tree(path: str, recursive: bool = False) -> list[dict[str, Any]]:
    url = (f"https://huggingface.co/api/datasets/{HF_DS}/tree/{HF_REV}/{urllib.parse.quote(path)}"
           + ("?recursive=true" if recursive else ""))
    out = []
    while url:
        req = urllib.request.Request(url, headers={"User-Agent": "public-trajectories-research/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            out += json.loads(r.read())
            link = r.headers.get("Link") or ""
        url = ""
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part[part.find("<") + 1:part.find(">")]
    return out


def _hf_url(path: str) -> str:
    return f"https://huggingface.co/datasets/{HF_DS}/resolve/{HF_REV}/{urllib.parse.quote(path)}"


def _units(s: dict[str, Any]) -> dict[str, list[tuple[str, str]]]:
    """unit -> [(url, relative path under the unit dir)]; listing only, no download."""
    if s["source"] == "s3":
        units: dict[str, list[tuple[str, str]]] = {}
        for k in _s3_keys(s["prefix"]):
            if not k.endswith(s["suffix"]):
                continue
            rel = k[len(s["prefix"]):]
            unit = rel.split("/")[0]
            if s["suffix"].startswith(".") and "/" not in rel:
                unit = rel[: -len(s["suffix"])]
            units.setdefault(unit, []).append((f"{S3}/{urllib.parse.quote(k)}", Path(rel).name))
        return units
    base = f"submissions/terminal-bench/2.0/{s['submission']}"
    units = {}
    for job in (x for x in _hf_tree(base) if x["type"] == "directory"):
        for tr in (x for x in _hf_tree(job["path"]) if x["type"] == "directory"):
            unit = tr["path"][len(base) + 1:]
            units[unit] = [] if s["format"] == "claude_code" else [
                (_hf_url(tr["path"] + "/agent/trajectory.json"), "agent/trajectory.json")]
    return units


def _fetch_unit(root: Path, s: dict[str, Any], u: str, files: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Download one sampled unit (skipping files already on disk); 404s are recorded as absent."""
    if s["format"] == "claude_code":
        tr = f"submissions/terminal-bench/2.0/{s['submission']}/{u}"
        files = [(_hf_url(x["path"]), "agent/sessions/" + x["path"].split("/agent/sessions/", 1)[1])
                 for x in _hf_tree(tr + "/agent/sessions", True)
                 if x["type"] == "file" and "/projects/" in x["path"] and x["path"].endswith(".jsonl")]
    got = []
    for url, rel in files:
        dest = root / s["id"] / u / rel
        if not dest.is_file():
            try:
                data = _get(url)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    continue
                raise
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
        got.append({"path": f"{s['id']}/{u}/{rel}", "url": url,
                    "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()})
    return got


def fetch(root: Path, n: int = N_PER_SYSTEM, seed: int = SEED, only: list[str] | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    mpath = root / "manifest.json"
    man = _load(mpath) if mpath.is_file() else {"seed": seed, "n_per_system": n, "systems": {}}
    if (man["seed"], man["n_per_system"]) != (seed, n):
        sys.exit("FAIL: existing manifest was drawn with a different seed or N")
    man["sources"] = SOURCES
    for s in SYSTEMS:
        if only and s["id"] not in only:
            continue
        units = _units(s)
        names = sorted(units)
        pick = sorted(random.Random(f"{seed}:{s['id']}").sample(names, min(n, len(names))))
        entry = {"n_available": len(names), "units": [], "missing": [], "files": []}
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:  # latency-bound link: overlap requests
            results = list(pool.map(lambda u: (u, _fetch_unit(root, s, u, units[u])), pick))
        for u, got in results:
            (entry["units"] if got else entry["missing"]).append(u)
            entry["files"] += got
        man["systems"][s["id"]] = entry
        mpath.write_text(json.dumps(man, indent=1, sort_keys=True), encoding="utf-8")
        print(f"{s['id']}: {len(entry['units'])} units, {len(entry['missing'])} missing, "
              f"{len(entry['files'])} files of {len(names)} available", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--data", type=Path, required=True)
    f.add_argument("--n", type=int, default=N_PER_SYSTEM)
    f.add_argument("--seed", type=int, default=SEED)
    f.add_argument("--only", nargs="*")
    b = sub.add_parser("build")
    b.add_argument("--data", type=Path, required=True)
    b.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch(a.data, a.n, a.seed, a.only)
        return 0
    data = build(a.data, a.out)
    for r in data["aggregates"]["systems"]:
        share = r["cache_read_share_pct"]
        print(f"{r['id']:24s} {r['status']:13s} traj={r['n_trajectories']:3d} calls={r['n_calls']:5d} "
              f"share={'-' if share is None else f'{share:.1f}%'} conv={r['convention']} anomalies={r['anomalous_calls']}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
