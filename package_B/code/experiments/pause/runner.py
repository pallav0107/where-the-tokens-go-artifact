#!/usr/bin/env python3
"""Dispatcher for the pause experiment (experiments/PREREG_pause.md). Runs on the execution host.

    python3 runner.py build                      # agent images for the pilot + 24 candidates
    python3 runner.py infra up [--stub]          # egress proxy (+ API stub for dry runs)
    python3 runner.py run --plan dry             # stub API, never contacts the real API
    python3 runner.py run --plan authcheck       # real endpoint, invalid key: must fail at auth
    python3 runner.py run --plan pilot --key-file <.env holding ANTHROPIC_API_KEY>
    python3 runner.py run --plan full  --key-file <...> --pause-seconds <lifetime + 60>
    python3 runner.py ledger

Each run is one fresh container from the task's agent image (official SWE-bench instance image +
pinned Claude Code + pause hook), on a docker --internal network whose only way out is the
allowlist proxy (api.anthropic.com:443). The API key is read from --key-file (only the
ANTHROPIC_API_KEY line), written to a 0600 env file that docker reads at create time and that is
deleted right after; it never appears on a command line or in a log. After the run the runner
copies out /out (CLI result, patch, pause-hook log) and the transcripts, costs the run from the
transcripts at list prices, appends the ledger, and removes the container.

Budget stop (ledger.may_start): no run starts if spent + max per-run cost x concurrency > USD 140.
Conditions A and B of a task start together (same API conditions); infra failures re-run once.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ledger as ledger_mod  # noqa: E402

REPO = HERE.parents[1]
PRICES = REPO / "evidence" / "params" / "anthropic_list_prices.json"
CC_VERSION = "2.1.326"
CC_IMAGE = f"cachepause-cc:{CC_VERSION}"
MODEL = "claude-sonnet-5-5"
MAX_TURNS = 60
WALL_SECONDS = 2700            # agent working time; condition B gets its pauses on top
PAUSE_AT = "5,15,25"
# Conditions (PREREG_pause.md and its Amendment 1): pause points, pause seconds (None = --pause-seconds),
# one-hour caching env. C has one 3660 s pause after tool call 10 on the first N_C tasks.
CONDITIONS = {"A": ("5,15,25", 0, False), "B": ("5,15,25", None, False), "C": ("10", 3660, True)}
N_C = 10
NET_INT, NET_OUT = "cachepause-int", "cachepause-egress"
PROXY, STUB = "cachepause-proxy", "cachepause-stub"
PROXY_IMAGE = "python:3.12-slim"
PROMPT = """{problem}

The repository for this issue is checked out at /testbed (your current directory), with its
Python environment ready. Resolve the issue above by editing the repository's source files. Do not
modify the existing tests. Work autonomously until the fix is complete; do not ask questions."""
ENV_COMMON = {
    "DISABLE_AUTOUPDATER": "1", "DISABLE_TELEMETRY": "1", "DISABLE_ERROR_REPORTING": "1",
    "DISABLE_BUG_COMMAND": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "IS_SANDBOX": "1",
}
LOCK = threading.Lock()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sh(*args: str, check: bool = True, capture: bool = True, timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(list(args), check=check, text=True, timeout=timeout,
                          stdout=subprocess.PIPE if capture else None, stderr=subprocess.PIPE if capture else None)


def agent_image(instance_id: str) -> str:
    return f"cachepause/{instance_id.lower()}:cc{CC_VERSION}"


def load_tasks(work: Path) -> dict:
    return json.loads((work / "tasks.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- images
def session_image(session_id: str) -> str:
    return f"cachepause/session-{session_id.lower()}:cc{CC_VERSION}"


def load_sessions(work: Path) -> dict:
    return json.loads((work / "sessions.json").read_text(encoding="utf-8"))


def cmd_build(args) -> int:
    if args.sessions:
        return build_sessions(args)
    t = load_tasks(args.work)
    todo = [t["pilot"]] + t["tasks"]
    if args.only:
        todo = [x for x in todo if x["instance_id"] in args.only]
    status_path = args.work / "builds.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    for task in todo:
        iid = task["instance_id"]
        if status.get(iid, {}).get("ok") and not args.force:
            continue
        r = sh("nice", "-n", "10", "docker", "build", "--network", "host", "-q",
               "--build-arg", f"BASE={task['image']}", "--build-arg", f"CC={CC_IMAGE}",
               "-t", agent_image(iid), str(HERE / "image"), check=False)
        ok = r.returncode == 0
        status[iid] = {"ok": ok, "image": agent_image(iid), "base": task["image"], "at": now(),
                       "id": r.stdout.strip() if ok else None, "error": None if ok else r.stderr[-400:]}
        print(("built " if ok else "FAILED ") + iid, flush=True)
        status_path.write_text(json.dumps(status, indent=1))
    return 0


def build_sessions(args) -> int:
    """Amendment 2: one agent image per session, on the instance image of the session's first task."""
    d = load_sessions(args.work)
    status_path = args.work / "builds_sessions.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    for sess in [d["pilot"]] + d["sessions"]:
        sid = sess["session_id"]
        if (args.only and sid not in args.only) or (status.get(sid, {}).get("ok") and not args.force):
            continue
        r = sh("nice", "-n", "10", "docker", "build", "--network", "host", "-q",
               "--build-arg", f"BASE={sess['image']}", "--build-arg", f"CC={CC_IMAGE}",
               "-t", session_image(sid), str(HERE / "image"), check=False)
        ok, missing = r.returncode == 0, []
        if ok:   # every task's base commit must be reachable in the image's repository
            missing = [t["base_commit"] for t in sess["tasks"]
                       if sh("docker", "run", "--rm", "--entrypoint", "git", session_image(sid), "-C", "/testbed",
                             "cat-file", "-e", t["base_commit"] + "^{commit}", check=False).returncode]
            ok = not missing
        status[sid] = {"ok": ok, "image": session_image(sid), "base": sess["image"], "at": now(),
                       "error": None if ok else (r.stderr[-400:] if r.returncode else f"missing base commits {missing}")}
        print(("built " if ok else "FAILED ") + sid, flush=True)
        status_path.write_text(json.dumps(status, indent=1))
    return 0


# ---------------------------------------------------------------- infra
def cmd_infra(args) -> int:
    # the stub is only replaced by a stub "up" or removed by "down": a real-mode "up" leaves a running dry run's
    # stub alone (real plans never point at it)
    for name in (PROXY, STUB) if (args.action == "down" or args.stub) else (PROXY,):
        sh("docker", "rm", "-f", name, check=False)
    if args.action == "down":
        return 0
    for net, internal in ((NET_INT, True), (NET_OUT, False)):
        if sh("docker", "network", "inspect", net, check=False).returncode:
            sh("docker", "network", "create", *(["--internal"] if internal else []), net)
    allow = [] if args.stub else ["api.anthropic.com"]
    sh("docker", "run", "-d", "--name", PROXY, "--network", NET_OUT, "--dns-option", "use-vc",
       "--restart", "unless-stopped", "-v", f"{(HERE / 'egress_proxy.py').resolve()}:/app/egress_proxy.py:ro",
       PROXY_IMAGE, "python3", "-u", "/app/egress_proxy.py", "--port", "3128", "--allow", *(allow or ["none.invalid"]))
    sh("docker", "network", "connect", NET_INT, PROXY)
    if args.stub:
        sh("docker", "run", "-d", "--name", STUB, "--network", NET_INT,
           "-e", f"STUB_STEPS={args.stub_steps}", "-e", f"STUB_TTL_SECONDS={args.stub_ttl}",
           "-v", f"{(HERE / 'stub_api.py').resolve()}:/app/stub_api.py:ro", "-v", f"{(args.work / 'stub').resolve()}:/log",
           PROXY_IMAGE, "python3", "-u", "/app/stub_api.py", "--port", "8080", "--log", "/log/stub.jsonl")
    print(f"infra up: proxy allow={allow or 'NOTHING'} stub={'yes' if args.stub else 'no'}")
    return 0


# ---------------------------------------------------------------- one run
def read_key(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*(?:export\s+)?ANTHROPIC_API_KEY\s*=\s*(.+?)\s*$", line)
        if m:
            return m.group(1).strip().strip("'\"")
    raise SystemExit("BLOCKED: no ANTHROPIC_API_KEY line in the key file")


def classify(out: Path, cost: dict) -> tuple[str, str]:
    """ok | infra_fail, with a reason. Agent outcomes (max turns, timeout) are ok; API/infra errors are not."""
    exit_info = json.loads((out / "exit.json").read_text()) if (out / "exit.json").exists() else {}
    res = {}
    try:
        res = json.loads((out / "result.json").read_text())
    except (OSError, json.JSONDecodeError):
        pass
    if not cost.get("requests"):
        return "infra_fail", "no API request in the transcripts"
    text = json.dumps(res)[:4000].lower()
    if res.get("is_error") and any(s in text for s in ("api error", "authentication", "overloaded", "rate limit", "529", "401")):
        return "infra_fail", "API error: " + str(res.get("subtype"))
    if exit_info.get("rc") in (124, 137):
        return "ok", "wall-clock timeout"
    return "ok", str(res.get("subtype") or f"rc={exit_info.get('rc')}")


def run_one(spec: dict, args, lg: ledger_mod.Ledger, prices: dict) -> dict:
    run_id = spec["run_id"]
    d = args.runs / run_id
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    # unique per runs directory, so two runners (e.g. a dry run beside the pilot) never share a container
    name = "cp_" + re.sub(r"[^a-zA-Z0-9_.-]", "_", f"{args.runs.name}_{run_id}")
    sh("docker", "rm", "-f", name, check=False)
    pause_s, pause_at = spec["pause_seconds"], spec["pause_at"]
    wall = args.wall_seconds + len(pause_at.split(",")) * pause_s
    env = {**ENV_COMMON, "MODEL": args.model, "MAX_TURNS": str(args.max_turns), "WALL_SECONDS": str(wall),
           "PAUSE_AT": pause_at, "PAUSE_SECONDS": str(pause_s), "HTTPS_PROXY": f"http://{PROXY}:3128",
           "https_proxy": f"http://{PROXY}:3128", "NO_PROXY": STUB, "no_proxy": STUB}
    env["CC_MAX_BUDGET_USD"] = str(args.run_budget_usd)
    if args.cache_1h or spec.get("cache_1h"):
        env["ENABLE_PROMPT_CACHING_1H"] = "1"
    if args.plan == "dry":
        env["ANTHROPIC_BASE_URL"] = f"http://{STUB}:8080"
    keyfile = d / ".key.env"
    keyfile.touch(mode=0o600)
    os.chmod(keyfile, 0o600)
    key = "dry-run-not-a-key" if args.plan in ("dry", "authcheck") else read_key(args.key_file)
    keyfile.write_text(f"ANTHROPIC_API_KEY={key}\n", encoding="utf-8")
    meta = {"run_id": run_id, "instance_id": spec["instance_id"], "condition": spec["condition"],
            "attempt": spec["attempt"], "pilot": spec["plan"] != "full", "plan": spec["plan"],
            "pause_seconds": pause_s, "pause_at": pause_at, "model": args.model, "max_turns": args.max_turns,
            "wall_seconds": wall, "image_override": spec.get("image"), "cache_1h_env": bool(args.cache_1h or spec.get("cache_1h")), "run_budget_usd": args.run_budget_usd,
            "image": agent_image(spec["instance_id"]), "claude_code": CC_VERSION}
    try:
        create = ["docker", "create", "--name", name, "--network", NET_INT, "--env-file", str(keyfile),
                  "--cpus", "2", "--memory", "8g", "--pids-limit", "4096"]
        for k, v in env.items():
            create += ["-e", f"{k}={v}"]
        sh(*create, spec.get("image") or agent_image(spec["instance_id"]))
    finally:
        keyfile.unlink(missing_ok=True)
    (d / "prompt.txt").write_text(PROMPT.format(problem=spec["problem"]), encoding="utf-8")
    sh("docker", "cp", str(d / "prompt.txt"), f"{name}:/out/prompt.txt")
    meta["start"] = now()
    sh("docker", "start", name)
    deadline = time.time() + wall + 600
    while time.time() < deadline:
        st = sh("docker", "inspect", "-f", "{{.State.Running}}", name, check=False).stdout.strip()
        if st != "true":
            break
        time.sleep(10)
    else:
        sh("docker", "kill", name, check=False)
        meta["host_killed"] = True
    meta["end"] = now()
    sh("docker", "cp", f"{name}:/out", str(d / "out"), check=False)
    sh("docker", "cp", f"{name}:/agent-home/.claude/projects", str(d / "projects"), check=False)
    meta["container_exit"] = sh("docker", "inspect", "-f", "{{.State.ExitCode}}", name, check=False).stdout.strip()
    sh("docker", "rm", "-f", name, check=False)
    for p in d.rglob("*"):                       # defence in depth: the key must never be on disk
        if p.is_file() and key.encode() in p.read_bytes():
            p.write_bytes(p.read_bytes().replace(key.encode(), b"[REDACTED]"))
            meta["key_redacted_from"] = meta.get("key_redacted_from", []) + [p.name]
    cost = ledger_mod.transcript_cost(d / "projects", prices) if (d / "projects").is_dir() else {"requests": 0}
    meta["status"], meta["reason"] = classify(d / "out", cost)
    res = {}
    try:
        res = json.loads((d / "out" / "result.json").read_text())
    except (OSError, json.JSONDecodeError):
        pass
    pauses = (d / "out" / "pause" / "pauses.jsonl")
    n_inj = sum(1 for x in pauses.read_text().splitlines() if x.strip() and json.loads(x)["seconds"] > 0) if pauses.exists() else 0
    row = {"run_id": run_id, "instance_id": spec["instance_id"], "condition": spec["condition"], "plan": spec["plan"],
           "attempt": spec["attempt"], "start": meta["start"], "end": meta["end"], "status": meta["status"],
           "reason": meta["reason"], "requests": cost.get("requests", 0), "tokens": cost.get("tokens"),
           "usd": round(cost.get("usd", 0.0), 6), "cli_usd": res.get("total_cost_usd"), "num_turns": res.get("num_turns"),
           "models": cost.get("models"), "unpriced_requests": cost.get("unpriced_requests"), "pauses_injected": n_inj,
           "first_request": cost.get("first_request"), "last_request": cost.get("last_request"),
           "usd_by_hour": cost.get("usd_by_hour"), "usd_by_day": cost.get("usd_by_day"),
           "request_ids": cost.get("request_ids")}
    (d / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    with LOCK:
        lg.append(row)
    print(f"{now()} done {run_id}: {meta['status']} ({meta['reason']}) requests={row['requests']} "
          f"usd={row['usd']:.3f} cli_usd={row['cli_usd']} pauses={n_inj}", flush=True)
    return row


# ---------------------------------------------------------------- one chained session (Amendment 2)
SESSION_CAP_USD = 15.0
NEXT_TASK = """Next task. The repository at /testbed has been reset to a different commit for this new issue;
your earlier changes are no longer present.

"""


def _exec_json(name: str, path: str) -> dict:
    r = sh("docker", "exec", name, "cat", path, check=False)
    try:
        return json.loads(r.stdout) if r.returncode == 0 else {}
    except json.JSONDecodeError:
        return {}


def task_api_error(res: dict) -> bool:
    text = json.dumps(res)[:4000].lower()
    return bool(res.get("is_error")) and any(x in text for x in ("api error", "authentication", "overloaded",
                                                                 "rate limit", "529", "401"))


# ---------------------------------------------------------------- Amendment 3 (PREREG_pause.md)
# Specificity arm: at each task boundary b (after task b) the listed invalidators are injected; every
# configuration change stays in force for the rest of the session (cumulative), so it changes the
# request only at its own boundary. "pause" = the condition-B 360 s sleep in the dispatcher.
A3_SWITCH_MODEL = "claude-opus-5-5"
A3_MCP_CONFIG = "/opt/pause/mcp.json"
A3_TEMPLATES = {"S1": {"1": ("mcp",), "2": ("tools", "pause"), "3": ("model",)},
                "S2": {"1": ("tools",), "2": ("mcp", "pause"), "3": ("model", "pause")}}
A3_K_SESSIONS = 4          # concurrency arm: the first 4 sessions in session order, condition-B pauses, one at a time
A3_FILES = ("session.sh", "mcp_stub.js", "mcp.json")   # copied into an S container (images keep Amendment 2's)


def a3_task_settings(template: dict, k: int, model: str) -> tuple[str, list[str]]:
    """Model and extra CLI arguments of task k: every injection at a boundary before task k stays in force."""
    m, extra = model, []
    for b in range(1, k):
        for x in template.get(str(b), ()):
            if x == "model":
                m = A3_SWITCH_MODEL
            elif x == "mcp":
                extra += ["--mcp-config", A3_MCP_CONFIG]
            elif x == "tools":
                extra += ["--disallowedTools", "Edit"]
    return m, extra


def a3_pauses(template: dict, pause_seconds: int) -> dict:
    """Pause after task b: pause_seconds where "pause" is injected, else a 0 s marker (as condition A)."""
    return {b: (pause_seconds if "pause" in template.get(b, ()) else 0) for b in ("1", "2", "3")}


def run_session(spec: dict, args, lg: ledger_mod.Ledger, prices: dict) -> dict:
    run_id = spec["run_id"]
    d = args.runs / run_id
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    name = "cp_" + re.sub(r"[^a-zA-Z0-9_.-]", "_", f"{args.runs.name}_{run_id}")
    sh("docker", "rm", "-f", name, check=False)
    env = {**ENV_COMMON, "MODEL": args.model, "MAX_TURNS": str(args.max_turns), "WALL_SECONDS": str(args.wall_seconds),
           "CC_MAX_BUDGET_USD": str(args.session_budget_usd), "HTTPS_PROXY": f"http://{PROXY}:3128",
           "https_proxy": f"http://{PROXY}:3128", "NO_PROXY": STUB, "no_proxy": STUB}
    if spec.get("cache_1h"):
        env["ENABLE_PROMPT_CACHING_1H"] = "1"
    if args.plan == "dry":
        env["ANTHROPIC_BASE_URL"] = f"http://{STUB}:8080"
    image = spec.get("image") or session_image(spec["session_id"])
    keyfile = d / ".key.env"
    keyfile.touch(mode=0o600)
    os.chmod(keyfile, 0o600)
    key = "dry-run-not-a-key" if args.plan in ("dry", "authcheck") else read_key(args.key_file)
    keyfile.write_text(f"ANTHROPIC_API_KEY={key}\n", encoding="utf-8")
    meta = {"run_id": run_id, "instance_id": spec["session_id"], "session_id": spec["session_id"],
            "condition": spec["condition"], "attempt": spec["attempt"], "pilot": spec["plan"] != "full",
            "plan": spec["plan"], "design": "chained-session (Amendment 2)", "pauses_planned": spec["pauses"],
            "model": args.model, "max_turns_per_task": args.max_turns, "wall_seconds_per_task": args.wall_seconds,
            "session_budget_usd": args.session_budget_usd, "cache_1h_env": bool(spec.get("cache_1h")),
            "image": image, "claude_code": CC_VERSION, "tasks": []}
    try:
        create = ["docker", "create", "--name", name, "--network", NET_INT, "--env-file", str(keyfile),
                  "--cpus", "2", "--memory", "8g", "--pids-limit", "4096", "--entrypoint", "/opt/pause/session.sh"]
        for k, v in env.items():
            create += ["-e", f"{k}={v}"]
        sh(*create, image, "hold")
    finally:
        keyfile.unlink(missing_ok=True)
    inject = spec.get("inject")
    if inject is not None:                       # Amendment 3: the per-task-settings driver and the MCP server
        meta["inject"] = inject
        for f in A3_FILES:                       # docker cp keeps the host mode: stage with 0755 (git stores 0644)
            staged = d / f"a3_{f}"
            shutil.copyfile(HERE / "image" / f, staged)
            os.chmod(staged, 0o755)
            sh("docker", "cp", str(staged), f"{name}:/opt/pause/{f}")
    meta["start"] = now()
    sh("docker", "start", name)
    pauses: list[dict] = []
    injections: list[dict] = []
    spent, api_error = 0.0, False
    for k, task in enumerate(spec["tasks"], start=1):
        budget = round(args.session_budget_usd - spent, 2)
        if budget < 0.5:
            meta["stopped"] = f"session budget used before task {k}"
            break
        prompt = (NEXT_TASK if k > 1 else "") + PROMPT.format(problem=task["problem"])
        (d / f"prompt_{k}.txt").write_text(prompt, encoding="utf-8")
        sh("docker", "exec", name, "mkdir", "-p", f"/out/task_{k}")
        sh("docker", "cp", str(d / f"prompt_{k}.txt"), f"{name}:/out/task_{k}/prompt.txt")
        if inject is not None:
            model, extra = a3_task_settings(inject, k, args.model)
            if model != args.model:
                (d / f"model_{k}.txt").write_text(model + "\n", encoding="utf-8")
                sh("docker", "cp", str(d / f"model_{k}.txt"), f"{name}:/out/task_{k}/model")
            if extra:
                (d / f"extra_args_{k}.txt").write_text("".join(a + "\n" for a in extra), encoding="utf-8")
                sh("docker", "cp", str(d / f"extra_args_{k}.txt"), f"{name}:/out/task_{k}/extra_args")
            if k > 1:
                injections.append({"boundary": k - 1, "types": list(inject.get(str(k - 1), ())),
                                   "model": model, "extra_args": extra, "dispatched": now()})
        try:
            sh("docker", "exec", "-e", f"CC_MAX_BUDGET_USD={budget}", name, "/opt/pause/session.sh", "task", str(k),
               task["base_commit"], check=False, timeout=args.wall_seconds + 300)
        except subprocess.TimeoutExpired:
            meta["stopped"] = f"host timeout in task {k}"
        res = _exec_json(name, f"/out/task_{k}/result.json")
        ex = _exec_json(name, f"/out/task_{k}/exit.json")
        cost = float(res.get("total_cost_usd") or 0.0)
        spent = max(spent, cost)          # the CLI reports the continued session's cumulative cost
        err = task_api_error(res)
        api_error = api_error or err or not res
        meta["tasks"].append({"k": k, "instance_id": task["instance_id"], "base_commit": task["base_commit"],
                              "subtype": res.get("subtype"), "terminal_reason": res.get("terminal_reason"),
                              "num_turns": res.get("num_turns"), "cli_usd": cost, "rc": ex.get("rc"),
                              "start": ex.get("start"), "end": ex.get("end"), "api_error": err,
                              "session_id_cli": res.get("session_id"), "model": ex.get("model")})
        if meta.get("stopped"):
            break
        secs = spec["pauses"].get(str(k))
        if secs is not None and k < len(spec["tasks"]):
            p0 = now()
            time.sleep(secs)
            pauses.append({"call": k, "start": p0, "end": now(), "seconds": secs})
    meta["end"] = now()
    sh("docker", "cp", f"{name}:/out", str(d / "out"), check=False)
    sh("docker", "cp", f"{name}:/agent-home/.claude/projects", str(d / "projects"), check=False)
    sh("docker", "rm", "-f", name, check=False)
    (d / "out" / "pause").mkdir(parents=True, exist_ok=True)
    (d / "out" / "pause" / "pauses.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pauses), encoding="utf-8")
    if inject is not None:                       # Amendment 3 ground truth: what was injected at each boundary
        (d / "out" / "inject").mkdir(parents=True, exist_ok=True)
        (d / "out" / "inject" / "injections.jsonl").write_text("".join(json.dumps(x) + "\n" for x in injections),
                                                               encoding="utf-8")
    for p in d.rglob("*"):                       # defence in depth: the key must never be on disk
        if p.is_file() and key.encode() in p.read_bytes():
            p.write_bytes(p.read_bytes().replace(key.encode(), b"[REDACTED]"))
            meta["key_redacted_from"] = meta.get("key_redacted_from", []) + [p.name]
    cost = ledger_mod.transcript_cost(d / "projects", prices) if (d / "projects").is_dir() else {"requests": 0}
    if not cost.get("requests"):
        meta["status"], meta["reason"] = "infra_fail", "no API request in the transcripts"
    elif api_error:
        meta["status"], meta["reason"] = "infra_fail", "API error or missing result in a task"
    else:
        meta["status"], meta["reason"] = "ok", meta.get("stopped") or "all tasks ran"
    meta["transcripts"] = cost.get("transcripts")
    (d / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    row = {"run_id": run_id, "instance_id": spec["session_id"], "condition": spec["condition"], "plan": spec["plan"],
           "attempt": spec["attempt"], "start": meta["start"], "end": meta["end"], "status": meta["status"],
           "reason": meta["reason"], "requests": cost.get("requests", 0), "tokens": cost.get("tokens"),
           "usd": round(cost.get("usd", 0.0), 6), "cli_usd": round(spent, 6),
           "num_turns": sum(t["num_turns"] or 0 for t in meta["tasks"]), "models": cost.get("models"),
           "unpriced_requests": cost.get("unpriced_requests"), "pauses_injected": sum(p["seconds"] > 0 for p in pauses),
           "first_request": cost.get("first_request"), "last_request": cost.get("last_request"),
           "usd_by_hour": cost.get("usd_by_hour"), "usd_by_day": cost.get("usd_by_day"),
           "request_ids": cost.get("request_ids"), "tasks_run": len(meta["tasks"]), "transcripts": cost.get("transcripts")}
    with LOCK:
        lg.append(row)
    print(f"{now()} done {run_id}: {meta['status']} ({meta['reason']}) tasks={len(meta['tasks'])} requests={row['requests']} "
          f"transcripts={row['transcripts']} usd={row['usd']:.3f} cli_usd={row['cli_usd']:.3f} "
          f"pauses={row['pauses_injected']}", flush=True)
    return row


def session_pauses(cond: str, args) -> tuple[dict, bool]:
    """Amendment 2: pauses after task k in seconds (A logs 0 s markers), and one-hour caching for C."""
    if cond == "A":
        return {"1": 0, "2": 0, "3": 0}, False
    if cond == "B":
        return {"1": args.pause_seconds, "2": args.pause_seconds, "3": args.pause_seconds}, False
    return {"2": args.c_pause_seconds}, True


def session_specs(args) -> list[dict]:
    d = load_sessions(args.work)
    prompts = json.loads((args.work / "data" / "session_prompts.json").read_text(encoding="utf-8"))
    bpath = args.work / "builds_sessions.json"
    builds = json.loads(bpath.read_text()) if bpath.exists() else {}
    if args.plan in ("dry", "pilot", "authcheck"):
        pool = d["sessions"] + [d["pilot"]]
        chosen = [x for x in pool if x["session_id"] == args.session] if args.session else [d["pilot"]]
    else:
        chosen = []
        for x in d["sessions"]:
            b = builds.get(x["session_id"])
            if b is None and getattr(args, "wait_images", False):
                chosen.append(x)            # its thread waits for the image (run --wait-images)
                continue
            if b is None:
                raise SystemExit(f"BLOCKED: session {x['session_id']} has no build attempt yet; run build --sessions first")
            if b.get("ok"):
                chosen.append(x)
            else:
                print(f"WARNING: session {x['session_id']} image failed; reported, not replaced", flush=True)
    if getattr(args, "amend3", None):
        return a3_specs(args, d, chosen, prompts)
    conds = args.conditions or {"authcheck": "A", "dry": "AB", "pilot": "A", "full": "ABC"}[args.plan]

    def spec(x: dict, c: str) -> dict:
        pauses, one_h = session_pauses(c, args)
        return {"run_id": f"{x['session_id']}__{c}__r1", "session_id": x["session_id"], "instance_id": x["session_id"],
                "condition": c, "attempt": 1, "plan": args.plan, "pauses": pauses, "cache_1h": one_h,
                "image": args.image, "tasks": [{**t, "problem": prompts[t["instance_id"]]} for t in x["tasks"]]}
    ab = [spec(x, c) for x in chosen for c in "AB" if c in conds]
    cs = [spec(x, "C") for x in chosen if "C" in conds and (x.get("condition_c") or args.plan != "full")]
    out, k = [], 0
    for i in range(0, len(ab), 4):
        out += ab[i:i + 4]
        if k < len(cs):
            out.append(cs[k])
            k += 1
    return out + cs[k:]


def a3_specs(args, d: dict, chosen: list[dict], prompts: dict) -> list[dict]:
    """Amendment 3. S: every built session under templates S1 and S2 (started together, like A and B).
    K: condition-B pauses on the first A3_K_SESSIONS sessions in session order, run one at a time."""
    def spec(x: dict, c: str) -> dict:
        tmpl = A3_TEMPLATES.get(c)
        pauses = a3_pauses(tmpl, args.pause_seconds) if tmpl else {b: args.pause_seconds for b in ("1", "2", "3")}
        return {"run_id": f"{x['session_id']}__{c}__r1", "session_id": x["session_id"], "instance_id": x["session_id"],
                "condition": c, "attempt": 1, "plan": args.plan, "pauses": pauses, "cache_1h": False,
                "image": args.image, "inject": ({b: list(v) for b, v in tmpl.items()} if tmpl else None),
                "tasks": [{**t, "problem": prompts[t["instance_id"]]} for t in x["tasks"]]}
    if args.amend3 == "K":
        order = [x["session_id"] for x in d["sessions"]][:A3_K_SESSIONS]
        return [spec(x, "K") for x in chosen if x["session_id"] in order]
    return [spec(x, c) for x in chosen for c in ("S1", "S2")]


# ---------------------------------------------------------------- plans
def plan_specs(args) -> list[dict]:
    t = load_tasks(args.work)
    prompts = json.loads((args.work / "data" / "prompts.json").read_text(encoding="utf-8"))
    builds = json.loads((args.work / "builds.json").read_text()) if (args.work / "builds.json").exists() else {}
    if args.plan in ("dry", "authcheck", "pilot"):
        chosen = [t["pilot"]]
    else:
        # spares replace candidates only for recorded build failures (PREREG_pause.md): every candidate up to
        # the twentieth usable one must have been attempted, so an image still downloading cannot be skipped
        chosen = []
        for x in t["tasks"]:
            b = builds.get(x["instance_id"])
            if b is None:
                raise SystemExit(f"BLOCKED: {x['instance_id']} (rank {x['rank']}) has no build attempt yet; run build first")
            if b.get("ok"):
                chosen.append(x)
            if len(chosen) == t["n_use"]:
                break
        if len(chosen) < t["n_use"]:
            print(f"WARNING: only {len(chosen)} candidate images built", flush=True)
    conds = args.conditions or {"authcheck": "A", "dry": "AB", "pilot": "AB", "full": "ABC"}[args.plan]

    def spec(x: dict, c: str) -> dict:
        pause_at, secs, one_h = CONDITIONS[c]
        return {"run_id": f"{x['instance_id']}__{c}__r1", "instance_id": x["instance_id"], "condition": c,
                "attempt": 1, "plan": args.plan, "pause_at": pause_at,
                "pause_seconds": args.pause_seconds if secs is None else (args.c_pause_seconds if c == "C" else secs),
                "cache_1h": one_h, "problem": prompts[x["instance_id"]], "image": args.image}
    ab = [spec(x, c) for x in chosen for c in "AB" if c in conds]
    cs = [spec(x, "C") for x in (chosen[:N_C] if args.plan == "full" else chosen)] if "C" in conds else []
    # A and B of a task start together; a C run (mostly idle) is slotted after every two pairs
    out, k = [], 0
    for i in range(0, len(ab), 4):
        out += ab[i:i + 4]
        if k < len(cs):
            out.append(cs[k])
            k += 1
    return out + cs[k:]


def finished(runs: Path, spec: dict) -> bool:
    """Resume: a (task, condition) with an ok run, or with both attempts used, is not run again."""
    metas = [runs / spec["run_id"].replace("__r1", f"__r{a}") / "meta.json" for a in (1, 2)]
    got = [json.loads(m.read_text()) for m in metas if m.exists()]
    return any(g.get("status") == "ok" for g in got) or len(got) == 2


def cmd_run(args) -> int:
    if args.image and args.plan in ("pilot", "full"):
        raise SystemExit("BLOCKED: --image is for dry and authcheck plans only")
    if args.plan in ("pilot", "full") and not args.key_file:
        raise SystemExit("BLOCKED: --key-file is required for a paid plan")
    if args.plan in ("pilot", "full"):
        read_key(args.key_file)                     # fail early, print nothing
    sub = args.runs_subdir or {"dry": "runs-dry", "authcheck": "runs-authcheck", "pilot": "runs", "full": "runs"}[args.plan]
    args.runs = args.work / sub
    args.runs.mkdir(parents=True, exist_ok=True)
    lg = ledger_mod.Ledger(args.runs / "ledger.jsonl")
    prices = json.loads(PRICES.read_text(encoding="utf-8"))["models"]
    specs = session_specs(args) if args.sessions else plan_specs(args)
    one = run_session if args.sessions else run_one
    if args.limit:
        specs = specs[: args.limit]
    skipped = [x["run_id"] for x in specs if finished(args.runs, x)]
    specs = [x for x in specs if x["run_id"] not in skipped]
    if skipped:
        print(f"{now()} resume: {len(skipped)} runs already done, skipped", flush=True)
    in_flight = {"n": 0}
    stop = {"hit": False}
    ab_waiting = {x["run_id"] for x in specs if x["condition"] in "AB"}

    def observed_max() -> float:
        return max([lg.charged(r) for r in lg.rows() if r.get("plan") != "dry"] or [args.prior_run_usd])

    def image_ready(spec: dict) -> bool:
        """--wait-images: block until the session's image is built; False if its build failed."""
        if not (args.sessions and getattr(args, "wait_images", False)):
            return True
        bpath = args.work / "builds_sessions.json"
        while True:
            b = (json.loads(bpath.read_text()) if bpath.exists() else {}).get(spec["session_id"])
            if b and b.get("ok"):
                return True
            if b and not b.get("ok") and "missing base commits" in str(b.get("error")):
                print(f"{now()} not starting {spec['run_id']}: image check failed ({b.get('error')})", flush=True)
                return False
            time.sleep(60)

    first = {"started": False}

    def gated(spec: dict) -> None:
        if getattr(args, "amend3", None) == "K":   # one at a time, with an idle gap before every session but the first
            if first["started"]:
                print(f"{now()} K: idle {args.run_gap_seconds}s before {spec['run_id']}", flush=True)
                time.sleep(args.run_gap_seconds)
            first["started"] = True
        if not image_ready(spec):
            with LOCK:
                ab_waiting.discard(spec["run_id"])
            return
        is_c = spec["condition"] == "C"
        while True:
            with LOCK:
                if stop["hit"]:
                    return
                maxc = lg.max_run_cost(args.prior_run_usd)
                if not is_c:
                    ab_waiting.discard(spec["run_id"])
                    if not ledger_mod.may_start(lg.spent(), maxc, args.concurrency, args.stop_usd):
                        stop["hit"] = True
                        print(f"{now()} BUDGET STOP: spent {lg.spent():.2f} + {maxc:.2f} x {args.concurrency} > "
                              f"{args.stop_usd}; not starting {spec['run_id']}", flush=True)
                        return
                    in_flight["n"] += 1
                    break
                # C (Amendment 1): A/B have priority, so keep room for every A/B run not yet started
                reserve = len(ab_waiting) * observed_max()
                if ledger_mod.may_start(lg.spent() + reserve, maxc, args.concurrency, args.stop_usd):
                    in_flight["n"] += 1
                    break
                if not ab_waiting:
                    print(f"{now()} BUDGET: not starting {spec['run_id']} (C runs come after A/B)", flush=True)
                    return
            time.sleep(60)             # wait for A/B runs to start or finish, then re-check
        try:
            try:
                row = one(spec, args, lg, prices)
            except Exception as e:  # noqa: BLE001  docker/infra error: record it, never crash the dispatcher
                row = {"run_id": spec["run_id"], "instance_id": spec["instance_id"], "condition": spec["condition"],
                       "plan": spec["plan"], "attempt": spec["attempt"], "status": "infra_fail",
                       "reason": f"runner exception {type(e).__name__}", "requests": 0, "usd": 0.0, "cli_usd": None}
                d = args.runs / spec["run_id"]
                d.mkdir(parents=True, exist_ok=True)
                (d / "meta.json").write_text(json.dumps({**row, "pilot": spec["plan"] != "full"}, indent=1))
                with LOCK:
                    lg.append(row)
                print(f"{now()} done {spec['run_id']}: infra_fail ({row['reason']})", flush=True)
            if row["status"] == "infra_fail" and spec["attempt"] == 1 and not stop["hit"]:
                again = {**spec, "attempt": 2, "run_id": spec["run_id"].replace("__r1", "__r2")}
                with LOCK:
                    if not ledger_mod.may_start(lg.spent(), lg.max_run_cost(args.prior_run_usd), args.concurrency, args.stop_usd):
                        stop["hit"] = True
                        return
                one(again, args, lg, prices)
        finally:
            with LOCK:
                in_flight["n"] -= 1

    a3 = getattr(args, "amend3", None)
    print(f"{now()} plan {args.plan}: {len(specs)} runs, concurrency {args.concurrency}, model {args.model}, "
          + (f"Amendment 3 arm {a3}: " + (f"templates {A3_TEMPLATES}, pause {args.pause_seconds}s, switch model "
                                          f"{A3_SWITCH_MODEL}" if a3 == "S" else
                                          f"condition-B pauses {args.pause_seconds}s, one session at a time, "
                                          f"{args.run_gap_seconds}s idle between sessions")
             if a3 else
             f"chained sessions (Amendment 2): B {args.pause_seconds}s between consecutive tasks; C {args.c_pause_seconds}s "
             f"between tasks 2 and 3 with one-hour caching; hook logs tool calls only"
             if args.sessions else
             f"B: {args.pause_seconds}s at tool calls {CONDITIONS['B'][0]}; C: {args.c_pause_seconds}s at tool call "
             f"{CONDITIONS['C'][0]} with one-hour caching")
          + f"; spent so far {lg.spent():.2f}", flush=True)
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(gated, specs))
    print(f"{now()} finished: spent {lg.spent():.2f} USD over {len(lg.rows())} ledger rows", flush=True)
    return 0


def cmd_ledger(args) -> int:
    for sub in ("runs", "runs-dry", "runs-authcheck"):
        lg = ledger_mod.Ledger(args.work / sub / "ledger.jsonl")
        rows = lg.rows()
        if rows:
            print(f"{sub}: {len(rows)} runs, spent {lg.spent():.2f} USD, max run {lg.max_run_cost(0):.2f}")
            for r in rows:
                print(f"  {r['run_id']:<48} {r['status']:<10} req={r['requests']:<4} usd={r['usd']:.3f} cli={r['cli_usd']}"
                      f" {r.get('start')} .. {r.get('end')}")
            per = lg.by_period()
            print("  list-price USD per UTC day: " + json.dumps(per["day"]))
            print("  list-price USD per UTC hour: " + json.dumps(per["hour"]))
            (args.work / sub / "billing_by_period.json").write_text(json.dumps(per, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", type=Path, default=Path.home() / "work" / "cache-pause")
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--force", action="store_true")
    b.add_argument("--only", nargs="+", default=None, help="instance or session ids to build")
    b.add_argument("--sessions", action="store_true", help="Amendment 2: build the chained-session images")
    i = sp.add_parser("infra")
    i.add_argument("action", choices=("up", "down"))
    i.add_argument("--stub", action="store_true")
    i.add_argument("--stub-steps", type=int, default=30)
    i.add_argument("--stub-ttl", type=float, default=20.0)
    r = sp.add_parser("run")
    r.add_argument("--plan", choices=("dry", "authcheck", "pilot", "full"), required=True)
    r.add_argument("--key-file", type=Path)
    r.add_argument("--concurrency", type=int, default=8)
    r.add_argument("--pause-seconds", type=int, default=360)
    r.add_argument("--model", default=MODEL)
    r.add_argument("--max-turns", type=int, default=MAX_TURNS)
    r.add_argument("--wall-seconds", type=int, default=WALL_SECONDS)
    r.add_argument("--prior-run-usd", type=float, default=6.0)
    r.add_argument("--stop-usd", type=float, default=ledger_mod.STOP_USD)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--sessions", action="store_true", help="Amendment 2: chained sessions (sessions.json)")
    r.add_argument("--session", default=None, help="dry/pilot: run this session id instead of the pilot session")
    r.add_argument("--wait-images", action="store_true", help="full sessions plan: start each session when its image is built")
    r.add_argument("--session-budget-usd", type=float, default=SESSION_CAP_USD, help="USD cap per chained session")
    r.add_argument("--conditions", default=None, help="subset of ABC (default: pilot/dry AB, full ABC)")
    r.add_argument("--c-pause-seconds", type=int, default=3660, help="condition C pause (Amendment 1: 3660)")
    r.add_argument("--image", default=None, help="dry/authcheck only: run this agent image instead of the task's")
    r.add_argument("--run-budget-usd", type=float, default=8.0, help="per-run --max-budget-usd safety cap")
    r.add_argument("--cache-1h", action="store_true",
                   help="set ENABLE_PROMPT_CACHING_1H=1 (one-hour writes; pauses must then be 3660 s)")
    r.add_argument("--runs-subdir", default=None, help="override the runs directory name (e.g. a second dry run)")
    r.add_argument("--amend3", choices=("S", "K"), default=None,
                   help="Amendment 3 with --sessions: S = injected invalidators (templates S1, S2); K = condition-B "
                        "sessions one at a time (forces concurrency 1)")
    r.add_argument("--run-gap-seconds", type=int, default=420, help="Amendment 3 K: idle seconds between sessions")
    sp.add_parser("ledger")
    args = ap.parse_args()
    if args.cmd == "run" and args.concurrency > 8:
        raise SystemExit("BLOCKED: concurrency is capped at 8 on the shared host")
    if args.cmd == "run" and args.amend3:
        if not args.sessions:
            raise SystemExit("BLOCKED: --amend3 needs --sessions")
        if args.amend3 == "K":
            args.concurrency = 1
    return {"build": cmd_build, "infra": cmd_infra, "run": cmd_run, "ledger": cmd_ledger}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
