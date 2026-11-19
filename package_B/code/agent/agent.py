"""The ReAct agent loop (PREREG section 2). One LLM request per step."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

from agent.policies import build_messages
from agent.prompts import SYSTEM_PROMPT

SCHEMA = "agent-request/v1"
KNOWN_ACTIONS = ("search", "read", "finish")
ERROR_OBSERVATION = (
    'Error: the reply must be exactly one JSON object with keys "thought", "notes", '
    '"action", "args" and "action" one of "search", "read", "finish". '
    "This step counts; recover with a valid action."
)


@dataclass
class AgentResult:
    answer: str = ""
    finished: bool = False
    turns: int = 0
    total_input_tokens: int = 0
    total_cached_tokens: int | None = None  # None if any request has None
    total_output_tokens: int = 0
    total_latency_s: float = 0.0
    parse_errors: int = 0
    steps: list = field(default_factory=list)  # one record per request
    grounded: bool | None = None  # v2 arms with a cites field only


def _extract_json(text: str) -> dict | None:
    """Parse the first JSON object in the reply."""
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def _request_usd(provider, usage) -> float | None:
    from agent.providers import CLOUD_PROVIDERS
    if provider.name not in CLOUD_PROVIDERS:
        return None  # scripted/local providers: no USD
    from agent.budget import request_usd
    try:
        return request_usd(provider.name, provider.model, usage)
    except Exception:  # noqa: BLE001 - cost logging must never break the run
        return None


def make_record(provider, resp, messages, step, policy, k, run_meta, action,
                parse_error, obs_chars) -> dict:
    """The per-request log record (shared by every arm, so the schema is identical)."""
    usage = resp.usage
    raw = usage.raw or {}
    run_meta = run_meta or {}
    return {
        "schema": SCHEMA,
        "run_id": run_meta.get("run_id"),
        "question_id": run_meta.get("question_id"),
        "turn": run_meta.get("turn"),
        "step": step,
        "provider": provider.name,
        "model": provider.model,
        "policy": policy,
        "k": k,
        "control": run_meta.get("control"),
        "input_tokens": usage.input_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "output_tokens": usage.output_tokens,
        "latency_s": round(usage.latency_s, 6),
        "provider_timing": (usage.raw or {}).get("provider_timing"),
        "usage_metadata": (usage.raw or {}).get("usage_metadata"),
        "thinking_config": (usage.raw or {}).get("thinking_config"),
        "response_model": (usage.raw or {}).get("response_model"),
        # OpenAI (PREREG_v2 Amendment 2, B5/B7); None for the other providers
        "cache_write_tokens": getattr(usage, "cache_write_tokens", None),
        "reasoning_tokens": getattr(usage, "reasoning_tokens", None),
        "service_tier": raw.get("service_tier"),
        "system_fingerprint": raw.get("system_fingerprint"),
        "finish_reason": raw.get("finish_reason"),
        "input_upper_bound": raw.get("input_upper_bound"),
        "usd_worst": raw.get("usd_worst"),
        "unconfirmed_attempts": raw.get("unconfirmed_attempts"),
        "cum_usd": raw.get("cum_usd"),
        "action": action,
        "parse_error": parse_error,
        "obs_chars": obs_chars,
        "reply": resp.text,  # raw visible reply (what is carried); thinking is not
        "thinking_chars": len(resp.thinking or ""),
        "n_messages": len(messages),
        "usd": _request_usd(provider, usage),
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _missing_tools():
    def _raiser(*args, **kwargs):
        raise RuntimeError("run_agent was called without tools")
    return {"search": _raiser, "read": _raiser}


def run_agent(question, provider, policy, k, tools=None, system_prompt=None,
              max_steps=10, logger=None, run_meta=None,
              error_observation=None) -> AgentResult:
    """Run the ReAct loop. tools maps "search"/"read" to callables; if omitted,
    a placeholder is used that raises (the grid runner always passes tools).
    error_observation replaces the v1 parse-error text (PREREG_v3 notes-free arms)."""
    tools = tools if tools is not None else _missing_tools()
    system_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    run_meta = run_meta or {}
    history: list[tuple[str, str]] = []  # (raw assistant reply, observation)
    res = AgentResult()
    cached_seen: list = []
    for step in range(max_steps):
        messages = build_messages(policy, system_prompt, question, history)
        resp = provider.chat(messages)
        usage = resp.usage
        res.turns += 1
        res.total_input_tokens += usage.input_tokens
        res.total_output_tokens += usage.output_tokens
        res.total_latency_s += usage.latency_s
        cached_seen.append(usage.cached_input_tokens)

        obj = _extract_json(resp.text)
        action, parse_error = None, True
        observation = error_observation or ERROR_OBSERVATION
        args = obj.get("args") if isinstance(obj, dict) else None
        if (obj is not None and obj.get("action") in KNOWN_ACTIONS
                and (args is None or isinstance(args, dict))):
            action = obj["action"]
            args = args or {}
            parse_error = False
            if action == "search":
                observation = tools["search"](str(args.get("query", "")), k)
            elif action == "read":
                observation = tools["read"](str(args.get("doc_id", "")))
            else:  # finish
                res.finished = True
                res.answer = str(args.get("answer", ""))
                observation = ""
        else:
            res.parse_errors += 1
        # The carried assistant message is the model's raw reply text; thinking is never carried.
        history.append((resp.text, observation))

        record = make_record(provider, resp, messages, step, policy, k, run_meta,
                             action, parse_error, len(observation))
        res.steps.append(record)
        if logger is not None:
            logger.log(record)
        if res.finished:
            break
    res.total_cached_tokens = (
        sum(cached_seen) if all(c is not None for c in cached_seen) else None
    )
    return res
