"""LLM providers for the experiment grid.

Only the Python stdlib is used for HTTP (urllib). API keys and request headers
are never printed or logged.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

_BACKOFF_S = (5, 15, 45)  # PREREG: 3 retries with 5/15/45 s backoff
_TIMEOUT_S = 600
_QUOTA_BACKOFF_S = (10, 30, 90, 270)  # HTTP 429 without a RetryInfo delay
_MAX_RETRY_DELAY_S = 120.0  # cap on a server-supplied retryDelay
_DASHSCOPE_DEFAULT_BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
_GEMINI_DEFAULT_BASE = "https://generativelanguage.googleapis.com"
_OPENAI_DEFAULT_BASE = "https://api.openai.com/v1"
# PREREG_v2 Amendment 2, B7 ([REDACTED: unrelated]):
_OPENAI_RATE_BACKOFF_S = (2, 4, 8, 16, 32, 64)  # HTTP 429 rate limit, then the block stops
_OPENAI_MAX_CONCURRENCY = 2
_OPENAI_SLOTS = threading.BoundedSemaphore(_OPENAI_MAX_CONCURRENCY)
_OPENAI_EFFORT_LADDER = ("none", "minimal", "low")  # B3: lowest first
_OPENAI_FALLBACK_PARAMS = ("temperature", "seed", "reasoning_effort")
_SECRET_RE = re.compile(r"sk-[A-Za-z0-9_*.\-]{3,}")

CLOUD_PROVIDERS = ("gemini", "dashscope", "openai")  # billed providers (USD per request)


class ProviderUnavailable(Exception):
    """The provider cannot serve requests (missing key, network failure, HTTP 5xx...)."""


class ProviderQuotaExhausted(Exception):
    """Rate limit persisted after backoff, or a per-day quota is used up. Deliberately NOT a
    ProviderUnavailable: the grid stops the model's block instead of recording run errors."""


class ProviderInsufficientQuota(Exception):
    """OpenAI `insufficient_quota`: the account has no credit left. Never retried; the grid
    stops the WHOLE run at once (PREREG_v2 Amendment 2, B7: [REDACTED: unrelated]
    [REDACTED: unrelated])."""


class ProviderAuthInvalid(ProviderInsufficientQuota, ProviderUnavailable):
    """HTTP 401: the key is missing, invalid or revoked. Never retried; halts the whole run
    like a billing stop (PREREG_v3 Amendment 2), since every further request would fail too."""


class ProviderBadRequest(ProviderUnavailable):
    """HTTP 400 from OpenAI; `error` is the parsed (secret-scrubbed) error object."""

    def __init__(self, message, error=None):
        super().__init__(message)
        self.error = error or {}


def _sleep(seconds):
    """Backoff sleep; tests patch agent.providers._sleep (or time.sleep)."""
    time.sleep(seconds)


def _retry_delay_s(detail):
    """google.rpc.RetryInfo retryDelay (e.g. "17s" / "1.5s") from an error body, or None."""
    try:
        err = json.loads(detail).get("error") or {}
    except (ValueError, AttributeError):
        return None
    for d in err.get("details") or []:
        if isinstance(d, dict) and str(d.get("@type", "")).endswith("RetryInfo"):
            m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)s\s*", str(d.get("retryDelay") or ""))
            if m:
                return min(float(m.group(1)), _MAX_RETRY_DELAY_S)
    return None


def _is_daily_quota(detail):
    low = detail.lower()
    if "perday" in low or "per day" in low:
        return True
    return "resource_exhausted" in low and "daily" in low


@dataclass
class Usage:
    input_tokens: int
    cached_input_tokens: int | None  # null when the provider does not report it (never 0)
    output_tokens: int
    latency_s: float
    raw: dict = field(default_factory=dict)
    cache_write_tokens: int | None = None  # OpenAI only: prompt tokens billed as cache writes
    reasoning_tokens: int | None = None  # OpenAI only: part of output_tokens


@dataclass
class LLMResponse:
    text: str
    usage: Usage
    thinking: str | None = None


class Provider(ABC):
    name: str = ""
    model: str = ""

    @abstractmethod
    def chat(self, messages, **opts) -> LLMResponse:
        ...


def _http_json(method, url, headers=None, payload=None, timeout=_TIMEOUT_S, retries=3,
               quota_aware=False):
    """JSON HTTP call with retries on network errors and 5xx (backoff 5/15/45 s).

    quota_aware: HTTP 429 honours RetryInfo retryDelay (else 10/30/90/270 s); a daily quota
    or a 429 persisting after 4 retries raises ProviderQuotaExhausted."""
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    hdrs = {"Content-Type": "application/json"}
    hdrs.update(headers or {})
    last_exc = None
    quota_tries = 0
    attempt = 0
    while attempt <= retries:
        try:
            req = urllib.request.Request(url, data=body, headers=hdrs, method=method)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_exc = exc
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:2000]
            except Exception:  # noqa: BLE001 - best effort; body is not always readable
                pass
            if quota_aware and exc.code == 429:
                if _is_daily_quota(detail):
                    raise ProviderQuotaExhausted(
                        f"per-day quota exhausted (HTTP 429): {detail[:300]}") from exc
                if quota_tries >= len(_QUOTA_BACKOFF_S):
                    raise ProviderQuotaExhausted(
                        f"HTTP 429 persisted after {quota_tries} retries: {detail[:300]}") from exc
                delay = _retry_delay_s(detail)
                _sleep(delay if delay is not None else _QUOTA_BACKOFF_S[quota_tries])
                quota_tries += 1
                continue
            if exc.code >= 500 and attempt < retries:
                _sleep(_BACKOFF_S[attempt])
                attempt += 1
                continue
            detail = detail[:500]
            suffix = f": {detail}" if detail else ""
            raise ProviderUnavailable(f"HTTP {exc.code} from {url}{suffix}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt < retries:
                _sleep(_BACKOFF_S[attempt])
                attempt += 1
                continue
            break
    raise ProviderUnavailable(f"network error calling {url}: {last_exc}")


def _scrub(text, secret=None):
    """Remove the API key (and anything shaped like an OpenAI key) from text that may be
    logged or raised: error bodies can echo a partially masked key."""
    text = str(text)
    if secret:
        text = text.replace(secret, "[redacted]")
    return _SECRET_RE.sub("sk-[redacted]", text)


def _openai_error(detail):
    try:
        err = json.loads(detail).get("error")
    except (ValueError, AttributeError):
        return {}
    return err if isinstance(err, dict) else {}


# Account-level billing stops: never retried; the whole run halts (Amendment 2, B7).
_OPENAI_BILLING_STOPS = ("insufficient_quota", "billing_hard_limit_reached",
                         "billing_not_active", "access_terminated")


def _is_insufficient_quota(detail):
    err = _openai_error(detail)
    codes = (str(err.get("code")), str(err.get("type")))
    return any(c in codes or c in detail for c in _OPENAI_BILLING_STOPS)


def _retry_after_s(exc):
    """Retry-After / retry-after-ms of an HTTP error, in seconds (capped), or None."""
    hdrs = getattr(exc, "headers", None) or {}
    try:
        ms = hdrs.get("retry-after-ms")
        if ms is not None:
            return min(float(ms) / 1000.0, _MAX_RETRY_DELAY_S)
        sec = hdrs.get("retry-after")
        if sec is not None:
            return min(float(sec), _MAX_RETRY_DELAY_S)
    except (TypeError, ValueError):
        return None
    return None


def _openai_post(url, api_key, payload, stats, timeout=_TIMEOUT_S):
    """One Chat Completions call under Amendment 2 B7.

    429 insufficient_quota -> ProviderInsufficientQuota at once (never retried); other 429s
    -> backoff 2..64 s (or Retry-After, capped) then ProviderQuotaExhausted; 5xx and network
    errors -> 5/15/45 s then ProviderUnavailable; 400 -> ProviderBadRequest; other 4xx ->
    ProviderUnavailable. stats["unconfirmed"] counts timeouts after the request was sent
    (possibly billed without a response). Messages are scrubbed of the key."""
    body = json.dumps(payload).encode("utf-8")
    hdrs = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    rate_tries = err_tries = 0
    while True:
        try:
            req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
            with _OPENAI_SLOTS:  # at most 2 requests in flight per process
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:2000]
            except Exception:  # noqa: BLE001 - best effort; body is not always readable
                pass
            detail = _scrub(detail, api_key)
            err = _openai_error(detail)
            msg = _scrub(err.get("message") or detail[:300], api_key)
            if 400 <= exc.code < 500 and _is_insufficient_quota(detail):
                raise ProviderInsufficientQuota(
                    f"{err.get('code') or err.get('type') or 'insufficient_quota'} "
                    f"(HTTP {exc.code}): {msg}") from None
            if exc.code == 401:
                raise ProviderAuthInvalid(
                    f"{err.get('code') or err.get('type') or 'unauthorized'} (HTTP 401): "
                    f"{msg}") from None
            if exc.code == 429:
                if rate_tries >= len(_OPENAI_RATE_BACKOFF_S):
                    raise ProviderQuotaExhausted(
                        f"HTTP 429 persisted after {rate_tries} retries: {msg}") from None
                delay = max(_OPENAI_RATE_BACKOFF_S[rate_tries], _retry_after_s(exc) or 0.0)
                _sleep(min(delay, _MAX_RETRY_DELAY_S))
                rate_tries += 1
                continue
            if exc.code >= 500 and err_tries < len(_BACKOFF_S):
                _sleep(_BACKOFF_S[err_tries])
                err_tries += 1
                continue
            text = (f"HTTP {exc.code} from {url}: {err.get('type') or ''} "
                    f"{err.get('code') or ''} param={err.get('param')!r}: {msg}")
            if exc.code == 400:
                raise ProviderBadRequest(text, error={k: _scrub(v, api_key) if isinstance(v, str)
                                                      else v for k, v in err.items()}) from None
            raise ProviderUnavailable(text) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if isinstance(exc, TimeoutError) or "timed out" in str(exc).lower():
                stats["unconfirmed"] = stats.get("unconfirmed", 0) + 1
            if err_tries < len(_BACKOFF_S):
                _sleep(_BACKOFF_S[err_tries])
                err_tries += 1
                continue
            raise ProviderUnavailable(
                f"network error calling {url}: {type(exc).__name__}") from None


def _rejected_param(error):
    """Which B3 parameter an HTTP 400 rejects (temperature, seed, reasoning_effort) or None."""
    param = str(error.get("param") or "")
    msg = str(error.get("message") or "").lower()
    for name in _OPENAI_FALLBACK_PARAMS:
        if param == name or param.endswith("." + name):
            return name
    if any(w in msg for w in ("unsupported", "not supported", "does not support",
                              "is not allowed", "unrecognized")):
        for name in _OPENAI_FALLBACK_PARAMS:
            if name in msg:
                return name
    return None


def _roles(messages):
    return [{"role": m["role"], "content": m["content"]} for m in messages]


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, spec):
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ProviderUnavailable("GEMINI_API_KEY not set in job environment")
        self.api_key = key
        self.base = os.environ.get("GEMINI_BASE_URL", _GEMINI_DEFAULT_BASE).rstrip("/")
        self.thinking_budget = spec.get("thinking_budget")
        self.max_output_tokens = spec.get("max_output_tokens")
        self.model = spec["model"]
        self.thinking_level = "minimal"  # downgraded to "low" if the API rejects "minimal"
        if self.model == "auto-flash":
            self.model = self._resolve_tier("flash")
        elif self.model == "auto-flash-lite":
            self.model = self._resolve_tier("flash-lite")

    def _thinking_config(self):
        """Thinking setting actually sent; gemini-3 ids only accept thinkingLevel."""
        if self.model.startswith("gemini-3"):
            return {"thinkingLevel": self.thinking_level}
        if self.thinking_budget is not None:
            return {"thinkingBudget": self.thinking_budget}
        return None

    def _resolve_tier(self, suffix):
        """PREREG A1: highest-version exact gemini-<version>-<suffix> with generateContent.

        Preview/exp/image/tts/live/customtools ids and any extra suffix (e.g. -001) never
        match the exact form, so they are excluded."""
        pat = re.compile(r"gemini-(\d+(?:\.\d+)*)-" + re.escape(suffix))
        best = None  # (version_tuple, id)
        token = None
        while True:
            url = f"{self.base}/v1beta/models"
            if token:
                url += "?pageToken=" + urllib.parse.quote(token)
            data = _http_json("GET", url, headers={"x-goog-api-key": self.api_key},
                              quota_aware=True)
            for m in data.get("models") or []:
                mid = str(m.get("name") or "").removeprefix("models/")
                match = pat.fullmatch(mid)
                if not match:
                    continue
                if "generateContent" not in (m.get("supportedGenerationMethods") or []):
                    continue
                ver = tuple(int(x) for x in match.group(1).split("."))
                if best is None or ver > best[0]:
                    best = (ver, mid)
            token = data.get("nextPageToken")
            if not token:
                break
        if best is None:
            raise ProviderUnavailable(
                f"no gemini-<version>-{suffix} model supporting generateContent")
        return best[1]

    def chat(self, messages, **opts):
        system_parts, contents = [], []
        for m in messages:
            if m["role"] == "system":
                system_parts.append({"text": m["content"]})
                continue
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        gen_cfg = {"temperature": 0, "seed": 42, "responseMimeType": "application/json"}
        if self.max_output_tokens is not None:
            gen_cfg["maxOutputTokens"] = self.max_output_tokens
        thinking_cfg = self._thinking_config()
        if thinking_cfg is not None:
            gen_cfg["thinkingConfig"] = dict(thinking_cfg)
        payload = {"contents": contents, "generationConfig": gen_cfg}
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}
        url = f"{self.base}/v1beta/models/{self.model}:generateContent"
        t0 = time.monotonic()
        try:
            data = _http_json("POST", url, headers={"x-goog-api-key": self.api_key},
                              payload=payload, quota_aware=True)
        except ProviderUnavailable as exc:
            # gemini-3 rejects some thinkingLevel values with HTTP 400: retry once with "low".
            if (thinking_cfg is None or "thinkingLevel" not in thinking_cfg
                    or "400" not in str(exc) or "thinking" not in str(exc).lower()):
                raise
            thinking_cfg = {"thinkingLevel": "low"}
            payload["generationConfig"]["thinkingConfig"] = dict(thinking_cfg)
            data = _http_json("POST", url, headers={"x-goog-api-key": self.api_key},
                              payload=payload, quota_aware=True)
            self.thinking_level = "low"  # the retry worked: remember it for later requests
        latency = time.monotonic() - t0
        parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text") or "" for p in parts if not p.get("thought"))
        thinking = "".join(p.get("text") or "" for p in parts if p.get("thought")) or None
        um = data.get("usageMetadata") or {}
        usage = Usage(
            input_tokens=int(um.get("promptTokenCount") or 0),
            cached_input_tokens=int(um.get("cachedContentTokenCount") or 0),  # absent -> 0
            output_tokens=int(um.get("candidatesTokenCount") or 0) + int(um.get("thoughtsTokenCount") or 0),
            latency_s=latency,
            raw={"thinking_config": thinking_cfg, "usage_metadata": dict(um)},
        )
        return LLMResponse(text=text, usage=usage, thinking=thinking)


class DashScopeProvider(Provider):
    name = "dashscope"

    def __init__(self, spec):
        key = os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            raise ProviderUnavailable("DASHSCOPE_API_KEY not set in job environment")
        self.api_key = key
        self.base = os.environ.get("DASHSCOPE_BASE_URL", _DASHSCOPE_DEFAULT_BASE).rstrip("/")
        # Pay-as-you-go only: the Alibaba Coding Plan (sk-sp- keys, coding-* endpoints) forbids
        # automated scripts and batch calls, so it is refused outright (PREREG section 3).
        if key.startswith("sk-sp-") or "coding" in self.base.lower():
            raise ProviderUnavailable("Coding Plan key/endpoint refused: pay-as-you-go DashScope only")
        self.model = spec["model"]
        self.max_tokens = spec.get("max_tokens", spec.get("max_output_tokens"))
        # Only sent when the spec carries the key (qwen-max has no thinking mode).
        self.enable_thinking = spec.get("enable_thinking") if "enable_thinking" in spec else None

    def chat(self, messages, **opts):
        payload = {
            "model": self.model,
            "messages": _roles(messages),
            "temperature": 0,
            "seed": 42,
            "response_format": {"type": "json_object"},
        }
        if self.enable_thinking is not None:
            payload["enable_thinking"] = self.enable_thinking
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        t0 = time.monotonic()
        data = _http_json("POST", f"{self.base}/chat/completions",
                          headers={"Authorization": f"Bearer {self.api_key}"},
                          payload=payload)
        latency = time.monotonic() - t0
        text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        u = data.get("usage") or {}
        details = u.get("prompt_tokens_details") or {}
        usage = Usage(
            input_tokens=int(u.get("prompt_tokens") or 0),
            cached_input_tokens=int(details.get("cached_tokens") or 0),  # absent -> 0
            output_tokens=int(u.get("completion_tokens") or 0),
            latency_s=latency,
            raw={"response_model": data.get("model")},
        )
        return LLMResponse(text=text, usage=usage, thinking=None)


class OpenAIProvider(Provider):
    """OpenAI Chat Completions (PREREG_v2 Amendment 2). Stdlib HTTPS only.

    Refuses to send anything until a spend guard is attached (attach_budget), so every
    request is admitted against the project cap first. The key is held privately and never
    logged; error text is scrubbed of it."""

    name = "openai"

    def __init__(self, spec):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ProviderUnavailable("OPENAI_API_KEY not set in job environment")
        self.__key = key
        self.base = os.environ.get("OPENAI_BASE_URL", _OPENAI_DEFAULT_BASE).rstrip("/")
        self.model = spec["model"]
        self.max_completion_tokens = int(spec.get("max_output_tokens")
                                         or spec.get("max_completion_tokens") or 4096)
        effort = spec.get("reasoning_effort", _OPENAI_EFFORT_LADDER[0])
        self._efforts = (list(_OPENAI_EFFORT_LADDER[_OPENAI_EFFORT_LADDER.index(effort):])
                         if effort in _OPENAI_EFFORT_LADDER else [effort])
        self.reasoning_effort = self._efforts[0]
        self.send_temperature = spec.get("temperature", 0) is not None
        self.send_seed = spec.get("seed", 42) is not None
        self.rejected: list = []  # B3: parameters the API rejected, in order
        self.guard = None

    def __repr__(self):
        return f"OpenAIProvider(model={self.model!r})"

    def attach_budget(self, max_usd, spent):
        """Attach the per-request spend guard (agent.budget.SpendGuard) at the given prior
        spend; the price entry must exist and be verified."""
        from agent.budget import SpendGuard
        self.guard = SpendGuard(self.name, self.model, max_usd, spent)
        return self.guard

    def sent_params(self):
        return {"temperature": 0 if self.send_temperature else None,
                "seed": 42 if self.send_seed else None,
                "reasoning_effort": self.reasoning_effort,
                "max_completion_tokens": self.max_completion_tokens,
                "service_tier": "default", "store": False,
                "response_format": "json_object", "prompt_cache_key": None,
                "rejected": list(self.rejected)}

    def _payload(self, msgs):
        # B4: no prompt_cache_key / prompt_cache_options / retention, identically for all arms.
        payload = {"model": self.model, "messages": msgs,
                   "response_format": {"type": "json_object"},
                   "max_completion_tokens": self.max_completion_tokens,
                   "reasoning_effort": self.reasoning_effort,
                   "service_tier": "default", "store": False}
        if self.send_temperature:
            payload["temperature"] = 0
        if self.send_seed:
            payload["seed"] = 42
        return payload

    def _fall_back(self, name):
        """B3 rule: drop temperature/seed, or step reasoning effort down the ladder."""
        if name == "temperature" and self.send_temperature:
            self.send_temperature = False
        elif name == "seed" and self.send_seed:
            self.send_seed = False
        elif name == "reasoning_effort" and len(self._efforts) > 1:
            self._efforts.pop(0)
            self.reasoning_effort = self._efforts[0]
        else:
            return False
        self.rejected.append(name)
        return True

    def chat(self, messages, **opts):
        from agent.budget import BudgetExceeded, input_token_upper_bound, request_usd
        if self.guard is None:
            raise BudgetExceeded("no spend guard attached: OpenAI requests are refused "
                                 "(PREREG_v2 Amendment 2, B7)")
        msgs = _roles(messages)
        bound = input_token_upper_bound(msgs)
        worst = self.guard.admit(bound, self.max_completion_tokens)
        url = f"{self.base}/chat/completions"
        stats: dict = {}
        t0 = time.monotonic()
        try:
            while True:
                try:
                    data = _openai_post(url, self.__key, self._payload(msgs), stats)
                    break
                except ProviderBadRequest as exc:
                    if not self._fall_back(_rejected_param(exc.error)):
                        raise
        except BaseException:
            self.guard.release(worst)
            raise
        finally:
            if stats.get("unconfirmed"):
                self.guard.charge_unconfirmed(worst * stats["unconfirmed"])
        latency = time.monotonic() - t0
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        u = data.get("usage") or {}
        pdet = u.get("prompt_tokens_details") or {}
        cdet = u.get("completion_tokens_details") or {}
        usage = Usage(
            input_tokens=int(u.get("prompt_tokens") or 0),
            cached_input_tokens=int(pdet.get("cached_tokens") or 0),  # absent -> 0
            output_tokens=int(u.get("completion_tokens") or 0),  # includes reasoning tokens
            latency_s=latency,
            cache_write_tokens=int(pdet.get("cache_write_tokens") or 0),
            reasoning_tokens=int(cdet.get("reasoning_tokens") or 0),
            raw={"response_model": data.get("model"),
                 "usage_metadata": dict(u),
                 "thinking_config": self.sent_params(),
                 "service_tier": data.get("service_tier"),
                 "system_fingerprint": data.get("system_fingerprint"),
                 "finish_reason": choice.get("finish_reason"),
                 "refusal": message.get("refusal"),
                 "input_upper_bound": bound,
                 "usd_worst": round(worst, 8),
                 "unconfirmed_attempts": int(stats.get("unconfirmed") or 0)},
        )
        # no usage object = cost unknown: unpriced (fail closed), never 0
        usd = request_usd(self.name, self.model, usage) if u else None
        usage.raw["cum_usd"] = self.guard.charge(usd, worst, usage.input_tokens, bound)
        return LLMResponse(text=message.get("content") or "", usage=usage, thinking=None)


class MockProvider(Provider):
    """Deterministic scripted LLM for tests (model name: mock-scripted).

    step 0: search(question); step 1: read(first [doc_id] in the last observation);
    step 2: finish(first 2 words of that read observation after the title).
    input_tokens = number of whitespace words over all message contents.
    """

    name = "mock"

    def __init__(self, spec):
        self.model = spec.get("model", "mock-scripted")
        self._step = 0

    def chat(self, messages, **opts):
        n_in = sum(len(str(m.get("content") or "").split()) for m in messages)
        last = str(messages[-1].get("content") or "") if messages else ""
        step, self._step = self._step, self._step + 1
        if step == 0:
            question = next(
                (str(m["content"])[len("Question: "):]
                 for m in messages if str(m.get("content") or "").startswith("Question: ")),
                last,
            )
            action, args = "search", {"query": question}
        elif step == 1:
            match = re.search(r"\[([^\]]+)\]", last)
            action, args = "read", {"doc_id": match.group(1) if match else "unknown"}
        else:
            body = last.removeprefix("Observation: ")
            body = re.sub(r"^\[[^\]]*\]\s*", "", body)
            if ":" in body:
                body = body.split(":", 1)[1]
            words = body.split()
            action, args = "finish", {"answer": " ".join(words[:2]) if words else "unknown"}
        text = json.dumps(
            {"thought": "scripted", "notes": "scripted notes", "action": action, "args": args},
            separators=(",", ":"),
        )
        usage = Usage(input_tokens=n_in, cached_input_tokens=None,
                      output_tokens=len(text.split()), latency_s=0.0, raw={})
        return LLMResponse(text=text, usage=usage, thinking=None)


_PROVIDERS = {
    "gemini": GeminiProvider,
    "dashscope": DashScopeProvider,
    "openai": OpenAIProvider,
    "mock": MockProvider,
}


def make_provider(spec: dict) -> Provider:
    kind = spec.get("provider")
    cls = _PROVIDERS.get(kind)
    if cls is None:
        raise ValueError(f"unknown provider: {kind!r}")
    return cls(spec)
