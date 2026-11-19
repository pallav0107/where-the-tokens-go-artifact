#!/usr/bin/env python3
"""Stub of the Anthropic Messages API for dry runs: never contacts the real API, costs nothing.

The runner's --stub mode points the containers' ANTHROPIC_BASE_URL at this server. After each user
prompt it answers the first STUB_STEPS main-loop requests (those offering a Bash tool) with one Bash
tool call each, then ends the turn (so a continued conversation gets STUB_STEPS steps per task); other requests get a short text. It emulates prefix caching so the transcripts carry
realistic usage fields: a request reads the cached prefix of its session if the previous request
was within STUB_TTL_SECONDS, otherwise re-writes it; write tokens go to the TTL tier the request's
cache_control asked for. It logs, per request, the model, the anthropic-beta header, whether an
API key was sent (never the key) and the cache_control TTLs requested, which shows which cache
lifetime the pinned client asks for before any paid call.

    STUB_STEPS=30 STUB_TTL_SECONDS=20 python3 stub_api.py --port 8080 --log stub.jsonl
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE: dict[str, tuple[int, float]] = {}
LOCK = threading.Lock()
LOG_PATH = "stub.jsonl"


def cache_ttls(body: dict) -> collections.Counter:
    """Count cache_control markers in system, tools and messages by requested ttl ('5m' default)."""
    c: collections.Counter = collections.Counter()

    def walk(x):
        if isinstance(x, dict):
            cc = x.get("cache_control")
            if isinstance(cc, dict):
                c[cc.get("ttl", "5m")] += 1
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk([body.get("system"), body.get("tools"), body.get("messages")])
    return c


def usage_for(body: dict, now: float, ttl_s: float) -> dict:
    """Emulated usage: read the session's cached prefix if within ttl_s, else re-write it."""
    tokens = max(10, len(json.dumps(body, sort_keys=True)) // 4)
    ttls = cache_ttls(body)
    msgs = body.get("messages") or [{}]
    key = hashlib.sha256((str(body.get("model")) + json.dumps(msgs[0], sort_keys=True)).encode()).hexdigest()
    fresh = 3
    if not ttls:
        return {"input_tokens": tokens, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
                "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
                "output_tokens": 1}
    with LOCK:
        cached, last = STATE.get(key, (0, 0.0))
        read = min(cached, tokens - fresh) if now - last <= ttl_s else 0
        write = max(0, tokens - fresh - read)
        STATE[key] = (read + write, now)
    one_h = "1h" in ttls
    return {"input_tokens": fresh, "cache_creation_input_tokens": write, "cache_read_input_tokens": read,
            "cache_creation": {"ephemeral_5m_input_tokens": 0 if one_h else write,
                               "ephemeral_1h_input_tokens": write if one_h else 0},
            "output_tokens": 1}


def reply_blocks(body: dict, steps: int) -> tuple[list[dict], str]:
    tools = {t.get("name") for t in body.get("tools") or [] if isinstance(t, dict)}
    msgs = [m for m in body.get("messages") or [] if isinstance(m, dict)]

    def is_prompt(m: dict) -> bool:     # a user turn carrying text, not only tool results
        c = m.get("content")
        return m.get("role") == "user" and (isinstance(c, str) or any(
            isinstance(b, dict) and b.get("type") == "text" for b in c or []) and not any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in c or []))
    last = max([i for i, m in enumerate(msgs) if is_prompt(m)] or [0])
    n_assistant = sum(m.get("role") == "assistant" for m in msgs[last:])   # steps since the latest prompt
    if "Bash" in tools and n_assistant < steps:
        return [{"type": "tool_use", "id": "toolu_" + uuid.uuid4().hex[:24], "name": "Bash",
                 "input": {"command": f"echo stub step {n_assistant + 1}", "description": "stub step"}}], "tool_use"
    return [{"type": "text", "text": "Done (stub)."}], "end_turn"


def sse(events: list[tuple[str, dict]]) -> bytes:
    return "".join("event: %s\ndata: %s\n\n" % (e, json.dumps(d)) for e, d in events).encode()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _record(self, row: dict) -> None:
        row["ts"] = dt.datetime.now(dt.timezone.utc).isoformat()
        row["api_key_sent"] = bool(self.headers.get("x-api-key") or self.headers.get("authorization"))
        row["beta"] = self.headers.get("anthropic-beta")
        with LOCK, open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    def do_GET(self):
        self._record({"path": self.path, "method": "GET"})
        self._send(404, {"type": "error", "error": {"type": "not_found_error", "message": "stub"}})

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("content-length", "0")
        self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("content-length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            body = {}
        path = self.path.split("?")[0]
        if path.endswith("/count_tokens"):
            self._record({"path": path, "model": body.get("model")})
            return self._send(200, {"input_tokens": max(10, len(json.dumps(body)) // 4)})
        if not path.endswith("/v1/messages"):
            self._record({"path": path, "method": "POST"})
            return self._send(404, {"type": "error", "error": {"type": "not_found_error", "message": "stub"}})
        usage = usage_for(body, time.time(), float(os.environ.get("STUB_TTL_SECONDS", "300")))
        blocks, stop = reply_blocks(body, int(os.environ.get("STUB_STEPS", "30")))
        mid = "msg_stub_" + uuid.uuid4().hex[:20]
        model = body.get("model") or "stub"
        self._record({"path": path, "model": model, "stream": bool(body.get("stream")),
                      "n_tools": len(body.get("tools") or []), "n_messages": len(body.get("messages") or []), "cache_ttls": dict(cache_ttls(body)),
                      "usage": usage, "stop": stop})
        out_tokens = 20
        if not body.get("stream"):
            return self._send(200, {"id": mid, "type": "message", "role": "assistant", "model": model,
                                    "content": blocks, "stop_reason": stop, "stop_sequence": None,
                                    "usage": {**usage, "output_tokens": out_tokens}})
        events = [("message_start", {"type": "message_start", "message": {
            "id": mid, "type": "message", "role": "assistant", "model": model, "content": [],
            "stop_reason": None, "stop_sequence": None, "usage": usage}})]
        for i, b in enumerate(blocks):
            if b["type"] == "tool_use":
                events.append(("content_block_start", {"type": "content_block_start", "index": i, "content_block": {
                    "type": "tool_use", "id": b["id"], "name": b["name"], "input": {}}}))
                events.append(("content_block_delta", {"type": "content_block_delta", "index": i, "delta": {
                    "type": "input_json_delta", "partial_json": json.dumps(b["input"])}}))
            else:
                events.append(("content_block_start", {"type": "content_block_start", "index": i,
                                                       "content_block": {"type": "text", "text": ""}}))
                events.append(("content_block_delta", {"type": "content_block_delta", "index": i,
                                                       "delta": {"type": "text_delta", "text": b["text"]}}))
            events.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
        events.append(("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                                         "usage": {"output_tokens": out_tokens}}))
        events.append(("message_stop", {"type": "message_stop"}))
        data = sse(events)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> int:
    global LOG_PATH
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--log", default="stub.jsonl")
    args = ap.parse_args()
    LOG_PATH = args.log
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
