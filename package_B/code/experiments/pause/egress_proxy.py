#!/usr/bin/env python3
"""Allowlist HTTPS CONNECT proxy: the agent containers' only way out (PREREG_pause.md, Setup).

Agent containers sit on a docker --internal network with no route out; this proxy sits on that
network and on the default bridge, and tunnels CONNECT host:443 only for hosts in ALLOW (default
api.anthropic.com). Everything else gets 403 and a log line, so a blocked call is visible. It never
reads the tunnelled bytes (TLS end to end). Log: one JSON line per CONNECT on stdout.

    python3 egress_proxy.py --port 3128 --allow api.anthropic.com
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json

DEFAULT_ALLOW = ("api.anthropic.com",)


def allowed(target: str, allow: tuple[str, ...]) -> bool:
    """True for 'host:443' with host exactly in the allowlist (no suffix or port games)."""
    host, sep, port = target.rpartition(":")
    return bool(sep) and port == "443" and host.lower() in {a.lower() for a in allow}


def log(**kw) -> None:
    kw["ts"] = dt.datetime.now(dt.timezone.utc).isoformat()
    print(json.dumps(kw, sort_keys=True), flush=True)


async def pipe(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
    try:
        while data := await r.read(65536):
            w.write(data)
            await w.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        try:
            w.close()
        except Exception:  # noqa: BLE001
            pass


async def handle(cr: asyncio.StreamReader, cw: asyncio.StreamWriter, allow: tuple[str, ...]) -> None:
    peer = cw.get_extra_info("peername")
    try:
        head = await asyncio.wait_for(cr.readuntil(b"\r\n\r\n"), 30)
    except (asyncio.TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
        cw.close()
        return
    parts = head.split(b"\r\n", 1)[0].decode("latin-1").split()
    method, target = (parts + ["", ""])[:2]
    if method != "CONNECT" or not allowed(target, allow):
        log(event="deny", method=method, target=target, client=str(peer))
        cw.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
        await cw.drain()
        cw.close()
        return
    host, _, port = target.rpartition(":")
    try:
        ur, uw = await asyncio.wait_for(asyncio.open_connection(host, int(port)), 30)
    except (OSError, asyncio.TimeoutError) as e:
        log(event="upstream_error", target=target, error=type(e).__name__)
        cw.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
        await cw.drain()
        cw.close()
        return
    log(event="allow", target=target, client=str(peer))
    cw.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
    await cw.drain()
    await asyncio.gather(pipe(cr, uw), pipe(ur, cw))


async def serve(port: int, allow: tuple[str, ...]) -> None:
    server = await asyncio.start_server(lambda r, w: handle(r, w, allow), "0.0.0.0", port)
    log(event="listening", port=port, allow=list(allow))
    async with server:
        await server.serve_forever()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=3128)
    ap.add_argument("--allow", nargs="+", default=list(DEFAULT_ALLOW))
    args = ap.parse_args()
    asyncio.run(serve(args.port, tuple(args.allow)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
