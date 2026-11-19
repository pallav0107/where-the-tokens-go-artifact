"""Compare a re-computed evidence envelope with the published one.

Usage: python cmp_evidence.py PUBLISHED.json RECOMPUTED.json
Exact comparison of `aggregates` and `figures` (canonical JSON, sorted keys); `provenance`
is reported but not compared (it carries generation time, source hashes and descriptions).
Exit 0 only when both sections are identical.
"""
import json
import sys


def canon(o):
    return json.dumps(o, sort_keys=True, ensure_ascii=False)


def diff_paths(a, b, path="", out=None, limit=20):
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(path + "/" + str(k) + " (missing on one side)")
            else:
                diff_paths(a[k], b[k], path + "/" + str(k), out, limit)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            diff_paths(x, y, path + "[" + str(i) + "]", out, limit)
    elif canon(a) != canon(b):
        out.append("{}: {!r} != {!r}".format(path, a, b))
    return out


def main():
    pub = json.load(open(sys.argv[1], encoding="utf-8"))
    new = json.load(open(sys.argv[2], encoding="utf-8"))
    ok = True
    for sec in ("aggregates", "figures"):
        same = canon(pub.get(sec)) == canon(new.get(sec))
        n = len(pub.get(sec) or {})
        print("{}: {} ({} top-level entries)".format(sec, "IDENTICAL" if same else "DIFFERENT", n))
        if not same:
            ok = False
            for d in diff_paths(pub.get(sec), new.get(sec)):
                print("   ", d)
    ps = {s["id"]: s.get("sha256") for s in pub["provenance"].get("sources", [])}
    ns = {s["id"]: s.get("sha256") for s in new["provenance"].get("sources", [])}
    for k in sorted(set(ps) | set(ns)):
        print("source {}: published {} recomputed {} {}".format(
            k, (ps.get(k) or "-")[:12], (ns.get(k) or "-")[:12],
            "same" if ps.get(k) == ns.get(k) else "differs"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
