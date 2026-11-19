#!/usr/bin/env python3
"""READ-stage gate (runs in CI): every evidence/*.json obeys the envelope contract.

Checks, per file:
  * envelope schema (and evidence/schema/<kind>.schema.json if present)
  * every figure cites a source id declared in provenance.sources
  * macro names are unique across ALL evidence files (gen_macros would collide)
  * no session-id-shaped strings and no absolute paths anywhere in the file
Exit 1 on any failure.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_lib import SCHEMA_DIR, evidence_files, load_json, validate  # noqa: E402

UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
# A drive letter must stand alone (not the tail of a URL scheme such as "https:"), and a
# forward slash after it must not be "//"; so "https://..." passes while C:\... (also as
# JSON's C:\\...), c:/... and file:///C:/... fail.
ABS_PATH = re.compile(r"(?:(?<![A-Za-z0-9+.\-])[A-Za-z]:(?:\\|/(?!/))[^\s\"]+"
                      r"|(?<![\w~.])/(?:home|Users|root|srv|mnt|tmp)/[^\s\"]*)")


def check_file(path: Path, seen: dict[str, str]) -> list[str]:
    errors: list[str] = []
    try:
        data = load_json(path)
    except json.JSONDecodeError as exc:
        return [f"{path.name}: invalid JSON: {exc}"]
    envelope = load_json(SCHEMA_DIR / "envelope.schema.json")
    errors += [f"{path.name}: {e}" for e in validate(data, envelope)]
    if errors:
        return errors
    kind_schema = SCHEMA_DIR / f"{data['kind']}.schema.json"
    if kind_schema.is_file():
        errors += [f"{path.name}: {e}" for e in validate(data, load_json(kind_schema))]
    ids = {s["id"] for s in data["provenance"]["sources"]}
    for name, fig in data["figures"].items():
        if fig["source"] not in ids:
            errors.append(f"{path.name}: figure {name} cites undeclared source {fig['source']!r}")
        if name in seen:
            errors.append(f"{path.name}: macro {name} already defined in {seen[name]}")
        seen[name] = path.name
    text = path.read_text(encoding="utf-8")
    for pat, what in ((UUID, "session-id-shaped string"), (ABS_PATH, "absolute path")):
        m = pat.search(text)
        if m:
            errors.append(f"{path.name}: {what} found: {m.group(0)[:40]!r}")
    return errors


def main() -> int:
    files = evidence_files()
    if not files:
        print("FAIL: no evidence files")
        return 1
    seen: dict[str, str] = {}
    errors: list[str] = []
    for path in files:
        errs = check_file(path, seen)
        errors += errs
        print(f"{'FAIL' if errs else 'PASS'}  {path.name}")
    for e in errors:
        print(f"  - {e}")
    print(f"{len(files)} evidence files, {len(seen)} figures, {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
