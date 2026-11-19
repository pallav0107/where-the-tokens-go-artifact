"""Shared helpers for the evidence -> macro pipeline.

Evidence contract (every evidence/*.json, whoever writes it):

    {
      "schema_version": 1,
      "kind": "<short name>",
      "provenance": {
        "generated_at": "<ISO-8601 UTC>",
        "generator": "<script that wrote this file>",
        "sources": [
          {"id": "...", "description": "...", "command": "... (optional)",
           "sha256": "<64 hex>", "grade": "measured|documented-record|model-parameter|prediction"}
        ]
      },
      "aggregates": { ... free-form, aggregates only ... },
      "figures": {
        "MacroName": {"value": <number|string>, "format": "<fmt>",
                      "source": "<source id>", "desc": "..."}
      }
    }

gen_macros.py turns every figure into a LaTeX macro. Producers compute derived
numbers from rows; gen_macros only formats. No number reaches the paper by hand.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
EVIDENCE = REPO / "evidence"
SCHEMA_DIR = EVIDENCE / "schema"

GRADES = {"measured", "documented-record", "model-parameter", "prediction"}
FORMATS = {
    "int",         # 207820 -> 207{,}820
    "pct0",        # 84.07 -> 84\%
    "pct1",        # 97.1 -> 97.1\%
    "pct2", "pct3",
    "pctlt1",      # 97.14 -> 97.1\% ; 0.002 -> $<$0.1\% (never a printed zero for a positive share)
    "signedpct1",  # 13.8 -> +13.8\% ; -43.0 -> $-$43.0\%
    "float1", "float2", "float3",
    "signedfloat2",  # 0.12 -> +0.12 ; -0.53 -> $-$0.53
    "millions1",   # 48813889 -> 48.8~million
    "billions1",   # 70181692882 -> 70.2~billion
    "kilo",        # 150000 -> 150k
    "word",        # 3 -> three
    "date",        # 2026-11-15 -> 15~November 2026
    "text",        # literal (escaped)
}
MACRO_NAME = re.compile(r"^[A-Za-z]{3,60}$")
_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
          "nine", "ten", "eleven", "twelve"]
_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def figure(value: Any, fmt: str, source: str, desc: str) -> dict[str, Any]:
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt}")
    return {"value": value, "format": fmt, "source": source, "desc": desc}


def _latex_escape(text: str) -> str:
    repl = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
            "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}
    return "".join(repl.get(c, c) for c in text)


def format_value(value: Any, fmt: str) -> str:
    """Render one figure as LaTeX. Deterministic, so a re-run diffs cleanly."""
    if fmt == "int":
        # Round half up; int() truncated fractional figures (27.64 printed as 27).
        from decimal import Decimal, ROUND_HALF_UP
        rounded = int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        return f"{rounded:,}".replace(",", "{,}")
    if fmt == "pct0":
        return f"{float(value):.0f}\\%"
    if fmt == "pctlt1":
        v = float(value)
        return "$<$0.1\\%" if 0 < v < 0.05 else f"{v:.1f}\\%"
    if fmt in ("pct1", "pct2", "pct3"):
        return f"{float(value):.{fmt[-1]}f}\\%"
    if fmt == "signedpct1":
        v = float(value)
        return (f"+{v:.1f}\\%" if v >= 0 else f"$-${abs(v):.1f}\\%")
    if fmt in ("float1", "float2", "float3"):
        return f"{float(value):.{fmt[-1]}f}"
    if fmt == "signedfloat2":
        v = float(value)
        return (f"+{v:.2f}" if v >= 0 else f"$-${abs(v):.2f}")
    if fmt == "millions1":
        return f"{float(value) / 1e6:.1f}~million"
    if fmt == "billions1":
        return f"{float(value) / 1e9:.1f}~billion"
    if fmt == "kilo":
        return f"{float(value) / 1e3:.0f}k"
    if fmt == "word":
        v = int(value)
        return _WORDS[v] if 0 <= v < len(_WORDS) else f"{v:,}".replace(",", "{,}")
    if fmt == "date":
        d = _dt.date.fromisoformat(str(value)[:10])
        return f"{d.day}~{_MONTHS[d.month - 1]} {d.year}"
    if fmt == "text":
        return _latex_escape(str(value))
    raise ValueError(f"unknown format {fmt}")


# --- minimal JSON-schema subset validator (dependency-free on purpose) ---------
def validate(instance: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Supports: type, required, properties, additionalProperties (bool|schema),
    items, enum, pattern, minimum, maximum, minLength, minItems, patternProperties."""
    errors: list[str] = []
    types = schema.get("type")
    if types is not None:
        allowed = types if isinstance(types, list) else [types]
        if not any(_is_type(instance, t) for t in allowed):
            return [f"{path}: expected {allowed}, got {type(instance).__name__}"]
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} not in {schema['enum']}")
    if isinstance(instance, str):
        if "pattern" in schema and not re.search(schema["pattern"], instance):
            errors.append(f"{path}: {instance!r} does not match {schema['pattern']}")
        if len(instance) < schema.get("minLength", 0):
            errors.append(f"{path}: shorter than {schema['minLength']}")
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: {instance} < minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: {instance} > maximum {schema['maximum']}")
    if isinstance(instance, list):
        if len(instance) < schema.get("minItems", 0):
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if "items" in schema:
            for i, item in enumerate(instance):
                errors += validate(item, schema["items"], f"{path}[{i}]")
    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}: missing required '{key}'")
        props = schema.get("properties", {})
        pprops = schema.get("patternProperties", {})
        extra = schema.get("additionalProperties", True)
        for key, val in instance.items():
            if key in props:
                errors += validate(val, props[key], f"{path}.{key}")
                continue
            matched = False
            for pat, sub in pprops.items():
                if re.search(pat, key):
                    matched = True
                    errors += validate(val, sub, f"{path}.{key}")
            if matched:
                continue
            if extra is False:
                errors.append(f"{path}: unexpected key '{key}'")
            elif isinstance(extra, dict):
                errors += validate(val, extra, f"{path}.{key}")
    return errors


def _is_type(value: Any, name: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }[name]


def evidence_files() -> list[Path]:
    """Every committed evidence file: evidence/*.json (not schema/, not raw/)."""
    return sorted(p for p in EVIDENCE.glob("*.json") if p.is_file())


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # LF on every OS: gen_macros records each file's hash, and CI checks out LF.
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")
