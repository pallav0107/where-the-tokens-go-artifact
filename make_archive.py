#!/usr/bin/env python3
"""Build the upload archive of this package reproducibly, without local metadata.

    python make_archive.py --dry-run            # list what would be archived; writes nothing
    python make_archive.py --out ../agent2027_release.zip

Not run yet: the archive is built only when the package is approved for release.

What it does:
- Checks every file listed in MANIFEST.json against its sha256 and refuses on any mismatch,
  missing file or unlisted package file.
- Writes a zip with sorted entries, one fixed timestamp (FIXED_TIME) for every entry, fixed
  permissions (0644), no directory entries and no owner, user, group or host information.
  The zip format stores no owner, and no extra fields are written.
- Uses the published names (package_A/, package_B/, README.md), so the review copy with
  long folder names and the repository copy give the same archive.
- Leaves out review aids and repository files: README_REVIEW.md, .gitattributes and this
  script's own output.

A tar alternative with the same properties (GNU tar):
    tar --sort=name --mtime='2026-11-17 00:00:00Z' --owner=0 --group=0 --numeric-owner \\
        --exclude=README_REVIEW.md --exclude=.gitattributes -cf - . | gzip -n > ../agent2027_release.tar.gz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile

FIXED_TIME = (2026, 11, 17, 0, 0, 0)
EXCLUDE = {"README_REVIEW.md", ".gitattributes"}
RENAME = {"package_A_preregistrations": "package_A",
          "package_B_experiment_data_and_code": "package_B",
          "README_RELEASE.md": "README.md"}
TOP_LEVEL = {"MANIFEST.json", "README.md", "REDACTION_LOG.md", "VERIFICATION.md", "make_archive.py"}


def published_name(rel: str) -> str:
    head, _, rest = rel.partition("/")
    head = RENAME.get(head, head)
    return f"{head}/{rest}" if rest else head


def collect(root: str, out_path: str | None) -> dict[str, str]:
    files = {}
    out_abs = os.path.abspath(out_path) if out_path else None
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in (".git", "__pycache__"))
        for fn in filenames:
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            if rel in EXCLUDE or os.path.abspath(path) == out_abs:
                continue
            files[published_name(rel)] = path
    return files


def check(files: dict[str, str], root: str) -> list[str]:
    manifest = json.load(open(os.path.join(root, "MANIFEST.json"), encoding="utf-8"))
    listed = {f["path"]: f["sha256"] for f in manifest["files"]}
    problems = []
    for name, sha in listed.items():
        if name not in files:
            problems.append(f"missing: {name}")
        elif hashlib.sha256(open(files[name], "rb").read()).hexdigest() != sha:
            problems.append(f"sha256 differs: {name}")
    for name in files:
        if name not in listed and name not in TOP_LEVEL:
            problems.append(f"not in MANIFEST.json: {name}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", help="zip file to write (outside the package folder)")
    ap.add_argument("--dry-run", action="store_true", help="check and list only; write nothing")
    args = ap.parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    if not args.dry_run and not args.out:
        ap.error("give --out or --dry-run")
    files = collect(root, args.out)
    problems = check(files, root)
    if problems:
        print("\n".join(problems))
        print(f"REFUSED: {len(problems)} problem(s); nothing written")
        return 1
    total = sum(os.path.getsize(p) for p in files.values())
    if args.dry_run:
        for name in sorted(files):
            print(name)
        print(f"dry run: {len(files)} files, {total:,} bytes; every MANIFEST sha256 matches")
        return 0
    with zipfile.ZipFile(args.out, "w") as zf:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            with open(files[name], "rb") as fh:
                zf.writestr(info, fh.read(), compresslevel=9)
    digest = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    print(f"wrote {args.out}: {len(files)} files, sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
