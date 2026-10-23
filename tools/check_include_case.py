#!/usr/bin/env python3
"""Fail if any project #include "..." does not match the on-disk path case.

Windows (NTFS) and default macOS (APFS) resolve paths case-insensitively,
Linux does not: `#include "kenshin/Core/Log.h"` builds on the first two
and breaks on the third. This walks engine/, app/, tools/ and tests/ and
checks each quoted include, resolved against the including file's
directory and the project include roots, component by component against
the real directory listing.

Usage: python tools/check_include_case.py   (exit 1 on any mismatch)
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN_DIRS = ["engine", "app", "tools", "tests"]
INCLUDE_ROOTS = ["engine/include", "engine/src", "ThirdParty",
                 "ThirdParty/imgui-backends", "engine/src/backend/vulkan"]
EXTS = (".h", ".hpp", ".cpp", ".cc", ".inl", ".slang")
INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)


def exact_exists(base, rel):
    """True if base/rel exists with exactly this case in every component."""
    cur = base
    for part in rel.replace("\\", "/").split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            cur = os.path.dirname(cur)
            continue
        try:
            entries = os.listdir(cur)
        except OSError:
            return False
        if part not in entries:
            return False
        cur = os.path.join(cur, part)
    return True


def case_insensitive_exists(base, rel):
    return os.path.exists(os.path.join(base, rel))


def main():
    problems = []
    checked = 0
    for d in SCAN_DIRS:
        for dirpath, _, files in os.walk(os.path.join(ROOT, d)):
            for f in files:
                if not f.endswith(EXTS):
                    continue
                path = os.path.join(dirpath, f)
                with open(path, encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
                for inc in INCLUDE_RE.findall(text):
                    bases = [dirpath] + [os.path.join(ROOT, r) for r in INCLUDE_ROOTS]
                    hit = [b for b in bases if case_insensitive_exists(b, inc)]
                    if not hit:
                        continue  # a system/dependency header, not ours
                    checked += 1
                    if not any(exact_exists(b, inc) for b in hit):
                        rel = os.path.relpath(path, ROOT)
                        problems.append(f"{rel}: #include \"{inc}\" "
                                        "differs in case from the file on disk")
    for p in problems:
        print(p)
    print(f"check_include_case: {checked} project includes checked, "
          f"{len(problems)} case mismatches")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
