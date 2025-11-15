#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Index all .py files in the repository (from the repository root) and write them
to PYFILES_INDEX.txt at the repo root. The script excludes common VCS and IDE
folders (.git, .idea) and Python cache directories (__pycache__).

Usage:
    python tools/index_py_files.py

Output:
    - Prints the list of .py files (relative paths)
    - Writes the same list to PYFILES_INDEX.txt in the repository root

Note: Run this from any location; the script infers the repo root as the parent
of the "tools" directory where this script is stored.

Example command to run from repo root:
    python tools\\index_py_files.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Set


IGNORED_DIRS: Set[str] = {".git", ".idea", "__pycache__", "patch_analysis_result"}


def find_py_files(root: Path) -> Iterable[Path]:
    """
    Yield relative paths to all .py files under `root`, skipping ignored dirs.
    Returned paths are POSIX-style (forward slashes) as strings.
    """
    for p in root.rglob("*.py"):
        # Skip files that live inside any ignored directory
        if any(part in IGNORED_DIRS for part in p.parts):
            continue
        yield p.relative_to(root)


def write_index(root: Path, out_name: str = "PYFILES_INDEX.txt") -> int:
    """
    Write the list of .py files to out_name in `root`. Returns count of files.
    """
    pyfiles = sorted(str(p.as_posix()) for p in find_py_files(root))
    out_path = root / out_name
    out_path.write_text("\n".join(pyfiles) + ("\n" if pyfiles else ""), encoding="utf-8")
    return len(pyfiles)


def main() -> None:
    # Assume this script lives in tools/, so repo root is parent of that folder.
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    count = write_index(repo_root)
    print(f"Indexed {count} .py files and wrote PYFILES_INDEX.txt at: {repo_root}")
    if count:
        print("--- files ---")
        for p in sorted(find_py_files(repo_root)):
            print(p.as_posix())


if __name__ == "__main__":
    main()
