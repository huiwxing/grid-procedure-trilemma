"""How a path is written into a record or a log line: `display` spells it in POSIX form relative to a root
directory (its callers pass the repository root), or as given when it lies outside that root.
"""
from __future__ import annotations

from pathlib import Path


def display(path: Path, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()
