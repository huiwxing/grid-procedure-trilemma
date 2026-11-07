"""Reading the released run files, which are stored gzip-compressed.

The large files of a run directory (the scored rows and the API ledgers) are released as <name>.gz. Every reader keeps
naming the plain file, as the run's invocations.jsonl and the command lines do: `stored(path)` is the plain file when it
exists, else its .gz neighbour, and `read_bytes` returns the content either way. The released records name every file
by its path in this repository, relative to the repository root.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


def stored(path: Path) -> Path:
    """`path` when it exists, else `path`.gz when that exists, else `path` (so a missing file is reported by its name)."""
    path = Path(path)
    if path.exists():
        return path
    packed = path.with_name(path.name + ".gz")
    return packed if packed.exists() else path


def read_bytes(path: Path) -> bytes:
    found = stored(path)
    data = found.read_bytes()
    return gzip.decompress(data) if found.suffix == ".gz" else data


def read_text(path: Path) -> str:
    return read_bytes(path).decode("utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in read_text(path).splitlines() if x.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(read_bytes(path)).hexdigest()


def recorded_run_id(run_dir: Path, rows: list[dict]) -> str:
    """The run id a run's rows record (`run_id`, the same on every row), whatever its directory is called now; the
    directory name for rows without one."""
    ids = sorted({r.get("run_id") for r in rows} - {None})
    if len(ids) > 1:
        raise ValueError(f"{run_dir}: rows of several runs {ids}")
    return ids[0] if ids else Path(run_dir).name


def plain_name(path: Path) -> str:
    """The file name without a trailing .gz: what a ledger or rows file is called in a run's records."""
    name = Path(path).name
    return name[:-3] if name.endswith(".gz") else name
