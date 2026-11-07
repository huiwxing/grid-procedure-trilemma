"""Which call-ledger requests made each LLM-built memory: memory cache_key -> the ledger file stem and the request
hashes of its LLM calls, so that a cost is attributed to the memory that incurred it by request identity, never
inferred from item ids or revisions.

* Summary (M5) and records (M7) memories (built once by the run's builder, call ledger "build__<builder>"): every cap
  level of every sub-block of the memory's plan (`MemoryResult.build["blocks"][].levels`), whether or not the level
  was sent: a level-2 request exists in a ledger only when its level-1 answer was truncated, so the requests of a
  memory that a ledger holds are exactly the ones sent for it. A budget-free M7 build shares its requests across the
  budgets of the ladder; each memory still lists all of them (the deployment view: every memory needs its whole
  build), and a consumer that wants the bill counts each request of a ledger once.
* Compression (M6) memories (per question, call ledger "query__<compressor>"): the one compression request of the
  memory, none when the memory had no input (nothing was sent). Several questions whose compressions are the
  identical request share it the same way.

`memory_requests(memories)` takes MemoryResults, planned (dry run) or live: the plan already names every request hash
(a dry-run level carries the hash the live request will have; procedures.matrix.run_matrix checks a live memory keeps
its planned cache_key). A planned memory lists every cap level; a live one only the levels it sent.
procedures.matrix.run_matrix writes a run's map from the plan, and procedures.scoring.rescore reuses the run's map or
writes one from a re-plan (run_matrix.replan_memories), so a run has one map whichever pass wrote it.
`write_memory_requests(run_dir, memories)` stores the map as memory_requests_<sha12>.json (named after its content)
and returns its file name, which a run's rows name (procedures.scoring.score_path).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "procedures.memory.requests.v1"
BUILT_ONCE = ("M5", "M7")
QUERY_BUILT = ("M6",)
FILE_STEM = "memory_requests"


def _entry(memory) -> dict | None:
    build = memory.build or {}
    if memory.representation in BUILT_ONCE:
        profile, ledger = build["builder_profile"], f"build__{build['builder_profile']}"
        requests = sorted({(level["request_hash"], level["cap_level"], level["block_key"])
                           for block in build["blocks"] for level in block["levels"]})
        listed = [{"request_hash": h, "cap_level": c, "block_key": k} for h, c, k in requests]
    elif memory.representation in QUERY_BUILT:
        profile, ledger = build["compressor_profile"], f"query__{build['compressor_profile']}"
        listed = [] if build["mode"] == "no_input" else [{"request_hash": build["request_hash"]}]
    else:
        return None
    return {"representation": memory.representation, "budget_tokens": memory.budget_tokens, "profile": profile,
            "ledger": ledger, "revisions": [list(r) for r in memory.revisions_used], "requests": listed}


def memory_requests(memories) -> dict:
    """{"schema", "memories": cache_key -> entry} over the LLM-built memories among `memories` (an iterable of
    MemoryResult); two memories with one cache_key must list the same requests."""
    out: dict[str, dict] = {}
    for memory in memories:
        entry = _entry(memory)
        if entry is None:
            continue
        if out.setdefault(memory.cache_key, entry) != entry:
            raise ValueError(f"memory {memory.cache_key[:12]} is listed with two different request sets")
    return {"schema": SCHEMA, "memories": dict(sorted(out.items()))}


def map_bytes(mapping: dict) -> bytes:
    return (json.dumps(mapping, ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8")


def write_memory_requests(run_dir: Path, memories) -> str:
    """Write the map of `memories` into `run_dir` as memory_requests_<sha12>.json; returns the name."""
    data = map_bytes(memory_requests(memories))
    name = f"{FILE_STEM}_{hashlib.sha256(data).hexdigest()[:12]}.json"
    (Path(run_dir) / name).write_bytes(data)
    return name


def load_memory_requests(path: Path) -> dict:
    """A map file (its schema checked)."""
    path = Path(path)
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if mapping.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {mapping.get('schema')!r} is not {SCHEMA}")
    return mapping
