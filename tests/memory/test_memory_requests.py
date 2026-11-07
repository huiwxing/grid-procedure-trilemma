"""procedures.memory.requests: the map from a memory's cache_key to the call-ledger requests that built it (every cap
level of a records memory, M7, built once per revision; the one call of a query-focused compression, M6). The map is
named after its content, written by the live matrix's scoring and named on every row, re-planned for a finished run
(procedures.matrix.run_matrix.replan_memories) and checked against what the run's call ledgers hold;
procedures.scoring.score_path.with_memory_requests. Synthetic corpus and mocked endpoints; no network."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from items_fixture import make_items
from test_run_matrix_m6_m7 import BUILDER, C1, C2, Grid, index, load, run  # noqa: F401  (index is a fixture)
from procedures.readers.ledger import read_ledger
from procedures.memory import build_memory
from procedures.memory.requests import SCHEMA, load_memory_requests, memory_requests, write_memory_requests
from procedures.matrix.run_matrix import replan_memories
from procedures.scoring.score_path import with_memory_requests

CONDS = ["M2@400", "M6@400", "M7@400", "M7@800"]


def test_map_lists_every_cap_level_of_a_built_once_memory_and_the_one_compression(index):
    item = make_items()[0]
    m7a = build_memory(item, "M7", 400, index, builder_profile=BUILDER)
    m7b = build_memory(item, "M7", 800, index, builder_profile=BUILDER)
    m6 = build_memory(item, "M6", 400, index, compressor_profile=C1)
    m2 = build_memory(item, "M2", 400, index)
    got = memory_requests([m7a, m7b, m6, m2, m7a])
    assert got["schema"] == SCHEMA and set(got["memories"]) == {m7a.cache_key, m7b.cache_key, m6.cache_key}
    a, b = got["memories"][m7a.cache_key], got["memories"][m7b.cache_key]
    assert a["ledger"] == f"build__{BUILDER}" and a["representation"] == "M7" and a["budget_tokens"] == 400
    levels = [lv for blk in m7a.build["blocks"] for lv in blk["levels"]]
    assert len(a["requests"]) == len(levels) == 2 * len(m7a.build["blocks"])
    assert {r["request_hash"] for r in a["requests"]} == {lv["request_hash"] for lv in levels}
    assert a["requests"] == b["requests"]                   # M7 is built once: the budgets share every request
    c = got["memories"][m6.cache_key]
    assert c["ledger"] == f"query__{C1}" and c["requests"] == [{"request_hash": m6.build["request_hash"]}]


def test_a_written_map_reads_back_under_any_name_and_load_checks_its_schema(index, tmp_path):
    item = make_items()[0]
    mems = [build_memory(item, "M7", 400, index, builder_profile=BUILDER)]
    name = write_memory_requests(tmp_path, mems)
    assert name.startswith("memory_requests_") and write_memory_requests(tmp_path, mems) == name
    assert load_memory_requests(tmp_path / name) == memory_requests(mems)
    moved = tmp_path / "memory_requests.json"
    moved.write_bytes((tmp_path / name).read_bytes())
    assert load_memory_requests(moved) == memory_requests(mems)
    other = tmp_path / "other.json"
    other.write_text(json.dumps({**memory_requests(mems), "schema": "another.map"}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_memory_requests(other)


def test_with_memory_requests_names_the_map_on_rows_of_listed_memories_only():
    row = {"provenance": {"memory_cache_key": "k1", "request_hash": "h"}, "outcome": "observed"}
    assert with_memory_requests(row, "memory_requests_x.json", {"k1"})["provenance"]["memory_requests"] ==\
        "memory_requests_x.json"
    other = with_memory_requests(row, "memory_requests_x.json", {"k2"})
    assert other["provenance"]["memory_requests"] is None and other["provenance"]["request_hash"] == "h"
    assert "memory_requests" not in row["provenance"]                                  # the input is not changed


def test_replanned_map_of_a_live_run_attributes_every_ledger_request_and_its_tokens(index, tmp_path):
    grid = Grid(make_items(), (C1, C2))
    report = run(tmp_path, index, CONDS, transport=grid, dry_run=False)
    run_dir = Path(report["run_dir"])
    manifest = json.loads((run_dir / "manifest.json").read_bytes())
    memories = replan_memories(manifest, {it["item_id"]: it for it in make_items()}, index, build_memory)
    mapping = memory_requests(memories.values())
    rows = load(report, report["outputs"]["rows"])
    # the live scoring wrote the map (the same one the plan gives) and every row names it (None for M2 rows)
    name = report["outputs"]["memory_requests"]
    assert (run_dir / name).exists() and load_memory_requests(run_dir / name) == mapping
    assert sorted(p.name for p in run_dir.glob("memory_requests_*.json")) == [name]
    for r in rows:
        want = name if r["representation"] in ("M6", "M7") else None
        assert r["provenance"]["memory_requests"] == want
    ledgers = {p.stem: {r["request_hash"]: r for r in read_ledger(p)[1] if r["outcome"] == "observed"}
               for p in run_dir.glob("*__*.jsonl")}
    used: dict[str, set] = {}
    for r in rows:
        if r["representation"] not in ("M6", "M7"):
            continue
        entry = mapping["memories"][r["provenance"]["memory_cache_key"]]
        recs = [ledgers[entry["ledger"]][q["request_hash"]] for q in entry["requests"]
                if q["request_hash"] in ledgers[entry["ledger"]]]
        tokens = sum(x["usage"]["input_total"] + x["usage"]["output_total"] for x in recs)
        want = r["sustainability"]["build_tokens"] if r["representation"] == "M7" else\
            r["sustainability"]["query_memory_tokens"]
        assert recs and tokens == want
        used.setdefault(entry["ledger"], set()).update(x["request_hash"] for x in recs)
    for stem, records in ledgers.items():
        if not stem.startswith("read__"):
            assert used[stem] == set(records)                  # every build / compression answer has its memory
    tampered = {**manifest, "memory_cache_keys": {k: "0" * 64 for k in manifest["memory_cache_keys"]}}
    with pytest.raises(ValueError, match="differ from the run's cache keys"):
        replan_memories(tampered, {it["item_id"]: it for it in make_items()}, index, build_memory)
    with pytest.warns(UserWarning, match="items hash"):
        replan_memories(manifest, {it["item_id"]: it for it in make_items()[1:]}, index, build_memory)
