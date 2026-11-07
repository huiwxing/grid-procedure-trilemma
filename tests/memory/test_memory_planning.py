"""procedures.memory planning at grid scale: a revision index restricted to some documents keeps exactly their ladders,
dry-run summary (M5) and records (M7) memories are planned once per revision set yet stay item-specific where the
item enters (as-of, question), and incremental packing counts exactly what whole-text counting does. Synthetic
corpora from tests/memory/test_memory.py; no network."""
from __future__ import annotations

from datetime import date

import pytest

import procedures.memory.chunking as chunking
import procedures.memory.summary as m5
import procedures.memory.records as m7
from test_memory import DOC, DOC2, LONG, long_item, write_corpus, write_long_corpus
from procedures.readers.tokens import estimate_tokens
from procedures.memory import RevisionIndex, build_memory

BUILDER = "B1_dsv41flash_hf_deepinfra"


def test_an_index_of_some_documents_keeps_exactly_their_ladders(tmp_path):
    manifest = write_corpus(tmp_path)
    full = RevisionIndex.from_manifest(manifest)
    only = RevisionIndex.from_manifest(manifest, docs={DOC2, DOC + "-IP"})
    assert only.documents() == sorted({DOC2, DOC + "-IP"}) and len(full.documents()) == 3
    for doc in only.documents():
        assert only.ladder(doc) == full.ladder(doc)
    with pytest.raises(KeyError):
        only.ladder(DOC)
    assert RevisionIndex.from_manifest(manifest, docs=set()).documents() == []


def test_dry_m5_and_m7_are_planned_once_per_revision_set_and_keep_the_item_parts(tmp_path, monkeypatch):
    index = write_long_corpus(tmp_path)
    a = long_item()
    b = {**long_item(), "item_id": "other", "as_of_date": "2025-02-01",
         "question": a["question"].replace("2025-01-01", "2025-02-01").replace("percent", "minutes")}
    assert index.in_effect(LONG, date(2025, 2, 1)) == index.in_effect(LONG, date(2025, 1, 1))
    for f in (m5._plan, m5._dry_m5, m7._plan, m7._dry_m7):
        f.cache_clear()
    got = {rep: [build_memory(it, rep, 4_000, index, builder_profile=BUILDER) for it in (a, b)] for rep in ("M5", "M7")}
    assert m5._dry_m5.cache_info().misses == 1 and m5._dry_m5.cache_info().hits == 1
    assert m7._dry_m7.cache_info().misses == 1 and m7._dry_m7.cache_info().hits == 1
    for rep, (ma, mb) in got.items():
        assert ma.build["as_of"] == "2025-01-01" and mb.build["as_of"] == "2025-02-01"
        assert ma.build["blocks"] == mb.build["blocks"] and ma.text == mb.text
    assert got["M5"][0].cache_key == got["M5"][1].cache_key            # an M5 memory does not read the question
    assert got["M7"][0].cache_key != got["M7"][1].cache_key            # an M7 memory does (its ranking)
    uncached = {}
    for rep, mod in (("M5", m5), ("M7", m7)):                          # the same memories without the caches
        for name in ("_plan", f"_dry_{rep.lower()}"):
            monkeypatch.setattr(mod, name, getattr(mod, name).__wrapped__)
    for rep in ("M5", "M7"):
        uncached[rep] = [build_memory(it, rep, 4_000, index, builder_profile=BUILDER) for it in (a, b)]
        assert [m.to_dict() for m in uncached[rep]] == [m.to_dict() for m in got[rep]]


def test_incremental_packing_equals_whole_text_counting(tmp_path, monkeypatch):
    index = write_long_corpus(tmp_path)
    it = long_item()
    budgets = (300, 1_000, 4_000, 16_000)
    fast = {(rep, b): build_memory(it, rep, b, index).to_dict() for rep in ("M2", "M4") for b in budgets}
    monkeypatch.setattr(chunking, "recount", lambda before, before_tokens, after: estimate_tokens(after))
    slow = {(rep, b): build_memory(it, rep, b, index).to_dict() for rep in ("M2", "M4") for b in budgets}
    assert fast == slow and all(m["text_tokens"] <= b for (_, b), m in fast.items())
