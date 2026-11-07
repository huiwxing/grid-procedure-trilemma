"""Offline mechanism check of the memory builders on an item file. Nothing is sent, no key is read.

  py -3.12 scripts/memory_sanity.py [--items data/items/questions.jsonl] [--budget 4000]

--items defaults to the paper's item file. For every item it builds B0 (closed book), M0 (full text, budget None),
and M2 (BM25 retrieval) / M4 (constraint sentences) / M5 (LLM summary) at --budget (M5 as a dry run by --builder,
so no LLM is called). Mechanism checks (any failure -> exit 1): the as-of revision set is exactly the item's
`revision`; M0 reads exactly the item's `m0_paths`; no memory exceeds its budget (a dry-run M5 memory without the
budget line appended after its fit, procedures.memory.summary.DRY_BUDGET_LINE).
Descriptive only (never a pass / fail criterion): the items whose M2 / M4 memory is empty (the as-of revision
holds no passage or constraint sentence for the question; the analysis keeps such cells in their condition and
reports them), whether the M2 / M4 provenance overlaps the item's gold
`source_span` in its evidence file, and the M5 dry-run builds (distinct sub-block summaries and revisions, reference
build tokens, run_cells HIGH-scenario dollars with the real prompts).
Prints one JSON object. Needs the corpus texts of the items' documents (scripts/corpus/rebuild.py --items <file>).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from procedures import ROOT
from procedures.corpus.frozen_manifest import DEFAULT_MANIFEST
from procedures.items.publish import display
from procedures.matrix.grids import BUILDER
from procedures.readers.tokens import estimate_tokens, token_method
from procedures.memory import RevisionIndex, build_memory
from procedures.memory.summary import DRY_BUDGET_LINE
from procedures.corpus.revisions import asof_revisions

BUDGETED = ("M2", "M4", "M5")


def overlaps(mem, path: str, span) -> bool:
    return any(p["path"] == path and p["char_start"] < span[1] and span[0] < p["char_end"] for p in mem.provenance)


def fitted_tokens(mem, budget: int) -> int:
    """The memory's tokens as fitted to the budget: a dry-run M5 memory ends with DRY_BUDGET_LINE, appended after the
    budget fit, so that line is not counted."""
    tail = "\n" + DRY_BUDGET_LINE.format(budget=budget)
    return estimate_tokens(mem.text[:-len(tail)]) if mem.text.endswith(tail) else mem.text_tokens


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", type=Path, default=ROOT / "data" / "items" / "questions.jsonl",
                    help="item file (default: the paper's item file)")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                    help="corpus manifest the memories are built from (default: the paper's corpus manifest)")
    ap.add_argument("--external", type=Path, default=ROOT / "data" / "corpus" / "version_dates_external.csv",
                    help="external version dates of the procedure revisions (CSV)")
    ap.add_argument("--budget", type=int, default=4_000, help="memory budget in tokens for M2 / M4 / M5")
    ap.add_argument("--builder", default=BUILDER, help="M5 builder profile (default: the DeepSeek builder)")
    a = ap.parse_args()
    items = [json.loads(x) for x in a.items.read_text(encoding="utf-8").splitlines() if x.strip()]
    index = RevisionIndex.from_manifest(a.manifest, a.external, docs={it["doc"] for it in items})   # the items' ladders
    failures: list[dict] = []
    empty: list[dict] = []
    covered, with_span = Counter(), 0
    tokens: dict[str, list[int]] = {rep: [] for rep in ("M0", *BUDGETED)}
    builds: dict[str, dict] = {}
    for it in items:
        asof = [r.revision for r in asof_revisions(it, index).revisions]
        if asof != [it["revision"]]:
            failures.append({"item_id": it["item_id"], "check": "as_of_revision", "got": asof, "want": it["revision"]})
        build_memory(it, "B0", None, index)
        m0 = build_memory(it, "M0", None, index)
        want = sorted(index.by_path(p).source_path for p in it.get("m0_paths", []))
        if want and sorted(p["path"] for p in m0.provenance) != want:
            failures.append({"item_id": it["item_id"], "check": "m0_paths", "got": m0.provenance, "want": want})
        tokens["M0"].append(m0.text_tokens)
        span = it.get("source_span")
        evidence = index.by_path(it["evidence_path"]).source_path
        with_span += bool(span)
        for rep in BUDGETED:
            kw = {"builder_profile": a.builder} if rep == "M5" else {}
            mem = build_memory(it, rep, a.budget, index, **kw)
            tokens[rep].append(mem.text_tokens)
            if fitted_tokens(mem, a.budget) > a.budget:
                failures.append({"item_id": it["item_id"], "check": f"{rep}_budget", "tokens": mem.text_tokens,
                                 "notes": mem.notes})
            elif rep != "M5" and not mem.text:
                empty.append({"item_id": it["item_id"], "representation": rep, "notes": mem.notes})
            if rep == "M5":
                for b in mem.build["blocks"]:
                    builds[b["block_key"]] = b
            elif span:
                covered[rep] += overlaps(mem, evidence, span)
    report = {
        "items": len(items), "items_file": display(a.items, ROOT), "budget": a.budget, "token_method": token_method(),
        "failures": failures,
        "empty_memories": empty,
        "tokens": {rep: {"min": min(v), "max": max(v), "mean": round(sum(v) / len(v), 1)} for rep, v in tokens.items()},
        "descriptive_gold_span_in_memory": {rep: f"{covered[rep]}/{with_span}" for rep in ("M2", "M4")},
        "m5_dry_run_builds": {"builder": a.builder, "distinct_blocks": len(builds),
                              "revisions": len({(b["doc_id"], b["revision"]) for b in builds.values()}),
                              "build_tokens": sum(b["build_tokens"] for b in builds.values()),
                              "usd_high": round(sum(b["usd"] for b in builds.values()), 4)},
    }
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
