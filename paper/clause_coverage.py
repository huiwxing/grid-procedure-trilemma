"""Was the item's own condition clause inside the BM25 retrieval context the reader saw? (Sec. III-C.)

  py -3.12 paper/clause_coverage.py [--run-dir data/runs/main] [--rows rows_final.jsonl]
      [--items data/items/questions.jsonl]

For every primary-population question (procedures.analysis.spec.in_population), rebuilds the retrieval context at the
4k budget (M2@4000: BM25 over the named revision's paragraphs) from the corpus texts (scripts/corpus/rebuild.py) and
compares it with the context the run sent (the sha256 the run manifest recorded for that question,
memory_text_sha256): a warning gives the number of contexts that differ. It then tests whether the item's
`condition_clause.text` occurs verbatim (whitespace-normalised) in the rebuilt context. The context is the same for
every reader, so the presence count is per question; per reader it prints the first-read (repeat 0) wrong answers whose
context held the clause. Exact matching, so a paraphrased or footnote-broken clause counts as absent (a lower bound on
presence). Offline; sends nothing. Needs the rebuilt corpus texts and tiktoken with its o200k_base encoding cached
(the token counts that pack the context): without them the rebuilt contexts differ from the ones the run sent, and the
counts are those of the rebuilt contexts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from procedures import stored
from procedures.analysis.spec import in_population
from procedures.corpus.frozen_manifest import run_corpus_manifest
from procedures.corpus.revisions import RevisionIndex
from procedures.memory import build_memory
from procedures.readers.tokens import token_method

READERS = ["C1_llama31_8b_hf_deepinfra", "C2_qwen3_8b_hf_nscale_nothink", "C3_gptoss20b_hf_deepinfra_low",
           "tierC2_qwen3_32b_hf_nscale_nothink"]   # the constrained panel


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("’", "'").replace("‘", "'").replace("­", "")).strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", type=Path, default=ROOT / "data" / "runs" / "main",
                    help="the main run's directory (default: data/runs/main)")
    ap.add_argument("--rows", default="rows_final.jsonl",
                    help="final rows file name inside --run-dir (default: rows_final.jsonl)")
    ap.add_argument("--items", type=Path, default=ROOT / "data" / "items" / "questions.jsonl",
                    help="the run's item file (default: data/items/questions.jsonl)")
    ap.add_argument("--external", type=Path, default=ROOT / "data" / "corpus" / "version_dates_external.csv",
                    help="external version dates of the procedure revisions (CSV)")
    ap.add_argument("--budget", type=int, default=4000, help="retrieval budget in tokens (default 4000)")
    a = ap.parse_args(argv)
    items = {it["item_id"]: it for it in stored.read_jsonl(a.items)}
    population = sorted(iid for iid, it in items.items() if in_population(it))
    recorded = json.loads((a.run_dir / "manifest.json").read_text(encoding="utf-8"))["memory_text_sha256"]
    index = RevisionIndex.from_manifest(run_corpus_manifest(a.run_dir), a.external,
                                        docs={items[i]["doc"] for i in population})
    condition = f"M2@{a.budget}"
    if f"{population[0]}|{condition}" not in recorded:
        raise SystemExit(f"the run recorded no {condition} contexts")
    present, differ = {}, []
    for iid in population:
        text = build_memory(items[iid], "M2", a.budget, index).text
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != recorded[f"{iid}|{condition}"]:
            differ.append(iid)
        present[iid] = norm(items[iid]["condition_clause"]["text"]) in norm(text)
    if differ:
        print(f"warning: {len(differ)} of {len(population)} rebuilt {condition} contexts differ from the ones the run "
              f"sent (e.g. {differ[0]}); the counts below are those of the rebuilt contexts. The run's contexts need "
              "the corpus texts (scripts/corpus/rebuild.py) and the o200k_base tokenizer (token method here: "
              f"{token_method()})", file=sys.stderr)
    exceptions = [i for i in population if items[i]["cell_kind"] == "exception"]
    print(f"{condition}: clause present in {sum(present.values())}/{len(population)} contexts; "
          f"exception questions {sum(present[i] for i in exceptions)}/{len(exceptions)}")
    rows = stored.read_jsonl(a.run_dir / a.rows)
    for reader in READERS:
        wrong = [r["item_id"] for r in rows if r["profile_id"] == reader and r["stage"] == f"read/{condition}"
                 and r["repeat_idx"] == 0 and r["item_id"] in present and r["outcome"] == "observed"
                 and (r.get("capability") or {}).get("correct") is False]
        print(f"{reader}: first-read wrong answers with the clause in context {sum(present[i] for i in wrong)}/{len(wrong)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
