"""Re-score an existing reader run from its call ledgers, sending nothing (procedures.scoring.rescore).

  py -3.12 scripts/rescore.py --run-dir data/runs/main
      --decisions data/analysis/arbitration/decisions_main.json    # (one line)

Joins every row of the run's last live scoring to its reader call ledger by request_hash and scores it again with the
current score path: the answer-field mapping (an answer left blank in the content but given in the reasoning field is
scored from there) for the readers whose panel entry sets answer_from_reasoning_field, and the blind arbitration
decisions on the answers the scorer could not match (--decisions, optional). Writes new rows_ / summary_ files (and an
arbitration_ queue when some answer still needs arbitration) and rescore_<sha12>.json, each named after its content,
appends one line to invocations.jsonl and prints the changes and the accuracy per reader x condition before and after.
The outputs go into --out-dir (default out/rescore/<run directory name>/; the new rows are written uncompressed,
rows_<sha12>.jsonl), so the released run directory stays as it is: on a released run with its decisions file the new
rows equal the run's rows_final.jsonl. --items is the run's item file (items that differ from the ones the run
manifest records are warned about). A run with an LLM-built condition also needs its memory request map
(memory_requests.json in a released run, named on every row, so that the analysis can charge each LLM-built memory
its build and compression cost): the run directory's own, or, for a run without one, a map written from its memories
re-planned on the corpus index of --manifest (default: the corpus manifest the run recorded, else the paper's corpus
manifest; procedures.corpus.frozen_manifest.run_corpus_manifest) and --external, which must keep the cache keys the
run bound.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from procedures import ROOT
from procedures.corpus.frozen_manifest import run_corpus_manifest
from procedures.readers import load_panel
from procedures.memory import RevisionIndex
from procedures.scoring.rescore import read_jsonl, rescore_run


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", type=Path, required=True, help="the run directory to re-score")
    ap.add_argument("--items", type=Path, default=ROOT / "data" / "items" / "questions.jsonl",
                    help="the run's item file (default: the paper's item file)")
    ap.add_argument("--decisions", type=Path, help="blind arbitration decisions file of this run")
    ap.add_argument("--manifest", type=Path, help="corpus manifest for the memory re-plan (default: the run's "
                                                  "recorded one, else the paper's corpus manifest)")
    ap.add_argument("--external", type=Path, default=ROOT / "data" / "corpus" / "version_dates_external.csv",
                    help="external version dates of the procedure revisions (CSV)")
    ap.add_argument("--out-dir", type=Path,
                    help="directory for the new rows, summary, arbitration queue, rescore record and invocation line "
                         "(default: out/rescore/<run directory name>)")
    a = ap.parse_args(argv)
    items = read_jsonl(a.items)
    index = RevisionIndex.from_manifest(a.manifest or run_corpus_manifest(a.run_dir), a.external,
                                        docs={it["doc"] for it in items})
    out_dir = a.out_dir or ROOT / "out" / "rescore" / a.run_dir.resolve().name
    out = rescore_run(a.run_dir, items, load_panel(), decisions_path=a.decisions, corpus_index=index, out_dir=out_dir)
    brief = {k: out[k] for k in ("run_id", "rescore", "outputs", "arbitration_queue_after")}
    brief["changes"] = {k: v for k, v in out["changes"].items() if k != "list"}
    print(json.dumps(brief, indent=1, ensure_ascii=False))
    print("| reader | condition | n | before acc | after acc |")
    print("|---|---|---|---|---|")
    for key, v in out["accuracy"].items():
        pid, stage = key.split("|")
        before = v["before"]
        print(f"| {pid} | {stage} | {v['after']['n']} | "
              f"{'-' if before is None else format(before['acc'], '.3f')} | {v['after']['acc']:.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
