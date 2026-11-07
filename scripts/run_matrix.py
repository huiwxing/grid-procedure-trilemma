"""Run a matrix of readers x memory conditions over an item file (by default a dry run: it prices, sends nothing).

The default dry run prices the pending calls and sends nothing; --execute makes the paid calls. Inputs: an item file
(--items), a corpus manifest (--manifest) and the external version dates (--external) from which the memories are built,
the reader panel (data/panel) and, for a paid run, an approval (--approval). Output: a run directory
<out-root>/<label>_<sha12>/ (default out/runs/; procedures.matrix.run_matrix) holding the run manifest, the call ledgers
(read__<reader>.jsonl, build__<builder>.jsonl, query__<compressor>.jsonl), invocations.jsonl and the scored rows; a JSON
report is printed.

  py -3.12 scripts/run_matrix.py --label main --grid main --repeats 3 --k1-slice   # the paper's main run, dry run
  py -3.12 scripts/run_matrix.py --label records --grid cbuilt_gptoss20b --repeats 3 --k1-slice   # dry run
  py -3.12 scripts/run_matrix.py --label cloud --grid cloud_reference           # the cloud comparator, dry run
  py -3.12 scripts/run_matrix.py --label main --grid main --repeats 3 --k1-slice
      --execute --approval <id>                               # (one line) paid: only under an approval file
  py -3.12 scripts/run_matrix.py --label main --grid main --repeats 3 --k1-slice
      --execute --approval <id> --build-only --stage-cap 5    # (one line) paid: LLM builds only, at most $5 of them
  py -3.12 scripts/run_matrix.py --label main --grid main --repeats 3 --k1-slice --reuse-build-from data/runs/main
      # (one line) dry run: count the build requests another run's build ledger already answered (imported, not
      # sent); with --execute they are copied in
  py -3.12 scripts/run_matrix.py --label try --condition B0 --item-id 05807dfc --item-id 074ea795
      --readers C2_qwen3_8b_hf_nscale_nothink                                 # (one line) a few named items only

A condition is written <rep>@<budget>[/<builder or compressor>]: B0 closed book (no procedure text), M0 full text of
the revisions in effect on the as-of date (@full), M2 BM25 retrieval, M4 constraint sentences, M5 an LLM summary and
M7 typed constraint records (both built once per revision by the builder), M6 a query-focused LLM compression (one
call per question, by the compressor).

--grid picks the default conditions, readers and builder (procedures.matrix.grids.GRIDS; default "main"). The grids
are those of the paper's runs: "main" (the server and single-GPU readers, memories built by DeepSeek-V4.1-Flash;
budgets 2k / 4k / 16k, M6 at 4k only), "cbuilt_gptoss20b" and "cbuilt_qwen32b" (records memories, M7 at 2k / 4k /
16k, built by gpt-oss-20b or Qwen3-32B and read by the single-GPU readers) and "cloud_reference" (the cloud
comparator on B0, M0@full and M2@4000, read once: --repeats 1, no --k1-slice). --condition / --readers / --builder
override the grid.
--repeats K re-reads a fixed 20% item subset K times in all (procedures.matrix.run_matrix.repeat_subset); with
--k1-slice the 4k slice (B0, M0@full and every condition at 4k) re-reads that subset united with the primary analysis
population, K times as well, so that every population cell of the paper's primary slice is read more than once
(procedures.matrix.run_matrix.repeat_sets; part of the run identity). The paper's main and records runs used
--repeats 3 --k1-slice.
--item-id keeps only the items whose id starts with each given prefix (each must match exactly one item).
Memories come from procedures.memory.build_memory over the corpus manifest (the revision ladders of the items'
documents only) with the external version dates of data/corpus/version_dates_external.csv. --manifest defaults to the
corpus manifest the paper's runs read (data/corpus/manifest.csv). The run identity binds the manifest file (its path
and sha256, recorded in the run manifest as corpus_manifest), so a run resumes in its directory only while the
manifest is unchanged: edit the manifest between runs, not during one.
Budget (procedures.readers.budget): --execute requires --approval <id>, the approval file
data/budget/approval_<id>.json that grants the spend for one batch of paid runs (procedures.readers.budget
.write_approval writes it); --label must be one of its arms, --cap-usd defaults to that arm's cap and may not exceed
it. Every reservation is checked against the approval (all runs bound to it, pending unknown costs and other
executors' reservations included). --stage-cap USD (with --build-only) checks and caps the build stage only; the full
command re-checks the whole run. A dry run with --approval reports the approval's tally and the pre-flight amount.
Exit status 1 = refused or stopped (cap, approval, circuit breaker): the ledgers are intact and the same command
resumes.
--timeout-override PROFILE_ID=SECONDS (repeatable) raises one sender's client read timeout above its panel timeout
for this invocation (e.g. a route whose answers outlast 120 s); like concurrency it is transport: the panel file, the
run id, the fingerprints and the ledger headers are unchanged, and invocations.jsonl records the overrides.
--workers-override PROFILE_ID=N (repeatable) likewise raises one sender's in-flight limit (its panel max_workers) to N
(at most 8) for this invocation; a lower limit is not an override (the caller's concurrency lowers it).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from procedures import ROOT
from procedures.corpus.frozen_manifest import DEFAULT_MANIFEST, manifest_record
from procedures.matrix.sanity import select_items
from procedures.matrix.grids import GRIDS
from procedures.readers import SpendCapExceeded, load_panel
from procedures.readers.budget import APPROVALS_ROOT, load_approval
from procedures.readers.ledger import LockBusy
from procedures.readers.executor import RouteTripped, RunPaused, parse_timeout_override, parse_workers_override
from procedures.memory import RevisionIndex, build_memory
from procedures.readers.cells import BUILT_ONCE
from procedures.matrix.run_matrix import RUNS_ROOT, lf_sha256, parse_condition, run_matrix


def timeout_override(text: str) -> tuple[str, float]:
    try:
        return parse_timeout_override(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def workers_override(text: str) -> tuple[str, int]:
    try:
        return parse_workers_override(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", required=True, help="run label; the run id is <label>_<sha12 of the run identity>")
    ap.add_argument("--grid", choices=sorted(GRIDS), default="main",
                    help="default conditions, readers and builder (procedures.matrix.grids; default: main, the "
                         "paper's main run)")
    ap.add_argument("--items", type=Path, default=ROOT / "data" / "items" / "questions.jsonl",
                    help="item file, one JSON item per line (default: the paper's item file)")
    ap.add_argument("--item-id", action="append", dest="item_ids", metavar="PREFIX",
                    help="keep only the item whose id starts with PREFIX (repeatable; each must match one item)")
    ap.add_argument("--readers", nargs="+", help="reader profiles (default: the grid's)")
    ap.add_argument("--condition", action="append", dest="conditions",
                    help='"REP[@BUDGET|@full][/COMPRESSOR][:reader,...]", repeatable (M6 without /COMPRESSOR: '
                         'each reader compresses for itself); default: the grid\'s conditions')
    ap.add_argument("--builder", help="profile that builds the M5 / M7 memories (default: the grid's)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="reads per cell of the repeated 20%% item subset (1 = no repeats)")
    ap.add_argument("--k1-slice", action="store_true",
                    help="the 4k slice re-reads the 20%% subset united with the primary population, so each of its "
                         "cells is read more than once (needs --repeats > 1)")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                    help="corpus manifest (default: data/corpus/manifest.csv, the paper's)")
    ap.add_argument("--external", type=Path, default=ROOT / "data" / "corpus" / "version_dates_external.csv",
                    help="external version dates of the procedure revisions (CSV)")
    ap.add_argument("--cap-usd", type=float,
                    help="spend cap over every ledger of the run (default: the approval's arm; a dry run without "
                         "--approval: 10)")
    ap.add_argument("--approval", metavar="ID", help="approval data/budget/approval_<ID>.json (required by --execute)")
    ap.add_argument("--approvals-root", type=Path, default=APPROVALS_ROOT, help=argparse.SUPPRESS)
    ap.add_argument("--stage-cap", type=float, metavar="USD",
                    help="with --build-only: cap on the build ledger (spent + pending builds); only the build stage "
                         "is pre-checked")
    ap.add_argument("--out-root", type=Path, default=RUNS_ROOT,
                    help="parent of the run directory <label>_<sha12> (default out/runs)")
    ap.add_argument("--execute", action="store_true", help="paid run (default: dry run, nothing is sent)")
    ap.add_argument("--build-only", action="store_true",
                    help="with --execute: build the LLM memories live and stop before any reader call")
    ap.add_argument("--timeout-override", action="append", type=timeout_override, dest="timeout_overrides",
                    metavar="PROFILE_ID=SECONDS",
                    help="raise the client read timeout of one profile this run sends (repeatable; > its panel "
                         "timeout, which stays in the run identity; transport only, recorded in invocations.jsonl)")
    ap.add_argument("--workers-override", action="append", type=workers_override, dest="workers_overrides",
                    metavar="PROFILE_ID=N",
                    help="raise the in-flight limit (max_workers) of one profile this run sends (repeatable; >= its "
                         "panel max_workers, <= 8; transport only, recorded in invocations.jsonl)")
    ap.add_argument("--reuse-build-from", type=Path, metavar="RUN_DIR",
                    help="import the build records of another run (same builder identity) whose requests this run "
                         "would send; a dry run only counts them")
    a = ap.parse_args(argv)
    timeouts = dict(a.timeout_overrides or [])
    if len(timeouts) != len(a.timeout_overrides or []):
        ap.error("--timeout-override names a profile twice")
    in_flight = dict(a.workers_overrides or [])
    if len(in_flight) != len(a.workers_overrides or []):
        ap.error("--workers-override names a profile twice")
    if a.build_only and not a.execute:
        ap.error("--build-only builds for real; it needs --execute")
    if a.k1_slice and a.repeats < 2:
        ap.error("--k1-slice re-reads the 4k slice; it needs --repeats > 1")
    if a.stage_cap is not None and not a.build_only:
        ap.error("--stage-cap caps the build stage of a --build-only run")
    manifest = a.manifest
    if a.execute and not a.approval:
        ap.error("--execute needs --approval <id> (data/budget/approval_<id>.json, the spend grant for this batch)")
    approval = load_approval(a.approval, a.approvals_root) if a.approval else None
    if approval is not None and a.label not in approval.arms:
        ap.error(f"approval {approval.approval_id} has no arm for label {a.label!r} (arms: {sorted(approval.arms)})")
    cap_usd = a.cap_usd if a.cap_usd is not None else (approval.arms[a.label] if approval is not None else 10.0)
    grid = GRIDS[a.grid]
    conditions = [parse_condition(c) for c in (a.conditions or grid.conditions)]
    items = [json.loads(x) for x in a.items.read_text(encoding="utf-8").splitlines() if x.strip()]
    if a.item_ids:
        items = select_items(items, a.item_ids)
    index = RevisionIndex.from_manifest(manifest, a.external, docs={it["doc"] for it in items})  # the items' ladders
    corpus_sha = hashlib.sha256(f"{lf_sha256(manifest)}|{lf_sha256(a.external)}".encode()).hexdigest()
    built_once = any(c.representation in BUILT_ONCE for c in conditions)
    try:
        report = run_matrix(items, conditions, a.readers or list(grid.readers), index, build_memory=build_memory,
                            panel=load_panel(), label=a.label, cap_usd=cap_usd, corpus_index_sha256=corpus_sha,
                            dry_run=not a.execute, builder_id=(a.builder or grid.builder) if built_once else None,
                            out_root=a.out_root, build_only=a.build_only, reuse_build_from=a.reuse_build_from,
                            repeats=a.repeats, corpus_manifest=manifest_record(manifest), approval=approval,
                            stage_cap_usd=a.stage_cap, k1_slice=a.k1_slice, timeout_overrides=timeouts,
                            workers_overrides=in_flight)
    except (SpendCapExceeded, RouteTripped, RunPaused, LockBusy) as exc:
        print(json.dumps({"stopped": type(exc).__name__, "error": str(exc),
                          "resume": "ledgers intact; rerun the same command once the cause is understood"},
                         indent=1, ensure_ascii=False))
        return 1
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
