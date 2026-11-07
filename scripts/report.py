"""Write the analysis report (capability, access and sustainability tables) of one or more reader-matrix runs.

Zero network, zero spend: it reads run directories, the item file, the item pool, the price book and the energy-band
file, and writes the report as JSON + markdown.

  py -3.12 scripts/report.py --run-dir data/runs/cloud_comparator --label cloud
  py -3.12 scripts/report.py --rows data/runs/main/rows_final.jsonl \
      --summary data/runs/main/summary_final.json --label main     # a re-scoring's rows
  py -3.12 scripts/report.py --run-dir data/runs/<a> --run-dir data/runs/<b> --label both \
      --items data/items/questions.jsonl --spec data/analysis/spec.json \
      --price-book <corrected book>.json --price-book-label corrected --billing pass

The paper's report, data/analysis/reports/paper_report.json (one command line):

  py -3.12 scripts/report.py --run-dir data/runs/main --run-dir data/runs/records_built_by_gptoss20b
      --run-dir data/runs/records_built_by_qwen32b --run-dir data/runs/cloud_comparator
      --rows data/runs/main/rows_final.jsonl --summary data/runs/main/summary_final.json
      --rows data/runs/records_built_by_gptoss20b/rows_final.jsonl
      --summary data/runs/records_built_by_gptoss20b/summary_final.json
      --rows data/runs/records_built_by_qwen32b/rows_final.jsonl
      --summary data/runs/records_built_by_qwen32b/summary_final.json --spec data/analysis/spec.json --label paper

A run's rows are the file its last live invocation names, unless --rows names another rows file of that run directory
(the directory is the file's parent; the run is added if --run-dir does not list it); --summary gives the summary of
those rows for the self-check (without it the report lists the run as having no summary). Every $ is re-priced on the
one --price-book (default: the snapshot the runs were priced on); --billing records the state of the billing reconciliation
(usage x price within 5% of the provider invoice), pending / pass / fail, and anything but pass keeps the report AMBER
for dollars (its dollar figures are not results). The supply / power check (whether the item pool holds enough branch
pairs and units) is counted on the item pool (--pool, repeatable; default DEFAULT_POOL: the three admitted batches of
branch questions (as-of date x branch x exception) and the pool of branch pairs), each file's LF sha256 recorded.
--spec executes an analysis spec (scripts/make_spec.py: its population, tiers, primary comparisons and sensitivities;
the report records its path); without it every table is descriptive. --manifest (the corpus manifest the runs were
planned with) and --external (the external version dates of the revisions) enable the memory read-back
(procedures.analysis.readback: whether a summary (M5) build was truncated, and whether each record of a records memory
(M7) is located in its revision) for the two exploratory memory sensitivities. The read-back reads the builders'
outputs, which only the raw build ledgers hold: on the released runs (slimmed ledgers) the two sensitivities are
reported as not computed without --manifest, and with it the read-back stops with that message.
Outputs: <out-dir>/<label>_report.json and .md (default out/reports/); the markdown is also printed. The report the
paper was written from is data/analysis/reports/paper_report.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from procedures import ROOT, stored
from procedures.analysis.capability import REFERENCE_PROFILE
from procedures.analysis.costs import load_bands
from procedures.analysis.data import load_run
from procedures.analysis.readback import memory_readback
from procedures.analysis.report import BILLING_STATES, analyse, write_report
from procedures.analysis.spec import load_spec
from procedures.readers import load_panel, load_price_book
from procedures.readers.pricing import DEFAULT_PRICE_BOOK
from procedures.corpus.revisions import RevisionIndex

ITEMS_DIR = ROOT / "data" / "items"
DEFAULT_POOL = (ITEMS_DIR / "pool_branch_questions_1.jsonl", ITEMS_DIR / "pool_branch_questions_2.jsonl",
                ITEMS_DIR / "pool_branch_questions_3.jsonl", ITEMS_DIR / "pool_branch_pairs.jsonl")


def read_jsonl(path: Path) -> list[dict]:
    return stored.read_jsonl(path)


def load_pool(paths) -> tuple[list[dict], list[dict]]:
    """(the pool's items, one per item id; per file its path and LF sha256). An item id in two files must be the same
    item."""
    items: dict[str, dict] = {}
    sources = []
    for path in paths:
        path = Path(path)
        data = path.read_bytes().replace(b"\r\n", b"\n")
        shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        sources.append({"path": shown.as_posix(), "sha256": hashlib.sha256(data).hexdigest()})
        for it in read_jsonl(path):
            if items.setdefault(it["item_id"], it) != it:
                raise ValueError(f"{path}: item {it['item_id']} differs from the same id in an earlier pool file")
    return list(items.values()), sources


def repo_path(path: Path) -> str:
    """`path` relative to the repository root when it lies inside it (POSIX form), else as given."""
    path = Path(path)
    return (path.resolve().relative_to(ROOT) if path.resolve().is_relative_to(ROOT) else path).as_posix()


def file_identity(path: Path) -> dict:
    return {"path": repo_path(path),
            "sha256": hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()}


def readback_of(runs, manifest: Path | None, external: Path | None) -> dict | None:
    """The memory read-back of the runs' LLM-built memories over the manifest's revision texts, or None without one."""
    if manifest is None:
        if external is not None:
            raise ValueError("--external is given without --manifest")
        return None
    docs = {doc for run in runs for e in ((run.memory_requests or {}).get("memories") or {}).values()
            for doc, _ in e["revisions"]}
    index = RevisionIndex.from_manifest(manifest, external, docs=docs)
    return memory_readback(runs, index, source={"manifest": file_identity(manifest),
                                                "external": file_identity(external) if external else None})


def run_files(run_dirs, rows, summaries) -> list[tuple[Path, str | None, str | None]]:
    """(run dir, rows file name or None, summary file name or None) per run, in order."""
    chosen = {Path(p).resolve().parent: Path(p).name for p in rows}
    if len(chosen) != len(rows):
        raise ValueError("--rows names two rows files of one run directory")
    given = {Path(p).resolve().parent: Path(p).name for p in summaries}
    stray = sorted(str(d) for d in given if d not in chosen)
    if stray or len(given) != len(summaries):
        raise ValueError(f"--summary without its --rows (or twice for one run): {stray}")
    dirs = list(dict.fromkeys([Path(d).resolve() for d in run_dirs] + list(chosen)))
    return [(d, chosen.get(d), given.get(d)) for d in dirs]


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", type=Path, action="append", default=[], dest="run_dirs",
                    help="a run directory, e.g. data/runs/main (repeatable)")
    ap.add_argument("--rows", type=Path, action="append", default=[],
                    help="a rows file of a run directory to read instead of the last live invocation's (repeatable)")
    ap.add_argument("--summary", type=Path, action="append", default=[],
                    help="the summary of a --rows file, for the self-check (repeatable)")
    ap.add_argument("--items", type=Path, default=ROOT / "data" / "items" / "questions.jsonl",
                    help="the runs' item file (default: the paper's item file)")
    ap.add_argument("--pool", type=Path, action="append",
                    help="item pool file for the supply / power check (repeatable; default DEFAULT_POOL)")
    ap.add_argument("--price-book", type=Path, default=DEFAULT_PRICE_BOOK,
                    help="price book every $ is re-priced on (default: the snapshot the runs were priced on)")
    ap.add_argument("--price-book-label", default="snapshot", help="name of the book in the report (e.g. corrected)")
    ap.add_argument("--energy-bands", type=Path, default=ROOT / "data" / "analysis" / "energy_bands.json",
                    help="per-profile energy bands for the report's energy section")
    ap.add_argument("--reference", default=REFERENCE_PROFILE,
                    help="the reference reader of the reader-level Gap (read at M0@full; default the B1 reader)")
    ap.add_argument("--billing", choices=BILLING_STATES, default="pending",
                    help="state of the billing reconciliation; anything but pass keeps dollars AMBER")
    ap.add_argument("--spec", type=Path,
                    help="an analysis spec (the paper's: data/analysis/spec.json)")
    ap.add_argument("--manifest", type=Path, help="the corpus manifest the runs were planned with (memory "
                    "read-back for the two exploratory memory sensitivities; needs the raw build ledgers)")
    ap.add_argument("--external", type=Path, help="the external version dates of the revisions, with --manifest")
    ap.add_argument("--label", required=True, help="report label; the outputs are <label>_report.json and .md")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "out" / "reports",
                    help="output directory (default out/reports)")
    a = ap.parse_args(argv)
    if not a.run_dirs and not a.rows:
        ap.error("give at least one --run-dir or --rows")
    runs = [load_run(d, rows, summary) for d, rows, summary in run_files(a.run_dirs, a.rows, a.summary)]
    pool, sources = load_pool(a.pool or DEFAULT_POOL)
    report = analyse(runs, read_jsonl(a.items), load_panel(), load_price_book(a.price_book),
                     load_bands(a.energy_bands), pool=pool, pool_sources=sources, book_label=a.price_book_label,
                     reference=a.reference, billing=a.billing, spec=load_spec(a.spec) if a.spec else None,
                     spec_path=repo_path(a.spec) if a.spec else None,
                     readback=readback_of(runs, a.manifest, a.external))
    json_path, md_path = write_report(report, a.out_dir, a.label)
    print(md_path.read_text(encoding="utf-8"))
    shown = [str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p) for p in (json_path, md_path)]
    print("outputs: " + " , ".join(shown))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
