"""Mechanism sanity check of a finished reader run: did every memory really reach the reader?

    py -3.12 scripts/sanity_check.py --run-dir <run directory> [--items ...] [--manifest ...] [--out-dir ...]

Offline, reads no answer and scores nothing (procedures.matrix.sanity.check_run with mock=False): for every recorded
reader call it rebuilds the memory the cell planned from the corpus texts and checks that the prompt the call ledger
recorded carried it (so a representation that silently fell back to another, or to nothing, is caught before any result
is read), and checks the build calls behind every LLM-built memory. Writes sanity_report.json into --out-dir
(default out/sanity/<run directory name>/) and exits 1 on any failure. Needs the corpus texts
(scripts/corpus/rebuild.py) and the raw call ledgers with their prompts: the released ledgers are slimmed (no prompt
messages), so on a released run the check stops with that message, and the run's recorded check is the
sanity_report.json of its directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from procedures import ROOT, stored  # noqa: E402
from procedures.corpus.frozen_manifest import manifest_record, run_corpus_manifest  # noqa: E402
from procedures.corpus.revisions import RevisionIndex  # noqa: E402
from procedures.items.publish import display  # noqa: E402
from procedures.matrix.run_matrix import lf_sha256  # noqa: E402
from procedures.matrix.sanity import check_run, without_prompts  # noqa: E402
from procedures.memory import build_memory  # noqa: E402

ITEMS = ROOT / "data" / "items" / "questions.jsonl"
EXTERNAL = ROOT / "data" / "corpus" / "version_dates_external.csv"


def json_bytes(obj) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--items", type=Path, default=ITEMS, help="the run's item file (default: the paper's)")
    ap.add_argument("--manifest", type=Path, help="corpus manifest (default: the one the run recorded)")
    ap.add_argument("--external", type=Path, default=EXTERNAL, help="external version dates of the revisions (CSV)")
    ap.add_argument("--out-dir", type=Path, help="directory of the report (default: out/sanity/<run directory name>)")
    a = ap.parse_args(argv)
    slimmed = without_prompts(a.run_dir)
    if slimmed:
        raise SystemExit(f"{a.run_dir.name}: released (slimmed) call ledgers without prompts: {', '.join(slimmed)}; "
                         "the mechanism check needs the raw ledgers (the run's recorded check is its "
                         "sanity_report.json); nothing was written")
    items = stored.read_jsonl(a.items)
    manifest = a.manifest or run_corpus_manifest(a.run_dir)
    index = RevisionIndex.from_manifest(manifest, a.external, docs={it["doc"] for it in items})
    checks = check_run(a.run_dir, items, index, build_memory, None, mock=False)
    data = json_bytes({"run_dir": display(a.run_dir, ROOT), "items_sha256_lf": lf_sha256(a.items),
                       "corpus_manifest": manifest_record(manifest), **checks})
    name = "sanity_report.json"
    out_dir = a.out_dir or ROOT / "out" / "sanity" / a.run_dir.resolve().name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_bytes(data)
    print(json.dumps({"report": name, "reader_calls_by_condition": checks["reader_calls_by_condition"],
                      "calls_without_request": checks["calls_without_request"], "builds": len(checks["builds"]),
                      "build_calls": checks["build_calls"], "failures": checks["failures"][:20],
                      "n_failures": len(checks["failures"])}, indent=1))
    return 1 if checks["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
