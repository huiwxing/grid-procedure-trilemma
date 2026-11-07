"""The released layout: every file a released record names is here under that name (relative to the repository root,
or inside its run directory), stored as it is or gzip-compressed."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import re

from procedures import ROOT, stored

RUN_DIRS = sorted(p for p in (ROOT / "data" / "runs").iterdir() if (p / "manifest.json").is_file())


def exists(path: Path) -> bool:
    """A released file, stored as it is or as <name>.gz."""
    return stored.stored(path).is_file()


def doc(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def released_invocations(run_dir: Path) -> list[dict]:
    """The invocations whose outputs are released: the last live scoring and every rescoring."""
    log = stored.read_jsonl(run_dir / "invocations.jsonl")
    live = [x for x in log if x.get("outcome") == "live" and x.get("outputs")]
    return [live[-1], *[x for x in log if x.get("mode") == "rescore"]]


def run_dir_of(run_id: str) -> Path:
    found = [d for d in RUN_DIRS if doc(d / "summary_final.json")["run_id"] == run_id]
    assert len(found) == 1, run_id
    return found[0]


def test_the_released_runs_name_their_files_by_their_names_here():
    assert [d.name for d in RUN_DIRS] == ["cloud_comparator", "main", "records_built_by_gptoss20b",
                                          "records_built_by_qwen32b"]
    for run_dir in RUN_DIRS:
        assert doc(run_dir / "summary_final.json")["run_id"] == run_dir.name
        for inv in released_invocations(run_dir):
            names = [v for v in inv["outputs"].values() if v] + [inv[k] for k in ("estimate", "rescore") if k in inv]
            assert all(exists(run_dir / n) for n in names), (run_dir.name, names)
        assert (ROOT / doc(run_dir / "manifest.json")["corpus_manifest"]["path"]).is_file()
        sanity = doc(run_dir / "sanity_report.json")
        assert ROOT / sanity["run_dir"] == run_dir and (ROOT / sanity["corpus_manifest"]["path"]).is_file()
        for rows in sorted(run_dir.glob("rows_*.jsonl.gz")):
            data = gzip.decompress(rows.read_bytes())
            maps = set(re.findall(rb'"memory_requests": "([^"]+)"', data))
            assert all((run_dir / m.decode()).is_file() for m in maps), (rows.name, maps)
            assert set(re.findall(rb'"run_id": "([^"]+)"', data)) == {run_dir.name.encode()}, rows.name
        if (run_dir / "rescore.json").is_file():
            rec = doc(run_dir / "rescore.json")
            names = [rec["source"]["rows"], rec["source"]["summary"], rec["memory_requests"]["file"],
                     *(v for v in rec["outputs"].values() if v)]
            assert all(exists(run_dir / n) for n in names), names
            assert (ROOT / rec["decisions"]["path"]).is_file() and (ROOT / rec["panel"]["path"]).is_file()
            assert ROOT / rec["decisions"]["queue_file"] == run_dir / "arbitration_queue.jsonl"
    decisions = sorted((ROOT / "data" / "analysis" / "arbitration").glob("decisions_*.json"))
    assert len(decisions) == 3
    for path in decisions:
        d = doc(path)
        queue = ROOT / d["queue_file"]
        assert queue.is_file() and run_dir_of(d["run_id"]) == queue.parent
        assert path.name == f"decisions_{queue.parent.name}.json"


def test_the_analysis_panel_and_paper_records_name_files_here():
    spec = doc(ROOT / "data" / "analysis" / "spec.json")
    assert (ROOT / spec["items"]["path"]).is_file()
    panel = doc(ROOT / "data" / "panel" / "panel.json")
    assert (ROOT / panel["price_snapshot"]["path"]).is_file() and (ROOT / panel["pilot_profiles"]["path"]).is_file()
    routes = [p["route_snapshot"] for p in panel["profiles"].values() if p.get("route_snapshot")]
    assert routes and all((ROOT / r).is_file() for r in routes)
    assert (ROOT / doc(ROOT / "data" / "prices" / "prices.json")["raw_snapshots_dir"]).is_dir()
    report = doc(ROOT / "data" / "analysis" / "reports" / "paper_report.json")
    meta = report["meta"]
    assert (ROOT / meta["spec"]["path"]).is_file() and (ROOT / meta["price_book"]["path"]).is_file()
    assert (ROOT / report["energy"]["bands"]["path"]).is_file()
    assert all((ROOT / s["path"]).is_file() for s in report["status"]["supply"]["pool_sources"])
    assert len(meta["runs"]) == 4
    for run in meta["runs"]:
        run_dir = run_dir_of(run["run_id"])
        assert all(exists(run_dir / run[k]) for k in ("rows", "summary", "memory_requests") if run[k]), run
    numbers = doc(ROOT / "paper" / "numbers.json")
    assert numbers["report"] == "paper_report.json" and numbers["spec"] == meta["spec"]
    assert all(exists(ROOT / r) for r in doc(ROOT / "paper" / "exploratory.json")["rows"])
    assert all(exists(ROOT / s["path"]) for s in doc(ROOT / "paper" / "printed_values.json")["sources"])


def test_the_item_files_name_corpus_texts_relative_to_the_repository_root():
    """Every corpus path an item names is data/corpus/<family>/<file>.txt and is a text_path of the corpus manifest."""
    import csv
    with open(ROOT / "data" / "corpus" / "manifest.csv", encoding="utf-8", newline="") as f:
        manifest_paths = {r["text_path"] for r in csv.DictReader(f)}
    for name in ("questions", "questions_with_earlier_cells", "pool_branch_questions_1", "pool_branch_questions_2",
                 "pool_branch_questions_3"):
        for it in stored.read_jsonl(ROOT / "data" / "items" / f"{name}.jsonl"):
            paths = [*it.get("m0_paths", []), *(it[k] for k in ("evidence_path", "base_doc_path") if it.get(k))]
            assert paths, (name, it["item_id"])
            for p in paths:
                assert p.startswith("data/corpus/") and p in manifest_paths, (name, it["item_id"], p)


def test_a_released_run_directory_under_a_plain_name_is_still_its_run(tmp_path):
    """scripts/slim_ledgers.py writes the released ledgers of a raw run into a released run directory of any name: the
    two manifests must agree on everything but the schema id and the corpus manifest's recorded path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("slim_ledgers", ROOT / "scripts" / "slim_ledgers.py")
    slim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(slim)
    raw = {"schema": "old.run.v1", "label": "x", "items_sha256": "a" * 64,
           "corpus_manifest": {"path": "corpus/manifest_old.csv", "sha256": "b" * 64}}
    released = {**raw, "schema": "procedures.matrix.run.v1",
                "corpus_manifest": {"path": "data/corpus/manifest.csv", "sha256": "b" * 64}}
    for name, manifest in (("x_raw", raw), ("main", released),
                           ("other", {**released, "items_sha256": "c" * 64}),
                           ("edited", {**released, "corpus_manifest": {"path": "m.csv", "sha256": "d" * 64}})):
        (tmp_path / name).mkdir()
        (tmp_path / name / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert slim.same_run(tmp_path / "x_raw", tmp_path / "main")
    assert not slim.same_run(tmp_path / "x_raw", tmp_path / "other")
    assert not slim.same_run(tmp_path / "x_raw", tmp_path / "edited")
