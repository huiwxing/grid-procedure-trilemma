"""Repeats (a fixed 20% item subset re-read k times, per-item agreement) and the named grids of the paper's four runs
(the main grid over every sendable B and C reader, the two C-built grids and the cloud reference; cells over a
reader's context ceiling planned, not sent, counted)."""
from collections import Counter
import json

import pytest

from procedures import ROOT, stored
from procedures.readers.ledger import read_ledger
from procedures.matrix.run_matrix import REPEAT_SEED, Condition, parse_condition, primary_population, repeat_subset
from procedures.matrix.run_matrix import run_matrix

READER = "C1_llama31_8b_hf_deepinfra"


def run(items, panel, book, tmp_path, builder, reader=None, **kw):
    args = dict(build_memory=builder, panel=panel, label="t", cap_usd=1.0, corpus_index_sha256="0" * 64,
                builder_id=None, out_root=tmp_path, book=book, transport=reader, sleep=lambda s: None,
                concurrency=1, text_of=lambda p: "59.0 Hz 15 minutes")
    args.update(kw)
    return run_matrix(items, args.pop("conditions"), args.pop("readers", [READER]), {"corpus": "fake"}, **args)


def load(tmp_path, report, key):
    return json.loads((tmp_path / report["run_id"] / report[key]).read_text(encoding="utf-8"))


def test_repeat_subset_is_an_exact_share_chosen_by_the_hash_of_each_item_id():
    import hashlib
    from procedures.matrix.run_matrix import REPEAT_SEED, repeat_subset
    ids = [f"item-{i:04d}" for i in range(102)]
    subset = repeat_subset(ids)
    assert len(subset) == 20 and subset == sorted(subset) == repeat_subset(list(reversed(ids)))   # 20.4 -> 20
    rank = sorted(ids, key=lambda i: hashlib.sha256(f"{REPEAT_SEED}|{i}".encode()).hexdigest())
    assert subset == sorted(rank[:20])
    assert len(repeat_subset(ids[:3])) == 1 and len(repeat_subset(ids[:8])) == 2 and repeat_subset([]) == []
    assert repeat_subset(ids, 1.0) == sorted(ids) and repeat_subset(ids, 0.2, seed="other") != subset
    for bad in (0, 1.5, True):
        with pytest.raises(ValueError, match="fraction"):
            repeat_subset(ids, bad)


def test_the_papers_runs_re_read_the_subset_and_population_this_code_picks():
    """The repeated items the paper's run manifests record are repeat_subset of the paper's item file with the
    recorded seed and share, and the 4k slice's population is the primary population of those items."""
    items = stored.read_jsonl(ROOT / "data" / "items" / "questions.jsonl")
    ids = [it["item_id"] for it in items]
    repeated = 0
    for path in sorted((ROOT / "data" / "runs").glob("*/manifest.json")):
        record = json.loads(path.read_text(encoding="utf-8"))["repeats"]
        assert record["seed"] == REPEAT_SEED
        if record["k"] > 1:
            repeated += 1
            assert record["item_ids"] == repeat_subset(ids, record["fraction"]) and len(record["item_ids"]) == 49
            assert record["population_item_ids"] == primary_population(items)
    assert repeated == 3


def test_repeats_resend_the_subset_and_report_per_item_agreement(items, panel, book, tmp_path, fake_builder,
                                                                 fake_reader, gold_answer):
    """k = 3 on a 50% subset: repeat 0 of every item first, then repeats 1-2 of the subset, each its own request
    (distinct hash); capability reads repeat 0 only; one item answered wrong on its last M2 repeat is the only
    disagreement."""
    from procedures.matrix.run_matrix import repeat_subset
    subset = repeat_subset([it["item_id"] for it in items], 0.5)
    assert 0 < len(subset) < len(items)
    flipper, seen = subset[0], Counter()

    def answer(item):
        seen[item["item_id"]] += 1
        if item["item_id"] == flipper and seen[flipper] == 6:   # B0, M2 (repeat 0), B0, M2 (1), B0, M2 (2)
            return {"value": "insufficient_evidence"}
        return gold_answer(item)
    conditions = [Condition("B0"), Condition("M2", 4000)]
    reader = fake_reader(items, answer)
    report = run(items, panel, book, tmp_path, fake_builder(), reader, dry_run=False, conditions=conditions,
                 repeats=3, repeat_fraction=0.5)
    per_condition = len(items) + 2 * len(subset)
    assert len(reader.calls) == 2 * per_condition
    _, records = read_ledger(tmp_path / report["run_id"] / f"read__{READER}.jsonl")
    assert len({r["request_hash"] for r in records}) == len(records) == 2 * per_condition
    assert Counter(r["repeat_idx"] for r in records) == {0: 2 * len(items), 1: 2 * len(subset), 2: 2 * len(subset)}
    assert {r["repeat_idx"] for r in records[:2 * len(items)]} == {0}                   # repeat 0 goes first
    estimate = load(tmp_path, report, "estimate")
    assert estimate["repeats"] == {"k": 3, "items": len(subset)}
    assert estimate["readers"][READER]["repeat_cells"] == 2 * 2 * len(subset)
    conditions_out = json.loads((tmp_path / report["run_id"] / report["outputs"]["summary"])
                                .read_text(encoding="utf-8"))["conditions"]
    b0, m2 = conditions_out[f"{READER}|read/B0@full"], conditions_out[f"{READER}|read/M2@4000"]
    assert b0["cells"] == len(items) and b0["capability"]["n"] == len(items) and b0["capability"]["acc"] == 1.0
    assert b0["repeats"]["correct_agreement"] == 1.0 and b0["repeats"]["usd_counted"] > 0
    rep = m2["repeats"]
    assert (rep["k"], rep["items"], rep["complete"], rep["cells"]) == (3, len(subset), len(subset), 2 * len(subset))
    assert rep["correct_agree"] == rep["answer_agree"] == len(subset) - 1
    assert rep["per_item"][flipper] == {"correct": [True, True, False], "correct_agree": False,
                                        "answer_agree": False}
    assert rep["acc_by_repeat"][:2] == [1.0, 1.0] and m2["capability"]["acc"] == 1.0    # repeat 0 is untouched
    manifest = json.loads((tmp_path / report["run_id"] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["repeats"] == {"k": 3, "fraction": 0.5, "seed": REPEAT_SEED, "item_ids": subset}
    single = run(items, panel, book, tmp_path, fake_builder(), None, conditions=conditions)
    assert single["run_id"] != report["run_id"]                                          # repeats are identity
    assert load(tmp_path, single, "estimate")["repeats"] == {"k": 1, "items": 0}
    with pytest.raises(ValueError, match="repeats"):
        run(items, panel, book, tmp_path, fake_builder(), None, conditions=conditions, repeats=0)


def test_repeat_agreement_needs_every_repeat_and_counts_incomplete_items():
    from procedures.scoring.score_path import repeat_agreement

    def row(iid, idx, correct, outcome="observed", pred="15"):
        cap = {"correct": correct, "status": "predicted", "matched_option": None, "pred": pred, "pred_unit": "min"}
        return {"item_id": iid, "repeat_idx": idx, "outcome": outcome, "capability": cap,
                "access": {"usd_counted": 0.001}}
    rows = [row("a", 0, True), row("a", 1, True), row("b", 0, True), row("b", 1, True, pred="15.0"),
            row("c", 0, False), row("c", 1, False, outcome="no_response"), row("d", 0, True)]
    out = repeat_agreement(rows, 2)
    assert (out["items"], out["complete"], out["correct_agree"], out["answer_agree"]) == (3, 2, 2, 1)
    assert out["per_item"]["c"] == {"correct": [False, None], "correct_agree": None, "answer_agree": None}
    assert "d" not in out["per_item"] and out["cells"] == 3 and out["usd_counted"] == pytest.approx(0.003)
    with pytest.raises(ValueError, match="repeats"):
        repeat_agreement([row("a", 0, True), row("a", 2, True)], 2)


POPULATION = ("l3u:a@cur", "l3u:b@cur", "l3v:a@new", "l3w:b@new")


def with_population(items, ids=POPULATION):
    """The fixture items with `ids` in the primary analysis population (procedures.analysis.spec.in_population)."""
    ruled = {"analysis_stratum": "primary", "load_bearing": True, "domain": "ops", "rule2_spirit": "abstracted"}
    return [dict(it, **ruled) if it["item_id"] in ids else dict(it, analysis_stratum="controls") for it in items]


def test_k1_slice_rereads_the_primary_population_on_the_4k_slice_only(items, panel, book, tmp_path, fake_builder,
                                                                       fake_reader, gold_answer):
    """With k1_slice the 4k slice (B0, M0@full, every condition at 4k) re-reads the union of the 20% subset and the
    primary population k times, so no primary result on the slice rests on a single read (k = 1); a condition off the
    slice (M2@8000) re-reads the subset alone. The manifest binds the slice and the population; the estimate counts
    the repeat cells per condition; every repeated cell comes back as rows with repeat_idx 0..k-1."""
    from procedures.matrix.run_matrix import K1_SLICE_BUDGET, in_k1_slice, population_sha256, primary_population, repeat_subset, repeated_for
    items = with_population(items)
    ids = sorted(it["item_id"] for it in items)
    subset = repeat_subset(ids, 0.2)
    union = sorted(set(subset) | set(POPULATION))
    assert primary_population(items) == sorted(POPULATION) and len(union) > len(subset) and K1_SLICE_BUDGET == 4000
    conditions = [Condition("B0"), Condition("M0", None, (READER,)), Condition("M2", 4000), Condition("M2", 8000)]
    assert [in_k1_slice(c) for c in conditions] == [True, True, True, False]
    report = run(items, panel, book, tmp_path, fake_builder(), None, conditions=conditions, repeats=3,
                 repeat_fraction=0.2, k1_slice=True)
    manifest = json.loads((tmp_path / report["run_id"] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["repeats"] == {"k": 3, "fraction": 0.2, "seed": REPEAT_SEED, "item_ids": subset,
                                   "k1_slice": True, "k1_slice_conditions": ["B0@full", "M0@full", "M2@4000"],
                                   "population_item_ids": sorted(POPULATION), "population_n": 4,
                                   "population_sha256": population_sha256(POPULATION)}
    estimate = load(tmp_path, report, "estimate")
    assert estimate["repeats"] == {"k": 3, "items": len(subset),
                                   "k1_slice": {"conditions": ["B0@full", "M0@full", "M2@4000"], "items": len(union)}}
    assert estimate["readers"][READER]["repeat_cells"] == 2 * (3 * len(union) + len(subset))
    assert repeated_for(manifest["repeats"], "M2", 4000) == set(union) == repeated_for(manifest["repeats"], "B0", None)
    assert repeated_for(manifest["repeats"], "M2", 8000) == set(subset) and repeated_for(None, "M2", 4000) == set()
    plain = run(items, panel, book, tmp_path, fake_builder(), None, conditions=conditions, repeats=3,
                repeat_fraction=0.2)
    assert plain["run_id"] != report["run_id"]                                  # the slice is part of the identity
    plain_manifest = json.loads((tmp_path / plain["run_id"] / "manifest.json").read_text(encoding="utf-8"))
    assert plain_manifest["repeats"] == {"k": 3, "fraction": 0.2, "seed": REPEAT_SEED,
                                         "item_ids": subset}                  # no k1 key without the slice
    assert repeated_for(plain_manifest["repeats"], "M2", 4000) == set(subset)
    live = run(items, panel, book, tmp_path, fake_builder(), fake_reader(items, gold_answer), dry_run=False,
               conditions=conditions, repeats=3, repeat_fraction=0.2, k1_slice=True)
    assert live["run_id"] == report["run_id"]
    rows = [json.loads(x) for x in (tmp_path / live["run_id"] / live["outputs"]["rows"]).read_text(encoding="utf-8")
            .splitlines()]
    reads = Counter((r["stage"], r["item_id"]) for r in rows if r["outcome"] == "observed")
    for stage in ("read/B0@full", "read/M0@full", "read/M2@4000"):
        assert {i: reads[(stage, i)] for i in ids} == {i: 3 if i in union else 1 for i in ids}, stage
    assert {i: reads[("read/M2@8000", i)] for i in ids} == {i: 3 if i in subset else 1 for i in ids}
    summary = json.loads((tmp_path / live["run_id"] / live["outputs"]["summary"]).read_text(encoding="utf-8"))
    assert summary["conditions"][f"{READER}|read/M2@4000"]["repeats"]["complete"] == len(union)
    assert summary["conditions"][f"{READER}|read/M2@8000"]["repeats"]["complete"] == len(subset)


def test_k1_slice_refuses_a_run_it_could_not_widen(items, panel, book, tmp_path, fake_builder):
    """k1_slice needs repeats > 1, a primary population among the items and a 4k-slice condition: otherwise it would
    silently re-read nothing more."""
    with pytest.raises(ValueError, match="repeats > 1"):
        run(with_population(items), panel, book, tmp_path, fake_builder(), None, conditions=[Condition("B0")],
            k1_slice=True)
    with pytest.raises(ValueError, match="primary population"):
        run(items, panel, book, tmp_path, fake_builder(), None, conditions=[Condition("B0")], repeats=3,
            k1_slice=True)
    with pytest.raises(ValueError, match="4k-slice condition"):
        run(with_population(items), panel, book, tmp_path, fake_builder(), None, conditions=[Condition("M2", 8000)],
            repeats=3, k1_slice=True)


def test_main_grid_is_the_papers_matrix(panel):
    """The paper's main run: B0, M0@full for the three server readers, M2 / M4 / M5 / M7 at 2k / 4k / 16k, M6 at 4k
    only in two forms (the server readers read the B1-direct compression, the single-GPU readers compress for
    themselves); the eight B and C readers, every sendable B and C profile of the panel; the B1-direct builder."""
    from procedures.matrix.grids import BUILDER, GRIDS, SERVER_READERS, SINGLE_GPU_READERS
    grid = GRIDS["main"]
    conditions = [parse_condition(c) for c in grid.conditions]
    assert grid.readers == SERVER_READERS + SINGLE_GPU_READERS and len(grid.readers) == 8
    assert len(SERVER_READERS) == 3 and len(SINGLE_GPU_READERS) == 5 and grid.builder == BUILDER
    assert [c.key for c in conditions[:2]] == ["B0@full", "M0@full"] and conditions[1].readers == SERVER_READERS
    assert conditions[0].readers is None
    assert [(c.representation, c.budget_tokens) for c in conditions[2:] if c.representation != "M6"] == [
        (rep, b) for rep in ("M2", "M4", "M5", "M7") for b in (2000, 4000, 16000)]
    assert all(c.readers is None for c in conditions[2:] if c.representation != "M6")
    m6 = [c for c in conditions if c.representation == "M6"]
    assert [(c.budget_tokens, c.compressor, c.readers) for c in m6] == [
        (4000, "B1_dsv41flash_direct_nothink", SERVER_READERS), (4000, None, SINGLE_GPU_READERS)]
    assert len(conditions) == 2 + 4 * 3 + 2 and len({c.key for c in conditions}) == len(conditions)
    assert not any(c.budget_tokens == 8000 for c in conditions)
    assert panel.get(BUILDER).tier == "B" and {panel.get(pid).tier for pid in SERVER_READERS} == {"B"}
    sendable_bc = {pid for pid, pp in panel.profiles.items()
                   if pp.role in ("panel", "extension") and pp.tier in ("B", "C")}
    assert set(grid.readers) == sendable_bc and "tierB_llama33_70b_hf_deepinfra" not in grid.readers
    assert sorted(panel.get(pid).access_tier for pid in grid.readers) == ["B", "B", "B", "C1", "C1", "C1", "C2", "C3"]


def test_cbuilt_grids_and_the_cloud_reference_grid(panel):
    """The C-built grids (records memories, M7, at 2k / 4k / 16k, read by the five single-GPU readers only) once per
    C builder, and the cloud reference (A2 Sonnet 5 on B0, M0@full, M2@4k; no builder), which the reader matrix plans
    without a builder."""
    from procedures.matrix.grids import C_BUILDER, GRIDS, SINGLE_GPU_READERS
    for name, builder, access in (("cbuilt_gptoss20b", C_BUILDER, "C1"),
                                  ("cbuilt_qwen32b", "tierC2_qwen3_32b_hf_nscale_nothink", "C2")):
        grid = GRIDS[name]
        assert [(c.representation, c.budget_tokens, c.readers) for c in map(parse_condition, grid.conditions)] == [
            ("M7", b, None) for b in (2000, 4000, 16000)], name
        assert grid.readers == SINGLE_GPU_READERS and grid.builder == builder, name
        assert panel.get(builder).tier == "C" and panel.get(builder).access_tier == access, name
    assert SINGLE_GPU_READERS == tuple(pid for pid in GRIDS["main"].readers if panel.get(pid).tier == "C")
    assert GRIDS["main"].builder != C_BUILDER
    aref = GRIDS["cloud_reference"]
    assert aref.readers == ("A2_sonnet5_nothink",) and aref.builder is None
    assert [(c.key, c.readers) for c in map(parse_condition, aref.conditions)] == [
        ("B0@full", None), ("M0@full", ("A2_sonnet5_nothink",)), ("M2@4000", None)]
    assert panel.get("A2_sonnet5_nothink").tier == "A"
    assert set(GRIDS) == {"main", "cbuilt_gptoss20b", "cbuilt_qwen32b", "cloud_reference"}


def test_main_grid_plans_every_cell_and_counts_those_over_the_ceiling(items, panel, book, tmp_path, fake_builder):
    """The main grid planned with the fake builder: the B readers alone get M0@full; a memory too long for the C
    ceiling (the fake M7 is 40k tokens at every budget) is planned, not sent, and counted per stage. The fake's
    zero-LLM memories are the same text at every budget, so each is one request per item (the rest are shared: real
    M2 / M4 memories saturate the same way once everything fits); its M5 is still a dry-run stand-in, so each budget
    is its own request (the built summaries can differ). M6 (per-question compression, its own compressor and ledger
    in procedures.memory.compressor) is left to that module's tests."""
    from procedures.matrix.grids import BUDGETS, GRIDS
    grid = GRIDS["main"]
    conditions = [parse_condition(c) for c in grid.conditions if not c.startswith("M6@")]
    report = run(items, panel, book, tmp_path, fake_builder({"M7": 40_000, "M0": 41_000}), None,
                 readers=list(grid.readers), conditions=conditions, builder_id=grid.builder, dry_run=True,
                 cap_usd=100.0)
    est = load(tmp_path, report, "estimate")["readers"]
    n = len(items)
    for pid in grid.readers:
        e = est[pid]
        if panel.get(pid).tier == "B":   # B0, M0, M2 / M4 / M7 once each, M5 at each of the three budgets
            assert (e["cells"], e["infeasible"], e["infeasible_by_stage"]) == (14 * n, 0, {}), pid
            assert (e["pending"], e["shared_requests"]) == (8 * n, 6 * n), pid
        else:                            # B0, M2 / M4 once each, M5 at each budget; M7 never
            assert (e["cells"], e["infeasible"]) == (13 * n, 3 * n), pid
            assert e["infeasible_by_stage"] == {f"read/M7@{b}/{grid.builder}": n for b in BUDGETS}
            assert (e["pending"], e["shared_requests"]) == (6 * n, 4 * n), pid


def test_identical_requests_across_budgets_are_sent_once_and_scored_per_cell(items, panel, book, tmp_path,
                                                                             fake_builder, fake_reader, gold_answer):
    """M2 at 4k and 8k with the same memory text is one request: one call, two rows (the second marked
    shared_request). The same memory at a 32k budget is over the C ceiling: it stays not sent, not borrowing the
    answer of its feasible twin."""
    reader = fake_reader(items, gold_answer)
    conditions = [Condition("M2", 4000), Condition("M2", 8000), Condition("M2", 32000)]
    report = run(items, panel, book, tmp_path, fake_builder(), reader, dry_run=False, conditions=conditions)
    n = len(items)
    assert len(reader.calls) == n
    rows = [json.loads(x) for x in (tmp_path / report["run_id"] / report["outputs"]["rows"])
            .read_text(encoding="utf-8").splitlines()]
    by_stage = Counter((r["stage"], r["outcome"], r["provenance"]["shared_request"]) for r in rows)
    assert by_stage == {("read/M2@4000", "observed", False): n, ("read/M2@8000", "observed", True): n,
                        ("read/M2@32000", "not_sent_infeasible", False): n}
    assert all(r["capability"]["correct"] for r in rows if r["outcome"] == "observed")
    summary = json.loads((tmp_path / report["run_id"] / report["outputs"]["summary"])
                         .read_text(encoding="utf-8"))["conditions"]
    assert [summary[f"{READER}|read/M2@{b}"]["shared_requests"] for b in (4000, 8000, 32000)] == [0, n, 0]
    assert summary[f"{READER}|read/M2@32000"]["outcomes"] == {"not_sent_infeasible": n}


def test_an_estimate_never_merges_two_planned_memories_on_their_stand_in_text(panel):
    """Two budgets of an LLM-built memory not yet built are two requests even when their stand-ins read the same (the
    built texts can differ); built or zero-LLM memories with identical requests are one."""
    from dataclasses import replace
    from procedures.readers import Cell
    from procedures.readers.cells import ReaderCell
    from procedures.matrix.run_matrix import distinct_requests
    pp = panel.get(READER)
    cell = Cell(item_id="i", stage="read/M7@4000/b", messages=({"role": "user", "content": "same text"},),
                generation={"max_output_tokens": 64})
    base = ReaderCell(cell=cell, reader_profile_id=READER, item_id="i", form="value", representation="M7",
                      budget_tokens=4000, builder_profile_id="b", memory_cache_key="k4", memory_tokens=2,
                      build_tokens=0, revisions_used=(), memory_notes="", context_ceiling=None, feasible=True)
    other = replace(base, budget_tokens=8000, memory_cache_key="k8", cell=replace(cell, stage="read/M7@8000/b"))
    assert distinct_requests([base, other], pp) == [base]
    planned = [replace(base, memory_planned=True), replace(other, memory_planned=True)]
    assert distinct_requests(planned, pp) == planned
    assert distinct_requests([planned[0], replace(planned[0], cell=replace(cell, stage="x"))], pp) == [planned[0]]


def test_dry_run_stand_ins_of_llm_built_memories_name_their_budget(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "memory"))    # the memory tests' long corpus
    from test_memory import long_item, write_long_corpus
    from procedures.memory import build_memory
    index = write_long_corpus(tmp_path)
    it = long_item()
    for rep, kw in (("M5", {"builder_profile": "B1_dsv41flash_hf_deepinfra"}),
                    ("M6", {"compressor_profile": READER}), ("M7", {"builder_profile": "B1_dsv41flash_hf_deepinfra"})):
        texts = {b: build_memory(it, rep, b, index, **kw) for b in (4_000, 8_000, 16_000)}
        assert all(m.build["mode"] == "dry_run" for m in texts.values()), rep
        assert len({m.text for m in texts.values()}) == 3, rep


def test_cli_k1_slice_needs_repeats(tmp_path, capsys):
    """scripts/run_matrix.py --k1-slice re-reads the 4k slice: without --repeats > 1 it stops before planning."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "run_matrix_cli", Path(__file__).resolve().parents[2] / "scripts" / "run_matrix.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    with pytest.raises(SystemExit) as exc:
        cli.main(["--label", "main", "--k1-slice", "--out-root", str(tmp_path / "runs")])
    assert exc.value.code == 2 and "--k1-slice re-reads the 4k slice" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()
