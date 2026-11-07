"""The matrix runner with a mocked transport and a fake memory builder: a gold-echo reader scores 100% through the
real scorer, (C) on a not_in_effect cell is wrong, resume skips answered hashes, the spend cap refuses before any
build or call, dry runs send nothing, cells over the tier ceiling are recorded and not sent."""
import hashlib
import json

import pytest

from procedures.readers import SpendCapExceeded
from procedures.readers.ledger import Ledger, LedgerError, identity_clash, read_ledger
from procedures.readers.executor import ledger_header
from procedures.memory import BuilderLedger
from procedures.matrix.run_matrix import Condition, parse_condition, run_matrix

READERS = ["C1_llama31_8b_hf_deepinfra", "C2_qwen3_8b_hf_nscale_nothink"]
BUILDER = "B1_dsv41flash_hf_deepinfra"
CONDITIONS = [Condition("B0"), Condition("M2", 4000), Condition("M4", 2000), Condition("M5", 4000)]


def run(items, panel, book, tmp_path, builder, reader=None, **kw):
    args = dict(build_memory=builder, panel=panel, label="t", cap_usd=1.0, corpus_index_sha256="0" * 64,
                builder_id=BUILDER, out_root=tmp_path, book=book, transport=reader, sleep=lambda s: None,
                concurrency=1, text_of=lambda p: "59.0 Hz 15 minutes")
    args.update(kw)
    return run_matrix(items, args.pop("conditions", CONDITIONS), args.pop("readers", READERS), {"corpus": "fake"},
                      **args)


def rows_of(report, tmp_path):
    path = tmp_path / report["run_id"] / report["outputs"]["rows"]
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def test_gold_echo_reader_scores_every_cell_through_the_real_scorer(items, panel, book, tmp_path, fake_builder,
                                                                    fake_reader, gold_answer):
    builder = fake_builder()
    reader = fake_reader(items, gold_answer)
    report = run(items, panel, book, tmp_path, builder, reader, dry_run=False)
    rows = rows_of(report, tmp_path)
    assert len(rows) == len(items) * len(CONDITIONS) * len(READERS) == len(reader.calls)
    assert all(r["outcome"] == "observed" and r["capability"]["correct"] for r in rows)
    summary = json.loads((tmp_path / report["run_id"] / report["outputs"]["summary"]).read_text(encoding="utf-8"))
    assert len(summary["conditions"]) == len(CONDITIONS) * len(READERS)
    for s in summary["conditions"].values():
        assert s["capability"]["acc"] == 1.0 and s["pair_consistency"]["rate"] == 1.0
        assert s["l3_branch_pair_consistency"]["rate"] == 1.0 and s["c_on_nie"] == 0
    m5 = [c for c in builder.calls if c["representation"] == "M5"]
    assert {c["ledger"] is None for c in m5} == {True, False}  # planned dry, then built live
    live_ledgers = {c["ledger"] for c in m5 if c["ledger"]}
    assert all(isinstance(led, BuilderLedger) for led in live_ledgers)
    assert {(led.path.name, 0 < led.cap_usd < 1.0, led.transport is reader) for led in live_ledgers} ==\
        {(f"build__{BUILDER}.jsonl", True, True)}  # the cap left after the pending reads
    live_m5 = [p for p in reader.calls if f"summary written by {BUILDER}" in p["messages"][-1]["content"]]
    assert len(live_m5) == len(items) * len(READERS)  # readers saw the live-built summary, not the planned stand-in
    manifest = json.loads((tmp_path / report["run_id"] / "manifest.json").read_text(encoding="utf-8"))
    for key in ("items_sha256", "panel_sha256", "price_book_sha256", "scorer_sha256", "memory_cache_keys",
                "templates_sha256", "corpus_index_sha256"):
        assert manifest[key]
    assert len(manifest["memory_cache_keys"]) == len(items) * len(CONDITIONS)
    m5_row = next(r for r in rows if r["representation"] == "M5")
    assert m5_row["builder_profile_id"] == BUILDER and m5_row["sustainability"]["build_tokens"] == 4000
    assert m5_row["access"]["tier"] == "C" and m5_row["access"]["feasible_at_B"] is True
    assert m5_row["sustainability"]["tokens_input_uncached"] == 400 and m5_row["access"]["usd"] > 0
    assert m5_row["provenance"]["request_hash"] and m5_row["provenance"]["memory_cache_key"]


def test_c_answered_on_a_not_in_effect_cell_is_scored_wrong(items, panel, book, tmp_path, fake_builder, fake_reader):
    reader = fake_reader(items, lambda it: {"value": "C", "unit": None, "evidence_span": ""})
    report = run(items, panel, book, tmp_path, fake_builder(), reader, dry_run=False, readers=READERS[:1],
                 conditions=[Condition("M2", 4000)], builder_id=None)
    rows = {r["item_id"]: r for r in rows_of(report, tmp_path)}
    nie = rows["l3v:a@old"]["capability"]
    assert nie["correct"] is False and nie["c_on_nie"] is True
    assert rows["l3u:a@cur"]["capability"]["c_on_nie"] is False
    summary = json.loads((tmp_path / report["run_id"] / report["outputs"]["summary"]).read_text(encoding="utf-8"))
    assert summary["conditions"][f"{READERS[0]}|read/M2@4000"]["c_on_nie"] == 1


def test_resume_skips_answered_request_hashes(items, panel, book, tmp_path, fake_builder, fake_reader, gold_answer):
    first = fake_reader(items, gold_answer)
    report = run(items, panel, book, tmp_path, fake_builder(), first, dry_run=False)
    again = fake_reader(items, gold_answer)
    resumed = run(items, panel, book, tmp_path, fake_builder(), again, dry_run=False)
    assert resumed["run_id"] == report["run_id"] and again.calls == []
    assert all(v["sent"] == 0 and v["skipped_done"] == len(items) * len(CONDITIONS) for v in resumed["sent"].values())
    assert resumed["outputs"] == report["outputs"]  # same ledgers -> same content-addressed outputs
    estimate = json.loads((tmp_path / report["run_id"] / resumed["estimate"]).read_text(encoding="utf-8"))
    assert estimate["spent_usd"] > 0
    for e in estimate["readers"].values():  # only the M5 cells, planned on dry-run stand-ins, still count as pending
        assert e["pending"] == len(items) and set(e["by_stage_high"]) == {f"read/M5@4000/{BUILDER}"}
    for pid in READERS:
        _, records = read_ledger(tmp_path / report["run_id"] / f"read__{pid}.jsonl")
        assert len(records) == len(items) * len(CONDITIONS)
    modes = [json.loads(x)["outcome"] for x in
             (tmp_path / report["run_id"] / "invocations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert modes == ["live", "live"]


def test_spend_cap_refuses_before_building_or_sending(items, panel, book, tmp_path, fake_builder, fake_reader,
                                                      gold_answer):
    builder, reader = fake_builder(), fake_reader(items, gold_answer)
    with pytest.raises(SpendCapExceeded, match="nothing was built or sent"):
        run(items, panel, book, tmp_path, builder, reader, dry_run=False, cap_usd=1e-6)
    assert reader.calls == [] and all(c["ledger"] is None for c in builder.calls)
    (run_dir,) = tmp_path.iterdir()
    assert not list(run_dir.glob("*__*.jsonl"))
    assert json.loads((run_dir / "invocations.jsonl").read_text(encoding="utf-8"))["outcome"] == "refused"


def test_pending_builds_count_toward_the_cap(items, panel, book, tmp_path, fake_builder, fake_reader, gold_answer):
    dry = run(items, panel, book, tmp_path, fake_builder(), None, dry_run=True)
    assert dry["build_usd_high"] == pytest.approx(2 * 0.01)  # two distinct summaries, each priced once
    reader = fake_reader(items, gold_answer)
    with pytest.raises(SpendCapExceeded, match="pending builds"):  # the reads fit, reads + builds do not
        run(items, panel, book, tmp_path, fake_builder(), reader, dry_run=False,
            cap_usd=dry["pending_usd_high"] + 0.015)
    assert dry["build_usd_high_possible"] == pytest.approx(2 * 0.01)   # each could come back truncated: level 2
    with pytest.raises(SpendCapExceeded, match="possible escalations"):  # reads + level 1 fit, + level 2 does not
        run(items, panel, book, tmp_path, fake_builder(), reader, dry_run=False,
            cap_usd=dry["pending_usd_high"] + 0.03)
    assert reader.calls == []


STOP = {"outcome": "observed", "finish_reason": "stop"}
CUT = {"outcome": "observed", "finish_reason": "length"}


def planned(fake_builder, items) -> dict:
    build = fake_builder()
    memories = {(it["item_id"], "M5@4000"): build(it, "M5", 4000, None, builder_profile=BUILDER) for it in items}
    memories[("x", "M2@4000")] = build(items[0], "M2", 4000, None)
    return memories


def test_pending_builds_climb_the_cap_ladder_only_after_a_truncated_answer(fake_builder, items):
    from procedures.matrix.run_matrix import pending_builds
    memories = planned(fake_builder, items)
    a, b = sorted({p for it in items for p in it["m0_paths"]})          # the two revision files: two sub-blocks
    fresh = pending_builds(memories, {})
    assert fresh == {"live": {f"sum:{a}": (1, 0.01), f"sum:{b}": (1, 0.01)},
                     "possible": {f"sum:{a}@2": (2, 0.01), f"sum:{b}@2": (2, 0.01)}, "imports": {}}
    done = {f"req:{a}": STOP, f"req:{b}": CUT}                        # b was cut at level 1: its level 2 is due
    assert pending_builds(memories, done) == {"live": {f"sum:{b}@2": (2, 0.01)}, "possible": {}, "imports": {}}
    assert pending_builds(memories, {**done, f"req:{b}@2": CUT})["live"] == {}                    # no level 3
    retry = {f"req:{p}": {"outcome": "no_response"} for p in (a, b)}
    assert pending_builds(memories, retry) == fresh


def test_pending_builds_import_what_another_run_answered(fake_builder, items):
    from procedures.matrix.run_matrix import pending_builds
    memories = planned(fake_builder, items)
    a, b = sorted({p for it in items for p in it["m0_paths"]})
    source = {f"req:{a}": STOP, f"req:{b}": CUT, "req:unrelated": STOP}
    reuse = pending_builds(memories, {}, source)
    assert reuse == {"live": {f"sum:{b}@2": (2, 0.01)}, "possible": {},
                     "imports": {f"req:{a}": STOP, f"req:{b}": CUT}}     # only requests this run would send
    assert pending_builds(memories, {f"req:{a}": STOP}, source)["imports"] == {f"req:{b}": CUT}   # answered here
    both = pending_builds(memories, {}, {**source, f"req:{b}@2": STOP})
    assert both["live"] == {} and set(both["imports"]) == {f"req:{a}", f"req:{b}", f"req:{b}@2"}
    assert pending_builds(memories, {}, {f"req:{a}": {"outcome": "no_response"}}) == pending_builds(memories, {})


def build_record(hash_: str, finish: str, usd: float = 0.002) -> dict:
    return {"kind": "call", "request_hash": hash_, "item_id": hash_, "stage": "build/M5@84", "outcome": "observed",
            "finish_reason": finish, "text": "- a rule.", "usd": usd, "usd_counted": usd}


def source_run(root, panel, book, records, header=None):
    """A run directory as run_matrix writes one (<label>_<sha12 of its manifest.json>) with a builder ledger."""
    manifest = json.dumps({"schema": "synthetic", "records": len(records)}).encode()
    run_dir = root / f"old_{hashlib.sha256(manifest).hexdigest()[:12]}"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_bytes(manifest)
    ledger = Ledger(run_dir / f"build__{BUILDER}.jsonl", header or ledger_header(panel.get(BUILDER), book),
                    cap_usd=1.0)
    for record in records:
        ledger.append(record)
    return run_dir


def invocations(run_dir) -> list[dict]:
    return [json.loads(x) for x in (run_dir / "invocations.jsonl").read_text(encoding="utf-8").splitlines()]


def test_reuse_build_from_imports_the_answered_builds_into_this_run(items, panel, book, tmp_path, fake_builder):
    a, b = sorted({p for it in items for p in it["m0_paths"]})
    src = source_run(tmp_path / "elsewhere", panel, book, [build_record(f"req:{a}", "stop"),
                                                           build_record(f"req:{b}", "length"),
                                                           build_record("req:unrelated", "stop")])
    dry = run(items, panel, book, tmp_path, fake_builder(), None, dry_run=True, reuse_build_from=src)
    run_dir = tmp_path / dry["run_id"]
    assert dry["run_id"] == run(items, panel, book, tmp_path, fake_builder(), None)["run_id"]   # not identity
    est = json.loads((run_dir / dry["estimate"]).read_text(encoding="utf-8"))["builds"]
    assert est["imports"] == {"from": src.name, "records": 2, "finish_reasons": {"length": 1, "stop": 1}}
    assert (est["pending"], est["pending_by_cap_level"], est["usd_high"], est["escalations_possible"]) == (
        1, {"2": 1}, pytest.approx(0.01), 0)
    assert (dry["imports"], dry["builds_pending"], dry["build_usd_high_possible"]) == (2, 1, 0)
    assert not list(run_dir.glob("build__*"))                                        # a dry run imports nothing
    built = run(items, panel, book, tmp_path, fake_builder(), None, dry_run=False, build_only=True,
                reuse_build_from=src)
    header, records = read_ledger(run_dir / f"build__{BUILDER}.jsonl")
    assert identity_clash(header, ledger_header(panel.get(BUILDER), book)) == {}
    assert sorted(r["request_hash"] for r in records) == [f"req:{a}", f"req:{b}"]
    source = {r["request_hash"]: r for r in read_ledger(src / f"build__{BUILDER}.jsonl")[1]}
    for r in records:                             # the same record, marked, and no spend of this run
        assert {k: v for k, v in r.items() if k not in ("imported_from", "usd_counted", "imported_usd_counted")}\
            == {k: v for k, v in source[r["request_hash"]].items() if k != "usd_counted"}
        assert (r["imported_from"], r["usd_counted"], r["imported_usd_counted"], r["usd"]) == (
            src.name, 0.0, 0.002, 0.002)
    assert built["spent_usd"] == 0.0
    again = run(items, panel, book, tmp_path, fake_builder(), None, dry_run=False, build_only=True,
                reuse_build_from=src)
    assert again["imports"] == 0 and len(read_ledger(run_dir / f"build__{BUILDER}.jsonl")[1]) == 2
    log = [(x["outcome"], x.get("imported_from"), x.get("importable"), x.get("imported")) for x in
           invocations(run_dir)]
    assert log == [("dry_run", src.name, 2, None), ("dry_run", None, None, None), ("built", src.name, None, 2),
                   ("built", src.name, None, 0)]


@pytest.mark.parametrize("field, value", [("price_book_sha256", "0" * 64), ("profile_fingerprint", "0" * 64),
                                          ("user_message_suffix", " /no_think"), ("profile_id", "other")])
def test_reuse_build_from_refuses_a_ledger_of_another_identity(items, panel, book, tmp_path, fake_builder, field,
                                                               value):
    header = {**ledger_header(panel.get(BUILDER), book), field: value}
    src = source_run(tmp_path / "elsewhere", panel, book, [build_record("req:x", "stop")], header=header)
    for dry_run in (True, False):
        with pytest.raises(LedgerError, match=field):
            run(items, panel, book, tmp_path, fake_builder(), None, dry_run=dry_run, build_only=not dry_run,
                reuse_build_from=src)
    (run_dir,) = [d for d in tmp_path.iterdir() if d.name.startswith("t_")]
    assert not list(run_dir.glob("*__*.jsonl")) and [x["outcome"] for x in invocations(run_dir)] == ["refused"] * 2


def test_reuse_build_from_needs_a_run_directory_with_a_build_ledger(items, panel, book, tmp_path, fake_builder):
    src = source_run(tmp_path / "elsewhere", panel, book, [build_record("req:x", "stop")])
    renamed = src.rename(src.parent / "a renamed run")              # a run directory under any name
    dry = run(items, panel, book, tmp_path, fake_builder(), None, dry_run=True, reuse_build_from=renamed)
    assert dry["imports"] == 0
    (tmp_path / "no_run").mkdir()
    with pytest.raises(ValueError, match="not a run directory"):
        run(items, panel, book, tmp_path, fake_builder(), None, reuse_build_from=tmp_path / "no_run")
    back = renamed.rename(src)
    (back / f"build__{BUILDER}.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="no build ledger"):
        run(items, panel, book, tmp_path, fake_builder(), None, reuse_build_from=back)
    with pytest.raises(ValueError, match="reuse_build_from needs"):
        run(items, panel, book, tmp_path, fake_builder(), None, conditions=[Condition("M2", 4000)], builder_id=None,
            reuse_build_from=back)


def test_an_llm_memory_that_was_not_built_live_stops_the_run(items, panel, book, tmp_path, fake_builder, fake_reader,
                                                             gold_answer):
    reader = fake_reader(items, gold_answer)  # e.g. the builder's credential is unset: procedures.memory falls back
    with pytest.raises(RuntimeError, match="not built live"):
        run(items, panel, book, tmp_path, fake_builder(live_m5=False), reader, dry_run=False)
    assert reader.calls == []
    (run_dir,) = tmp_path.iterdir()
    assert json.loads((run_dir / "invocations.jsonl").read_text(encoding="utf-8"))["outcome"] == "stopped"


def test_dry_run_estimates_and_sends_nothing(items, panel, book, tmp_path, fake_builder):
    builder = fake_builder()
    report = run(items, panel, book, tmp_path, builder, None, dry_run=True)
    run_dir = tmp_path / report["run_id"]
    estimate = json.loads((run_dir / report["estimate"]).read_text(encoding="utf-8"))
    assert set(estimate["readers"]) == set(READERS)
    for est in estimate["readers"].values():
        assert est["pending"] == len(items) * len(CONDITIONS) and 0 < est["usd_low"] <= est["usd_high"]
        assert set(est["by_stage_high"]) == {"read/B0@full", "read/M2@4000", "read/M4@2000",
                                             f"read/M5@4000/{BUILDER}"}
    assert not list(run_dir.glob("*__*.jsonl")) and all(c["ledger"] is None for c in builder.calls)


def test_cells_over_the_tier_ceiling_are_recorded_not_sent(items, panel, book, tmp_path, fake_builder, fake_reader,
                                                           gold_answer):
    reader = fake_reader(items, gold_answer)
    conditions = [Condition("M0"), Condition("M2", 4000, readers=(READERS[0],))]
    report = run(items, panel, book, tmp_path, fake_builder({"M0": 40_000}), reader, dry_run=False,
                 conditions=conditions, builder_id=None)
    rows = rows_of(report, tmp_path)
    m0 = [r for r in rows if r["representation"] == "M0"]
    assert len(m0) == 2 * len(items) and all(r["outcome"] == "not_sent_infeasible" for r in m0)
    assert all(r["access"]["feasible_at_B"] is False and r["capability"] is None for r in m0)
    assert {r["stratum"] for r in m0} == {"S1", "S2", "S3", "L3"} and all(r["type"] and r["family"] for r in m0)
    assert len(reader.calls) == len(items)  # only the M2 cells, only for the reader the condition names
    assert {r["profile_id"] for r in rows if r["representation"] == "M2"} == {READERS[0]}


def test_current_revision_numeric_cells_find_their_sibling_in_the_run_item_set(items, panel, book, tmp_path,
                                                                               fake_builder, fake_reader):
    """Two numeric L3 cells on the current revision (no other_side): the sibling of the answered cell is too long
    for the C tier and never sent, yet its structured gold is the answered cell's condition counterpart; a third cell
    without a sibling is scored without one and counted."""
    common = {**{k: items[0][k] for k in ("family", "doc", "evidence_path", "m0_paths")}, "type": "L3-value",
              "frame": "L3", "answer_type": "numeric", "branch_role": "parallel", "cell_kind": "parallel",
              "other_side": None, "other_branch_outcome": "fifty-nine (59.0 Hz)"}
    q = "Under PJM Manual 99 as in effect on 2026-03-01, when {}, what is the threshold for frequency?"
    cells = [
        {**common, "item_id": "l3n:a@cur", "unit_id": "l3n", "cell": "a@cur", "branch": "a",
         "question": q.format("a single outage is screened"), "gold": {"value": "59.0", "unit": "Hz"}},
        {**common, "item_id": "l3n:b@cur", "unit_id": "l3n", "cell": "b@cur", "branch": "b",
         "question": q.format("x " * 80_000 + "a multiple outage is screened"), "gold": {"value": "59.6", "unit": "Hz"}},
        {**common, "item_id": "l3z:a@cur", "unit_id": "l3z", "cell": "a@cur", "branch": "a",
         "question": q.format("the relay is tested"), "gold": {"value": "59.5", "unit": "Hz"}},
    ]
    answers = {"l3n:a@cur": {"value": "59.6", "unit": "Hz"}, "l3z:a@cur": {"value": "59.5", "unit": "Hz"}}
    reader = fake_reader(cells, lambda it: answers[it["item_id"]])
    report = run(cells, panel, book, tmp_path, fake_builder(), reader, dry_run=False, readers=READERS[:1],
                 conditions=[Condition("M2", 4000)], builder_id=None)
    rows = {r["item_id"]: r for r in rows_of(report, tmp_path)}
    assert rows["l3n:b@cur"]["outcome"] == "not_sent_infeasible" and len(reader.calls) == 2
    assert rows["l3n:a@cur"]["capability"]["error_class"] == "E-cond"
    assert rows["l3n:a@cur"]["capability"]["counterparts_absent"] == []
    lone = rows["l3z:a@cur"]["capability"]
    assert lone["correct"] and lone["counterparts_absent"] == ["condition"]
    summary = json.loads((tmp_path / report["run_id"] / report["outputs"]["summary"]).read_text(encoding="utf-8"))
    assert summary["conditions"][f"{READERS[0]}|read/M2@4000"]["counterparts_absent"] == 1


def test_a_scoring_pass_with_nothing_to_arbitrate_writes_no_queue(tmp_path):
    """The rows and the summary are written under their content's name; an empty arbitration queue is no file, and
    the outputs record it as None."""
    from procedures.matrix.run_matrix import write_scored
    out = write_scored(tmp_path, "t_000000000000", [], {}, [])
    assert out["arbitration"] is None
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([out["rows"], out["summary"]])


def test_conditions_parse_and_validate():
    assert parse_condition("M0@full:B1_x,C1_y") == Condition("M0", None, ("B1_x", "C1_y"))
    assert parse_condition("M2@4000") == Condition("M2", 4000) and parse_condition("B0") == Condition("B0")
    with pytest.raises(ValueError, match="takes no budget"):
        Condition("B0", 4000)
    with pytest.raises(ValueError, match="positive int"):
        Condition("M2", 0)


def test_llm_built_conditions_need_exactly_one_builder(items, panel, book, tmp_path, fake_builder):
    with pytest.raises(ValueError, match="builder profile"):
        run(items, panel, book, tmp_path, fake_builder(), None, builder_id=None)
    with pytest.raises(ValueError, match="builder profile"):
        run(items, panel, book, tmp_path, fake_builder(), None, conditions=[Condition("M2", 4000)])


def test_build_only_builds_live_calls_no_reader_and_the_full_run_follows(items, panel, book, tmp_path, fake_builder,
                                                                        fake_reader, gold_answer):
    builder, reader = fake_builder(), fake_reader(items, gold_answer)
    built = run(items, panel, book, tmp_path, builder, reader, dry_run=False, build_only=True)
    assert built["mode"] == "build_only" and reader.calls == [] and "outputs" not in built
    assert any(c["ledger"] is not None for c in builder.calls if c["representation"] == "M5")
    full = run(items, panel, book, tmp_path, fake_builder(), reader, dry_run=False)
    assert full["run_id"] == built["run_id"] and len(reader.calls) == len(items) * len(CONDITIONS) * len(READERS)
    modes = [json.loads(x)["outcome"] for x in
             (tmp_path / built["run_id"] / "invocations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert modes == ["built", "live"]
    with pytest.raises(ValueError, match="build_only needs"):
        run(items, panel, book, tmp_path, fake_builder(), reader, dry_run=False, build_only=True,
            conditions=[Condition("M2", 4000)], builder_id=None)
