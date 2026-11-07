"""The answer read from reasoning_content for the soft-switch readers (a Qwen3 reader that, with thinking switched
off, can return a blank content and the answer JSON in reasoning_content) and the scoring-only rescore of an existing
run (procedures.scoring.rescore): mapping rule edge cases, the live matrix path, and a rescore that joins the call
ledgers by request_hash, applies the mapping and blind arbitration decisions, writes new files next to the old ones,
refuses inputs of another run and warns about files that changed since the run recorded them."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from procedures.memory.requests import map_bytes
from procedures.readers.parse import answer_from_reasoning
from procedures.scoring.rescore import cell_meta, rescore_run
from procedures.matrix.run_matrix import Condition, run_matrix
from procedures.scoring.score_path import condition_summary, ledger_row, score_entries

QWEN, LLAMA = "C2_qwen3_8b_hf_nscale_nothink", "C1_llama31_8b_hf_deepinfra"
ANSWER = '{"value": "15", "unit": "min", "evidence_span": ""}'


def by_id(items, iid):
    return next(it for it in items if it["item_id"] == iid)


def observed(text, reasoning):
    return {"outcome": "observed", "text": text, "finish_reason": "stop", "reasoning_text": reasoning}


def test_only_a_blank_content_with_exactly_one_json_object_in_the_reasoning_field_is_mapped():
    assert answer_from_reasoning(observed(None, "\n\n" + ANSWER + "\n")) == ANSWER
    assert answer_from_reasoning(observed(" \n", ANSWER)) == ANSWER                 # whitespace content is blank
    assert answer_from_reasoning(observed('{"value": "2"}', ANSWER)) is None          # the content answers
    assert answer_from_reasoning(observed(None, ANSWER + " " + ANSWER)) is None      # two objects
    assert answer_from_reasoning(observed(None, "Let me think. " + ANSWER)) is None  # deliberation around it
    assert answer_from_reasoning(observed(None, '["15"]')) is None                   # not an object
    assert answer_from_reasoning(observed(None, "\n\n")) is None                     # the empty /no_think block
    assert answer_from_reasoning({**observed(None, ANSWER), "outcome": "no_response"}) is None


def test_mapping_applies_only_for_a_flagged_reader_and_is_flagged_and_counted(items, text_of, panel):
    item = by_id(items, "s2p:value:a")
    record = {**observed(None, "\n\n" + ANSWER), "reasoning_leak": True}
    plain, = score_entries([{"item": item, "record": record}], text_of)
    mapped, = score_entries([{"item": item, "record": record}], text_of, answer_from_reasoning_field=True)
    assert plain["verdict"]["status"] == "unparseable" and plain["error_class"] == "E-parse"
    assert not plain["answer_from_reasoning_field"]
    assert mapped["verdict"]["correct"] is True and mapped["answer_from_reasoning_field"] is True
    cell = {"item_id": item["item_id"], "stage": "read/B0@full", "repeat_idx": 0, "form": "value",
            "representation": "B0", "budget_tokens": None, "builder_profile_id": None, "context_ceiling": 32_768,
            "planned_input_tokens": 300, "feasible": True, "memory_tokens": 0, "build_tokens": 0,
            "memory_cache_key": "k", "revisions_used": [], "query_tokens": 0, "query_usd": None, "memory_empty": False}
    row = ledger_row("r", item, cell, panel.get(QWEN), record, mapped)
    assert row["capability"]["answer_from_reasoning_field"] and row["capability"]["reasoning_only"]
    summary = condition_summary([mapped], [row])
    assert (summary["answer_from_reasoning_field"], summary["reasoning_only"]) == (1, 1)
    assert summary["capability"]["acc"] == 1.0
    lone, = score_entries([{"item": item, "record": observed(None, "I think " + ANSWER)}], text_of,
                          answer_from_reasoning_field=True)
    assert lone["error_class"] == "E-parse" and not lone["answer_from_reasoning_field"]


class ReasoningReader:
    """Answers the gold in reasoning_content with a null content for the Qwen3 model (as the nscale route returns it),
    in the content for every other model, which drops the unit on the `drop_unit` items (an ambiguous answer)."""

    def __init__(self, items, drop_unit=()):
        self.by_question = {it["question"]: it for it in items}
        self.drop_unit = set(drop_unit)
        self.calls = 0

    def __call__(self, url, headers, payload, timeout):
        self.calls += 1
        item = self.by_question[payload["messages"][-1]["content"].removesuffix(" /no_think")
                                .rsplit("Question: ", 1)[1]]
        qwen = payload["model"].startswith("Qwen/Qwen3-8B")
        gold = item["gold"]
        if gold["value"] == "not_in_effect":
            answer = {"value": "not_in_effect"}
        elif item.get("options"):
            answer = {"value": item["gold_label"], "unit": None, "evidence_span": ""}
        else:
            dropped = not qwen and item["item_id"] in self.drop_unit
            answer = {"value": gold["value"], "unit": None if dropped else gold.get("unit"), "evidence_span": ""}
        text = json.dumps(answer)
        message = ({"role": "assistant", "content": None, "reasoning_content": "\n\n" + text}
                   if qwen else {"role": "assistant", "content": text})
        body = {"id": f"c{self.calls}", "model": payload["model"].split(":")[0],
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 300, "completion_tokens": 20, "total_tokens": 320}}
        return 200, {"x-request-id": f"r{self.calls}"}, json.dumps(body).encode("utf-8")


def unflagged(panel):
    """The panel as it was before the reasoning-field mapping: same file and fingerprints, the Qwen3 mapping off."""
    return replace(panel, profiles={**panel.profiles,
                                    QWEN: replace(panel.get(QWEN), answer_from_reasoning_field=False)})


def small_run(items, panel, book, tmp_path, fake_builder):
    reader = ReasoningReader(items, drop_unit={"s2p:value:a"})
    report = run_matrix(items, [Condition("B0"), Condition("M2", 4000)], [LLAMA, QWEN], {"corpus": "fake"},
                        build_memory=fake_builder(), panel=panel, label="t", cap_usd=1.0,
                        corpus_index_sha256="0" * 64, dry_run=False, out_root=tmp_path, book=book, transport=reader,
                        sleep=lambda s: None, concurrency=1, text_of=lambda p: "59.0 Hz 15 minutes")
    return report, tmp_path / report["run_id"]


def read_rows(run_dir, name):
    return [json.loads(x) for x in (run_dir / name).read_text(encoding="utf-8").splitlines()]


def test_live_matrix_maps_the_flagged_reader_only(items, panel, book, tmp_path, fake_builder):
    report, run_dir = small_run(items, panel, book, tmp_path, fake_builder)
    rows = read_rows(run_dir, report["outputs"]["rows"])
    qwen = [r for r in rows if r["profile_id"] == QWEN]
    assert all(r["capability"]["answer_from_reasoning_field"] and r["capability"]["reasoning_only"] for r in qwen)
    assert not any(r["capability"]["answer_from_reasoning_field"] for r in rows if r["profile_id"] == LLAMA)
    summary = json.loads((run_dir / report["outputs"]["summary"]).read_text(encoding="utf-8"))["conditions"]
    assert summary[f"{QWEN}|read/B0@full"]["answer_from_reasoning_field"] == len(items)


def decisions_file(tmp_path, run_dir, decisions, **over):
    queue = next(run_dir.glob("arbitration_*.jsonl"))
    doc = {"run_id": run_dir.name, "queue_file": str(queue), "queue_sha256": hashlib.sha256(queue.read_bytes())
           .hexdigest(), "n_arbiters": 1, "decisions": decisions, **over}
    path = tmp_path / f"decisions_{len(list(tmp_path.glob('decisions_*')))}.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_rescore_applies_the_mapping_and_the_decisions_and_keeps_the_old_files(items, panel, book, tmp_path,
                                                                              fake_builder, text_of):
    report, run_dir = small_run(items, unflagged(panel), book, tmp_path, fake_builder)
    before = read_rows(run_dir, report["outputs"]["rows"])
    assert all(r["capability"]["error_class"] == "E-parse" for r in before if r["profile_id"] == QWEN)
    queue = [json.loads(x) for x in (run_dir / report["outputs"]["arbitration"]).read_text().splitlines()]
    assert len(queue) == 1 and queue[0]["note"].startswith("missing unit")   # the Llama answers without a unit
    arb = queue[0]["arb_id"]
    old_files = {p.name: p.read_bytes() for p in run_dir.iterdir()}

    out = rescore_run(run_dir, items, panel, decisions_path=decisions_file(tmp_path, run_dir, {arb: True}),
                      text_of=text_of)
    assert all(run_dir.joinpath(n).read_bytes() == b for n, b in old_files.items() if n != "invocations.jsonl")
    rows = read_rows(run_dir, out["outputs"]["rows"])
    assert len(rows) == len(before) and [r["provenance"]["request_hash"] for r in rows] ==\
        [r["provenance"]["request_hash"] for r in before]
    assert [r["memory_empty"] for r in rows] == [r["memory_empty"] for r in before]
    ch = out["changes"]
    n_qwen = sum(r["profile_id"] == QWEN for r in rows)
    ambiguous = sum(r["capability"]["arb_id"] == arb for r in rows)
    assert ch["by_reason"] == {"answer_from_reasoning_field": n_qwen, "arbitration": ambiguous}
    assert ch["cells"] == n_qwen + ambiguous and ch["became_correct"] == n_qwen + ambiguous
    assert out["accuracy"][f"{QWEN}|read/B0@full"]["after"]["acc"] == 1.0
    assert out["accuracy"][f"{QWEN}|read/B0@full"]["before"]["acc"] == 0.0
    assert out["arbitration_queue_after"] == 0 and out["scorer_sha256"]["changed"] == []
    assert out["outputs"]["arbitration"] is None                          # nothing left to arbitrate: no queue file
    assert sorted(p.name for p in run_dir.glob("arbitration_*.jsonl")) == [report["outputs"]["arbitration"]]
    assert all(v["observed_unreferenced"] == 0 for v in out["ledgers"].values())
    record = json.loads((run_dir / out["rescore"]).read_text(encoding="utf-8"))
    assert record["decisions"]["decisions"] == {arb: True}
    assert record["panel"]["answer_from_reasoning_field"] == {LLAMA: False, QWEN: True}
    log = [json.loads(x) for x in (run_dir / "invocations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert (log[-1]["mode"], log[-1]["rescore"], log[-1]["outputs"]) == ("rescore", out["rescore"], out["outputs"])
    again = rescore_run(run_dir, items, panel, decisions_path=decisions_file(tmp_path, run_dir, {arb: True}),
                        text_of=text_of)
    assert again["outputs"] == out["outputs"]          # deterministic: the same files, written once


def test_rescore_into_an_out_dir_leaves_the_run_directory_as_it_is(items, panel, book, tmp_path, fake_builder,
                                                                    text_of):
    """With out_dir the new rows, summary, arbitration queue, rescore record and invocation line are written there:
    the run directory keeps every byte, its invocations.jsonl included, and the outputs equal an in-place rescore's."""
    report, run_dir = small_run(items, unflagged(panel), book, tmp_path / "runs", fake_builder)
    before = {p.name: p.read_bytes() for p in run_dir.iterdir()}
    out_dir = tmp_path / "rescored"
    out = rescore_run(run_dir, items, panel, text_of=text_of, out_dir=out_dir)
    assert {p.name: p.read_bytes() for p in run_dir.iterdir()} == before
    assert all((out_dir / name).is_file() for name in (out["rescore"], *out["outputs"].values()))
    log = [json.loads(x) for x in (out_dir / "invocations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(x["mode"], x["rescore"]) for x in log] == [("rescore", out["rescore"])]
    in_place = rescore_run(run_dir, items, panel, text_of=text_of)
    assert (in_place["rescore"], in_place["outputs"]) == (out["rescore"], out["outputs"])
    assert all((run_dir / n).read_bytes() == (out_dir / n).read_bytes() for n in out["outputs"].values())


def test_a_decisions_file_must_name_a_queue_file_of_the_run_directory(items, panel, book, tmp_path,
                                                                      fake_builder, text_of):
    """A decisions file whose queue file is not in the run directory stops the re-scoring."""
    report, run_dir = small_run(items, unflagged(panel), book, tmp_path, fake_builder)
    queue = run_dir / report["outputs"]["arbitration"]
    arb = json.loads(queue.read_text().splitlines()[0])["arb_id"]
    elsewhere = "data/runs/earlier/arbitration_0123456789ab.jsonl"
    path = decisions_file(tmp_path, run_dir, {arb: True}, queue_file=elsewhere)
    with pytest.raises(FileNotFoundError, match="arbitration_0123456789ab.jsonl"):
        rescore_run(run_dir, items, panel, decisions_path=path, text_of=text_of, out_dir=tmp_path / "a")


def test_cell_meta_reads_memory_empty_and_derives_it_for_rows_before_the_field(items, panel, book, tmp_path,
                                                                                fake_builder):
    """A rescored row keeps its memory_empty; a row written before the field is empty when its non-closed-book memory
    had 0 tokens."""
    report, run_dir = small_run(items, unflagged(panel), book, tmp_path, fake_builder)
    rows = read_rows(run_dir, report["outputs"]["rows"])
    b0 = next(r for r in rows if r["representation"] == "B0")
    m2 = next(r for r in rows if r["representation"] != "B0")
    assert cell_meta(m2)["memory_empty"] is False and cell_meta(b0)["memory_empty"] is False
    legacy = [{k: v for k, v in r.items() if k != "memory_empty"} for r in (b0, m2)]
    for r in legacy:
        r["sustainability"] = {**r["sustainability"], "memory_tokens": 0}
    assert [cell_meta(r)["memory_empty"] for r in legacy] == [False, True]
    assert cell_meta({**legacy[1], "memory_empty": False})["memory_empty"] is False


def test_rescore_refuses_inputs_of_another_run_and_warns_about_changed_files(items, panel, book, tmp_path,
                                                                             fake_builder, text_of):
    """Inputs that cannot belong to the run stop it (items its rows name missing, decisions for another run or for
    arb_ids not in the queue, a ledger record that is not the row's); files that changed since the run recorded them
    (the items, the queue, the panel profile) are warned about and the re-scoring goes on."""
    report, run_dir = small_run(items, panel, book, tmp_path, fake_builder)
    arb = json.loads((run_dir / report["outputs"]["arbitration"]).read_text().splitlines()[0])["arb_id"]
    with pytest.raises(ValueError, match="not in the item file"):
        rescore_run(run_dir, items[:-1], panel, text_of=text_of)
    edited = [dict(it, question=it["question"] + " ") if k == 0 else it for k, it in enumerate(items)]
    with pytest.warns(UserWarning, match="differ from the ones"):
        assert rescore_run(run_dir, edited, panel, text_of=text_of, out_dir=tmp_path / "a")["outputs"]
    with pytest.raises(ValueError, match="decisions for run"):
        rescore_run(run_dir, items, panel, decisions_path=decisions_file(tmp_path, run_dir, {arb: True},
                                                                          run_id="t_000000000000"))
    with pytest.warns(UserWarning, match="recorded sha256"):
        rescore_run(run_dir, items, panel, text_of=text_of, out_dir=tmp_path / "b",
                    decisions_path=decisions_file(tmp_path, run_dir, {arb: True}, queue_sha256="0" * 64))
    with pytest.raises(ValueError, match="not in the queue"):
        rescore_run(run_dir, items, panel, decisions_path=decisions_file(tmp_path, run_dir, {"feedfacefeedface": True}))
    other = replace(panel, profiles={**panel.profiles, LLAMA: replace(panel.get(LLAMA), user_message_suffix=" x")})
    with pytest.warns(UserWarning, match="differs from the one"):
        rescore_run(run_dir, items, other, text_of=text_of, out_dir=tmp_path / "c")
    ledger = run_dir / f"read__{LLAMA}.jsonl"
    original = ledger.read_text(encoding="utf-8").splitlines()
    first = json.loads(original[1])
    for tampered in ({"item_id": "s3x:deadline" if first["item_id"] != "s3x:deadline" else "s2p:side:a"},
                     {"outcome": "no_response"}):
        ledger.write_text("\n".join([original[0], json.dumps({**first, **tampered}), *original[2:]]) + "\n",
                          encoding="utf-8")
        with pytest.raises(ValueError, match="not the one the row scored"):
            rescore_run(run_dir, items, panel, text_of=text_of)


def test_scoring_and_rescoring_attribute_rows_to_one_memory_request_map(items, panel, book, tmp_path, fake_builder,
                                                                        fake_reader, gold_answer, text_of):
    """The live scoring writes memory_requests_<sha12>.json from the plan and every row of an LLM-built memory (M5, M6,
    M7) names it (None on the others); a rescore names the run's map on its rows: the one in the run directory when
    its keys are the run's LLM-built memories, else one re-planned (which needs the corpus index and the run's cache
    keys)."""
    builder = "B1_dsv41flash_hf_deepinfra"
    report = run_matrix(items, [Condition("B0"), Condition("M5", 4000)], [LLAMA], {"corpus": "fake"},
                        build_memory=fake_builder(), panel=panel, label="attr", cap_usd=1.0,
                        corpus_index_sha256="0" * 64, builder_id=builder, dry_run=False, out_root=tmp_path, book=book,
                        transport=fake_reader(items, gold_answer), sleep=lambda s: None, concurrency=1,
                        text_of=lambda p: "59.0 Hz 15 minutes")
    run_dir = tmp_path / report["run_id"]
    name = report["outputs"]["memory_requests"]
    mapping = json.loads((run_dir / name).read_text(encoding="utf-8"))
    rows = read_rows(run_dir, report["outputs"]["rows"])
    assert {r["provenance"]["memory_requests"] for r in rows if r["representation"] == "M5"} == {name}
    assert {r["provenance"]["memory_requests"] for r in rows if r["representation"] == "B0"} == {None}
    m5_keys = {r["provenance"]["memory_cache_key"] for r in rows if r["representation"] == "M5"}
    assert m5_keys == set(mapping["memories"])
    assert all(len(e["requests"]) == 2 * len(e["revisions"]) for e in mapping["memories"].values())  # every level

    out = rescore_run(run_dir, items, panel, text_of=text_of)                  # the map the run directory holds
    record = json.loads((run_dir / out["rescore"]).read_text(encoding="utf-8"))
    assert out["outputs"]["memory_requests"] == name and record["memory_requests"]["from"] == "run directory"
    assert [r["provenance"] for r in read_rows(run_dir, out["outputs"]["rows"])] == [r["provenance"] for r in rows]

    (run_dir / name).unlink()                                                   # a run scored before the map existed
    with pytest.raises(ValueError, match="needs the corpus index"):
        rescore_run(run_dir, items, panel, text_of=text_of)
    other = fake_builder()

    def shifted(item, rep, budget, index, **kw):              # a plan that no longer gives the run's cache keys
        return replace(other(item, rep, budget, index, **kw), cache_key="0" * 64)
    with pytest.raises(ValueError, match="differ from the run's cache keys"):
        rescore_run(run_dir, items, panel, text_of=text_of, corpus_index={"corpus": "fake"}, build_memory=shifted)
    out = rescore_run(run_dir, items, panel, text_of=text_of, corpus_index={"corpus": "fake"},
                      build_memory=fake_builder())
    record = json.loads((run_dir / out["rescore"]).read_text(encoding="utf-8"))
    assert out["outputs"]["memory_requests"] == name and record["memory_requests"]["from"] == "re-plan"
    assert [r["provenance"] for r in read_rows(run_dir, out["outputs"]["rows"])] == [r["provenance"] for r in rows]

    wrong = {**mapping, "memories": dict(list(mapping["memories"].items())[1:])}  # a map of other memories
    data = map_bytes(wrong)
    (run_dir / name).unlink()
    (run_dir / f"memory_requests_{hashlib.sha256(data).hexdigest()[:12]}.json").write_bytes(data)
    with pytest.raises(ValueError, match="not the run's LLM-built memories"):
        rescore_run(run_dir, items, panel, text_of=text_of)
