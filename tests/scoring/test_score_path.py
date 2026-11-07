"""Raw reader output -> frozen scorer: think tags, lettered answers, truncation, missing responses, counterparts and
error classes per frame (including current-revision numeric cells of the L3 frame, as-of date x branch x exception,
whose other branch is the sibling cell's structured gold), and the (C)-on-not_in_effect diagnostic."""
import json
import logging

import pytest

from procedures import ROOT

from procedures.readers.parse import chosen_option, parse_record, record_answer
from procedures.scoring.score_path import c_on_nie, counterparts, score_entries, scorer_item
from procedures.scoring.normalize import parse_quantity
from procedures.scoring.parse import Extraction

FULL_ITEMS = ROOT / "data" / "items" / "questions_with_earlier_cells.jsonl"   # the item file the spec was made from
S2_ECFB = {"b": "2f390ae7", "a": "ba5b881e"}  # unit s2_ecfb287b9a4e357e: b@cur gold 120 %, a@cur gold 115 %


def by_id(items, iid):
    return next(it for it in items if it["item_id"] == iid)


def observed(text, finish="stop"):
    return {"outcome": "observed", "text": text, "finish_reason": finish}


def score(items, text_of, iid, text, finish="stop"):
    return score_entries([{"item": by_id(items, iid), "record": observed(text, finish)}], text_of)[0]


def test_record_answer_only_reads_answered_records():
    assert record_answer(observed("x", "length")) == ("x", "length")
    assert record_answer({"outcome": "no_response", "text": '{"value": "15"}', "finish_reason": "stop"}) == \
        (None, None)
    assert record_answer({"outcome": "harness_error", "text": '{"value": "15"}', "result": {"text": "{}"}}) == \
        (None, None)


def test_think_tags_are_stripped_by_the_frozen_extraction(items, text_of):
    text = '<think>maybe {"value": "2", "unit": "h"}</think>\n{"value": "15", "unit": "min", "evidence_span": "x"}'
    ex = parse_record(observed(text))
    assert isinstance(ex, Extraction) and ex.status == "ok" and ex.think_stripped and ex.answer["value"] == "15"
    row = score(items, text_of, "s2p:value:a", text)
    assert row["verdict"]["correct"] and row["verdict"]["think_stripped"]
    orphan = 'the template opened the block ... </think>{"value": 15, "unit": "minutes"}'
    assert score(items, text_of, "s2p:value:a", orphan)["verdict"]["correct"]
    unclosed = '{"value": "2", "unit": "h"} <think>{"value": "15", "unit": "min"}'
    assert score(items, text_of, "s2p:value:a", unclosed)["error_class"] == "E-cond"


@pytest.mark.parametrize("answer,correct", [("B", True), ("(B)", True), ("option b", True),
                                            ("(B) file a report by day end", True),
                                            ("file a report by day end", True),
                                            ("(A) file a report by day end", False), ("D", False), ("A", False)])
def test_lettered_answers_go_through_the_frozen_closed_set_judge(items, text_of, answer, correct):
    row = score(items, text_of, "s2p:side:b", json.dumps({"value": answer, "unit": None}))
    assert row["verdict"]["correct"] is correct
    if answer == "A":
        assert row["error_class"] == "E-cond"


def test_truncation_and_missing_responses_score_as_parse_errors(items, text_of):
    cut = score(items, text_of, "s2p:value:a", '{"value": "15", "unit": "min"}', finish="length")
    assert cut["verdict"]["status"] == "truncated" and cut["error_class"] == "E-parse"
    missing = score_entries([{"item": by_id(items, "s2p:value:a"),
                              "record": {"outcome": "no_response", "text": None}}], text_of)[0]
    assert missing["verdict"]["status"] == "unparseable" and missing["outcome"] == "no_response"


def test_counterparts_and_axes_per_frame(items):
    def axes(iid):
        it = by_id(items, iid)
        return [(c["axes"][0], c["gtype"], c["value"]) for c in scorer_item(it).get("counterparts", [])]

    assert axes("s2p:side:a") == [("condition", "closed_set", "file a report by day end")]
    assert axes("s2p:value:a") == [("condition", "numeric", "2")]
    assert axes("s1q:value:old") == [("version", "numeric", "59.5")]
    assert axes("s3x:deadline") == []
    assert axes("l3u:a@cur") == [("condition", "closed_set", "file a report by day end")]
    assert axes("l3u:b@cur") == [("exception", "closed_set", "notify the operator within the hour")]
    assert axes("l3v:a@new") == [("version", "not_in_effect", "not_in_effect"),
                                 ("condition", "closed_set", "hold the schedule")]
    assert axes("l3w:b@old") == [("version", "not_in_effect", "not_in_effect"), ("condition", "numeric", "59.6")]
    assert axes("l3w:b@new") == [("version", "numeric", "59.0")]  # the other branch is not_in_effect = the gold
    nie = scorer_item(by_id(items, "l3v:a@old"))
    assert nie["gold"]["gtype"] == "not_in_effect" and nie["gold"]["lettered"] is True
    assert nie["gold"]["options"][2] == "none of these"
    assert scorer_item(by_id(items, "s2p:side:a"))["pair"] == {"pair_id": "s2p|S2-side", "side": "a", "n_sides": 2}
    assert scorer_item(by_id(items, "s2p:value:a"))["pair"]["pair_id"] == "s2p|S2-value"
    assert "pair" not in scorer_item(by_id(items, "l3u:a@cur"))
    bad = {**by_id(items, "s2p:side:a"), "other_side": {"value": "an outcome nobody saw", "unit": None}}
    with pytest.raises(ValueError, match="not a displayed option"):
        counterparts(bad, scorer_item(by_id(items, "s2p:side:b"))["gold"])


def test_error_classes_follow_the_counterpart_axes(items, text_of):
    assert score(items, text_of, "s1q:value:old", '{"value": 59.5, "unit": "Hz"}')["error_class"] == "E-ver"
    assert score(items, text_of, "l3u:b@cur", '{"value": "A"}')["error_class"] == "E-exc"
    assert score(items, text_of, "l3u:a@cur", '{"value": "B"}')["error_class"] == "E-cond"
    other_date = score(items, text_of, "l3v:a@old", '{"value": "A"}')  # the option in effect on the other date
    assert not other_date["verdict"]["correct"] and other_date["error_class"] == "E-ver"
    assert score(items, text_of, "l3w:b@old", '{"value": "59.6", "unit": "Hz"}')["error_class"] == "E-cond"
    assert score(items, text_of, "l3w:b@new", '{"value": 59, "unit": "Hz"}')["error_class"] == "E-ver"
    assert score(items, text_of, "s2p:value:a", '{"value": 15, "unit": "h"}')["error_class"] == "E-unit"


def test_current_revision_numeric_cells_take_the_sibling_structured_gold():
    items = [json.loads(x) for x in FULL_ITEMS.read_text(encoding="utf-8").splitlines() if x.strip()]
    cell = {br: next(it for it in items if it["item_id"].startswith(p)) for br, p in S2_ECFB.items()}
    assert {br: (it["cell"], it["other_side"], it["gold"]["value"], it["gold"]["unit"]) for br, it in cell.items()} == \
        {"b": ("b@cur", None, "120", "%"), "a": ("a@cur", None, "115", "%")}
    assert parse_quantity(cell["b"]["other_branch_outcome"]).number == 100  # why the surface text is never re-read
    for it, sib in ((cell["b"], cell["a"]), (cell["a"], cell["b"])):
        for pool in (items, [it, sib]):  # the run's whole item set, or just the unit

            def scored(answer):
                return score_entries([{"item": it, "record": observed(json.dumps(answer))}], lambda path: "",
                                     items=pool)[0]

            echo = scored({"value": it["gold"]["value"], "unit": "%", "evidence_span": ""})
            assert echo["verdict"]["correct"] and echo["error_class"] is None and echo["counterparts_absent"] == []
            assert echo["item"]["counterparts"] == [{"value": sib["gold"]["value"], "unit": "%", "gtype": "numeric",
                                                     "axes": ["condition"]}]
            other_branch = scored({"value": sib["gold"]["value"], "unit": "%", "evidence_span": ""})
            assert not other_branch["verdict"]["correct"] and other_branch["error_class"] == "E-cond"
            assert scored({"value": 100, "unit": "%"})["error_class"] == "E-other"


def lone_numeric_cell(items, **fields) -> dict:
    return {**{k: items[0][k] for k in ("family", "doc", "evidence_path", "m0_paths")},
            "item_id": "l3z:a@cur", "type": "L3-value", "frame": "L3", "answer_type": "numeric", "unit_id": "l3z",
            "cell": "a@cur", "branch": "a", "branch_role": "parallel", "cell_kind": "parallel",
            "question": "Under PJM Manual 99 as in effect on 2026-03-01, when a single outage is screened, what is the "
                        "threshold for frequency?", "gold": {"value": "59.6", "unit": "Hz"}, "other_side": None,
            "other_branch_outcome": "fifty-nine (59.0 Hz)", **fields}


def test_a_current_revision_cell_without_a_sibling_is_scored_without_the_branch_counterpart(items, text_of, caplog):
    lone = lone_numeric_cell(items)
    with caplog.at_level(logging.WARNING, logger="procedures.scoring.score_path"):
        echo = score_entries([{"item": lone, "record": observed('{"value": "59.6", "unit": "Hz"}')}], text_of,
                             items=items + [lone])[0]
    assert echo["verdict"]["correct"] and "counterparts" not in echo["item"]
    assert echo["counterparts_absent"] == ["condition"] and "l3z:a@cur" in caplog.text
    other = score_entries([{"item": lone, "record": observed('{"value": "59.0", "unit": "Hz"}')}], text_of)[0]
    assert other["error_class"] == "E-other" and other["counterparts_absent"] == ["condition"]
    exception = lone_numeric_cell(items, branch_role="exception", cell_kind="exception")
    assert score_entries([{"item": exception, "record": observed("{}")}], text_of)[0]["counterparts_absent"] == \
        ["exception"]
    with pytest.raises(ValueError, match="without other_side"):  # an as-of cell still needs its other date
        scorer_item(lone_numeric_cell(items, cell="a@old"))
    assert score(items, text_of, "l3u:a@cur", '{"value": "A"}')["counterparts_absent"] == []


def test_c_on_a_not_in_effect_cell_is_wrong_and_flagged(items, text_of):
    row = score(items, text_of, "l3v:a@old", '{"value": "C", "unit": null}')
    assert row["verdict"]["correct"] is False and c_on_nie(row)
    assert c_on_nie(score(items, text_of, "l3v:a@old", '{"value": "none of these"}'))
    assert not c_on_nie(score(items, text_of, "l3v:a@old", '{"value": "not_in_effect"}'))
    assert not c_on_nie(score(items, text_of, "s2p:side:a", '{"value": "C"}'))  # gold is an option, not NIE
    assert chosen_option('{"value": "(C)"}', "stop", ["x y", "z", "none of these"]) == "none of these"


def test_blank_content_with_reasoning_content_is_e_parse_and_flagged(items, text_of):
    """Qwen3-8B closed book on the nscale route: content null, the answer JSON in reasoning_content. Without the
    reader's answer_from_reasoning_field mapping (tests/scoring/test_rescore.py) the content alone is scored (E-parse);
    the row says so and the summary counts it."""
    from procedures.scoring.score_path import condition_summary, ledger_row, reasoning_only
    item = by_id(items, "s2p:value:a")
    record = {"outcome": "observed", "text": None, "finish_reason": "stop",
              "reasoning_text": '\n\n{"value": "15", "unit": "min", "evidence_span": ""}', "reasoning_leak": True}
    assert reasoning_only(record) and not reasoning_only({**record, "text": '{"value": "15"}'})
    assert reasoning_only({**record, "text": " \n"})                     # whitespace content is blank too
    assert not reasoning_only({**record, "reasoning_text": "\n\n"})     # the empty think block of /no_think
    assert not reasoning_only({**record, "outcome": "no_response"})
    (scored,) = score_entries([{"item": item, "record": record}], text_of)
    assert scored["verdict"]["status"] == "unparseable" and scored["error_class"] == "E-parse"

    class PP:
        profile_id, tier, access_tier, slot = "C2_qwen3_8b_hf_nscale_nothink", "C", "C1", "C2"
    cell = {"item_id": item["item_id"], "stage": "read/B0@full", "repeat_idx": 0, "form": "value",
            "representation": "B0", "budget_tokens": None, "builder_profile_id": None, "context_ceiling": 32_768,
            "planned_input_tokens": 300, "feasible": True, "memory_tokens": 0, "build_tokens": 0,
            "memory_cache_key": "k", "revisions_used": [], "query_tokens": 0, "query_usd": None, "memory_empty": False}
    row = ledger_row("r", item, cell, PP, record, scored)
    assert row["capability"]["reasoning_only"] is True and row["access"]["access_tier"] == "C1"
    assert condition_summary([scored], [row])["reasoning_only"] == 1
