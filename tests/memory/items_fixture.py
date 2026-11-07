"""Shared test items over two synthetic documents: an S1 revision pair (the old and the new value of one clause,
two items) and one side of an S2 branch pair (a branch pair written by the procedure's owner)."""
from __future__ import annotations

from test_memory import DOC, DOC2
from procedures.scoring import render_lettered_options

NIE = " If no such requirement applies on that date, answer not_in_effect."
OPTS = ["the operator must shed load", "the operator should hold the schedule", "none of these"]


def make_items() -> list[dict]:
    q = "Under Test Operating Procedure 1 as in effect on {}, what is the ramp limit for the Test unit?" + NIE
    s1 = {"type": "S1-value", "frame": "S1", "answer_type": "numeric", "family": "TEST", "doc": DOC,
          "pair_id": "p1", "m0_paths": ["proj/data/corpus/test/r2.txt", "proj/data/corpus/test/r3.txt"]}
    return [
        {**s1, "item_id": "p1:value:new", "question": q.format("2023-06-01"), "as_of_date": "2023-06-01",
         "revision": "3", "side": "new", "evidence_path": "proj/data/corpus/test/r3.txt",
         "gold": {"value": "30", "unit": "MW"}, "other_side": {"value": "20", "unit": "MW"}},
        {**s1, "item_id": "p1:value:old", "question": q.format("2021-06-01"), "as_of_date": "2021-06-01",
         "revision": "2", "side": "old", "evidence_path": "proj/data/corpus/test/r2.txt",
         "gold": {"value": "20", "unit": "MW"}, "other_side": {"value": "30", "unit": "MW"}},
        {"item_id": "s2p:side:a", "type": "S2-side", "frame": "S2", "answer_type": "closed_set", "family": "TEST2",
         "doc": DOC2, "revision": "1", "effective_date": "2024-01-01", "pair_id": "s2p", "side": "a",
         "question": "Under Test Operating Procedure 2 (Revision 1, effective 2024-01-01), when the frequency falls "
                     f"below the trigger, which of the following applies? {render_lettered_options(OPTS)}.",
         "options": OPTS, "gold": {"value": OPTS[0], "unit": None}, "gold_label": "A",
         "other_side": {"value": OPTS[1], "unit": None}, "evidence_path": "proj/data/corpus/test/op2.txt",
         "m0_paths": ["proj/data/corpus/test/op2.txt"]},
    ]
