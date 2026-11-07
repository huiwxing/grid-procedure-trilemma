"""Helpers over the frozen item sets: the scorer's item view, the consistency checks and the JSONL form of rows.

* `scorer_item`: the frozen scorer's item: closed-set golds are lettered with the options in display order; `docs` =
  the item's evidence file (the revision it is asked about), which the scorer reads to split the E-other error class
  (another value; e.g. `value_not_in_doc`). `NONE_OF_THESE` is the third option of a two-outcome closed set.
* `branch_pair_consistency` / `date_pair_consistency`: both branches of an L3 unit right on the same date, and both
  dates of one branch of an L3 unit right, via the frozen scorer.
* `jsonl_bytes`: the JSONL bytes of a row list (one sorted-key JSON object per line).
"""
from __future__ import annotations

import json

from procedures.scoring.score import pair_consistency

NONE_OF_THESE = "none of these"


def stratum_of(item: dict) -> str:
    return item["type"].split("-")[0]


def scorer_item(item: dict) -> dict:
    g = item["gold"]
    if g["value"] == "not_in_effect":
        gold = {"gtype": "not_in_effect", "value": "not_in_effect"}
    elif item.get("options"):
        gold = {"gtype": "closed_set", "value": " ".join(g["value"].split()),
                "options": [" ".join(o.split()) for o in item["options"]], "lettered": True}
    elif item["answer_type"] == "numeric":
        gold = {"gtype": "numeric", "value": g["value"], "unit": g.get("unit")}
    else:
        gold = {"gtype": "categorical", "value": g["value"]}
    return {"instance_id": item["item_id"], "question": item["question"], "stratum": stratum_of(item), "gold": gold,
            "docs": [item["evidence_path"]]}


# --------------------------------------------------------------------------- rows as JSONL

def jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode("utf-8")


# --------------------------------------------------------------------------- pair consistency of L3 units

def branch_pair_consistency(scored: list[dict], items: list[dict]) -> dict:
    """Both branches of an L3 unit right on the same date (cells `{branch}@{date}`), via the frozen scorer."""
    by_id = {it["item_id"]: it for it in items}
    rows = []
    for r in scored:
        it = by_id[r["item"]["instance_id"]]
        pair = {"pair_id": f"{it['unit_id']}@{it['cell'].split('@')[1]}", "side": it["branch"]}
        rows.append({**r, "item": {**r["item"], "pair": pair}})
    out = pair_consistency(rows)
    return {k: out[k] for k in ("n_pairs", "n_consistent", "rate", "incomplete")}


def date_pair_consistency(scored: list[dict], items: list[dict]) -> dict:
    """Both dates of one branch of an L3 unit right (cells `{branch}@old` / `{branch}@new`; the as-of analogue of
    `branch_pair_consistency`, reported for value cells where a constant not_in_effect answer scores one date of
    every added / deleted clause). Cells on a single current date (`@cur`) have no date pair."""
    by_id = {it["item_id"]: it for it in items}
    rows = []
    for r in scored:
        it = by_id[r["item"]["instance_id"]]
        branch, date = it["cell"].split("@")
        if date != "cur":
            rows.append({**r, "item": {**r["item"], "pair": {"pair_id": f"{it['unit_id']}:{branch}", "side": date}}})
    out = pair_consistency(rows)
    return {k: out[k] for k in ("n_pairs", "n_consistent", "rate", "incomplete")}
