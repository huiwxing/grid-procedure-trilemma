"""Call-ledger record -> frozen scorer row -> one row of the capability / access / sustainability tables.

The scorer item is procedures.items.frozen.scorer_item (gold typing, lettered closed sets in display order) plus what
the error classes and pair consistency need:

* `counterparts` (the other side's value, with its axis):
  - S2 (branch pairs written by the procedure's owner): the other branch (`other_side`) -> condition.
  - S1 (revision pairs) and L2 (as-of date x branch): the other date's value (`other_side`) -> version.
  - L3 (as-of date x branch x exception) on the current date (`@cur`): the other branch (`other_side`) -> condition,
    or exception when this cell's branch is the exception branch (answering the general rule where the exception
    applies). A numeric question on the current revision has no other date, and its builder leaves `other_side`
    empty: the other branch is then the structured gold (value + unit) of each sibling cell (same unit, same date,
    other branch) in the run's item set, never the other branch's surface text (the frozen `parse_quantity` reads
    "one-hundred fifteen (115%) percent" as 100). With no sibling cell the axis is absent: the cell is scored without
    that counterpart, its row lists the axis under `counterparts_absent`, and a warning is logged.
  - L3 on an as-of date (`@old` / `@new`): the same branch on the other date (`other_side`) -> version, and the other
    branch on the same date (`other_branch_outcome`, a surface string: an option text, not_in_effect, or a quantity
    read with the frozen `parse_quantity`) -> condition / exception as above.
  - S3 (unrevised control sentences): none. A counterpart equal to the gold is left out.
  A closed-set counterpart must be one of the displayed options. On a not_in_effect cell of a closed-set question
  the gold also carries the displayed `options` and `lettered` (ignored by the not_in_effect judge), so that a letter
  naming the other date's option is classified E-ver like its text.
* `pair`: S1 / S2 / L2 sides, keyed by pair id and item type (an S2 pair has a side form and a value form);
  L3 branch and date consistency use the frozen `branch_pair_consistency` / `date_pair_consistency`.
* `docs`: the item's memory scope (`m0_paths`, plus the evidence file) for the E-other in-document split.

The run scorer passes the run's whole item set to `score_entries(items=...)`, so that a sibling cell is found whether
or not its own record has been answered.

Rows of records that never got an answer (`no_response`, `harness_error`) are scored with raw None (unparseable) and
keep their `outcome`; summaries read capability only from answered records.

An answered record whose content is blank while its separate reasoning field (`reasoning_text`, i.e. the provider's
reasoning_content) is not carries `reasoning_only` in its row, and the reader x condition summary counts it. For a
reader whose panel entry sets `answer_from_reasoning_field` (the Qwen3 profiles served by nscale, whose server can
return the whole answer in reasoning_content), such a record is scored on its reasoning_content when that parses as
exactly one JSON object (procedures.readers.parse.answer_from_reasoning), and the row and summary count
`answer_from_reasoning_field`. Every other blank-content record, and every such record of any other reader, is scored
on the content alone: unparseable / E-parse.

Repeats: capability, pair and L3 consistency summaries read repeat 0 only; `repeat_agreement` compares, item by item,
the repeats 0..k-1 of the repeated subset.
"""
from __future__ import annotations

from collections import Counter
import logging

from procedures.items.frozen import NONE_OF_THESE, branch_pair_consistency, date_pair_consistency
from procedures.items.frozen import scorer_item as frozen_scorer_item
from procedures.scoring import DocIndex, arbitration_queue, pair_consistency, score_rows, summarize
from procedures.scoring.normalize import parse_quantity

from procedures.readers.parse import answer_from_reasoning, chosen_option, record_answer

log = logging.getLogger(__name__)

NIE = "not_in_effect"
PAIRED_FRAMES = ("S1", "S2", "L2")
VERSION_FRAMES = ("S1", "L2")
SUBSET_KEYS = ("n", "correct", "acc", "chance_expected", "acc_chance_corrected", "abstained", "unparseable",
               "ambiguous", "truncated", "error_classes")


def _squash(text: str) -> str:
    return " ".join(text.split())


def _branch_axis(item: dict) -> str:
    return "exception" if item.get("branch_role") == "exception" else "condition"


def _date(item: dict) -> str:
    return item["cell"].split("@")[1]


def _has_other_side(item: dict) -> bool:
    side = item.get("other_side")
    return isinstance(side, dict) and "value" in side


def unit_cells(items) -> dict[tuple[str, str], list[dict]]:
    """(unit_id, date side) -> the L3 cells of that unit on that date, each item id once: the sibling pool."""
    out: dict[tuple[str, str], list[dict]] = {}
    seen: set[str] = set()
    for it in items:
        if it["frame"] == "L3" and it["item_id"] not in seen:
            seen.add(it["item_id"])
            out.setdefault((it["unit_id"], _date(it)), []).append(it)
    return out


def siblings_of(item: dict, pool: dict) -> list[dict]:
    """The cells of an L3 item's unit on the same date with another branch (`pool` = `unit_cells`)."""
    if item["frame"] != "L3":
        return []
    return [c for c in pool.get((item["unit_id"], _date(item)), []) if c["branch"] != item["branch"]]


def absent_axes(item: dict, siblings) -> list[str]:
    """Counterpart axes an item cannot carry: an L3 current-date cell with neither `other_side` nor a sibling cell."""
    if item["frame"] == "L3" and _date(item) == "cur" and not _has_other_side(item) and not siblings:
        return [_branch_axis(item)]
    return []


def _counterpart(item: dict, gold: dict, value, unit, axis: str) -> dict | None:
    if value == NIE:
        return None if gold["gtype"] == "not_in_effect" else {"value": NIE, "gtype": "not_in_effect", "axes": [axis]}
    if gold.get("options"):
        text = _squash(value)
        if text not in gold["options"]:
            raise ValueError(f"{item['item_id']}: counterpart {text!r} is not a displayed option")
        return None if text == gold.get("value") else {"value": text, "gtype": "closed_set", "axes": [axis]}
    if item["answer_type"] == "numeric":
        if gold["gtype"] == "numeric" and (value, unit or None) == (gold["value"], gold.get("unit") or None):
            return None
        return {"value": value, "unit": unit or "", "gtype": "numeric", "axes": [axis]}
    return None if value == gold.get("value") else {"value": value, "gtype": "categorical", "axes": [axis]}


def _surface_counterpart(item: dict, gold: dict, surface: str, axis: str) -> dict | None:
    """The other branch's outcome as the L3 item states it (surface text)."""
    if surface == NIE or gold.get("options") or item["answer_type"] != "numeric":
        return _counterpart(item, gold, surface, None, axis)
    q = parse_quantity(surface)
    if q is None or q.number is None:
        raise ValueError(f"{item['item_id']}: other-branch outcome {surface!r} is not a quantity")
    return _counterpart(item, gold, str(q.number), q.unit_text, axis)


def _other_side(item: dict) -> dict:
    side = item.get("other_side")
    if not isinstance(side, dict) or "value" not in side:
        raise ValueError(f"{item['item_id']}: {item['frame']} item without other_side")
    return side


def counterparts(item: dict, gold: dict, siblings=()) -> list[dict]:
    """`siblings` (`siblings_of`) is read only for an L3 current-date cell whose `other_side` is empty; without any,
    that cell gets no branch counterpart (`absent_axes`)."""
    frame = item["frame"]
    if frame == "S3":
        return []
    if frame in VERSION_FRAMES or frame == "S2":
        side = _other_side(item)
        found = [_counterpart(item, gold, side["value"], side.get("unit"),
                              "version" if frame in VERSION_FRAMES else "condition")]
    elif frame == "L3":
        if _date(item) == "cur":
            others = [item["other_side"]] if _has_other_side(item) else [s["gold"] for s in siblings]
            found = [_counterpart(item, gold, o["value"], o.get("unit"), _branch_axis(item)) for o in others]
        else:
            side = _other_side(item)
            found = [_counterpart(item, gold, side["value"], side.get("unit"), "version")]
            if item.get("other_branch_outcome") is not None:
                found.append(_surface_counterpart(item, gold, item["other_branch_outcome"], _branch_axis(item)))
    else:
        raise ValueError(f"{item['item_id']}: no counterpart rule for frame {frame!r}")
    out: list[dict] = []
    for c in found:
        if c is not None and c not in out:
            out.append(c)
    return out


def pair_of(item: dict) -> dict | None:
    if item["frame"] in PAIRED_FRAMES and item.get("pair_id") and item.get("side"):
        return {"pair_id": f"{item['pair_id']}|{item['type']}", "side": item["side"], "n_sides": 2}
    return None


def scorer_item(item: dict, siblings=()) -> dict:
    base = frozen_scorer_item(item)
    gold = dict(base["gold"])
    if gold["gtype"] == "not_in_effect" and item.get("options"):
        gold.update(options=[_squash(o) for o in item["options"]], lettered=True)
    out = {**base, "gold": gold, "docs": sorted({*(item.get("m0_paths") or []), item["evidence_path"]})}
    found = counterparts(item, gold, siblings)
    if found:
        out["counterparts"] = found
    pair = pair_of(item)
    if pair:
        out["pair"] = pair
    return out


def score_entries(entries: list[dict], text_of, decisions: dict | None = None,
                  items: list[dict] | None = None, answer_from_reasoning_field: bool = False) -> list[dict]:
    """entries: {"item": frame item, "record": ledger record}. Returns frozen-scorer rows in the same order, each with
    the frame item under "frame_item", the counterpart axes it cannot carry under "counterparts_absent", and whether
    the answer was read from the reasoning field under "answer_from_reasoning_field"; `text_of(path)` = text of a
    memory-scope document (E-other split); `decisions` = blind arbitration decisions (arb_id -> bool, the frozen
    scorer's arbitration input, N11); `items` = the item set in which an L3 cell looks for its sibling cells
    (default: the entries' items); `answer_from_reasoning_field` = the reader's panel flag (module docstring)."""
    pool = unit_cells(items if items is not None else [e["item"] for e in entries])
    rows, absent = [], {}
    for e in entries:
        item = e["item"]
        siblings = siblings_of(item, pool)
        axes = absent_axes(item, siblings)
        if axes:
            absent[item["item_id"]] = axes
        raw, finish = record_answer(e["record"])
        mapped = answer_from_reasoning(e["record"]) if answer_from_reasoning_field else None
        rows.append({"item": scorer_item(item, siblings), "raw": raw if mapped is None else mapped,
                     "finish_reason": finish, "frame_item": item, "outcome": e["record"].get("outcome"),
                     "counterparts_absent": axes, "answer_from_reasoning_field": mapped is not None})
    if absent:
        log.warning("%d L3 current-date cell(s) without other_side and without a sibling cell in the item set are "
                    "scored without that counterpart: %s", len(absent), sorted(absent.items()))
    paths = sorted({d for r in rows for d in r["item"]["docs"]})
    return score_rows(rows, {p: DocIndex.from_text(text_of(p)) for p in paths}, decisions)


def reasoning_only(record: dict) -> bool:
    """An answered record with blank content and non-blank reasoning_content (scored E-parse on the content)."""
    return (record.get("outcome") == "observed" and not (record.get("text") or "").strip()
            and bool((record.get("reasoning_text") or "").strip()))


def c_on_nie(scored: dict) -> bool:
    """A not_in_effect cell of a closed-set question answered with the "none of these" option."""
    gold = scored["item"]["gold"]
    if gold["gtype"] != "not_in_effect" or not gold.get("options"):
        return False
    return chosen_option(scored["raw"], scored["finish_reason"], gold["options"]) == NONE_OF_THESE


def ledger_row(run_id: str, item: dict, cell: dict, pp, record: dict | None, scored: dict | None, *,
               shared_request: bool = False) -> dict:
    """One row of the capability / access / sustainability tables for one reader cell (`item` = the frame item,
    `cell` = `ReaderCell.meta()`). `record` None = not sent: `outcome` "not_sent_infeasible" when the cell exceeds its
    reader's local ceiling or route context (access `infeasible_reason`; None on a feasible cell and on a cell meta
    written before the field existed), else "pending"; `scored` is None exactly then. `shared_request`: the record
    answers an earlier cell with the identical request (procedures.matrix.run_matrix.distinct_requests). A compression
    (M6) cell also carries its memory's per-question compression (access `query_memory_usd`, sustainability
    `query_memory_tokens`): the query-side cost of the memory, apart from the one-time `build_tokens` of the summary
    (M5) and records (M7) memories. `memory_empty`: the cell's memory had no text and its prompt carried the explicit
    marker procedures.readers.prompts.EMPTY_MEMORY; the condition summary counts these cells."""
    record = record or {}
    usage = record.get("usage") or {}
    outcome = record.get("outcome") or ("pending" if cell["feasible"] else "not_sent_infeasible")
    capability = None
    if scored is not None:
        v = scored["verdict"]
        capability = {"correct": v["correct"], "status": v["status"], "error_class": scored["error_class"],
                      "other_subclass": scored["other_subclass"], "abstained": v["status"] == "abstained",
                      "truncated": v["status"] == "truncated", "arb_id": scored["arb_id"], "pred": v["pred"],
                      "pred_unit": v["pred_unit"], "matched_option": v["matched_option"],
                      "unit_converted": v["unit_converted"], "note": v["note"],
                      "think_stripped": v["think_stripped"], "n_json_objects": v["n_json_objects"],
                      "pair": scored["item"].get("pair"), "c_on_nie": c_on_nie(scored),
                      "counterparts_absent": scored.get("counterparts_absent") or [],
                      "reasoning_leak": record.get("reasoning_leak"), "reasoning_only": reasoning_only(record),
                      "answer_from_reasoning_field": scored["answer_from_reasoning_field"]}
    return {
        "run_id": run_id, "profile_id": pp.profile_id, "item_id": cell["item_id"], "stage": cell["stage"],
        "repeat_idx": cell["repeat_idx"], "form": cell["form"], "representation": cell["representation"],
        "budget_tokens": cell["budget_tokens"], "builder_profile_id": cell["builder_profile_id"],
        "stratum": item["type"].split("-")[0], "type": item["type"], "family": item.get("family"),
        "cell_kind": item.get("cell_kind"), "outcome": outcome, "memory_empty": cell["memory_empty"],
        "capability": capability,
        "access": {"tier": pp.tier, "access_tier": pp.access_tier, "slot": pp.slot,
                   "local_context_ceiling": cell["context_ceiling"], "route_context": cell.get("route_context"),
                   "planned_input_tokens": cell["planned_input_tokens"], "feasible_at_B": cell["feasible"],
                   "infeasible_reason": cell.get("infeasible_reason"),
                   "usd": record.get("usd"), "usd_counted": record.get("usd_counted"),
                   "requested_model": record.get("requested_model"), "served_model": record.get("served_model"),
                   "observed_provider": record.get("observed_provider"), "query_memory_usd": cell["query_usd"]},
        "sustainability": {"tokens_input_uncached": usage.get("input_uncached"),
                           "tokens_input_cached": usage.get("input_cached"),
                           "tokens_cache_write": usage.get("cache_write"),
                           "tokens_output": usage.get("output_total"),
                           "tokens_output_visible": usage.get("output_visible"),
                           "tokens_reasoning": usage.get("reasoning"),
                           "reasoning_tokens_estimate": usage.get("reasoning_tokens_estimate"),
                           "ref_input_tokens": record.get("ref_input_tokens"),
                           "memory_tokens": cell["memory_tokens"], "build_tokens": cell["build_tokens"],
                           "query_memory_tokens": cell["query_tokens"],
                           "token_method": record.get("token_method")},
        "provenance": {"request_hash": record.get("request_hash"), "memory_cache_key": cell["memory_cache_key"],
                       "revisions_used": cell["revisions_used"], "system_fingerprint": record.get("system_fingerprint"),
                       "server_request_id": record.get("server_request_id"),
                       "finish_reason": record.get("finish_reason"), "n_attempts": len(record.get("attempts") or []),
                       "shared_request": shared_request,
                       "price_book_sha256": (record.get("price") or {}).get("price_book_sha256"),
                       "harness_digest": record.get("harness_digest"), "started_at": record.get("started_at"),
                       "finished_at": record.get("finished_at")},
    }


def row_memory_empty(row: dict) -> bool:
    """A ledger row's `memory_empty`; a row written before empty memories carried the EMPTY_MEMORY marker has no
    field and is read from its memory tokens (0 on a non-closed-book cell: an empty memory, sent then as an empty
    block without the marker)."""
    if "memory_empty" in row:
        return row["memory_empty"]
    return row["representation"] != "B0" and row["sustainability"]["memory_tokens"] == 0


def _subset(rows: list[dict]) -> dict:
    s = summarize(rows)
    return {k: s[k] for k in SUBSET_KEYS}


def condition_summary(scored: list[dict], ledger_rows: list[dict]) -> dict:
    """One reader x condition: outcome counts and costs over all its cells; capability (frozen `summarize`, pair and
    L3 branch / date consistency, per stratum and type) over answered cells only."""
    answered = [r for r in scored if r["outcome"] == "observed"]
    sus = [r["sustainability"] for r in ledger_rows]
    out = {"cells": len(ledger_rows), "outcomes": dict(sorted(Counter(r["outcome"] for r in ledger_rows).items())),
           "usd_counted": sum(r["access"]["usd_counted"] or 0.0 for r in ledger_rows),
           "query_memory_usd": sum(r["access"]["query_memory_usd"] or 0.0 for r in ledger_rows),
           "query_memory_tokens": sum(s["query_memory_tokens"] for s in sus),
           "tokens_input": sum((s["tokens_input_uncached"] or 0) + (s["tokens_input_cached"] or 0)
                               + (s["tokens_cache_write"] or 0) for s in sus),
           "tokens_output": sum(s["tokens_output"] or 0 for s in sus),
           "c_on_nie": sum(bool(r["capability"] and r["capability"]["c_on_nie"]) for r in ledger_rows),
           "counterparts_absent": sum(bool(r["capability"] and r["capability"]["counterparts_absent"])
                                      for r in ledger_rows),
           "reasoning_only": sum(bool(r["capability"] and r["capability"]["reasoning_only"]) for r in ledger_rows),
           "answer_from_reasoning_field": sum(bool(r["capability"] and r["capability"]["answer_from_reasoning_field"])
                                              for r in ledger_rows),
           "shared_requests": sum(r["provenance"]["shared_request"] for r in ledger_rows),
           "memory_empty": sum(r["memory_empty"] for r in ledger_rows)}
    reasons = Counter(r["access"].get("infeasible_reason") for r in ledger_rows
                      if r["outcome"] == "not_sent_infeasible")
    if reasons:
        out["infeasible_by_reason"] = {str(k): v for k, v in sorted(reasons.items(), key=lambda kv: str(kv[0]))}
    if not answered:
        return out
    paired = [r for r in answered if r["item"].get("pair")]
    l3 = [r for r in answered if r["item"]["stratum"] == "L3"]
    out["capability"] = summarize(answered)
    out["by_stratum"] = {s: _subset([r for r in answered if r["item"]["stratum"] == s])
                         for s in sorted({r["item"]["stratum"] for r in answered})}
    out["by_type"] = {t: _subset([r for r in answered if r["frame_item"]["type"] == t])
                      for t in sorted({r["frame_item"]["type"] for r in answered})}
    pc = pair_consistency(paired)
    out["pair_consistency"] = {k: pc[k] for k in ("n_pairs", "n_consistent", "rate", "incomplete", "pending")}
    if l3:
        l3_items = [r["frame_item"] for r in l3]
        out["l3_branch_pair_consistency"] = branch_pair_consistency(l3, l3_items)
        out["l3_date_pair_consistency"] = date_pair_consistency(l3, l3_items)
    return out


def answer_key(capability: dict) -> tuple:
    """What one answer said, for agreement across repeats: the scorer status plus the displayed option it named
    (a letter and the option text are the same answer), else the parsed value and unit."""
    if capability["matched_option"] is not None:
        return capability["status"], "option", capability["matched_option"]
    return capability["status"], "value", capability["pred"], capability["pred_unit"]


def repeat_agreement(ledger_rows: list[dict], k: int) -> dict:
    """Agreement across the repeats 0..k-1 of one reader x condition (`ledger_rows` = its `ledger_row` rows, every
    repeat). The repeated items are those with a row of repeat_idx > 0; an item is complete when all k repeats were
    answered (outcome observed). Over complete items: `correct_agree` = the same verdict on every repeat,
    `answer_agree` = the same `answer_key` on every repeat, `acc_by_repeat` = accuracy of repeat r. `per_item` gives
    each repeated item's verdicts in repeat order (None = not answered) and its two agreement flags (None when
    incomplete). `cells` / `usd_counted` cover the extra repeats (repeat_idx > 0) only."""
    by_item: dict[str, dict[int, dict]] = {}
    for r in ledger_rows:
        by_item.setdefault(r["item_id"], {})[r["repeat_idx"]] = r
    per_item, complete = {}, []
    for iid in sorted(i for i, reps in by_item.items() if any(idx > 0 for idx in reps)):
        reps = by_item[iid]
        if sorted(reps) != list(range(k)):
            raise ValueError(f"{iid}: repeats {sorted(reps)} are not 0..{k - 1}")
        answered = [reps[i] if reps[i]["outcome"] == "observed" else None for i in range(k)]
        verdicts = [None if r is None else r["capability"]["correct"] for r in answered]
        full = all(r is not None for r in answered)
        per_item[iid] = {"correct": verdicts,
                         "correct_agree": len(set(verdicts)) == 1 if full else None,
                         "answer_agree": len({answer_key(r["capability"]) for r in answered}) == 1 if full else None}
        if full:
            complete.append(iid)
    extra = [r for r in ledger_rows if r["repeat_idx"] > 0]
    n = len(complete)
    agree = {name: sum(per_item[i][name] for i in complete) for name in ("correct_agree", "answer_agree")}
    return {"k": k, "items": len(per_item), "complete": n, **agree,
            "correct_agreement": agree["correct_agree"] / n if n else None,
            "answer_agreement": agree["answer_agree"] / n if n else None,
            "acc_by_repeat": [sum(per_item[i]["correct"][r] for i in complete) / n if n else None for r in range(k)],
            "cells": len(extra), "usd_counted": sum(r["access"]["usd_counted"] or 0.0 for r in extra),
            "per_item": per_item}


def blind_arbitration_queue(scored: list[dict]) -> list[dict]:
    """The frozen scorer's blind queue (no system identity) over every answered row of a run."""
    return arbitration_queue([r for r in scored if r["outcome"] == "observed"])


def with_memory_requests(row: dict, requests_file: str, mapped_keys) -> dict:
    """`row` (a `ledger_row`) with provenance `memory_requests`: the name of the run's memory request map
    (procedures.memory.requests: the memory_requests_<sha12>.json a run writes, memory_requests.json in the released
    runs) when the row's memory is one of `mapped_keys` (an LLM-built memory: its build or compression requests are
    listed there), else None. The analysis attributes build and compression cost to a memory by these request
    hashes."""
    listed = row["provenance"]["memory_cache_key"] in mapped_keys
    return {**row, "provenance": {**row["provenance"], "memory_requests": requests_file if listed else None}}
