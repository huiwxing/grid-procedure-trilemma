"""Frozen scorer: verdicts, closed-set labels, error classes, blind arbitration, summaries.

Everything here is a deterministic function of (raw model text, finish_reason, item,
arbitration decisions, source documents).  The rules are stated here and in normalize.py and
parse.py (numbered N1-N11, procedures.scoring).

Gold types (item["gold"]["gtype"]; inferred for golds that carry none):
  numeric | categorical | closed_set | yes_no | not_in_effect | time_of_day | date
  ("interval" is not scorable: an interval gold is split into one numeric gold per bound at admission, rule N9).
  A closed_set gold with "lettered": true lists its options in display order; the answer
  may then name an option by its letter (A, B, C, ...) instead of, or together with, its text (rule N6').
Statuses: predicted | abstained | unparseable | ambiguous | truncated.  Only "predicted"
rows (and ambiguous rows an arbitrator marked correct in the blind arbitration, rule N11) can be correct.
Error classes, first match wins: E-parse > E-abs > E-ver > E-cond > E-exc > E-unit > E-other,
E-other split into value_in_doc / value_not_in_doc.  E-parse: no usable answer; E-abs: abstained;
E-ver: another revision's value; E-cond: another branch's value; E-exc: the general rule's value on an
exception item; E-unit: the gold number with a wrong unit; E-other: any other value.
"""
from __future__ import annotations

import hashlib
import json
import re
import string
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

from procedures.scoring.normalize import UNDECIDABLE, DocIndex, Unit, lookup_unit, norm_date, norm_label, norm_time, parse_quantity, plural_forms, strip_terminal_punct, to_decimal, unit_equal
from procedures.scoring.parse import extract_answer

GTYPES = ("numeric", "categorical", "closed_set", "yes_no", "not_in_effect", "time_of_day", "date")
STATUSES = ("predicted", "abstained", "unparseable", "ambiguous", "truncated")
ERROR_CLASSES = ("E-parse", "E-abs", "E-ver", "E-cond", "E-exc", "E-unit", "E-other")
OTHER_SUBCLASSES = ("value_in_doc", "value_not_in_doc")
AXIS_CLASSES = (("version", "E-ver"), ("condition", "E-cond"), ("exception", "E-exc"))
NIE_LABEL = "not in effect"
ABSTAIN_LABEL = "insufficient evidence"
ABSTAIN_TOKEN = "insufficient_evidence"
OPTION_LETTERS = string.ascii_uppercase
_LETTER_ONLY = re.compile(r"(?:option\s+)?\(?([a-z])\)?", re.IGNORECASE)
_LETTER_TEXT = re.compile(r"(?:option\s+)?(?:\(([a-z])\)|([a-z])[).:])\s*(\S.*)", re.IGNORECASE | re.DOTALL)


# --------------------------------------------------------------------------- gold

@dataclass(frozen=True)
class GoldSpec:
    gtype: str
    value: object                 # Decimal (numeric) | normalised str | option str (closed_set)
    unit: Unit | None             # numeric only; None = dimensionless count
    forms: frozenset              # categorical / yes_no accepted labels
    options: tuple                # closed_set: ((option, frozenset(forms)), ...) in display order
    chance: Fraction              # guessing rate: 1/2 yes_no, 1/k closed_set, else 0
    lettered: bool = False        # closed_set shown as "(A) .. (B) ..": letters name options


def _infer_gtype(value) -> str:
    if isinstance(value, str):
        return "not_in_effect" if norm_label(value) == NIE_LABEL else "categorical"
    if to_decimal(value) is not None:
        return "numeric"
    raise ValueError(f"cannot infer gtype for gold value {value!r}")


def _label_forms(texts) -> frozenset:
    forms = set()
    for t in texts:
        if not isinstance(t, str) or not norm_label(t):
            raise ValueError(f"label/alias must be a non-empty string, got {t!r}")
        forms |= plural_forms(norm_label(t))
    return frozenset(forms)


def gold_spec(gold: dict) -> GoldSpec:
    """Validate a gold dict and pre-normalise it; malformed gold raises ValueError."""
    value = gold.get("value")
    gtype = gold.get("gtype") or _infer_gtype(value)
    none = Fraction(0)
    if gtype not in GTYPES + ("interval",):
        raise ValueError(f"unknown gtype {gtype!r}")
    if gtype == "interval":
        raise ValueError("interval gold is not scorable: split it into one numeric gold per bound at admission (N9)")
    if gtype == "numeric":
        number = to_decimal(value)
        if number is None:
            raise ValueError(f"numeric gold needs a finite number, got {value!r}")
        unit_text = gold.get("unit")
        unit = None if unit_text in (None, "") else lookup_unit(unit_text)
        if unit_text not in (None, "") and unit is None:
            raise ValueError(f"gold unit {unit_text!r} is not in the frozen unit table")
        return GoldSpec(gtype, number, unit, frozenset(), (), none)
    if gtype == "not_in_effect":
        if not isinstance(value, str) or norm_label(value) != NIE_LABEL:
            raise ValueError(f"not_in_effect gold must be 'not_in_effect', got {value!r}")
        return GoldSpec(gtype, NIE_LABEL, None, frozenset({NIE_LABEL}), (), none)
    if gtype == "categorical":
        forms = _label_forms([value, *(gold.get("accepted_aliases") or [])])
        return GoldSpec(gtype, norm_label(value), None, forms, (), none)
    if gtype == "yes_no":
        label = norm_label(value) if isinstance(value, str) else None
        if label not in ("yes", "no"):
            raise ValueError(f"yes_no gold must be yes/no, got {value!r}")
        return GoldSpec(gtype, label, None, frozenset({label}), (), Fraction(1, 2))
    if gtype == "closed_set":
        return _closed_set_spec(gold)
    canon = (norm_time if gtype == "time_of_day" else norm_date)(value)
    if not isinstance(canon, str):
        raise ValueError(f"{gtype} gold is not a decidable {gtype}: {value!r}")
    return GoldSpec(gtype, canon, None, frozenset(), (), none)


def _closed_set_spec(gold: dict) -> GoldSpec:
    options = gold.get("options")
    if not isinstance(options, list) or len(options) < 2 or len(set(options)) != len(options):
        raise ValueError("closed_set gold needs >= 2 distinct verbatim options")
    if gold.get("value") not in options:
        raise ValueError(f"closed_set gold value {gold.get('value')!r} is not one of its options")
    aliases = gold.get("option_aliases") or {}
    stray = set(aliases) - set(options)
    if stray:
        raise ValueError(f"option_aliases for non-options: {sorted(stray)}")
    lettered = gold.get("lettered", False)
    if not isinstance(lettered, bool):
        raise ValueError(f"closed_set 'lettered' must be true or false, got {lettered!r}")
    if lettered and len(options) > len(OPTION_LETTERS):
        raise ValueError(f"a lettered closed_set has at most {len(OPTION_LETTERS)} options")
    table = tuple((opt, _label_forms([opt, *aliases.get(opt, [])])) for opt in options)
    for i, (opt_i, forms_i) in enumerate(table):
        for opt_j, forms_j in table[i + 1:]:
            if forms_i & forms_j:
                raise ValueError(f"closed_set options {opt_i!r} and {opt_j!r} collide after "
                                 f"normalisation: {sorted(forms_i & forms_j)}")
        if lettered:
            letter = OPTION_LETTERS[i]
            clash = [opt for j, (opt, forms) in enumerate(table) if j != i and letter.lower() in forms]
            if clash:
                raise ValueError(f"closed_set letter {letter} of {opt_i!r} is also the text of "
                                 f"option {clash[0]!r}")
    return GoldSpec("closed_set", gold["value"], None, frozenset(), table,
                    Fraction(1, len(options)), lettered)


# --------------------------------------------------------------------------- verdicts

def _jsonable(x):
    if isinstance(x, Decimal):
        return str(x)
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    return x


def _verdict(status, correct=False, **kw) -> dict:
    v = {"status": status, "correct": bool(correct), "pred": None, "pred_unit": None,
         "unit_converted": False, "matched_option": None, "note": None, "answer": None,
         "n_json_objects": 0, "think_stripped": False, "arbitrated": False}
    v.update(kw)
    return v


def _is_label(value, label: str) -> bool:
    return isinstance(value, str) and norm_label(value) == label


def _unit_claim(embedded: str | None, obj: dict):
    """-> (kind, Unit|None) with kind in known | missing | empty | unknown | conflict."""
    key = obj.get("unit")
    key_text = None if key is None else str(key).strip()
    if embedded:
        e_unit = lookup_unit(embedded)
        if key_text:
            k_unit = lookup_unit(key_text)
            if e_unit is None or k_unit is None:
                return "unknown", None
            return ("known", e_unit) if e_unit == k_unit else ("conflict", None)
        return ("known", e_unit) if e_unit else ("unknown", None)
    if key_text is None:
        return "missing", None
    if key_text == "":
        return "empty", None
    k_unit = lookup_unit(key_text)
    return ("known", k_unit) if k_unit else ("unknown", None)


def _judge_numeric(obj: dict, spec: GoldSpec) -> dict:
    q = parse_quantity(obj["value"])
    if q is None or q.number is None:
        return _verdict("predicted", pred=_jsonable(obj["value"]),
                        note=(q.note if q else None) or "non-numeric on numeric gold")
    pred = str(q.number)
    kind, unit = _unit_claim(q.unit_text, obj)
    if spec.unit is None:  # dimensionless count: unit ignored unless it is a physical unit
        if kind in ("known", "conflict"):
            return _verdict("predicted", pred=pred, pred_unit=unit.symbol if unit else None,
                            note="physical unit on dimensionless gold")
        return _verdict("predicted", q.number == spec.value, pred=pred)
    if kind in ("missing", "unknown", "conflict"):
        return _verdict("ambiguous", pred=pred, note=f"{kind} unit on dimensional gold")
    if kind == "empty":
        return _verdict("predicted", pred=pred, pred_unit="",
                        note="unitless claim on dimensional gold")
    ok, converted = unit_equal(q.number, unit, spec.value, spec.unit)
    return _verdict("predicted", ok, pred=pred, pred_unit=unit.symbol,
                    unit_converted=bool(ok and converted))


def _judge_letter(value: str, spec: GoldSpec) -> dict:
    """N6' for a lettered closed set: "B", "(B)", "B)", "option B", or a letter followed by
    option text ("(B) notify PJM", "B. notify PJM").  Letter and text naming different options
    is wrong (neither reading is taken); a letter with text outside the set goes to arbitration."""
    s = strip_terminal_punct(value.strip())
    m = _LETTER_ONLY.fullmatch(s)
    if m:
        letter, text = m[1], None
    else:
        m = _LETTER_TEXT.fullmatch(s)
        if m is None:
            return _verdict("predicted", pred=norm_label(value), note="outside the closed set")
        letter, text = m[1] or m[2], m[3]
    letter = letter.upper()
    index = OPTION_LETTERS.index(letter)
    if index >= len(spec.options):
        return _verdict("predicted", pred=norm_label(value), note="outside the closed set")
    option, forms = spec.options[index]
    if text is not None and norm_label(text) not in forms:
        if any(norm_label(text) in other for _, other in spec.options):
            return _verdict("predicted", pred=norm_label(value),
                            note=f"option letter {letter} and text disagree")
        return _verdict("ambiguous", pred=norm_label(value),
                        note=f"option letter {letter} with text outside the closed set")
    return _verdict("predicted", option == spec.value, pred=option, matched_option=option,
                    note=f"option letter {letter}")


def _judge(obj: dict, spec: GoldSpec) -> dict:
    """Score a parsed answer object (has "value") against a validated gold."""
    av = obj["value"]
    if spec.gtype == "not_in_effect":
        return _verdict("predicted", _is_label(av, NIE_LABEL), pred=_jsonable(av))
    if spec.gtype == "numeric":
        return _judge_numeric(obj, spec)
    if spec.gtype in ("time_of_day", "date"):
        canon = (norm_time if spec.gtype == "time_of_day" else norm_date)(av)
        if canon is UNDECIDABLE:
            return _verdict("ambiguous", pred=_jsonable(av), note=f"undecidable {spec.gtype}")
        if canon is None:
            return _verdict("predicted", pred=_jsonable(av), note=f"not a {spec.gtype}")
        return _verdict("predicted", canon == spec.value, pred=canon)
    if spec.gtype == "yes_no" and isinstance(av, bool):
        label = "yes" if av else "no"
    elif isinstance(av, str) and norm_label(av):
        label = norm_label(av)
    else:
        return _verdict("predicted", pred=_jsonable(av), note=f"non-string on {spec.gtype} gold")
    if spec.gtype == "closed_set":
        hit = [opt for opt, forms in spec.options if label in forms]  # forms are disjoint
        if not hit and spec.lettered:
            return _judge_letter(av, spec)
        if not hit:
            return _verdict("predicted", pred=label, note="outside the closed set")
        return _verdict("predicted", hit[0] == spec.value, pred=hit[0], matched_option=hit[0])
    return _verdict("predicted", label in spec.forms, pred=label)


def score_answer(raw_text: str | None, gold: dict, finish_reason: str | None = None) -> dict:
    """One answer -> verdict dict.  raw_text None means the call produced no output (N10: scored unparseable)."""
    spec = gold_spec(gold)
    ex = extract_answer(raw_text, finish_reason)
    meta = {"answer": _jsonable(ex.answer), "n_json_objects": ex.n_objects,
            "think_stripped": ex.think_stripped}
    if ex.status == "truncated":
        return _verdict("truncated", note=f"finish_reason={finish_reason}", **meta)
    if ex.status == "empty":
        return _verdict("unparseable", note="empty output", **meta)
    if ex.status == "no_json":
        return _verdict("unparseable", note="no JSON object", **meta)
    if ex.status == "no_value_key":
        return _verdict("ambiguous", note="value key missing", **meta)
    if _is_label(ex.answer["value"], ABSTAIN_LABEL):
        return _verdict("abstained", pred=ABSTAIN_TOKEN, **meta)
    return {**_judge(ex.answer, spec), **meta}


# --------------------------------------------------------------------------- error classes

def _counterpart_specs(item: dict) -> list[tuple[tuple, GoldSpec]]:
    """Counterpart = another side's value (other version / branch / exception).  Its gtype is
    explicit, or not_in_effect for the nie token, or inferred from its own value when the
    item's gold is not_in_effect (deleted/added requirement), else the gold's gtype."""
    gold = item["gold"]
    gold_gtype = gold_spec(gold).gtype
    out = []
    for cp in item.get("counterparts") or []:
        axes = tuple(cp.get("axes") or ())
        if not axes or set(axes) - {a for a, _ in AXIS_CLASSES}:
            raise ValueError(f"{item.get('instance_id')}: counterpart axes must be a non-empty "
                             f"subset of version/condition/exception, got {axes!r}")
        pseudo = {"value": cp.get("value"), "unit": cp.get("unit", ""),
                  "accepted_aliases": cp.get("accepted_aliases") or []}
        if cp.get("gtype"):
            pseudo["gtype"] = cp["gtype"]
        elif _is_label(cp.get("value"), NIE_LABEL):
            pseudo["gtype"] = "not_in_effect"
        elif gold_gtype == "not_in_effect":
            pseudo["gtype"] = _infer_gtype(cp.get("value"))
        else:
            pseudo["gtype"] = gold_gtype
        if pseudo["gtype"] == "closed_set":
            pseudo["options"] = gold.get("options")
            pseudo["option_aliases"] = gold.get("option_aliases") or {}
            pseudo["lettered"] = gold.get("lettered", False)
        out.append((axes, gold_spec(pseudo)))
    return out


def _value_in_docs(av, spec: GoldSpec, indexes: list[DocIndex]) -> bool:
    if isinstance(av, str) and norm_label(av) in (NIE_LABEL, ABSTAIN_LABEL):
        return False
    if spec.gtype in ("date", "time_of_day"):
        canon = (norm_date if spec.gtype == "date" else norm_time)(av)
        if isinstance(canon, str):
            pool = "dates" if spec.gtype == "date" else "times"
            return any(canon in getattr(ix, pool) for ix in indexes)
    q = parse_quantity(av)
    if q is not None and q.number is not None:
        return any(q.number in ix.numbers for ix in indexes)
    if isinstance(av, str):
        return any(ix.contains_label(av) for ix in indexes)
    return False


def classify_error(row: dict, doc_indexes: dict[str, DocIndex]) -> tuple[str | None, str | None]:
    """(error_class, other_subclass) of a scored row; (None, None) when correct."""
    verdict, item = row["verdict"], row["item"]
    if verdict["correct"]:
        return None, None
    if verdict["status"] == "ambiguous" and not verdict["arbitrated"]:
        raise ValueError(f"{item.get('instance_id')}: unresolved ambiguous row (arbitrate first, N11)")
    if verdict["status"] in ("unparseable", "truncated"):
        return "E-parse", None
    if verdict["status"] == "abstained":
        return "E-abs", None
    ex = extract_answer(row.get("raw"), row.get("finish_reason"))
    if ex.answer is None or "value" not in ex.answer:
        return "E-parse", None
    specs = _counterpart_specs(item)
    for axis, cls in AXIS_CLASSES:
        if any(axis in axes and _judge(ex.answer, spec)["correct"] for axes, spec in specs):
            return cls, None
    spec = gold_spec(item["gold"])
    if spec.gtype == "numeric":
        q = parse_quantity(ex.answer["value"])
        if q is not None and q.number == spec.value:
            return "E-unit", None
    doc_ids = item.get("docs") or []
    if not doc_ids:
        raise ValueError(f"{item.get('instance_id')}: item lists no docs; E-other needs them")
    indexes = [doc_indexes[d] for d in doc_ids]
    # a closed-set answer that picked an option (by text or by letter) stands for that option
    value = verdict["matched_option"] if verdict["matched_option"] is not None else ex.answer["value"]
    inside = _value_in_docs(value, spec, indexes)
    return "E-other", "value_in_doc" if inside else "value_not_in_doc"


# --------------------------------------------------------------------------- rows, arbitration

def _judged_answer(verdict: dict) -> dict:
    """What an arbitrator sees and decides on: the parsed answer minus evidence_span."""
    return {k: v for k, v in (verdict["answer"] or {}).items() if k != "evidence_span"}


def render_lettered_options(options) -> str:
    """The fixed display form of a lettered closed set, the one the questions show (procedures.items.template):
    "(A) o1 (B) o2 ..."."""
    if len(options) > len(OPTION_LETTERS):
        raise ValueError(f"a lettered closed_set has at most {len(OPTION_LETTERS)} options")
    return " ".join(f"({OPTION_LETTERS[i]}) {opt}" for i, opt in enumerate(options))


def check_lettered_display(item: dict) -> None:
    """A lettered closed set must appear in the question exactly as "(A) o1 (B) o2 ...", in gold
    option order and with no further lettered option, so that each letter names what the model
    saw.  Items without a lettered closed-set gold pass unchecked."""
    gold = item["gold"]
    if gold.get("gtype") != "closed_set" or not gold.get("lettered"):
        return
    options = gold["options"]
    shown = render_lettered_options(options)
    question = item.get("question")
    if not isinstance(question, str) or shown not in question:
        raise ValueError(f"{item.get('instance_id')}: question does not display the lettered "
                         f"options in gold order as {shown!r}")
    if len(options) < len(OPTION_LETTERS) and f"{shown} ({OPTION_LETTERS[len(options)]})" in question:
        raise ValueError(f"{item.get('instance_id')}: question displays more lettered options "
                         f"than the gold lists")


def arbitration_id(item: dict, verdict: dict) -> str:
    """Same instance + same judged answer -> same id, whichever system produced it."""
    payload = json.dumps([item["instance_id"], _judged_answer(verdict)], sort_keys=True,
                         ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def score_rows(rows: list[dict], doc_indexes: dict[str, DocIndex],
               decisions: dict[str, bool] | None = None) -> list[dict]:
    """rows: [{"item": {...}, "raw": str|None, "finish_reason": str|None, ...}].
    Returns copies with "verdict", "arb_id" (ambiguous rows), "error_class",
    "other_subclass".  `decisions` maps arb_id -> bool from blind arbitration (N11)."""
    decisions = decisions or {}
    bad = {k: v for k, v in decisions.items() if not isinstance(v, bool)}
    if bad:
        raise ValueError(f"arbitration decisions must be bool: {bad}")
    out = []
    for row in rows:
        item = row["item"]
        _counterpart_specs(item)  # validate counterparts even when the answer is right
        check_lettered_display(item)
        verdict = score_answer(row.get("raw"), item["gold"], row.get("finish_reason"))
        scored = {**row, "verdict": verdict, "arb_id": None,
                  "error_class": None, "other_subclass": None}
        if verdict["status"] == "ambiguous":
            scored["arb_id"] = arbitration_id(item, verdict)
            if scored["arb_id"] in decisions:
                verdict["arbitrated"] = True
                verdict["correct"] = decisions[scored["arb_id"]]
        if verdict["status"] != "ambiguous" or verdict["arbitrated"]:
            scored["error_class"], scored["other_subclass"] = classify_error(scored, doc_indexes)
        out.append(scored)
    return out


def arbitration_queue(scored_rows: list[dict]) -> list[dict]:
    """Unresolved ambiguous rows, stripped to a whitelist (no system identity, no evidence
    span), de-duplicated by arb_id and ordered by arb_id so grouping reveals nothing."""
    queue = {}
    for r in scored_rows:
        v = r["verdict"]
        if v["status"] != "ambiguous" or v["arbitrated"]:
            continue
        queue[r["arb_id"]] = {"arb_id": r["arb_id"], "instance_id": r["item"]["instance_id"],
                              "question": r["item"].get("question"), "gold": r["item"]["gold"],
                              "answer": _judged_answer(v), "note": v["note"]}
    return [queue[k] for k in sorted(queue)]


# --------------------------------------------------------------------------- summaries

def _rate(num, den):
    return round(float(Fraction(num) / den), 4) if den else None


def summarize(scored_rows: list[dict], final: bool = False) -> dict:
    """Headline counts for one system/condition.  final=True refuses a non-empty
    arbitration queue (N11)."""
    n = len(scored_rows)
    status = {s: sum(r["verdict"]["status"] == s for r in scored_rows) for s in STATUSES}
    unresolved = sum(r["verdict"]["status"] == "ambiguous" and not r["verdict"]["arbitrated"]
                     for r in scored_rows)
    if final and unresolved:
        raise ValueError(f"{unresolved} ambiguous rows still in the arbitration queue (N11)")
    correct = sum(r["verdict"]["correct"] for r in scored_rows)
    answered = status["predicted"] + sum(r["verdict"]["arbitrated"] for r in scored_rows)
    chance = sum((gold_spec(r["item"]["gold"]).chance for r in scored_rows), Fraction(0))
    return {
        "n": n, "correct": correct, "acc": _rate(correct, n), **status,
        "ambiguous_unresolved": unresolved,
        "arbitrated": sum(r["verdict"]["arbitrated"] for r in scored_rows),
        "arbitrated_correct": sum(r["verdict"]["arbitrated"] and r["verdict"]["correct"]
                                  for r in scored_rows),
        "coverage": _rate(answered, n), "acc_answered": _rate(correct, answered),
        "unit_conversions": sum(r["verdict"]["unit_converted"] for r in scored_rows),
        "chance_expected": round(float(chance), 4),
        "acc_chance_corrected": _rate(correct - chance, n - chance) if n > chance else None,
        "error_classes": {c: sum(r["error_class"] == c for r in scored_rows) for c in ERROR_CLASSES},
        "other_subclasses": {c: sum(r["other_subclass"] == c for r in scored_rows)
                             for c in OTHER_SUBCLASSES},
    }


def pair_consistency(scored_rows: list[dict]) -> dict:
    """Pair-level "all sides correct" for ONE system's rows.  item["pair"] =
    {"pair_id", "side", "n_sides" (default 2)}."""
    groups: dict[str, list[dict]] = {}
    for r in scored_rows:
        pair = r["item"].get("pair")
        if pair:
            groups.setdefault(pair["pair_id"], []).append(r)
    by_pair, incomplete, pending = {}, [], []
    for pid in sorted(groups):
        rs = groups[pid]
        sides = [r["item"]["pair"]["side"] for r in rs]
        n_sides = rs[0]["item"]["pair"].get("n_sides", 2)
        if len(set(sides)) != len(sides) or len(rs) > n_sides:
            raise ValueError(f"pair {pid}: sides {sides} do not fit n_sides={n_sides} "
                             f"(side listed twice, or rows from more than one system?)")
        if len(rs) < n_sides:
            incomplete.append(pid)
        elif any(r["verdict"]["status"] == "ambiguous" and not r["verdict"]["arbitrated"]
                 for r in rs):
            pending.append(pid)
        else:
            by_pair[pid] = all(r["verdict"]["correct"] for r in rs)
    consistent = sum(by_pair.values())
    return {"n_pairs": len(by_pair), "n_consistent": consistent,
            "rate": _rate(consistent, len(by_pair)), "by_pair": by_pair,
            "incomplete": incomplete, "pending": pending}
