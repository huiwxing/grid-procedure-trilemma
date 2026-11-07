"""Contract tests for the scoring rules the inherited cases (tests/scoring/test_scorer_regression_cases.py) do not cover
(N1-N11, closed set, error classes). Rule ids are those listed in the procedures.scoring package docstring."""
import json
from decimal import Decimal

import pytest

from procedures.scoring import DocIndex, arbitration_queue, extract_answer, pair_consistency, render_lettered_options, score_answer, score_rows, summarize
from procedures.scoring.normalize import UNDECIDABLE, norm_date, norm_label, norm_time, parse_quantity


def J(**kw):
    return json.dumps(kw)


def ok(raw, gold, **kw):
    v = score_answer(raw, gold, **kw)
    return v["correct"], v["status"]


G_MIN = {"gtype": "numeric", "value": 30, "unit": "min"}
G_MW = {"gtype": "numeric", "value": 1380, "unit": "MW"}


# ---------------------------------------------------------------- N1 extraction

def test_n1_think_block_json_is_ignored():
    raw = "<think>draft " + J(value=45, unit="min") + "</think>\n" + J(value=30, unit="min")
    assert ok(raw, G_MIN) == (True, "predicted")
    assert score_answer(raw, G_MIN)["think_stripped"] is True


def test_n1_answer_only_inside_think_is_unparseable():
    raw = "<think>" + J(value=30, unit="min") + "</think> The answer is thirty minutes."
    assert ok(raw, G_MIN) == (False, "unparseable")


def test_n1_orphan_closing_tag_drops_the_prefix():
    raw = "reasoning " + J(value=45, unit="min") + " more </think>" + J(value=30, unit="min")
    assert ok(raw, G_MIN) == (True, "predicted")


def test_n1_orphan_opening_tag_drops_the_suffix():
    raw = J(value=30, unit="min") + "\n<think>second thoughts " + J(value=45, unit="min")
    assert ok(raw, G_MIN) == (True, "predicted")


def test_n1_thinking_tag_and_case_insensitive():
    raw = "<THINKING>" + J(value=45, unit="min") + "</Thinking>" + J(value=30, unit="min")
    assert ok(raw, G_MIN) == (True, "predicted")


def test_n1_last_json_with_value_wins_over_first():
    raw = "Draft: " + J(value=45, unit="min") + "\nFinal answer: " + J(value=30, unit="min")
    assert ok(raw, G_MIN) == (True, "predicted")
    raw_rev = "Draft: " + J(value=30, unit="min") + "\nFinal answer: " + J(value=45, unit="min")
    assert ok(raw_rev, G_MIN) == (False, "predicted")


def test_n1_trailing_object_without_value_does_not_hide_the_answer():
    raw = J(value=30, unit="min") + "\n" + J(confidence="high")
    assert ok(raw, G_MIN) == (True, "predicted")


def test_n1_nested_value_object_is_not_a_candidate():
    raw = json.dumps({"answer": {"value": 30, "unit": "min"}})
    v = score_answer(raw, G_MIN)
    assert (v["correct"], v["status"], v["note"]) == (False, "ambiguous", "value key missing")


def test_n1_object_inside_a_json_array_is_a_candidate():
    assert ok("[" + J(value=30, unit="min") + "]", G_MIN) == (True, "predicted")


def test_n1_malformed_json_is_unparseable():
    assert ok("{'value': 30, 'unit': 'min'}", G_MIN) == (False, "unparseable")
    assert ok('{"value": 30, "unit": "min",}', G_MIN) == (False, "unparseable")


@pytest.mark.parametrize("reason", ["length", "max_tokens", "MAX_TOKENS",
                                    "model_context_window_exceeded"])
def test_n1_truncation_scores_zero_even_with_a_right_json(reason):
    v = score_answer(J(value=30, unit="min"), G_MIN, finish_reason=reason)
    assert (v["correct"], v["status"]) == (False, "truncated")


@pytest.mark.parametrize("reason", [None, "stop", "end_turn", "STOP"])
def test_n1_normal_stop_reasons_are_not_truncation(reason):
    assert ok(J(value=30, unit="min"), G_MIN, finish_reason=reason) == (True, "predicted")


def test_n1_numbers_decode_exactly():
    ex = extract_answer('{"value": 0.30000000000000004}')
    assert ex.answer["value"] == Decimal("0.30000000000000004")


# ---------------------------------------------------------------- N2 / N3

@pytest.mark.parametrize("val", ["Insufficient evidence.", "insufficient_evidence!", "INSUFFICIENT-EVIDENCE"])
def test_n2_abstention_after_terminal_punctuation(val):
    assert ok(J(value=val, unit=""), G_MIN) == (False, "abstained")


@pytest.mark.parametrize("val", ["not_in_effect.", "Not in effect;", "NOT-IN-EFFECT"])
def test_n3_not_in_effect_after_terminal_punctuation(val):
    assert ok(J(value=val, unit=""), {"value": "not_in_effect"}) == (True, "predicted")


def test_n3_not_in_effect_is_strict():
    gold = {"gtype": "not_in_effect", "value": "not_in_effect"}
    assert ok(J(value="The requirement is not in effect", unit=""), gold) == (False, "predicted")
    assert ok(J(value=None, unit=None), gold) == (False, "predicted")
    assert ok(J(value=30, unit="min"), gold) == (False, "predicted")


def test_n3_not_in_effect_answer_on_value_gold_is_wrong():
    assert ok(J(value="not_in_effect", unit=""), G_MIN) == (False, "predicted")


# ---------------------------------------------------------------- N4 numbers

@pytest.mark.parametrize("raw", [
    J(value="thirty (30)", unit="min"), J(value="30 (thirty)", unit="min"),
    J(value="Thirty", unit="minutes"), J(value="thirty (30) minutes"), J(value="30 minutes"),
    J(value="30-minute"), J(value="30.0", unit="min"), '{"value": 30.00, "unit": "min"}',
    J(value="0.5", unit="h"),
])
def test_n4_numeric_forms(raw):
    assert ok(raw, G_MIN) == (True, "predicted")


def test_n4_cardinal_words_cover_zero_to_hundred():
    assert parse_quantity("zero").number == 0
    assert parse_quantity("seventy-two").number == 72
    assert parse_quantity("ninety nine").number == 99
    assert parse_quantity("one hundred").number == 100
    assert parse_quantity("a few") is None


def test_n4_word_numeral_disagreement_is_wrong_not_either_reading():
    v = score_answer(J(value="thirty (31)", unit="min"), G_MIN)
    assert (v["correct"], v["status"]) == (False, "predicted")
    assert "disagree" in v["note"]


def test_n4_exact_decimal_no_tolerance():
    assert ok(J(value="30.0000001", unit="min"), G_MIN) == (False, "predicted")
    gold = {"gtype": "numeric", "value": 0.3, "unit": "pu"}
    assert ok('{"value": 0.30, "unit": "pu"}', gold) == (True, "predicted")
    assert ok('{"value": 0.30000000000000004, "unit": "pu"}', gold) == (False, "predicted")


def test_n4_thousands_separator_must_be_grouped():
    assert ok(J(value="1,380", unit="MW"), G_MW) == (True, "predicted")
    assert ok(J(value="13,80", unit="MW"), G_MW) == (False, "predicted")


def test_n4_booleans_are_not_numbers():
    assert ok('{"value": true, "unit": "MW"}', G_MW) == (False, "predicted")


def test_n4_value_with_unexplained_suffix_goes_to_arbitration():
    v = score_answer(J(value="30 minutes or less"), G_MIN)
    assert v["status"] == "ambiguous"


# ---------------------------------------------------------------- N5 units

@pytest.mark.parametrize("val,unit", [(0.5, "h"), (1800, "s"), ("0.5", "hours"), (30, "Minutes"),
                                       ("1800000", "ms")])
def test_n5_time_converts_exactly_and_is_counted(val, unit):
    v = score_answer(J(value=val, unit=unit), G_MIN)
    assert (v["correct"], v["status"]) == (True, "predicted")
    assert v["unit_converted"] is (unit.lower() not in ("minutes", "min"))


def test_n5_day_hour_week_conversion():
    assert ok(J(value=48, unit="hr"), {"gtype": "numeric", "value": 2, "unit": "days"}) == (True, "predicted")
    assert ok(J(value=14, unit="days"), {"gtype": "numeric", "value": 2, "unit": "weeks"}) == (True, "predicted")
    assert ok(J(value=48, unit="hr"), {"gtype": "numeric", "value": 10, "unit": "day"}) == (False, "predicted")


@pytest.mark.parametrize("val,unit,gold_unit,gold_val", [
    (12, "months", "year", 1), (1, "year", "months", 12), (365, "days", "year", 1),
    (30, "days", "month", 1), (5, "days", "business days", 5),
])
def test_n5_calendar_units_never_convert(val, unit, gold_unit, gold_val):
    gold = {"gtype": "numeric", "value": gold_val, "unit": gold_unit}
    assert ok(J(value=val, unit=unit), gold) == (False, "predicted")


def test_n5_same_calendar_unit_still_matches():
    assert ok(J(value="twelve", unit="calendar months"),
              {"gtype": "numeric", "value": 12, "unit": "months"}) == (True, "predicted")


def test_n5_percent_is_never_a_fraction():
    gold = {"gtype": "numeric", "value": 15.6, "unit": "%"}
    assert ok(J(value=0.156, unit=""), gold) == (False, "predicted")
    assert ok(J(value=0.156, unit="%"), gold) == (False, "predicted")
    assert ok(J(value="15.6%"), gold) == (True, "predicted")
    assert ok(J(value=15.6, unit="percent"), gold) == (True, "predicted")


def test_n5_dimensionless_count_ignores_count_nouns():
    gold = {"gtype": "numeric", "value": 3, "unit": ""}
    assert ok(J(value=3), gold) == (True, "predicted")
    assert ok(J(value=3, unit="contingencies"), gold) == (True, "predicted")
    assert ok(J(value="three transformers"), gold) == (True, "predicted")
    assert ok(J(value=3, unit=""), gold) == (True, "predicted")


def test_n5_dimensionless_count_rejects_physical_units():
    gold = {"gtype": "numeric", "value": 3, "unit": ""}
    v = score_answer(J(value=3, unit="MW"), gold)
    assert (v["correct"], v["status"]) == (False, "predicted")


def test_n5_cross_dimension_is_wrong():
    assert ok(J(value=30, unit="MW"), G_MIN) == (False, "predicted")
    assert ok(J(value=1380, unit="MVA"), G_MW) == (False, "predicted")
    assert ok(J(value=1380, unit="MWh"), G_MW) == (False, "predicted")


@pytest.mark.parametrize("val,unit,gold", [
    (1.38, "GW", G_MW), (1380000, "kW", G_MW),
    (0.036, "Hz", {"gtype": "numeric", "value": 36, "unit": "mHz"}),
    (138000, "V", {"gtype": "numeric", "value": 138, "unit": "kV"}),
    (100, "Mvar", {"gtype": "numeric", "value": 100, "unit": "MVAR"}),
    (2, "MWh", {"gtype": "numeric", "value": 2000, "unit": "kWh"}),
])
def test_n5_electrical_units_convert_inside_their_dimension(val, unit, gold):
    assert ok(J(value=val, unit=unit), gold) == (True, "predicted")


def test_n5_conflicting_unit_claims_go_to_arbitration():
    assert ok(J(value="30 minutes", unit="h"), G_MIN) == (False, "ambiguous")


def test_n5_unit_inside_value_wins_over_empty_unit_key():
    assert ok(J(value="30 minutes", unit=""), G_MIN) == (True, "predicted")


def test_n5_null_unit_is_missing_not_unitless():
    assert ok(J(value=30, unit=None), G_MIN) == (False, "ambiguous")


def test_n5_gold_unit_outside_table_is_a_construction_error():
    with pytest.raises(ValueError, match="unit table"):
        score_answer(J(value=5, unit="$"), {"gtype": "numeric", "value": 5, "unit": "$/MWh"})


# ---------------------------------------------------------------- N6 categorical

G_TO = {"gtype": "categorical", "value": "Transmission Owner", "accepted_aliases": ["TO"]}


@pytest.mark.parametrize("val", ["the Transmission Owner.", "Transmission Owners", "TO", "the TO",
                                 "transmission-owner", "A transmission owner"])
def test_n6_articles_punctuation_plural_alias(val):
    assert ok(J(value=val, unit=""), G_TO) == (True, "predicted")


@pytest.mark.parametrize("val", ["Transmission Owner or PJM", "Owner", "TOs and GOs", "Transmission Operator"])
def test_n6_no_fuzzy_matching(val):
    assert ok(J(value=val, unit=""), G_TO) == (False, "predicted")


def test_n6_label_letters_are_not_articles():
    gold = {"gtype": "categorical", "value": "Phase A"}
    assert ok(J(value="Phase A", unit=""), gold) == (True, "predicted")
    assert ok(J(value="Phase", unit=""), gold) == (False, "predicted")
    assert norm_label("Category A facility") == "category a facility"


# ---------------------------------------------------------------- N6' closed set

G_CS = {"gtype": "closed_set", "value": "open the breaker",
        "options": ["open the breaker", "notify the PJM dispatcher"],
        "option_aliases": {"notify the PJM dispatcher": ["notify PJM"]}}


def test_n6p_closed_set_maps_answer_to_its_option():
    v = score_answer(J(value="Open the breaker.", unit=""), G_CS)
    assert (v["correct"], v["matched_option"]) == (True, "open the breaker")


def test_n6p_other_branch_option_is_wrong_and_identified():
    v = score_answer(J(value="notify PJM", unit=""), G_CS)
    assert (v["correct"], v["status"], v["matched_option"]) == (False, "predicted", "notify the PJM dispatcher")


def test_n6p_outside_the_set_is_wrong():
    v = score_answer(J(value="open the breaker and notify PJM", unit=""), G_CS)
    assert (v["correct"], v["matched_option"], v["note"]) == (False, None, "outside the closed set")


def test_n6p_plural_of_an_option():
    gold = {"gtype": "closed_set", "value": "shunt reactor", "options": ["shunt reactor", "capacitor bank"]}
    assert ok(J(value="shunt reactors", unit=""), gold) == (True, "predicted")


@pytest.mark.parametrize("bad", [
    {"gtype": "closed_set", "value": "x", "options": ["x"]},
    {"gtype": "closed_set", "value": "z", "options": ["x", "y"]},
    {"gtype": "closed_set", "value": "unit", "options": ["unit", "units"]},
    {"gtype": "closed_set", "value": "x", "options": ["x", "y"], "option_aliases": {"w": ["q"]}},
    {"gtype": "closed_set", "value": "x", "options": ["x", "y"], "option_aliases": {"x": ["Y"]}},
])
def test_n6p_malformed_closed_sets_raise(bad):
    with pytest.raises(ValueError):
        score_answer(J(value="x", unit=""), bad)


# S2-side items (one side of an owner-written branch pair): options in display order, answered by text or by letter
G_SIDE = {"gtype": "closed_set", "lettered": True, "value": "notify the PJM dispatcher",
          "options": ["open the breaker", "notify the PJM dispatcher", "none of these"],
          "option_aliases": {"notify the PJM dispatcher": ["notify PJM"]}}
Q_SIDE = ("Under PJM Manual 3 (Revision 71, effective 2026-01-01), when the line trips, which of the "
          "following applies? (A) open the breaker (B) notify the PJM dispatcher (C) none of these.")


@pytest.mark.parametrize("val,note", [
    ("B", "option letter B"), ("b", "option letter B"), ("(B)", "option letter B"), ("B)", "option letter B"),
    ("B.", "option letter B"), ("Option B", "option letter B"), ("option (b)", "option letter B"),
    ("(B) notify the PJM dispatcher", "option letter B"), ("B. notify PJM", "option letter B"), ("B: notify PJM", "option letter B"),
    ("B) Notify the PJM dispatcher.", "option letter B"), ("notify the PJM dispatcher", None), ("Notify PJM.", None),
])
def test_n6p_lettered_set_accepts_letter_or_text(val, note):
    v = score_answer(J(value=val, unit=""), G_SIDE)
    assert (v["correct"], v["status"], v["matched_option"]) == (True, "predicted", "notify the PJM dispatcher")
    assert v["note"] == note


def test_n6p_letter_of_another_option_is_wrong_and_identified():
    v = score_answer(J(value="(A)"), G_SIDE)
    assert (v["correct"], v["matched_option"], v["note"]) == (False, "open the breaker", "option letter A")
    v = score_answer(J(value="C"), G_SIDE)
    assert (v["correct"], v["matched_option"]) == (False, "none of these")


def test_n6p_letter_and_text_must_agree():
    v = score_answer(J(value="(A) notify the PJM dispatcher"), G_SIDE)
    assert (v["correct"], v["status"], v["matched_option"]) == (False, "predicted", None)
    assert v["note"] == "option letter A and text disagree"
    v = score_answer(J(value="(B) tell the dispatcher"), G_SIDE)
    assert (v["correct"], v["status"], v["note"]) == (False, "ambiguous",
                                                      "option letter B with text outside the closed set")


@pytest.mark.parametrize("val", ["D", "(D)", "AB", "B and C", "A/B"])
def test_n6p_letters_outside_the_display_are_outside_the_set(val):
    v = score_answer(J(value=val), G_SIDE)
    assert (v["correct"], v["matched_option"], v["note"]) == (False, None, "outside the closed set")


def test_n6p_unlettered_set_does_not_read_letters():
    unlettered = {k: v for k, v in G_SIDE.items() if k != "lettered"}
    v = score_answer(J(value="B"), unlettered)
    assert (v["correct"], v["matched_option"], v["note"]) == (False, None, "outside the closed set")
    assert score_answer(J(value="notify PJM"), unlettered)["correct"] is True


@pytest.mark.parametrize("bad", [
    {**G_SIDE, "lettered": "yes"},
    {**G_SIDE, "options": ["B", "notify the PJM dispatcher", "none of these"], "option_aliases": {}},
    {**G_SIDE, "option_aliases": {"none of these": ["a"]}},
])
def test_n6p_malformed_lettered_sets_raise(bad):
    with pytest.raises(ValueError):
        score_answer(J(value="B"), bad)


def test_n6p_letter_that_is_its_own_option_text_is_allowed():
    gold = {"gtype": "closed_set", "lettered": True, "value": "B", "options": ["A", "B"]}
    assert ok(J(value="(B)"), gold) == (True, "predicted")


def _side_row(raw, gold=G_SIDE, question=Q_SIDE, counterparts=()):
    return {"item": {"instance_id": "s2-side-1", "question": question, "gold": gold, "stratum": "S2",
                     "counterparts": list(counterparts), "docs": ["d"]}, "raw": raw}


def test_n6p_render_matches_the_template_display():
    assert render_lettered_options(G_SIDE["options"]) in Q_SIDE


@pytest.mark.parametrize("question", [
    Q_SIDE.replace("(A) open the breaker (B) notify the PJM dispatcher",
                   "(A) notify the PJM dispatcher (B) open the breaker"),
    Q_SIDE.replace("(C) none of these.", "(C) none of these (D) both."),
    "when the line trips, what applies?",
    None,
])
def test_n6p_score_rows_checks_the_displayed_letter_order(question):
    with pytest.raises(ValueError, match="lettered options"):
        score_rows([_side_row(J(value="B"), question=question)], {"d": DocIndex.from_text("")})


def test_n6p_lettered_chance_rate_counts_every_displayed_option():
    s = summarize(score_rows([_side_row(J(value="B"))], {"d": DocIndex.from_text("")}), final=True)
    assert (s["correct"], s["chance_expected"]) == (1, round(1 / 3, 4))


# ---------------------------------------------------------------- N7 yes/no

def test_n7_yes_no_scoring():
    gold = {"gtype": "yes_no", "value": "Yes"}
    assert ok(J(value="yes.", unit=""), gold) == (True, "predicted")
    assert ok('{"value": true, "unit": ""}', gold) == (True, "predicted")
    assert ok(J(value="Yes, the TO must notify PJM", unit=""), gold) == (False, "predicted")
    assert ok('{"value": false, "unit": ""}', gold) == (False, "predicted")


def test_n7_chance_correction_in_summary():
    gold = {"gtype": "yes_no", "value": "yes"}
    rows = [{"item": {"instance_id": f"i{k}", "gold": gold, "docs": ["d"]},
             "raw": J(value="yes" if k < 3 else "no", unit="")} for k in range(4)]
    s = summarize(score_rows(rows, {"d": DocIndex.from_text("yes no")}))
    assert (s["acc"], s["chance_expected"], s["acc_chance_corrected"]) == (0.75, 2.0, 0.5)


# ---------------------------------------------------------------- N8 time / date

@pytest.mark.parametrize("val", ["11 a.m.", "11:00 AM", "1100", "11:00", "1100 EPT", "11 am EPT",
                                 "11:00 Eastern Prevailing Time", "1100 hours"])
def test_n8_times_normalise_to_24h(val):
    assert ok(J(value=val, unit=""), {"gtype": "time_of_day", "value": "11:00"}) == (True, "predicted")


def test_n8_time_edge_cases():
    assert norm_time("12 a.m.") == "00:00" and norm_time("12:30 pm") == "12:30"
    assert norm_time("2400") == "24:00" and norm_time("2460") is None and norm_time("11") is None
    assert norm_time("noon") == "12:00" and norm_time("midnight") is UNDECIDABLE
    gold = {"gtype": "time_of_day", "value": "23:00"}
    assert ok(J(value="11 p.m.", unit=""), gold) == (True, "predicted")
    assert ok(J(value="11 a.m.", unit=""), gold) == (False, "predicted")
    assert ok(J(value="midnight", unit=""), {"gtype": "time_of_day", "value": "00:00"}) == (False, "ambiguous")


@pytest.mark.parametrize("val", ["June 1, 2022", "Jun. 1st, 2022", "06/01/2022", "6/1/2022",
                                 "JUN-01-2022", "2022-06-01", "1 June 2022", "06-01-2022"])
def test_n8_dates_normalise_to_iso_month_first(val):
    assert ok(J(value=val, unit=""), {"gtype": "date", "value": "2022-06-01"}) == (True, "predicted")


def test_n8_date_edge_cases():
    assert norm_date("01/06/2022") == "2022-01-06"          # North-American MM/DD
    assert norm_date("02/30/2024") is None
    assert norm_date("06/01/22") is UNDECIDABLE
    gold = {"gtype": "date", "value": "June 1, 2022"}
    assert ok(J(value="06/01/22", unit=""), gold) == (False, "ambiguous")
    assert ok(J(value="effective June 1, 2022", unit=""), gold) == (False, "predicted")


# ---------------------------------------------------------------- N9 intervals

def test_n9_interval_split_into_bounds():
    upper = {"unit": "Hz", "gtype": "numeric", "value": 60.05, "bound": "upper"}   # the upper bound of 59.95-60.05 Hz
    assert ok(J(value=60.05, unit="Hz"), upper) == (True, "predicted")
    assert ok(J(value="59.95-60.05", unit="Hz"), upper) == (False, "predicted")


def test_n9_interval_gold_is_not_scorable_unsplit():
    gold = {"gtype": "interval", "value": {"lower": 1, "upper": 2}, "unit": "h"}
    with pytest.raises(ValueError, match="N9"):
        score_answer(J(value=1, unit="h"), gold)


def test_unknown_or_malformed_gold_raises():
    for bad in ({"gtype": "free_text", "value": "x"}, {"gtype": "numeric", "value": "about 5"},
                {"gtype": "yes_no", "value": "maybe"}, {"gtype": "date", "value": "06/01/22"},
                {"value": None}):
        with pytest.raises(ValueError):
            score_answer(J(value="x"), bad)


# ---------------------------------------------------------------- N10 producer failure

def test_n10_no_output_is_unparseable_and_e_parse():
    rows = [{"item": {"instance_id": "x", "gold": G_MIN, "docs": ["d"]}, "raw": None}]
    r = score_rows(rows, {"d": DocIndex.from_text("")})[0]
    assert (r["verdict"]["status"], r["error_class"]) == ("unparseable", "E-parse")


# ---------------------------------------------------------------- N11 arbitration

def _amb_rows():
    item = {"instance_id": "m03-s2-007", "question": "Q?", "gold": G_MIN, "docs": ["d"]}
    return [
        {"item": item, "raw": J(value=30, evidence_span="span"), "system": "reader-A", "tier": "A"},
        {"item": item, "raw": J(value=30, evidence_span="other"), "system": "reader-C", "tier": "C"},
    ]


def test_n11_final_summary_refuses_open_queue():
    scored = score_rows(_amb_rows(), {"d": DocIndex.from_text("30 minutes")})
    assert summarize(scored)["ambiguous_unresolved"] == 2
    with pytest.raises(ValueError, match="N11"):
        summarize(scored, final=True)


def test_n11_queue_is_identity_stripped_and_deduplicated():
    scored = score_rows(_amb_rows(), {"d": DocIndex.from_text("30 minutes")})
    queue = arbitration_queue(scored)
    assert len(queue) == 1
    assert set(queue[0]) == {"arb_id", "instance_id", "question", "gold", "answer", "note"}
    assert queue[0]["answer"] == {"value": 30}
    assert "reader" not in json.dumps(queue) and "tier" not in json.dumps(queue)


def test_n11_decisions_resolve_rows_and_classify_them():
    idx = {"d": DocIndex.from_text("30 minutes")}
    arb_id = arbitration_queue(score_rows(_amb_rows(), idx))[0]["arb_id"]
    accepted = score_rows(_amb_rows(), idx, decisions={arb_id: True})
    s = summarize(accepted, final=True)
    assert (s["correct"], s["arbitrated"], s["arbitrated_correct"]) == (2, 2, 2)
    rejected = score_rows(_amb_rows(), idx, decisions={arb_id: False})
    assert [r["error_class"] for r in rejected] == ["E-unit", "E-unit"]
    with pytest.raises(ValueError):
        score_rows(_amb_rows(), idx, decisions={arb_id: "yes"})


# ---------------------------------------------------------------- error classes

DOC = DocIndex.from_text(
    "The TO shall restore within thirty (30) minutes. Under emergency conditions the limit is "
    "15 minutes; otherwise 60 minutes. Effective June 1, 2022 at 1100 EPT the Transmission "
    "Owner notifies PJM.")


def _row(raw, gold, counterparts=(), finish_reason=None):
    return {"item": {"instance_id": "i", "gold": gold, "counterparts": list(counterparts),
                     "docs": ["d"]}, "raw": raw, "finish_reason": finish_reason}


def _cls(row):
    r = score_rows([row], {"d": DOC})[0]
    return r["error_class"], r["other_subclass"]


def test_err_priority_parse_then_abstain():
    assert _cls(_row("no json here", G_MIN)) == ("E-parse", None)
    assert _cls(_row(J(value=30, unit="min"), G_MIN, finish_reason="length")) == ("E-parse", None)
    assert _cls(_row(J(value="insufficient_evidence"), G_MIN)) == ("E-abs", None)


def test_err_version_beats_exception_for_deleted_requirement():
    gold = {"gtype": "not_in_effect", "value": "not_in_effect"}
    cps = [{"value": 30, "unit": "min", "axes": ["exception", "version"]}]
    assert _cls(_row(J(value=30, unit="min"), gold, cps)) == ("E-ver", None)


def test_err_condition_and_exception_axes():
    cond = [{"value": 60, "unit": "min", "axes": ["condition"]}]
    exc = [{"value": 60, "unit": "min", "axes": ["exception"]}]
    gold = {"gtype": "numeric", "value": 15, "unit": "min"}
    assert _cls(_row(J(value=1, unit="h"), gold, cond)) == ("E-cond", None)
    assert _cls(_row(J(value=60, unit="min"), gold, exc)) == ("E-exc", None)


def test_err_counterpart_not_in_effect_matches_nie_answer():
    cps = [{"value": "not_in_effect", "axes": ["version"]}]
    assert _cls(_row(J(value="not in effect"), G_MIN, cps)) == ("E-ver", None)


def test_err_closed_set_other_option_is_e_cond():
    cps = [{"value": "notify the PJM dispatcher", "axes": ["condition"]}]
    assert _cls(_row(J(value="notify PJM"), G_CS, cps)) == ("E-cond", None)


def test_err_lettered_other_branch_by_letter_is_e_cond():
    cps = [{"value": "open the breaker", "axes": ["condition"]}]
    r = score_rows([_side_row(J(value="A"), counterparts=cps)], {"d": DOC})[0]
    assert (r["error_class"], r["verdict"]["matched_option"]) == ("E-cond", "open the breaker")


def test_err_lettered_answer_is_looked_up_as_its_option_text():
    cps = [{"value": "open the breaker", "axes": ["condition"]}]
    doc = {"d": DocIndex.from_text("Category C facilities open the breaker.")}  # contains the token "c"
    by_letter = score_rows([_side_row(J(value="C"), counterparts=cps)], doc)[0]
    by_text = score_rows([_side_row(J(value="none of these"), counterparts=cps)], doc)[0]
    assert (by_letter["error_class"], by_letter["other_subclass"]) == ("E-other", "value_not_in_doc")
    assert (by_text["error_class"], by_text["other_subclass"]) == ("E-other", "value_not_in_doc")


def test_err_unit_when_number_right_unit_wrong():
    assert _cls(_row(J(value=30, unit="h"), G_MIN)) == ("E-unit", None)
    assert _cls(_row(J(value=30, unit=""), G_MIN)) == ("E-unit", None)
    assert _cls(_row(J(value=30, unit="MW"), G_MIN)) == ("E-unit", None)


def test_err_other_split_by_document_presence():
    assert _cls(_row(J(value=15, unit="h"), G_MIN)) == ("E-other", "value_in_doc")
    assert _cls(_row(J(value=47, unit="min"), G_MIN)) == ("E-other", "value_not_in_doc")
    assert _cls(_row(J(value="PJM", unit=""), G_TO)) == ("E-other", "value_in_doc")
    assert _cls(_row(J(value="Generator Owner"), G_TO)) == ("E-other", "value_not_in_doc")
    date_gold = {"gtype": "date", "value": "2023-01-01"}
    assert _cls(_row(J(value="2022-06-01"), date_gold)) == ("E-other", "value_in_doc")
    time_gold = {"gtype": "time_of_day", "value": "12:00"}
    assert _cls(_row(J(value="11 a.m."), time_gold)) == ("E-other", "value_in_doc")
    assert _cls(_row(J(value="not_in_effect"), G_MIN)) == ("E-other", "value_not_in_doc")


def test_err_s3_without_counterparts_never_gets_ver_or_cond():
    assert _cls(_row(J(value=60, unit="min"), G_MIN)) == ("E-other", "value_in_doc")


def test_err_bad_counterpart_axes_raise_even_on_a_right_answer():
    with pytest.raises(ValueError, match="axes"):
        score_rows([_row(J(value=30, unit="min"), G_MIN, [{"value": 60, "unit": "min"}])], {"d": DOC})


def test_err_other_requires_docs():
    row = _row(J(value=47, unit="min"), G_MIN)
    row["item"]["docs"] = []
    with pytest.raises(ValueError, match="docs"):
        score_rows([row], {"d": DOC})


def test_doc_index_contents():
    assert {Decimal(30), Decimal(15), Decimal(60), Decimal(1100)} <= DOC.numbers
    assert "2022-06-01" in DOC.dates and "11:00" in DOC.times
    assert DOC.contains_label("the Transmission Owner") and not DOC.contains_label("Owner PJM")


# ---------------------------------------------------------------- pair consistency

def _pair_row(pid, side, raw, n_sides=2):
    return {"item": {"instance_id": f"{pid}-{side}", "gold": G_MIN, "docs": ["d"],
                     "pair": {"pair_id": pid, "side": side, "n_sides": n_sides}}, "raw": raw}


def test_pair_consistency_counts_complete_pending_incomplete():
    right, wrong, amb = J(value=30, unit="min"), J(value=60, unit="min"), J(value=30)
    rows = [_pair_row("p1", "a", right), _pair_row("p1", "b", right),
            _pair_row("p2", "a", right), _pair_row("p2", "b", wrong),
            _pair_row("p3", "a", right), _pair_row("p3", "b", amb),
            _pair_row("p4", "a", right)]
    res = pair_consistency(score_rows(rows, {"d": DOC}))
    assert (res["n_pairs"], res["n_consistent"], res["rate"]) == (2, 1, 0.5)
    assert res["by_pair"] == {"p1": True, "p2": False}
    assert (res["pending"], res["incomplete"]) == (["p3"], ["p4"])


def test_pair_consistency_rejects_mixed_systems():
    rows = [_pair_row("p1", "a", J(value=30, unit="min"))] * 2
    with pytest.raises(ValueError, match="twice"):
        pair_consistency(score_rows(rows, {"d": DOC}))
    extra = [_pair_row("p1", s, J(value=30, unit="min")) for s in ("a", "b", "c")]
    with pytest.raises(ValueError, match="n_sides=2"):
        pair_consistency(score_rows(extra, {"d": DOC}))


def test_json_string_with_braces_does_not_break_the_scan():
    raw = '{"value": "open {the} breaker", "unit": "", "evidence_span": "x}{"} trailing }'
    gold = {"gtype": "categorical", "value": "open {the} breaker"}
    assert ok(raw, gold) == (True, "predicted")


# ---------------------------------------------------------------- summary

def test_summary_counts_and_conversions():
    rows = [_row(J(value=0.5, unit="h"), G_MIN), _row(J(value="insufficient_evidence"), G_MIN),
            _row("", G_MIN), _row(J(value=30, unit="min"), G_MIN, finish_reason="length")]
    s = summarize(score_rows(rows, {"d": DOC}), final=True)
    assert (s["n"], s["correct"], s["abstained"], s["unparseable"], s["truncated"]) == (4, 1, 1, 1, 1)
    assert (s["unit_conversions"], s["coverage"], s["acc_answered"]) == (1, 0.25, 1.0)
    assert s["error_classes"]["E-parse"] == 2 and s["error_classes"]["E-abs"] == 1
