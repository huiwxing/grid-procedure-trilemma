"""38 answer / gold cases with expected verdicts, inherited from the test suite of the scorer this one extends (source
file sha256 260dc3fa26c48b7ca168c8e3ac9f86f62176a218c58e3c75d22ac73df266bad8, LF-normalised). The case data in CASES
are copied verbatim (comments translated); DELIBERATE_CHANGES lists the one expectation a scoring rule changed.
"""
import json

import pytest

from procedures.scoring import score_answer

G_NUM = {"value": 1380, "unit": "MW"}
G_NUM_F = {"value": 109.3, "unit": "kV"}
G_NIE = {"value": "not_in_effect", "unit": ""}
G_QUAL = {"value": "Energize shunt reactors", "unit": "",
          "accepted_aliases": ["energize shunt reactor", "switch in shunt reactors"]}
G_DERIV = {"value": 312, "unit": "MVA", "derivation": "1.30 x 240 MVA"}


def J(**kw):
    return json.dumps(kw)


CASES = [
    # ---- numeric gold: positives (3) ----
    ("num+ exact",        J(value=1380, unit="MW"), G_NUM, True, "predicted"),
    ("num+ str+comma",    J(value="1,380", unit="mw"), G_NUM, True, "predicted"),
    ("num+ float",        J(value=109.3, unit="kV"), G_NUM_F, True, "predicted"),
    # ---- numeric gold: negatives (4) ----
    ("num- wrong value",  J(value=1500, unit="MW"), G_NUM, False, "predicted"),
    ("num- near miss",    J(value=1380.0005, unit="MW"), G_NUM, False, "predicted"),
    ("num- wrong unit",   J(value=1380, unit="MVA"), G_NUM, False, "predicted"),
    ("num- non-numeric",  J(value="the LTE figure", unit="MW"), G_NUM, False, "predicted"),
    ("num- null",         J(value=None, unit="MW"), G_NUM, False, "predicted"),
    # ---- numeric gold: unit-missing => ambiguous (arbitration, scored False) ----
    ("num~ missing unit", J(value=1380), G_NUM, False, "ambiguous"),
    ("num~ unknown unit", J(value=1380, unit="megawatts??"), G_NUM, False, "ambiguous"),
    ("num~ kA not schema", J(value=1.38, unit="kA"), {"value": 1380, "unit": "A"}, False, "ambiguous"),
    # ---- not_in_effect gold: positives (3) ----
    ("nie+ token",        J(value="not_in_effect", unit=""), G_NIE, True, "predicted"),
    ("nie+ spaced",       J(value="Not in effect", unit=""), G_NIE, True, "predicted"),
    ("nie+ hyphen",       J(value="not-in-effect", unit=""), G_NIE, True, "predicted"),
    # ---- not_in_effect gold: negatives (4) ----
    ("nie- number",       J(value=98.0, unit="kV"), G_NIE, False, "predicted"),
    ("nie- prose",        J(value="no limit applies today I think", unit=""), G_NIE, False, "predicted"),
    ("nie- wrong token",  J(value="n/a", unit=""), G_NIE, False, "predicted"),
    ("nie- json null",    J(value=None, unit=""), G_NIE, False, "predicted"),
    # ---- qualitative gold: positives (3) ----
    ("qual+ exact",       J(value="Energize shunt reactors", unit=""), G_QUAL, True, "predicted"),
    ("qual+ case/space",  J(value="  energize   shunt reactors ", unit=""), G_QUAL, True, "predicted"),
    ("qual+ alias",       J(value="switch in shunt reactors", unit=""), G_QUAL, True, "predicted"),
    # ---- qualitative gold: negatives (3; includes the token-overlap case of an early iteration, now scored wrong) ----
    ("qual- overlap",     J(value="shunt reactors should be considered", unit=""), G_QUAL, False, "predicted"),
    ("qual- superstring", J(value="Energize shunt reactors and drop load", unit=""), G_QUAL, False, "predicted"),
    ("qual- number",      J(value=42, unit=""), G_QUAL, False, "predicted"),
    # ---- derived-value gold (scored as numeric): 3+ / 3- ----
    ("drv+ exact",        J(value=312, unit="MVA"), G_DERIV, True, "predicted"),
    ("drv+ str",          J(value="312", unit="mva"), G_DERIV, True, "predicted"),
    ("drv+ float",        J(value=312.0, unit="MVA"), G_DERIV, True, "predicted"),
    ("drv- input value",  J(value=240, unit="MVA"), G_DERIV, False, "predicted"),
    # explicit empty unit on numeric gold = a wrong unitless claim, NOT ambiguous
    ("drv- pct",          J(value=1.30, unit=""), G_DERIV, False, "predicted"),
    ("drv- wrong calc",   J(value=276, unit="MVA"), G_DERIV, False, "predicted"),
    # ---- abstention (legal, scored 0, tracked) ----
    ("abstain num",       J(value="insufficient_evidence", unit=""), G_NUM, False, "abstained"),
    ("abstain qual",      J(value="Insufficient evidence", unit=""), G_QUAL, False, "abstained"),
    ("abstain nie",       J(value="insufficient_evidence", unit=""), G_NIE, False, "abstained"),
    # ---- unparseable / structural ----
    ("unparse prose",     "The applicable limit is 1380 MW.", G_NUM, False, "unparseable"),
    ("unparse empty",     "", G_NUM, False, "unparseable"),
    ("ambig missing key", J(unit="MW", tier_or_basis="LTE"), G_NUM, False, "ambiguous"),
    ("ambig legacy key",  J(applied_value=1380, unit="MW"), G_NUM, False, "ambiguous"),
    # ---- fenced JSON still parses ----
    ("fence+ ok",         "```json\n" + J(value=1380, unit="MW") + "\n```", G_NUM, True, "predicted"),
]

# name -> (new expectation, scoring rule).  The inherited case expected kA to fail (no conversion); this scorer
# converts exactly inside a dimension, so 1.38 kA == 1380 A.
DELIBERATE_CHANGES = {
    "num~ kA not schema": ((True, "predicted"), "N5 exact in-dimension conversion (kA -> A)"),
}


def test_case_count_is_38():
    assert len(CASES) == 38
    assert len({c[0] for c in CASES}) == 38
    assert set(DELIBERATE_CHANGES) <= {c[0] for c in CASES}


@pytest.mark.parametrize("name,raw,gold,want_ok,want_status", CASES, ids=[c[0] for c in CASES])
def test_inherited_case(name, raw, gold, want_ok, want_status):
    if name in DELIBERATE_CHANGES:
        (want_ok, want_status), _rule = DELIBERATE_CHANGES[name]
    v = score_answer(raw, gold)
    assert (v["correct"], v["status"]) == (want_ok, want_status), v


def test_ka_change_records_a_conversion():
    v = score_answer(J(value=1.38, unit="kA"), {"value": 1380, "unit": "A"})
    assert v["unit_converted"] is True and v["pred_unit"] == "kA"
