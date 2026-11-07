"""Analysis layer on synthetic runs: loading and joins, the cluster bootstrap, capability / Gap (reference accuracy
minus a reader's) / dGap (how much a memory narrows that gap) / tier random effect / outcomes, the family-cluster
sensitivity, the summary hypothesis H_S (branch errors of an LLM-built summary against a zero-LLM memory), the
empty-cell void rule, the supply / power check on a pool, the build and compression attribution by request
identity, one-book pricing, energy bands, the Codec roll-up, the self-check against the run summary, a re-scoring's
rows file and the report files. Zero network."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

import analysis_synth as S
from procedures.analysis import capability as cap
from procedures.analysis.costs import build_costs, energy, load_bands, sustainability
from procedures.analysis.data import load_run, observations
from procedures.analysis.report import analyse, render_markdown, report_bytes, write_report
from procedures.analysis.stats import boot
from procedures.analysis.taxonomy import taxonomy
from procedures.readers import load_panel, load_price_book
from procedures.readers.pricing import DEFAULT_PRICE_BOOK

ROOT = Path(__file__).resolve().parents[2]
ITEMS = S.make_items()
IDS = [it["item_id"] for it in ITEMS]
BUILD = {"M-01@1": [(500, 20), (400, 30)], "BPM-001@1": [(300, 10)]}
BUILD_TOTAL = {"M-01": 950, "BPM-001": 310}


@pytest.fixture(scope="module")
def panel():
    return load_panel()


@pytest.fixture(scope="module")
def book():
    return load_price_book()


@pytest.fixture(scope="module")
def bands():
    return load_bands(ROOT / "data" / "analysis" / "energy_bands.json")


def spec_rows(spec: dict, per_cell: dict | None = None):
    """spec: (profile, condition) -> correct item ids; per_cell: (profile, condition, item id) -> row() keywords;
    M5 rows carry the synthetic build."""
    per_cell = per_cell or {}
    def rows_of(run_id):
        out = []
        for (pid, cond), good in spec.items():
            for it in ITEMS:
                kw = dict(per_cell.get((pid, cond, it["item_id"]), {}))
                if cond.startswith("M5"):
                    kw.setdefault("build_tokens", BUILD_TOTAL[it["doc"]])
                    kw.setdefault("memory_key", f"m5-{cond}-{it['doc']}")
                if cond == "M0@full":
                    kw.setdefault("tokens", (80_000, 50))
                out.append(S.row(run_id, pid, cond, it, it["item_id"] in good, **kw))
        return out
    return rows_of


def run_of(tmp_path, panel, book, spec, builds=BUILD, per_cell=None, requests=True):
    m5 = sorted({c for _, c in spec if c.startswith("M5")})
    records = S.build_calls(panel, book, builds, imported=1) if m5 else None
    mapping = S.m5_requests(records, {f"m5-{c}-{doc}": f"{doc}@1" for c in m5 for doc in ("M-01", "BPM-001")}) \
        if m5 and requests else None
    return load_run(S.write_run(tmp_path, spec_rows(spec, per_cell), ITEMS, panel, book, build_records=records,
                                requests=mapping))


ALL = set(IDS)


def base_spec() -> dict:
    """Reference B1 M0 all right; B1 M2 all right, M5 half; C readers: M2 all right, M5 12 / 8 / 4 of 16 right."""
    spec = {(S.READER_B, "M0@full"): ALL, (S.READER_B, "M2@4000"): ALL, (S.READER_B, S.M5): set(IDS[:8])}
    for pid, k in zip(S.READERS_C, (12, 8, 4)):
        spec[(pid, "M2@4000")] = ALL
        spec[(pid, S.M5)] = set(IDS[:k])
    return spec


# ---------------------------------------------------------------------- loading


def test_load_run_reads_a_run_anywhere_and_checks_items_and_overlay(tmp_path, panel, book):
    """A run directory loads under any name with the run id its rows record; items other than the run's are warned
    about, items its rows name must be there, and two runs are never overlaid."""
    run = run_of(tmp_path, panel, book, {(S.READER_B, "M2@4000"): ALL})
    assert len(observations([run], ITEMS, panel)) == len(ITEMS)
    moved = tmp_path / "renamed run"
    shutil.copytree(run.run_dir, moved)
    again = load_run(moved)
    assert again.run_id == run.run_id and again.rows == run.rows
    changed = [dict(it, question="Another question?") if it["item_id"] == "u1a" else it for it in ITEMS]
    with pytest.warns(UserWarning, match="differ from the ones the run was planned with"):
        assert len(observations([run], changed, panel)) == len(ITEMS)
    with pytest.raises(ValueError, match="not in the item file"):
        observations([run], [it for it in ITEMS if it["item_id"] != "u1a"], panel)
    with pytest.raises(ValueError, match="never overlaid"):
        observations([run, run], ITEMS, panel)


def test_observation_fields(tmp_path, panel, book):
    run = run_of(tmp_path, panel, book, {(S.READER_B, S.M5): ALL})
    obs = {o.item_id: o for o in observations([run], ITEMS, panel)}
    assert obs["u1a"].cluster == "L3:U1" and obs["u1a"].layer == "abstracted" and obs["u4a"].layer == "transparent"
    assert obs["p1a"].cluster == "S2:P1" and obs["s3x"].cluster == "S3:M-01" and obs["s3x"].layer is None
    assert obs["u1a"].chance == pytest.approx(1 / 3) and obs["u2b"].chance == 0 and obs["s3x"].chance == 0
    assert obs["u1a"].arm == "B-built" and obs["u1a"].condition == S.M5 and obs["u1a"].budget == 4000
    assert not any(o.memory_empty for o in obs.values())


def test_memory_empty_is_read_from_the_row_and_derived_for_rows_before_the_field(tmp_path, panel, book):
    """An empty memory stays in its condition, flagged. A row written before the field is read from its memory tokens
    (0 on a non-closed-book cell = empty); closed book is never an empty memory."""
    per_cell = {(S.READER_B, "M2@4000", "u1a"): {"memory_empty": True},
                (S.READER_B, "M2@4000", "u2b"): {"memory_empty": None},
                (S.READER_B, "B0@full", "u2b"): {"memory_empty": None}}
    run = run_of(tmp_path, panel, book, {(S.READER_B, "M2@4000"): ALL, (S.READER_B, "B0@full"): ALL},
                 per_cell=per_cell)
    obs = {(o.condition, o.item_id): o.memory_empty for o in observations([run], ITEMS, panel)}
    assert {k for k, v in obs.items() if v} == {("M2@4000", "u1a"), ("M2@4000", "u2b")}
    assert len(obs) == 2 * len(ITEMS)


# ---------------------------------------------------------------------- bootstrap


def test_bootstrap_is_seeded_by_key_and_needs_two_clusters():
    clusters = {"a": "c1", "b": "c1", "c": "c2", "d": "c3"}
    v = {"a": 1.0, "b": 0.0, "c": 1.0, "d": 0.0}
    one, two = boot([v], clusters, "k1"), boot([v], clusters, "k1")
    assert np.array_equal(one.reps, two.reps) and not np.array_equal(one.reps, boot([v], clusters, "k2").reps)
    assert one.point[0] == 0.5 and one.n_clusters == 3 and one.items == ("a", "b", "c", "d")
    assert boot([{"a": 1.0, "b": 0.0}], clusters, "k").reps is None                 # one cluster: no interval
    same = boot([{"a": 1.0, "b": 0.0, "c": 0.5, "d": 0.5}], {"a": "c1", "b": "c1", "c": "c2", "d": "c2"}, "k")
    assert np.allclose(same.reps, 0.5)                                              # equal clusters: degenerate
    paired = boot([{"a": 1.0, "c": 1.0}, {"a": 0.0, "b": 1.0, "c": 1.0}], clusters, "k")
    assert paired.items == ("a", "c")                                               # arms compared on common items


# ---------------------------------------------------------------------- capability


def test_accuracy_and_chance_corrected_lettered(tmp_path, panel, book):
    good = ALL - {"u1a", "u3b", "u5a", "s3x"}
    run = run_of(tmp_path, panel, book, {(S.READERS_C[0], "M2@4000"): good})
    grid = cap.Grid(observations([run], ITEMS, panel))
    key = (S.READERS_C[0], "M2@4000")
    acc = grid.accuracy(key, "all")
    assert acc["est"] == 0.75 and acc["n"] == 16 and acc["ci"][0] <= 0.75 <= acc["ci"][1]
    assert acc == grid.accuracy(key, "all")
    ab = grid.accuracy(key, "L3-abstracted")                   # u1a..u3b: 6 lettered cells, u2b chance 0
    assert (ab["n"], ab["clusters"]) == (6, 3)
    assert ab["lettered"]["chance_corrected"]["est"] == round((4 - 5 / 3) / (6 - 5 / 3), 4)
    assert "lettered" not in grid.accuracy(key, "S3")


def test_gap_dgap_orientation_and_tier_random_effect(tmp_path, panel, book):
    grid = cap.Grid(observations([run_of(tmp_path, panel, book, base_spec())], ITEMS, panel))
    ref = grid.reference()
    assert ref == (S.READER_B, "M0@full") and grid.reference(S.READERS_C[0]) is None     # no M0@full: no reference
    c1 = S.READERS_C[0]
    assert grid.gap((c1, "M2@4000"), "all", ref)["est"] == 0.0
    g5 = grid.gap((c1, S.M5), "all", ref)
    assert g5["est"] == 0.25 and g5["n_discordant"] == 4 and g5["token_ratio"] == round(1000 / 80_000, 4)
    fam = grid.gap((c1, S.M5), "all", ref, by="family")
    assert fam["est"] == g5["est"] and fam["clusters"] == 2 and fam["ci"] != g5["ci"]      # PJM, MISO
    with pytest.raises(ValueError, match="cluster level"):
        grid.gap((c1, S.M5), "all", ref, by="doc")
    assert grid.arm_pairs(c1) == [("M2@4000", S.M5)]
    p = grid.pair(c1, "M2@4000", S.M5, "all")
    assert p["arms"] == ["zero-LLM", "B-built"] and p["dgap"]["est"] == -0.25 and p["n_discordant"] == 4
    assert p["h_t"] == "not detected" and p["dgap"]["ci"][1] >= 0 and not p["n_eff_ok"]   # 4 misses in 3 clusters
    p3 = grid.pair(S.READERS_C[2], "M2@4000", S.M5, "all")
    assert p3["dgap"]["est"] == -0.75 and p3["h_t"] == "effective" and p3["dgap"]["ci"][1] < 0
    assert grid.tiers() == ["B", cap.CONSTRAINED_TIER]            # the three synthetic C readers are access tier C1
    t = grid.tier_pair(cap.CONSTRAINED_TIER, "M2@4000", S.M5, "all")
    assert t["models"] == sorted(S.READERS_C) and t["dgap"]["est"] == -0.5
    assert t["between_model_sd"] == 0.25 and t["model_level_resampled"]
    assert t["dgap"]["ci"][0] <= -0.5 <= t["dgap"]["ci"][1]
    tb = grid.tier_pair("B", "M2@4000", S.M5, "all")
    assert tb["dgap"]["est"] == -0.5 and not tb["model_level_resampled"] and tb["between_model_sd"] is None
    inter = grid.interaction("M2@4000", S.M5, "all")
    assert inter["interaction"]["est"] == 0.0
    assert inter["models"] == {cap.CONSTRAINED_TIER: sorted(S.READERS_C), "B": [S.READER_B]}
    tg = grid.tier_gap(cap.CONSTRAINED_TIER, S.M5, "all", ref)
    assert tg["est"] == 0.5 and tg["models"] == sorted(S.READERS_C)


def test_outcome_rules():
    def cond(c, budget):
        return cap.Cond("p", c, "C", c.split("@")[0], budget, None, "zero-LLM", None)
    conds = {"M2@4000": cond("M2@4000", 4000), "M4@4000": cond("M4@4000", 4000), "M2@65536": cond("M2@65536", 65536),
             "M0@full": cond("M0@full", None)}

    def g(est, lo, hi, ratio=0.1):
        return {"est": est, "ci": [lo, hi], "token_ratio": ratio}
    far = {"M2@65536": g(0.0, -0.01, 0.01), "M0@full": g(0.0, 0.0, 0.0)}      # not local: never counted
    assert cap.outcome({"M2@4000": g(0.2, 0.1, 0.3), "M4@4000": g(0.1, 0.02, 0.2), **far}, conds)["outcome"] == "binds"
    esc = cap.outcome({"M2@4000": g(0.0, -0.05, 0.04), "M4@4000": g(0.3, 0.2, 0.4), **far}, conds)
    assert esc["outcome"] == "escapable" and esc["escapable_by"] == ["M2@4000"]
    heavy = cap.outcome({"M2@4000": g(0.0, -0.05, 0.04, ratio=0.3), "M4@4000": g(0.3, 0.2, 0.4)}, conds)
    assert heavy["outcome"] == "indeterminate"                                # escaping needs <= 1/4 of the tokens
    assert cap.outcome({"M2@4000": g(0.09, 0.02, 0.2), "M4@4000": g(0.3, 0.2, 0.4)}, conds)["outcome"] == \
        "indeterminate"
    assert cap.outcome({"M2@4000": {"est": 0.3, "ci": None, "token_ratio": 0.1}}, conds)["outcome"] == "indeterminate"
    assert cap.outcome(far, conds)["outcome"] == "no local representation"


def test_h_s_rates_and_verdict(tmp_path, panel, book):
    c1 = S.READERS_C[0]
    wrong = ("u1a", "u3a", "u4a", "u5b")
    per_cell = {(c1, S.M5, i): {"error_class": "E-cond"} for i in wrong}
    run = run_of(tmp_path, panel, book, {(c1, "M2@4000"): ALL, (c1, S.M5): ALL - set(wrong)}, per_cell=per_cell)
    grid = cap.Grid(observations([run], ITEMS, panel))
    assert (cap.HS_MAIN_SCOPE, cap.HS_CONTROL_SCOPE) == ("L2+L3", "S2")          # S2 is the control
    main = grid.h_s(c1, S.M5, "M2@4000", cap.HS_MAIN_SCOPE)
    control = grid.h_s(c1, S.M5, "M2@4000", cap.HS_CONTROL_SCOPE)
    assert main["n"] == 12 and main["rate_diff"]["est"] == round(4 / 12, 4)       # denominator: all cells
    assert main["share_diff"]["est"] is None                                  # zero-LLM arm has no wrong cell
    assert control["n"] == 2 and control["rate_diff"]["est"] == 0.0
    assert cap.h_s_verdict([main], [control], "rate_diff") == "holds"
    assert cap.h_s_verdict([main], [control], "share_diff") == "not evaluable"
    assert cap.h_s_verdict([main], [], "rate_diff") == "not evaluable"
    s2_wrong = {(c1, S.M5, "p1a"): {"error_class": "E-cond"}}
    run2 = run_of(tmp_path / "s2", panel, book, {(c1, "M2@4000"): ALL, (c1, S.M5): ALL - set(wrong) - {"p1a"}},
                  per_cell={**per_cell, **s2_wrong})
    grid2 = cap.Grid(observations([run2], ITEMS, panel))
    control2 = grid2.h_s(c1, S.M5, "M2@4000", cap.HS_CONTROL_SCOPE)
    assert control2["rate_diff"]["est"] == 0.5                                # S2 control differs by 50pp
    assert cap.h_s_verdict([grid2.h_s(c1, S.M5, "M2@4000", cap.HS_MAIN_SCOPE)], [control2], "rate_diff") == \
        "does not hold"
    report = analyse([run], ITEMS, panel, book, load_bands(ROOT / "data" / "analysis" / "energy_bands.json"))
    h = report["h_s"][0]
    assert h["pre_registered"] and h["main_scope"] == "L2+L3" and h["verdict"]["rate_diff"] == "holds"
    assert h["main"][0]["family"]["clusters"] == 2


def ruled(items: list[dict]) -> list[dict]:
    """ITEMS with the manual ruling fields on the abstracted units: U1, U2 ops, U3 market; U1 literal exception on its
    exception cell; U2's not_in_effect cell ruled not literal; U2 carries same_rule medium."""
    domain = {"U1": "ops", "U2": "technical", "U3": "market"}
    out = []
    for it in items:
        u = it.get("unit_id")
        if u in domain:
            it = dict(it, domain=domain[u], load_bearing=True,
                      k6_exception_literal="exception" if it["item_id"] == "u1b" else "none",
                      same_rule_confidence="medium" if u == "U2" else None)
        out.append(it)
    return out


def test_supply_counts_the_rulings_with_the_earlier_batch_fallback():
    """The L3 supply check counts the ops + technical abstracted units (load_bearing true), literal by
    k6_exception_literal; cells without the fields fall back (domain by document, literal by cell kind); cell_kind
    stays a diagnostic."""
    bare = cap.supply(ITEMS)["L3"]
    assert bare["units"] == 0 and bare["abstracted_units_other_domains"] == 3 and not bare["met"]   # no domain: none
    got = cap.supply(ruled(ITEMS))
    l3 = got["L3"]
    assert (l3["units"], l3["cells"], l3["families"], l3["max_units_per_doc"]) == (2, 4, 2, 1)
    assert l3["literal_units"] == 1 and l3["literal_share"] == 0.5 and l3["cell_kind_share"] == 1.0   # U2: NIE gold
    assert l3["literal_units_by_kind"] == {"exception": 1, "not_in_effect": 0}
    assert l3["abstracted_units_other_domains"] == 1 and l3["domain_fallback_units"] == 0
    assert l3["sensitivity"]["drop same_rule_medium"] == {"units": 1, "literal_units": 1, "literal_share": 1.0}
    assert got["S2"] == {"pairs": 1, "met": False}
    prior = [dict(it, item_id=f"prior-{it['item_id']}", unit_id="P9", doc="M-03") for it in ITEMS
             if it.get("unit_id") == "U1"]                                  # batch 1-2 cells: no ruling fields
    mixed = cap.supply(ruled(ITEMS) + prior)["L3"]
    assert mixed["units"] == 3 and mixed["literal_units"] == 2              # M-03 -> ops; its exception cell literal
    assert mixed["domain_fallback_units"] == 1 and mixed["literal_fallback_units"] == 1
    off = [dict(it, load_bearing=False) if it.get("unit_id") == "U2" else it for it in ruled(ITEMS)]
    held = [dict(it, admissible=False) if it.get("unit_id") == "U1" else it for it in off]
    left = cap.supply(held + [dict(ruled(ITEMS)[0], item_id="dev", split="dev", unit_id="U9")])
    assert left["L3"]["units"] == 0 and left["ineligible"] == 3 and left["items"] == 14


def test_supply_drops_pin_and_normative_medium_and_reads_the_pool_both_ways():
    """Whole units carrying pin_confidence / normative_confidence medium are dropped (a unit carries a tag when any of
    its cells does); the pool is read two ways: "drop outside_pool" (the original pool, read literally) drops every
    outside_pool unit, "drop outside_union_pool" keeps those whose exception sentence is in the extended pool
    (cue_pool extension). Supply units: U1 literal, U2 not, and copies of them: P1 (U1, pin medium on one cell), N1 (U2,
    normative medium), O1 (U1, outside_pool), E1 (U1, outside_pool + extension), X1 (U1, normative medium)."""
    base = ruled(ITEMS)

    def copy(unit: str, of: str, first_only: bool = False, **tags) -> list[dict]:
        cells = [it for it in base if it.get("unit_id") == of]
        return [dict(it, item_id=f"{unit}-{it['item_id']}", unit_id=unit, **({} if first_only and k else tags))
                for k, it in enumerate(cells)]
    pool = (base + copy("P1", "U1", first_only=True, pin_confidence="medium")
            + copy("N1", "U2", normative_confidence="medium") + copy("O1", "U1", cue_relation="outside_pool")
            + copy("E1", "U1", cue_relation="outside_pool", cue_pool="extension")
            + copy("X1", "U1", normative_confidence="medium"))
    l3 = cap.supply(pool)["L3"]
    assert (l3["units"], l3["literal_units"]) == (7, 5)
    got = {k: (v["units"], v["literal_units"]) for k, v in l3["sensitivity"].items()}
    assert got["drop pin_medium"] == (6, 4)                          # P1 out
    assert got["drop normative_medium"] == (5, 4)                    # N1, X1 out
    assert got["drop outside_pool"] == (5, 3)                        # O1, E1 out
    assert got["drop outside_union_pool"] == (6, 4)                  # O1 out, E1 kept
    assert l3["sensitivity"]["drop outside_union_pool"]["literal_share"] == round(4 / 6, 4)
    assert got["drop same_rule_medium"] == (5, 5)                    # U2, N1 (the U2 copy) out
    from procedures.analysis.spec import unit_tags
    tags = unit_tags(pool)
    assert tags["pin_medium"] == ["P1"] and tags["normative_medium"] == ["N1", "X1"]
    assert tags["outside_pool"] == ["E1", "O1"] and tags["outside_union_pool"] == ["O1"]


def test_the_spec_freezes_the_pin_normative_and_union_pool_drops_and_refuses_an_older_spec(panel):
    from procedures.analysis.spec import check_spec, make_paper_spec
    new = ("drop:pin_medium", "drop:normative_medium", "drop:outside_union_pool")
    spec = make_paper_spec(ITEMS, "synthetic.jsonl", "0" * 64, panel, {})
    assert set(new) <= {s["id"] for s in spec["sensitivities"]}
    assert check_spec(spec, ITEMS, panel)["missing"] == []
    older = dict(spec, sensitivities=[s for s in spec["sensitivities"] if s["id"] not in new])
    with pytest.raises(ValueError, match="made before they existed"):
        check_spec(older, ITEMS, panel)


def test_empty_cells_void_an_arm_only_above_ten_percent_and_deviations_are_listed(tmp_path, panel, book):
    c2 = S.READERS_C[1]
    empty = {"status": "unparseable", "error_class": "E-parse"}
    per_cell = {(c2, "B0@full", "u1a"): {"leak": True, "from_reasoning": True},
                (c2, "B0@full", "u2a"): {"from_reasoning": True}, (c2, "B0@full", "u3a"): empty,
                (c2, "M2@4000", "u1a"): empty, (c2, "M2@4000", "u4a"): empty}
    spec = {(S.READER_B, "M0@full"): ALL, (c2, "B0@full"): ALL - {"u3a"}, (c2, "M2@4000"): ALL - {"u1a", "u4a"}}
    run = run_of(tmp_path, panel, book, spec, per_cell=per_cell)
    grid = cap.Grid(observations([run], ITEMS, panel))
    b0, m2 = grid.conds[(c2, "B0@full")], grid.conds[(c2, "M2@4000")]
    assert b0.void is None and (b0.empty, b0.from_reasoning, b0.leaks) == (1, 2, 1)     # 1 / 16 empty: read
    assert m2.void.startswith("2 of 16 answered cells are empty") and m2.empty == 2      # 12.5% > 10%: void
    assert grid.accuracy(b0.key, "all")["est"] == 0.9375                                  # the empty cell is wrong
    report = analyse([run], ITEMS, panel, book, load_bands(ROOT / "data" / "analysis" / "energy_bands.json"))
    reasons = report["status"]["reasons"]
    assert any(r.startswith(f"void (not read): {c2} M2@4000") for r in reasons)
    assert not any(f"{c2} B0@full" in r for r in reasons)
    dev = {d["condition"]: d for d in report["status"]["panel_deviations"]}
    assert dev["B0@full"]["from_reasoning_field"] == 2 and dev["B0@full"]["reasoning_leaks"] == 1
    assert all(c["condition"] != "M2@4000" or c["profile_id"] != c2 for c in report["capability"])
    # the mapped arm and the reading without the mapping, side by side
    assert grid.unmapped_accuracy(b0.key, "all")["est"] == 0.8125                        # 2 mapped right -> wrong
    assert b0.void_unmapped.startswith("3 of 16 answered cells are empty")                # 18.75% > 10%
    (m,) = report["reasoning_field_mapping"]
    assert (m["condition"], m["from_reasoning_field"], m["empty_mapped"], m["empty_unmapped"]) == ("B0@full", 2, 1, 3)
    assert m["mapped"]["all"]["est"] == 0.9375 and m["unmapped"]["all"]["est"] == 0.8125
    assert m["void_mapped"] is None and m["void_unmapped"] is not None
    md = render_markdown(report, "x")
    assert "## Answer taken from the reasoning field: both readings" in md and "| 0.812 [" in md


def test_repeats_are_averaged(tmp_path, panel, book):
    c1 = S.READERS_C[0]

    def rows_of(run_id):
        return [S.row(run_id, c1, "M2@4000", it, True, repeat=0) for it in ITEMS] + \
               [S.row(run_id, c1, "M2@4000", it, it["item_id"] != "u1a", repeat=1) for it in ITEMS]
    run = load_run(S.write_run(tmp_path, rows_of, ITEMS, panel, book))
    grid = cap.Grid(observations([run], ITEMS, panel))
    assert grid.repeats == 2 and grid.values["correct"][(c1, "M2@4000")]["u1a"] == 0.5
    assert grid.accuracy((c1, "M2@4000"), "all")["est"] == round(15.5 / 16, 4)


def test_repeat_coverage_counts_the_named_conditions_only(tmp_path, panel, book):
    """The k = 1 warning (a result read only once) is withdrawn on the 4k slice alone, so repeat_coverage takes the
    conditions to count: condition strings as the observations name them ("{reader}" resolved per reader, a
    self-compressed M6) or a predicate on the Cond; None keeps every readable condition, as before."""
    c1 = S.READERS_C[0]

    def rows_of(run_id):
        twice = [(c1, "M2@4000"), (c1, S.M6)]
        return [S.row(run_id, pid, cond, it, True, repeat=r, query_tokens=10 if cond == S.M6 else None)
                for pid, cond in twice for r in (0, 1) for it in ITEMS] + \
               [S.row(run_id, pid, cond, it, True) for pid, cond in ((c1, "M2@16000"), (S.READER_B, "M0@full"))
                for it in ITEMS]
    grid = cap.Grid(observations([load_run(S.write_run(tmp_path, rows_of, ITEMS, panel, book))], ITEMS, panel))
    n = len(ITEMS)
    assert grid.repeat_coverage("all") == {"scope": "all", "cells": 4 * n, "repeated": 2 * n}
    assert grid.repeat_coverage("all", conditions=["M2@4000", "M6@4000/{reader}"]) == \
        {"scope": "all", "cells": 2 * n, "repeated": 2 * n}
    assert grid.repeat_coverage("all", conditions=("M2@4000", "M0@full")) == \
        {"scope": "all", "cells": 2 * n, "repeated": n}
    assert grid.repeat_coverage("all", conditions=lambda c: c.budget == 16000) == \
        {"scope": "all", "cells": n, "repeated": 0}
    assert grid.repeat_coverage("all", conditions=[]) == {"scope": "all", "cells": 0, "repeated": 0}
    with pytest.raises(TypeError, match="collection"):
        grid.repeat_coverage("all", conditions="M2@4000")


# ---------------------------------------------------------------------- costs


def test_build_attribution_is_by_request_identity_and_checked(tmp_path, panel, book):
    run = run_of(tmp_path / "a", panel, book, {(S.READER_B, S.M5): ALL})
    assert run.memory_requests_file.startswith("memory_requests_")
    obs = observations([run], ITEMS, panel)
    builds = build_costs([run], obs, book)
    m = builds["memories"][f"m5-{S.M5}-M-01"]
    assert (m["calls"], m["imported_calls"], m["tokens"]["total"]) == (2, 1, 950)
    assert m["tokens"]["input_total"] == 900 and m["tokens"]["output"] == 50 and m["ledger"] == f"build__{S.BUILDER}"
    assert m["usd"] == pytest.approx(m["usd_recorded"])
    assert builds["unattributed_calls"] == {run.run_id: {f"build__{S.BUILDER}": 0}}
    bill = builds["bill"][run.run_id][f"build__{S.BUILDER}"]
    assert (bill["calls"], bill["imported_records"]) == (1, 2)          # the first call of each revision is imported
    sus = {s["condition"]: s for s in sustainability(obs, builds["memories"], {})}[S.M5]
    per_query = (9 * 950 + 7 * 310) / 16                  # 9 items read M-01's memory, 7 read BPM-001's
    assert sus["build"]["per_query_total"] == per_query and sus["build"]["memories"] == 2
    assert sus["amortised_per_query"]["10"] == round(1050 + per_query / 10, 4)

    extra = dict(BUILD, **{"X-9@1": [(10, 1)]})
    run = run_of(tmp_path / "b", panel, book, {(S.READER_B, S.M5): ALL}, builds=extra)
    assert build_costs([run], observations([run], ITEMS, panel), book)["unattributed_calls"] == \
        {run.run_id: {f"build__{S.BUILDER}": 1}}

    lie = {(S.READER_B, S.M5, i): {"build_tokens": 999} for i in IDS if i.startswith(("u1", "u3", "u5", "p", "s3x"))}
    run = run_of(tmp_path / "c", panel, book, {(S.READER_B, S.M5): ALL}, per_cell=lie)
    with pytest.raises(ValueError, match="records 999 tokens"):
        build_costs([run], observations([run], ITEMS, panel), book)

    two = {(S.READER_B, S.M5): ALL, (S.READER_B, f"M5@8000/{S.BUILDER}"): ALL}      # two memories, one build
    run = run_of(tmp_path / "d", panel, book, two)
    both = build_costs([run], observations([run], ITEMS, panel), book)
    a, b = (both["memories"][f"m5-{c}-M-01"] for c in (S.M5, f"M5@8000/{S.BUILDER}"))
    assert a["tokens"] == b["tokens"] and a["calls"] == b["calls"] == 2                  # deployment view: each whole
    assert both["bill"][run.run_id][f"build__{S.BUILDER}"]["calls"] == 1                 # bill view: once

    run = run_of(tmp_path / "e", panel, book, {(S.READER_B, S.M5): ALL}, requests=False)
    with pytest.raises(ValueError, match="no memory request map"):
        build_costs([run], observations([run], ITEMS, panel), book)


def test_the_memory_request_map_is_found_under_its_plain_name(tmp_path, panel, book):
    """The released runs' maps are memory_requests.json: a run whose map was renamed to that name is read."""
    run = run_of(tmp_path, panel, book, {(S.READER_B, S.M5): ALL})
    (run.run_dir / run.memory_requests_file).rename(run.run_dir / "memory_requests.json")
    plain = load_run(run.run_dir)
    assert plain.memory_requests_file == "memory_requests.json" and plain.memory_requests == run.memory_requests


def test_m6_compression_cost_goes_to_its_question_and_compressor(tmp_path, panel, book, bands):
    c1 = S.READERS_C[0]
    pp = panel.get(c1)
    per_cell, recs, mapping = {}, [], {}
    for k, it in enumerate(ITEMS):
        key, h = f"m6-{it['item_id']}", f"q{k:063d}"
        q_in, q_out = 2000 + k, 100
        recs.append(S.call_record(pp, h, it["item_id"], f"query/M6@4000/{c1}", S.usage(q_in, q_out), book))
        mapping[key] = {"representation": "M6", "budget_tokens": 4000, "profile": c1, "ledger": f"query__{c1}",
                        "revisions": [[it["doc"], "1"]], "requests": [{"request_hash": h}]}
        per_cell[(c1, S.M6, it["item_id"])] = {"memory_key": key, "query_tokens": q_in + q_out}
    spec = {(S.READER_B, "M0@full"): ALL, (c1, "M2@4000"): ALL, (c1, S.M6): ALL}
    run = load_run(S.write_run(tmp_path, spec_rows(spec, per_cell), ITEMS, panel, book,
                               query_records={c1: recs}, requests=mapping))
    obs = observations([run], ITEMS, panel)
    assert {o.arm for o in obs if o.condition == S.M6} == {"C-built"}
    builds = build_costs([run], obs, book)
    assert builds["memories"]["m6-u1a"]["tokens"]["total"] == 2100
    report = analyse([run], ITEMS, panel, book, bands)
    sus = {s["condition"]: s for s in report["sustainability"] if s["profile_id"] == c1}[S.M6]
    mean_q = (sum(2000 + k for k in range(16)) / 16) + 100
    assert sus["query"]["per_query_total"] == mean_q and sus["build"] is None
    assert sus["amortised_per_query"]["10"] == sus["amortised_per_query"]["10000"] == round(1050 + mean_q, 4)
    acc = {a["condition"]: a for a in report["access"] if a["profile_id"] == c1}[S.M6]
    assert acc["usd_per_1k_query_memory"] > 0
    assert acc["usd_per_1k_at_N"]["10"] == pytest.approx(acc["usd_per_1k_read"] + acc["usd_per_1k_query_memory"],
                                                         abs=2e-4)
    lie = {**per_cell, (c1, S.M6, "u1a"): {"memory_key": "m6-u1a", "query_tokens": 1}}
    bad = load_run(S.write_run(tmp_path / "bad", spec_rows(spec, lie), ITEMS, panel, book,
                               query_records={c1: recs}, requests=mapping))
    with pytest.raises(ValueError, match="records 1 tokens"):
        build_costs([bad], observations([bad], ITEMS, panel), book)


def test_cells_not_sent_are_listed_as_absences_by_reason(tmp_path, panel, book, bands):
    """route_context (the cell is over the context of its reader's route) and route_timeout (the cell's M6
    compression timed out) are absences of one kind: counted per reader x condition by reason in the report and the
    access rows, not scored."""
    c1 = S.READERS_C[0]
    pp = panel.get(c1)
    timed = {"u1b", "u2a"}
    per_cell, recs, mapping = {(c1, "M2@4000", "u1a"): {"not_sent": "route_context"}}, [], {}
    for k, it in enumerate(ITEMS):
        key, h = f"m6-{it['item_id']}", f"q{k:063d}"
        mapping[key] = {"representation": "M6", "budget_tokens": 4000, "profile": c1, "ledger": f"query__{c1}",
                        "revisions": [[it["doc"], "1"]], "requests": [{"request_hash": h}]}
        if it["item_id"] in timed:      # its compression timed out: no answer, the reader cell is not sent
            per_cell[(c1, S.M6, it["item_id"])] = {"memory_key": key, "query_tokens": 0, "not_sent": "route_timeout"}
            continue
        recs.append(S.call_record(pp, h, it["item_id"], f"query/M6@4000/{c1}", S.usage(2000, 100), book))
        per_cell[(c1, S.M6, it["item_id"])] = {"memory_key": key, "query_tokens": 2100}
    spec = {(S.READER_B, "M0@full"): ALL, (c1, "M2@4000"): ALL, (c1, S.M6): ALL}
    run = load_run(S.write_run(tmp_path, spec_rows(spec, per_cell), ITEMS, panel, book, query_records={c1: recs},
                               requests=mapping))
    obs = observations([run], ITEMS, panel)
    assert {o.infeasible_reason for o in obs if not o.observed} == {"route_context", "route_timeout"}
    report = analyse([run], ITEMS, panel, book, bands)
    assert report["absences"] == {
        "by_condition": [{"profile_id": c1, "condition": "M2@4000", "not_sent": 1, "by_reason": {"route_context": 1}},
                         {"profile_id": c1, "condition": S.M6, "not_sent": 2, "by_reason": {"route_timeout": 2}}],
        "by_reason": {"route_context": 1, "route_timeout": 2}, "cells": 3}
    acc = {(a["profile_id"], a["condition"]): a for a in report["access"]}
    assert (acc[(c1, S.M6)]["not_sent_infeasible"], acc[(c1, S.M6)]["not_sent_by_reason"]) == (2, {"route_timeout": 2})
    assert acc[(c1, S.M6)]["feasible"] == 14 and acc[(S.READER_B, "M0@full")]["not_sent_by_reason"] == {}
    md = render_markdown(report, "x")
    assert "## Cells not sent" in md and "3 cells: route_context 1, route_timeout 2." in md


def test_one_price_book_for_every_dollar(tmp_path, panel, book, bands):
    spec = {(S.READER_B, "M0@full"): ALL, (S.READER_B, S.M5): ALL, (S.READERS_C[0], "M2@4000"): ALL}
    run = run_of(tmp_path, panel, book, spec)
    snap = analyse([run], ITEMS, panel, book, bands)
    assert snap["meta"]["price_book"]["same_as_ledgers"]
    for a in snap["access"]:
        assert a["usd_per_1k_read"] == a["usd_per_1k_read_recorded"]
    doc = json.loads(Path(DEFAULT_PRICE_BOOK).read_text(encoding="utf-8"))
    renamed = tmp_path / "renamed.json"            # the same prices in another file: the ledgers' prices
    renamed.write_text(json.dumps({**doc, "schema": "another.price_snapshot"}), encoding="utf-8")
    same = analyse([run], ITEMS, panel, load_price_book(renamed), bands)["meta"]["price_book"]
    assert same["same_as_ledgers"] and same["sha256"] != book.sha256 and same["ledgers_priced_on"] == [book.sha256]
    key = panel.get(S.READER_B).price_key
    doc["models"][key] = {**doc["models"][key], "in": 2 * doc["models"][key]["in"],
                          "out": 2 * doc["models"][key]["out"]}
    path = tmp_path / "corrected.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    corrected = analyse([run], ITEMS, panel, load_price_book(path), bands, book_label="corrected", billing="pass")
    assert not corrected["meta"]["price_book"]["same_as_ledgers"] and corrected["meta"]["price_book"]["label"] == \
        "corrected"
    got = {(a["profile_id"], a["condition"]): a for a in corrected["access"]}
    b_m0 = got[(S.READER_B, "M0@full")]
    assert b_m0["usd_per_1k_read"] == pytest.approx(2 * b_m0["usd_per_1k_read_recorded"], abs=2e-4)
    c1 = got[(S.READERS_C[0], "M2@4000")]
    assert c1["usd_per_1k_read"] == c1["usd_per_1k_read_recorded"]         # its own rate is unchanged
    b_m5 = got[(S.READER_B, S.M5)]
    assert b_m5["build_usd_per_memory"] is not None and set(b_m5["usd_per_1k_at_N"]) == {"10", "30", "100", "300",
                                                                                         "1000", "3000", "10000"}
    assert not any("billing reconciliation" in r for r in corrected["status"]["reasons"])
    assert any(r.startswith("billing reconciliation pending") for r in snap["status"]["reasons"])


def test_energy_bands(tmp_path, panel, book, bands):
    run = run_of(tmp_path, panel, book, {(c, "M2@4000"): ALL for c in S.READERS_C})
    obs = observations([run], ITEMS, panel)
    sus = sustainability(obs, {}, {})
    none = energy(sus, bands)
    assert all(not v["joules_per_query"] and not v["rank_statements"] for v in none["by_N"].values())
    assert set(none["by_N"]["10"]["unbanded"]) == {f"{c}|M2@4000" for c in S.READERS_C}
    doc = json.loads(Path(bands["path"]).read_text(encoding="utf-8"))
    doc["profiles"][S.READERS_C[0]].update(input={"low": 0.1, "high": 0.2}, output={"low": 1, "high": 2},
                                           source="synthetic")
    doc["profiles"][S.READERS_C[2]].update(input={"low": 1, "high": 2}, output={"low": 10, "high": 20},
                                           source="synthetic")
    path = tmp_path / "bands.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    out = energy(sus, load_bands(path))["by_N"]["100"]
    c1, c2, c3 = (f"{c}|M2@4000" for c in S.READERS_C)
    assert out["joules_per_query"][c1] == {"low": 150.0, "high": 300.0}
    assert out["rank_statements"] == [f"{c1} < {c3}"] and list(out["unbanded"]) == [c2]
    doc["profiles"][S.READERS_C[1]].update(input={"low": 0.15, "high": 0.3}, output={"low": 1.5, "high": 3},
                                           source="synthetic")          # 225-450 J: overlaps c1's 150-300 J
    path.write_text(json.dumps(doc), encoding="utf-8")
    out = energy(sus, load_bands(path))["by_N"]["100"]
    assert out["rank_statements"] == [f"{c1} < {c3}", f"{c2} < {c3}"] and not out["unbanded"]
    doc["profiles"][S.READERS_C[1]].update(input=None, output=None, source=None)
    for bad, match in (({"input": {"low": 2, "high": 1}, "source": "x"}, "low <= high"),
                       ({"input": {"low": 1, "high": 2}, "source": None}, "without a source")):
        doc["profiles"][S.READERS_C[1]].update(bad)
        path.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_bands(path)
        doc["profiles"][S.READERS_C[1]].update(input=None, source=None)
    path.write_text(json.dumps({**doc, "units": "Wh/token"}), encoding="utf-8")
    with pytest.raises(ValueError, match="J/token"):
        load_bands(path)


# ---------------------------------------------------------------------- taxonomy, self-check, report


def test_taxonomy_codec_rollup_and_polarity_overlay(tmp_path, panel, book):
    c1 = S.READERS_C[0]
    cells = {"u1a": {"status": "abstained", "error_class": "E-abs", "pred": "insufficient_evidence"},
             "u1b": {"error_class": "E-exc"}, "u3a": {"error_class": "E-cond"}, "u3b": {"error_class": "E-ver"},
             "u4a": {"error_class": "E-unit"}, "u4b": {"error_class": "E-other", "other_subclass": "value_in_doc"},
             "u5a": {"error_class": "E-other", "other_subclass": "value_not_in_doc"},
             "u5b": {"status": "unparseable", "error_class": "E-parse"},
             "u6a": {"status": "ambiguous", "error_class": None},
             "u2b": {"error_class": "E-other", "other_subclass": "value_in_doc", "pred": "B"},
             "p1a": {"error_class": "E-ver", "pred": "not_in_effect"},
             "p1b": {"error_class": "E-cond", "pred": "C", "matched_option": "none of these"}}
    run = run_of(tmp_path, panel, book, {(c1, "M2@4000"): ALL - set(cells)},
                 per_cell={(c1, "M2@4000", i): kw for i, kw in cells.items()})
    t = taxonomy(observations([run], ITEMS, panel))[0]
    assert (t["n"], t["correct"], t["pending_arbitration"]) == (16, 4, 1)
    assert t["frozen"]["E-other/value_in_doc"] == 2 and t["frozen"]["E-parse"] == 1
    assert t["codec"] == {"omission": 1, "weakening": None, "mutation": 3, "scope": 3, "temporal-decision": 2,
                          "hallucinated commitment": 1, "format (not a Codec class)": 1}
    assert t["polarity_flips"] == 3           # u2b (NIE gold, a letter), p1a (not_in_effect), p1b (none of these)
    run2 = run_of(tmp_path / "nie", panel, book, {(c1, "M2@4000"): ALL - {"u2b"}},
                  per_cell={(c1, "M2@4000", "u2b"): {"c_on_nie": True, "pred": "C"}})
    t2 = taxonomy(observations([run2], ITEMS, panel))[0]
    assert (t2["polarity_flips"], t2["c_on_nie"]) == (0, 1)       # "none of these" on a NIE cell is not a flip


def test_arbitrated_ambiguous_rows_are_not_pending(tmp_path, panel, book, bands):
    """The frozen scorer keeps an arbitrated row's status "ambiguous" and sets correct or an error class: only the
    unresolved one (neither) awaits arbitration, in the status block and in the taxonomy alike."""
    b = S.READER_B
    arbitrated = {"u1a": {"status": "ambiguous"},                                        # arbitrated correct
                  "u1b": {"status": "ambiguous", "error_class": "E-other", "other_subclass": "value_not_in_doc"}}
    queued = {**arbitrated, "u3a": {"status": "ambiguous", "error_class": None}}         # still in the queue
    for sub, cells, pending in (("arb", arbitrated, 0), ("queued", queued, 1)):
        spec = {(b, "M0@full"): ALL - set(cells) | {"u1a"}}
        run = run_of(tmp_path / sub, panel, book, spec, per_cell={(b, "M0@full", i): kw for i, kw in cells.items()})
        got = analyse([run], ITEMS, panel, book, bands)
        assert got["status"]["arbitration_pending"] == pending
        assert got["taxonomy"]["all"][0]["pending_arbitration"] == pending
        assert any("await blind arbitration" in r for r in got["status"]["reasons"]) == bool(pending)
        assert got["taxonomy"]["all"][0]["frozen"]["E-other/value_not_in_doc"] == 1
        assert got["taxonomy"]["all"][0]["status"]["ambiguous"] == len(cells)
    wrong = run_of(tmp_path / "bad", panel, book, {(b, "M0@full"): ALL - {"u1b"}},
                   per_cell={(b, "M0@full", "u1b"): {"status": "abstained", "error_class": None}})
    with pytest.raises(ValueError, match="without an error class"):
        analyse([wrong], ITEMS, panel, book, bands)


def test_self_check_against_the_run_summary(tmp_path, panel, book, bands):
    spec = {(S.READER_B, "M2@4000"): ALL - {"u1a"}}
    ok = run_of(tmp_path / "ok", panel, book, spec)
    assert analyse([ok], ITEMS, panel, book, bands)["meta"]["self_checks"]["summary_conditions_matched"] == 1
    wrong = {"conditions": {f"{S.READER_B}|read/M2@4000": {"capability": {"n": 16, "correct": 16,
                                                                          "acc_chance_corrected": 1.0}}}}
    bad = load_run(S.write_run(tmp_path / "bad", spec_rows(spec), ITEMS, panel, book, summary_override=wrong))
    with pytest.raises(ValueError, match="run summary"):
        analyse([bad], ITEMS, panel, book, bands)


def test_report_is_deterministic_and_rendered(tmp_path, panel, book, bands):
    run = run_of(tmp_path / "run", panel, book, base_spec())
    one = analyse([run], ITEMS, panel, book, bands)
    assert report_bytes(one) == report_bytes(analyse([run], ITEMS, panel, book, bands))
    assert one["status"]["label"] == "AMBER" and one["status"]["reasons"][0].startswith("k = 1")
    assert "supply check counted on the analysed items only: no item pool was given" in one["status"]["reasons"]
    pooled = analyse([run], ITEMS, panel, book, bands, pool=ITEMS + [dict(ITEMS[0], item_id="extra", unit_id="U7")],
                     pool_sources=[{"path": "synthetic.jsonl", "sha256": "0" * 64}])
    assert pooled["status"]["supply"]["pool"]["items"] == 17 and pooled["status"]["supply"]["analysed"]["items"] == 16
    assert pooled["status"]["supply"]["population"] is None and one["primary"] is None
    assert not any(r.startswith("supply check counted on the analysed") for r in pooled["status"]["reasons"])
    assert all("family" in g for g in one["gap"]) and all("family" in p for p in one["dgap"])
    assert one["meta"]["reference"] == {"profile_id": S.READER_B, "condition": "M0@full", "readable": True}
    none = analyse([run], ITEMS, panel, book, bands, reference=S.READERS_C[0])
    assert not none["gap"] and any(r.startswith("no readable reference") for r in none["status"]["reasons"])
    assert one["outcomes"]["all"]["readers"][S.READERS_C[0]]["outcome"] == "escapable"      # M2 equals the reference
    assert one["outcomes"]["all"]["tiers"][cap.CONSTRAINED_TIER]["escapable_by"] == ["M2@4000"]
    assert {p["scope"] for p in one["dgap"]} == {"L3-abstracted", "L3", "S2+L2+L3", "all"}
    json_path, md_path = write_report(one, tmp_path / "out", "syn")
    assert write_report(one, tmp_path / "out", "syn") == (json_path, md_path)
    md = md_path.read_text(encoding="utf-8")
    for heading in ("## Supply check", "## Capability", "## Gap", "## ΔGap", "## H_S", "## Sustainability",
                    "### LLM-built memory calls, bill view", "## Access", "## Energy", "## Error taxonomy"):
        assert heading in md
    assert md == render_markdown(one, json_path.stem)
    assert json.loads(json_path.read_text(encoding="utf-8")) == json.loads(report_bytes(one))
    with pytest.raises(ValueError, match="billing"):
        analyse([run], ITEMS, panel, book, bands, billing="maybe")


def test_rows_file_of_a_rescoring_is_read_when_named(tmp_path, panel, book, bands):
    spec = {(S.READER_B, "M2@4000"): ALL - {"u1a"}}
    run = run_of(tmp_path, panel, book, spec)
    rescored = [dict(r, capability=dict(r["capability"], correct=True, error_class=None))
                if r["item_id"] == "u1a" else r for r in run.rows]
    (run.run_dir / "rows_rescored.jsonl").write_bytes(S.jsonl_bytes(rescored))
    again = load_run(run.run_dir, rows="rows_rescored.jsonl")
    assert again.rows_file == "rows_rescored.jsonl" and again.summary is None
    got = analyse([again], ITEMS, panel, book, bands)
    assert got["meta"]["self_checks"]["runs_without_summary"] == [run.run_id]
    assert got["capability"][0]["scopes"]["all"]["est"] == 1.0
    with pytest.raises(ValueError, match="summary file is given only with"):
        load_run(run.run_dir, summary="summary_x.json")
