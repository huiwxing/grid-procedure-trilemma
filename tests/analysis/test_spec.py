"""The executable analysis spec of the paper (procedures.analysis.spec): the counts it must reproduce on the released
item files, and on synthetic runs the ruling-tag wiring into observations, the constrained panel (tier C_load_bearing:
the readers of access tiers C1 and C2 pooled; C3 apart), the named primary comparisons with their sensitivities, the
spec checks, the k = 1 rule (a result read only once is marked) and the empty-memory count. Zero network: sockets are
blocked."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket

import pytest

import analysis_synth as S
from procedures.analysis import capability as cap
from procedures.analysis.costs import load_bands
from procedures.analysis.data import load_run, observations
from procedures.analysis.report import _k1_scope, _populations, _primary, analyse, render_markdown
from procedures.analysis.spec import COMPARISON_KINDS, CONSTRAINED_TIER, RULED_FIELDS, SCHEMA, check_spec, in_population, load_spec, make_paper_spec, tier_group, write_spec
from procedures.readers import load_panel, load_price_book

ROOT = Path(__file__).resolve().parents[2]
ITEMS_DIR = ROOT / "data" / "items"
# the final item files (with the ruling confidences and the cue pool of every cell) and the spec made from them
FULL = ITEMS_DIR / "questions_with_earlier_cells.jsonl"
SUBSET = ITEMS_DIR / "questions.jsonl"
PAPER_SPEC = ROOT / "data" / "analysis" / "spec.json"          # the paper's spec
C3_ACCESS_C1 = "C3_gptoss20b_hf_deepinfra_low"           # slot C3, access tier C1: in the constrained tier


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise RuntimeError("network blocked in tests")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture(scope="module")
def panel():
    return load_panel()


@pytest.fixture(scope="module")
def book():
    return load_price_book()


@pytest.fixture(scope="module")
def bands():
    return load_bands(ROOT / "data" / "analysis" / "energy_bands.json")


def read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def lf_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ---------------------------------------------------------------------- the real files


def test_population_is_158_cells_79_units_on_both_item_files():
    ids = {}
    for path in (FULL, SUBSET):
        pop = [it for it in read(path) if in_population(it)]
        assert (len(pop), len({it["unit_id"] for it in pop})) == (158, 79)
        ids[path.name] = sorted(it["item_id"] for it in pop)
    assert ids[FULL.name] == ids[SUBSET.name]


def test_tiers_pool_c1_c2_and_keep_c3_apart(panel):
    from procedures.analysis.spec import tiers
    got = tiers(panel)
    assert got["B"] == sorted(["B1_dsv41flash_hf_deepinfra", "B2_qwen3_235b_2507_hf_deepinfra",
                               "tierB_llama33_70b_hf_ovhcloud"])
    assert got[CONSTRAINED_TIER] == sorted(["C1_llama31_8b_hf_deepinfra", "C2_qwen3_8b_hf_nscale_nothink",
                                             C3_ACCESS_C1, "tierC2_qwen3_32b_hf_nscale_nothink"])
    assert got["C3"] == ["tierC3_gptoss120b_hf_deepinfra_low"]
    assert all(panel.get(p).access_tier in ("C1", "C2") for p in got[CONSTRAINED_TIER])


def test_k6_reproduces_the_final_rulings_82_43_and_79_43():
    """The final rulings of the supply / power check (L3) on the full item file, which holds every admitted L3 batch:
    82 units, 43 literal, 7 families, at most 5 units per document, with the reconciliation sensitivities and the
    pool read both ways (the ruled earlier-batch units carry load_bearing false and are outside the supply count
    either way); and
    79 / 43 on the primary population (the 3 earlier-batch supply units are outside it; the original pool drops 10 units,
    the union pool 2); the old cell_kind count is the diagnostic (44 / 82 and 44 / 79)."""
    k6 = cap.supply(read(FULL))["L3"]
    assert (k6["units"], k6["literal_units"], k6["families"], k6["max_units_per_doc"]) == (82, 43, 7, 5)
    assert k6["met"] and k6["domain_fallback_units"] == 3 and k6["literal_fallback_units"] == 3
    got = {k: (v["units"], v["literal_units"]) for k, v in k6["sensitivity"].items()}
    assert got == {"drop same_rule_medium": (70, 40), "k6_medium not literal": (82, 30), "drop pin_medium": (79, 42),
                   "drop normative_medium": (74, 39), "exemption_outcome not literal": (82, 20),
                   "drop outside_pool": (72, 33), "drop outside_union_pool": (80, 41),
                   "one unit per template_family": (80, 41), "drop owner_restates_inconsistent": (81, 42)}
    assert k6["cell_kind_share"] == round(44 / 82, 4)
    for path in (FULL, SUBSET):
        pop = cap.supply([it for it in read(path) if in_population(it)])["L3"]
        assert (pop["units"], pop["literal_units"], pop["cell_kind_share"]) == (79, 43, round(44 / 79, 4))
        assert {k: pop["sensitivity"][k]["units"] for k in ("drop outside_pool", "drop outside_union_pool")} == {
            "drop outside_pool": 69, "drop outside_union_pool": 77}


ANALYSIS_FIELDS = ("population", "reference", "tiers", "load_bearing_tier", "server_tier", "intermediate_tier",
                   "reference_arms", "builders", "comparisons", "comparison_kinds", "sensitivities", "repeats",
                   "status_rules", "supply")


def test_the_paper_spec_follows_from_the_item_file(panel):
    """The paper's spec, made before any result of the main run was read: the analysis the code derives from the full
    item file (population, tiers, comparisons, sensitivities, repeats, supply check) is the one the spec holds; the
    244-cell subset holds all of its population."""
    items = read(FULL)
    spec = load_spec(PAPER_SPEC)
    population = [it for it in items if in_population(it)]
    supply_check = {"item_file": cap.supply(items)["L3"], "population": cap.supply(population)["L3"]}
    again = make_paper_spec(items, spec["items"]["path"], lf_sha(FULL), panel, supply_check)
    assert {f: again[f] for f in ANALYSIS_FIELDS} == {f: spec[f] for f in ANALYSIS_FIELDS}
    assert [c["id"] for c in spec["comparisons"]] == PAPER_IDS and len(spec["sensitivities"]) == 18
    assert spec["tiers"]["B"] == sorted(["B1_dsv41flash_hf_deepinfra", "B2_qwen3_235b_2507_hf_deepinfra",
                                         "tierB_llama33_70b_hf_ovhcloud"]) and len(spec["tiers"][CONSTRAINED_TIER]) == 4
    assert check_spec(spec, read(SUBSET), panel)["missing"] == []


def grid_conditions(grid_names) -> dict[str, bool]:
    """Analysis-side condition name (the reader stage without "read/"; a self-compressed M6 as "{reader}") -> whether
    procedures.matrix.run_matrix puts it in the 4k slice, over the named grids and every reader of each."""
    from procedures.matrix.grids import GRIDS
    from procedures.readers.cells import BUILT_ONCE, stage_name
    from procedures.matrix.run_matrix import in_k1_slice, parse_condition
    out = {}
    for name in grid_names:
        grid = GRIDS[name]
        for cond in map(parse_condition, grid.conditions):
            for reader in cond.readers or grid.readers:
                maker = grid.builder if cond.representation in BUILT_ONCE else\
                    (cond.compressor or reader if cond.representation == "M6" else None)
                stage = stage_name(cond.representation, cond.budget_tokens, maker).removeprefix("read/")
                out[stage.replace(f"/{reader}", "/{reader}") if cond.representation == "M6" else stage] =\
                    in_k1_slice(cond)
    return out


def test_the_k1_scope_of_the_paper_spec_is_the_run_side_4k_slice_of_the_papers_runs():
    """The conditions the paper's spec counts for withdrawing the k = 1 warning are exactly the conditions the reader
    grids of the paper's runs (main, cbuilt_*) re-read on the whole population
    (procedures.matrix.run_matrix.in_k1_slice), in the same names; every condition a primary comparison names is read
    by one of those grids (the cloud comparator's by cloud_reference)."""
    from procedures.analysis.spec import k1_scope_conditions, paper_comparisons, spec_conditions
    tiered = grid_conditions(("main", "cbuilt_gptoss20b", "cbuilt_qwen32b"))
    assert sorted(k1_scope_conditions()) == sorted(c for c, sliced in tiered.items() if sliced)
    assert len(k1_scope_conditions()) == 10 == len(set(k1_scope_conditions()))
    named = spec_conditions({"comparisons": paper_comparisons(), "repeats": {}})
    assert set(named) <= set(tiered) | set(grid_conditions(("cloud_reference",)))


def test_template_family_clusters_collapse_79_units_to_77(tmp_path, panel, book):
    pop = [it for it in read(FULL) if in_population(it)]
    rows_of = lambda run_id: [S.row(run_id, S.READER_B, "M2@4000", it, True) for it in pop]  # noqa: E731
    run = load_run(S.write_run(tmp_path, rows_of, pop, panel, book))
    grid = cap.Grid(observations([run], pop, panel))
    key = (S.READER_B, "M2@4000")
    assert grid.accuracy(key, "L3")["clusters"] == 79
    assert grid.accuracy(key, "L3", by="template_family")["clusters"] == 77
    assert grid.accuracy(key, "L3", by="document")["clusters"] == 50


# ---------------------------------------------------------------------- synthetic


ITEMS = S.main_run_items()
POP = S.POPULATION
M2, M5 = "M2@4000", S.M5


def rows(cells: dict, repeats: dict | None = None):
    """cells: (profile, condition) -> the wrong item ids; repeats: (profile, condition) -> item ids read twice."""
    repeats = repeats or {}

    def rows_of(run_id):
        out = []
        for (pid, cond), wrong in cells.items():
            for it in ITEMS:
                kw = {"build_tokens": 10, "memory_key": f"k-{cond}-{it['doc']}"} if cond == M5 else {}
                kw["memory_empty"] = (cond == M2 and it["item_id"] == "u1a")
                if cond == "M0@full":
                    kw["tokens"] = (80_000, 50)
                out.append(S.row(run_id, pid, cond, it, it["item_id"] not in wrong, **kw))
                if it["item_id"] in repeats.get((pid, cond), ()):
                    out.append(S.row(run_id, pid, cond, it, it["item_id"] not in wrong, repeat=1, **kw))
        return out
    return rows_of


def cells_of(extra=None) -> dict:
    cells = {(S.READER_B, "M0@full"): set(), (S.READER_B, M2): set(), (S.READER_B, M5): {"u2b"}}
    for pid in (*S.READERS_C, S.READER_C3):
        cells[(pid, M2)] = set()
        cells[(pid, M5)] = {"u1a"} if pid == S.READERS_C[0] else set()
    cells[(S.READER_C3, M5)] = {"u1a", "u1b", "u2a", "u2b"}                  # C3 would drag C down if pooled
    return {**cells, **(extra or {})}


def spec_of(panel) -> dict:
    return make_paper_spec(ITEMS, "synthetic.jsonl", "0" * 64, panel, {})


def run_with(tmp_path, panel, book, cells, repeats=None):
    records = S.build_calls(panel, book, {"M-01@1": [(5, 5)], "BPM-001@1": [(5, 5)]})     # 10 tokens per memory
    mapping = S.m5_requests(records, {f"k-{M5}-M-01": "M-01@1", f"k-{M5}-BPM-001": "BPM-001@1"})
    return load_run(S.write_run(tmp_path, rows(cells, repeats), ITEMS, panel, book, build_records=records,
                                requests=mapping))


def test_observations_carry_the_ruling_tags_and_access_tier(tmp_path, panel, book):
    run = load_run(S.write_run(tmp_path, rows({(S.READER_C3, M2): set()}), ITEMS, panel, book))
    obs = {o.item_id: o for o in observations([run], ITEMS, panel)}
    assert obs["u1a"].template_family == "twin" and obs["u2a"].same_rule_confidence == "medium"
    assert obs["u1a"].analysis_stratum == "primary" and obs["u1a"].domain == "ops" and obs["u1a"].load_bearing
    assert obs["u1b"].k6_exception_literal == "exception" and obs["u1a"].unit_id == "U1" and obs["u1a"].doc == "M-01"
    assert obs["u4a"].analysis_stratum == "earlier_batches" and all(getattr(obs["u4a"], f) is None
                                                               for f in RULED_FIELDS if f != "analysis_stratum")
    assert obs["p1a"].unit_id is None and obs["p1a"].analysis_stratum == "controls"
    assert obs["u1a"].access_tier == "C3" and tier_group("C", obs["u1a"].access_tier) == "C3"
    assert obs["u1a"].memory_empty and not obs["u1b"].memory_empty


def test_spec_population_tiers_and_checks(panel):
    spec = spec_of(panel)
    assert spec["population"]["item_ids"] == sorted(POP) and spec["population"]["units"] == 2
    u1a = next(it for it in ITEMS if it["item_id"] == "u1a")
    assert in_population(u1a)
    for change in ({"load_bearing": False}, {"rule2_spirit": "transparent"}, {"analysis_stratum": "earlier_batches"},
                   {"domain": "market"}, {"load_bearing": None}):
        assert not in_population({**u1a, **change}), change
    assert spec["population"]["template_families"] == {"twin": ["U1"]}                 # U3 is outside
    drops = {s["id"]: s["units"] for s in spec["sensitivities"] if s["kind"] == "drop_units"}
    assert drops["drop:same_rule_medium"] == ["U2"] and drops["leave_out:PJM"] == ["U1"]
    assert S.READER_C3 in spec["tiers"]["C3"] and S.READER_C3 not in spec["tiers"][CONSTRAINED_TIER]
    assert check_spec(spec, ITEMS, panel) == {"present": sorted(POP), "missing": []}
    partial = [it for it in ITEMS if it["item_id"] != "u1b"]           # a partial run: U1's tagged cell is absent
    assert check_spec(spec, partial, panel)["missing"] == ["u1b"]
    untagged = [dict(it, subtype=None) if it["item_id"] == "u1b" else it for it in ITEMS]
    with pytest.raises(ValueError, match="unit tag exemption_outcome"):
        check_spec(spec, untagged, panel)
    drift = [dict(it, same_rule_confidence="medium") if it["item_id"] == "u1a" else it for it in ITEMS]
    with pytest.raises(ValueError, match="unit tag same_rule_medium"):
        check_spec(spec, drift, panel)
    moved = [dict(it, domain="market") if it["item_id"] == "u2a" else it for it in ITEMS]
    with pytest.raises(ValueError, match="disagree with the spec's population"):
        check_spec(spec, moved, panel)
    extra = [dict(it, domain="ops") if it.get("unit_id") == "U3" else it for it in ITEMS]
    with pytest.raises(ValueError, match="outside the spec"):
        check_spec(spec, extra, panel)


def test_a_written_spec_reads_back_under_any_name(tmp_path, panel):
    path = write_spec(spec_of(panel), tmp_path)
    assert load_spec(path) == spec_of(panel) and write_spec(spec_of(panel), tmp_path) == path
    renamed = tmp_path / "analysis_spec.json"
    renamed.write_bytes(path.read_bytes())
    assert load_spec(renamed) == spec_of(panel)


def test_primary_comparisons_and_sensitivities(tmp_path, panel, book, bands):
    """Population u1a..u2b. B: M5 misses u2b -> dGap -0.25. C_load_bearing (the three access-C1 readers): C1 Llama
    misses u1a -> tier mean -1/12; the access-C3 reader misses all four but stays out. Interaction = -1/12 + 1/4."""
    run = run_with(tmp_path, panel, book, cells_of())
    report = analyse([run], ITEMS, panel, book, bands, spec=spec_of(panel))
    got = {p["id"]: p for p in report["primary"]}
    assert got["P0-B1"]["estimate"]["est"] == 1.0 and got["P0-B1"]["estimate"]["n"] == 4
    assert got["dGap-M5-B"]["estimate"]["est"] == -0.25
    c = got[f"dGap-M5-{CONSTRAINED_TIER}"]["estimate"]
    assert c["est"] == round(-1 / 12, 4) and c["models"] == sorted(S.READERS_C)
    assert got["interaction-M5"]["estimate"]["est"] == round(-1 / 12 + 1 / 4, 4)
    assert got["dGap-M7-B"]["estimate"] is None and got["dGap-M6-B"]["estimate"] is None       # not in this run
    sens = got["dGap-M5-B"]["sensitivities"]
    assert sens["drop:same_rule_medium"]["est"] == 0.0 and sens["drop:same_rule_medium"]["n"] == 2   # U2 out
    assert sens["leave_out:MISO"]["est"] == 0.0 and sens["leave_out:PJM"]["n"] == 2
    assert sens["clusters:document"]["clusters"] == 2 and sens["clusters:template_family"]["clusters"] == 2
    assert sens["unmapped"]["est"] == -0.25
    c3 = [g for g in report["gap_tier"] if g["tier"] == "C3"]
    assert c3 and all(g["models"] == [S.READER_C3] for g in c3)                            # reported on its own
    assert report["meta"]["load_bearing"]["scope"] == "primary"
    assert report["status"]["supply"]["population"]["L3"]["units"] == 2
    md = render_markdown(report, "syn")
    assert "## Primary comparisons" in md and "(exploratory)" in md and "interaction-M5" in md


def test_k1_warning_is_withdrawn_only_when_every_population_cell_is_repeated(tmp_path, panel, book, bands):
    cells = {(S.READER_B, "M0@full"): set(), (S.READER_B, M2): set()}
    partly = run_with(tmp_path / "a", panel, book, cells, repeats={(S.READER_B, M2): POP})
    reasons = analyse([partly], ITEMS, panel, book, bands, spec=spec_of(panel))["status"]["reasons"]
    assert reasons[0].startswith("k = 1: 4 of 8 answered cells of primary")
    every = run_with(tmp_path / "b", panel, book, cells, repeats={k: POP for k in cells})
    report = analyse([every], ITEMS, panel, book, bands, spec=spec_of(panel))
    assert not any(r.startswith("k = 1") for r in report["status"]["reasons"])
    cov = report["status"]["repeat_coverage"]
    assert (cov["scope"], cov["cells"], cov["repeated"]) == ("primary (4k slice)", 8, 8)


def test_empty_memories_are_counted_per_condition(tmp_path, panel, book, bands):
    run = run_with(tmp_path, panel, book, {(S.READER_B, M2): set(), (S.READERS_C[0], M2): set()})
    empty = analyse([run], ITEMS, panel, book, bands)["empty_memory"]
    assert empty["cells"] == 2 and empty["distinct_read_requests"] == 2
    assert {(r["profile_id"], r["memory_empty"]) for r in empty["by_condition"]} == {(S.READER_B, 1),
                                                                                      (S.READERS_C[0], 1)}


# ---------------------------------------------------------------------- the paper's comparisons, arms and rules

PAPER_IDS = ["P0-B1", "P0-A", "REF-B", f"OUTCOME-{CONSTRAINED_TIER}", f"OUTCOME-{CONSTRAINED_TIER}-A",
            "dGap-M5-B", f"dGap-M5-{CONSTRAINED_TIER}", "dGap-M7-B", f"dGap-M7-{CONSTRAINED_TIER}", "dGap-M6-B",
            f"dGap-M6-{CONSTRAINED_TIER}", "interaction-M5", "interaction-M7", "H_M7", "H_S",
            "CBUILT-gptoss20b", "CBUILT-qwen32b"]
B_DIRECT = S.BUILDER
QWEN32 = S.READER_C2
C_LB = (*S.READERS_C, QWEN32)
SLICE = ["B0@full", "M0@full", "M2@4000", "M4@4000", f"M5@4000/{B_DIRECT}", f"M7@4000/{B_DIRECT}",
         f"M6@4000/{B_DIRECT}", "M6@4000/{reader}", f"M7@4000/{S.READERS_C[2]}", f"M7@4000/{QWEN32}"]


def test_paper_spec_schema_comparisons_and_rules(panel):
    spec = spec_of(panel)
    assert spec["schema"] == SCHEMA and spec["reference"]["condition"] == "M0@full"
    assert [c["id"] for c in spec["comparisons"]] == PAPER_IDS and len(PAPER_IDS) == 17
    assert {c["kind"] for c in spec["comparisons"]} == set(COMPARISON_KINDS) == set(spec["comparison_kinds"])
    assert spec["reference_arms"] == {"A": [S.READER_A]}
    assert all(S.READER_A not in pids for pids in spec["tiers"].values())             # A is in no tier
    assert spec["builders"] == {"B-built": B_DIRECT, "C-built": [S.READERS_C[2], QWEN32]}
    assert spec["repeats"]["k1_scope_conditions"] == SLICE
    assert "k1_scope_conditions" in spec["repeats"]["k1_warning_withdrawn_when"]
    assert spec["status_rules"]["min_models_per_tier"] == 3 and "sensitivity_callout" in spec
    got = {c["id"]: c for c in spec["comparisons"]}
    assert got["REF-B"]["candidates"] == ["M0@full", "M2@4000"] and got["P0-A"]["k1"] is True
    assert got[f"OUTCOME-{CONSTRAINED_TIER}"]["reference"] == "REF-B"
    assert got[f"OUTCOME-{CONSTRAINED_TIER}-A"]["reference"] == "P0-A"
    assert got["dGap-M6-B"]["to"] == f"M6@4000/{B_DIRECT}"
    assert got[f"dGap-M6-{CONSTRAINED_TIER}"]["to"] == "M6@4000/{reader}"
    # two interactions (M5, M7; B1-built, both tiers); the M6 one is exploratory
    inter = [c for c in spec["comparisons"] if c["kind"] == "interaction"]
    assert [(c["id"], c["to"]) for c in inter] == [("interaction-M5", f"M5@4000/{B_DIRECT}"),
                                                   ("interaction-M7", f"M7@4000/{B_DIRECT}")]
    assert (got["CBUILT-qwen32b"]["from"], got["CBUILT-qwen32b"]["to"]) == (f"M7@4000/{QWEN32}", f"M7@4000/{B_DIRECT}")
    assert got["H_S"]["main_scope"] == "primary" and got["H_S"]["control_scope"] == "S2"
    assert all(got[i]["tier"] == CONSTRAINED_TIER for i in ("H_M7", "H_S", "CBUILT-gptoss20b", "CBUILT-qwen32b"))
    assert check_spec(spec, ITEMS, panel)["missing"] == []


def test_check_spec_checks_the_reference_arm_and_condition_names(panel):
    spec = spec_of(panel)
    with pytest.raises(ValueError, match="reference arm"):
        check_spec(dict(spec, reference_arms={"A": [S.READER_B]}), ITEMS, panel)
    bad = [dict(c, to="M5 at 4k") if c["id"] == "dGap-M5-B" else c for c in spec["comparisons"]]
    with pytest.raises(ValueError, match="no reader stage"):
        check_spec(dict(spec, comparisons=bad), ITEMS, panel)
    stray = dict(spec, repeats=dict(spec["repeats"], k1_scope_conditions=[*SLICE, "M7@4000/no_such_builder"]))
    with pytest.raises(ValueError, match="no panel profile"):
        check_spec(stray, ITEMS, panel)


def conf_cells() -> dict:
    """B1: M0@full misses u2b (0.75 on the population), M2@4000 all right (1.0): REF-B = M2@4000. The C readers as in
    cells_of, Qwen3-32B added, M4@4000 all right; the A arm all right; B1 M2@16000 (outside the 4k slice)."""
    cells = cells_of({(S.READER_B, "M0@full"): {"u2b"}, (S.READER_B, "B0@full"): set(POP),
                      (S.READER_B, "M2@16000"): set(), (QWEN32, M2): set(), (QWEN32, M5): set()})
    for pid in C_LB:
        cells[(pid, "M4@4000")] = set()
    for cond in ("B0@full", "M0@full", M2):
        cells[(S.READER_A, cond)] = set()
    return cells


def slice_repeats(cells: dict, skip=()) -> dict:
    return {(pid, cond): POP for pid, cond in cells
            if pid != S.READER_A and cond in SLICE and (pid, cond) not in skip}


def test_paper_report_ref_b_outcomes_k1_scope_amber_and_callout(tmp_path, panel, book, bands):
    cells = conf_cells()
    run = run_with(tmp_path, panel, book, cells, repeats=slice_repeats(cells))
    report = analyse([run], ITEMS, panel, book, bands, spec=spec_of(panel))
    got = {p["id"]: p for p in report["primary"]}
    assert [p["id"] for p in report["primary"]] == PAPER_IDS
    ref = got["REF-B"]["estimate"]
    assert ref["chosen"] == "M2@4000" and ref["est"] == 1.0                    # higher than M0@full's 0.75
    assert got["P0-B1"]["estimate"]["est"] == 0.75 and got["P0-A"]["estimate"]["est"] == 1.0
    out = got[f"OUTCOME-{CONSTRAINED_TIER}"]["estimate"]
    assert out["reference"] == [S.READER_B, "M2@4000"] and out["token_reference"] == [S.READER_B, "M0@full"]
    # the token clause is against B1's full text (1000 / 80000), not against the M2 reference (ratio 1)
    assert out["gaps"]["M2@4000"]["token_ratio"] == 0.0125 and out["outcome"] == "escapable"
    assert out["tier_models"] == {CONSTRAINED_TIER: 4} and out["models"] == sorted(C_LB)
    a_out = got[f"OUTCOME-{CONSTRAINED_TIER}-A"]["estimate"]
    assert a_out["reference"] == [S.READER_A, "M0@full"] and a_out["outcome"] == "escapable"
    # A is a reference only: no tier estimate is made of it
    assert all(g["tier"] != "A" for g in report["gap_tier"] + report["dgap_tier"])
    assert report["meta"]["load_bearing"]["tiers"].keys() == {"B", "C3", CONSTRAINED_TIER}
    arm = report["reference_arm_outcomes"][S.READER_A]
    assert arm["k1"] and set(arm["tiers"]) == {"B", "C3", CONSTRAINED_TIER} and S.READER_A not in arm["readers"]
    # the k = 1 rule: the 4k slice of the tiered readers is re-read; the A arm and M2@16000 are not, and do not count
    reasons = report["status"]["reasons"]
    assert not any(r.startswith("k = 1") for r in reasons)
    cov = report["status"]["repeat_coverage"]
    assert cov["cells"] == cov["repeated"] > 0 and cov["conditions"] == SLICE
    marked = {(c["profile_id"], c["condition"]): c["k1_marked"] for c in report["capability"]}
    assert marked[(S.READER_A, "M0@full")] and marked[(S.READER_B, "M2@16000")]
    assert not marked[(S.READER_B, "M2@4000")] and not marked[(QWEN32, M5)]
    # fewer than 3 models in a tier -> AMBER (tier B has B1 alone here); C_load_bearing has 4
    assert got["dGap-M5-B"]["small_tiers"] == ["B"] and got["interaction-M5"]["small_tiers"] == ["B"]
    assert got[f"dGap-M5-{CONSTRAINED_TIER}"]["small_tiers"] == []
    assert "tier estimate on too few models (AMBER): dGap-M5-B: tier B on 1 < 3 models" in reasons
    assert all(r["small_tier"] == (r["tier"] != CONSTRAINED_TIER) for r in report["dgap_tier"])
    # dGap-M5-B = -0.25 (M5 misses u2b); dropping U2 (same_rule medium; the MISO family) makes it 0: called out
    call = {c["id"]: c["sensitivities"] for c in report["sensitivity_callout"]}
    assert {"drop:same_rule_medium", "leave_out:MISO"} <= set(call["dGap-M5-B"])
    assert "P0-B1" not in call and "REF-B" not in call
    assert got["H_S"]["estimate"]["verdict"] == "does not hold"             # no branch error anywhere
    assert got["H_M7"]["estimate"] is None and got["CBUILT-gptoss20b"]["estimate"] is None     # not read here
    md = render_markdown(report, "conf")
    for text in ("Sensitivities that move a primary comparison", "against the A reference arm", "M2@4000: 1.000",
                 "k = 1 (reference arm)", "AMBER: tier B on 1 models", " †"):
        assert text in md, text
    one_missing = run_with(tmp_path / "b", panel, book, cells,
                           repeats=slice_repeats(cells, skip={(S.READERS_C[1], M5)}))
    reasons = analyse([one_missing], ITEMS, panel, book, bands, spec=spec_of(panel))["status"]["reasons"]
    assert any(r.startswith("k = 1: 4 of 72 answered cells of primary (4k slice)") for r in reasons)


def test_paper_m6_dgap_h_m7_and_cbuilt_on_the_grid(tmp_path, panel, book):
    """M2 misses u1a, u2a and M4 u1b, u2b for every tiered reader; the B1-built M7 is right everywhere (H_M7 +0.5 on
    both legs: pass). M6: B1 reads the B1 compression and misses three (dGap -0.25); the C readers self-compress, C1
    Llama misses all four (-0.5), the others match M2 (0): tier -0.125. C-built M7 by gpt-oss-20b misses u2b (CBUILT
    +0.25), by Qwen3-32B nothing (0)."""
    m2_wrong, m4_wrong = {"u1a", "u2a"}, {"u1b", "u2b"}
    cells = {(S.READER_B, "M0@full"): set(), (S.READER_B, M2): m2_wrong, (S.READER_B, "M4@4000"): m4_wrong,
             (S.READER_B, f"M6@4000/{B_DIRECT}"): {"u1a", "u1b", "u2a"},
             (S.READER_B, f"M7@4000/{B_DIRECT}"): set()}
    for pid in C_LB:
        cells.update({(pid, M2): m2_wrong, (pid, "M4@4000"): m4_wrong, (pid, f"M7@4000/{B_DIRECT}"): set(),
                      (pid, f"M6@4000/{pid}"): set(POP) if pid == S.READERS_C[0] else m2_wrong,
                      (pid, f"M7@4000/{S.READERS_C[2]}"): {"u2b"}, (pid, f"M7@4000/{QWEN32}"): set()})
    spec = spec_of(panel)

    def rows_of(run_id):
        return [S.row(run_id, pid, cond, it, it["item_id"] not in wrong,
                      tokens=(80_000, 50) if cond == "M0@full" else (1000, 50),
                      query_tokens=10 if cond.startswith("M6") else None)
                for (pid, cond), wrong in cells.items() for it in ITEMS]
    run = load_run(S.write_run(tmp_path, rows_of, ITEMS, panel, book))
    grid = cap.Grid(observations([run], ITEMS, panel), tiers=spec["tiers"],
                    populations=_populations(spec, sorted(POP)))
    primary = {p["id"]: p for p in _primary(grid, spec)}
    got = {k: p["estimate"] for k, p in primary.items()}
    assert primary["H_M7"]["sensitivities"]["unmapped"]["verdict"] == "pass"          # tier_pair reads either value
    assert primary["H_S"]["sensitivities"]["unmapped"] is None                      # no unmapped branch-error value
    assert got["dGap-M6-B"]["est"] == -0.25 and got[f"dGap-M6-{CONSTRAINED_TIER}"]["est"] == -0.125
    x = got["interaction-M7"]                                          # M2 -> B1-built M7: +0.5 in both tiers
    assert x["est"] == 0.0 and x["tier_models"] == {CONSTRAINED_TIER: 4, "B": 1}
    h = got["H_M7"]
    assert h["verdict"] == "pass" and {z: v["est"] for z, v in h["legs"].items()} == {M2: 0.5, "M4@4000": 0.5}
    assert got["CBUILT-gptoss20b"]["est"] == 0.25 and got["CBUILT-qwen32b"]["est"] == 0.0
    assert got["CBUILT-qwen32b"]["models"] == sorted(C_LB)            # Qwen3-32B reads its own C-built M7 too
    out = got[f"OUTCOME-{CONSTRAINED_TIER}"]
    assert out["reference"] == [S.READER_B, "M0@full"]                 # REF-B: M0@full 1.0 > M2@4000 0.5
    assert "M6@4000/{reader}" in out["gaps"] and f"M7@4000/{QWEN32}" in out["gaps"]
    assert "M7@4000/{reader}" not in out["gaps"] and out["gaps"]["M6@4000/{reader}"]["models"] == sorted(C_LB)
    assert out["outcome"] == "escapable" and f"M7@4000/{B_DIRECT}" in out["escapable_by"]
    assert _k1_scope(grid.conds[(QWEN32, f"M7@4000/{QWEN32}")], spec)
    assert _k1_scope(grid.conds[(QWEN32, f"M6@4000/{QWEN32}")], spec)
