"""The executable analysis spec of the paper: the analysis of the main run, written down as data before any main-run
answer was read. `make_paper_spec(items, items_path, items_sha256, panel, supply)` derives it from the main-run item file,
the panel and the constants below; scripts/make_spec.py writes it (`write_spec`: <out-dir>/spec.json) and the report
(procedures.analysis.report.analyse, `spec=`) executes it (`load_spec` reads a spec file of any name). The released
spec of the paper is data/analysis/spec.json.

Conditions are named `<rep>@<budget>[/<builder>]`: B0 closed book (no procedure text), M0 the full text of the
revisions in effect (`M0@full`), M2 BM25 retrieval, M4 constraint sentences (regex extraction + BM25), M5 an LLM
summary and M7 typed constraint records (both built once per revision by a builder model), M6 a query-focused LLM
compression (one call per question, by a named compressor or by the reader itself, "{reader}"). The budget is the
memory's token budget; the primary slice is 4,000 tokens.

* population: the items the primary comparisons are computed on: `analysis_stratum` primary, `load_bearing`, and in
  the main-analysis domains and sub-layer of MAIN_ANALYSIS (the ops / technical domains; the abstracted L3 items,
  whose scenario changes the description level of the clause instead of paraphrasing it). Frozen as item ids, with
  its cells, units and per-family units (158 cells in 79 units in the released spec).
* tiers: reader profiles per analysis tier (`tier_group`): B = governance tier B (open weights on a multi-GPU
  server); C_load_bearing = the constrained panel (access tiers C1 and C2: open weights on one 24 GB consumer card or
  one 48 GB workstation card; assigned by access tier, not by panel slot); C3 = access tier C3 (one 80 GB data-centre
  card), the intermediate step, reported on its own and never pooled into C_load_bearing; A = governance tier A
  (cloud frontier models). Readers = the main-run grids' readers (procedures.matrix.grids).
* reference_arms: {"A": [A2_sonnet5_nothink]} (Claude Sonnet 5, the cloud comparator), a reference only: in no tier,
  so in no tier estimate; its estimates carry the k = 1 mark.
* builders: the B-built arm (the B1 model, DeepSeek-V4.1-Flash, through DeepSeek's own API with thinking off) and the
  two C-built builders (gpt-oss-20b, access tier C1; Qwen3-32B, access tier C2).
* comparisons: the named primary estimates (kinds in COMPARISON_KINDS), in order; every other estimate of the report
  is exploratory. A condition may name "{reader}" for the reader's own compressor (a C reader's self-compressed M6).
  dGap = acc(to) - acc(from) = Gap(from) - Gap(to): positive = the `to` memory narrows the gap to the reference.
  - P0-B1 and P0-A (accuracy): the B1 reader and the cloud comparator reading the full text (M0@full).
  - REF-B (reference_rule): the open-weight reference, since B1's full text is no upper bound: of B1's M0@full and
    M2@4000, the one with the higher population accuracy (a tie goes to M0@full), chosen once and held fixed under the
    sensitivities.
  - OUTCOME-C_load_bearing (tier_outcome, the verdict) and OUTCOME-C_load_bearing-A (reported): the constrained
    panel's outcome against REF-B and against P0-A, over its measured local conditions (representations x budgets,
    budget <= 32k). The outcome rule (procedures.analysis.capability.outcome): `binds` when every local condition has
    Gap >= 10pp with the interval's lower end > 0; else `escapable` when some local condition has the interval's
    upper end <= 5pp at <= 1/4 of the input tokens of the token reference's full text (B1's for the verdict, the
    cloud comparator's for the reported outcome; one reference tokenizer), which only budgets <= 4k can meet; else
    `indeterminate`.
  - dGap-M5, dGap-M7 and dGap-M6, each in B and in C_load_bearing (tier_dgap): dGap(M2@4000 -> M5 / M7 / M6 @4000),
    with M5 and M7 built by B1, and M6 compressed by B1 in B and by each reader itself in C_load_bearing.
  - interaction-M5 and interaction-M7: dGap(C_load_bearing) - dGap(B) for M2@4000 -> the B1-built M5@4000 and
    M7@4000 (the M6 interaction is exploratory).
  - H_M7 (h_m7, the records hypothesis): the B1-built M7@4000 against M2@4000 and M4@4000 in C_load_bearing; passes
    when both differences are >= 5pp with the interval's lower end > 0.
  - H_S (h_s, the summary hypothesis): branch-error rate (E-cond + E-ver + E-exc over all cells) of the B1-built
    M5@4000 minus M2 / M4 @4000 in C_load_bearing, main pool the population, control S2; holds when both main
    differences have the interval's lower end > 0 and both control differences are under 5pp.
  - CBUILT-gptoss20b and CBUILT-qwen32b (tier_dgap): from the C-built to the B-built M7@4000 in C_load_bearing
    (positive = the B-side build narrows the gap).
* sensitivities: each primary comparison is recomputed under every one of them: other bootstrap clusters (template
  family, document); whole units dropped by a ruling tag (UNIT_TAGS: the authors' medium-confidence rulings,
  exemption outcomes, an owner's inconsistent restatement of a standard, and exception sentences outside the cue
  pool, the cue pool read two ways: the first cue pool alone, or the union with the second) or by family (leave one
  procedure owner out); and the reading without the reasoning-field mapping (answers a reader gave only in its
  reasoning field counted wrong). Unit lists are frozen in the spec.
* repeats: the correctness of a cell is the mean over its reads before the bootstrap; replicates resample clusters
  (and models), never repeats. repeats.k1_scope_conditions is the 4k slice: the k = 1 warning (cells read once:
  intervals cover item sampling, not re-read noise) is withdrawn when every answered cell of the population in those
  conditions, read by a tiered reader, has at least 2 reads; the reference arm and the other conditions keep the
  k = 1 mark.
* status_rules.min_models_per_tier = 3: a tier estimate on fewer models is AMBER (descriptive only), beside the
  report's other AMBER conditions (procedures.analysis.report).
* sensitivity_callout: a primary comparison that a sensitivity flips in sign, or whose interval it makes span 0
  (a verdict comparison: whose verdict it changes), is named in the report.

`check_spec(spec, items, panel)` ties a spec to the items and panel of an analysis: every population id in the items
must still satisfy the population rule and every item satisfying it must be a population id; a cell carrying a tag
must belong to a unit of the frozen list, and a unit whose cells are all among the items must carry exactly its frozen
tags (a partial run may hold only some cells of a unit, and a tag such as subtype sits on one cell); each tier's
profiles must still map to that tier; each reference arm's profiles must be of that governance tier; every condition
a comparison or the k = 1 scope names must be a reader stage whose builder / compressor is a panel profile (or
"{reader}").
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from procedures.matrix.grids import A_REFERENCE, BUILDER, C_BUILDER, C_BUILDER_QWEN32B, M6_B_COMPRESSOR, SERVER_READERS, SINGLE_GPU_READERS
from procedures.readers.cells import STAGE

SCHEMA = "procedures.analysis.spec.v1"
SPEC_FILE = "spec.json"              # the name write_spec gives a spec
PRIMARY_STRATUM = "primary"
REFERENCE = {"profile_id": "B1_dsv41flash_hf_deepinfra", "condition": "M0@full"}   # the reader-level Gap reference
BUDGET = 4000
READER = "{reader}"
# Tier names, unit tags, sensitivity and comparison ids are strings of the frozen spec and of the bootstrap seeds
# (procedures.analysis.stats.seed_of): keep them.
SERVER_TIER, CONSTRAINED_TIER, INTERMEDIATE_TIER = "B", "C_load_bearing", "C3"
CONSTRAINED_ACCESS = ("C1", "C2")           # the constrained panel: C1 u C2 (24 / 48 GB card), C3 apart
# the main-analysis subset of the primary stratum: the ops and technical domains, L3-abstracted units
MAIN_ANALYSIS = {"domain": ("ops", "technical"), "rule2_spirit": ("abstracted",)}
RULED_FIELDS = ("analysis_stratum", "domain", "load_bearing", "subtype", "same_rule_confidence", "k6_exception_literal",
                "k6_exception_confidence", "pin_confidence", "normative_confidence", "cue_relation", "cue_pool",
                "template_family", "owner_restates_standard")
# a unit carries a tag when any of its cells does (the tags are cell fields; subtype sits on the exemption cell)
UNIT_TAGS = {
    # medium confidence that the unit's cells are branches of one rule
    "same_rule_medium": lambda it: it.get("same_rule_confidence") == "medium",
    # medium confidence in the literal-exception ruling (`k6_exception_literal`)
    "k6_medium": lambda it: it.get("k6_exception_confidence") == "medium",
    # medium confidence that the scenario pins the clause (the clause governs the scenario as written)
    "pin_medium": lambda it: it.get("pin_confidence") == "medium",
    # medium confidence that the clause is normative (binding on the operator)
    "normative_medium": lambda it: it.get("normative_confidence") == "medium",
    # the exception's outcome is an exemption from the rule
    "exemption_outcome": lambda it: it.get("subtype") == "exemption_outcome",
    # the pool read literally: the exception sentence is no row of the first cue pool
    "outside_pool": lambda it: it.get("cue_relation") == "outside_pool",
    # the pool read as the union: no row of the first or the second cue pool
    "outside_union_pool": lambda it: it.get("cue_relation") == "outside_pool" and it.get("cue_pool") is None,
    # the owner's procedure restates a standard inconsistently
    "owner_restates_inconsistent": lambda it: it.get("owner_restates_standard") == "inconsistent",
}
CLUSTER_SENSITIVITIES = ("template_family", "document")    # the bootstrap re-clustered, as sensitivities
C_BUILDERS = {"gptoss20b": C_BUILDER, "qwen32b": C_BUILDER_QWEN32B}   # the tier-C builders of M7
MIN_MODELS_PER_TIER = 3                                     # fewer models -> the tier estimate is AMBER
COMPARISON_KINDS = ("accuracy", "reference_rule", "tier_outcome", "tier_dgap", "interaction", "h_m7", "h_s")


def in_population(item: dict) -> bool:
    return (item.get("analysis_stratum") == PRIMARY_STRATUM and item.get("load_bearing") is True
            and item.get("domain") in MAIN_ANALYSIS["domain"]
            and item.get("rule2_spirit") in MAIN_ANALYSIS["rule2_spirit"])


def tier_group(tier: str, access_tier: str) -> str:
    """The analysis tier of a reader: its governance tier, with C split by access tier (C1 u C2 | C3)."""
    if tier != "C":
        return tier
    if access_tier in CONSTRAINED_ACCESS:
        return CONSTRAINED_TIER
    if access_tier == INTERMEDIATE_TIER:
        return INTERMEDIATE_TIER
    raise ValueError(f"a C reader with access tier {access_tier!r}: not in {CONSTRAINED_ACCESS} or C3")


def unit_tags(items: list[dict]) -> dict[str, list[str]]:
    """tag -> sorted unit ids of `items` carrying it."""
    return {tag: sorted({it["unit_id"] for it in items if keep(it)}) for tag, keep in UNIT_TAGS.items()}


def template_families(items: list[dict]) -> dict[str, list[str]]:
    fam: dict[str, set] = defaultdict(set)
    for it in items:
        if it.get("template_family"):
            fam[it["template_family"]].add(it["unit_id"])
    return {k: sorted(v) for k, v in sorted(fam.items())}


def _units_by_family(items: list[dict]) -> dict[str, list[str]]:
    fam: dict[str, set] = defaultdict(set)
    for it in items:
        fam[it["family"]].add(it["unit_id"])
    return {k: sorted(v) for k, v in sorted(fam.items())}


def sensitivities(population: list[dict]) -> list[dict]:
    tags = unit_tags(population)
    out = [{"id": f"clusters:{c}", "kind": "clusters", "clusters": c} for c in CLUSTER_SENSITIVITIES]
    out += [{"id": f"drop:{tag}", "kind": "drop_units", "units": units} for tag, units in tags.items()]
    out += [{"id": f"leave_out:{fam}", "kind": "drop_units", "units": units}
            for fam, units in _units_by_family(population).items()]
    out.append({"id": "unmapped", "kind": "values", "values": "correct_unmapped",
                "note": "cells answered from the reasoning field counted wrong"})
    return out


def tiers(panel, readers=SERVER_READERS + SINGLE_GPU_READERS) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for pid in readers:
        pp = panel.get(pid)
        out[tier_group(pp.tier, pp.access_tier)].append(pid)
    return {k: sorted(v) for k, v in sorted(out.items())}


def paper_comparisons() -> list[dict]:
    """The paper's primary comparisons, in order (described in the module docstring)."""
    m2, m4 = f"M2@{BUDGET}", f"M4@{BUDGET}"
    built = {rep: f"{rep}@{BUDGET}/{BUILDER}" for rep in ("M5", "M7")}
    m6 = {SERVER_TIER: f"M6@{BUDGET}/{M6_B_COMPRESSOR}", CONSTRAINED_TIER: f"M6@{BUDGET}/{READER}"}
    ref_b = REFERENCE["profile_id"]
    outcome_note = ("over every readable local condition (budget <= 32k) of the tier's models: the measured "
                    "representations x budgets, nothing beyond them; escapable's token clause (query input <= 1/4 of "
                    "the full text's, reference tokenizer, against the reference reader's M0@full) can only be met at "
                    "budgets <= 4k")
    out = [{"id": "P0-B1", "kind": "accuracy", **REFERENCE,
            "question": "the B reference: B1 DeepSeek-V4.1-Flash reading the full text"},
           {"id": "P0-A", "kind": "accuracy", "profile_id": A_REFERENCE, "condition": "M0@full", "k1": True,
            "question": "the A reference arm reading the full text (k = 1)"},
           {"id": "REF-B", "kind": "reference_rule", "profile_id": ref_b, "candidates": ["M0@full", m2],
            "rule": "the candidate with the higher population accuracy (point estimate, primary clusters, mapped "
                    "answers); a tie goes to the first; chosen once, held fixed under every sensitivity",
            "question": "the B tier's best available reference (B1's full text is no upper bound)"},
           {"id": f"OUTCOME-{CONSTRAINED_TIER}", "kind": "tier_outcome", "tier": CONSTRAINED_TIER,
            "reference": "REF-B", "token_reference": REFERENCE, "role": "verdict", "note": outcome_note,
            "question": f"trilemma outcome of tier {CONSTRAINED_TIER} against REF-B"},
           {"id": f"OUTCOME-{CONSTRAINED_TIER}-A", "kind": "tier_outcome", "tier": CONSTRAINED_TIER,
            "reference": "P0-A", "token_reference": {"profile_id": A_REFERENCE, "condition": "M0@full"},
            "role": "reported", "k1": True, "note": outcome_note,
            "question": f"trilemma outcome of tier {CONSTRAINED_TIER} against the A reference arm"}]
    for rep in ("M5", "M7", "M6"):
        to = {SERVER_TIER: m6[SERVER_TIER], CONSTRAINED_TIER: m6[CONSTRAINED_TIER]} if rep == "M6" else\
            {SERVER_TIER: built[rep], CONSTRAINED_TIER: built[rep]}
        for tier in (SERVER_TIER, CONSTRAINED_TIER):
            out.append({"id": f"dGap-{rep}-{tier}", "kind": "tier_dgap", "tier": tier, "from": m2, "to": to[tier],
                        "question": f"zero-LLM retrieval -> {rep} at 4k, tier {tier}"
                                    + (" (B1-compressed in B, self-compressed in C)"
                                       if rep == "M6" else " (B1-built)")})
    for rep in ("M5", "M7"):
        out.append({"id": f"interaction-{rep}", "kind": "interaction", "local": CONSTRAINED_TIER,
                    "server": SERVER_TIER, "from": m2, "to": built[rep],
                    "question": f"dGap({CONSTRAINED_TIER}) - dGap({SERVER_TIER}) for M2 -> {rep} at 4k"})
    out += [{"id": "H_M7", "kind": "h_m7", "tier": CONSTRAINED_TIER, "to": built["M7"], "from": [m2, m4],
             "question": "H_M7: the B1-built M7 against M2 and M4 at 4k, tier C_load_bearing "
                         "(>= 5pp with CI low > 0 on both)"},
            {"id": "H_S", "kind": "h_s", "tier": CONSTRAINED_TIER, "llm": built["M5"], "zero_llm": [m2, m4],
             "main_scope": "primary", "control_scope": "S2", "reading": "rate_diff",
             "question": "H_S: branch-error rate (E-cond + E-ver + E-exc over all cells) of the "
                         "B1-built M5 minus M2 / M4 at 4k, tier C_load_bearing; main pool the population (L3), "
                         "control S2 (|diff| < 5pp)"}]
    for name, builder in C_BUILDERS.items():
        out.append({"id": f"CBUILT-{name}", "kind": "tier_dgap", "tier": CONSTRAINED_TIER,
                    "from": f"M7@{BUDGET}/{builder}", "to": built["M7"],
                    "question": f"C-built ({builder}) -> B-built M7 at 4k, tier {CONSTRAINED_TIER} (positive = the "
                                "B-side build narrows the gap)"})
    return out


def k1_scope_conditions() -> list[str]:
    """The 4k slice on which the k = 1 warning is withdrawn: B0, M0@full, M2 / M4 / M5 / M7 @4k, M6@4k in both forms
    (B1-compressed and self-compressed), the C-built M7@4k."""
    return ["B0@full", "M0@full", f"M2@{BUDGET}", f"M4@{BUDGET}", f"M5@{BUDGET}/{BUILDER}", f"M7@{BUDGET}/{BUILDER}",
            f"M6@{BUDGET}/{M6_B_COMPRESSOR}", f"M6@{BUDGET}/{READER}",
            *(f"M7@{BUDGET}/{b}" for b in C_BUILDERS.values())]


def make_paper_spec(items: list[dict], items_path: str, items_sha256: str, panel, supply: dict) -> dict:
    """The spec of the paper (module docstring) over `items`, the main-run item file; `supply` = the supply check
    (procedures.analysis.capability.supply, key L3) of the same items (recorded beside the population)."""
    population = sorted((it for it in items if in_population(it)), key=lambda it: it["item_id"])
    units = {it["unit_id"] for it in population}
    return {
        "schema": SCHEMA,
        "items": {"path": items_path, "sha256": items_sha256, "n": len(items)},
        "panel_sha256": panel.sha256,
        "population": {"rule": {"analysis_stratum": PRIMARY_STRATUM, "load_bearing": True,
                                **{k: list(v) for k, v in MAIN_ANALYSIS.items()}},
                       "cells": len(population), "units": len(units),
                       "units_by_family": {k: len(v) for k, v in _units_by_family(population).items()},
                       "template_families": template_families(population),
                       "item_ids": [it["item_id"] for it in population],
                       "unit_of": {it["item_id"]: it["unit_id"] for it in population}},
        "reference": REFERENCE,
        "tiers": tiers(panel),
        "load_bearing_tier": CONSTRAINED_TIER, "server_tier": SERVER_TIER, "intermediate_tier": INTERMEDIATE_TIER,
        "reference_arms": {"A": [A_REFERENCE]},
        "builders": {"B-built": BUILDER, "C-built": list(C_BUILDERS.values())},
        "comparisons": paper_comparisons(),
        "comparison_kinds": list(COMPARISON_KINDS),
        "exploratory": "every estimate of the report that is not a primary comparison",
        "sensitivities": sensitivities(population),
        "repeats": {"cell_value": "mean over the cell's reads, before the bootstrap",
                    "repeat_level_resampling": False,
                    "k1_scope_conditions": k1_scope_conditions(),
                    "k1_warning_withdrawn_when": "every answered cell of the population in k1_scope_conditions, read "
                                                 "by a reader of the tiers, has at least 2 reads",
                    "k1_marked": "the reference arms and every condition outside k1_scope_conditions"},
        "status_rules": {"min_models_per_tier": MIN_MODELS_PER_TIER,
                         "rule": "a tier estimate on fewer models is AMBER"},
        "sensitivity_callout": "a primary comparison whose sign a sensitivity flips, or whose interval (excluding 0) "
                               "it makes span 0, or whose verdict it changes, is named in the text",
        "supply": supply,
    }


def spec_bytes(spec: dict) -> bytes:
    return (json.dumps(spec, ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8")


def write_spec(spec: dict, out_dir: Path) -> Path:
    path = Path(out_dir) / SPEC_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(spec_bytes(spec))
    return path


def load_spec(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def spec_conditions(spec: dict) -> list[str]:
    """Every condition the spec's comparisons and k = 1 scope name (sorted, once each)."""
    out = set(spec.get("repeats", {}).get("k1_scope_conditions", []))
    for comp in spec["comparisons"]:
        for field in ("condition", "from", "to", "llm"):
            v = comp.get(field)
            out.update([v] if isinstance(v, str) else v or [])
        out.update(comp.get("candidates", []))
        out.update(comp.get("zero_llm", []))
    return sorted(out)


def check_spec(spec: dict, items: list[dict], panel) -> dict:
    """Raises when the items or the panel no longer agree with the spec; returns the population ids present in
    `items` and those missing from them."""
    ids = set(spec["population"]["item_ids"])
    present = [it for it in items if it["item_id"] in ids]
    drift = sorted(it["item_id"] for it in present if not in_population(it))
    stray = sorted(it["item_id"] for it in items if in_population(it) and it["item_id"] not in ids)
    if drift or stray:
        raise ValueError(f"the items disagree with the spec's population: {len(drift)} population ids no longer "
                         f"satisfy its rule (e.g. {drift[:2]}), {len(stray)} items satisfy it outside the spec "
                         f"(e.g. {stray[:2]})")
    unit_of = spec["population"]["unit_of"]
    cells: dict[str, set] = defaultdict(set)
    for i, u in unit_of.items():
        cells[u].add(i)
    whole = {u for u, ids_u in cells.items() if ids_u <= {it["item_id"] for it in present}}
    frozen = {s["id"][len("drop:"):]: set(s["units"]) for s in spec["sensitivities"] if s["id"].startswith("drop:")}
    unfrozen = sorted(set(UNIT_TAGS) - set(frozen))
    if unfrozen:
        raise ValueError(f"the spec froze no unit list for the tags {unfrozen} (made before they existed): remake it")
    for tag, units in unit_tags(present).items():
        # a tagged cell puts its unit in the frozen list; a unit whose cells are all here carries exactly its tags
        if not set(units) <= frozen[tag] or (frozen[tag] & whole) != (set(units) & whole):
            raise ValueError(f"unit tag {tag}: the items give {units}, the spec froze {sorted(frozen[tag])}")
    for tier, pids in spec["tiers"].items():
        for pid in pids:
            pp = panel.get(pid)
            if tier_group(pp.tier, pp.access_tier) != tier:
                raise ValueError(f"{pid}: the panel puts it in tier {tier_group(pp.tier, pp.access_tier)}, "
                                 f"the spec in {tier}")
    for tier, pids in spec.get("reference_arms", {}).items():
        for pid in pids:
            if panel.get(pid).tier != tier:
                raise ValueError(f"{pid}: the panel puts it in governance tier {panel.get(pid).tier}, the spec's "
                                 f"reference arm in {tier}")
    for cond in spec_conditions(spec):
        m = STAGE.fullmatch(f"read/{cond.replace(READER, 'reader')}")
        if m is None:
            raise ValueError(f"the spec names {cond!r}, which is no reader stage")
        if m["builder"] and READER not in cond and m["builder"] not in panel.profiles:
            raise ValueError(f"the spec names {cond!r}, whose builder / compressor is no panel profile")
    return {"present": sorted(ids & {it["item_id"] for it in items}),
            "missing": sorted(ids - {it["item_id"] for it in items})}
