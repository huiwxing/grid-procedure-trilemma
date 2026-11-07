"""The analysis report of one or more reader-matrix runs: the capability, access and sustainability tables, the
pre-registered tests and the status, as one JSON document and its markdown tables. The status gathers the supply /
power check (enough L3 units of the main analysis and S2 branch pairs in the item pool, n_eff of 30 or more per
comparison; supply keys L3 / S2), the billing reconciliation (until it passes, dollar figures are not
results) and the blind arbitration of the answers the scorer could not match (scorer rule N11: the tables are final
only when its queue is empty).

`analyse(runs, items, panel, book, bands, pool=..., spec=...)` joins the runs' rows with their items
(procedures.analysis.data), first checks itself against every run's own summary (per reader x stage, on the first
read as the summaries count it: answered cells, correct cells and the frozen chance-corrected accuracy must match; a
mismatch raises), then builds the capability grid and the tables. The supply check is counted on `pool` (the full
admitted item pool); the analysed items' own counts are reported beside it as the subset, and the spec's population
as a third line. Every Gap, dGap, tier estimate, interaction and H_S difference carries, besides its interval over the
primary clusters, the interval with the document family (the procedure owner) as the cluster (`family`: the
second-level sensitivity).

With `spec` (procedures.analysis.spec): the spec is checked against the items and the panel
(procedures.analysis.spec.check_spec), its tiers replace the derived ones, its population is the scope "primary",
and its named comparisons are computed first, each under every sensitivity of the spec; every other table is
exploratory. Without a spec the tiers are procedures.analysis.spec.tier_group of each reader (B, open weights on a
multi-GPU server; C_load_bearing = the constrained panel, the C1 and C2 single-card readers; C3 apart) and the main
scope is L3-abstracted (capability.ABSTRACTED_SCOPE).

The paper's spec (procedures.analysis.spec.make_paper_spec; recognised by repeats.k1_scope_conditions) adds: the
comparison kinds reference_rule (the open-weight reference REF-B: chosen once on the population, held under the
sensitivities), tier_outcome (the tier's three-outcome verdict, binds / escapable / indeterminate, against REF-B or
the cloud comparator, the token clause against the reference reader's full text), h_m7 (the records hypothesis: typed
constraint records, M7, against BM25 retrieval, M2, and constraint sentences, M4) and h_s (the summary hypothesis:
the tier-level branch-error rate difference of the LLM summary, M5, control S2); the k = 1 rule counted on the 4k
slice read by the tiers' readers (Grid.repeat_coverage(scope, conditions=...)), with the reference arms and the other
conditions marked k = 1 (†); AMBER on a primary tier estimate over fewer than status_rules.min_models_per_tier models
(exploratory tier rows are marked); the sensitivity call-out (the primary comparisons a sensitivity flips in sign,
makes span 0 or changes the verdict of); the outcomes against the cloud comparator (the A reference arm) as their own
table; and two
exploratory memory-level sensitivities, never among the primary comparisons: the M5 comparisons without the cells
whose M5 memory is build_truncated (a summary block still cut at the top of its output-cap ladder), and the C-built
comparisons (CBUILT-*: tier-C-built against B-built M7) on the population cells whose revisions every builder located
at >= LOCATED_MIN (the share of records whose source span is found in the revision text), from `readback`
(procedures.analysis.readback; listed as not computed without it).
A primary comparison the analysed runs cannot give (its population cells, conditions, readers or reference absent) has
estimate None, `computable` false and is named in the status; `missing` lists what the runs lack, also beside a
computed estimate that rests on the rest of a tier. A condition with no answered cell (every cell recorded and not
sent) is an absence, never a readable condition (capability.Cond.readable).
`write_report` stores <label>_report.json and <label>_report.md.

Status: AMBER (descriptive only) whenever any of these holds: k = 1 (some answered cell of the main scope has fewer
than 2 reads: a cell's value is the mean of its reads and the bootstrap never resamples repeats, so intervals cover
item sampling only), the supply check unmet on the pool (L3 units) or for the branch pairs (S2), no pool given, spec
population cells missing from the analysed items, the reference condition missing, a comparison below n_eff 30
(flagged per comparison), the arbitration queue not empty (scorer rule N11: tables not final), the billing
reconciliation not passed ($ not reportable), a void condition. Panel deviations (answers taken from the reasoning field,
where some readers put them; reasoning leaks) are listed; they void nothing. Every condition with answers taken from
the reasoning field is reported in both readings: the mapped arm (the table's) and the arm without the mapping, whose
mapped cells are empty (E-parse), with its own void verdict under the same rule (void when more than 10% of its
answered cells are empty). Empty memories (a legitimate outcome of
a representation) are counted per condition; the cells stay in their condition. Cells recorded and not sent are
absences, listed per reader x condition by reason (`absences`: route_context, the serving route's context exceeded,
and route_timeout, a query-focused compression (M6) that timed out on every attempt, alike, beside local_ceiling and
compression_<reason>); they are in no estimate.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from procedures.readers.cells import QUERY_BUILT

from procedures.analysis import capability as cap
from procedures.analysis.costs import AMORTISE_N, access, book_identity, build_costs, energy, not_sent_by_reason, sustainability
from procedures.analysis.data import Obs, Run, observations
from procedures.analysis.spec import CONSTRAINED_TIER, READER, check_spec
from procedures.analysis.stats import LEVEL, REPLICATES, r4
from procedures.analysis.taxonomy import CODEC, CODEC_CLASSES, NOT_IDENTIFIABLE, is_pending, taxonomy

SCHEMA = "procedures.analysis.report.v1"
BILLING_STATES = ("pending", "pass", "fail")
PACKAGE = Path(__file__).resolve().parent


def code_digest() -> str:
    h = hashlib.sha256()
    for p in sorted(PACKAGE.glob("*.py")):
        h.update(p.name.encode() + b"\0" + p.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()


def self_check(runs: list[Run], obs: list[Obs]) -> dict:
    """Per reader x stage of every run summary: answered cells, correct cells and chance-corrected accuracy of the
    first read (repeat 0), which is what the run summaries count (procedures.scoring.score_path); the analysis itself
    averages every read of a cell."""
    ours: dict[tuple, list[Obs]] = defaultdict(list)
    for o in obs:
        if o.observed and o.repeat_idx == 0:
            ours[(o.run_id, o.profile_id, f"read/{o.condition}")].append(o)
    checked = 0
    for run in runs:
        if run.summary is None:
            continue
        for key, s in sorted(run.summary["conditions"].items()):
            got = s.get("capability")
            if got is None:
                continue
            pid, stage = key.split("|", 1)
            mine = ours.get((run.run_id, pid, stage), [])
            n, correct = len(mine), sum(o.correct for o in mine)
            chance = sum((Fraction(o.chance).limit_denominator(100) for o in mine), Fraction(0))
            acc_cc = r4(Fraction(correct - chance) / (n - chance)) if n > chance else None
            if (n, correct, acc_cc) != (got["n"], got["correct"], got["acc_chance_corrected"]):
                raise ValueError(f"{run.run_id} {key}: analysis reads n={n} correct={correct} acc_cc={acc_cc}, the "
                                 f"run summary {got['n']}/{got['correct']}/{got['acc_chance_corrected']}")
            checked += 1
    return {"summary_conditions_matched": checked,
            "runs_without_summary": [r.run_id for r in runs if r.summary is None]}


def _cond_fields(c: cap.Cond) -> dict:
    return {"profile_id": c.profile_id, "condition": c.condition, "tier": c.tier, "arm": c.arm,
            "representation": c.representation, "budget": c.budget, "builder": c.builder}


def _status(grid: cap.Grid, supply: dict, pairs: list[dict], pending: int, billing: str, ref, coverage: dict,
            missing: int | None, small_tiers: list[str] = (), not_computable: list[str] = ()) -> dict:
    """`small_tiers`: the primary comparisons whose tier estimates rest on too few models (the paper's spec's
    status_rules); `not_computable`: the primary comparisons the analysed runs cannot give. The reason strings are those
    of the released report's status (module docstring)."""
    reasons = []
    if coverage["cells"] == 0:
        reasons.append(f"k = 1: no answered cell of {coverage['scope']} in the analysed runs, so no cell has 2 reads")
    elif coverage["repeated"] < coverage["cells"]:
        reasons.append(f"k = 1: {coverage['cells'] - coverage['repeated']} of {coverage['cells']} answered cells of "
                       f"{coverage['scope']} have fewer than 2 reads; a cell is the mean of its reads and replicates "
                       "never resample repeats, so intervals cover item (cluster) sampling only, not re-read noise")
    if missing:
        reasons.append(f"{missing} cells of the spec's population are not in the analysed items")
    if not_computable:
        reasons.append(f"{len(not_computable)} primary comparisons not computable from the analysed runs (their "
                       f"missing inputs are listed with them): {', '.join(not_computable)}")
    reasons += [f"tier estimate on too few models (AMBER): {s}" for s in small_tiers]
    counted = supply["pool"] if supply["pool"] is not None else supply["analysed"]
    where = "pool" if supply["pool"] is not None else "analysed items (no pool given)"
    if supply["pool"] is None:
        reasons.append("supply check counted on the analysed items only: no item pool was given")
    lb = counted["L3"]
    if not lb["met"]:
        unmet = sorted(k for k, ok in lb["checks"].items() if not ok)
        reasons.append(f"supply check unmet on the L3 units ({where}): {', '.join(unmet)}")
    if not counted["S2"]["met"]:
        reasons.append(f"supply check unmet on the S2 pairs ({where}): {counted['S2']['pairs']} pairs < "
                       f"{cap.SUPPLY_S2_PAIRS}")
    if ref is None:
        reasons.append(f"no readable reference {cap.REFERENCE_PROFILE} {cap.REFERENCE_CONDITION}: no Gap")
    low = sum(not p["n_eff_ok"] for p in pairs)
    if low:
        reasons.append(f"{low} of {len(pairs)} paired comparisons have n_eff < {cap.N_EFF_MIN}")
    if pending:
        reasons.append(f"{pending} answers await blind arbitration (N11: tables are not final)")
    if billing != "pass":
        reasons.append(f"billing reconciliation {billing}: dollar figures are not reportable as results")
    void = [f"{c.profile_id} {c.condition}: {c.void}" for c in grid.conds.values() if c.void]
    reasons += [f"void (not read): {v}" for v in void]
    deviations = [{"profile_id": c.profile_id, "condition": c.condition, "answered": c.answered,
                   "from_reasoning_field": c.from_reasoning, "reasoning_leaks": c.leaks, "empty": c.empty}
                  for c in grid.conds.values() if c.from_reasoning or c.leaks or c.empty]
    return {"label": "AMBER" if reasons else "GREEN", "reasons": reasons, "repeats": grid.repeats,
            "repeat_coverage": coverage, "void": void, "panel_deviations": deviations,
            "arbitration_pending": pending, "supply": supply}


SIGNED_KINDS = ("tier_dgap", "interaction")         # a sign and an interval to call out (sensitivity call-out)


def _readable(grid: cap.Grid, key: tuple) -> bool:
    return key in grid.conds and grid.conds[key].readable


def _accuracy(grid: cap.Grid, key: tuple, scope: str, by: str, kind: str) -> dict | None:
    if not _readable(grid, key):
        return None
    a = grid.accuracy(key, scope, by) if kind == "correct" else grid.unmapped_accuracy(key, scope, by)
    return {"n": a["n"], "clusters": a["clusters"], "est": a["est"], "ci": a["ci"], "models": [key[0]]}


def _references(grid: cap.Grid, spec: dict) -> dict[str, tuple | None]:
    """comparison id -> the (profile, condition) it reads, for the accuracy comparisons and the reference rules. A
    reference rule (REF-B: B1's full text is no upper bound) takes the readable candidate with the higher population
    accuracy (point estimate, primary clusters, mapped answers; a tie keeps the earlier), once: its sensitivities keep
    that choice."""
    out = {}
    for comp in spec["comparisons"]:
        if comp["kind"] == "accuracy":
            key = (comp["profile_id"], comp["condition"])
            out[comp["id"]] = key if _readable(grid, key) else None
        elif comp["kind"] == "reference_rule":
            best = None
            for cond in comp["candidates"]:
                key = (comp["profile_id"], cond)
                est = grid.accuracy(key, cap.POPULATION)["est"] if _readable(grid, key) else None
                if est is not None and (best is None or est > best[1]):
                    best = (key, est)
            out[comp["id"]] = None if best is None else best[0]
    return out


def _template(c: cap.Cond) -> str:
    """A self-compressed condition (a query-built memory made by its own reader) as its "{reader}" template: one tier
    condition over the tier's readers. A revision-built memory whose builder also reads it (the C-built M7 of
    Qwen3-32B read by Qwen3-32B) keeps its name."""
    if c.representation in QUERY_BUILT and c.builder == c.profile_id:
        return c.condition.replace(f"/{c.profile_id}", f"/{READER}")
    return c.condition


def _tier_conditions(grid: cap.Grid, tier: str, ref: tuple) -> dict[str, cap.Cond]:
    """condition (templated) -> one Cond of it, over the readable conditions of the tier's readers, less `ref`."""
    out: dict[str, cap.Cond] = {}
    for c in grid.readable():
        if grid.tier_of.get(c.profile_id) == tier and c.key != ref:
            out.setdefault(_template(c), c)
    return dict(sorted(out.items()))


def _tier_outcome(grid: cap.Grid, tier: str, ref: tuple | None, token_ref: tuple, scope: str,
                  by: str = "unit") -> dict | None:
    """The three-outcome verdict (binds / escapable / indeterminate) of a tier against `ref` over its readable local
    conditions (cap.outcome); the token clause is taken against `token_ref`'s input (the reference reader's full
    text), whatever `ref` is."""
    if ref is None:
        return None
    gaps, conds = {}, {}
    for cond, c in _tier_conditions(grid, tier, ref).items():
        g = grid.tier_gap(tier, cond, scope, ref, by)
        if g is None or g.get("est") is None:
            continue
        if token_ref != ref:
            t = grid.tier_gap(tier, cond, scope, token_ref, by) if _readable(grid, token_ref) else None
            g = {**g, "token_ratio": None if t is None else t.get("token_ratio")}
        gaps[cond], conds[cond] = g, c
    if not gaps:
        return None
    o = cap.outcome(gaps, conds)
    return {**o, "reference": list(ref), "token_reference": list(token_ref),
            "n": max(g["n"] for g in gaps.values()), "clusters": max(g["clusters"] for g in gaps.values()),
            "gaps": {c: {k: g.get(k) for k in ("est", "ci", "n", "token_ratio", "models")} for c, g in gaps.items()},
            "models": sorted({p for g in gaps.values() for p in g["models"]}),
            "tier_models": {tier: min(len(g["models"]) for g in gaps.values())}}


def _leg(t: dict) -> dict:
    return {"n": t["n"], "clusters": t["clusters"], **t["dgap"], "models": t["models"]}


def _estimate(grid: cap.Grid, comp: dict, scope: str, by: str = "unit", kind: str = "correct",
              refs: dict | None = None) -> dict | None:
    """One primary comparison of a spec as {n, clusters, est, ci, models, tier_models} (a verdict comparison: its
    verdict and legs instead of est / ci); None when its conditions are not read or the sensitivity does not apply
    (the unmapped values sensitivity on a Gap outcome or a branch-error rate)."""
    refs = refs or {}
    if comp["kind"] == "accuracy":
        return _accuracy(grid, (comp["profile_id"], comp["condition"]), scope, by, kind)
    if comp["kind"] == "reference_rule":
        chosen = refs.get(comp["id"])
        a = None if chosen is None else _accuracy(grid, chosen, scope, by, kind)
        return None if a is None else {**a, "chosen": chosen[1]}
    if comp["kind"] == "tier_dgap":
        t = grid.tier_pair(comp["tier"], comp["from"], comp["to"], scope, by, kind)
        return None if t is None else {**_leg(t), "h_t": t["h_t"], "tier_models": {comp["tier"]: len(t["models"])}}
    if comp["kind"] == "interaction":
        x = grid.interaction(comp["from"], comp["to"], scope, comp["local"], comp["server"], by, kind)
        return None if x is None else {"n": x["n"], "clusters": x["clusters"], **x["interaction"],
                                       "models": x["models"][comp["local"]] + x["models"][comp["server"]],
                                       "tier_models": {t: len(x["models"][t]) for t in (comp["local"],
                                                                                        comp["server"])}}
    if comp["kind"] == "h_m7":
        legs = {f: grid.tier_pair(comp["tier"], f, comp["to"], scope, by, kind) for f in comp["from"]}
        if any(t is None or t["dgap"]["est"] is None for t in legs.values()):
            return None
        ok = all(t["dgap"]["ci"] is not None and t["dgap"]["est"] >= cap.RECORDS_MIN_DIFF and t["dgap"]["ci"][0] > 0
                 for t in legs.values())
        return _verdict_estimate("pass" if ok else "fail", {f: _leg(t) for f, t in legs.items()}, comp["tier"])
    if kind != "correct":
        return None
    if comp["kind"] == "tier_outcome":
        tok = comp["token_reference"]
        return _tier_outcome(grid, comp["tier"], refs.get(comp["reference"]), (tok["profile_id"], tok["condition"]),
                             scope, by)
    if comp["kind"] == "h_s":
        if comp["main_scope"] != cap.POPULATION:
            raise ValueError(f"{comp['id']}: an H_S main pool other than the population ({comp['main_scope']!r})")
        legs, control = {}, {}
        for z in comp["zero_llm"]:
            m = grid.tier_pair(comp["tier"], z, comp["llm"], scope, by, "branch_error")
            c = grid.tier_pair(comp["tier"], z, comp["llm"], comp["control_scope"], by, "branch_error")
            if m is None or m["dgap"]["est"] is None:
                return None
            legs[z] = _leg(m)
            control[z] = None if c is None or c["dgap"]["est"] is None else _leg(c)
        verdict = cap.h_s_verdict([{"n": x["n"], "rate_diff": x} for x in legs.values()],
                                  [{"n": 0} if x is None else {"n": x["n"], "rate_diff": x} for x in control.values()],
                                  comp["reading"])
        return {**_verdict_estimate(verdict, legs, comp["tier"]), "control": control}
    raise ValueError(f"unknown comparison kind {comp['kind']!r}")


def _verdict_estimate(verdict: str, legs: dict, tier: str) -> dict:
    return {"verdict": verdict, "legs": legs, "n": min(x["n"] for x in legs.values()),
            "clusters": min(x["clusters"] for x in legs.values()),
            "models": sorted({p for x in legs.values() for p in x["models"]}),
            "tier_models": {tier: min(len(x["models"]) for x in legs.values())}}


def _sensitivity_args(sens: dict) -> dict:
    if sens["kind"] == "clusters":
        return {"scope": cap.POPULATION, "by": sens["clusters"]}
    if sens["kind"] == "drop_units":
        return {"scope": f"{cap.POPULATION}|{sens['id']}"}
    if sens["kind"] == "values":
        return {"scope": cap.POPULATION, "kind": sens["values"]}
    raise ValueError(f"unknown sensitivity kind {sens['kind']!r}")


def _populations(spec: dict, present: list[str]) -> dict[str, frozenset]:
    """The population (its cells among the analysed items) and, per unit-dropping sensitivity, the population less
    the frozen units (units by the spec's frozen unit_of)."""
    unit_of = spec["population"]["unit_of"]
    out = {cap.POPULATION: frozenset(present)}
    for sens in spec["sensitivities"]:
        if sens["kind"] == "drop_units":
            drop = set(sens["units"])
            out[f"{cap.POPULATION}|{sens['id']}"] = frozenset(i for i in present if unit_of[i] not in drop)
    return out


VERDICT_KINDS = ("tier_outcome", "h_m7", "h_s")


def _computed(kind: str, est: dict | None) -> dict | None:
    """The estimate when it holds a number (a verdict kind: a verdict or outcome), else None (not computable)."""
    if est is None:
        return None
    if kind in VERDICT_KINDS:
        return est
    return est if est.get("est") is not None else None


def _absent(grid: cap.Grid, profile_id: str, condition: str) -> str | None:
    c = grid.conds.get((profile_id, cap.resolve(condition, profile_id)))
    if c is None:
        return "not read"
    if c.void:
        return "void"
    return None if c.answered else "not sent (no answered cell)"


def _named(comp: dict) -> list[str]:
    out = []
    for field in ("from", "to", "llm", "zero_llm"):
        v = comp.get(field)
        out += [v] if isinstance(v, str) else list(v or [])
    return list(dict.fromkeys(out))


def _missing(grid: cap.Grid, comp: dict, spec: dict, refs: dict) -> list[str]:
    """What a primary comparison needs and the analysed runs lack: the population, a (reader, condition) of its tiers
    or its reference. An estimate over the rest is still made when it can be; this lists what it rests without."""
    out = []
    if not grid.populations.get(cap.POPULATION):
        out.append(f"population: none of its {len(spec['population']['item_ids'])} cells among the analysed items")
    if comp["kind"] in ("accuracy", "reference_rule"):
        for cond in comp.get("candidates") or [comp["condition"]]:
            why = _absent(grid, comp["profile_id"], cond)
            if why:
                out.append(f"{comp['profile_id']} {cond}: {why}")
        return out
    if comp["kind"] == "tier_outcome":
        if refs.get(comp["reference"]) is None:
            out.append(f"reference {comp['reference']}: not available")
        tok = comp["token_reference"]
        why = _absent(grid, tok["profile_id"], tok["condition"])
        if why:
            out.append(f"token reference {tok['profile_id']} {tok['condition']}: {why}")
        readers = spec["tiers"].get(comp["tier"], [])
        if not any(c.profile_id in readers and c.local for c in grid.readable()):
            out.append(f"tier {comp['tier']}: no readable local condition of its readers {readers}")
        return out
    tiers = [comp["local"], comp["server"]] if comp["kind"] == "interaction" else [comp["tier"]]
    for tier in tiers:
        readers = spec["tiers"].get(tier, [])
        for cond in _named(comp):
            lacking = [f"{p} ({why})" for p in readers if (why := _absent(grid, p, cond))]
            if lacking:
                out.append(f"tier {tier}, {cond}: {len(lacking)} of {len(readers)} readers lack it: "
                           + ", ".join(lacking))
    return out


def _primary(grid: cap.Grid, spec: dict) -> list[dict]:
    """The spec's comparisons with their estimates and sensitivities. An estimate without a number (its conditions,
    readers, reference or population cells are not in the analysed runs) is None with `computable` false; `missing`
    lists what the runs lack (also when an estimate rests on the rest of a tier). With the spec's status_rules, each
    carries `small_tiers` (the tiers of a computed estimate on fewer models than the rule's minimum: AMBER)."""
    refs = _references(grid, spec)
    least = spec.get("status_rules", {}).get("min_models_per_tier")
    out = []
    for comp in spec["comparisons"]:
        est = _computed(comp["kind"], _estimate(grid, comp, cap.POPULATION, refs=refs))
        row = {**comp, "estimate": est, "computable": est is not None, "missing": _missing(grid, comp, spec, refs),
               "sensitivities": {sens["id"]: _computed(comp["kind"],
                                                       _estimate(grid, comp, **_sensitivity_args(sens), refs=refs))
                                 for sens in spec["sensitivities"]}}
        if least is not None:
            row["small_tiers"] = sorted(t for t, n in ((est or {}).get("tier_models") or {}).items() if n < least)
        out.append(row)
    return out


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def _signed_move(main: dict | None, sens: dict | None) -> bool:
    """A sensitivity changes the sign of a nonzero estimate (a move to exactly 0 counts), or makes an interval that
    excludes 0 span it."""
    if not main or not sens or main.get("est") is None or sens.get("est") is None:
        return False
    if main["est"] != 0 and _sign(sens["est"]) != _sign(main["est"]):
        return True
    excludes = main.get("ci") is not None and (main["ci"][0] > 0 or main["ci"][1] < 0)
    return excludes and sens.get("ci") is not None and sens["ci"][0] <= 0 <= sens["ci"][1]


def _moves(kind: str, main: dict | None, sens: dict | None) -> bool:
    if main is None or sens is None:
        return False
    if kind in SIGNED_KINDS:
        return _signed_move(main, sens)
    if kind == "tier_outcome":
        return sens["outcome"] != main["outcome"]
    if kind in ("h_m7", "h_s"):
        return sens["verdict"] != main["verdict"] or any(_signed_move(leg, sens["legs"].get(z))
                                                         for z, leg in main["legs"].items())
    return False


def _callouts(rows: list[dict]) -> list[dict]:
    """The sensitivity call-out: the primary comparisons a sensitivity moves (sign, interval across 0, verdict), to
    be named in the text."""
    out = []
    for p in rows:
        hits = [name for name, s in p["sensitivities"].items() if _moves(p["kind"], p["estimate"], s)]
        if hits:
            out.append({"id": p["id"], "sensitivities": hits})
    return out


def _k1_scope(c: cap.Cond, spec: dict) -> bool:
    """Whether a condition is in the spec's k = 1 scope (the paper's spec: the 4k slice, tiered readers only)."""
    if any(c.profile_id in ps for ps in spec.get("reference_arms", {}).values()):
        return False
    return _template(c) in spec["repeats"]["k1_scope_conditions"]


def _k1_coverage(obs: list[Obs], spec: dict, populations: dict) -> dict:
    """The paper's k = 1 rule: repeat coverage of the population over the spec's k1_scope_conditions, read by the
    tiers' readers (the reference arms stay k = 1 and outside the rule)."""
    tiered = sorted({p for ps in spec["tiers"].values() for p in ps})
    grid = cap.Grid([o for o in obs if o.profile_id in tiered], tiers=spec["tiers"], populations=populations)
    scope = spec["repeats"]["k1_scope_conditions"]          # "{reader}" is resolved per reader by repeat_coverage
    if cap.POPULATION not in grid.scopes():
        return {"scope": f"{cap.POPULATION} (4k slice)", "cells": 0, "repeated": 0, "conditions": scope}
    return {**grid.repeat_coverage(cap.POPULATION, conditions=scope), "scope": f"{cap.POPULATION} (4k slice)",
            "conditions": scope}


def _reference_arm_outcomes(grid: cap.Grid, spec: dict) -> dict:
    """The three-outcome verdicts of every tier and every tiered reader against each reference arm's full text
    (M0@full), on the population (reported apart; the reference arm is k = 1)."""
    out = {}
    for pid in sorted(p for ps in spec.get("reference_arms", {}).values() for p in ps):
        ref = (pid, cap.REFERENCE_CONDITION)
        if not _readable(grid, ref):
            out[pid] = None
            continue
        tiers = {t: _tier_outcome(grid, t, ref, ref, cap.POPULATION) for t in grid.tiers()}
        readers = {}
        for p in sorted(grid.tier_of):
            mine = {c.condition: c for c in grid.readable() if c.profile_id == p and c.key != ref}
            gaps = {cond: grid.gap(c.key, cap.POPULATION, ref) for cond, c in mine.items()}
            readers[p] = cap.outcome({k: g for k, g in gaps.items() if g.get("est") is not None}, mine)
        out[pid] = {"reference": list(ref), "scope": cap.POPULATION, "k1": True, "tiers": tiers, "readers": readers}
    return out


M5_TRUNCATION_IDS = ("dGap-M5-B", f"dGap-M5-{CONSTRAINED_TIER}", "interaction-M5", "H_S")    # M5 truncation
LOCATED_IDS = ("CBUILT-gptoss20b", "CBUILT-qwen32b")                                            # M7 located
LOCATED_MIN = 0.90          # the memory build check's located-share line (set on a B1-built probe), unchanged
LOCATED_SCOPE = f"{cap.POPULATION}|m7_located"


def _m5_truncation(obs: list[Obs], spec: dict, populations: dict, truncated: dict[str, bool]) -> dict:
    """The M5 truncation sensitivity: the primary M5 comparisons with every cell whose M5 memory is build_truncated
    dropped (the cell leaves its condition, so the comparison runs on the items still read in both arms)."""
    comps = [c for c in spec["comparisons"] if c["id"] in M5_TRUNCATION_IDS]
    flagged = {k for k, v in truncated.items() if v}
    dropped = [o for o in obs if o.representation == "M5" and o.memory_cache_key in flagged]
    grid = cap.Grid([o for o in obs if not (o.representation == "M5" and o.memory_cache_key in flagged)],
                    tiers=spec["tiers"], populations=populations)
    by_condition: dict[str, int] = defaultdict(int)
    for o in dropped:
        if o.observed:
            by_condition[o.condition] += 1
    population = populations.get(cap.POPULATION, frozenset())
    return {"exploratory": True,
            "rule": "drop every cell whose M5 memory is build_truncated (a sub-block answer still capped at cap level "
                    "2), then recompute; memory-level, reported apart from the spec's unit-tag sensitivities",
            "memories": len(truncated), "build_truncated_memories": len(flagged),
            "build_truncated_memories_read": len({o.memory_cache_key for o in dropped}),
            "answered_cells_dropped": dict(sorted(by_condition.items())),
            "population_items_dropped": len({o.item_id for o in dropped if o.item_id in population}),
            "comparisons": {c["id"]: _computed(c["kind"], _estimate(grid, c, cap.POPULATION)) for c in comps}}


def _m7_located(obs: list[Obs], spec: dict, populations: dict, located: dict[str, dict]) -> dict:
    """The M7 located sensitivity: the two CBUILT comparisons on the population cells whose as-of revisions every
    builder of the spec located at LOCATED_MIN or more (M7 located share per revision,
    procedures.analysis.readback)."""
    builders = [spec["builders"]["B-built"], *spec["builders"]["C-built"]]
    comps = [c for c in spec["comparisons"] if c["id"] in LOCATED_IDS]
    base = {"exploratory": True, "threshold": LOCATED_MIN, "builders": builders,
            "rule": "keep the population cells whose as-of revisions have M7 located share >= the threshold for every "
                    "builder, then recompute the CBUILT comparisons"}
    lacking = [b for b in builders if b not in located]
    if lacking:
        return {**base, "computed": False, "reason": f"no M7 read-back for builders {lacking}"}
    revisions = sorted({r for b in builders for r in located[b]})
    unknown = {b: sorted(set(revisions) - set(located[b])) for b in builders}
    good = {r for r in revisions if all((located[b].get(r) or {}).get("share") is not None
                                        and located[b][r]["share"] >= LOCATED_MIN for b in builders)}
    of_item: dict[str, set] = defaultdict(set)
    for o in obs:
        if o.representation == "M7":
            of_item[o.item_id].update(f"{d}@{r}" for d, r in o.revisions)
    population = populations.get(cap.POPULATION, frozenset())
    kept = frozenset(i for i in population if of_item.get(i) and of_item[i] <= good)
    grid = cap.Grid(obs, tiers=spec["tiers"], populations={**populations, LOCATED_SCOPE: kept})
    return {**base, "computed": True, "revisions": len(revisions), "revisions_kept": len(good),
            "below_threshold": {b: sum(1 for r in revisions if r in located[b] and located[b][r]["share"] is not None
                                       and located[b][r]["share"] < LOCATED_MIN) for b in builders},
            "without_read_back": {b: len(v) for b, v in unknown.items()},
            "population_cells": len(population), "population_cells_kept": len(kept),
            "comparisons": {c["id"]: _computed(c["kind"], _estimate(grid, c, LOCATED_SCOPE)) for c in comps}}


def _memory_sensitivities(obs: list[Obs], spec: dict, populations: dict, readback: dict | None) -> dict:
    """The two exploratory memory-level sensitivities (M5 truncation, M7 located): not primary, not in the spec's
    list."""
    if readback is None:
        why = "no memory read-back: it needs the raw build ledgers and scripts/report.py --manifest (and --external)"
        return {"m5_build_truncated": {"exploratory": True, "computed": False, "reason": why},
                "m7_located": {"exploratory": True, "computed": False, "reason": why},
                "source": None}
    def unreadable(rep: str) -> dict | None:
        bad = [u for u in readback.get("unreadable", []) if u["representation"] == rep]
        if not bad:
            return None
        return {"exploratory": True, "computed": False, "unreadable": len(bad),
                "reason": f"{len(bad)} {rep} memories cannot be read back (their request map disagrees with the "
                          f"current plan), e.g. {bad[0]['reason']}"}
    m5_out = unreadable("M5") or {**_m5_truncation(obs, spec, populations, readback["m5_build_truncated"]),
                                            "computed": True}
    m7_out = unreadable("M7") or _m7_located(obs, spec, populations, readback["m7_located"])
    return {"m5_build_truncated": m5_out, "m7_located": m7_out, "source": readback.get("source")}


def _empty_memory(grid: cap.Grid, obs: list[Obs]) -> dict:
    """Empty memories per condition (a legitimate outcome, read and kept in the condition); a read shared by several
    conditions (the same empty memory at several budgets, one request) is counted once in `distinct_read_requests`."""
    rows = [{"profile_id": c.profile_id, "condition": c.condition, "answered": c.answered,
             "memory_empty": c.memory_empty} for c in grid.conds.values() if c.memory_empty]
    reads = {(o.profile_id, o.request_hash) for o in obs if o.observed and o.memory_empty}
    return {"by_condition": rows, "cells": sum(r["memory_empty"] for r in rows), "distinct_read_requests": len(reads)}


def _absences(obs: list[Obs]) -> dict:
    """Cells recorded and not sent (outcome not_sent_infeasible; every read, repeats included), per reader x
    condition and in total, by reason (procedures.analysis.costs.not_sent_by_reason)."""
    groups: dict[tuple, list[Obs]] = defaultdict(list)
    for o in obs:
        if o.outcome == "not_sent_infeasible":
            groups[(o.profile_id, o.condition)].append(o)
    rows = [{"profile_id": p, "condition": c, "not_sent": len(os), "by_reason": not_sent_by_reason(os)}
            for (p, c), os in sorted(groups.items())]
    return {"by_condition": rows, "by_reason": not_sent_by_reason(obs), "cells": sum(r["not_sent"] for r in rows)}


def _family(result: dict | None, field: str | None = None) -> dict | None:
    """{est, ci, clusters} of an estimate recomputed with family clusters (the sensitivity column): `result` is the
    estimate's output, `field` the key of its interval when the interval is nested (dGap, interaction, H_S)."""
    if result is None or not result.get("n"):
        return None
    est = result[field] if field else result
    return {"est": est.get("est"), "ci": est.get("ci"), "clusters": result.get("clusters")}


def analyse(runs: list[Run], items: list[dict], panel, book, bands: dict, *, pool: list[dict] | None = None,
            pool_sources: list[dict] | None = None, book_label: str = "snapshot",
            reference: str = cap.REFERENCE_PROFILE, billing: str = "pending", spec: dict | None = None,
            spec_path: str | None = None, readback: dict | None = None) -> dict:
    """`spec_path`: the spec's file as the report records it (meta.spec.path). `readback`
    (procedures.analysis.readback.memory_readback): the memory read-back for the exploratory memory sensitivities of
    the paper's spec; without it they are listed as not computed."""
    if billing not in BILLING_STATES:
        raise ValueError(f"billing must be one of {BILLING_STATES}")
    obs = observations(runs, items, panel)
    checks = self_check(runs, obs)
    by_id = {it["item_id"]: it for it in items}
    analysed = sorted({o.item_id for o in obs})
    spec_check = None
    paper_rules = spec is not None and "k1_scope_conditions" in spec["repeats"]     # the paper's spec's rules apply
    if spec is not None:
        spec_check = check_spec(spec, [by_id[i] for i in analysed], panel)
        populations = _populations(spec, spec_check["present"])
        grid = cap.Grid(obs, tiers=spec["tiers"], populations=populations)
    else:
        grid = cap.Grid(obs)
    lb_scope = cap.POPULATION if spec is not None else cap.ABSTRACTED_SCOPE
    ref = grid.reference(reference)
    scopes = grid.scopes()
    outcome_scopes = [s for s in ([cap.POPULATION] if spec is not None else []) + list(cap.OUTCOME_SCOPES)
                      if s in scopes]
    readable = grid.readable()
    void = {c.key: c.void for c in grid.conds.values() if c.void}
    by_condition = {c.condition: c for c in readable}

    capability_rows = [{**_cond_fields(c), "scopes": {s: grid.accuracy(c.key, s) for s in scopes},
                        **({"k1_marked": not _k1_scope(c, spec)} if paper_rules else {})} for c in readable]
    gaps, tier_gaps, outcomes = [], [], {}
    if ref is not None:
        for c in readable:
            if c.key == ref:
                continue
            gaps += [{**_cond_fields(c), "scope": s, **grid.gap(c.key, s, ref),
                      "family": _family(grid.gap(c.key, s, ref, by="family"))} for s in outcome_scopes]
        for tier in grid.tiers():
            for cond in sorted({c.condition for c in readable if grid.tier_of.get(c.profile_id) == tier} - {ref[1]}):
                for s in outcome_scopes:
                    g = grid.tier_gap(tier, cond, s, ref)
                    if g is not None:
                        tier_gaps.append({"tier": tier, "condition": cond, "scope": s, **g,
                                          "family": _family(grid.tier_gap(tier, cond, s, ref, by="family"))})
        for s in outcome_scopes:
            per_reader = {}
            for pid in sorted({g["profile_id"] for g in gaps}):
                mine = {g["condition"]: g for g in gaps if g["profile_id"] == pid and g["scope"] == s}
                per_reader[pid] = cap.outcome(mine, by_condition)
            per_tier = {t: cap.outcome({g["condition"]: g for g in tier_gaps if g["tier"] == t and g["scope"] == s},
                                       by_condition) for t in grid.tiers()}
            outcomes[s] = {"readers": per_reader, "tiers": per_tier}

    pairs, tier_pairs, interactions = [], [], []
    tier_arm_pairs: dict[str, set] = defaultdict(set)
    for pid in sorted({c.profile_id for c in readable}):
        for lower, higher in grid.arm_pairs(pid):
            if pid in grid.tier_of:
                tier_arm_pairs[grid.tier_of[pid]].add((lower, higher))
            for s in outcome_scopes:
                p = grid.pair(pid, lower, higher, s)
                p["family"] = _family(grid.pair(pid, lower, higher, s, by="family"), "dgap")
                pairs.append(p)
    for tier, arm_pairs in sorted(tier_arm_pairs.items()):
        for lower, higher in sorted(arm_pairs):
            for s in outcome_scopes:
                t = grid.tier_pair(tier, lower, higher, s)
                if t is not None:
                    t["family"] = _family(grid.tier_pair(tier, lower, higher, s, by="family"), "dgap")
                    tier_pairs.append(t)
    for lower, higher in sorted(tier_arm_pairs.get(cap.CONSTRAINED_TIER, set())
                                & tier_arm_pairs.get(cap.SERVER_TIER, set())):
        for s in outcome_scopes:
            x = grid.interaction(lower, higher, s)
            if x is not None:
                x["family"] = _family(grid.interaction(lower, higher, s, by="family"), "interaction")
                interactions.append(x)

    h_s = []
    for pid in sorted({c.profile_id for c in readable}):
        mine = [c for c in readable if c.profile_id == pid]
        for llm in [c for c in mine if c.arm.endswith("-built")]:
            zeros = [c for c in mine if c.arm == "zero-LLM" and c.budget == llm.budget]
            if not zeros:
                continue
            main = [grid.h_s(pid, llm.condition, z.condition, cap.HS_MAIN_SCOPE) for z in zeros]
            control = [grid.h_s(pid, llm.condition, z.condition, cap.HS_CONTROL_SCOPE) for z in zeros]
            for m, z in zip(main, zeros):
                m["family"] = _family(grid.h_s(pid, llm.condition, z.condition, cap.HS_MAIN_SCOPE, by="family"),
                                      "rate_diff")
            h_s.append({"profile_id": pid, "llm": llm.condition, "zero_llm": [z.condition for z in zeros],
                        "pre_registered": (llm.representation, llm.budget) == cap.HS_CRITERION,
                        "main_scope": cap.HS_MAIN_SCOPE, "control_scope": cap.HS_CONTROL_SCOPE,
                        "main": main, "control": control,
                        "verdict": {r: cap.h_s_verdict(main, control, r) for r in cap.HS_READINGS}})

    least = (spec or {}).get("status_rules", {}).get("min_models_per_tier")
    if least is not None:           # the tier-size rule: an exploratory tier estimate on too few models is marked too
        for row in tier_gaps + tier_pairs:
            row["small_tier"] = len(row["models"]) < least
    supply = {"pool": cap.supply(pool) if pool is not None else None, "pool_sources": pool_sources,
              "analysed": cap.supply([by_id[i] for i in analysed]),
              "population": cap.supply([by_id[i] for i in spec_check["present"]]) if spec is not None else None}
    pending = sum(o.observed and is_pending(o) for o in obs)
    mapping = [{"profile_id": c.profile_id, "condition": c.condition, "answered": c.answered,
                "from_reasoning_field": c.from_reasoning, "empty_mapped": c.empty,
                "mapped": {s: grid.accuracy(c.key, s) for s in scopes} if c.void is None else None,
                "void_mapped": c.void,
                "unmapped": {s: grid.unmapped_accuracy(c.key, s) for s in scopes},
                "empty_unmapped": c.empty + c.from_reasoning, "void_unmapped": c.void_unmapped}
               for c in grid.conds.values() if c.from_reasoning]
    builds = build_costs(runs, obs, book)
    sus = sustainability(obs, builds["memories"], void)
    acc_rows = access(runs, obs, builds["memories"], book, void)
    lb_gap = {(g["profile_id"], g["condition"]): g for g in gaps if g["scope"] == "all"}
    for row in acc_rows:
        g = lb_gap.get((row["profile_id"], row["condition"]))
        row["gap_all"] = {"est": g["est"], "ci": g["ci"]} if g else None
    primary = _primary(grid, spec) if spec is not None else None
    if paper_rules:
        coverage = _k1_coverage(obs, spec, populations)
    else:
        coverage = grid.repeat_coverage(lb_scope) if lb_scope in scopes else {"scope": lb_scope, "cells": 0,
                                                                              "repeated": 0}
    small = [f"{p['id']}: tier {t} on {p['estimate']['tier_models'][t]} < {least} models"
             for p in primary or [] for t in p.get("small_tiers", [])]
    report = {
        "schema": SCHEMA,
        "meta": {"spec": None if spec is None else {"path": spec_path, "population_cells": len(spec_check["present"]),
                                                    "population_missing": len(spec_check["missing"]),
                                                    **({"k1_scope_conditions": spec["repeats"]["k1_scope_conditions"],
                                                        "reference_arms": spec["reference_arms"],
                                                        "min_models_per_tier": least} if paper_rules else {})},
                 "runs": [{"run_id": r.run_id, "rows": r.rows_file, "summary": r.summary_file,
                           "memory_requests": r.memory_requests_file,
                           "items_sha256": r.manifest["items_sha256"], "n_items": r.manifest["n_items"]} for r in runs],
                 "price_book": book_identity(book, runs, book_label), "billing_reconciliation": billing,
                 "panel_sha256": panel.sha256, "code_sha256": code_digest(),
                 "bootstrap": {"replicates": REPLICATES, "level": LEVEL, "interval": "percentile",
                               "clusters": "L3 unit / S2, S1, L2 pair / S3 document",
                               "sensitivity_clusters": "document family; with a spec also template family and "
                                                       "document",
                               "models": "resampled within tier for tier estimates",
                               "repeats": "a cell is the mean of its reads before the bootstrap; replicates never "
                                          "resample repeats"},
                 "thresholds": {"h_t_min_diff": cap.HT_MIN_DIFF, "binds_gap": cap.BINDS_GAP,
                                "escape_gap": cap.ESCAPE_GAP, "escape_token_ratio": cap.ESCAPE_TOKEN_RATIO,
                                "local_budget_max": cap.LOCAL_BUDGET_MAX, "records_min_diff": cap.RECORDS_MIN_DIFF,
                                "h_s_control_max_diff": cap.HS_CONTROL_MAX_DIFF, "n_eff_min": cap.N_EFF_MIN,
                                "empty_void_share": cap.EMPTY_VOID_SHARE,
                                "supply_l3": cap.SUPPLY_L3, "supply_s2_pairs": cap.SUPPLY_S2_PAIRS},
                 "h_s": {"main_scope": cap.HS_MAIN_SCOPE, "control_scope": cap.HS_CONTROL_SCOPE,
                         "readings": list(cap.HS_READINGS), "criterion": list(cap.HS_CRITERION)},
                 "reference": {"profile_id": reference, "condition": cap.REFERENCE_CONDITION,
                               "readable": ref is not None},
                 "load_bearing": {"tier": cap.CONSTRAINED_TIER, "server_tier": cap.SERVER_TIER,
                                  "scope": lb_scope, "tiers": {t: sorted(p for p, x in grid.tier_of.items() if x == t)
                                                               for t in sorted(set(grid.tier_of.values()))}},
                 "amortise_n": list(AMORTISE_N), "codec_map": CODEC, "self_checks": checks},
        "status": _status(grid, supply, pairs, pending, billing, ref, coverage,
                          len(spec_check["missing"]) if spec is not None else None, small,
                          [p["id"] for p in primary or [] if not p["computable"]]),
        "primary": primary,
        **({"sensitivity_callout": _callouts(primary), "reference_arm_outcomes": _reference_arm_outcomes(grid, spec),
            "memory_sensitivities": _memory_sensitivities(obs, spec, populations, readback)}
           if paper_rules else {}),
        "empty_memory": _empty_memory(grid, obs),
        "absences": _absences(obs),
        "capability": capability_rows, "gap": gaps, "gap_tier": tier_gaps, "outcomes": outcomes,
        "dgap": pairs, "dgap_tier": tier_pairs, "interaction": interactions, "h_s": h_s,
        "reasoning_field_mapping": mapping,
        "sustainability": sus, "builds": builds, "access": acc_rows, "energy": energy(sus, bands),
        "taxonomy": {"all": taxonomy(obs, lambda o: (o.profile_id, o.condition) not in void),
                     "L3": taxonomy(obs, lambda o: o.stratum == "L3" and (o.profile_id, o.condition) not in void)},
    }
    return report


def report_bytes(report: dict) -> bytes:
    return (json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8")


def write_report(report: dict, out_dir: Path, label: str) -> tuple[Path, Path]:
    data = report_bytes(report)
    stem = f"{label}_report"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / f"{stem}.json", out_dir / f"{stem}.md"
    json_path.write_bytes(data)
    md_path.write_bytes(render_markdown(report, stem).encode("utf-8"))
    return json_path, md_path


# ---------------------------------------------------------------------- markdown

def _iv(e: dict | None) -> str:
    if not e or e.get("est") is None:
        return "—"
    return f"{e['est']:.3f}" + (f" [{e['ci'][0]:.3f}, {e['ci'][1]:.3f}]" if e.get("ci") else " [no CI]")


def _cond(row: dict) -> str:
    rep_budget = row["condition"].split("/")[0]
    return rep_budget + (f" ({row['arm']})" if row.get("builder") else "")


def _num(x, fmt: str = ",.0f") -> str:
    return "—" if x is None else format(x, fmt)


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |"
                                                                            for r in rows]


def render_markdown(report: dict, stem: str) -> str:
    st, meta = report["status"], report["meta"]
    out = [f"# Analysis {stem}", "",
           f"Status **{st['label']}**; runs {', '.join(r['run_id'] for r in meta['runs'])}; price book "
           f"`{meta['price_book']['path']}` ({meta['price_book']['label']}); billing reconciliation "
           f"{meta['billing_reconciliation']}; bootstrap "
           f"{meta['bootstrap']['replicates']} replicates, 95% percentile, clusters = {meta['bootstrap']['clusters']}.",
           ""]
    out += [f"- {r}" for r in st["reasons"]] + [""]
    if report["primary"] is not None:
        out += _primary_md(report)
    out += _reference_arm_md(report)
    out += _memory_sensitivities_md(report.get("memory_sensitivities"))
    out += _supply_md(st["supply"])
    out += _empty_memory_md(report["empty_memory"])
    out += _absences_md(report["absences"])
    if st["panel_deviations"]:
        out += ["## Panel deviations (listed, not voided)", ""]
        out += _table(["reader", "condition", "answered", "answer from reasoning field", "reasoning leaks",
                       "empty (E-parse)"],
                      [[d["profile_id"], d["condition"].split("/")[0], str(d["answered"]),
                        str(d["from_reasoning_field"]), str(d["reasoning_leaks"]), str(d["empty"])]
                       for d in st["panel_deviations"]]) + [""]
    if report["reasoning_field_mapping"]:
        out += ["## Answer taken from the reasoning field: both readings", "",
                "Mapped arm = the tables below. Without the mapping the same cells are empty (E-parse, counted wrong); "
                "that arm is void when more than 10% of its answered cells are empty.", ""]
        out += _table(["reader", "condition", "answered", "from reasoning field", "accuracy (all), mapped",
                       "accuracy (all), without mapping", "empty without mapping", "void without mapping"],
                      [[m["profile_id"], m["condition"].split("/")[0], str(m["answered"]),
                        str(m["from_reasoning_field"]),
                        _iv((m["mapped"] or {}).get("all")) if m["mapped"] else f"void: {m['void_mapped']}",
                        _iv(m["unmapped"].get("all")), f"{m['empty_unmapped']} / {m['answered']}",
                        "yes" if m["void_unmapped"] else "no"]
                       for m in report["reasoning_field_mapping"]]) + [""]
    scopes = ["all", "L3-abstracted", "L3-transparent", "S2", "S3", "S1"]
    present = [s for s in scopes if any(s in c["scopes"] for c in report["capability"])]
    if report["primary"] is not None and any("primary" in c["scopes"] for c in report["capability"]):
        present = ["primary", *present]
    out += ["## Capability: accuracy [95% CI]" + _exploratory(report), ""]
    out += _table(["reader", "condition", *present, "lettered, chance-corrected (all)"],
                  [[c["profile_id"], _cond(c) + (" †" if c.get("k1_marked") else ""),
                    *(_iv(c["scopes"].get(s)) for s in present),
                    _iv((c["scopes"]["all"].get("lettered") or {}).get("chance_corrected"))]
                   for c in report["capability"]])
    sizes = {s: max((c["scopes"][s]["n"], c["scopes"][s]["clusters"]) for c in report["capability"]
                    if s in c["scopes"]) for s in present}
    out += ["", "Largest n per scope (items / clusters): " + ", ".join(f"{s} {n} / {k}" for s, (n, k) in sizes.items()),
            ""]
    if any("k1_marked" in c for c in report["capability"]):
        out += ["† outside the k = 1 rule's scope (a reference arm, or a condition outside the 4k slice): its estimates "
                "stay k = 1.", ""]
    if report["gap"]:
        out += [f"## Gap = acc({meta['reference']['profile_id']} M0@full) − acc(reader, condition)"
                + _exploratory(report), ""]
        lb = meta["load_bearing"]["scope"]
        rows = defaultdict(dict)
        for g in report["gap"]:
            rows[(g["profile_id"], _cond(g))][g["scope"]] = g
        def disc(g):
            return str((g or {}).get("n_discordant", "—"))
        out += _table(["reader", "condition", "all", "all, family clusters", lb, f"n_discordant (all / {lb})",
                       "token ratio"],
                      [[p, c, _iv(v.get("all")), _iv((v.get("all") or {}).get("family")), _iv(v.get(lb)),
                        f"{disc(v.get('all'))} / {disc(v.get(lb))}",
                        _num((v.get("all") or {}).get("token_ratio"), ".3f")] for (p, c), v in rows.items()])
        out += ["", "### Tier random effect (mean over the tier's models; models resampled within tier; tiers "
                + "; ".join(f"{t} = {len(p)} readers" for t, p in meta["load_bearing"]["tiers"].items()) + ")", ""]
        out += _table(["tier", "condition", "scope", "models", "Gap", "Gap, family clusters"],
                      [[g["tier"], g["condition"].split("/")[0], g["scope"], _models(g), _iv(g),
                        _iv(g.get("family"))] for g in report["gap_tier"]])
        out += ["", "### Trilemma outcome (local representations, budget ≤ 32k)", ""]
        out += _table(["scope", "unit", "outcome", "escapable by"],
                      [[s, u, o["outcome"], ", ".join(o["escapable_by"]) or "—"]
                       for s, v in report["outcomes"].items()
                       for u, o in [*v["readers"].items(), *((f"tier {t}", x) for t, x in v["tiers"].items())]])
        out += [""]
    if report["dgap"]:
        out += ["## ΔGap between builder arms = acc(higher arm) − acc(lower arm), same reader and budget"
                + _exploratory(report), ""]
        out += _table(["reader", "lower → higher", "scope", "ΔGap", "ΔGap, family clusters", "n_discordant", "H_T"],
                      [[p["profile_id"], f"{p['lower'].split('/')[0]} ({p['arms'][0]}) → "
                        f"{p['higher'].split('/')[0]} ({p['arms'][1]})", p["scope"], _iv(p["dgap"]),
                        _iv(p.get("family")), str(p["n_discordant"]), p.get("h_t", "—")] for p in report["dgap"]])
        out += ["", "### Tier random effect and tier interaction", ""]
        out += _table(["tier", "lower → higher", "scope", "models", "ΔGap", "ΔGap, family clusters",
                       "between-model SD"],
                      [[t["tier"], f"{t['lower'].split('/')[0]} → {t['higher'].split('/')[0]}", t["scope"],
                        _models(t), _iv(t["dgap"]), _iv(t.get("family")),
                        _num(t["between_model_sd"], ".3f")] for t in report["dgap_tier"]])
        if report["interaction"]:
            out += [""] + _table(["lower → higher", "scope",
                                  f"ΔGap({meta['load_bearing']['tier']}) − ΔGap({meta['load_bearing']['server_tier']})",
                                  "family clusters"],
                                 [[f"{x['lower'].split('/')[0]} → {x['higher'].split('/')[0]}", x["scope"],
                                   _iv(x["interaction"]), _iv(x.get("family"))] for x in report["interaction"]])
        out += [""]
    if report["h_s"]:
        hs = meta["h_s"]
        out += ["## H_S: branch errors (E-cond + E-ver + E-exc), LLM-built − zero-LLM" + _exploratory(report), "",
                f"Primary reading: error-rate difference over all cells of {hs['main_scope']}; secondary: difference "
                f"in the share of branch errors among wrong cells. Control: {hs['control_scope']} (|rate diff| < 5pp; "
                f"S3 is an empty test under the frozen scorer). Pre-registered comparison: "
                f"{hs['criterion'][0]}@{hs['criterion'][1]}; other LLM-built conditions are descriptive.", ""]
        rows = []
        for h in report["h_s"]:
            for z, m, c in zip(h["zero_llm"], h["main"], h["control"]):
                rows.append([h["profile_id"], f"{h['llm'].split('/')[0]} − {z}", "yes" if h["pre_registered"] else "no",
                             _iv(m.get("rate_diff")), _iv(m.get("family")), _iv(m.get("share_diff")),
                             _iv(c.get("rate_diff")), f"{h['verdict']['rate_diff']} / {h['verdict']['share_diff']}"])
        out += _table(["reader", "comparison", "pre-registered", f"rate diff ({hs['main_scope']}, primary)",
                       "rate diff, family clusters", "share-of-errors diff (secondary)",
                       f"rate diff ({hs['control_scope']} control)", "verdict (rate / share)"], rows) + [""]
    out += ["## Sustainability: provider tokens per query", ""]
    ns = ["10", "100", "1000", "10000"]
    out += _table(["reader", "condition", "input uncached", "input cached", "output", "reasoning (reported / est.)",
                   "compression tokens / query", "build tokens per memory", *(f"total @N={n}" for n in ns)],
                  [[s["profile_id"], _cond({**s, "builder": s["build"] or s["query"]}),
                    _num(s["per_query"]["input_uncached"]),
                    _num(s["per_query"]["input_cached"]), _num(s["per_query"]["output"]),
                    f"{_num(s['per_query']['reasoning'])} ({s['calls_reporting']['reasoning']}) / "
                    f"{_num(s['per_query']['reasoning_estimate'])} ({s['calls_reporting']['reasoning_estimate']})",
                    _num((s["query"] or {}).get("per_query_total")),
                    _num((s["build"] or {}).get("tokens_per_memory", {}).get("mean")),
                    *(_num((s["amortised_per_query"] or {}).get(n)) for n in ns)] for s in report["sustainability"]])
    out += _bill_md(report["builds"])
    out += ["", "## Access", ""]
    out += _table(["reader", "condition", "tier", "ceiling", "feasible / cells", "$ / 1k read",
                   "$ / 1k compression", "$ / 1k @N=100", "$ / 1k @N=1000", "build $ / memory",
                   "recorded $ / 1k read", "Gap (all)"],
                  [[a["profile_id"], a["condition"].split("/")[0], a["tier"], _num(a["local_context_ceiling"]),
                    f"{a['feasible']} / {a['cells']}", _num(a["usd_per_1k_read"], ".4f"),
                    _num(a["usd_per_1k_query_memory"], ".4f"),
                    _num((a["usd_per_1k_at_N"] or {}).get("100"), ".4f"),
                    _num((a["usd_per_1k_at_N"] or {}).get("1000"), ".4f"), _num(a["build_usd_per_memory"], ".4f"),
                    _num(a["usd_per_1k_read_recorded"], ".4f"), _iv(a["gap_all"])] for a in report["access"]])
    en = report["energy"]["by_N"]
    stated = sum(len(v["rank_statements"]) for v in en.values())
    banded = sum(len(v["joules_per_query"]) for v in en.values())
    out += ["", "## Energy (rank-only)", "",
            f"Band file `{report['energy']['bands']['path']}`: {banded} banded condition-N cells, {stated} rank "
            "statements" + ("; no band is set, so no energy statement is made." if not banded else "."), ""]
    out += ["## Error taxonomy (all answered cells; Context Codec roll-up)", ""]
    classes = [c for c in CODEC_CLASSES if c not in NOT_IDENTIFIABLE]
    out += _table(["reader", "condition", "n", "correct", *classes, "pending arb.", "polarity flips (overlay)"],
                  [[t["profile_id"], t["condition"].split("/")[0], str(t["n"]), str(t["correct"]),
                    *(str(t["codec"][c]) for c in classes), str(t["pending_arbitration"]), str(t["polarity_flips"])]
                   for t in report["taxonomy"]["all"]])
    out += ["", "Weakening is not identifiable by the frozen scorer (values and options only). E-exc (the general rule "
            "answered where the exception applies) rolls up to scope: a lost exception carve-out is a scope error in "
            "Codec's terms, but Codec's scope class is broader, so the roll-up is interpretive.",
            ""]
    return "\n".join(out)


def _models(row: dict) -> str:
    return str(len(row["models"])) + (" (AMBER: too few models)" if row.get("small_tier") else "")


def _exploratory(report: dict) -> str:
    return " (exploratory)" if report["primary"] is not None else ""


def _primary_md(report: dict) -> list[str]:
    spec = report["meta"]["spec"]
    rows = report["primary"]
    out = [f"## Primary comparisons (spec `{spec['path']}`; population {spec['population_cells']} cells; every other "
           "table is exploratory)", "",
           "dGap = acc(to) − acc(from) = Gap(from) − Gap(to), same readers; tier estimates average the tier's readers "
           "(resampled within tier). Scope = the spec's population.", ""]
    flags = any(p.get("k1") or p.get("small_tiers") for p in rows)
    out += _table(["id", "comparison", "readers", "n / clusters", "estimate [95% CI]", *(["flags"] if flags else [])],
                  [[p["id"], p["question"], str(len((p["estimate"] or {}).get("models", []))),
                    f"{p['estimate']['n']} / {p['estimate']['clusters']}" if p["estimate"] else "—",
                    _shown(p["kind"], p["estimate"]) if p.get("computable", True) else "not computable",
                    *([_flags(p)] if flags else [])] for p in rows])
    lacking = [p for p in rows if p.get("missing")]
    if lacking:
        out += ["", "### Inputs the analysed runs lack (a computed estimate rests on the rest)", ""]
        out += [f"- {p['id']}{'' if p.get('computable', True) else ' (not computable)'}: " + "; ".join(p["missing"])
                for p in lacking]
    names = list(rows[0]["sensitivities"]) if rows else []
    out += ["", "### Sensitivities of the primary comparisons (estimate [95% CI]; a verdict comparison: its verdict)",
            ""]
    out += _table(["sensitivity", *(p["id"] for p in rows)],
                  [[name, *(_shown(p["kind"], p["sensitivities"][name], short=True) for p in rows)] for name in names])
    if "sensitivity_callout" in report:
        out += ["", "### Sensitivities that move a primary comparison (named in the text)", ""]
        out += [f"- {c['id']}: {', '.join(c['sensitivities'])}" for c in report["sensitivity_callout"]] or [
            "No sensitivity flips a primary comparison's sign, makes its interval span 0 or changes its verdict."]
    return out + [""]


def _flags(p: dict) -> str:
    out = ["k = 1 (reference arm)"] if p.get("k1") else []
    out += [f"AMBER: tier {t} on {p['estimate']['tier_models'][t]} models" for t in p.get("small_tiers", [])]
    return "; ".join(out) or "—"


def _conds(conditions) -> str:
    """Conditions named short and still apart: the builder / compressor by its slot ("M7@4000/C3",
    "M6@4000/{reader}")."""
    return ", ".join(c if "/" not in c else f"{c.split('/')[0]}/{c.split('/', 1)[1].split('_')[0]}" for c in conditions)


def _shown(kind: str, e: dict | None, short: bool = False) -> str:
    """A primary estimate in one cell: an interval, the reference rule's choice, an outcome or a verdict."""
    if e is None:
        return "—"
    if kind == "reference_rule":
        return f"{e['chosen']}: {_iv(e)}"
    if kind == "tier_outcome":
        by = f" by {_conds(e['escapable_by'])}" if e["escapable_by"] else ""
        return e["outcome"] + by
    if kind in ("h_m7", "h_s"):
        if short:
            return e["verdict"]
        legs = "; ".join(f"vs {z}: {_iv(x)}" for z, x in e["legs"].items())
        return f"{e['verdict']} ({legs})"
    return _iv(e)


def _reference_arm_md(report: dict) -> list[str]:
    arms = report.get("reference_arm_outcomes")
    if not arms:
        return []
    out = ["## Trilemma outcome against the A reference arm (k = 1, reported apart)", ""]
    for pid, got in arms.items():
        if got is None:
            out += [f"{pid}: M0@full not read.", ""]
            continue
        rows = [[f"tier {t}", "—" if o is None else o["outcome"],
                 "—" if o is None else _conds(o["escapable_by"]) or "—",
                 "—" if o is None else str(o["tier_models"][t])] for t, o in got["tiers"].items()]
        rows += [[p, o["outcome"], _conds(o["escapable_by"]) or "—", "1"]
                 for p, o in got["readers"].items()]
        out += [f"Reference {pid} M0@full, scope {got['scope']}:", ""]
        out += _table(["unit", "outcome", "escapable by", "models"], rows) + [""]
    return out


def _memory_sensitivities_md(ms: dict | None) -> list[str]:
    if not ms:
        return []
    out = ["## Memory-level sensitivities (exploratory; not primary, not in the spec's list)", ""]
    for name, title in (("m5_build_truncated", "M5 cells with a build_truncated memory dropped"),
                        ("m7_located", "revisions every builder located at >= 0.90")):
        s = ms[name]
        if not s.get("computed"):
            out += [f"{title}: not computed ({s['reason']}).", ""]
            continue
        if name == "m5_build_truncated":
            head = (f"{title}: {s['build_truncated_memories']} of {s['memories']} M5 memories build_truncated "
                    f"({s['build_truncated_memories_read']} read here); answered cells dropped "
                    + (", ".join(f"{c.split('/')[0]} {n}" for c, n in s["answered_cells_dropped"].items()) or "none")
                    + f"; population items touched {s['population_items_dropped']}.")
        else:
            head = (f"{title}: {s['revisions_kept']} of {s['revisions']} revisions kept (below the line: "
                    + ", ".join(f"{b} {n}" for b, n in s["below_threshold"].items())
                    + f"); population cells kept {s['population_cells_kept']} of {s['population_cells']}.")
        out += [head, ""]
        out += _table(["comparison", "estimate [95% CI] or verdict"],
                      [[cid, _shown("h_s" if cid == "H_S" else "tier_dgap", e) if e else "not computable"]
                       for cid, e in s["comparisons"].items()]) + [""]
    return out


def _empty_memory_md(empty: dict) -> list[str]:
    out = ["## Empty memories (read and kept in their condition)", ""]
    if not empty["by_condition"]:
        return out + ["No answered cell had an empty memory.", ""]
    out += _table(["reader", "condition", "answered", "empty memory"],
                  [[r["profile_id"], r["condition"].split("/")[0], str(r["answered"]), str(r["memory_empty"])]
                   for r in empty["by_condition"]])
    return out + ["", f"{empty['cells']} cells; {empty['distinct_read_requests']} distinct read requests (a read "
                  "shared by several budgets counted once).", ""]


def _absences_md(absent: dict) -> list[str]:
    out = ["## Cells not sent (absences by reason: route context, route timeout, ceilings)", ""]
    if not absent["by_condition"]:
        return out + ["Every cell was sent.", ""]
    out += _table(["reader", "condition", "not sent", "by reason"],
                  [[r["profile_id"], r["condition"].split("/")[0], str(r["not_sent"]),
                    ", ".join(f"{k} {n}" for k, n in r["by_reason"].items())] for r in absent["by_condition"]])
    return out + ["", f"{absent['cells']} cells: " + ", ".join(f"{k} {n}" for k, n in absent["by_reason"].items())
                  + ".", ""]


def _supply_md(supply: dict) -> list[str]:
    def line(name: str, got: dict | None) -> list[str]:
        if got is None:
            return []
        l3 = got["L3"]
        return [f"| {name} | {got['items']} | {l3['units']} | {l3['cells']} | {l3['families']} | "
                f"{l3['max_units_per_doc']} | {l3['literal_units']} = {l3['literal_share']} | "
                f"{l3['cell_kind_share']} | {l3['domain_fallback_units']} / {l3['literal_fallback_units']} | "
                f"{l3['abstracted_units_other_domains']} | {'yes' if l3['met'] else 'no'} | {got['S2']['pairs']} |"]
    out = ["## Supply check (load-bearing ops + technical abstracted L3 units, the authors' literal rulings; counted "
           "on the pool)", "",
           "| items | eligible items | units | cells | families | max units / doc | literal units = share | "
           "diagnostic: cell_kind share | earlier-batch fallback units (domain / literal) | abstracted units, other "
           "domains | L3 check met | S2 pairs |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    out += line("pool", supply["pool"]) + line("analysed subset", supply["analysed"])
    out += line("spec population", supply.get("population"))
    counted = supply["pool"] or supply["analysed"]
    sens = counted["L3"]["sensitivity"]
    out += ["", "Reconciliation sensitivities (" + ("pool" if supply["pool"] else "analysed subset") + "; the verdict "
            "stays on the main count): " + "; ".join(f"{k} {v['literal_units']}/{v['units']} = {v['literal_share']}"
                                                     for k, v in sens.items())]
    sources = supply.get("pool_sources") or []
    if sources:
        out += ["", "Pool files: " + ", ".join(f"`{x['path']}` ({x['sha256'][:12]})" for x in sources)]
    return out + [""]


def _bill_md(builds: dict) -> list[str]:
    if not builds["bill"]:
        return []
    out = ["", "### LLM-built memory calls, bill view (each answered ledger record once; imported records apart)", ""]
    out += _table(["run", "ledger", "calls (this run)", "imported", "tokens", "re-priced $", "recorded usd_counted",
                   "unattributed"],
                  [[run_id, stem, str(b["calls"]), str(b["imported_records"]), _num(b["tokens"]["total"]),
                    _num(b["usd"], ".4f"), _num(b["usd_counted_recorded"], ".4f"),
                    str(builds["unattributed_calls"][run_id][stem])]
                   for run_id, ledgers in builds["bill"].items() for stem, b in ledgers.items()])
    return out
