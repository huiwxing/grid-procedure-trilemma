"""Capability table and the pre-registered tests. Every threshold here was fixed before any model output was
read; nothing here is tuned on results.

`Grid(obs)`: per (reader profile, condition) the answered cells, averaged over repeats per item: correctness, the
branch-error indicator (E-cond / E-ver / E-exc, for the summary hypothesis H_S below), wrongness and
reference-tokenizer input tokens. Some readers return their answer only in the reasoning field; the reasoning-field
mapping takes the answer from there when that field holds exactly one answer object. An empty cell (blank content
after that mapping) is scored E-parse like any unparseable answer; the condition is void (listed, not read) only
when more than EMPTY_VOID_SHARE of its answered cells are empty. Cells answered from the reasoning field and
reasoning leaks are counted per condition as a panel deviation, never a void. `unmapped_accuracy` is the reading
without that mapping (reported beside the mapped one): a mapped cell has blank content, which the frozen scorer
scores unparseable, so it counts wrong and empty; `Cond.void_unmapped` applies the void rule to those counts.

Every interval is a cluster bootstrap over the primary clusters (L3 unit, S2 / S1 / L2 pair, S3 document); `by`
recomputes an estimate with other clusters, the sensitivities: "family" (the document family, i.e. the procedure
owner: the second level), "template_family" (units of different documents that instantiate one rule template,
merged into one cluster) and "document". A cell's value is the mean over its reads before the bootstrap; replicates
never resample repeats (`repeat_coverage` counts the cells with at least 2 reads).

Scopes are the SCOPES below plus the named item-id sets a Grid is given (`populations`: the spec's primary
population and its sensitivity variants, procedures.analysis.spec). Tiers are the analysis tiers of
procedures.analysis.spec.tier_group (B = open weights on a multi-GPU server; C_load_bearing = open weights on one
24 GB or 48 GB card, access tiers C1 u C2; C3 = one 80 GB card, on its own; A = cloud frontier models), or the
spec's frozen profile lists; a condition may name "{reader}" (procedures.analysis.spec.READER), resolved per reader
(a query-focused compression, M6, made by the reader itself).

* `accuracy`: cluster-bootstrap accuracy per scope, and on the lettered (side-form) cells the chance-corrected
  accuracy (correct - chance) / (n - chance) with the frozen scorer's chance (as the run summaries compute it).
* `gap`: Gap(reader, condition) = acc(reference, M0@full) - acc(reader, condition) on the same items, M0@full being
  the full text of the revisions in effect; the reference is one named reader, REFERENCE_PROFILE at M0@full (the B1
  reader, DeepSeek-V4.1-Flash); no maximum over several references.
* `pair`: dGap between two builder arms for one reader at one budget: dGap = Gap(lower arm) - Gap(higher arm)
  = acc(higher) - acc(lower), arms ordered zero-LLM < C-built < B-built < A-built (the builder hypothesis H_T:
  positive = the build by the more capable, frontier-side model narrows the gap; `h_t` is "effective" when |dGap| >=
  HT_MIN_DIFF with the interval excluding 0). n_discordant = items where the two arms differ (n_eff, the effective
  sample size of the supply / power check: below N_EFF_MIN a comparison is descriptive only).
* `tier_pair` / `tier_gap`: the same with reader models as a random effect: the tier estimate is the mean over the
  tier's models; each replicate draws the clusters (shared) and the models (with replacement).
  `interaction`: the constrained panel's dGap (tier C_load_bearing = access tiers C1 and C2) minus the server panel's
  (tier B) (models drawn within each tier): whether a memory narrows the gap more for the single-card readers than for
  the server readers.
* `outcome`: the pre-registered three-outcome rule over the local representations (budget <= 32k, the C readers'
  context ceiling): binds = every one has Gap >= 10pp with CI low > 0 (no local memory closes the gap); escapable =
  some one has CI high <= 5pp at <= 1/4 of the reference's input tokens (counted by one reference tokenizer, o200k,
  for every reader); else indeterminate.
* `h_s`: the summary hypothesis H_S: branch errors of an LLM-built memory minus a zero-LLM one at the same budget
  (positive = the LLM-built memory loses more branches). Primary reading: the error-rate difference over all cells;
  secondary: the difference in the share of branch errors among wrong cells. Main pool HS_MAIN_SCOPE (L2 + L3),
  control HS_CONTROL_SCOPE (S2): the S3 control first planned is an empty test under the frozen scorer (an S3 cell
  has no counterpart, so no branch error can occur) and is replaced by S2.
* `supply`: the supply / power check, keys L3 (units) and S2 (pairs), of an item pool (counted on the full admitted
  pool; the analysed subset is reported beside it). L3 counts on the authors' manual rulings recorded per cell
  (k6_exception_literal, domain, load_bearing, rule2_spirit): eligible L3 cells that are load_bearing, abstracted
  (rule2_spirit) and in the ops / technical domains; a unit is literal when a cell's `k6_exception_literal` ruling is
  exception or not_in_effect. Cells of the L3 item batches 1-2, which carry no such fields, use the
  batch-3 counting rule: load_bearing, the domain from EARLIER_BATCH_DOMAIN by document, literal by cell kind or a
  not_in_effect gold; the fallbacks are counted. The pre-ruling count (cell_kind exception or not_in_effect gold) is
  kept as a diagnostic column, and the reconciliation sensitivities (the count again without the units of each ruling
  tag) are recomputed from the tags, with the pin / normative medium-confidence drops and both readings of the cue
  pool.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from procedures.analysis.data import ARM_RANK, Obs
from procedures.analysis.spec import CONSTRAINED_TIER, MAIN_ANALYSIS, READER, SERVER_TIER, UNIT_TAGS, tier_group
from procedures.analysis.stats import boot, interval, model_weights, r4

HT_MIN_DIFF = 0.05              # H_T: |dGap| >= 5pp with the CI excluding 0
RECORDS_MIN_DIFF = 0.05         # H_M7 (the records hypothesis; a pair's "records" key): M7 - M2 / M4 >= 5pp, CI low > 0
BINDS_GAP = 0.10
ESCAPE_GAP = 0.05
ESCAPE_TOKEN_RATIO = 0.25
LOCAL_BUDGET_MAX = 32_768       # a local representation: budget <= 32k (the C readers' context ceiling)
HS_CONTROL_MAX_DIFF = 0.05      # H_S control: |difference| < 5pp
HS_READINGS = ("rate_diff", "share_diff")          # primary, secondary
HS_CRITERION = ("M5", 4000)     # the pre-registered H_S names M5 at 4k; other LLM-built conditions: descriptive
N_EFF_MIN = 30                  # power check: fewer discordant items -> descriptive only
EMPTY_VOID_SHARE = 0.10         # an arm with a larger share of empty answered cells is void
SUPPLY_S2_PAIRS = 75
# counted on the ops + technical abstracted units of the main analysis, literal by the manual ruling
SUPPLY_L3 = {"units": 60, "cells": 120, "families": 4, "max_units_per_doc": 15, "literal_share": 0.5}
SUPPLY_L3_DOMAINS = MAIN_ANALYSIS["domain"]
SUPPLY_L3_LAYERS = MAIN_ANALYSIS["rule2_spirit"]
SUPPLY_LITERAL = ("exception", "not_in_effect")
# L3 batches 1-2 carry no domain field: their abstracted units are labelled by document (an interpretation; the MISO
# BPM-015 / 020 / 030 units are market / planning, not counted, and stay unlabelled here)
EARLIER_BATCH_DOMAIN = {"M-03": "ops", "NERC-PRC-023": "technical"}
BRANCH_ERRORS = ("E-cond", "E-ver", "E-exc")
REFERENCE_CONDITION = "M0@full"
REFERENCE_PROFILE = "B1_dsv41flash_hf_deepinfra"   # the named reference reader (B1, DeepSeek-V4.1-Flash)
# These strings are part of the bootstrap seeds (procedures.analysis.stats.seed_of) and of the frozen spec: renaming
# one moves every interval of the report.
ABSTRACTED_SCOPE = "L3-abstracted"          # the main scope without a spec (abstracted L3 cells); with one, POPULATION
POPULATION = "primary"
CLUSTERINGS = ("unit", "family", "template_family", "document")    # primary; sensitivities
SCOPES = {
    "all": lambda m: True,
    "L3": lambda m: m.stratum == "L3",
    "L3-abstracted": lambda m: m.layer == "abstracted",
    "L3-transparent": lambda m: m.layer == "transparent",
    "S2+L2+L3": lambda m: m.stratum in ("S2", "L2", "L3"),      # H_T pool
    "L2+L3": lambda m: m.stratum in ("L2", "L3"),               # H_S main pool
    "S2": lambda m: m.stratum == "S2",
    "S3": lambda m: m.stratum == "S3",
    "S1": lambda m: m.stratum == "S1",
    "L2": lambda m: m.stratum == "L2",
}
OUTCOME_SCOPES = ("L3-abstracted", "L3", "S2+L2+L3", "all")
HS_MAIN_SCOPE, HS_CONTROL_SCOPE = "L2+L3", "S2"


def resolve(condition: str, profile_id: str) -> str:
    return condition.replace(READER, profile_id)


@dataclass(frozen=True)
class Cond:
    profile_id: str
    condition: str
    tier: str
    representation: str
    budget: int | None
    builder: str | None
    arm: str
    void: str | None
    answered: int = 0
    empty: int = 0                  # answered cells with blank content after the reasoning-field mapping (E-parse)
    from_reasoning: int = 0         # answered cells whose answer came from the reasoning field
    leaks: int = 0                  # answered cells the infer layer flagged reasoning_leak
    memory_empty: int = 0           # answered cells whose memory was empty (a legitimate outcome: read, not voided)

    @property
    def key(self) -> tuple[str, str]:
        return self.profile_id, self.condition

    @property
    def void_unmapped(self) -> str | None:
        """The void rule on the cells as scored without the reasoning-field mapping (mapped cells are empty)."""
        return void_reason(self.answered, self.empty + self.from_reasoning)

    @property
    def local(self) -> bool:
        return self.budget is not None and self.budget <= LOCAL_BUDGET_MAX

    @property
    def readable(self) -> bool:
        """Not void and with at least one answered cell (a condition whose every cell was recorded and not sent, e.g.
        a reader's M6 whose compressions all timed out, is an absence, not a condition of no items)."""
        return self.void is None and self.answered > 0


@dataclass(frozen=True)
class ItemMeta:
    stratum: str
    layer: str | None
    cluster: str
    family: str
    chance: float
    lettered: bool
    doc: str | None
    template_family: str | None


def _mean(xs) -> float:
    return float(np.mean(xs))


def _verdict(est: dict, min_diff: float) -> str:
    if est["ci"] is None:
        return "no interval"
    if abs(est["est"]) >= min_diff and (est["ci"][0] > 0 or est["ci"][1] < 0):
        return "effective"
    return "not detected"


def void_reason(answered: int, empty: int) -> str | None:
    if answered and empty / answered > EMPTY_VOID_SHARE:
        return (f"{empty} of {answered} answered cells are empty after the reasoning-field mapping "
                f"(> {EMPTY_VOID_SHARE:.0%})")
    return None


class Grid:
    def __init__(self, obs: list[Obs], tiers: dict[str, list[str]] | None = None,
                 populations: dict[str, frozenset] | None = None):
        """`tiers`: analysis tier -> reader profiles (the spec's frozen lists), default tier_group of each reader;
        `populations`: named item-id scopes beside SCOPES."""
        groups: dict[tuple, list[Obs]] = defaultdict(list)
        self.meta: dict[str, ItemMeta] = {}
        derived: dict[str, set] = defaultdict(set)
        for o in obs:
            groups[(o.profile_id, o.condition)].append(o)
            derived[tier_group(o.tier, o.access_tier)].add(o.profile_id)
            if not o.family:
                raise ValueError(f"{o.item_id}: item without a family (the sensitivity cluster)")
            self.meta.setdefault(o.item_id, ItemMeta(o.stratum, o.layer, o.cluster, o.family, o.chance, o.lettered,
                                                     o.doc, o.template_family))
        self.tier_of = {p: t for t, ps in (tiers or derived).items() for p in ps}
        self.populations = {k: frozenset(v) for k, v in (populations or {}).items()}
        clash = sorted(set(self.populations) & set(SCOPES))
        if clash:
            raise ValueError(f"population names {clash} are SCOPES names")
        self.clusters = {"unit": {i: m.cluster for i, m in self.meta.items()},
                         "family": {i: f"family:{m.family}" for i, m in self.meta.items()},
                         "template_family": {i: f"template:{m.template_family}" if m.template_family else m.cluster
                                             for i, m in self.meta.items()},
                         "document": {i: f"doc:{m.doc}" for i, m in self.meta.items()}}
        self.conds: dict[tuple, Cond] = {}
        self.values: dict[str, dict[tuple, dict]] = {"correct": {}, "correct_unmapped": {}, "branch_error": {},
                                                     "wrong": {}, "ref_tokens": {}}
        self.reads: dict[tuple, dict[str, int]] = {}
        for key, rows in sorted(groups.items()):
            first = rows[0]
            answered = [o for o in rows if o.observed]
            empty = sum(o.empty for o in answered)
            self.conds[key] = Cond(first.profile_id, first.condition, first.tier, first.representation, first.budget,
                                   first.builder, first.arm, void_reason(len(answered), empty), len(answered), empty,
                                   sum(o.from_reasoning for o in answered), sum(o.reasoning_leak for o in answered),
                                   sum(o.memory_empty for o in answered))
            per_item: dict[str, list[Obs]] = defaultdict(list)
            for o in answered:
                per_item[o.item_id].append(o)
            self.reads[key] = {i: len(v) for i, v in per_item.items()}
            self.values["correct"][key] = {i: _mean([o.correct for o in v]) for i, v in per_item.items()}
            self.values["correct_unmapped"][key] = {i: _mean([o.correct and not o.from_reasoning for o in v])
                                                    for i, v in per_item.items()}
            self.values["branch_error"][key] = {i: _mean([o.error_class in BRANCH_ERRORS for o in v])
                                                for i, v in per_item.items()}
            self.values["wrong"][key] = {i: _mean([not o.correct for o in v]) for i, v in per_item.items()}
            self.values["ref_tokens"][key] = {i: _mean([o.ref_input_tokens for o in v]) for i, v in per_item.items()
                                              if all(o.ref_input_tokens is not None for o in v)}

    # ------------------------------------------------------------------ selection

    @property
    def repeats(self) -> int:
        """The most reads of any answered cell."""
        return max((n for v in self.reads.values() for n in v.values()), default=0)

    def repeat_coverage(self, scope: str, conditions=None) -> dict:
        """Answered (reader, condition, item) cells of `scope` in readable conditions, and those read >= 2 times.
        `conditions` (the paper's spec withdraws the k = 1 warning on the 4k slice alone) keeps only some
        conditions: a collection of condition strings as the observations name them ("B0@full", "M0@full",
        "M2@4000", "M5@4000/<builder>", "M6@4000/<compressor>"; "{reader}" is resolved per reader, e.g.
        "M6@4000/{reader}" for a self-compressed M6), or a predicate on the Cond; None = every readable condition."""
        if conditions is None:
            chosen = self.readable()
        elif callable(conditions):
            chosen = [c for c in self.readable() if conditions(c)]
        else:
            if isinstance(conditions, str):
                raise TypeError("conditions is a collection of condition strings, not one string")
            named = set(conditions)
            chosen = [c for c in self.readable() if any(resolve(x, c.profile_id) == c.condition for x in named)]
        counts = [n for c in chosen for i, n in self.reads[c.key].items() if self._keep(scope, i)]
        return {"scope": scope, "cells": len(counts), "repeated": sum(n >= 2 for n in counts)}

    def _keep(self, scope: str, item_id: str) -> bool:
        if scope in self.populations:
            return item_id in self.populations[scope]
        return SCOPES[scope](self.meta[item_id])

    def scopes(self) -> list[str]:
        """The scopes tabulated: the population POPULATION (when given) and SCOPES, where they hold items; the other
        populations (sensitivity variants) are named explicitly by their callers."""
        listed = [POPULATION] if POPULATION in self.populations else []
        return [s for s in [*listed, *SCOPES] if any(self._keep(s, i) for i in self.meta)]

    def vec(self, key: tuple, scope: str, kind: str = "correct") -> dict:
        return {i: x for i, x in self.values[kind][key].items() if self._keep(scope, i)}

    def readable(self) -> list[Cond]:
        return [c for c in self.conds.values() if c.readable]

    def reference(self, profile_id: str = REFERENCE_PROFILE) -> tuple | None:
        """The named reference reader's M0@full when it is readable, else None (no Gap is computed)."""
        c = self.conds.get((profile_id, REFERENCE_CONDITION))
        return c.key if c is not None and c.readable else None

    def tiers(self) -> list[str]:
        return sorted({self.tier_of[c.profile_id] for c in self.readable() if c.profile_id in self.tier_of})

    def models(self, tier: str, *conditions: str) -> list[str]:
        """Profiles of an analysis tier with every given condition (resolved per profile) readable."""
        return sorted(p for p in {c.profile_id for c in self.readable() if self.tier_of.get(c.profile_id) == tier}
                      if all(self.conds.get((p, resolve(x, p))) is not None
                             and self.conds[(p, resolve(x, p))].readable for x in conditions))

    def arm_pairs(self, profile_id: str) -> list[tuple[str, str]]:
        """(lower arm condition, higher arm condition) at the same budget, arms of different rank."""
        mine = [c for c in self.readable() if c.profile_id == profile_id and c.arm in ARM_RANK]
        return sorted((x.condition, y.condition) for x in mine for y in mine
                      if x.budget == y.budget and ARM_RANK[x.arm] < ARM_RANK[y.arm])

    # ------------------------------------------------------------------ estimates

    def _boot(self, vectors: list[dict], key: str, by: str = "unit"):
        if by not in CLUSTERINGS:
            raise ValueError(f"cluster level must be one of {CLUSTERINGS}, got {by!r}")
        return boot(vectors, self.clusters[by], key if by == "unit" else f"{key}|by={by}")

    def accuracy(self, key: tuple, scope: str, by: str = "unit") -> dict:
        v = self.vec(key, scope)
        b = self._boot([v], f"acc|{scope}|{key}", by)
        out = {"n": len(b.items), "clusters": b.n_clusters,
               **interval(b.point[0], None if b.reps is None else b.reps[:, 0])}
        lettered = {i: x for i, x in v.items() if self.meta[i].lettered}
        chance = {i: self.meta[i].chance for i in lettered}
        lb = self._boot([lettered, chance], f"lettered|{scope}|{key}")
        if lb.items:
            def cc(m):
                return (m[..., 0] - m[..., 1]) / (1 - m[..., 1])
            out["lettered"] = {"n": len(lb.items), "chance_mean": r4(lb.point[1]), "acc": r4(lb.point[0]),
                               "chance_corrected": interval(cc(lb.point), None if lb.reps is None else cc(lb.reps))}
        return out

    def unmapped_accuracy(self, key: tuple, scope: str, by: str = "unit") -> dict:
        """Accuracy with the cells answered from the reasoning field counted wrong (the reading without the mapping)."""
        b = self._boot([self.vec(key, scope, "correct_unmapped")], f"acc_unmapped|{scope}|{key}", by)
        return {"n": len(b.items), "clusters": b.n_clusters,
                **interval(b.point[0], None if b.reps is None else b.reps[:, 0])}

    def _token_ratio(self, key: tuple, ref: tuple, items) -> float | None:
        q, f = self.values["ref_tokens"][key], self.values["ref_tokens"][ref]
        if not items or any(i not in q or i not in f for i in items):
            return None
        return r4(_mean([q[i] for i in items]) / _mean([f[i] for i in items]))

    def gap(self, key: tuple, scope: str, ref: tuple, by: str = "unit") -> dict:
        b = self._boot([self.vec(ref, scope), self.vec(key, scope)], f"gap|{scope}|{[ref]}|{key}", by)
        if not b.items:
            return {"n": 0, "est": None, "ci": None}
        reps = None if b.reps is None else b.reps[:, 0] - b.reps[:, 1]
        ref_v, mine = self.vec(ref, scope), self.vec(key, scope)
        return {"n": len(b.items), "clusters": b.n_clusters, **interval(b.point[0] - b.point[1], reps),
                "reference": list(ref), "n_discordant": sum(ref_v[i] != mine[i] for i in b.items),
                "token_ratio": self._token_ratio(key, ref, b.items)}

    def pair(self, profile_id: str, lower: str, higher: str, scope: str, by: str = "unit") -> dict:
        x, y = self.vec((profile_id, lower), scope), self.vec((profile_id, higher), scope)
        b = self._boot([x, y], f"pair|{scope}|{profile_id}|{lower}|{higher}", by)
        est = interval(b.point[1] - b.point[0], None if b.reps is None else b.reps[:, 1] - b.reps[:, 0])
        n_disc = sum(x[i] != y[i] for i in b.items)
        out = {"profile_id": profile_id, "lower": lower, "higher": higher,
               "arms": [self.conds[(profile_id, lower)].arm, self.conds[(profile_id, higher)].arm], "scope": scope,
               "n": len(b.items), "clusters": b.n_clusters, "dgap": est, "n_discordant": n_disc,
               "n_eff_ok": n_disc >= N_EFF_MIN}
        if b.items:
            out["h_t"] = _verdict(est, HT_MIN_DIFF)
            reps = (self.conds[(profile_id, lower)].representation, self.conds[(profile_id, higher)].representation)
            if reps[1] == "M7" and reps[0] in ("M2", "M4"):
                out["records"] = "pass" if est["ci"] and est["est"] >= RECORDS_MIN_DIFF and est["ci"][0] > 0 else "fail"
        return out

    def _model_draw(self, vecs: list[dict], key: str, groups: list[int], by: str = "unit"):
        """Crossed bootstrap over `vecs` (model arms laid out group by group): (boot, one weight matrix per group)."""
        b = self._boot(vecs, key, by)
        weights = [model_weights(b.rng, n) for n in groups] if b.reps is not None else None
        return b, weights

    def tier_pair(self, tier: str, lower: str, higher: str, scope: str, by: str = "unit",
                  kind: str = "correct") -> dict | None:
        models = self.models(tier, lower, higher)
        if not models:
            return None
        vecs = [self.vec((p, resolve(c, p)), scope, kind) for p in models for c in (lower, higher)]
        key = f"tier_pair|{scope}|{tier}|{lower}|{higher}" + ("" if kind == "correct" else f"|{kind}")
        b, weights = self._model_draw(vecs, key, [len(models)], by)
        diffs = b.point[1::2] - b.point[0::2]
        reps = None if weights is None else (weights[0] * (b.reps[:, 1::2] - b.reps[:, 0::2])).sum(axis=1)
        est = interval(diffs.mean(), reps) if b.items else {"est": None, "ci": None}
        return {"tier": tier, "lower": lower, "higher": higher, "scope": scope, "models": models,
                "n": len(b.items), "clusters": b.n_clusters, "dgap": est,
                "per_model": {p: r4(d) for p, d in zip(models, diffs)} if b.items else {},
                "between_model_sd": r4(np.std(diffs, ddof=1)) if len(models) > 1 and b.items else None,
                "model_level_resampled": len(models) > 1,
                "h_t": _verdict(est, HT_MIN_DIFF) if b.items else None}

    def interaction(self, lower: str, higher: str, scope: str, local: str = CONSTRAINED_TIER,
                    server: str = SERVER_TIER, by: str = "unit", kind: str = "correct") -> dict | None:
        """dGap of the local tier minus dGap of the server tier (the interaction of memory and access tier)."""
        groups = [self.models(t, lower, higher) for t in (local, server)]
        if not all(groups):
            return None
        vecs = [self.vec((p, resolve(c, p)), scope, kind) for g in groups for p in g for c in (lower, higher)]
        key = f"interaction|{scope}|{lower}|{higher}" + ("" if kind == "correct" else f"|{kind}")
        b, weights = self._model_draw(vecs, key, [len(g) for g in groups], by)
        if not b.items:
            return None
        n_local = len(groups[0])
        diffs = b.point[1::2] - b.point[0::2]
        point = diffs[:n_local].mean() - diffs[n_local:].mean()
        reps = None
        if weights is not None:
            d = b.reps[:, 1::2] - b.reps[:, 0::2]
            reps = (weights[0] * d[:, :n_local]).sum(axis=1) - (weights[1] * d[:, n_local:]).sum(axis=1)
        return {"lower": lower, "higher": higher, "scope": scope, "tiers": [local, server],
                "models": {local: groups[0], server: groups[1]}, "n": len(b.items), "clusters": b.n_clusters,
                "interaction": interval(point, reps)}

    def tier_gap(self, tier: str, condition: str, scope: str, ref: tuple, by: str = "unit") -> dict | None:
        models = self.models(tier, condition)
        if not models:
            return None
        vecs = [self.vec(ref, scope)] + [self.vec((p, resolve(condition, p)), scope) for p in models]
        b, weights = self._model_draw(vecs, f"tier_gap|{scope}|{tier}|{condition}|{[ref]}", [len(models)], by)
        if not b.items:
            return {"n": 0, "est": None, "ci": None, "models": models}
        gaps = b.point[0] - b.point[1:]
        reps = None
        if weights is not None:
            reps = (weights[0] * (b.reps[:, :1] - b.reps[:, 1:])).sum(axis=1)
        ratios = [self._token_ratio((p, resolve(condition, p)), ref, b.items) for p in models]
        return {"n": len(b.items), "clusters": b.n_clusters, **interval(gaps.mean(), reps), "models": models,
                "reference": list(ref), "model_level_resampled": len(models) > 1,
                "token_ratio": r4(_mean(ratios)) if all(x is not None for x in ratios) else None}

    def h_s(self, profile_id: str, llm: str, zero: str, scope: str, by: str = "unit") -> dict:
        key_l, key_z = (profile_id, llm), (profile_id, zero)
        vecs = [self.vec(key_l, scope, "branch_error"), self.vec(key_l, scope, "wrong"),
                self.vec(key_z, scope, "branch_error"), self.vec(key_z, scope, "wrong")]
        b = self._boot(vecs, f"h_s|{scope}|{profile_id}|{llm}|{zero}", by)
        if not b.items:
            return {"n": 0}

        def share(m):
            with np.errstate(divide="ignore", invalid="ignore"):
                return m[..., 0] / m[..., 1] - m[..., 2] / m[..., 3]
        reps = None if b.reps is None else b.reps
        share_reps = None if reps is None else share(reps)
        if share_reps is not None and not np.all(np.isfinite(share_reps)):
            share_reps = None           # a replicate without wrong cells in an arm: no share interval
        return {"n": len(b.items), "clusters": b.n_clusters,
                "rate_diff": interval(b.point[0] - b.point[2], None if reps is None else reps[:, 0] - reps[:, 2]),
                "share_diff": interval(share(b.point) if b.point[1] and b.point[3] else None, share_reps)}


# ---------------------------------------------------------------------- outcomes: binds / escapable / indeterminate

def outcome(gaps: dict[str, dict], conds: dict[str, Cond]) -> dict:
    """gaps: condition -> gap estimate (with token_ratio); conds: condition -> Cond. Local representations only."""
    local = {c: g for c, g in gaps.items() if conds[c].local and g.get("est") is not None}
    if not local:
        return {"outcome": "no local representation", "binds_all": None, "escapable_by": []}
    binds = all(g["ci"] is not None and g["est"] >= BINDS_GAP and g["ci"][0] > 0 for g in local.values())
    escape = sorted(c for c, g in local.items() if g["ci"] is not None and g["ci"][1] <= ESCAPE_GAP
                    and g.get("token_ratio") is not None and g["token_ratio"] <= ESCAPE_TOKEN_RATIO)
    verdict = "binds" if binds else "escapable" if escape else "indeterminate"
    return {"outcome": verdict, "binds_all": binds, "escapable_by": escape, "representations": sorted(local)}


def h_s_verdict(main: list[dict], control: list[dict], reading: str) -> str:
    """H_S under one reading (rate_diff, primary | share_diff, secondary): every zero-LLM comparison has CI low > 0 on
    the main pool (L2 + L3), and every control (S2) difference is under 5pp."""
    if not main or not control or any(m.get("n", 0) == 0 for m in main + control):
        return "not evaluable"
    if any(m[reading]["ci"] is None for m in main) or any(c[reading]["est"] is None for c in control):
        return "not evaluable"
    holds = all(m[reading]["ci"][0] > 0 for m in main) and all(abs(c[reading]["est"]) < HS_CONTROL_MAX_DIFF
                                                               for c in control)
    return "holds" if holds else "does not hold"


# ---------------------------------------------------------------------- supply / power check (keys L3, S2)

def eligible(item: dict) -> bool:
    """An item of the supply pool: admitted (admissible) and on the eval split."""
    return item.get("admissible") is True and item.get("split") == "eval"


def supply_domain(item: dict) -> tuple[str | None, bool]:
    """(domain, whether it is the fallback of the L3 batches 1-2, which carry no domain field)."""
    if "domain" in item:
        return item["domain"], False
    return EARLIER_BATCH_DOMAIN.get(item["doc"]), True


def cell_kind_literal(item: dict) -> str | None:
    """The pre-ruling count (the diagnostic column): "not_in_effect" on a not_in_effect gold, "exception" on an
    exception cell, else None."""
    if item["gold"]["value"] == "not_in_effect":
        return "not_in_effect"
    return "exception" if item.get("cell_kind") == "exception" else None


def supply_literal(item: dict) -> tuple[str | None, bool]:
    """(the cell's literal kind in SUPPLY_LITERAL or None, by the manual ruling; whether it is the cell-kind
    fallback)."""
    if "k6_exception_literal" in item:
        kind = item["k6_exception_literal"]
        return (kind if kind in SUPPLY_LITERAL else None), False
    return cell_kind_literal(item), True


def supply_units(cells: list[dict]) -> dict[str, list[dict]]:
    """unit id -> its cells, over the eligible, load_bearing, abstracted, ops / technical L3 cells of `cells` (cells
    without a load_bearing field predate it and count as load_bearing)."""
    units: dict[str, list[dict]] = defaultdict(list)
    for it in cells:
        if (eligible(it) and it["type"].startswith("L3") and it.get("load_bearing", True) is True
                and it.get("rule2_spirit") in SUPPLY_L3_LAYERS and supply_domain(it)[0] in SUPPLY_L3_DOMAINS):
            units[it["unit_id"]].append(it)
    return dict(units)


def _tagged(units: dict[str, list[dict]], tag: str) -> set:
    return {u for u, v in units.items() if any(UNIT_TAGS[tag](c) for c in v)}


def _count(units: dict[str, list[dict]], literal: set) -> dict:
    return {"units": len(units), "literal_units": len(literal & set(units)),
            "literal_share": r4(len(literal & set(units)) / len(units)) if units else None}


def supply_sensitivity(units: dict[str, list[dict]]) -> dict:
    """The reconciliation readings of the L3 supply count (the verdict stays on the main count): the count with the units
    of each ruling tag dropped, or not counted literal, and with one unit per template family; the cue pool is read
    both ways, the first cue pool alone and the union with the second."""
    literal = {u for u, v in units.items() if any(supply_literal(c)[0] for c in v)}
    kept_families: dict[str, str] = {}
    for u in sorted(units):
        fam = units[u][0].get("template_family")
        if fam:
            kept_families.setdefault(fam, u)
    family_dropped = {u for u, v in units.items()
                      if v[0].get("template_family") and kept_families[v[0]["template_family"]] != u}

    def without(drop: set) -> dict:
        return _count({u: v for u, v in units.items() if u not in drop}, literal)
    return {"drop same_rule_medium": without(_tagged(units, "same_rule_medium")),
            "k6_medium not literal": _count(units, literal - _tagged(units, "k6_medium")),
            "drop pin_medium": without(_tagged(units, "pin_medium")),
            "drop normative_medium": without(_tagged(units, "normative_medium")),
            "exemption_outcome not literal": _count(units, literal - _tagged(units, "exemption_outcome")),
            "drop outside_pool": without(_tagged(units, "outside_pool")),
            "drop outside_union_pool": without(_tagged(units, "outside_union_pool")),
            "one unit per template_family": without(family_dropped),
            "drop owner_restates_inconsistent": without(_tagged(units, "owner_restates_inconsistent"))}


def _l3_supply(cells: list[dict]) -> dict:
    units = supply_units(cells)
    per_doc = Counter(v[0]["doc"] for v in units.values())
    literal = [u for u, v in units.items() if any(supply_literal(c)[0] for c in v)]
    abstracted_lb = {it["unit_id"] for it in cells if eligible(it) and it["type"].startswith("L3")
                     and it.get("load_bearing", True) is True and it.get("rule2_spirit") in SUPPLY_L3_LAYERS}
    got = {"units": len(units), "cells": sum(len(v) for v in units.values()),
           "families": len({v[0]["family"] for v in units.values()}),
           "units_by_family": dict(sorted(Counter(v[0]["family"] for v in units.values()).items())),
           "max_units_per_doc": max(per_doc.values(), default=0),
           "literal_units": len(literal), "literal_share": r4(len(literal) / len(units)) if units else None,
           "literal_units_by_kind": {k: sum(any(supply_literal(c)[0] == k for c in v) for v in units.values())
                                     for k in SUPPLY_LITERAL},
           "cell_kind_share": r4(sum(any(cell_kind_literal(c) for c in v) for v in units.values()) / len(units))
           if units else None,
           "domain_fallback_units": sum(any(supply_domain(c)[1] for c in v) for v in units.values()),
           "literal_fallback_units": sum(any(supply_literal(c)[1] for c in v) for v in units.values()),
           "abstracted_units_other_domains": len(abstracted_lb - set(units))}
    checks = {k: (got[k] is not None and (got[k] <= need if k == "max_units_per_doc" else got[k] >= need))
              for k, need in SUPPLY_L3.items()}
    return {**got, "checks": checks, "met": all(checks.values()), "sensitivity": supply_sensitivity(units)}


def supply(items: list[dict]) -> dict:
    """The supply / power check (keys L3, S2) over the eligible items of `items`: the full admitted pool, or
    an analysed subset."""
    pool = [it for it in items if eligible(it)]
    s2_pairs = {it["pair_id"] for it in pool if it["type"].startswith("S2")}
    return {"items": len(pool), "ineligible": len(items) - len(pool),
            "L3": _l3_supply([it for it in pool if it["type"].startswith("L3")]),
            "S2": {"pairs": len(s2_pairs), "met": len(s2_pairs) >= SUPPLY_S2_PAIRS}}
