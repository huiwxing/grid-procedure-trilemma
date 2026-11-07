"""Post hoc readings the paper reports beside the prespecified comparisons, from the final scored rows (offline).

  py -3.12 paper/exploratory.py [--items data/items/questions.jsonl] [--run-dir data/runs/main]
      [--rows rows_final.jsonl] [--cloud-dir data/runs/cloud_comparator] [--out-dir <dir>]

Inputs (the defaults are the paper's): the item file, the main run's directory and the name of its final rows file,
and the cloud comparator's run directory (Claude Sonnet 5, one read per question; its rows are the file its last live
invocation names). "Unit" below is a pair of questions on one branch point of a procedure; the intervals resample
units.
1. Error mix on the primary population (158 questions; every read; the constrained panel's four readers pooled):
   absolute rates of branch errors (E-cond: the wrong branch, E-exc: the general rule answered on an exception item,
   E-ver: the other date's value), of "no branch applies" answers (the scorer's E-other/value_not_in_doc whose answer
   is "none of these" or not_in_effect), of other wrong answers and of abstentions (E-abs), per memory at 4k.
   Unit-cluster bootstrap intervals.
2. Both questions of a unit correct in the same read (paired by repeat index), averaged over reads, per reader and
   memory; panel means average the panel's readers. Unit-cluster bootstrap intervals.
3. gpt-oss-20b with BM25 at 4k against the open-weight reference (REF-B: DeepSeek-V4.1-Flash with BM25 at 4k),
   DeepSeek's full text and the cloud comparator (full text and BM25 at 4k): per-question difference of mean
   accuracy, unit-cluster bootstrap intervals.
4. Accuracy by branch type (cell_kind rule / exception / parallel), with the wrong-answer mix and a stratification
   by the gold option's position; 4b. within the units that hold both a rule and an exception question, exception
   minus rule.
5. Provider tokens per query on the primary population (the same questions as the accuracies), with the build tokens
   amortized over 1,000 queries per memory.
The output keys keep the analysis spec's labels: "tier B" is the server panel, "tier C_load_bearing" the constrained
panel and "tier-A arm" the cloud comparator.

These are post hoc, not among the comparisons of the analysis spec; the paper marks them exploratory. Writes
exploratory.json into --out-dir (default out/paper/) and prints sections 1-3; the file the paper was written from is
paper/exploratory.json. Bootstrap: 2,000 resamples of units with replacement, percentile 95%, seed 20260928.
Questions are taken in sorted order (units in the order of their first question), so every rerun draws the same
resamples. Point estimates reproduce the recorded file exactly; its interval endpoints are Monte Carlo
draws made in the unit order of the process that wrote it (set by Python's per-process string hashing), so a rerun's
endpoints differ from them in the second or third decimal.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from procedures import stored
from procedures.analysis.spec import in_population
from procedures.items.publish import display

B1 = "B1_dsv41flash_direct_nothink"
B_TIER = ["B1_dsv41flash_hf_deepinfra", "B2_qwen3_235b_2507_hf_deepinfra", "tierB_llama33_70b_hf_ovhcloud"]  # servers
C_TIER = ["C1_llama31_8b_hf_deepinfra", "C2_qwen3_8b_hf_nscale_nothink", "C3_gptoss20b_hf_deepinfra_low",
          "tierC2_qwen3_32b_hf_nscale_nothink"]  # the constrained panel
GPTOSS20B = "C3_gptoss20b_hf_deepinfra_low"
A_ARM = "A2_sonnet5_nothink"  # the cloud comparator
CONDITIONS = ["read/B0@full", "read/M2@4000", "read/M4@4000", f"read/M5@4000/{B1}", f"read/M7@4000/{B1}", "read/M6@4000"]
BRANCH = ("E-cond", "E-exc", "E-ver")
N_BOOT, SEED = 2000, 20260928


def read_jsonl(path: Path) -> list[dict]:
    return stored.read_jsonl(path)


def last_live_rows(run_dir: Path) -> str:
    live = [x for x in read_jsonl(run_dir / "invocations.jsonl") if x.get("outcome") == "live" and x.get("outputs")]
    return live[-1]["outputs"]["rows"]


def stage_of(row: dict) -> str:
    """Server (B) readers read DeepSeek's (B1) compression (M6), the others their own: fold both into read/M6@4000."""
    return "read/M6@4000" if row["stage"].startswith("read/M6@4000/") else row["stage"]


def boot_ci(unit_values: dict, stat, rng: random.Random) -> list[float]:
    """Percentile CI of stat(list of unit values) under resampling of units with replacement."""
    units = list(unit_values)
    draws = []
    for _ in range(N_BOOT):
        sample = [unit_values[units[rng.randrange(len(units))]] for _ in units]
        draws.append(stat(sample))
    draws.sort()
    return [round(draws[int(0.025 * N_BOOT)], 4), round(draws[int(0.975 * N_BOOT) - 1], 4)]


def mean(xs) -> float:
    return statistics.fmean(xs) if xs else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--items", type=Path, default=ROOT / "data" / "items" / "questions.jsonl",
                    help="the runs' item file (default: data/items/questions.jsonl)")
    ap.add_argument("--run-dir", type=Path, default=ROOT / "data" / "runs" / "main",
                    help="the main run's directory (default: data/runs/main)")
    ap.add_argument("--rows", default="rows_final.jsonl",
                    help="final rows file name inside --run-dir (default: rows_final.jsonl)")
    ap.add_argument("--cloud-dir", type=Path, default=ROOT / "data" / "runs" / "cloud_comparator",
                    help="the cloud comparator's run directory (default: data/runs/cloud_comparator)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "out" / "paper",
                    help="where exploratory.json goes (default out/paper)")
    a = ap.parse_args(argv)
    items = {it["item_id"]: it for it in read_jsonl(a.items)}
    pop = dict.fromkeys(sorted(iid for iid, it in items.items() if in_population(it)))  # fixed order for the bootstrap
    unit_of = {iid: items[iid]["unit_id"] for iid in pop}
    rows = [r for r in read_jsonl(a.run_dir / a.rows) if r["item_id"] in pop and r["outcome"] == "observed"]
    cloud_rows = a.cloud_dir / last_live_rows(a.cloud_dir)
    rows += [r for r in read_jsonl(cloud_rows) if r["item_id"] in pop and r["outcome"] == "observed"]
    rng = random.Random(SEED)
    out = {"rows": [display(a.run_dir / a.rows, ROOT), display(cloud_rows, ROOT)], "population_cells": len(pop),
           "units": len(set(unit_of.values())),
           "bootstrap": {"replicates": N_BOOT, "seed": SEED, "cluster": "unit", "ci": "percentile 95%"}}

    # per (reader, condition, item): list of (repeat_idx, correct, capability) over reads
    cells: dict = defaultdict(list)
    for r in rows:
        cap = r.get("capability") or {}
        cells[(r["profile_id"], stage_of(r), r["item_id"])].append((r["repeat_idx"], bool(cap.get("correct")), cap))

    # 1. error mix, the constrained panel pooled, primary population, every read
    mix = {}
    for cond in CONDITIONS:
        per_unit = defaultdict(lambda: [0, 0, 0, 0, 0, 0])  # n, wrong, branch, none_or_nie, abstain, other_wrong
        for reader in C_TIER:
            for iid in pop:
                for _, correct, cap in cells.get((reader, cond, iid), []):
                    u = per_unit[unit_of[iid]]
                    u[0] += 1
                    if correct:
                        continue
                    u[1] += 1
                    ec = cap.get("error_class")
                    pred = str(cap.get("pred")).strip().lower()
                    if ec in BRANCH:
                        u[2] += 1
                    elif ec == "E-other" and cap.get("other_subclass") == "value_not_in_doc" and \
                            (pred in ("c", "none of these") or "not_in_effect" in pred or "not in effect" in pred):
                        u[3] += 1
                    elif ec == "E-abs":
                        u[4] += 1
                    else:
                        u[5] += 1
        totals = [sum(v[k] for v in per_unit.values()) for k in range(6)]
        n = totals[0]

        def rate(k):
            return lambda sample: sum(v[k] for v in sample) / max(1, sum(v[0] for v in sample))

        mix[cond] = {"answered": n, "wrong": totals[1] / n,
                     "branch": {"est": round(totals[2] / n, 4), "ci": boot_ci(per_unit, rate(2), rng)},
                     "none_or_not_in_effect": {"est": round(totals[3] / n, 4), "ci": boot_ci(per_unit, rate(3), rng)},
                     "abstain": {"est": round(totals[4] / n, 4), "ci": boot_ci(per_unit, rate(4), rng)},
                     "other_wrong": {"est": round(totals[5] / n, 4), "ci": boot_ci(per_unit, rate(5), rng)}}
    out["error_mix_C_tier_primary"] = mix

    # 2. both questions of a unit correct in the same read
    def unit_pair_values(reader: str, cond: str) -> dict:
        by_unit = defaultdict(dict)
        for iid in pop:
            for rep, correct, _ in cells.get((reader, cond, iid), []):
                by_unit[unit_of[iid]].setdefault(rep, {})[iid] = correct
        values = {}
        for unit, reps in by_unit.items():
            both = [all(v.values()) for v in reps.values() if len(v) == 2]
            if both:
                values[unit] = mean(both)
        return values

    pairs = {}
    for reader in B_TIER + C_TIER + ["tierC3_gptoss120b_hf_deepinfra_low", A_ARM]:
        for cond in CONDITIONS + ["read/M0@full"]:
            vals = unit_pair_values(reader, cond)
            if not vals:
                continue
            pairs[f"{reader}|{cond}"] = {"units": len(vals), "est": round(mean(vals.values()), 4),
                                         "ci": boot_ci(vals, lambda s: mean(s), rng)}
    for tier, readers in (("B", B_TIER), ("C_load_bearing", C_TIER)):
        for cond in CONDITIONS:
            per_reader = [unit_pair_values(r, cond) for r in readers]
            units = sorted(set.intersection(*[set(v) for v in per_reader]))
            if not units:
                continue
            tier_vals = {u: mean([v[u] for v in per_reader]) for u in units}
            pairs[f"tier {tier}|{cond}"] = {"units": len(units), "est": round(mean(tier_vals.values()), 4),
                                            "ci": boot_ci(tier_vals, lambda s: mean(s), rng),
                                            "note": "mean over the tier's readers per unit; units resampled, readers not"}
    out["both_branches_correct"] = pairs

    # 3. gpt-oss-20b + BM25@4k against the references, paired by question
    def cell_means(reader: str, cond: str) -> dict:
        return {iid: mean([c for _, c, _ in v]) for (p, s, iid), v in cells.items() if p == reader and s == cond}

    left = cell_means(GPTOSS20B, "read/M2@4000")
    diffs = {}
    for label, reader, cond in (("REF-B (DeepSeek, BM25@4k)", "B1_dsv41flash_hf_deepinfra", "read/M2@4000"),
                                ("DeepSeek, full text", "B1_dsv41flash_hf_deepinfra", "read/M0@full"),
                                ("tier-A arm, full text (k = 1)", A_ARM, "read/M0@full"),
                                ("tier-A arm, BM25@4k (k = 1)", A_ARM, "read/M2@4000")):
        right = cell_means(reader, cond)
        common = sorted(set(left) & set(right))
        per_unit = defaultdict(list)
        for iid in common:
            per_unit[unit_of[iid]].append(left[iid] - right[iid])
        unit_vals = {u: mean(v) for u, v in per_unit.items()}
        cell_diffs = [left[i] - right[i] for i in common]
        diffs[label] = {"cells": len(common), "units": len(unit_vals), "gptoss20b": round(mean(left[i] for i in common), 4),
                        "reference": round(mean(right[i] for i in common), 4), "diff": round(mean(cell_diffs), 4),
                        "ci": boot_ci(unit_vals, lambda s: mean(s), rng),
                        "cells_where_20b_higher": sum(1 for d in cell_diffs if d > 0),
                        "cells_where_lower": sum(1 for d in cell_diffs if d < 0)}
    out["gptoss20b_bm25_4k_vs_references"] = diffs

    # 4. accuracy by branch type (rule / exception / parallel), wrong-answer mix, and a gold-letter stratification
    def gold_letter(it: dict) -> str:
        opts = [o.lower() for o in (it.get("options") or [])]
        g = str(it["gold"]["value"]).lower()
        return "A" if opts and g == opts[0] else "B" if len(opts) > 1 and g == opts[1] else "value"

    def by_kind(readers: list, cond: str) -> dict:
        per_unit_kind: dict = defaultdict(lambda: defaultdict(list))  # kind -> unit -> [cell means]
        mix: dict = defaultdict(lambda: [0, 0, 0, 0, 0])  # kind -> n, other_branch, none_or_nie, abstain, other
        strat: dict = defaultdict(list)  # (kind, letter) -> cell means
        for reader in readers:
            for iid in pop:
                reads = cells.get((reader, cond, iid), [])
                if not reads:
                    continue
                it = items[iid]
                kind = it["cell_kind"]
                m = mean([c for _, c, _ in reads])
                per_unit_kind[kind][unit_of[iid]].append(m)
                strat[(kind, gold_letter(it))].append(m)
                opts = [o.lower() for o in (it.get("options") or [])]
                gold = str(it["gold"]["value"]).lower()
                other = [o for o in opts if o != gold and o != "none of these"]
                for _, correct, cap in reads:
                    mix[kind][0] += 1
                    if correct:
                        continue
                    pred = str(cap.get("pred")).strip().lower()
                    if pred in ("c", "none of these") or "not_in_effect" in pred or "not in effect" in pred:
                        mix[kind][2] += 1
                    elif cap.get("error_class") == "E-abs":
                        mix[kind][3] += 1
                    elif other and (pred == other[0] or pred in ("a", "b")):
                        mix[kind][1] += 1
                    else:
                        mix[kind][4] += 1
        out_k = {}
        for kind, units in per_unit_kind.items():
            unit_vals = {u: mean(v) for u, v in units.items()}
            n = mix[kind][0]
            out_k[kind] = {"cells": sum(len(v) for v in units.values()), "units": len(unit_vals),
                           "acc": round(mean(unit_vals.values()), 4), "ci": boot_ci(unit_vals, lambda s: mean(s), rng),
                           "wrong_mix_per_answer": {"other_branch": round(mix[kind][1] / n, 4),
                                                    "none_or_not_in_effect": round(mix[kind][2] / n, 4),
                                                    "abstain": round(mix[kind][3] / n, 4), "other": round(mix[kind][4] / n, 4)},
                           "by_gold_letter": {f"{k[1]}": {"acc": round(mean(v), 4), "cells": len(v)}
                                              for k, v in sorted(strat.items()) if k[0] == kind}}
        return out_k

    kinds = {}
    for label, readers in (("tier C_load_bearing", C_TIER), ("tier B", B_TIER), ("C3_gptoss20b_hf_deepinfra_low", [GPTOSS20B]),
                           ("B1_dsv41flash_hf_deepinfra", ["B1_dsv41flash_hf_deepinfra"]), (A_ARM, [A_ARM])):
        for cond in CONDITIONS + ["read/M0@full", "read/M2@16000"]:
            k = by_kind(readers, cond)
            if k:
                kinds[f"{label}|{cond}"] = k
    out["accuracy_by_branch_type"] = {"note": "unit means over cells of that kind; units resampled; 'acc' averages the "
                                              "tier's readers cell by cell; wrong-answer mix is per answered cell", **kinds}

    # 4b. strictly paired: within the units that have both a rule and an exception question, exception minus rule
    def paired_kind(readers: list, cond: str) -> dict | None:
        per_unit = defaultdict(list)  # unit -> [exception - rule per reader]
        rule_vals, exc_vals = [], []
        for reader in readers:
            by_unit = defaultdict(dict)
            for iid in pop:
                reads = cells.get((reader, cond, iid), [])
                kind = items[iid]["cell_kind"]
                if reads and kind in ("rule", "exception"):
                    by_unit[unit_of[iid]][kind] = mean([c for _, c, _ in reads])
            for u, d in by_unit.items():
                if "rule" in d and "exception" in d:
                    per_unit[u].append(d["exception"] - d["rule"])
                    rule_vals.append(d["rule"])
                    exc_vals.append(d["exception"])
        if not per_unit:
            return None
        unit_vals = {u: mean(v) for u, v in per_unit.items()}
        return {"units": len(unit_vals), "rule": round(mean(rule_vals), 4), "exception": round(mean(exc_vals), 4),
                "exception_minus_rule": round(mean(unit_vals.values()), 4), "ci": boot_ci(unit_vals, lambda s: mean(s), rng)}

    paired = {}
    for label, readers in (("tier C_load_bearing", C_TIER), ("tier B", B_TIER), ("C3_gptoss20b_hf_deepinfra_low", [GPTOSS20B]),
                           ("B1_dsv41flash_hf_deepinfra", ["B1_dsv41flash_hf_deepinfra"]), (A_ARM, [A_ARM])):
        for cond in CONDITIONS + ["read/M0@full", "read/M2@16000"]:
            p = paired_kind(readers, cond)
            if p:
                paired[f"{label}|{cond}"] = p
    out["paired_exception_minus_rule"] = {"note": "units with both a rule and an exception cell (44 in the population); "
                                                  "per-unit difference averaged over the tier's readers; units resampled", **paired}

    # 5. provider tokens per query on the primary population (same questions as the accuracies), from the token
    #    counts the rows carry: read tokens (input, cached input, output) + the compressor's per-query tokens +
    #    build tokens / 1,000
    tok = {}
    per: dict = defaultdict(list)
    for r in rows:
        s = r.get("sustainability") or {}
        if "tokens_input_uncached" not in s:
            continue
        total = (s.get("tokens_input_uncached") or 0) + (s.get("tokens_input_cached") or 0) + (s.get("tokens_output") or 0)
        per[(r["profile_id"], stage_of(r))].append((total, s.get("query_memory_tokens") or 0, s.get("build_tokens") or 0))
    for (pid, cond), v in per.items():
        tok[f"{pid}|{cond}"] = {"cells": len(v), "read": round(mean(x[0] for x in v)), "compression": round(mean(x[1] for x in v)),
                                "build_per_memory": round(mean(x[2] for x in v)),
                                "per_query_at_1000": round(mean(x[0] + x[1] + x[2] / 1000 for x in v))}
    out["tokens_primary"] = {"note": "provider counts per answered read on the primary population; build amortized over "
                                     "1,000 queries per memory; compression counted per query", **tok}

    a.out_dir.mkdir(parents=True, exist_ok=True)
    target = a.out_dir / "exploratory.json"
    target.write_bytes((json.dumps(out, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
    print(target)
    print("\n== error mix (constrained panel, primary, all reads)")
    for cond, m in mix.items():
        print(f"  {cond[:36]:36s} n {m['answered']} wrong {m['wrong']:.3f} branch {m['branch']['est']:.3f} {m['branch']['ci']} "
              f"none/NIE {m['none_or_not_in_effect']['est']:.3f} {m['none_or_not_in_effect']['ci']} abstain {m['abstain']['est']:.3f} other {m['other_wrong']['est']:.3f}")
    print("\n== both questions of a unit correct")
    for k, v in pairs.items():
        print(f"  {k[:70]:70s} units {v['units']} {v['est']:.3f} {v['ci']}")
    print("\n== gpt-oss-20b + BM25@4k vs references (paired)")
    for k, v in diffs.items():
        print(f"  vs {k:34s} cells {v['cells']} 20b {v['gptoss20b']:.3f} ref {v['reference']:.3f} diff {v['diff']:+.3f} {v['ci']} (+{v['cells_where_20b_higher']} / -{v['cells_where_lower']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
