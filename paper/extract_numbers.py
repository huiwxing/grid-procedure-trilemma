"""Select the numbers the paper takes from the analysis report into one file.

  py -3.12 paper/extract_numbers.py [<report>] [--out-dir <dir>]

Input: the report scripts/report.py wrote (default: data/analysis/reports/paper_report.json, the paper's). Writes
numbers.json into --out-dir (default out/paper/; the file the paper was written from is paper/numbers.json): the 17
primary comparisons, the Table I cells (primary population, 4k budget plus no document and full text), the
constrained panel's (tiers C1 and C2) gap to the open-weight reference (REF-B) and to the cloud comparator per path
and budget (Table II, Sec. III), the report's token counts per query (Table I's token row comes from the rows,
exploratory.py section 5), the error-class mix, the reader- and panel-level verdicts, the absences and the arbitration
count. Nothing is computed here that the
report did not compute; the script only selects and rounds. Table I prints the rows' accuracies rounded once
(printed_values.py); this file rounds the report's four-decimal estimates, which differs by 0.001 in two cells.

Spec vocabulary: REF-B is the open-weight reference, P0-A the cloud comparator on the full text, C_load_bearing the
constrained panel, and the primary population the 158 questions. The outcomes binds / escapable / indeterminate are
the paper's binding / navigable / undecided. Profile ids keep their run names, so their prefixes are not tiers
(C3_gptoss20b_hf_deepinfra_low is a 24 GB reader).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "paper"
REPORT = ROOT / "data" / "analysis" / "reports" / "paper_report.json"

READERS = [  # (profile_id, tier: A cloud, B server, C1 24 GB, C2 48 GB, C3 80 GB; display name)
    ("A2_sonnet5_nothink", "A", "Claude Sonnet 5"),
    ("B1_dsv41flash_hf_deepinfra", "B", "DeepSeek-V4.1-Flash"),
    ("B2_qwen3_235b_2507_hf_deepinfra", "B", "Qwen3-235B-A22B"),
    ("tierB_llama33_70b_hf_ovhcloud", "B", "Llama-3.3-70B"),
    ("tierC2_qwen3_32b_hf_nscale_nothink", "C2", "Qwen3-32B"),
    ("C3_gptoss20b_hf_deepinfra_low", "C1", "gpt-oss-20b"),
    ("C2_qwen3_8b_hf_nscale_nothink", "C1", "Qwen3-8B"),
    ("C1_llama31_8b_hf_deepinfra", "C1", "Llama-3.1-8B"),
    ("tierC3_gptoss120b_hf_deepinfra_low", "C3", "gpt-oss-120b"),
]
B_TIER = {"B1_dsv41flash_hf_deepinfra", "B2_qwen3_235b_2507_hf_deepinfra", "tierB_llama33_70b_hf_ovhcloud"}
C_LOAD_BEARING = ["C1_llama31_8b_hf_deepinfra", "C2_qwen3_8b_hf_nscale_nothink", "C3_gptoss20b_hf_deepinfra_low",
                  "tierC2_qwen3_32b_hf_nscale_nothink"]
B1_BUILT = "B1_dsv41flash_direct_nothink"
TABLE_COLUMNS = ["B0@full", "M0@full", "M2@4000", "M4@4000", f"M5@4000/{B1_BUILT}", f"M7@4000/{B1_BUILT}",
                 "M7@4000/C3_gptoss20b_hf_deepinfra_low", "M7@4000/tierC2_qwen3_32b_hf_nscale_nothink", "M6@4000"]


def m6_condition(profile_id: str) -> str:
    """Server (B) readers read the compression DeepSeek (B1) wrote; the other readers compress for themselves."""
    return f"M6@4000/{B1_BUILT}" if profile_id in B_TIER else f"M6@4000/{profile_id}"


def r3(x):
    return None if x is None else round(float(x), 3)


def est_ci(block: dict) -> dict:
    return {"est": r3(block["est"]), "ci": [r3(block["ci"][0]), r3(block["ci"][1])], "n": block.get("n"),
            "clusters": block.get("clusters")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("report", type=Path, nargs="?", default=REPORT,
                    help="the analysis report (default: data/analysis/reports/paper_report.json)")
    ap.add_argument("--out-dir", type=Path, default=OUT, help="where numbers.json goes (default out/paper)")
    args = ap.parse_args(argv)
    report = args.report
    doc = json.loads(report.read_text(encoding="utf-8"))
    out: dict = {"report": report.name, "code_sha256": doc["meta"]["code_sha256"],
                 "spec": doc["meta"]["spec"], "runs": doc["meta"]["runs"], "status": doc["status"]["reasons"],
                 "thresholds": doc["meta"]["thresholds"], "reference": doc["meta"]["reference"]}

    # 1. primary comparisons (spec order)
    prim = {}
    for p in doc["primary"]:
        e = p["estimate"]
        row = {"computable": p["computable"], "kind": p["kind"]}
        if "est" in e:
            row.update(est_ci(e))
        if "chosen" in e:
            row["chosen"] = e["chosen"]
        if "outcome" in e:
            row.update({"outcome": e["outcome"], "escapable_by": e["escapable_by"], "binds_all": e["binds_all"],
                        "reference": e["reference"]})
        if "legs" in e:
            row["legs"] = {k: est_ci(v) for k, v in e["legs"].items()}
            row["verdict"] = e.get("verdict")
        if "h_t" in e:
            row["h_t"] = e["h_t"]
        if "control" in e:
            row["control"] = {k: est_ci(v) for k, v in e["control"].items()}
        prim[p["id"]] = row
    out["primary"] = prim
    out["sensitivity_callout"] = doc["sensitivity_callout"]

    # 2. Table I cells: primary-population accuracy at the 4k budget (+ no document, full text), chance-corrected
    #    accuracy alongside
    cap = {(c["profile_id"], c["condition"]): c for c in doc["capability"]}
    table = {}
    for pid, tier, name in READERS:
        cells = {}
        for col in TABLE_COLUMNS:
            cond = m6_condition(pid) if col == "M6@4000" else col
            c = cap.get((pid, cond))
            if c is None:
                cells[col] = None
                continue
            s = c["scopes"]["primary"]
            cells[col] = {"condition": cond, "acc": r3(s["est"]), "ci": [r3(s["ci"][0]), r3(s["ci"][1])],
                          "n": s["n"], "chance_corrected": r3(s["lettered"]["chance_corrected"]["est"]),
                          "k1": bool(c.get("k1_marked"))}
        table[pid] = {"tier": tier, "name": name, "cells": cells}
    out["table1"] = table

    # 3. the constrained panel's gap per path and budget, to the open-weight reference and to the cloud comparator
    #    (Table II)
    def gaps(pid: str) -> dict:
        e = [p for p in doc["primary"] if p["id"] == pid][0]["estimate"]
        return {"reference": e["reference"], "outcome": e["outcome"],
                "gaps": {k: {"est": r3(v["est"]), "ci": [r3(v["ci"][0]), r3(v["ci"][1])], "n": v["n"],
                             "token_ratio": r3(v.get("token_ratio"))} for k, v in e["gaps"].items()}}
    out["gap_C_load_bearing"] = {"vs_REF_B": gaps("OUTCOME-C_load_bearing"), "vs_A": gaps("OUTCOME-C_load_bearing-A")}

    # 4. the report's token counts per query (provider tokens per answered query; build tokens per memory; amortised
    #    totals; not printed)
    sus = {(s["profile_id"], s["condition"]): s for s in doc["sustainability"]}
    tokens = {}
    for pid, tier, name in READERS:
        rows = {}
        for col in TABLE_COLUMNS + ["M2@16000", "M2@2000"]:
            cond = m6_condition(pid) if col == "M6@4000" else col
            s = sus.get((pid, cond))
            if s is None:
                continue
            q = s["per_query"]
            rows[col] = {"condition": cond, "input_total": round(s["per_query_input_total"]),
                         "input_uncached": round(q["input_uncached"]), "input_cached": round(q["input_cached"]),
                         "output": round(q["output"]), "read_total": round(s["per_query_read_total"]),
                         "build_per_memory": None if s.get("build") is None else round(s["build"]["tokens_per_memory"]["mean"]),
                         "build_memories": None if s.get("build") is None else s["build"]["memories"],
                         "compression_per_query": None if s.get("query") is None else round(s["query"]["per_query_total"]),
                         "amortised_total_at_N": {k: round(v) for k, v in s["amortised_per_query"].items()}}
        tokens[pid] = rows
    out["tokens"] = tokens

    # 5. error classes of the constrained panel at the 4k budget (+ no document), from the report's L3 taxonomy: the
    #    scorer's error classes and the report's roll-up to the classes of Trukhina and Vashkelis (not printed)
    tax = {}
    for t in doc["taxonomy"]["L3"]:
        if t["profile_id"] not in C_LOAD_BEARING or not ("@4000" in t["condition"] or t["condition"] == "B0@full"):
            continue
        cond = t["condition"]
        if cond.startswith("M6@"):  # self-compressed: one row for the panel; records (M7) keep their builder
            for r in C_LOAD_BEARING:
                cond = cond.replace(f"/{r}", "/{reader}")
        a = tax.setdefault(cond, {"n": 0, "correct": 0, "frozen": {}, "codec": {}})  # frozen: the scorer's classes
        a["n"] += t["n"]
        a["correct"] += t["correct"]
        for k, v in t["frozen"].items():
            a["frozen"][k] = a["frozen"].get(k, 0) + v
        for k, v in t["codec"].items():
            a["codec"][k] = a["codec"].get(k, 0) + (v or 0)
    for a in tax.values():
        wrong = a["n"] - a["correct"]
        a["wrong"] = wrong
        a["branch_share_of_wrong"] = r3((a["frozen"]["E-cond"] + a["frozen"]["E-exc"] + a["frozen"]["E-ver"]) / wrong)
        a["not_in_doc_share_of_wrong"] = r3(a["frozen"]["E-other/value_not_in_doc"] / wrong)
        a["abstain_share_of_wrong"] = r3(a["frozen"]["E-abs"] / wrong)
    out["taxonomy_C_load_bearing_L3"] = tax

    # 5b. reader- and panel-level verdicts (binds / escapable / indeterminate = binding / navigable / undecided),
    #     against the open-weight reference (report 'outcomes') and the cloud comparator (Sec. III-C)
    readers_ref_b = doc["outcomes"]["primary"]["readers"]
    readers_a = doc["reference_arm_outcomes"][READERS[0][0]]["readers"]
    out["reader_outcomes"] = {pid: {"tier": tier, "name": name,
                                    "vs_REF_B": {"outcome": readers_ref_b[pid]["outcome"],
                                                 "escapable_by": readers_ref_b[pid]["escapable_by"]} if pid in readers_ref_b else None,
                                    "vs_A": {"outcome": readers_a[pid]["outcome"],
                                             "escapable_by": readers_a[pid]["escapable_by"]} if pid in readers_a else None}
                              for pid, tier, name in READERS}
    out["tier_outcomes"] = {"vs_REF_B": {t: {"outcome": v["outcome"], "escapable_by": v["escapable_by"]}
                                         for t, v in doc["outcomes"]["primary"]["tiers"].items()},
                            "vs_A": {t: {"outcome": v["outcome"], "escapable_by": v["escapable_by"]}
                                     for t, v in doc["reference_arm_outcomes"][READERS[0][0]]["tiers"].items()}}

    # 6. absences, arbitration, supply
    out["absences"] = doc["absences"]
    out["arbitration_pending"] = doc["status"]["arbitration_pending"]
    out["supply"] = doc["status"]["supply"]
    out["memory_sensitivities"] = doc["memory_sensitivities"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    target = args.out_dir / "numbers.json"
    target.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(target, "primary", len(prim), "table readers", len(table), "gap conditions",
          len(out["gap_C_load_bearing"]["vs_REF_B"]["gaps"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
