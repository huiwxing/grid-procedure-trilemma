"""Recomputes the paper's printed values from the released rows and ledgers (the values that are not fields of the
analysis report) and compares each with the string the paper prints, offline.

  py -3.12 paper/printed_values.py [--numbers <file>] [--exploratory <file>] [--out-dir <dir>]

By section: II-A test set and arbitration; II-B budget share of a full procedure; II-C single-reader precision;
Table I (accuracies rounded once from the rows, token row, completed compression questions); Table II strings;
III DeepSeek's input tokens and the reader-input criterion; III-A memories vs retrieval, build share, error mix, the
PJM Manual 36 example; III-B compression token split, time-outs counted as failures; III-C gpt-oss-20b vs the
references, the PJM Manual 13 pair, rule vs exception; Table I note r (readers that reason first). The printed strings
of the prespecified comparisons and of the post hoc readings are read from the two recorded files (--numbers,
--exploratory; default: the files in paper/). The clause counts of Sec. III-C need the rebuilt corpus texts and come
from clause_coverage.py.

Every value is recomputed, then compared: a printed number or string with its recomputed form, a printed statement
with whether the records bear it out, and the consistency of the records the values rest on ("records"). The script
prints the table of value / printed / match, writes printed_values.json into --out-dir (default out/paper/; the file
the paper was written from is paper/printed_values.json, whose main_table_direct_3dp and completion_adjusted_readers
fig_readers.py reads) and exits 1 when any value does not match.

Reads the item file, the final rows of the four runs (the main run, the cloud comparator's, and the two that read
records built by gpt-oss-20b and by Qwen3-32B), the nine readers' read ledgers, the DeepSeek compressor's query ledger
and the arbitration decisions.

Labels: a unit is a pair of questions (unit_id); tier B is the server panel, C_load_bearing the constrained panel,
REF-B the open-weight reference (DeepSeek with retrieval at 4k) and the A reader the cloud comparator. Paths: B0 no
document, M0 full text, M2 retrieval, M4 rule sentences, M5_B / M7_B the summary / records built by DeepSeek, M7_20b /
M7_32b the records built by gpt-oss-20b / Qwen3-32B and M6 compression, all at the 4k budget except B0 and M0.
Intervals: 2,000 resamples of the 79 units with replacement from one shared draw sequence (seed 20260928), so every
contrast is paired; percentile 95%.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
from statistics import fmean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from procedures import stored  # noqa: E402
from procedures.analysis.spec import in_population  # noqa: E402
from procedures.memory import compressor  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = ROOT / "out" / "paper"
ITEMS = ROOT / "data/items/questions.jsonl"
ROWS = ROOT / "data/runs/main/rows_final.jsonl"                                   # the main run
AREF_ROWS = ROOT / "data/runs/cloud_comparator/rows_final.jsonl"                  # the cloud comparator
CBUILT20_ROWS = ROOT / "data/runs/records_built_by_gptoss20b/rows_final.jsonl"    # records built by gpt-oss-20b
CBUILT32_ROWS = ROOT / "data/runs/records_built_by_qwen32b/rows_final.jsonl"      # records built by Qwen3-32B
RUN_ROWS = (ROWS, AREF_ROWS, CBUILT20_ROWS, CBUILT32_ROWS)
ARBITRATION = ROOT / "data/analysis/arbitration"
EXPLORATORY = HERE / "exploratory.json"
NUMBERS = HERE / "numbers.json"
N_BOOT, SEED = 2000, 20260928
REPEATS = (0, 1, 2)
B_BUILDER = "B1_dsv41flash_direct_nothink"   # DeepSeek as builder and compressor
GPT20 = "C3_gptoss20b_hf_deepinfra_low"
QWEN32 = "tierC2_qwen3_32b_hf_nscale_nothink"
REF_B = "B1_dsv41flash_hf_deepinfra"
A_READER = "A2_sonnet5_nothink"
B_READERS = [REF_B, "B2_qwen3_235b_2507_hf_deepinfra", "tierB_llama33_70b_hf_ovhcloud"]
C_READERS = [QWEN32, GPT20, "C2_qwen3_8b_hf_nscale_nothink", "C1_llama31_8b_hf_deepinfra"]
C3_READER = "tierC3_gptoss120b_hf_deepinfra_low"
OPEN_WEIGHT = [*B_READERS, C3_READER, *C_READERS]
READERS = [A_READER, *OPEN_WEIGHT]
NAMES = dict(zip(READERS, ["Claude Sonnet 5", "DeepSeek-V4.1-Flash", "Qwen3-235B-A22B", "Llama-3.3-70B",
                         "gpt-oss-120b", "Qwen3-32B", "gpt-oss-20b", "Qwen3-8B", "Llama-3.1-8B"]))
CONDITIONS = ["B0", "M0", "M2", "M4", "M5_B", "M7_B", "M7_20b", "M7_32b", "M6"]
READ_LEDGERS = {r: (AREF_ROWS.parent if r == A_READER else ROWS.parent) / f"read__{r}.jsonl" for r in READERS}
OPENING_UNIT = "o_x_m13_dual_fuel_max_emergency"   # Fig. 1b, PJM Manual 13 Sec. 6.4
M36_UNIT = "o_x4_m36_dynamic_reserve_ei"           # PJM Manual 36 Sec. 5.1.2 exemption
PLACED, NOT_PLACED = "they are to be placed in Maximum Emergency", "should not be placed in Maximum Emergency"
M36_RULE = "Dynamic Reserve must be calculated"
NUMBER_WORDS = "zero one two three four five six seven eight nine ten".split()

# Table I as printed: the columns (header, path), question accuracy per reader ("--": not run), the token row.
TABLE_I_COLUMNS = (("Retrieval", "M2"), ("Rule sentences", "M4"), ("Summary", "M5_B"), ("Records", "M7_B"),
                   ("Compression", "M6"), ("Full text", "M0"), ("No document", "B0"))
TABLE_I = {
    "Claude Sonnet 5": ("0.899", "--", "--", "--", "--", "0.911", "0.411"),
    "DeepSeek-V4.1-Flash": ("0.772", "0.504", "0.686", "0.544", "0.793", "0.751", "0.441"),
    "Qwen3-235B-A22B": ("0.726", "0.475", "0.614", "0.584", "0.781", "0.719", "0.405"),
    "Llama-3.3-70B": ("0.620", "0.458", "0.565", "0.532", "0.715", "0.616", "0.464"),
    "gpt-oss-20b": ("0.878", "0.479", "0.654", "0.612", "0.734", "--", "0.390"),
    "Qwen3-8B": ("0.610", "0.449", "0.530", "0.494", "0.595", "--", "0.399"),
    "Llama-3.1-8B": ("0.451", "0.361", "0.441", "0.430", "0.455", "--", "0.409"),
    "Qwen3-32B": ("0.578", "0.373", "0.534", "0.470", "0.686", "--", "0.344"),
    "gpt-oss-120b": ("0.846", "0.506", "0.679", "0.641", "0.907", "--", "0.456"),
}
TABLE_I_TOKENS = ("3.9--4.0k", "2.3--2.4k", "3.8--3.9k", "3.7--3.8k", "9.9--15.0k", "20.2--20.7k", "0.4k")
TOKEN_STAGES = {"M2": "read/M2@4000", "M4": "read/M4@4000", "M5_B": f"read/M5@4000/{B_BUILDER}",
                "M7_B": f"read/M7@4000/{B_BUILDER}", "M6": "read/M6@4000", "M0": "read/M0@full", "B0": "read/B0@full"}
# Table II as printed (minus signs as "-"), by the spec's comparison ids: dGap-M5 / dGap-M7 / dGap-M6 summary /
# records / compression vs retrieval (-B the server panel, -C_load_bearing the constrained panel), CBUILT the larger
# records builder, H_M7 records vs rule sentences, H_S wrong-branch answers with summaries; in the note, interaction is
# the panel difference in loss from memories, REF-B and P0-A the two references, and a gap is the open-weight
# reference minus the constrained panel (compression on the 117 questions all four readers completed).
TABLE_II = {
    "dGap-M5-B": "-0.084 [-0.143, -0.032]", "dGap-M5-C_load_bearing": "-0.090 [-0.194, -0.008]",
    "dGap-M7-B": "-0.153 [-0.241, -0.068]", "dGap-M7-C_load_bearing": "-0.128 [-0.237, -0.029]",
    "interaction-M5": "-0.005 [-0.115, 0.096]", "interaction-M7": "0.025 [-0.094, 0.139]",
    "CBUILT-gptoss20b": "0.012 [-0.032, 0.060]", "CBUILT-qwen32b": "0.004 [-0.054, 0.060]",
    "dGap-M6-B": "0.057 [0.010, 0.112]", "dGap-M6-C_load_bearing": "-0.009 [-0.105, 0.092]",
    "H_M7 rule-sentence leg": "0.086 [0.019, 0.158]", "H_S retrieval leg": "0.011 [-0.044, 0.067]",
    "H_S rule-sentence leg": "0.068 [0.011, 0.122]", "REF-B": "0.772 [0.711, 0.829]", "P0-A": "0.911 [0.867, 0.949]",
    "gap retrieval": "0.143 [-0.021, 0.290]", "gap compression": "0.128 [-0.008, 0.265]",
    "gap retrieval 2k": "0.161 [-0.003, 0.297]", "gap retrieval 16k": "0.151 [0.021, 0.273]",
}
TABLE_II_MEMORY_GAPS = "0.17--0.28, lower bounds >=0.03"   # gap to the open-weight reference, DeepSeek-built memories
TABLE_II_CLOUD_GAPS = ">=0.28, lower bounds >=0.12"         # gap to the cloud comparator, every path and budget


class Checks:
    """The comparisons in the order they are made: what (where in the paper), the recomputed value, the printed one,
    whether they match. A printed statement that is not one number is "holds" when the records bear it out."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def value(self, where: str, what: str, computed, printed) -> None:
        self.rows.append({"where": where, "what": what, "value": computed, "printed": printed,
                          "match": computed == printed})

    def holds(self, where: str, statement: str, condition) -> None:
        self.rows.append({"where": where, "what": statement, "value": "holds" if condition else "does not hold",
                          "printed": "holds", "match": bool(condition)})

    def mismatches(self) -> list[dict]:
        return [r for r in self.rows if not r["match"]]

    def table(self) -> str:
        def shown(x) -> str:
            s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)
            return s if len(s) <= 60 else s[:57] + "..."
        lines = ["| where | what | value | printed | match |", "|---|---|---|---|---|"]
        lines += [f"| {r['where']} | {r['what']} | {shown(r['value'])} | {shown(r['printed'])} | "
                  f"{'yes' if r['match'] else 'NO'} |" for r in self.rows]
        return "\n".join(lines)


def condition(row: dict) -> str | None:
    """The row's path key (CONDITIONS), None for a path Table I does not report. Server readers read the compression
    DeepSeek wrote; the other readers compress for themselves."""
    stage = row["stage"]
    fixed = {"read/B0@full": "B0", "read/M0@full": "M0", "read/M2@4000": "M2",
             "read/M4@4000": "M4", f"read/M5@4000/{B_BUILDER}": "M5_B",
             f"read/M7@4000/{B_BUILDER}": "M7_B", f"read/M7@4000/{GPT20}": "M7_20b",
             f"read/M7@4000/{QWEN32}": "M7_32b"}
    if stage in fixed:
        return fixed[stage]
    compressor_profile = B_BUILDER if row["profile_id"] in B_READERS else row["profile_id"]
    if stage == f"read/M6@4000/{compressor_profile}":
        return "M6"
    return None


def source(path: Path) -> dict:
    """An input's path (relative to the repository when it lies inside it) and the sha256 of its content."""
    path = Path(path).resolve()
    shown = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name
    return {"path": shown, "sha256": stored.sha256(path)}


def reasoning_chars(record: dict) -> int | None:
    """Length of a read call's reasoning phase, None when it has none. Qwen3 routes with thinking off sometimes return
    the answer JSON in the reasoning field; such text is an answer, not a reasoning phase. A ledger that replaces the
    reasoning text by a placeholder records the text's length as reasoning_chars."""
    text = (record.get("reasoning_text") or "").strip()
    if not text or text.startswith("{"):
        return None
    return record["reasoning_chars"] if "reasoning_chars" in record else len(text)


def interval_3dp(v: dict) -> str:
    """est [lo, hi] at three decimals, as Table II prints an estimate of the numbers file."""
    return f"{v['est']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]"


def clear(v: dict) -> bool:
    """Table II's "clear": at least 5 points, with the interval excluding zero."""
    return abs(v["est"]) >= 0.05 and (v["ci"][0] > 0 or v["ci"][1] < 0)


def spans_zero(v: dict) -> bool:
    return v["ci"][0] < 0 < v["ci"][1]


def floor2(x: float) -> float:
    """x rounded down to two decimals: the bound a printed ">=" may claim."""
    return math.floor(round(100 * x, 6)) / 100


def words(n: int) -> str:
    return NUMBER_WORDS[n] if 0 <= n < len(NUMBER_WORDS) else str(n)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--numbers", type=Path, default=NUMBERS,
                    help="the file extract_numbers.py wrote (default: the recorded one in paper/)")
    ap.add_argument("--exploratory", type=Path, default=EXPLORATORY,
                    help="the file exploratory.py wrote (default: the recorded one in paper/)")
    ap.add_argument("--out-dir", type=Path, default=OUT, help="where printed_values.json goes (default out/paper)")
    args = ap.parse_args(argv)
    numbers = json.loads(args.numbers.read_text(encoding="utf-8"))
    exploratory = json.loads(args.exploratory.read_text(encoding="utf-8"))
    prim = numbers["primary"]
    gaps = numbers["gap_C_load_bearing"]
    ref_gaps = gaps["vs_REF_B"]["gaps"]
    mix = exploratory["error_mix_C_tier_primary"]
    primary_tokens = exploratory["tokens_primary"]
    check = Checks()

    # The test set (the primary population) and every scored read of it in the four runs.
    items = {it["item_id"]: it for it in stored.read_jsonl(ITEMS) if in_population(it)}
    units = sorted({it["unit_id"] for it in items.values()})
    unit_items = {u: sorted(i for i, it in items.items() if it["unit_id"] == u) for u in units}
    document_keys = sorted({(it["family"], it["doc"]) for it in items.values()})
    gold_letters = {}   # multiple-choice questions: the letter of the gold option
    unmatched_golds = []
    for iid, it in items.items():
        opts = it.get("options") or []
        gold = str(it["gold"]["value"]).strip().casefold()
        matches = [j for j, opt in enumerate(opts) if str(opt).strip().casefold() == gold]
        if opts:
            if len(matches) == 1:
                gold_letters[iid] = chr(65 + matches[0])
            else:
                unmatched_golds.append(iid)
    check.holds("records", "every multiple-choice gold is exactly one of its options", not unmatched_golds)
    letter_counts = dict(sorted(Counter(gold_letters.values()).items()))
    normalized_gold = [str(it["gold"]["value"]).strip().casefold() for it in items.values()]
    rows = [row for path in RUN_ROWS for row in stored.read_jsonl(path) if row["item_id"] in items]
    groups = defaultdict(list)   # (reader, path key) -> rows
    for row in rows:
        cond = condition(row)
        if row["profile_id"] in READERS and cond is not None:
            groups[row["profile_id"], cond].append(row)

    # Accuracy per reader and path: mean correctness over a question's observed reads, then the mean over the observed
    # questions, rounded once.
    observed_cells = {}   # (reader, path) -> {question: mean correctness of its observed reads}
    table = []
    duplicated = []
    for reader in READERS:
        row_output = {"reader": reader, "name": NAMES[reader], "methods": {}}
        for cond in CONDITIONS:
            cond_rows = groups.get((reader, cond), [])
            if not cond_rows:
                row_output["methods"][cond] = None
                continue
            keys = [(r["item_id"], r["repeat_idx"]) for r in cond_rows]
            if len(keys) != len(set(keys)):
                duplicated.append(f"{reader} {cond}")
            cells = defaultdict(list)
            observed = [r for r in cond_rows if r["outcome"] == "observed"]
            for r in observed:
                cells[r["item_id"]].append(int(bool((r.get("capability") or {}).get("correct"))))
            if not cells:
                row_output["methods"][cond] = None
                continue
            means = {iid: fmean(v) for iid, v in cells.items()}
            observed_cells[reader, cond] = means
            acc = fmean(means.values())
            lettered = [val for iid, val in means.items() if iid in gold_letters]
            row_output["methods"][cond] = {
                "accuracy": acc, "accuracy_3dp": f"{acc:.3f}",
                "observed_questions": len(cells), "observed_reads": len(observed),
                "correct_reads": sum(sum(v) for v in cells.values()),
                "reads_per_observed_question": dict(sorted(Counter(len(v) for v in cells.values()).items())),
                "outcomes": dict(sorted(Counter(r["outcome"] for r in cond_rows).items())),
                "lettered_questions": len(lettered), "lettered_accuracy": fmean(lettered) if lettered else None}
        table.append(row_output)
    check.holds("records", "no question-repeat is scored twice for one reader and path", not duplicated)
    acc3 = {row["reader"]: {c: (v["accuracy_3dp"] if v else None) for c, v in row["methods"].items()} for row in table}
    unrounded = {row["reader"]: row["methods"] for row in table}

    # One common set of resampled unit indices keeps every contrast paired.
    rng = random.Random(SEED)
    draws = [[rng.randrange(len(units)) for _ in units] for _ in range(N_BOOT)]

    def interval(values: list[float]) -> list[float]:
        estimates = sorted(fmean(values[j] for j in draw) for draw in draws)
        return [estimates[int(0.025 * N_BOOT)], estimates[int(0.975 * N_BOOT) - 1]]

    def paired_summary(left: list[float], right: list[float]) -> dict:
        diffs = [a - b for a, b in zip(left, right)]
        return {"left": fmean(left), "left_ci95": interval(left),
                "right": fmean(right), "right_ci95": interval(right),
                "difference": fmean(diffs), "difference_ci95": interval(diffs)}

    def units_from_cells(values: dict[str, float]) -> list[float]:
        return [fmean(values[i] for i in unit_items[u]) for u in units]

    # Completion: the 474 planned reads (158 questions x 3) of each open-weight reader with retrieval and with
    # compression, a read that did not complete counting as a failure; panel values average the named readers.
    completion = []
    complete_values = {}
    expected_keys = {(iid, rep) for iid in items for rep in REPEATS}
    for reader in OPEN_WEIGHT:
        arms = {}
        for cond in ("M2", "M6"):
            cond_rows = groups[reader, cond]
            keyed = {(r["item_id"], r["repeat_idx"]): r for r in cond_rows}
            check.holds("records", f"{NAMES[reader]} {cond}: one row per planned read (158 questions x 3)",
                        set(keyed) == expected_keys)
            observed = [r for r in cond_rows if r["outcome"] == "observed"]
            cells = observed_cells.get((reader, cond), {})

            def success(iid: str, rep: int) -> int:
                r = keyed.get((iid, rep))
                return int(r is not None and r["outcome"] == "observed"
                           and bool((r.get("capability") or {}).get("correct")))
            per_item = {iid: fmean(success(iid, rep) for rep in REPEATS) for iid in items}
            uv = units_from_cells(per_item)
            complete_values[reader, cond] = uv
            correct_reads = sum(int(bool((r.get("capability") or {}).get("correct"))) for r in observed)
            check.holds("records", f"{NAMES[reader]} {cond}: task success with incomplete reads as failures = "
                        "correct reads / 474", abs(fmean(uv) - correct_reads / 474) < 1e-12)
            check.holds("records", f"{NAMES[reader]} {cond}: a question has all three reads observed or none",
                        set(Counter(r["item_id"] for r in observed).values()) == {3})
            arms[cond] = {"observed_accuracy": fmean(cells.values()) if cells else None,
                          "observed_questions": len(cells), "observed_reads": len(observed),
                          "correct_reads": correct_reads, "planned_reads": 474,
                          "question_coverage": len(cells) / 158, "read_coverage": len(observed) / 474,
                          "success_with_incomplete_zero": fmean(uv),
                          "success_ci95": interval(uv),
                          "incomplete_questions": sorted(set(items) - set(cells)),
                          "nonobserved_reasons_by_reader_row": dict(sorted(Counter(
                              (r.get("access") or {}).get("infeasible_reason") or r["outcome"]
                              for r in cond_rows if r["outcome"] != "observed").items()))}
        completion.append({"reader": reader, "name": NAMES[reader], "arms": arms,
                           "paired_M6_minus_M2": paired_summary(complete_values[reader, "M6"],
                                                                 complete_values[reader, "M2"])})
    tier_completion = {}
    for tier, readers in (("B", B_READERS), ("C_load_bearing", C_READERS)):
        uv = {cond: [fmean(complete_values[r, cond][j] for r in readers) for j in range(len(units))]
              for cond in ("M2", "M6")}
        common = set(items)
        for r in readers:
            for cond in ("M2", "M6"):
                common &= set(observed_cells.get((r, cond), {}))
        common_points = {cond: fmean(fmean(observed_cells[r, cond][i] for i in common) for r in readers)
                         if common else None for cond in ("M2", "M6")}
        tier_completion[tier] = {"readers": readers, "readers_resampled": False,
                                 "paired_M6_minus_M2": paired_summary(uv["M6"], uv["M2"]),
                                 "observed_common_questions": len(common),
                                 "observed_common_accuracy": common_points,
                                 "observed_common_difference": common_points["M6"] - common_points["M2"]
                                 if common else None}

    # Provider tokens per answered read (input = uncached + cached), from the token counts the rows carry.
    def provider_means(reader: str, cond: str) -> dict:
        cond_rows = [r for r in groups[reader, cond] if r["outcome"] == "observed"]
        check.holds("records", f"{NAMES[reader]} {cond}: 474 answered reads of the 158 questions, each with its token "
                    "counts", len(cond_rows) == 474 and len({r["item_id"] for r in cond_rows}) == 158
                    and all("tokens_input_uncached" in r["sustainability"] for r in cond_rows))
        inputs = [(r["sustainability"].get("tokens_input_uncached") or 0)
                  + (r["sustainability"].get("tokens_input_cached") or 0) for r in cond_rows]
        outputs = [r["sustainability"].get("tokens_output") or 0 for r in cond_rows]
        builds = [r["sustainability"].get("build_tokens") or 0 for r in cond_rows]
        compression = [r["sustainability"].get("query_memory_tokens") or 0 for r in cond_rows]
        return {"questions": 158, "reads": len(cond_rows), "provider_input_mean": fmean(inputs),
                "provider_output_mean": fmean(outputs),
                "provider_input_plus_output_mean": fmean(a + b for a, b in zip(inputs, outputs)),
                "build_tokens_mean_per_answered_read": fmean(builds),
                "compression_tokens_mean_per_answered_read": fmean(compression),
                "per_query_at_1000": fmean(a + b + c + d / 1000
                                           for a, b, c, d in zip(inputs, outputs, compression, builds))}

    # Sec. II-A: "The test has 79 such pairs, 158 questions, drawn from 50 public documents of six North American system
    # operators and NERC: 21 pairs from AESO, 19 from NERC, 14 each from CAISO and PJM, 6 from IESO, 3 from NYISO and 2
    # from MISO. Forty-four pairs set a rule against its exception; the other 35 contrast parallel or complementary
    # conditions." "142 questions are multiple-choice and 16 are numeric. Every question has an applicable branch":
    # no gold answer is none of these or not in effect.
    units_by_family = {fam: len({it["unit_id"] for it in items.values() if it["family"] == fam})
                       for fam in sorted({it["family"] for it in items.values()})}
    pair_kinds = Counter(tuple(sorted(items[i]["cell_kind"] for i in unit_items[u])) for u in units)
    numeric = [i for i in items if i not in gold_letters]
    printed_families = {"AESO": 21, "NERC": 19, "CAISO": 14, "PJM": 14, "IESO": 6, "NYISO": 3, "MISO": 2}
    check.value("II-A", "pairs", len(units), 79)
    check.value("II-A", "questions", len(items), 158)
    check.holds("II-A", "two questions per pair", all(len(ids) == 2 for ids in unit_items.values()))
    check.value("II-A", "documents", len(document_keys), 50)
    for fam, n in printed_families.items():
        check.value("II-A", f"pairs from {fam}", units_by_family.get(fam, 0), n)
    check.holds("II-A", "no pair from another source", set(units_by_family) <= set(printed_families))
    rule_exception = pair_kinds[("exception", "rule")]
    check.value("II-A", "pairs that set a rule against its exception", rule_exception, 44)
    check.value("II-A", "pairs that contrast parallel or complementary conditions", len(units) - rule_exception, 35)
    check.value("II-A", "multiple-choice questions", len(gold_letters), 142)
    check.value("II-A", "numeric questions", len(numeric), 16)
    check.holds("II-A", "the questions without options are numeric",
                all(items[i]["answer_type"] == "numeric" for i in numeric))
    check.value("II-A", "gold option letters", letter_counts, {"A": 71, "B": 70, "C": 1})
    check.holds("II-A", "every question has an applicable branch (no gold is none of these or not in effect)",
                not any(g == "none of these" or g.replace(" ", "_") == "not_in_effect" for g in normalized_gold))
    # "the 60 of 47,739 answers (31 distinct cases) whose units it could not resolve went to two blind Claude arbiters,
    # which judged only answers from other model families and agreed on every case": the 158 questions the paper
    # reports, not the full 244-question run.
    arb_rows, arb_ids, observed_rows, arb_claude = 0, set(), 0, 0
    for row in rows:
        if row["outcome"] != "observed":
            continue
        observed_rows += 1
        arb = (row.get("capability") or {}).get("arb_id")
        if arb:
            arb_rows += 1
            arb_ids.add(arb)
            arb_claude += row["profile_id"] == A_READER
    disagreements = set()
    for path in sorted(ARBITRATION.glob("decisions_*.json")):
        disagreements |= set(json.loads(path.read_text(encoding="utf-8"))["disagreements"])
    check.value("II-A", "arbitrated answers", arb_rows, 60)
    check.value("II-A", "answers", observed_rows, 47739)
    check.value("II-A", "distinct arbitrated cases", len(arb_ids), 31)
    check.holds("II-A", "the Claude arbiters judged only answers from other model families", arb_claude == 0)
    check.holds("II-A", "the two arbiters agreed on every case", not (disagreements & arb_ids))
    arbitration = {"arbitrated_answers": arb_rows, "observed_answers": observed_rows, "unique_cases": len(arb_ids),
                   "disagreements_in_population": len(disagreements & arb_ids),
                   "disagreements_full_run": sorted(disagreements)}

    # Sec. II-B: "every path supplies at most 4,000 tokens (4k), about one fifth of an average full procedure in this
    # set": the budget over DeepSeek's mean provider input with the full text. "A compressor receives the retrieved
    # passages, up to four times the budget (16k tokens)". "Compression accuracy covers completed questions only,
    # excluding API time-outs."
    resource_arms = {label: provider_means(reader, cond) for label, reader, cond in (
        ("B1_full_text", REF_B, "M0"), ("B1_retrieval_4k", REF_B, "M2"), ("gptoss20b_retrieval_4k", GPT20, "M2"))}
    budget_share = 4000 / resource_arms["B1_full_text"]["provider_input_mean"]
    check.holds("II-B", "4k is about one fifth of an average full procedure (0.18 to 0.22 of DeepSeek's full-text "
                "input)", 0.18 <= budget_share <= 0.22)
    check.value("II-B", "compressor input, tokens", compressor.INPUT_FACTOR * 4000, 16000)
    check.holds("II-B", "the compression questions not completed are API time-outs",
                all(set(c["arms"]["M6"]["nonobserved_reasons_by_reader_row"]) <= {"route_timeout"} for c in completion))

    # Sec. II-C: "with 79 pairs, a reader's retrieval accuracy at 4k is estimated to within 4--6 points" = half-widths
    # of the nine readers' M2@4000 intervals, from the report and from this script's draws; the statement is limited to
    # retrieval at 4k because the memories' intervals at 4k are wider.
    def half_width(ci: list[float]) -> float:
        return (ci[1] - ci[0]) / 2
    report_hw = {NAMES[r]: half_width(numbers["table1"][r]["cells"]["M2@4000"]["ci"]) for r in READERS}
    own_hw = {NAMES[r]: half_width(interval(units_from_cells(observed_cells[r, "M2"]))) for r in READERS}
    for label, hw in (("report", report_hw), ("this script's draws", own_hw)):
        check.value("II-C", f"half-widths of the nine readers' retrieval intervals at 4k, points ({label})",
                    f"{round(100 * min(hw.values()))}--{round(100 * max(hw.values()))}", "4--6")
    widest_memory_hw = max(half_width(c["ci"]) for r in READERS for k, c in numbers["table1"][r]["cells"].items()
                           if c and c.get("ci") and k.startswith(("M5@", "M7@")))
    check.holds("II-C", "the memories' intervals at 4k are wider (widest half-width over 0.065)",
                widest_memory_hw > 0.065)

    # Table I: the 63 cells as printed (accuracies rounded once from the rows; the numbers file rounds the report's
    # four-decimal estimates, which differs by 0.001 in two cells). Token row: provider tokens per question of the
    # server and constrained panels with compression included and builds amortized over 1,000 queries per memory
    # (exploratory.py section 5). Note: "completed questions only (Qwen3-32B 121, Llama-3.1-8B 153, gpt-oss-120b 157)".
    printed_table = {NAMES[r]: tuple(acc3[r][key] or "--" for _, key in TABLE_I_COLUMNS) for r in READERS}
    for name, cells in TABLE_I.items():
        for (header, _), got, want in zip(TABLE_I_COLUMNS, printed_table[name], cells):
            check.value("Table I", f"{name}, {header}", got, want)
    token_row = {}
    for header, key in TABLE_I_COLUMNS:
        vals = [primary_tokens[f"{r}|{TOKEN_STAGES[key]}"]["per_query_at_1000"]
                for r in (B_READERS if key == "M0" else B_READERS + C_READERS)]
        lo, hi = f"{min(vals) / 1000:.1f}", f"{max(vals) / 1000:.1f}"
        token_row[header] = f"{lo}k" if lo == hi else f"{lo}--{hi}k"
    for (header, _), want in zip(TABLE_I_COLUMNS, TABLE_I_TOKENS):
        check.value("Table I", f"tokens per question, {header}", token_row[header], want)
    completed = {c["name"]: c["arms"]["M6"]["observed_questions"] for c in completion
                 if c["arms"]["M6"]["observed_questions"] < len(items)}
    printed_completed = {"gpt-oss-120b": 157, "Qwen3-32B": 121, "Llama-3.1-8B": 153}
    for name, n in printed_completed.items():
        check.value("Table I", f"compression questions completed, {name}", completed.get(name, len(items)), n)
    check.holds("Table I", "every other reader completed every compression question",
                set(completed) <= set(printed_completed))

    # Table II: every interval as printed, from the numbers file.
    table_ii_strings = {k: interval_3dp(prim[k]) for k in ("dGap-M5-B", "dGap-M5-C_load_bearing", "dGap-M7-B",
                                                  "dGap-M7-C_load_bearing", "interaction-M5", "interaction-M7",
                                                  "CBUILT-gptoss20b", "CBUILT-qwen32b", "dGap-M6-B",
                                                  "dGap-M6-C_load_bearing")}
    table_ii_strings["H_M7 rule-sentence leg"] = interval_3dp(prim["H_M7"]["legs"]["M4@4000"])
    table_ii_strings["H_S retrieval leg"] = interval_3dp(prim["H_S"]["legs"]["M2@4000"])
    table_ii_strings["H_S rule-sentence leg"] = interval_3dp(prim["H_S"]["legs"]["M4@4000"])
    table_ii_strings["REF-B"] = interval_3dp(prim["REF-B"])
    table_ii_strings["P0-A"] = interval_3dp(prim["P0-A"])
    table_ii_strings["gap retrieval"] = interval_3dp(ref_gaps["M2@4000"])
    table_ii_strings["gap compression"] = interval_3dp(ref_gaps["M6@4000/{reader}"])
    table_ii_strings["gap retrieval 2k"] = interval_3dp(ref_gaps["M2@2000"])
    table_ii_strings["gap retrieval 16k"] = interval_3dp(ref_gaps["M2@16000"])
    for key, want in TABLE_II.items():
        check.value("Table II", key, table_ii_strings[key], want)
    # The answers. "Clear loss in both panels" (summaries, records); "No gain detected from a larger builder"; records
    # beat rule sentences, "Yes, but not retrieval"; "No clear increase detected" in wrong-branch answers; compression
    # "Clear gain on servers only".
    check.holds("Table II", "summaries and records: clear loss in both panels",
                all(clear(prim[k]) and prim[k]["est"] < 0 for k in ("dGap-M5-B", "dGap-M5-C_load_bearing", "dGap-M7-B",
                                                                    "dGap-M7-C_load_bearing")))
    check.holds("Table II", "no gain detected from a larger builder",
                spans_zero(prim["CBUILT-gptoss20b"]) and spans_zero(prim["CBUILT-qwen32b"]))
    check.holds("Table II", "records beat rule sentences, but not retrieval",
                prim["H_M7"]["legs"]["M4@4000"]["ci"][0] > 0 and prim["H_M7"]["legs"]["M2@4000"]["est"] < 0)
    check.holds("Table II", "no clear increase detected in wrong-branch answers with summaries",
                spans_zero(prim["H_S"]["legs"]["M2@4000"]))
    check.value("Table II", "verdicts of H_M7 and H_S", [prim["H_M7"]["verdict"], prim["H_S"]["verdict"]],
                ["fail", "does not hold"])
    check.holds("Table II", "compression: clear gain on servers only",
                clear(prim["dGap-M6-B"]) and prim["dGap-M6-B"]["est"] > 0 and not clear(prim["dGap-M6-C_load_bearing"]))
    # Access rows. Retrieval and compression: "Undecided" (the class verdict against the open-weight reference); every
    # memory path meets the binding criterion (a gap of at least 10 points with its interval above zero), the
    # DeepSeek-built ones at "0.17--0.28, lower bounds >=0.03"; against the cloud comparator "Binding", at ">=0.28,
    # lower bounds >=0.12" on every path and budget.
    check.value("Table II", "constrained panel against the open-weight reference / the cloud comparator",
                [gaps["vs_REF_B"]["outcome"], gaps["vs_A"]["outcome"]], ["indeterminate", "binds"])
    check.holds("Table II", "the compression gap to the open-weight reference spans zero and reaches past 5 points",
                spans_zero(ref_gaps["M6@4000/{reader}"]) and ref_gaps["M6@4000/{reader}"]["ci"][1] > 0.05)
    memory_gaps = {k: v for k, v in ref_gaps.items() if k.startswith(("M5@", "M7@"))}
    check.holds("Table II", "every memory path binds against the open-weight reference (12 gaps of at least 10 "
                "points, intervals above zero)",
                len(memory_gaps) == 12 and all(v["est"] >= 0.10 and v["ci"][0] > 0 for v in memory_gaps.values()))
    deepseek_gaps = [v for k, v in memory_gaps.items() if k.endswith(f"/{B_BUILDER}")]
    deepseek_est = [v["est"] for v in deepseek_gaps]
    memory_row = (f"{min(deepseek_est):.2f}--{max(deepseek_est):.2f}, "
                  f"lower bounds >={floor2(min(v['ci'][0] for v in deepseek_gaps)):.2f}")
    cloud_gaps = list(gaps["vs_A"]["gaps"].values())
    cloud_row = (f">={floor2(min(v['est'] for v in cloud_gaps)):.2f}, "
                 f"lower bounds >={floor2(min(v['ci'][0] for v in cloud_gaps)):.2f}")
    check.value("Table II", "DeepSeek-built memory paths", len(deepseek_gaps), 6)
    check.value("Table II", "gaps of the DeepSeek-built memories to the open-weight reference", memory_row,
                TABLE_II_MEMORY_GAPS)
    check.value("Table II", "gaps to the cloud comparator, every path and budget", cloud_row, TABLE_II_CLOUD_GAPS)
    # Note c: "117 questions completed by all four readers"; on them the constrained panel's compression minus
    # retrieval, recomputed from the rows, is the printed -0.009.
    common_difference = tier_completion["C_load_bearing"]["observed_common_difference"]
    check.value("Table II", "compression questions all four constrained readers completed",
                tier_completion["C_load_bearing"]["observed_common_questions"], 117)
    check.value("Table II", "questions of the report's compression gap", ref_gaps["M6@4000/{reader}"]["n"], 117)
    check.value("Table II", "constrained compression minus retrieval on those questions (rows)",
                None if common_difference is None else f"{common_difference:.3f}", "-0.009")
    check.value("Table II", "constrained compression minus retrieval on those questions (report)",
                f"{prim['dGap-M6-C_load_bearing']['est']:.3f}", "-0.009")

    # Sec. III: "DeepSeek reads 3.82k input tokens per question with retrieval instead of 20.13k with the full
    # procedure, yet scores 0.772 against 0.751" (exploratory.py's token counts of these arms agree with the rows').
    # "Every path at 2k or 4k meets the reader-input criterion" (mean reader input at most a quarter of the reference
    # reader's mean full-text input; at 16k every path but rule sentences exceeds it). "Compression adds its own work,
    # bringing the total to 9.9--15.0k tokens per question" (Table I token row).
    deepseek_tokens = {"retrieval": f"{resource_arms['B1_retrieval_4k']['provider_input_mean'] / 1000:.2f}k",
                       "full_text": f"{resource_arms['B1_full_text']['provider_input_mean'] / 1000:.2f}k"}
    check.value("III", "DeepSeek input tokens per question, retrieval", deepseek_tokens["retrieval"], "3.82k")
    check.value("III", "DeepSeek input tokens per question, full text", deepseek_tokens["full_text"], "20.13k")
    check.value("III", "DeepSeek accuracy, retrieval", acc3[REF_B]["M2"], "0.772")
    check.value("III", "DeepSeek accuracy, full text", acc3[REF_B]["M0"], "0.751")
    exploratory_tokens_match = []
    for label, reader, stage in (("B1_full_text", REF_B, "read/M0@full"),
                                 ("B1_retrieval_4k", REF_B, "read/M2@4000"),
                                 ("gptoss20b_retrieval_4k", GPT20, "read/M2@4000")):
        recorded = primary_tokens[f"{reader}|{stage}"]
        arm = resource_arms[label]
        exploratory_tokens_match.append(recorded["cells"] == arm["reads"]
                                        and recorded["read"] == round(arm["provider_input_plus_output_mean"]))
    full_text = resource_arms["B1_full_text"]
    resource_ratios = {label: {
        "input_ratio_to_B1_full_text": vals["provider_input_mean"] / full_text["provider_input_mean"],
        "input_plus_output_ratio_to_B1_full_text":
            vals["provider_input_plus_output_mean"] / full_text["provider_input_plus_output_mean"]}
        for label, vals in resource_arms.items() if label != "B1_full_text"}
    check.holds("III", "every path at 2k or 4k meets the reader-input criterion (at most 1/4 of the full-text input)",
                all(v["token_ratio"] <= 0.25 for k, v in ref_gaps.items() if "@2000" in k or "@4000" in k))
    check.holds("III", "at 16k every path but rule sentences exceeds it",
                all(v["token_ratio"] > 0.25 for k, v in ref_gaps.items() if "@16000" in k and not k.startswith("M4@")))
    check.value("III", "tokens per question with compression", token_row["Compression"], "9.9--15.0k")

    # Sec. III-A: "For every open-weight reader, both memories score below retrieval at nearly the same tokens per
    # question, to which builds amortized over 1,000 queries per memory add less than 2%" (also Fig. 2). Build shares
    # are workload-weighted over the answered reads, not averages over the document revisions.
    below_retrieval = all(float(acc3[r][m]) < float(acc3[r]["M2"]) for r in OPEN_WEIGHT for m in ("M5_B", "M7_B"))
    check.holds("III-A", "for every open-weight reader both memories score below retrieval", below_retrieval)
    build_shares = []
    for reader in B_READERS + C_READERS:
        for cond, stage in (("M5_B", f"read/M5@4000/{B_BUILDER}"), ("M7_B", f"read/M7@4000/{B_BUILDER}")):
            vals = provider_means(reader, cond)
            recorded = primary_tokens[f"{reader}|{stage}"]
            exploratory_tokens_match.append(
                recorded["cells"] == vals["reads"]
                and recorded["read"] == round(vals["provider_input_plus_output_mean"])
                and recorded["build_per_memory"] == round(vals["build_tokens_mean_per_answered_read"])
                and recorded["per_query_at_1000"] == round(vals["per_query_at_1000"]))
            share = (vals["build_tokens_mean_per_answered_read"] / 1000) / vals["per_query_at_1000"]
            build_shares.append({"reader": reader, "name": NAMES[reader], "condition": cond, "reads": vals["reads"],
                                 "build_amortized_tokens_per_query": vals["build_tokens_mean_per_answered_read"] / 1000,
                                 "total_tokens_per_query_at_1000": vals["per_query_at_1000"],
                                 "build_share": share, "exploratory_tokens": recorded,
                                 "build_share_from_exploratory_rounded_fields":
                                     (recorded["build_per_memory"] / 1000) / recorded["per_query_at_1000"]})
    tokens_match = all(exploratory_tokens_match)
    check.holds("records", "exploratory.py's token counts per answered read agree with the rows' (17 configurations)",
                tokens_match)
    largest_build_share = max(build_shares, key=lambda x: x["build_share"])
    below_2_percent = largest_build_share["build_share"] < 0.02
    check.value("III-A", "memory configurations of the build-share check", len(build_shares), 14)
    check.holds("III-A", "builds amortized over 1,000 queries per memory add less than 2%", below_2_percent)
    # "Averaged per panel, summaries lose 8--9 points and records 13--15" (Sec. IV and the abstract: 8--15). "each
    # DeepSeek-built memory leaves the constrained panel's mean 17 to 28 points below the open-weight reference, with
    # gap intervals above zero" (Table II).
    points = [round(-100 * prim[k]["est"]) for k in ("dGap-M5-B", "dGap-M5-C_load_bearing", "dGap-M7-C_load_bearing",
                                                      "dGap-M7-B")]
    points_lost = {"summary": f"{points[0]}--{points[1]}", "records": f"{points[2]}--{points[3]}",
                   "memories": f"{min(points)}--{max(points)}"}
    check.value("III-A", "points lost per panel, summaries", points_lost["summary"], "8--9")
    check.value("III-A", "points lost per panel, records", points_lost["records"], "13--15")
    check.value("III-A", "points lost per panel, memories (abstract)", points_lost["memories"], "8--15")
    gap_points = f"{round(100 * min(deepseek_est))} to {round(100 * max(deepseek_est))}"
    check.value("III-A", "DeepSeek-built memories below the open-weight reference, points", gap_points, "17 to 28")
    # "Constrained readers answer this way in 3.6% of answers with retrieval, 9.4% with summaries and 14.5% with
    # records. Wrong-branch answers stay near their retrieval level (31%, 32% and 27%)" (exploratory.py section 1; Fig.
    # 3, whose compression bar counts the completed reads only).
    mix_keys = ("read/M2@4000", f"read/M5@4000/{B_BUILDER}", f"read/M7@4000/{B_BUILDER}")
    no_branch = [f"{100 * mix[k]['none_or_not_in_effect']['est']:.1f}%" for k in mix_keys]
    wrong_branch = [f"{100 * mix[k]['branch']['est']:.0f}%" for k in mix_keys]
    for label, got, want in zip(("retrieval", "summaries", "records"), no_branch, ("3.6%", "9.4%", "14.5%")):
        check.value("III-A", f"'no branch applies' answers, {label}", got, want)
    for label, got, want in zip(("retrieval", "summaries", "records"), wrong_branch, ("31%", "32%", "27%")):
        check.value("III-A", f"wrong-branch answers, {label}", got, want)
    check.value("records", "compression reads the constrained panel answered (exploratory.py)",
                mix["read/M6@4000"]["answered"], 1770)
    check.value("records", "compression reads the constrained panel answered (rows)",
                sum(c["arms"]["M6"]["observed_reads"] for c in completion if c["reader"] in C_READERS), 1770)
    # exploratory.py reads the letter c as "none of these"; resolving each answer letter to that question's own option
    # gives the same counts, so the printed shares hold either way.
    error_mix_check = {}
    for cond, stage in zip(("M2", "M5_B", "M7_B"), mix_keys):
        answered = letter_c_none = resolved_none = four_option_reads = raw_c_four_options = 0
        changed = []
        for reader in C_READERS:
            for row in groups[reader, cond]:
                if row["outcome"] != "observed":
                    continue
                answered += 1
                cap = row.get("capability") or {}
                pred = str(cap.get("pred")).strip().casefold()
                opts = [str(o).strip().casefold() for o in items[row["item_id"]].get("options") or []]
                if len(opts) == 4:
                    four_option_reads += 1
                    raw_c_four_options += pred == "c"
                if cap.get("correct"):
                    continue
                lettered = len(pred) == 1 and "a" <= pred <= "z" and ord(pred) - ord("a") < len(opts)
                resolved = opts[ord(pred) - ord("a")] if lettered else pred
                eligible = cap.get("error_class") == "E-other" and cap.get("other_subclass") == "value_not_in_doc"
                as_c = eligible and (pred in ("c", "none of these") or "not_in_effect" in pred
                                     or "not in effect" in pred)
                aware = eligible and (resolved == "none of these" or "not_in_effect" in resolved
                                      or "not in effect" in resolved)
                letter_c_none += bool(as_c)
                resolved_none += bool(aware)
                if as_c != aware:
                    changed.append({"reader": reader, "item_id": row["item_id"], "repeat_idx": row["repeat_idx"],
                                    "pred": pred, "resolved_option": resolved, "letter_c_as_none": bool(as_c),
                                    "option_aware": bool(aware)})
        exploratory_rate = mix[stage]["none_or_not_in_effect"]["est"]
        check.value("records", f"'no branch applies' share, {cond}: rows, the letter c read as none of these "
                    "(exploratory.py's reading)", round(letter_c_none / answered, 4), exploratory_rate)
        check.value("III-A", f"'no branch applies' share, {cond}, each answer letter resolved to its option",
                    f"{100 * resolved_none / answered:.1f}%", f"{100 * exploratory_rate:.1f}%")
        error_mix_check[cond] = {"answered_reads": answered, "letter_c_as_none_count": letter_c_none,
                                 "option_aware_none_or_not_in_effect_count": resolved_none,
                                 "letter_c_as_none_rate": letter_c_none / answered,
                                 "option_aware_rate": resolved_none / answered,
                                 "exploratory_rate": exploratory_rate, "changed_classifications": changed,
                                 "four_option_observed_reads": four_option_reads,
                                 "raw_literal_c_predictions_on_four_option_questions": raw_c_four_options}

    # The worked examples (Sec. III-A and III-C) check every read of every named reader, not a majority.
    def answers(reader: str, cond: str, unit: str, kind: str) -> list[tuple[int, str]]:
        iid = next(i for i in unit_items[unit] if items[i]["cell_kind"] == kind)
        found = [r for r in groups[reader, cond] if r["item_id"] == iid and r["outcome"] == "observed"]
        return [(int(bool((r.get("capability") or {}).get("correct"))),
                 str((r.get("capability") or {}).get("matched_option") or (r.get("capability") or {}).get("pred")))
                for r in sorted(found, key=lambda r: r["repeat_idx"])]

    def every(reader: str, cond: str, unit: str, kind: str, correct: int, option: str | None = None) -> bool:
        got = answers(reader, cond, unit, kind)
        return bool(got) and all(c == correct and (option is None or o == option) for c, o in got)

    # "In PJM Manual 36 ... Seven of the eight open-weight readers answer the exemption question correctly in every run
    # with retrieval and choose the rule in every run with the summary."
    m36_retrieval_right = [r for r in OPEN_WEIGHT if every(r, "M2", M36_UNIT, "exception", 1)]
    m36_summary_wrong = [r for r in OPEN_WEIGHT if every(r, "M5_B", M36_UNIT, "exception", 0)]
    m36_summary_rule = [r for r in OPEN_WEIGHT if every(r, "M5_B", M36_UNIT, "exception", 0, M36_RULE)]
    check.value("III-A", "Manual 36: open-weight readers right on the exemption in every retrieval run",
                len(m36_retrieval_right), 7)
    check.value("III-A", "Manual 36: open-weight readers wrong on it in every summary run", len(m36_summary_wrong), 8)
    check.holds("III-A", "Manual 36: each of them chooses the rule", m36_summary_rule == m36_summary_wrong)
    check.value("III-A", "Manual 36: the reader that misses it with retrieval",
                [NAMES[r] for r in OPEN_WEIGHT if r not in m36_retrieval_right], ["Llama-3.1-8B"])
    summary_readers = (f"all {words(len(m36_summary_rule))}" if len(m36_summary_rule) == len(OPEN_WEIGHT)
                       else f"{words(len(m36_summary_rule))} of the {words(len(OPEN_WEIGHT))}")
    m36_statement = (f"{words(len(m36_retrieval_right))} of the {words(len(OPEN_WEIGHT))} open-weight readers right in "
                     f"every retrieval run; {summary_readers} choose the rule in every summary run")

    # Sec. III-B: "per-question compression raises the server panel's accuracy by 0.057 at 12.0--12.1k instead of
    # 3.9--4.0k tokens per question: the compressor reads 9.0k and writes 1.4k, and each reader then reads 1.6k and
    # writes under 0.1k"; "triple the tokens" (heading and Sec. IV). The compressor's input is the retrieval packed to
    # at most 16k by the o200k_base count.
    server_tokens = {cond: [primary_tokens[f"{r}|{stage}"]["per_query_at_1000"] for r in B_READERS]
                     for cond, stage in (("M2", "read/M2@4000"), ("M6", "read/M6@4000"))}
    server_range = {cond: f"{min(v) / 1000:.1f}--{max(v) / 1000:.1f}k" for cond, v in server_tokens.items()}
    check.value("III-B", "server tokens per question, compression", server_range["M6"], "12.0--12.1k")
    check.value("III-B", "server tokens per question, retrieval", server_range["M2"], "3.9--4.0k")
    server_ratios = [c / r for c, r in zip(server_tokens["M6"], server_tokens["M2"])]
    check.holds("III-B", "triple the tokens (each server reader's ratio within 2.9 to 3.15)",
                all(2.9 <= x <= 3.15 for x in server_ratios))
    latest = {}
    for rec in stored.read_jsonl(ROWS.parent / f"query__{B_BUILDER}.jsonl"):
        if rec.get("kind") == "call":
            latest[rec["request_hash"]] = rec
    comp = [r for r in latest.values() if r["stage"] == "query/M6@4000" and r["item_id"] in items]
    check.holds("records", "DeepSeek compressed all 158 questions", len(comp) == 158
                and all(r["outcome"] == "observed" for r in comp))
    c_in = fmean(r["usage"]["input_total"] for r in comp)
    c_out = fmean(r["usage"]["output_total"] for r in comp)
    check.value("III-B", "the compressor reads", f"{c_in / 1000:.1f}k", "9.0k")
    check.value("III-B", "the compressor writes", f"{c_out / 1000:.1f}k", "1.4k")
    check.holds("III-B", "the compressor's input stays within the 16k packing (under 17,000 provider tokens)",
                max(r["usage"]["input_total"] for r in comp) < 17000)
    compression_split = {"compressor": B_BUILDER, "questions": len(comp), "compressor_input_mean": c_in,
                         "compressor_output_mean": c_out,
                         "compressor_input_min_max": [min(r["usage"]["input_total"] for r in comp),
                                                      max(r["usage"]["input_total"] for r in comp)],
                         "output_cap_hits": sum(r.get("finish_reason") == "length" for r in comp), "readers": {}}
    for reader in B_READERS:
        v = provider_means(reader, "M6")
        check.value("III-B", f"{NAMES[reader]} reads after compression", f"{v['provider_input_mean'] / 1000:.1f}k",
                    "1.6k")
        check.holds("III-B", f"{NAMES[reader]} writes under 0.1k after compression", v["provider_output_mean"] < 100)
        check.holds("records", f"{NAMES[reader]}: the compression tokens its rows carry are the compressor's",
                    abs(v["compression_tokens_mean_per_answered_read"] - (c_in + c_out)) < 1)
        compression_split["readers"][NAMES[reader]] = v
    # "Qwen3-32B completed compression for 121 of 158 questions, and counting the time-outs as failures lowers its
    # accuracy to 0.525, below its 0.578 with retrieval"
    qwen32_m6 = next(c for c in completion if c["reader"] == QWEN32)["arms"]["M6"]
    qwen32_failures = {"completed": qwen32_m6["observed_questions"],
                       "accuracy": f"{qwen32_m6['success_with_incomplete_zero']:.3f}", "retrieval": acc3[QWEN32]["M2"]}
    check.value("III-B", "Qwen3-32B compression questions completed", qwen32_failures["completed"], 121)
    check.value("III-B", "Qwen3-32B with time-outs counted as failures", qwen32_failures["accuracy"], "0.525")
    check.value("III-B", "Qwen3-32B with retrieval", qwen32_failures["retrieval"], "0.578")
    check.holds("III-B", "Qwen3-32B with time-outs as failures is below its retrieval accuracy",
                qwen32_m6["success_with_incomplete_zero"] < unrounded[QWEN32]["M2"]["accuracy"])

    # Sec. III-C: "Under the same retrieval path, the four constrained readers range from 0.451 to 0.878" (abstract and
    # Sec. IV: "0.45 to 0.88"). "Against the open-weight reference it is undecided: with retrieval the gap is 0.143
    # [-0.021, 0.290], an interval that includes zero and extends past 5 points": neither the binding rule (every path
    # at least 10 points short, interval above zero) nor the navigable rule (some path's interval within 5 points)
    # applies. Reader-level verdicts from the report: gpt-oss-20b is navigable ("escapable") against the open-weight
    # reference through retrieval and undecided ("indeterminate") against the cloud comparator.
    c_retrieval = sorted(unrounded[r]["M2"]["accuracy"] for r in C_READERS)
    retrieval_range = f"{c_retrieval[0]:.3f} to {c_retrieval[-1]:.3f}"
    check.value("III-C", "constrained readers with retrieval", retrieval_range, "0.451 to 0.878")
    check.value("III-C", "constrained readers with retrieval (abstract, Sec. IV)",
                f"{c_retrieval[0]:.2f} to {c_retrieval[-1]:.2f}", "0.45 to 0.88")
    retrieval_gap = ref_gaps["M2@4000"]
    check.value("III-C", "retrieval gap to the open-weight reference", interval_3dp(retrieval_gap),
                "0.143 [-0.021, 0.290]")
    check.holds("III-C", "an interval that includes zero and extends past 5 points",
                spans_zero(retrieval_gap) and retrieval_gap["ci"][1] > 0.05)
    gpt20_outcomes = numbers["reader_outcomes"][GPT20]
    check.value("III-C", "gpt-oss-20b against the open-weight reference", gpt20_outcomes["vs_REF_B"]["outcome"],
                "escapable")
    check.holds("III-C", "gpt-oss-20b is navigable through retrieval",
                "M2@4000" in gpt20_outcomes["vs_REF_B"]["escapable_by"])
    check.value("III-C", "gpt-oss-20b against the cloud comparator", gpt20_outcomes["vs_A"]["outcome"],
                "indeterminate")
    check.value("III-C", "constrained panel against the open-weight reference",
                numbers["tier_outcomes"]["vs_REF_B"]["C_load_bearing"]["outcome"], "indeterminate")
    check.value("III-C", "constrained panel against the cloud comparator",
                numbers["tier_outcomes"]["vs_A"]["C_load_bearing"]["outcome"], "binds")
    # Table II "readers on both sides": gpt-oss-20b's interval lies above the open-weight reference's and the other
    # three constrained readers' below it; those three score below both gpt-oss-20b and the open-weight reference.
    reader_ci = {name: numbers["table1"][pid]["cells"]["M2@4000"]["ci"] for name, pid in (
        ("gpt-oss-20b", GPT20), ("Qwen3-8B", "C2_qwen3_8b_hf_nscale_nothink"), ("Qwen3-32B", QWEN32),
        ("Llama-3.1-8B", "C1_llama31_8b_hf_deepinfra"))}
    ref_ci = prim["REF-B"]["ci"]
    check.holds("Table II", "gpt-oss-20b's retrieval interval lies above the open-weight reference's",
                reader_ci["gpt-oss-20b"][0] > ref_ci[1])
    check.holds("Table II", "the other three constrained readers' intervals lie below it",
                all(reader_ci[n][1] < ref_ci[0] for n in ("Qwen3-8B", "Qwen3-32B", "Llama-3.1-8B")))
    check.holds("Table II", "those three score below both gpt-oss-20b and the open-weight reference",
                all(unrounded[r]["M2"]["accuracy"] < min(unrounded[GPT20]["M2"]["accuracy"],
                                                         unrounded[REF_B]["M2"]["accuracy"])
                    for r in C_READERS if r != GPT20))
    # gpt-oss-20b "scores 0.878 against the open-weight reference's 0.772 at about the same 3.8k input tokens, a
    # difference of 0.105 [0.040, 0.171], which makes retrieval navigable for this reader; against the cloud comparator
    # on the full text, it differs by -0.034 [-0.084, 0.017], undecided" (exploratory.py section 3).
    vs = exploratory["gptoss20b_bm25_4k_vs_references"]
    contrasts = {label: f"{vs[key]['diff']:.3f} [{vs[key]['ci'][0]:.3f}, {vs[key]['ci'][1]:.3f}]"
                 for label, key in (("open-weight reference", "REF-B (DeepSeek, BM25@4k)"),
                                    ("cloud comparator, full text", "tier-A arm, full text (k = 1)"))}
    for label, want in (("open-weight reference", "0.105 [0.040, 0.171]"),
                        ("cloud comparator, full text", "-0.034 [-0.084, 0.017]")):
        check.value("III-C", f"gpt-oss-20b with retrieval minus the {label}", contrasts[label], want)
    check.value("III-C", "input tokens per question with retrieval, DeepSeek and gpt-oss-20b",
                sorted({f"{resource_arms[k]['provider_input_mean'] / 1000:.1f}k"
                        for k in ("B1_retrieval_4k", "gptoss20b_retrieval_4k")}), ["3.8k"])
    # The first contrast recomputed from the rows with this script's draws: the point estimates must equal
    # exploratory.py's; the interval endpoints are separate Monte Carlo draws.
    gpt_check = paired_summary(units_from_cells(observed_cells[GPT20, "M2"]),
                               units_from_cells(observed_cells[REF_B, "M2"]))
    recorded = vs["REF-B (DeepSeek, BM25@4k)"]
    check.holds("records", "gpt-oss-20b minus the open-weight reference recomputed from the rows: the point estimates "
                "of exploratory.py", abs(gpt_check["difference"] - recorded["diff"]) <= 0.000051
                and abs(gpt_check["left"] - recorded["gptoss20b"]) <= 0.000051
                and abs(gpt_check["right"] - recorded["reference"]) <= 0.000051)
    check.holds("III-C", "that difference's interval excludes zero (this script's draws and exploratory.py's)",
                gpt_check["difference_ci95"][0] > 0 and recorded["ci"][0] > 0)
    gpt_check.update({"left_label": "gpt-oss-20b M2@4000", "right_label": "DeepSeek M2@4000",
                      "exploratory_estimate": recorded,
                      "note": "recomputed in sorted pair order with its own draws; point estimates must equal "
                              "exploratory.py, interval endpoints are separate Monte Carlo draws"})
    # "On the opening PJM pair (Fig. 1b), only the two gpt-oss readers and the cloud comparator change their answer with
    # the fuel inventory, in every run. Llama-3.1-8B, Qwen3-8B and Qwen3-32B keep the turbine out of Maximum Emergency
    # on both questions (the rule), DeepSeek and Qwen3-235B place it there on both (the exception), and Llama-3.3-70B
    # varies between runs."
    switch_with_inventory = [r for r in READERS if every(r, "M2", OPENING_UNIT, "rule", 1)
                             and every(r, "M2", OPENING_UNIT, "exception", 1)]
    keep_rule_on_exception = [r for r in READERS if every(r, "M2", OPENING_UNIT, "rule", 1, NOT_PLACED)
                              and every(r, "M2", OPENING_UNIT, "exception", 0, NOT_PLACED)]
    place_on_both = [r for r in READERS if every(r, "M2", OPENING_UNIT, "rule", 0, PLACED)
                     and every(r, "M2", OPENING_UNIT, "exception", 1, PLACED)]
    opening_varies = [r for r in READERS if len({o for _, o in answers(r, "M2", OPENING_UNIT, "rule")
                                                 + answers(r, "M2", OPENING_UNIT, "exception")}) > 1
                      and not every(r, "M2", OPENING_UNIT, "rule", 1)]
    check.value("III-C", "opening pair: readers that change their answer with the fuel inventory in every run",
                sorted(NAMES[r] for r in switch_with_inventory),
                sorted(["gpt-oss-120b", "gpt-oss-20b", "Claude Sonnet 5"]))
    check.value("III-C", "opening pair: readers that keep the turbine out on both questions",
                sorted(NAMES[r] for r in keep_rule_on_exception), sorted(["Llama-3.1-8B", "Qwen3-8B", "Qwen3-32B"]))
    check.value("III-C", "opening pair: readers that place it there on both",
                sorted(NAMES[r] for r in place_on_both), sorted(["DeepSeek-V4.1-Flash", "Qwen3-235B-A22B"]))
    check.value("III-C", "opening pair: readers whose answer varies between runs",
                [NAMES[r] for r in opening_varies], ["Llama-3.3-70B"])
    # "On the 44 pairs that set a rule against its exception, constrained readers with retrieval score 0.72 on
    # exception questions and 0.54 on rule questions, a difference of 0.18 [0.03, 0.31]. Across all 48 rule questions,
    # 41% of their answers choose the other branch, against 22% for exception questions. The open-weight reference
    # shows the same asymmetry (0.93 versus 0.67)." The 41% covers the 48 rule questions (44 against an exception, 4
    # against a complement), the 22% the 44 exception questions; "other branch" = the non-gold option that is not
    # none of these (exploratory.py section 4).
    paired_c = exploratory["paired_exception_minus_rule"]["tier C_load_bearing|read/M2@4000"]
    by_type = exploratory["accuracy_by_branch_type"]["tier C_load_bearing|read/M2@4000"]
    ref_asym = exploratory["paired_exception_minus_rule"][f"{REF_B}|read/M2@4000"]
    rule_vs_exception = {
        "exception": f"{paired_c['exception']:.2f}", "rule": f"{paired_c['rule']:.2f}",
        "difference": f"{paired_c['exception_minus_rule']:.2f} [{paired_c['ci'][0]:.2f}, {paired_c['ci'][1]:.2f}]",
        "other_branch_rule": f"{100 * by_type['rule']['wrong_mix_per_answer']['other_branch']:.0f}%",
        "other_branch_exception": f"{100 * by_type['exception']['wrong_mix_per_answer']['other_branch']:.0f}%",
        "open_weight_reference": f"{ref_asym['exception']:.2f} versus {ref_asym['rule']:.2f}"}
    check.value("III-C", "pairs that set a rule against its exception (exploratory.py)", paired_c["units"], 44)
    check.value("III-C", "rule questions / exception questions", [by_type["rule"]["units"],
                                                                  by_type["exception"]["units"]], [48, 44])
    for key, want in (("exception", "0.72"), ("rule", "0.54"), ("difference", "0.18 [0.03, 0.31]"),
                      ("other_branch_rule", "41%"), ("other_branch_exception", "22%"),
                      ("open_weight_reference", "0.93 versus 0.67")):
        check.value("III-C", f"rule vs exception, {key.replace('_', ' ')}", rule_vs_exception[key], want)
    # "With retrieval, the constrained panel answers both questions correctly on 0.33 of pairs and the server panel on
    # 0.45; with summaries the constrained share falls to 0.24. Even gpt-oss-20b misses one pair in four (0.76, against
    # 0.82 for the cloud comparator on the full text)." (exploratory.py section 2)
    both = exploratory["both_branches_correct"]
    pair_consistency = {label: f"{both[key]['est']:.2f}" for label, key in (
        ("constrained panel, retrieval", "tier C_load_bearing|read/M2@4000"),
        ("server panel, retrieval", "tier B|read/M2@4000"),
        ("constrained panel, summary", f"tier C_load_bearing|read/M5@4000/{B_BUILDER}"),
        ("gpt-oss-20b, retrieval", f"{GPT20}|read/M2@4000"),
        ("cloud comparator, full text", f"{A_READER}|read/M0@full"))}
    for (label, got), want in zip(pair_consistency.items(), ("0.33", "0.45", "0.24", "0.76", "0.82")):
        check.value("III-C", f"both questions of a pair correct, {label}", got, want)
    check.value("III-C", "gpt-oss-20b misses one pair in", round(1 / (1 - both[f"{GPT20}|read/M2@4000"]["est"])), 4)

    # Table I note r: "Reasons before answering at low effort, which cannot be switched off; the other readers were run
    # without reasoning": the observed read calls of each reader whose reasoning text is a reasoning phase.
    reasoning_modes = {}
    for reader in READERS:
        calls = with_reasoning = 0
        lengths = []
        for rec in stored.read_jsonl(READ_LEDGERS[reader]):
            if rec.get("kind") != "call" or rec.get("outcome") != "observed":
                continue
            calls += 1
            chars = reasoning_chars(rec)
            if chars is not None:
                with_reasoning += 1
                lengths.append(chars)
        reasoning_modes[NAMES[reader]] = {"observed_calls": calls, "calls_with_reasoning_text": with_reasoning,
                                          "share": with_reasoning / calls,
                                          "mean_reasoning_chars": fmean(lengths) if lengths else 0}
    reasoners = sorted(n for n, v in reasoning_modes.items() if v["share"] > 0.99)
    direct = sorted(n for n, v in reasoning_modes.items() if v["share"] < 0.01)
    check.value("Table I", "readers that reason before answering (note r)", reasoners, ["gpt-oss-120b", "gpt-oss-20b"])
    check.holds("Table I", "the other readers answer without a reasoning phase", len(direct) == len(READERS) - 2)

    printed = {
        "II-A": {"pairs": len(units), "questions": len(items), "documents": len(document_keys),
                 "pairs_by_source": units_by_family, "pair_kinds": {"+".join(k): v for k, v in pair_kinds.items()},
                 "multiple_choice_questions": len(gold_letters), "numeric_questions": len(numeric),
                 "arbitration_158": arbitration},
        "II-B": {"budget_share_of_full_procedure": budget_share,
                 "compressor_input_tokens": compressor.INPUT_FACTOR * 4000},
        "II-C": {"single_reader_retrieval_4k_half_width": {"printed": "4--6 points", "report": report_hw,
                                                           "own_bootstrap": own_hw,
                                                           "widest_memory_half_width_at_4k": widest_memory_hw}},
        "Table I": {"accuracy": {name: dict(zip((h for h, _ in TABLE_I_COLUMNS), cells))
                                 for name, cells in printed_table.items()},
                    "token_row": token_row, "completed_compression_questions": completed,
                    "reason_before_answering": reasoners},
        "Table II": {"strings": table_ii_strings, "deepseek_built_memory_gaps": memory_row,
                     "cloud_comparator_gaps": cloud_row},
        "III": {"deepseek_input_tokens": deepseek_tokens},
        "III-A": {"memories_below_retrieval_every_open_weight_reader": below_retrieval,
                  "points_lost": points_lost,
                  "largest_build_share": largest_build_share["build_share"],
                  "deepseek_built_memory_gap_points": gap_points,
                  "no_branch_answers": dict(zip(("retrieval", "summary", "records"), no_branch)),
                  "wrong_branch_answers": dict(zip(("retrieval", "summary", "records"), wrong_branch)),
                  "manual_36_exemption": m36_statement},
        "III-B": {"server_compression_tokens": f"{server_range['M6']} vs {server_range['M2']}",
                  "server_compression_ratios": server_ratios,
                  "compression_token_split": compression_split,
                  "qwen3_32b_time_outs_as_failures": qwen32_failures},
        "III-C": {"constrained_retrieval_range": retrieval_range,
                  "verdict_wording": {"retrieval_gap_vs_REF_B": retrieval_gap,
                                      "gptoss20b_reader_outcomes": gpt20_outcomes,
                                      "class_outcomes": numbers["tier_outcomes"]},
                  "gptoss20b_differences": contrasts, "rule_vs_exception": rule_vs_exception,
                  "pair_consistency": pair_consistency},
    }
    mismatches = check.mismatches()
    report = {
        "schema": "paper.printed_values",
        "sources": [source(p) for p in (ITEMS, *RUN_ROWS, args.exploratory, args.numbers)],
        "population_rule": "procedures.analysis.spec.in_population; 158 questions, two per pair",
        "bootstrap": {"replicates": N_BOOT, "seed": SEED, "cluster": "unit", "unit_order": "sorted unit_id",
                      "ci": "percentile 95%, sorted draw indices 50 and 1949 (zero-based)",
                      "paired_draws": True, "repeat_resampling": False, "reader_resampling": False,
                      "note": "Completion-adjusted panel intervals condition on the named readers; the report's panel "
                              "intervals also resample readers."},
        "estimands": {
            "main_table": "Mean of observed repeat correctness within question, then equally across observed "
                          "questions; directly format unrounded mean to three decimals.",
            "completion_adjusted": "Correct observed answers / (158 questions x 3 planned reads); every non-observed "
                                   "planned read contributes zero task success.",
            "completion_scope": "the recorded run, route time-outs included",
            "tier": "Equal mean across fixed named readers, paired by question and unit; no tier verdicts recomputed.",
            "compressor": "B readers use DeepSeek-built M6; C readers use their own compressor. B/C differences "
                          "conflate reader panel and compressor assignment.",
        },
        "population": {
            "questions": len(items), "units": len(units), "documents_family_doc": len(document_keys),
            "documents_definition": "Unique (family, doc) identifiers; document revisions separately reported below.",
            "document_keys": [list(x) for x in document_keys],
            "document_revision_count": len({(it["family"], it["doc"], str(it.get("revision")))
                                            for it in items.values()}),
            "families": dict(sorted(Counter(it["family"] for it in items.values()).items())),
            "units_by_family": units_by_family,
            "lettered_questions": len(gold_letters), "numeric_or_unlettered_questions": len(items) - len(gold_letters),
            "gold_letter_counts": {letter: letter_counts.get(letter, 0) for letter in "ABC"},
            "raw_gold_label_counts": dict(sorted(Counter(str(it.get("gold_label")) for it in items.values()).items())),
            "none_of_these_gold_count": normalized_gold.count("none of these"),
            "not_in_effect_gold_count": sum(g.replace(" ", "_") == "not_in_effect" for g in normalized_gold),
            "abstain_or_na_gold_count": sum(g in {"abstain", "n/a", "na", "not applicable"} for g in normalized_gold),
            "lettered_number_of_options_counts": dict(sorted(Counter(len(items[i]["options"])
                                                                     for i in gold_letters).items()))},
        "main_table_direct_3dp": table,
        "completion_adjusted_readers": completion,
        "completion_adjusted_tiers": tier_completion,
        "gptoss20b_vs_fixed_REF_B": gpt_check,
        "option_aware_error_mix_check": {
            "scope": "The constrained panel's four readers, all three reads, primary population, M2/M5_B/M7_B at 4k",
            "definition": "exploratory.py's gate (a wrong answer of class E-other/value_not_in_doc), with a literal "
                          "answer letter resolved to that question's option before testing for none of these or "
                          "not_in_effect.",
            "conditions": error_mix_check},
        "primary_resource_check": {
            "population": "158 primary questions, three reads each; 474 observed reads in each checked condition",
            "provider_input_definition": "tokens_input_uncached + tokens_input_cached",
            "arms": resource_arms, "ratios_to_B1_full_text": resource_ratios,
            "exploratory_tokens_match": tokens_match,
            "build_amortization": {
                "queries_per_memory": 1000,
                "scope": "Server and constrained readers, DeepSeek-built summary and records at 4k (the memories of "
                         "Fig. 2)",
                "weighting": "Mean token workload across the primary population's answered reads; not an unweighted "
                             "mean over unique document revisions",
                "share_definition": "mean(build_tokens)/1000 divided by mean(input + output + query_memory_tokens + "
                                    "build_tokens/1000)",
                "configurations": build_shares, "maximum": largest_build_share,
                "maximum_share_below_2_percent": below_2_percent}},
        "example_units_check": {
            "scope": "All observed reads; retrieval = M2@4000, summary = DeepSeek-built M5@4000; open-weight readers "
                     "have three reads, the cloud comparator one",
            "opening_pair": {"unit": OPENING_UNIT,
                             "both_correct_every_read_retrieval": [NAMES[r] for r in switch_with_inventory],
                             "not_placed_on_both_every_read_retrieval": [NAMES[r] for r in keep_rule_on_exception],
                             "placed_on_both_every_read_retrieval": [NAMES[r] for r in place_on_both],
                             "answers": {NAMES[r]: {f"{cond}/{kind}": answers(r, cond, OPENING_UNIT, kind)
                                                    for cond in ("M2", "M5_B") for kind in ("rule", "exception")}
                                         for r in READERS}},
            "m36_exemption": {"unit": M36_UNIT, "open_weight_readers": len(OPEN_WEIGHT),
                              "correct_every_read_retrieval": [NAMES[r] for r in m36_retrieval_right],
                              "wrong_every_read_summary": [NAMES[r] for r in m36_summary_wrong],
                              "rule_outcome_every_read_summary": [NAMES[r] for r in m36_summary_rule],
                              "answers": {NAMES[r]: {cond: answers(r, cond, M36_UNIT, "exception")
                                                     for cond in ("M2", "M5_B")} for r in OPEN_WEIGHT},
                              "claim": "seven of eight open-weight readers answer correctly in every retrieval run; "
                                       "all eight choose the rule outcome in every summary run"},
            "opening_pair_varies_between_runs": [NAMES[r] for r in opening_varies]},
        "printed": printed,
        "reasoning_modes": {"definition": "Observed read calls whose provider reasoning text is non-empty and is not "
                                          "an answer JSON object",
                            "ledgers": {NAMES[r]: READ_LEDGERS[r].relative_to(ROOT).as_posix() for r in READERS},
                            "readers": reasoning_modes, "reason_before_answering": reasoners,
                            "answer_without_reasoning_phase": direct},
        "checks": {"passed": not mismatches, "values": len(check.rows), "mismatches": len(mismatches),
                   "table": check.rows},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target = args.out_dir / "printed_values.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(check.table())
    print()
    for tier, vals in tier_completion.items():
        p = vals["paired_M6_minus_M2"]
        print(f"tier {tier}, compression minus retrieval over every planned read: {p['difference']:.6f}; "
              f"paired unit CI {p['difference_ci95']}")
    print(f"gpt-oss-20b minus REF-B with retrieval (recomputed): {gpt_check['difference']:.6f}; "
          f"CI {gpt_check['difference_ci95']}")
    print("input ratio, gpt-oss-20b retrieval / DeepSeek full text: "
          f"{resource_ratios['gptoss20b_retrieval_4k']['input_ratio_to_B1_full_text']:.6f}")
    print(f"largest amortized build share at 1,000 queries: {largest_build_share['build_share']:.6%} "
          f"({largest_build_share['name']}, {largest_build_share['condition']})")
    print(f"{target}: {len(check.rows)} values compared, {len(mismatches)} not matching the printed ones (offline)")
    for r in mismatches:
        print(f"  NOT MATCHING  {r['where']}: {r['what']}: {r['value']!r} (printed {r['printed']!r})")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
