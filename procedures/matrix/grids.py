"""Named reader grids for procedures.matrix.run_matrix (scripts/run_matrix.py --grid): conditions in the CLI form
"REP[@BUDGET|@full][/COMPRESSOR][:reader,...]", the readers, and the builder of the memories built once per revision
(summary memories M5 and records memories M7). A run has one builder (its build ledger is build__<builder>.jsonl), so
the builder arm of the builder x reader matrix is one grid per builder. The other representations: B0 closed book,
M0 full text, M2 BM25 retrieval, M4 constraint sentences, M6 query-focused compression (one LLM call per question).

The grids of the paper's four runs:
* main (data/runs/main): the server readers (B1 DeepSeek-V4.1-Flash, B2 Qwen3-235B-A22B-Instruct-2507 and
  Llama-3.3-70B-Instruct on ovhcloud; the deepinfra Llama candidate has no priced route and stays out) and the five
  single-GPU readers (access tier C3 gpt-oss-120b, access tier C2 Qwen3-32B and the three access-tier C1 models of
  panel slots C1, C2, C3). B0; M0@full for the server readers only (full text is the upper-bound reference);
  M2 / M4 / M5 / M7 at 2k / 4k / 16k, M5 and M7 built by BUILDER (DeepSeek-V4.1-Flash served directly); M6 at 4k
  only, in two forms: the server readers read the compression made by the B1 DeepSeek-V4.1-Flash direct profile
  ("M6@4000/B1_dsv41flash_direct_nothink:<server readers>", one memory per item that every server reader reads:
  Llama-3.3-70B on ovhcloud and Qwen3-235B have no direct route, and a self-compression through the HF router runs
  into its 120 s gateway), and the single-GPU readers compress for themselves ("M6@4000:<single-GPU readers>"). A
  cell over its reader's local context ceiling or its route's context (the Llama ovhcloud route holds 131,072 tokens,
  so its longest M0@full cells are not sent), or whose compression its compressor cannot take, is planned, recorded
  and not sent (procedures.readers.cells feasibility, per reason).
* cbuilt_gptoss20b / cbuilt_qwen32b (data/runs/records_built_by_gptoss20b, data/runs/records_built_by_qwen32b):
  records memories built by a single-GPU model and read by the five single-GPU readers (the self-built control): M7
  at 2k / 4k / 16k, built by the access-tier C1 gpt-oss-20b profile or by the access-tier C2 Qwen3-32B profile; one
  grid per builder, since a run has one builder. M7 only: the builder comparison is made on M7 (a gpt-oss-20b M5 is
  not made, since its reasoning allowance takes its output cap).
* cloud_reference (data/runs/cloud_comparator): the cloud comparator, A2 Sonnet 5 without thinking, on B0, M0@full
  and M2@4000, read once (k = 1); nothing is built, so the grid has no builder.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Grid:
    conditions: tuple
    readers: tuple
    builder: str | None


# builder of M5 / M7: the B1 model served by DeepSeek directly (its Hugging Face route timed out on long generations);
# the B1 reader stays on the Hugging Face route
BUILDER = "B1_dsv41flash_direct_nothink"
BUDGETS = (2_000, 4_000, 16_000)
M6_BUDGETS = (4_000,)                                    # per-question compression is run at 4k only
LADDER_REPRESENTATIONS = ("M2", "M4", "M5", "M6", "M7")  # M7 carries the builder comparison, M5 is its control
C_BUILT_REPRESENTATIONS = ("M7",)                        # the builder comparison is made on M7 only
SERVER_READERS = ("B1_dsv41flash_hf_deepinfra", "B2_qwen3_235b_2507_hf_deepinfra", "tierB_llama33_70b_hf_ovhcloud")
SINGLE_GPU_READERS = ("tierC3_gptoss120b_hf_deepinfra_low", "tierC2_qwen3_32b_hf_nscale_nothink",
                      "C1_llama31_8b_hf_deepinfra", "C2_qwen3_8b_hf_nscale_nothink", "C3_gptoss20b_hf_deepinfra_low")
C_BUILDER = "C3_gptoss20b_hf_deepinfra_low"              # access tier C1: the first single-GPU builder of M7
C_BUILDER_QWEN32B = "tierC2_qwen3_32b_hf_nscale_nothink"  # access tier C2: the second single-GPU builder of M7
M6_B_COMPRESSOR = BUILDER                                # the server readers' M6 compressor (module docstring)
A_REFERENCE = "A2_sonnet5_nothink"                       # the cloud comparator, read once


def _ladder(rep: str, budget: int) -> tuple[str, ...]:
    if rep != "M6":
        return (f"{rep}@{budget}",)
    return (f"M6@{budget}/{M6_B_COMPRESSOR}:{','.join(SERVER_READERS)}",
            f"M6@{budget}:{','.join(SINGLE_GPU_READERS)}")


MAIN_CONDITIONS = ("B0", f"M0@full:{','.join(SERVER_READERS)}",
                   *(c for rep in LADDER_REPRESENTATIONS
                     for budget in (M6_BUDGETS if rep == "M6" else BUDGETS) for c in _ladder(rep, budget)))
CBUILT_CONDITIONS = tuple(f"{rep}@{budget}" for rep in C_BUILT_REPRESENTATIONS for budget in BUDGETS)
# No M5 / M7 condition, so no builder (procedures.matrix.run_matrix requires a builder exactly when one is built).
CLOUD_REFERENCE_CONDITIONS = ("B0", f"M0@full:{A_REFERENCE}", "M2@4000")

GRIDS = {
    "main": Grid(MAIN_CONDITIONS, SERVER_READERS + SINGLE_GPU_READERS, BUILDER),
    "cbuilt_gptoss20b": Grid(CBUILT_CONDITIONS, SINGLE_GPU_READERS, C_BUILDER),
    "cbuilt_qwen32b": Grid(CBUILT_CONDITIONS, SINGLE_GPU_READERS, C_BUILDER_QWEN32B),
    "cloud_reference": Grid(CLOUD_REFERENCE_CONDITIONS, (A_REFERENCE,), None),
}
