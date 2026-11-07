"""Route context ceilings: a cell is not sent when its planned reference input x 1.03 plus the output cap exceeds the
context of its reader's hosted route. The local ceiling of the tier-C readers (32,768 reference tokens + cap,
unchanged) is checked first. The reason is recorded per cell, counted per reader in the estimate and per reader x
condition in the summary, and carried in the `access` block of the scored row."""
import json

import pytest

from procedures.readers.cells import LOCAL_CEILING, ROUTE_CONTEXT, build_cell, infeasible_reason
from procedures.matrix.run_matrix import Condition, run_matrix

LLAMA = "tierB_llama33_70b_hf_ovhcloud"        # route context 131,072, cap 1,024
B2 = "B2_qwen3_235b_2507_hf_deepinfra"         # route context 262,144
C1 = "C1_llama31_8b_hf_deepinfra"              # local 32,768; route 131,072
QWEN8 = "C2_qwen3_8b_hf_nscale_nothink"        # local 32,768; route 40,960; cap 2,048
DIRECT = "B1_dsv41flash_direct_nothink"        # not on the HF router: no route context


# The largest planned reference input a route takes: ceil(edge x 1.03) + cap <= context < ceil((edge + 1) x 1.03) + cap.
# Llama: (131,072 - 1,024) / 1.03 = 126,260.2; B2: (262,144 - 1,024) / 1.03 = 253,514.6.
@pytest.mark.parametrize("pid, context, cap, edge", [(LLAMA, 131_072, 1_024, 126_260), (B2, 262_144, 1_024, 253_514)])
def test_route_context_boundary(panel, pid, context, cap, edge):
    pp = panel.get(pid)
    assert pp.route_context == context
    assert -(-edge * 103 // 100) + cap <= context < -(-(edge + 1) * 103 // 100) + cap
    assert infeasible_reason(pp, edge, cap) is None
    assert infeasible_reason(pp, edge + 1, cap) == ROUTE_CONTEXT
    assert infeasible_reason(pp, context - cap, cap) == ROUTE_CONTEXT     # fits the raw context, not the allowance


def test_the_local_ceiling_comes_first_and_is_unchanged(panel):
    c1, qwen = panel.get(C1), panel.get(QWEN8)
    assert infeasible_reason(c1, 32_768 - 1_024, 1_024) is None               # planned + cap = 32,768 fits
    assert infeasible_reason(c1, 32_768 - 1_024 + 1, 1_024) == LOCAL_CEILING  # no 1.03 on the local ceiling
    assert infeasible_reason(qwen, 200_000, 2_048) == LOCAL_CEILING           # over both: the local reason
    # route 40,960 never binds below the local 32,768 for a C reader: 30,720 x 1.03 + 2,048 = 33,690
    assert infeasible_reason(qwen, 32_768 - 2_048, 2_048) is None


def test_a_profile_without_a_route_context_has_no_route_limit(panel):
    pp = panel.get(DIRECT)
    assert pp.route_context is None and infeasible_reason(pp, 10_000_000, 4_096) is None


def test_build_cell_records_the_reason_and_the_route_context(items, panel, fake_builder):
    item = items[0]
    m0 = fake_builder({"M0": 128_000})(item, "M0", None, None)
    cells = {pid: build_cell(item, m0, panel.get(pid)) for pid in (LLAMA, B2, C1)}
    planned = cells[LLAMA].cell.est_input_tokens
    assert 126_260 < planned < 131_072 - 1_024          # fits the raw context, not with the 1.03 allowance
    assert {pid: (rc.feasible, rc.infeasible_reason, rc.route_context) for pid, rc in cells.items()} == {
        LLAMA: (False, ROUTE_CONTEXT, 131_072), B2: (True, None, 262_144), C1: (False, LOCAL_CEILING, 131_072)}
    meta = cells[LLAMA].meta()
    assert (meta["infeasible_reason"], meta["route_context"], meta["feasible"]) == (ROUTE_CONTEXT, 131_072, False)


def test_over_route_cells_are_recorded_not_sent_and_counted_per_reader(items, panel, book, tmp_path, fake_builder,
                                                                       fake_reader, gold_answer):
    reader = fake_reader(items, gold_answer)
    kw = dict(build_memory=fake_builder({"M0": 128_000}), panel=panel, cap_usd=5.0, corpus_index_sha256="0" * 64,
              out_root=tmp_path, book=book, sleep=lambda s: None, text_of=lambda p: "59.0 Hz 15 minutes")
    conditions = [Condition("M0"), Condition("B0")]
    dry = run_matrix(items, conditions, [LLAMA, B2, C1], {"corpus": "fake"}, label="dry", **kw)
    est = json.loads((tmp_path / dry["run_id"] / dry["estimate"]).read_text(encoding="utf-8"))["readers"]
    n = len(items)
    assert {pid: (e["infeasible"], e["infeasible_by_reason"]) for pid, e in est.items()} == {
        LLAMA: (n, {ROUTE_CONTEXT: n}), B2: (0, {}), C1: (n, {LOCAL_CEILING: n})}
    assert est[LLAMA]["pending"] == n and est[B2]["pending"] == 2 * n        # B0 only for Llama; M0 + B0 for B2
    live = run_matrix(items, conditions, [LLAMA, B2], {"corpus": "fake"}, label="live", dry_run=False,
                      transport=reader, concurrency=1, **kw)
    assert len(reader.calls) == 3 * n                                        # Llama B0 + B2 M0 and B0
    assert not any(c["model"].startswith("meta-llama/Llama-3.3") and "<memory>" in c["messages"][-1]["content"]
                   for c in reader.calls)
    run_dir = tmp_path / live["run_id"]
    rows = [json.loads(x) for x in (run_dir / live["outputs"]["rows"]).read_text(encoding="utf-8").splitlines()]
    llama_m0 = [r for r in rows if r["profile_id"] == LLAMA and r["representation"] == "M0"]
    assert len(llama_m0) == n and all(r["outcome"] == "not_sent_infeasible" and r["capability"] is None
                                      and r["access"]["infeasible_reason"] == ROUTE_CONTEXT
                                      and r["access"]["route_context"] == 131_072
                                      and r["access"]["local_context_ceiling"] is None for r in llama_m0)
    assert all(r["access"]["infeasible_reason"] is None for r in rows if r not in llama_m0)
    summary = json.loads((run_dir / live["outputs"]["summary"]).read_text(encoding="utf-8"))["conditions"]
    assert summary[f"{LLAMA}|read/M0@full"]["infeasible_by_reason"] == {ROUTE_CONTEXT: n}
    assert summary[f"{LLAMA}|read/M0@full"]["outcomes"] == {"not_sent_infeasible": n}
    assert "infeasible_by_reason" not in summary[f"{B2}|read/M0@full"]


def test_a_compression_its_compressor_cannot_take_names_that_reason(items, panel, fake_builder):
    from procedures.matrix.run_matrix import plan_grid
    by_id = {it["item_id"]: it for it in items}
    memories = {(iid, "B0@full"): fake_builder()(it, "B0", None, None) for iid, it in by_id.items()}
    first = next(iter(by_id))
    grid = plan_grid(by_id, [Condition("B0")], {C1: panel.get(C1)}, memories, None,
                     infeasible={(first, "B0@full"): LOCAL_CEILING})
    reasons = {rc.item_id: rc.infeasible_reason for rc in grid[C1]}
    assert reasons.pop(first) == f"compression_{LOCAL_CEILING}" and set(reasons.values()) == {None}
    assert all(rc.feasible is (rc.infeasible_reason is None) for rc in grid[C1])
