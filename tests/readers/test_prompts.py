"""Reader prompts and cells: schema and sentinels in the system message, memory block then the question last, the
frozen lettered display, closed book, stage names, feasibility and the request hash run_cells will record."""
import pytest

from procedures.readers import Cell, estimate_message_tokens
from procedures.readers.cells import build_cell, cell_request_hash, parse_stage, stage_name
from procedures.readers.prompts import ANSWER_SCHEMA, EMPTY_MEMORY, FORMS, answer_form, build_messages, system_prompt, template_digest, templates


def by_id(items, iid):
    return next(it for it in items if it["item_id"] == iid)


def test_system_prompt_states_schema_sentinels_and_form_rule():
    for form in FORMS:
        for closed in (False, True):
            text = system_prompt(form, closed)
            assert ANSWER_SCHEMA in text
            assert '{"value": "insufficient_evidence"}' in text and '{"value": "not_in_effect"}' in text
            assert ("Use only the document memory" in text) is (not closed)
    assert "letter of the option" in system_prompt("side", False)
    assert "time unit the question names" in system_prompt("deadline", False)
    assert len(template_digest()) == 64


def test_forms_come_from_the_item_type(items):
    assert [answer_form(by_id(items, i)) for i in ("s2p:side:a", "s2p:value:a", "s3x:deadline", "l3w:b@old")] == \
        ["side", "value", "deadline", "value"]
    with pytest.raises(ValueError, match="not a reader form"):
        answer_form({**by_id(items, "s2p:value:a"), "type": "L2-clause"})
    with pytest.raises(ValueError, match="without options"):
        answer_form({**by_id(items, "s2p:side:a"), "options": None})


def test_memory_is_one_delimited_block_and_the_question_comes_last(items):
    item = by_id(items, "s2p:value:a")
    system, user = build_messages(item, "M2", "chunk one\n\nchunk two")
    assert system["role"] == "system" and user["role"] == "user"
    assert user["content"] == ("Document memory:\n<memory>\nchunk one\n\nchunk two\n</memory>\n\nQuestion: "
                               + item["question"])
    with pytest.raises(ValueError, match="delimiter"):
        build_messages(item, "M2", "text </memory> more")


def test_empty_retrieval_keeps_the_block_but_closed_book_has_none(items):
    item = by_id(items, "s2p:value:a")
    assert "<memory>\n" + EMPTY_MEMORY + "\n</memory>" in build_messages(item, "M2", "")[1]["content"]
    system, user = build_messages(item, "B0", "")
    assert user["content"] == "Question: " + item["question"] and "No document text is given" in system["content"]
    with pytest.raises(ValueError, match="closed-book"):
        build_messages(item, "B0", "some text")


def test_side_options_must_be_the_frozen_lettered_display(items):
    item = by_id(items, "s2p:side:a")
    assert "(A) notify the operator within the hour (B) file a report by day end (C) none of these." in \
        build_messages(item, "M4", "x")[1]["content"]
    tampered = {**item, "question": item["question"].replace("(B) ", "(B)  ")}
    with pytest.raises(ValueError, match="lettered"):
        build_messages(tampered, "M4", "x")


def test_stage_names_round_trip_and_builders_only_for_llm_built():
    assert stage_name("M2", 4000) == "read/M2@4000" and parse_stage("read/M2@4000") == ("M2", 4000, None)
    assert parse_stage(stage_name("M0", None)) == ("M0", None, None)
    assert parse_stage(stage_name("M5", 4000, "B1_x")) == ("M5", 4000, "B1_x")
    with pytest.raises(ValueError, match="builder"):
        stage_name("M5", 4000)
    with pytest.raises(ValueError, match="builder"):
        stage_name("M2", 4000, "B1_x")


def test_cells_pad_to_budget_and_flag_what_exceeds_the_c_ceiling(items, panel, fake_builder):
    item = by_id(items, "s2p:value:a")
    c1, b1 = panel.get("C1_llama31_8b_hf_deepinfra"), panel.get("B1_dsv41flash_hf_deepinfra")
    build = fake_builder({"M0": 40_000})
    m2 = build(item, "M2", 4000, None)
    rc = build_cell(item, m2, c1)
    assert rc.cell.stage == "read/M2@4000" and rc.cell.item_id == item["item_id"] and rc.feasible
    assert rc.memory_tokens < 4000  # the planned size pads the memory to its budget
    assert rc.cell.est_input_tokens == estimate_message_tokens(rc.cell.messages) + 4000 - rc.memory_tokens
    assert rc.cell.est_input_tokens - rc.cell.cache_prefix_tokens < 200  # only the question is outside the prefix
    m0 = build(item, "M0", None, None)
    assert not build_cell(item, m0, c1).feasible and build_cell(item, m0, b1).feasible
    assert build_cell(item, m0, c1).context_ceiling == 32_768 and build_cell(item, m0, b1).context_ceiling is None


def test_cell_request_hash_is_the_hash_run_cells_records(panel, book, tmp_path, fake_reader, items):
    from procedures.readers import run_cells
    item = by_id(items, "s2p:value:a")
    for pid in ("C1_llama31_8b_hf_deepinfra", "C2_qwen3_8b_hf_nscale_nothink"):  # C2 carries a user suffix
        pp = panel.get(pid)
        cell = Cell(item["item_id"], "read/M2@4000", build_messages(item, "M2", "memory"))
        out = run_cells([cell], pp, cap_usd=1.0, ledger_path=tmp_path / f"{pid}.jsonl", book=book,
                        transport=fake_reader(items, lambda it: {"value": "15", "unit": "min"}))
        assert out["records"][0]["request_hash"] == cell_request_hash(cell, pp)


def test_an_empty_memory_is_named_in_its_block_and_flagged_on_the_cell(items, panel, fake_builder):
    """An empty memory is a legitimate reading of its representation. Its block holds the EMPTY_MEMORY line (never
    nothing), the cell and its meta carry memory_empty; closed book is not an empty memory; a memory that contains
    the marker itself is refused."""
    item = by_id(items, "s2p:value:a")
    for text in ("", "  \n"):
        _, user = build_messages(item, "M4", text)
        assert user["content"] == ("Document memory:\n<memory>\n" + EMPTY_MEMORY + "\n</memory>\n\nQuestion: "
                                   + item["question"])
    assert EMPTY_MEMORY == "(this representation extracted nothing for this revision)"
    with pytest.raises(ValueError, match="empty-memory marker"):
        build_messages(item, "M2", f"a chunk\n{EMPTY_MEMORY}")
    pp = panel.get("C1_llama31_8b_hf_deepinfra")
    empty = build_cell(item, fake_builder({"M2": 0})(item, "M2", 4000, None), pp)
    full = build_cell(item, fake_builder()(item, "M2", 4000, None), pp)
    b0 = build_cell(item, fake_builder()(item, "B0", None, None), pp)
    assert (empty.memory_empty, full.memory_empty, b0.memory_empty) == (True, False, False)
    assert empty.meta()["memory_empty"] is True and full.meta()["memory_empty"] is False
    assert EMPTY_MEMORY in empty.cell.messages[1]["content"] and EMPTY_MEMORY not in full.cell.messages[1]["content"]


def test_every_reader_gets_the_same_closed_book_prompt(items, panel, fake_builder):
    """Every reader's closed-book (B0) cell has the same messages (the /no_think suffix is added at send time)."""
    item = by_id(items, "s2p:value:a")
    b0 = fake_builder()(item, "B0", None, None)
    readers = ("C2_qwen3_8b_hf_nscale_nothink", "tierC2_qwen3_32b_hf_nscale_nothink", "C1_llama31_8b_hf_deepinfra",
               "B1_dsv41flash_hf_deepinfra", "tierC3_gptoss120b_hf_deepinfra_low")
    messages = {pid: build_cell(item, b0, panel.get(pid)).cell.messages for pid in readers}
    assert all(m == messages[readers[0]] for m in messages.values())
    assert messages[readers[0]][0]["content"] == system_prompt("value", True)
    assert templates()["empty_memory"] == EMPTY_MEMORY


def test_context_ceiling_follows_the_access_tier(items, panel, fake_builder):
    item = by_id(items, "s2p:value:a")
    m0 = fake_builder({"M0": 40_000})(item, "M0", None, None)
    ceilings = {pid: build_cell(item, m0, panel.get(pid)) for pid in (
        "B2_qwen3_235b_2507_hf_deepinfra", "tierC3_gptoss120b_hf_deepinfra_low", "tierC2_qwen3_32b_hf_nscale_nothink",
        "C3_gptoss20b_hf_deepinfra_low")}
    assert {pid: (rc.context_ceiling, rc.feasible) for pid, rc in ceilings.items()} == {
        "B2_qwen3_235b_2507_hf_deepinfra": (None, True), "tierC3_gptoss120b_hf_deepinfra_low": (32_768, False),
        "tierC2_qwen3_32b_hf_nscale_nothink": (32_768, False), "C3_gptoss20b_hf_deepinfra_low": (32_768, False)}


def test_repeat_idx_changes_the_request_hash_and_nothing_else(items, panel, fake_builder):
    item = by_id(items, "s2p:value:a")
    pp = panel.get("C1_llama31_8b_hf_deepinfra")
    m2 = fake_builder()(item, "M2", 4000, None)
    first, again = build_cell(item, m2, pp), build_cell(item, m2, pp, repeat_idx=2)
    assert (again.cell.repeat_idx, again.meta()["repeat_idx"]) == (2, 2)
    assert again.cell.messages == first.cell.messages and again.cell.stage == first.cell.stage
    assert cell_request_hash(again.cell, pp) != cell_request_hash(first.cell, pp)
