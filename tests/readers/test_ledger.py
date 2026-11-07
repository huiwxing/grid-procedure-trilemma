"""Append-only ledger: identity binding, resume semantics, spend-cap reservations, torn lines."""
import json

import pytest

from procedures.readers import Ledger, LedgerError, SpendCapExceeded, request_hash
from procedures.readers.ledger import SCHEMA, identity_clash
from procedures.readers.client.config import digest

HEADER = {"profile_id": "p", "profile_fingerprint": "f" * 64, "price_book_sha256": "b" * 64}


def call(hash_, outcome="observed", usd=0.01):
    return {"kind": "call", "request_hash": hash_, "outcome": outcome, "usd": usd, "usd_counted": usd}


def test_request_hash_normalises_crlf_and_separates_params_and_repeats():
    base = request_hash("p", "f", [{"role": "user", "content": "a\nb"}], {}, 0)
    assert base == request_hash("p", "f", [{"role": "user", "content": "a\r\nb"}], {}, 0)
    assert base != request_hash("p", "f", [{"role": "user", "content": "a\nb"}], {"max_output_tokens": 5}, 0)
    assert base != request_hash("p", "f", [{"role": "user", "content": "a\nb"}], {}, 1)
    assert base != request_hash("q", "f", [{"role": "user", "content": "a\nb"}], {}, 0)
    assert base != request_hash("p", "g", [{"role": "user", "content": "a\nb"}], {}, 0)


def test_request_hash_binds_the_user_message_suffix_only_when_set():
    sent = [{"role": "user", "content": "q /no_think"}]
    legacy = digest({"profile_id": "p", "profile_fingerprint": "f", "messages": sent, "params": {}, "repeat_idx": 0})
    assert request_hash("p", "f", sent, {}, 0) == request_hash("p", "f", sent, {}, 0, user_message_suffix=None) == legacy
    bound = request_hash("p", "f", sent, {}, 0, user_message_suffix=" /no_think")
    assert bound != legacy and bound != request_hash("p", "f", sent, {}, 0, user_message_suffix=" /think")


def test_user_message_suffix_is_part_of_the_run_identity(tmp_path):
    path = tmp_path / "run.jsonl"
    Ledger(path, HEADER, cap_usd=1.0)  # header written without the field (as before it existed)
    Ledger(path, {**HEADER, "user_message_suffix": None}, cap_usd=1.0)
    with pytest.raises(LedgerError, match="user_message_suffix"):
        Ledger(path, {**HEADER, "user_message_suffix": " /no_think"}, cap_usd=1.0)


def test_append_only_resume_and_latest_record_wins(tmp_path):
    path = tmp_path / "run.jsonl"
    ledger = Ledger(path, HEADER, cap_usd=1.0)
    ledger.append(call("h1"))
    ledger.append(call("h2", outcome="no_response", usd=0.0))
    first_bytes = path.read_bytes()
    reopened = Ledger(path, HEADER, cap_usd=1.0)
    assert reopened.done("h1")["usd"] == 0.01
    assert reopened.done("h2") is None  # never answered -> resent on resume
    reopened.append(call("h2"))
    assert path.read_bytes().startswith(first_bytes)  # nothing rewritten
    assert Ledger(path, HEADER, cap_usd=1.0).done("h2") is not None
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["kind"] == "header" and len(lines) == 4
    assert b"\r\n" not in path.read_bytes()


def test_reopening_with_another_identity_is_refused(tmp_path):
    path = tmp_path / "run.jsonl"
    Ledger(path, HEADER, cap_usd=1.0)
    with pytest.raises(LedgerError, match="run identity differs"):
        Ledger(path, {**HEADER, "profile_fingerprint": "e" * 64}, cap_usd=1.0)
    assert identity_clash(HEADER, {**HEADER, "created_at": "x", "harness_digest": "y"}) == {}   # not identity
    assert identity_clash(HEADER, {**HEADER, "price_book_sha256": "c" * 64}) == {
        "price_book_sha256": ("b" * 64, "c" * 64)}
    assert identity_clash(HEADER, {**HEADER, "user_message_suffix": " /no_think"}) == {
        "user_message_suffix": (None, " /no_think")}


def test_spend_cap_counts_spent_plus_in_flight(tmp_path):
    ledger = Ledger(tmp_path / "run.jsonl", HEADER, cap_usd=0.10)
    ledger.append(call("h1", usd=0.06))
    held = ledger.reserve(0.03)
    with pytest.raises(SpendCapExceeded, match="cap"):
        ledger.reserve(0.02)  # 0.06 spent + 0.03 in flight + 0.02 > 0.10
    ledger.append(call("h2", usd=0.01), held)  # settles the 0.03 reservation at its real cost
    assert ledger.spent_usd() == pytest.approx(0.07)
    ledger.reserve(0.02)
    assert Ledger(tmp_path / "run.jsonl", HEADER, cap_usd=0.10).spent_usd() == pytest.approx(0.07)


def test_torn_or_malformed_lines_are_reported_not_skipped(tmp_path):
    path = tmp_path / "run.jsonl"
    Ledger(path, HEADER, cap_usd=1.0).append(call("h1"))
    with open(path, "ab") as stream:
        stream.write(b'{"kind": "call", "request_ha')
    with pytest.raises(LedgerError, match="torn final line"):
        Ledger(path, HEADER, cap_usd=1.0)
    path.write_bytes(path.read_bytes().rsplit(b"\n", 1)[0] + b"\nnot json\n")
    with pytest.raises(LedgerError, match="malformed JSON"):
        Ledger(path, HEADER, cap_usd=1.0)


@pytest.fixture
def parses(monkeypatch):
    """Counts full parses of a ledger file (procedures.readers.ledger._parse_ledger calls)."""
    import procedures.readers.ledger as module
    calls = []
    real = module._parse_ledger
    monkeypatch.setattr(module, "_parse_ledger", lambda path, raw: calls.append(len(raw)) or real(path, raw))
    return calls


def readings(ledger):
    return (ledger.done("h1"), ledger.done("h2"), ledger.done("h3"), ledger.spent_usd(), ledger.committed_usd(),
            ledger.pending_unknown_usd(), sorted(r["request_hash"] for r in ledger.records()))


def test_reopening_the_same_path_parses_it_once(tmp_path, parses):
    path = tmp_path / "run.jsonl"
    writer = Ledger(path, HEADER, cap_usd=1.0)
    writer.append(call("h1"))
    writer.append(call("h2", outcome="no_response", usd=0.0))
    first, second = Ledger(path, HEADER, cap_usd=1.0), Ledger(path, HEADER, cap_usd=1.0)
    assert len(parses) == 1
    assert readings(first) == readings(second) == readings(writer)
    assert first.header == second.header == writer.header and first.header is not second.header


def test_reopening_reads_lines_appended_by_a_ledger_of_this_process(tmp_path, parses):
    path = tmp_path / "run.jsonl"
    Ledger(path, HEADER, cap_usd=1.0).append(call("h1"))
    first = Ledger(path, HEADER, cap_usd=1.0)
    first.append(call("h2", usd=0.02))
    second = Ledger(path, HEADER, cap_usd=1.0)
    assert len(parses) == 1
    assert second.done("h2")["usd"] == 0.02 and second.spent_usd() == pytest.approx(0.03)
    assert readings(second) == readings(first)
    second.append(call("h3"))       # the first instance still takes it from its own tail
    assert first.done("h3") is None and first.spent_usd() == pytest.approx(0.04) and first.done("h3") is not None


def test_reopening_reads_lines_appended_by_another_process(tmp_path, parses):
    import subprocess
    import sys
    path = tmp_path / "run.jsonl"
    Ledger(path, HEADER, cap_usd=1.0).append(call("h1"))
    before = Ledger(path, HEADER, cap_usd=1.0)
    line = json.dumps({"schema": SCHEMA, **call("h2", usd=0.05)}) + "\n"
    subprocess.run([sys.executable, "-c", f"open({str(path)!r}, 'ab').write({line.encode()!r})"], check=True)
    after = Ledger(path, HEADER, cap_usd=1.0)
    assert len(parses) == 1
    assert after.done("h2")["usd"] == 0.05 and after.spent_usd() == pytest.approx(0.06)
    assert before.spent_usd() == pytest.approx(0.06)        # the old instance's _refresh path is unchanged


def test_a_truncated_or_replaced_file_is_parsed_again(tmp_path, parses):
    path = tmp_path / "run.jsonl"
    writer = Ledger(path, HEADER, cap_usd=1.0)
    writer.append(call("h1"))
    writer.append(call("h2"))
    assert Ledger(path, HEADER, cap_usd=1.0).done("h2") is not None
    assert len(parses) == 1
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(lines[:2]))                               # truncated: h2 gone
    shorter = Ledger(path, HEADER, cap_usd=1.0)
    assert len(parses) == 2 and shorter.done("h2") is None and shorter.spent_usd() == pytest.approx(0.01)
    other = tmp_path / "other.jsonl"
    Ledger(other, HEADER, cap_usd=1.0).append(call("h3", usd=0.5))
    other.replace(path)                                                  # another file object at the path
    replaced = Ledger(path, HEADER, cap_usd=1.0)
    assert len(parses) == 3 and replaced.done("h1") is None and replaced.done("h3") is not None
    rewritten = path.read_bytes().replace(b'"h3"', b'"h4"')              # same length, rewritten in place
    path.write_bytes(rewritten)
    assert Ledger(path, HEADER, cap_usd=1.0).done("h4") is not None and len(parses) == 4


def test_a_torn_line_after_a_cached_parse_still_raises(tmp_path, parses):
    path = tmp_path / "run.jsonl"
    Ledger(path, HEADER, cap_usd=1.0).append(call("h1"))
    Ledger(path, HEADER, cap_usd=1.0)
    with open(path, "ab") as stream:
        stream.write(b'{"kind": "call"}\n{"kind": "call", "request_ha')
    with pytest.raises(LedgerError, match="torn final line"):
        Ledger(path, HEADER, cap_usd=1.0)
    path.write_bytes(path.read_bytes().rsplit(b"\n", 1)[0] + b"\n")     # complete, but not a call record
    with pytest.raises(LedgerError, match=r":3: not a call record"):
        Ledger(path, HEADER, cap_usd=1.0)
