"""procedures.readers.tokens: segment-memoised counting and the insertion re-count are exact (equal to encoding the whole text
with o200k_base), and the chars/4 fallback is unchanged. procedures.readers.pricing: a price book is parsed once per file
version (modification time, size) and again when the file changes."""
from __future__ import annotations

import json
import random

import pytest

import procedures.readers.tokens as tokens
from procedures.readers.pricing import DEFAULT_PRICE_BOOK, load_price_book
from procedures.readers.tokens import estimate_tokens, recount, segments

pytestmark = pytest.mark.skipif(tokens.token_method() != "tiktoken:o200k_base",
                                reason="the reference tokenizer is not in the local tiktoken cache")
# the characters around a piece boundary of o200k_base: newlines, the slash its punctuation branch takes after them,
# other whitespace (incl. non-ASCII), punctuation, letters, digits, CJK, a contraction, a special-token string
ALPHABET = [*"ab Z9 .,/-*#([\n\n\n\r\t  ", " ", " ", "　", "é", "中", "\x1c", "'s",
            "<|endoftext|>", "//", "\n/", ".\n", " \n", "\n ", "\n- ", "\n\n["]


def whole(text: str) -> int:
    return len(tokens._encoder()[0].encode(text, disallowed_special=()))


def random_text(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, n)))


def test_segments_tile_the_text_and_cut_only_after_a_newline_before_a_segment_start():
    text = "a.\n\nB c\n- x\n/y\n [z\r\n(q"
    assert segments(text) == ["a.\n\n", "B c\n", "- x\n/y\n [z\r\n", "(q"]
    rng = random.Random(7)
    for _ in range(2_000):
        t = random_text(rng, 40)
        parts = segments(t)
        assert "".join(parts) == t
        for left, right in zip(parts, parts[1:]):
            assert left[-1] in "\r\n" and right[0] in tokens.SEGMENT_START


def test_segment_sums_equal_whole_text_encoding():
    rng = random.Random(0)
    for _ in range(20_000):
        t = random_text(rng, 40)
        assert estimate_tokens(t) == whole(t), repr(t)
    paragraphs = [f"{i}.{i} Section {i}\n\nThe operator shall hold {i} MW; see /docs/{i}.\n- item {i}\n" for i in range(50)]
    text = "[Header, Revision 3 (effective 2024-01-01)]\n" + "\n\n".join(paragraphs)
    assert estimate_tokens(text) == whole(text) and len(segments(text)) > 100


def test_recount_after_an_insertion_or_any_other_change_is_exact():
    rng = random.Random(1)
    for _ in range(20_000):
        before = random_text(rng, 30)
        if rng.random() < 0.8:
            i = rng.randint(0, len(before))
            after = before[:i] + random_text(rng, 12) + before[i:]
        else:                                   # not an insertion: counted whole
            after = random_text(rng, 30)
        assert recount(before, whole(before), after) == whole(after), (before, after)


def test_recount_counts_only_around_the_insertion(monkeypatch):
    units = [f"Unit {i} says the limit is {i} MW." for i in range(40)]
    before = "[Doc, Revision 1]\n" + "\n\n".join(units[:20] + units[21:])
    after = "[Doc, Revision 1]\n" + "\n\n".join(units)
    seen = []
    real = tokens.estimate_tokens
    monkeypatch.setattr(tokens, "estimate_tokens", lambda text, **kw: seen.append(text) or real(text, **kw))
    assert recount(before, whole(before), after) == whole(after)
    assert max(len(t) for t in seen) < 3 * len(units[20]) + 10     # the neighbouring segments, not the memory


def test_fallback_counts_the_whole_text_as_chars_over_four(monkeypatch):
    monkeypatch.setattr(tokens, "_encoder", lambda: (None, tokens.FALLBACK_METHOD, "test"))
    assert estimate_tokens("a\nb\nc") == 2 and recount("a\nb", 1, "a\nXYZb") == 2
    with pytest.raises(tokens.ReferenceTokenizerUnavailable):
        estimate_tokens("x", strict=True)


def test_a_price_book_is_parsed_once_per_file_version_and_again_when_it_changes(tmp_path):
    assert load_price_book() is load_price_book(DEFAULT_PRICE_BOOK)
    path = tmp_path / "book.json"
    doc = json.loads(DEFAULT_PRICE_BOOK.read_text(encoding="utf-8"))
    path.write_text(json.dumps(doc), encoding="utf-8")
    first = load_price_book(path)
    assert load_price_book(path) is first and first.models == doc["models"]
    key = next(k for k, v in doc["models"].items() if "in" in v)
    doc["models"][key] = {**doc["models"][key], "in": 9.99}
    path.write_text(json.dumps(doc), encoding="utf-8")
    changed = load_price_book(path)
    assert changed is not first and changed.sha256 != first.sha256 and changed.models[key]["in"] == 9.99
