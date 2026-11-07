"""Reference token counts for planning and spend-cap bounds (never for billing).

The reference tokenizer is tiktoken ``o200k_base``. It is used only when its BPE
file is already in tiktoken's local cache with the published hash, so counting
never touches the network. Without it, ``estimate_tokens`` falls back to
ceil(characters / 4) and ``token_method()`` says so; callers that need the
reference tokenizer itself (budget fitting) pass ``strict=True`` to get an error
instead of the fallback.

Counting is exact and memoised by segment. o200k_base encodes each piece of its pre-tokenising regex on its own, and
a piece ends at the end of a run of newlines (CR / LF) whenever the next character is an ASCII letter or digit or one
of "[ - * # (": the only pattern branches that consume a newline (whitespace-then-newlines, and the newline-or-slash
tail of the punctuation branch) stop there, no branch looks behind, and the one lookahead (whitespace not followed by
non-whitespace) never reaches a newline because whitespace-then-newlines is tried before it. So the count of a text is
the sum of the counts of its segments cut at those points (tests/readers/test_tokens.py pins it against whole-text
encoding), and a segment seen before (a paragraph re-rendered into a larger memory, a prompt re-priced under another
scenario) is not encoded again. The chars/4 fallback counts the whole text, as before.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
import re
import tempfile

REFERENCE_ENCODING = "o200k_base"
_O200K_URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
_O200K_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
FALLBACK_METHOD = "chars/4"
# OpenAI chat-format convention: per-message framing and reply priming (an approximation
# for every provider; only used for estimates).
MESSAGE_OVERHEAD_TOKENS = 4
REPLY_PRIMING_TOKENS = 3


class ReferenceTokenizerUnavailable(RuntimeError):
    """strict counting was requested but o200k_base is not cached locally."""


def _cached_bpe_path() -> Path | None:
    """Mirror tiktoken.load.read_file_cached: the cache file, if it exists and verifies."""
    if "TIKTOKEN_CACHE_DIR" in os.environ:
        cache_dir = os.environ["TIKTOKEN_CACHE_DIR"]
    elif "DATA_GYM_CACHE_DIR" in os.environ:
        cache_dir = os.environ["DATA_GYM_CACHE_DIR"]
    else:
        cache_dir = os.path.join(tempfile.gettempdir(), "data-gym-cache")
    if not cache_dir:
        return None
    path = Path(cache_dir) / hashlib.sha1(_O200K_URL.encode()).hexdigest()
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != _O200K_SHA256:
        return None
    return path


@lru_cache(maxsize=1)
def _encoder():
    """Return (encoding or None, method, reason)."""
    try:
        import tiktoken
    except ImportError:
        return None, FALLBACK_METHOD, "tiktoken is not installed"
    if _cached_bpe_path() is None:
        return None, FALLBACK_METHOD, "o200k_base BPE file is not in the local tiktoken cache"
    return tiktoken.get_encoding(REFERENCE_ENCODING), f"tiktoken:{REFERENCE_ENCODING}", ""


def token_method() -> str:
    return _encoder()[1]


def estimate_tokens(text: str, *, strict: bool = False) -> int:
    if not isinstance(text, str):
        raise TypeError("estimate_tokens expects text")
    encoding, method, reason = _encoder()
    if encoding is None:
        if strict:
            raise ReferenceTokenizerUnavailable(reason)
        return math.ceil(len(text) / 4)
    if len(text) < LONG_TEXT_CHARS:
        return sum(_segment_tokens(segment) for segment in segments(text))
    return _long_text_tokens(text)


LONG_TEXT_CHARS = 2_048    # a text at least this long (a memory, a prompt) is also memoised whole: it is re-counted
                           # per reader, scenario and bound, and splitting it again is what would cost


@lru_cache(maxsize=1 << 12)
def _long_text_tokens(text: str) -> int:
    return sum(_segment_tokens(segment) for segment in segments(text))


SEGMENT_START = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789[-*#(")
SEGMENT_END = re.compile(r"[\r\n](?=[A-Za-z0-9\[\-*#(])")     # the newline before a piece boundary (docstring)


def segments(text: str) -> list[str]:
    """`text` cut after every newline followed by a SEGMENT_START character: a piece boundary of o200k_base."""
    out, start = [], 0
    for m in SEGMENT_END.finditer(text):
        out.append(text[start:m.end()])
        start = m.end()
    out.append(text[start:])
    return out


@lru_cache(maxsize=1 << 16)
def _segment_tokens(segment: str) -> int:
    return len(_encoder()[0].encode(segment, disallowed_special=()))


def _common_prefix(a: str, b: str) -> int:
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:                     # a[:lo] == b[:lo]; only the next window is compared
        mid = (lo + hi + 1) // 2
        if a[lo:mid] == b[lo:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _common_suffix(a: str, b: str, limit: int) -> int:
    lo, hi = 0, min(limit, len(a), len(b))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[len(a) - mid:len(a) - lo] == b[len(b) - mid:len(b) - lo]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def recount(before: str, before_tokens: int, after: str) -> int:
    """estimate_tokens(after), given before_tokens == estimate_tokens(before). When `after` is `before` with one string
    inserted (a packed memory grown by one unit), only the segments around the insertion are counted: the last piece
    boundary of `before` below the insertion point and the first above it stay boundaries of `after`, so the segments
    outside them are shared. Otherwise (or with a carriage return, or on the chars/4 fallback) the whole text is
    counted. Exact either way."""
    if _encoder()[0] is None or len(after) < len(before) or "\r" in after:
        return estimate_tokens(after)
    i = _common_prefix(before, after)
    n, grown = len(before), len(after) - len(before)
    if i + _common_suffix(before, after, n - i) != n:
        return estimate_tokens(after)
    lo, k = 0, before.rfind("\n", 0, i - 1) if i > 1 else -1
    while k >= 0:                      # a boundary p = k + 1 <= i - 1: its two characters precede the insertion
        if before[k + 1] in SEGMENT_START:
            lo = k + 1
            break
        k = before.rfind("\n", 0, k)
    hi, k = n, before.find("\n", i)
    while k >= 0 and k + 1 < n:        # a boundary p = k + 1 > i: its two characters follow the insertion
        if before[k + 1] in SEGMENT_START:
            hi = k + 1
            break
        k = before.find("\n", k + 1)
    return before_tokens - estimate_tokens(before[lo:hi]) + estimate_tokens(after[lo:hi + grown])


def estimate_message_tokens(messages, *, strict: bool = False) -> int:
    total = REPLY_PRIMING_TOKENS
    for message in messages:
        content = message["content"]
        if not isinstance(content, str):
            raise TypeError("estimate_message_tokens counts text messages only")
        total += MESSAGE_OVERHEAD_TOKENS + estimate_tokens(content, strict=strict)
    return total
