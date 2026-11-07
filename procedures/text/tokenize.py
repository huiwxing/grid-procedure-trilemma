"""Text normalisation and the BM25 token bags of the bounded memories (procedures.memory.chunking and
procedures.memory.records).

Everything here is deterministic and locale-independent: NFKC, typographic quotes and dashes
folded to ASCII, casefold.
"""
from __future__ import annotations

import re
import unicodedata

_FOLD_TABLE = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u00a0": " ", "\u2009": " ", "\u202f": " ",
})

STOPWORDS = frozenset(
    """a an the of to in for on at by with from as into onto upon via per is are be been being was
    were am and or nor if then than that this these those which who whom whose what when where why
    how it its itself their theirs they them there here such any all each every both either neither
    may must shall will would should can could do does did has have had having also only same so
    very just about through during again further once
    other another some own i we you he she his her our your my me us him""".split()
)


def fold(text: str) -> str:
    """NFKC + ASCII quotes/dashes + casefold."""
    return unicodedata.normalize("NFKC", text).translate(_FOLD_TABLE).casefold()


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*")


def stem(token: str) -> str:
    """Light plural stemmer (enough to align ``outages``/``outage``, ``limits``/``limit``)."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("sses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")) and not token[0].isdigit():
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Content tokens: folded, stopwords removed, lightly stemmed."""
    return [stem(t) for t in _TOKEN_RE.findall(fold(text)) if t not in STOPWORDS]
