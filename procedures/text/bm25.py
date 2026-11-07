"""Okapi BM25 ranking (zero LLM): `BM25` scores pre-tokenised documents against a query, with the parameters
`RetrievalParams` (k1, b). It ranks the units of the bounded memories (procedures.memory.chunking).

BM25 is a small pure-python Okapi implementation (no external dependency) with the Lucene non-negative
idf ``ln(1 + (N - df + .5) / (df + .5))``.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalParams:
    k1: float = 1.5
    b: float = 0.75


class BM25:
    """Okapi BM25 over pre-tokenised documents (Lucene non-negative idf)."""

    def __init__(self, docs: list[list[str]], k1: float, b: float):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avgdl = (sum(self.len) / len(docs)) if docs else 0.0
        df: Counter[str] = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1.0 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, dl in zip(self.tf, self.len):
            s = 0.0
            norm = self.k1 * (1.0 - self.b + self.b * dl / self.avgdl) if self.avgdl else self.k1
            for q in query:
                f = tf.get(q)
                if f:
                    s += self.idf[q] * f * (self.k1 + 1.0) / (f + norm)
            out.append(s)
        return out
