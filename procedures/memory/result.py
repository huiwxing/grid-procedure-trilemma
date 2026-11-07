"""The value every memory builder returns, and its content key."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

# a fixed string hashed into every memory cache key: the paper's run manifests and rows record keys made with it, so
# changing it would make every memory of those runs a different one
MEMORY_SCHEMA = "tg.memory/1"


@dataclass(frozen=True)
class MemoryResult:
    text: str
    representation: str
    budget_tokens: int | None
    revisions_used: list                # [(doc_id, revision)], ladder order
    provenance: list                    # [{doc_id, revision, path, char_start, char_end}], in rendered order
    build_tokens: int                   # one-time build cost attributable to this memory (0 for zero-LLM)
    cache_key: str                      # sha256 of the inputs (`cache_key`)
    notes: str
    token_method: str                   # procedures.readers.tokens.token_method() used for every count
    text_tokens: int                    # estimate_tokens(text)
    build: dict | None = None           # LLM-built representations: M5 / M7 the one-time build (mode, builder,
                                        # per-sub-block calls), M6 the per-question compression request
    query_tokens: int = 0               # per-question LLM cost attributable to this memory (M6's compression; else 0)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["revisions_used"] = [list(r) for r in self.revisions_used]
        return out


def cache_key(inputs: dict) -> str:
    """sha256 of the canonical JSON of `inputs` under the memory schema id."""
    payload = json.dumps({"schema": MEMORY_SCHEMA, **inputs}, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
