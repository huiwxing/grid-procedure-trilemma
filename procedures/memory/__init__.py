"""Memory representations for the reader harness.

B0 closed book (no procedure text), M0 full text, M2 retrieval (BM25 paragraph chunks), M4 constraint sentences
(regex), M5 summary (section-wise LLM summary), M6 compression (query-focused LLM compression), M7 records (typed
constraint records). The LLM-built ones (M5, M6, M7) call models through procedures.readers.executor.run_cells only.
Entry point: `build_memory`.
"""
from procedures.memory.build import REPRESENTATIONS, build_memory
from procedures.memory.summary import BuilderLedger
from procedures.memory.result import MemoryResult
from procedures.corpus.revisions import RevisionIndex, load_index

__all__ = ["REPRESENTATIONS", "BuilderLedger", "MemoryResult", "RevisionIndex", "build_memory", "load_index"]
