# grid-procedure-trilemma

Code and data for the paper *Navigating the Sustainable AI Trilemma in LLM Assistants for Grid Operating Procedures*
(Hui Wu, Xiaoyang Wang and Zhong Fan; IEEE PES International Meeting 2027, Beijing).

Grid operating procedures are public, but a question about live system conditions may have to stay on the
operator's premises. The paper asks whether a larger model can prepare a procedure once so that a smaller local
reader applies its conditions correctly, and what each way of spending LLM computation costs in accuracy and tokens.
The test is a set of **79 paired questions** from North American operating procedures: the two questions of a pair
sit on either side of one condition in one rule, so they require different actions. Nine readers (a cloud model,
three open-weight server models and five single-GPU models) answer them from the original passages, from reusable
summaries and records built once per revision, and from per-question compression, all under the same context budget.

## Start here

| what you are looking for | where it is |
|---|---|
| the questions, with gold answers and the clause each is scored against | `data/items/questions.jsonl` (244 questions; the paper's 158 are the `population.item_ids` of `data/analysis/spec.json`, two per pair) |
| the prompts the readers saw | `procedures/readers/prompts.py` |
| the scorer | `procedures/scoring/` (deterministic; the answers it could not match went to blind arbitration, recorded in `data/analysis/arbitration/`) |
| the knowledge paths | `procedures/memory/`: `retrieval.py` (BM25 passages), `constraints.py` (rule sentences), `summary.py` and `records.py` (built once per revision), `compressor.py` (per question), `fulltext.py` |
| the analysis the paper reports | `data/analysis/spec.json` (the comparisons, fixed before the main run) and `data/analysis/reports/paper_report.json` (the report every number traces to) |
| the four runs behind the paper | `data/runs/` (scored rows, call ledgers with token usage and prices, arbitration records) |
| the paper's numbers and figures | `paper/` |
| what each file, field and label means | [`docs/DATA.md`](docs/DATA.md) |

## Layout

| path | contents |
|---|---|
| `procedures/` | the pipeline as a Python package: `text` (sentences, BM25), `corpus` (manifest, revision ladders, PDF-to-text recipe), `items` (question templates, the scorer's view of a question, pair checks), `memory` (the knowledge paths), `readers` (API client, reader panel, prompts, call ledgers, pricing, spend caps), `scoring` (scorer, re-scoring with arbitration), `matrix` (the run driver and its sanity checks), `analysis` (observations, bootstrap, the report) |
| `scripts/` | command-line entry points: `report.py`, `rescore.py`, `make_spec.py`, `run_matrix.py`, `approval.py`, `smoke_plan.py`, `sanity_check.py`, `memory_sanity.py`, `slim_ledgers.py`, `corpus/rebuild.py` |
| `paper/` | the scripts that regenerate the paper's tables, figures and printed values from the released records, and the files the paper was written from |
| `data/items/` | the question files and the admitted pools they were drawn from |
| `data/corpus/` | the corpus manifest: one row per procedure revision with its source URL, fetch date, effective dates, conversion recipe and the publisher's terms (the texts themselves are not included; see below) |
| `data/panel/`, `data/prices/` | the reader panel and the price snapshot the runs were priced on |
| `data/analysis/` | the analysis spec, the arbitration decisions and the report |
| `data/runs/` | the four runs: `main` (eight open-weight readers, every path and budget), `records_built_by_gptoss20b` and `records_built_by_qwen32b` (records built by the smaller builders), `cloud_comparator` (the cloud model) |
| `tests/` | the test suite (offline: API calls are replayed from fixtures, corpus tests use synthetic documents) |

The rows and call ledgers of a run are stored gzip-compressed (`<name>.jsonl.gz`); the code and the records name the
plain file and read either form.

## Reproducing the paper's numbers

Python 3.12 with `pip install numpy pypdf matplotlib tiktoken` (pypdf 6.x). The analysis needs only numpy; pypdf is
for rebuilding the corpus texts, matplotlib for the figures, and tiktoken (with its `o200k_base` encoding cached
locally) for exact token counts when memories are rebuilt. Nothing below calls a model API, and everything is written
under `out/`.

**The report.** Every number in the paper traces to `data/analysis/reports/paper_report.json`. To rebuild it from the
released rows and ledgers (about two minutes, no network):

```
python scripts/report.py --run-dir data/runs/main --run-dir data/runs/records_built_by_gptoss20b --run-dir data/runs/records_built_by_qwen32b --run-dir data/runs/cloud_comparator --rows data/runs/main/rows_final.jsonl --summary data/runs/main/summary_final.json --rows data/runs/records_built_by_gptoss20b/rows_final.jsonl --summary data/runs/records_built_by_gptoss20b/summary_final.json --rows data/runs/records_built_by_qwen32b/rows_final.jsonl --summary data/runs/records_built_by_qwen32b/summary_final.json --spec data/analysis/spec.json --label paper
```

This writes `out/reports/paper_report.json` (and a markdown rendering of its tables) with the estimates, intervals and
verdicts of the released report.

**The paper's numbers and figures.** The scripts in `paper/` read the released records by default and write to
`out/paper/`:

```
python paper/extract_numbers.py     # the prespecified comparisons and verdicts (Table II, Sec. III) -> numbers.json
python paper/printed_values.py      # every other printed value, recomputed from the rows and ledgers and checked
python paper/fig_readers.py         # Fig. 2
python paper/fig_errors.py          # Fig. 3
python paper/exploratory.py         # the post hoc readings of Sec. III-A and III-C and the token row of Table I
python paper/clause_coverage.py     # the clause counts of Sec. III-C (needs the rebuilt corpus texts)
```

`paper/numbers.json`, `paper/exploratory.json` and `paper/printed_values.json` are the files the paper was written
from. Point estimates reproduce exactly; the bootstrap interval endpoints of `exploratory.py` are Monte Carlo draws
and may differ in the second decimal.

## Running the pipeline yourself

1. **Corpus.** The procedure texts are not redistributed (their publishers allow pointers only).
   `python scripts/corpus/rebuild.py --items data/items/questions.jsonl` fetches every revision the questions read
   from the URL (or the Internet Archive capture) the manifest records, converts it with the recorded recipe, writes
   it under `data/corpus/<family>/` and reports for each text whether it is the text the paper's runs read (183 texts
   of 71 documents).
2. **Memories.** `procedures/memory/build.py` builds any knowledge path for a question, a budget and, for the
   LLM-built ones, a builder; `python scripts/memory_sanity.py` checks the builders offline on the item file.
3. **Reader runs.** `python scripts/run_matrix.py --label main --grid main --repeats 3 --k1-slice` plans the paper's
   main run (cells, planned tokens, a priced estimate) under `out/runs/` and sends nothing; the grids
   `cbuilt_gptoss20b`, `cbuilt_qwen32b` and `cloud_reference` are the other three runs. `--execute --approval <id>`
   makes the paid calls: it needs the provider keys named in the panel file, a spend approval
   (`python scripts/approval.py new ...`) and, first, the connectivity check (`python scripts/smoke_plan.py`). Every
   call is appended to the run's ledgers and a killed run resumes from them; `python scripts/sanity_check.py
   --run-dir <run>` checks a finished run's prompts against the memories it planned, before any result is read.
4. **Scoring and analysis.** `python scripts/rescore.py --run-dir data/runs/main --decisions
   data/analysis/arbitration/decisions_main.json` re-scores a run from its ledgers (with the rebuilt texts and, where
   they exist, the arbitration decisions) into `out/rescore/<run>/`; on the released runs it writes the released final
   rows. `python scripts/make_spec.py --verify-items data/items/questions.jsonl` writes the analysis spec from the
   question file and the panel (`out/spec/spec.json`, equal to the released spec), and `scripts/report.py` executes
   it.

## Data labels

The paper's words map to the labels in the data as follows; [`docs/DATA.md`](docs/DATA.md) has the full glossary.
Knowledge paths: no document `B0`, full text `M0`, retrieval `M2`, rule sentences `M4`, summary `M5`, compression `M6`,
records `M7`, at budgets `2000`, `4000`, `16000` or `full`. Readers: the cloud comparator is `A2_sonnet5_nothink`; the
server readers are `B1_dsv41flash_hf_deepinfra`, `B2_qwen3_235b_2507_hf_deepinfra` and `tierB_llama33_70b_hf_ovhcloud`;
the constrained panel (one 24 or 48 GB GPU) is `C1_llama31_8b_hf_deepinfra`, `C2_qwen3_8b_hf_nscale_nothink`,
`C3_gptoss20b_hf_deepinfra_low` and `tierC2_qwen3_32b_hf_nscale_nothink`; `tierC3_gptoss120b_hf_deepinfra_low` is the
80 GB reader reported apart. The prefix of a profile id is its slot in the panel, not a GPU class. The open-weight
reference (`REF-B`) is the DeepSeek reader with retrieval at 4k; the builder and compressor of the server pipeline is
`B1_dsv41flash_direct_nothink`. The constrained panel is the tier `C_load_bearing` in the spec and the report; the
paper's verdicts binding / navigable / undecided are `binds` / `escapable` / `indeterminate` in the data.

## Tests

`python -m pytest tests -q` (offline, about two minutes).

## License and citation

Code and data are released under the MIT License (see `LICENSE`); the procedure excerpts quoted in the question files
remain subject to their publishers' terms. Please cite:

```
H. Wu, X. Wang, and Z. Fan, "Navigating the Sustainable AI Trilemma in LLM Assistants for Grid Operating Procedures,"
in Proc. IEEE PES International Meeting, Beijing, China, Jan. 2027.
```
