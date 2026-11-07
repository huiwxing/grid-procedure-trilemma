# Data: layout, how records refer to files, what is not released, rerunning, labels

This document is the companion of the README for anyone who reads the released records. It describes the data by
purpose, how the records refer to one another, what is left out and why, what each script reproduces, and what the
labels in the data mean in the paper's words.

## 1. Layout by purpose

**Questions** (`data/items/`)

- `questions.jsonl`: the 244 questions the runs read (220 branch questions and 24 controls). The paper analyses 158
  of them, the 79 pairs listed in `population.item_ids` of the analysis spec and selected by
  `procedures.analysis.spec.in_population`. Each question carries its text, the options, the gold answer, the clause it
  is scored against (`condition_clause`: document, revision, character offsets and the quoted sentence or two), its
  pair (`unit_id`), its side of the condition (`cell_kind`: `rule`, `exception`, `parallel` or `complement`), the
  corpus texts a full-text reader sees (`m0_paths`), the authors' rulings on it (section 5) and the authors' working
  notes (`notes`).
- `questions_with_earlier_cells.jsonl`: the same questions plus 82 cells of the two earlier batches of branch
  questions; the file the spec was made from.
- `pool_branch_questions_{1,2,3}.jsonl` and `pool_branch_pairs.jsonl`: the admitted pools the questions were drawn
  from, read only by the report's supply check (whether the pool holds enough branch pairs and units).

**Corpus pointers** (`data/corpus/`)

- `manifest.csv`: one row per procedure revision: publisher URL or Internet Archive capture, fetch date, effective
  dates, the sha256 of the fetched bytes and of the extracted text, the conversion recipe, the publisher's terms
  (`terms_flag`) and exclusion flags. The `effective_date_text` column quotes the effective-date clause of each
  revision verbatim (up to about a hundred words); it is the only column that carries procedure text. `text_path`
  names the text's place under `data/corpus/<family>/` once it is rebuilt.
- `version_dates_external.csv`: revision dates taken from publishers' revision histories.

**Readers and prices** (`data/panel/`, `data/prices/`): the reader panel (`panel.json`, with the pilot's profile file
`profiles.json` it was pinned from) and the price snapshot the runs were priced on (`prices.json`, with the router
snapshots it was built from).

**Runs** (`data/runs/<run>/`): four runs, named by their directory. `main` is the main run (eight open-weight readers;
no document, full text for the server readers, retrieval / rule sentences / summary / records at 2k, 4k and 16k,
compression at 4k; summaries and records built by DeepSeek-V4.1-Flash). `records_built_by_gptoss20b` and
`records_built_by_qwen32b` read records built by gpt-oss-20b and Qwen3-32B with the five single-GPU readers.
`cloud_comparator` is the cloud model (Claude Sonnet 5: no document, full text, retrieval at 4k; one read per
question). A run directory holds:

| file | role |
|---|---|
| `manifest.json` | the run's plan: question file, panel, price book, corpus manifest, prompt templates and scorer (each with its sha256 at run time), conditions, readers, repeats, the cache key of every memory |
| `invocations.jsonl` | one line per invocation of the run driver (plans, executions, resumes, scorings) |
| `estimate.json` | the priced plan before execution |
| `build__*.jsonl.gz`, `query__*.jsonl.gz`, `read__*.jsonl.gz` | the call ledgers of the builders, the compressors and the readers (section 3 says what they keep) |
| `memory_requests.json` | which calls built each LLM-built memory (the report charges build tokens through it) |
| `sanity_report.json` | the offline mechanism check run before any result was read: every memory really reached its reader |
| `rows_before_arbitration.jsonl.gz`, `summary_before_arbitration.json` | the scored cells as the run driver scored them |
| `arbitration_queue.jsonl` | the answers the scorer could not match, sent to blind arbitration |
| `rescore.json`, `rows_final.jsonl.gz`, `summary_final.json` | the record of the re-scoring that applied the arbitration decisions, and the final scored cells the report reads |

The cloud comparator's run needed no arbitration, so it has only `rows_final` and `summary_final`.

**Analysis** (`data/analysis/`): the spec `spec.json` (population, tiers, comparisons and sensitivity checks, fixed
before the main run after a pilot), the blind arbitration decisions per run (`arbitration/decisions_<run>.json`), the
energy band file read by the report (`energy_bands.json`, all bands unset: the paper makes no energy statement), and
the report the paper was written from (`reports/paper_report.json`).

**The paper's files** (`paper/`): `numbers.json` (the prespecified comparisons and verdicts, selected from the report
by `extract_numbers.py`), `exploratory.json` (the post hoc readings, computed from the rows by `exploratory.py`),
`printed_values.json` (every other printed value, recomputed by `printed_values.py`) and the two figures.

## 2. How the records refer to files

- Every path in a record is relative to the repository root (`data/items/questions.jsonl`,
  `data/corpus/pjm/m13_rev98.txt`); a run's own files are named by their file name inside the run directory, and the
  run id in every row, summary and decisions file is the run directory's name.
- The rows and ledgers are stored gzip-compressed. Code and records name the plain file (`rows_final.jsonl`) and
  `procedures/stored.py` reads the `.gz` when the plain file is absent.
- The run manifests and the re-scoring records carry the sha256 of their inputs (the item file, the panel, the price
  book, the corpus manifest, the prompt templates, the scorer's files, the ledgers) as provenance. For release the
  item files (corpus paths, two field names, the wording of the authors' notes), the corpus manifest (provenance
  notes), the panel and the price book (notes) were edited and their recorded hashes updated to the released files; the scorer's files were renamed and their
  comments rewritten, so the `scorer_sha256` entries name the original files, and the re-scoring in section 4 shows
  the scorer is the same; the ledgers were slimmed (section 3), so the ledger hashes in the re-scoring records are
  those of the raw ledgers. The code checks what the analysis depends on, never a byte identity: every item a run's
  rows name must be in the item file, a run's readers must keep the profile fingerprint the run used, decisions must
  name their run and its queue; a hash that differs from a recorded one is reported as a warning and the run goes on.
- The memory cache keys in the manifests bind the revision texts (by their sha256), the prompts and the builders'
  parameters: when the corpus texts are rebuilt and a memory is planned again, an equal key means the same request.

## 3. What is not released, and why

- **The procedure texts.** Publishers allow pointers only. The manifest records where each revision came from and the
  hash of its text; `scripts/corpus/rebuild.py` fetches and converts them and reports, per text, whether it is the one
  the runs read. The question files quote the clause each question is scored against (a sentence or two), and the
  manifest quotes the effective-date clauses.
- **The prompts and the readers' quotations.** The released call ledgers keep every accounting field (token usage
  as the provider reported it and as normalised, price, status, finish reason, timing and attempts, the hashes that
  tie a call to its row) and the generation parameters of each request, but not the prompt messages (they carry the
  procedure text), the provider's raw payload beyond its id, model, stop reason and usage, or the builders' and
  compressors' outputs (they restate the procedure), for which only `text_chars` and `text_sha256` remain. A reader's
  answer is released with the structure the scorer reads: in the answer object every member other than `value` and
  `unit` (the quoted `evidence_span` above all) is set to the empty string, text the reader wrote outside the object is
  replaced by `[text withheld]`, an object the reader did not close keeps its leading `value` and `unit` members, and
  the reader's reasoning text, where it is not itself the answer object, is replaced by `[reasoning text withheld]`
  (whitespace-only reasoning is kept as it was). The withheld strings are described by `evidence_span_chars` /
  `evidence_span_sha256`, `text_chars` / `text_sha256` and `reasoning_chars` / `reasoning_sha256`. The scorer reads
  only the answer's value, unit and structure, so re-scoring the released ledgers writes the released rows
  (section 4); `scripts/slim_ledgers.py` is the program that derives the released ledgers from the raw ones.
- **The tooling that found the documents and built the questions.** Its outputs are the manifest and the question
  files; the question templates remain in `procedures/items/template.py` as documentation.
- **Provider invoices.** The billing reconciliation (`billing_reconciliation` in the report's metadata) stays
  `pending`, so dollar figures are not results; token counts are.
- **Superseded files of the runs** named in `invocations.jsonl` (estimates of interrupted invocations, earlier live
  scorings) are not included.

## 4. Rerunning

- **The report.** `scripts/report.py` over the released rows, summaries and ledgers (the README command) writes a
  report with the estimates, intervals and verdicts of `reports/paper_report.json`. The two exploratory memory-level
  sensitivities need the builders' outputs, which only the raw build ledgers hold, and are reported as not computed.
- **The paper's files.** `paper/extract_numbers.py` on the report reproduces `paper/numbers.json`;
  `paper/exploratory.py` reproduces every point estimate of `paper/exploratory.json`, with bootstrap interval
  endpoints that are Monte Carlo draws (on the interval the paper prints, 0.18 [0.03, 0.31], a rerun moves an endpoint
  by at most 0.01); `paper/printed_values.py` recomputes every other printed value from the rows and ledgers and
  checks the printed strings; `paper/fig_readers.py` and `paper/fig_errors.py` draw the figures.
- **The corpus.** `scripts/corpus/rebuild.py --items data/items/questions.jsonl` rebuilds the 183 texts of the 71
  documents the questions read, from the manifest's pointers, and says for each whether it is the text the runs read.
- **The scorer.** `scripts/rescore.py` over each run's released ledgers, with the rebuilt texts and the arbitration
  decisions where they exist, writes the rows of `rows_final.jsonl.gz` for all four runs.
- **The spec.** `scripts/make_spec.py` on `questions_with_earlier_cells.jsonl` and the panel writes `spec.json`.

## 5. Labels

| in the paper | in the data |
|---|---|
| no document / full text / retrieval / rule sentences / summary / compression / records | conditions `B0@full`, `M0@full`, `M2@<budget>`, `M4@<budget>`, `M5@<budget>/<builder>`, `M6@4000[/<compressor>]`, `M7@<budget>/<builder>` |
| context budget 2k / 4k / 16k | `2000`, `4000`, `16000` |
| cloud comparator | `A2_sonnet5_nothink` (tier `A`); its run is the reference arm `P0-A` |
| server readers | `B1_dsv41flash_hf_deepinfra`, `B2_qwen3_235b_2507_hf_deepinfra`, `tierB_llama33_70b_hf_ovhcloud` (tier `B`) |
| constrained panel (one 24 or 48 GB GPU) | `C1_llama31_8b_hf_deepinfra`, `C2_qwen3_8b_hf_nscale_nothink`, `C3_gptoss20b_hf_deepinfra_low` (24 GB) and `tierC2_qwen3_32b_hf_nscale_nothink` (48 GB); tier `C_load_bearing` (access tiers `C1` and `C2`). The prefix of a profile id is its panel slot, not a GPU class |
| gpt-oss-120b (80 GB, reported apart) | `tierC3_gptoss120b_hf_deepinfra_low` (access tier `C3`) |
| open-weight reference | `REF-B`: the DeepSeek reader at the better of `M0@full` and `M2@4000` (retrieval won) |
| builder and compressor of the server pipeline | `B1_dsv41flash_direct_nothink` (DeepSeek's own API) |
| binding / navigable / undecided | `binds` / `escapable` / `indeterminate` in the report's `outcomes` |
| comparisons of Table II | `dGap-M5-*`, `dGap-M7-*` (summary and records vs retrieval), `CBUILT-*` (larger builder), `H_M7` (records vs rule sentences), `H_S` (summaries and wrong-branch answers), `dGap-M6-*` (compression), `gap_*` (gaps to the references), `interaction-*` (panel difference) |
| wrong branch / no branch applies / abstained | error classes `E-cond` (and `E-exc`, the rule chosen on an exception question), `E-other` with the answer `none of these` or `not_in_effect`, `E-abs` |
| pair, branch | `unit_id`; `cell_kind` `rule` / `exception` / `parallel` / `complement` |
| question strata | `type` `L3-*` (the paper's questions: a scenario on one side of a condition, with a revision date), `S2-*` (branch pairs without scenarios), `S1-*` (revision pairs), `S3-*` (controls); `analysis_stratum` `primary` / `controls` / `earlier_batches` and `source_batch` (`branch_questions_3`, `controls`, `revision_pair_controls`, ...) say where a cell of the main run's item file came from |
| the authors' rulings on a question | `load_bearing` (the cell counts for the main analysis), `domain` (`ops`, `technical`, `market`, ...), `rule2_spirit` (`abstracted`: the scenario changes the description level of the clause; `transparent`: it paraphrases it), `k6_exception_literal` / `k6_exception_confidence` (whether the pair's exception is literal in the text; `k6` is this ruling's label in the data, also in the spec's sensitivity `drop:k6_medium`), `same_rule_confidence`, `pin_confidence`, `normative_confidence`, `cue_relation` / `cue_pool` (how the exception sentence relates to the cue pool the questions were drawn from), `template_family`, `owner_restates_standard` |
| report status | `supply` with keys `L3` and `S2`: the supply check (enough units and branch pairs in the pool); `billing_reconciliation`: whether usage x price was reconciled with the provider invoices; `N11`: the scorer rule that sends unmatched answers to arbitration (the scorer's rules `N1`-`N11` are stated in `procedures/scoring/`); `k1_*`: the single-read rule and the 4k slice re-read three times |
| runs | the directory names `main`, `records_built_by_gptoss20b`, `records_built_by_qwen32b` and `cloud_comparator`, also the `run_id` of every row |
| schema ids | `procedures.<module>.<name>.v1` name the file formats and are checked when files are read |
