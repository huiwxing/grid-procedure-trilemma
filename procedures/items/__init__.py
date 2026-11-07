"""The question items: their strata, the scorer's view of an item, and the templates the questions were written with.

The items are data (data/items/), composed before any paid call; nothing in an item depends on any reader's answers.
Strata: S1 = revision pairs (an old and a new value of one clause; a positive control), S2 = branch pairs written by
the procedure's owner, S3 = unrevised control sentences, L3 = as-of date x branch x exception items whose scenario
names no clause (the paper's task). An L3 cell is one item of an L3 unit (one branch on one as-of date).

The main run's item file (data/items/questions.jsonl) keeps each source item verbatim and adds `analysis_stratum`,
`source_batch` and `m0_paths`:
  * primary = every load_bearing cell of the third batch of admitted L3 items (data/items/pool_branch_questions_3.jsonl;
    `source_batch` branch_questions_3);
  * secondary = the other cells of that batch (load_bearing false);
  * controls = the template controls (S2 / S3 controls, `source_batch` controls, and the S1 positive control,
    `source_batch` revision_pair_controls);
  * earlier_batches = the admitted cells of the L3 batches 1 and 2 (data/items/questions_with_earlier_cells.jsonl),
    less a few bases excluded for L3 before the main run; a unit a manual ruling keeps out of the primary set stays
    here with load_bearing false on every cell, and the other cells of these batches carry no load_bearing field.
`m0_paths` are the revisions a full-text reader (M0) sees: the evidence files of every item of the same unit or pair
(the version pair for an as-of unit, the current version otherwise) plus the text of each item's own revision, because
an S1 side whose clause is absent cites the other revision as evidence and the full-text reader still gets both.

* `frozen`: the scorer's item view, the pair-consistency checks of L3 units, the JSONL form of rows.
* `template`: the question templates (documentation of how the questions were generated).
* `publish`: how a path is written into a record.
"""
