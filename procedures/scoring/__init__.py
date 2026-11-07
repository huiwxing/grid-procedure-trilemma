"""Frozen answer scorer: raw model text + an item's gold -> verdict, error class, blind-arbitration queue, summaries.

"Frozen" means the scorer's files are hashed into every run's identity (procedures.matrix.run_matrix.SCORER_FILES),
so each run records exactly which scorer scored it. The scoring rules are numbered N1-N11; docstrings, error messages
and tests/scoring/test_scorer_rules.py cite them by number:
N1 answer extraction (parse.py); N2 abstention and N3 "not in effect" read after terminal punctuation; N4 numbers;
N5 units; N6 categorical labels and N6' closed sets (option letters); N7 yes/no golds and their per-stratum quota;
N8 times of day and dates (normalize.py, score.py); N9 interval golds, split into one numeric gold per bound; N10 a
call that produced no output scores unparseable; N11 the blind arbitration of answers the scorer could not match.
"""
from procedures.scoring.normalize import DocIndex
from procedures.scoring.parse import TRUNCATION_REASONS, extract_answer
from procedures.scoring.score import ERROR_CLASSES, GTYPES, OPTION_LETTERS, STATUSES, arbitration_queue, check_lettered_display, classify_error, gold_spec, pair_consistency, render_lettered_options, score_answer, score_rows, summarize

__all__ = [
    "DocIndex", "TRUNCATION_REASONS", "extract_answer", "ERROR_CLASSES", "GTYPES", "OPTION_LETTERS",
    "STATUSES", "arbitration_queue", "check_lettered_display", "classify_error", "gold_spec",
    "pair_consistency", "render_lettered_options", "score_answer", "score_rows", "summarize",
]
