"""Quantity and modal-verb detectors shared by the revision text views (procedures.corpus.revision_text).

One definition of "a quantity" and "a normative sentence" for every stage of the item construction. Pure string
work: no I/O, no LLM.
"""
from __future__ import annotations

import re

WORDNUM = (r"(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|twenty-four"
           r"|thirty|forty|forty-five|sixty|ninety)")
UNIT = (r"(?:MWh|MW|MVAR|MVAr|Mvar|MVA|kV|Hz|%|p\.u\.|pu|ms|°\s?[CF]"
        r"|(?i:percent|per[- ]unit|milliseconds?|seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|degrees?"
        r"|weeks?|months?|years?|cycles?))")
NUM = r"(?<![\w.])[±+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|(?<![\w.])\.\d+"
QTY = re.compile(
    rf"(?:\(?(?:{NUM})\)?|(?i:{WORDNUM})\s+\(\d+\))\s*-?\s*(?:(?i:business|calendar|consecutive|clock|operating)\s+)?"
    rf"({UNIT})(?![A-Za-z])")
QTY_WORD = re.compile(
    rf"\b(?:within|no later than|not later than|at least|no less than|not less than|no more than|not more than"
    rf"|up to|in excess of|greater than|less than|exceed(?:s|ing)?|every|each|per)\s+{WORDNUM}\s+"
    rf"(?:(?:business|calendar|consecutive|clock)\s+)?(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b",
    re.I)
MODAL_ANY = re.compile(r"\b(shall|must|required|prohibited|permitted)\b", re.I)
def has_qty(s: str) -> bool:
    return bool(QTY.search(s) or QTY_WORD.search(s))
