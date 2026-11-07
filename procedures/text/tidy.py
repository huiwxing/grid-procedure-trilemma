"""The one text normalisation for verbatim phrases shown to a reader.

Both segmenters (`sentences.segment` for the owner-written branch pairs S2 and the unrevised control sentences S3,
`procedures.corpus.revision_text.build_view` for the revision pairs S1 and the as-of date x branch items L2) join the
lines of a layout block with one space and never drop a character, so a sentence's text maps back to the LF corpus by
offsets and every verbatim check compares lower-cased alphanumeric streams (`patterns.alnum`), which spacing and
hyphens never change. PDF layout residue is removed only where a phrase is rendered into a question, an option or a
gold (`tidy`), by the same rules for every stratum's item builder:

  line-break hyphen   "Real- Time" -> "Real-Time", "one- hundred" -> "one-hundred", "PRC- 023-6" -> "PRC-023-6":
                      the corpus prints no soft hyphens (a survey of all 218 manifest texts: every line-end hyphen
                      is a compound's own hyphen), so a hyphen followed by the joining space is closed up; a
                      suspended hyphen before a coordinator stays ("pre- and post-contingency")
  footnote marker     "each calendar quarter10" -> "each calendar quarter": 1-2 digits glued to the end of a
                      lower-case word of >= 5 letters are a footnote reference (shorter words are left alone:
                      "than30", "rev20", "real32" are values or names)
  whitespace          runs of whitespace -> one space, trimmed (`squash`)
"""
from __future__ import annotations

import re

LINE_HYPHEN = re.compile(r"(?<=[A-Za-z0-9])- (?!(?:and|or|nor|to|through|and/or)\b|&)(?=[A-Za-z0-9])")
FOOTNOTE_DIGITS = re.compile(r"(?<![\w./:%-])([a-z]{5,})\d{1,2}(?![\w./%-])")


def squash(s: str) -> str:
    """Collapse whitespace runs (PDF extraction double spaces and line joins) into one space."""
    return " ".join(s.split())


def tidy(s: str) -> str:
    """Reader-facing form of a verbatim phrase: whitespace squashed, line-break hyphens closed up, footnote markers
    dropped (module docstring). Idempotent; changes no letter of the phrase."""
    return FOOTNOTE_DIGITS.sub(r"\1", LINE_HYPHEN.sub("-", squash(s)))
