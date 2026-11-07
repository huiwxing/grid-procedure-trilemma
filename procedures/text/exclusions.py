"""Revision-history regions of a procedure text: `history_regions(text, family)` gives the character spans of its
revision / version / document history. The summary memory (M5, procedures.memory.summary) sections and budgets a
revision's history apart from its body by this rule; the question builders kept their clauses out of the same regions
(the released items were built with it).

  PJM manuals         pages whose running footer names "Revision History" or "Current Revision", or in page-marker
                      text the pages from that heading on (pjm_history_regions)
  other documents     from the history heading to the next heading of the document's own structure (history_closes)
"""
from __future__ import annotations

import re

from procedures.text.sentences import PAGE_MARK

PJM_FOOTER = re.compile(r"\n([^\n]{3,120})\n\s*Revision: ?\d+, Effective Date:[^\n]*\s*$")
PJM_HISTORY_SECTIONS = ("revision history", "current revision")
PAGE_MARK_LINE = re.compile(r"^===\s*PAGE\s*\d+\s*===$")
# "Current Revision" sits in the front matter and is followed by the Introduction; "Revision History" is the last
# section of every on-disk M-03 / M-13 revision, and its own entries can open a page ("Attachment E: Updated ..."), so
# nothing but the Introduction heading ends a region
PJM_SECTION_START = re.compile(r"^Introduction$")
HISTORY_START = re.compile(
    r"^\s*(?:[IVX]+\.\s*)?(?:[\w-]+\s+){0,4}?(?:revision history|version history|document history"
    r"|document change history)\s*$", re.I)
# The end of a non-PJM history table. The heading opens the region; the region ends at the first later line that is a
# heading of the document's own structure, and runs to the end of the text only when none follows. Where the table
# stands decides which headings can follow it:
#   trailing history (a first-level body heading, BODY_START, precedes it): NERC standards print the Version History
#     after section E and before their attachments, normative tables (FAC-003 Table 2, PRC-005 Tables 1-1 to 5,
#     TPL-001 Table 1), guidelines and rationale; CAISO procedures before "N. Periodic Review Procedure" and the
#     "Appendix"; AESO rules before "Appendix 1" or a table (304.9 Table 1); ISO-NE before "IV. APPENDICES"
#   front history (no body heading before it): MISO / IESO / SPP / NYISO print it in the front matter, where it ends at
#     the contents or the first body section ("1. Introduction", NYISO "1. Overview"); its cells name the attachments
#     and tables they changed, one per line, even at a page top (NYISO "Attachment D", "Table A.4"; MISO "Attachment
#     BB"), so no attachment or table heading ends a front history
# The closing headings (history_closes):
#   HISTORY_END_ANY    anywhere, either position: contents, "1 Introduction" and NERC "A. Introduction" (VAR-002-4 and
#                      IRO-008-2 print a front Version History), supplemental material, guideline(s) and technical
#                      basis, application guidelines, violation severity levels, rationale headings
#   HISTORY_NUMBERED   after a page break or a blank line, either position: a numbered section heading ("5. Periodic
#                      Review Procedure", "1. Overview", "1. Contents")
#   trailing history only, after a page break or a blank line:
#     HISTORY_END_BREAK  appendix / appendices / (standard) attachments, a Roman-numbered "IV. APPENDICES", a NERC
#                        lettered section ("G. Rationale", "G. Reference")
#     HISTORY_ATTACH     an attachment / appendix / exhibit heading, optionally led by the standard's id ("PRC-025-2 -
#                        Attachment 1: Relay Settings", "MOD-025 Attachment 2")
#     HISTORY_TABLE      a numbered table heading ("FAC-003 - TABLE 2 - Minimum ...", "Table 1-4(a)",
#                        "UFLS Table 1: ...")
#     HISTORY_CAPS       an all-capitals heading of two or more words (BPA TOPIC: every procedure's "...
#                        Version History" is followed by the next procedure's title, "NERC TRANSMISSION AVAILABILITY
#                        DATA SYSTEM")
# History cells also wrap mid-table into heading-like lines ("Appendix 1 amended by combining ...", "Appendix 1: Minor
# edits for clarity ...", "Attachment 1 into the", MISO "11. Post - GIA Phase", "1. Annual Review Completed"), so a
# heading's title must read as one (heading_title): at most 120 characters, its first three significant words
# capitalised (HEADING_SMALL words aside), no sentence break, balanced parentheses, not ending in "," ";" "." or "-",
# not opening with a change verb (HISTORY_VERB); a numbered title also carries no digit and no past participle.
HISTORY_END_ANY = re.compile(
    r"^(?:(?:\d{1,2}\.?\s+)?(?:table of )?contents|(?:1\.?|A\.)\s+introduction|supplemental material"
    r"|guidelines? and technical basis|application guidelines|violation severity levels"
    r"|rationale(?:\s+box|\s+for\s+requirement\s+R\d+)?\s*:?)$", re.I)
HISTORY_END_BREAK = re.compile(
    r"^(?:(?:[IVX]{1,4}\.\s+)?appendi(?:x|ces)|(?:standard )?attachments"
    r"|[A-H]\.\s+(?:rationale|references?|regional variances|interpretations|associated documents"
    r"|supplemental material))$", re.I)
_DOC_ID = r"(?:[A-Z]{2,5}-\d{3}[\w.()-]*|Section\s+[A-Z])"
_SEP = r"\s*[—–:-]\s*"
_LABEL = r"(?:[A-Z]{1,2}|\d{1,2}[A-Z]?)(?:[-.]\d{1,2})*(?:\([a-z]\))?(?:\s*\((?i:cont|continued)\.?\))?"
HISTORY_ATTACH = re.compile(
    rf"^(?:{_DOC_ID}\s*(?:[—–-]\s*)?)?(?i:attachment|appendix|exhibit)\s+{_LABEL}(?P<title>(?:{_SEP}|\s+).*)?$")
HISTORY_TABLE = re.compile(
    rf"^(?:{_DOC_ID}{_SEP})?(?:[A-Z]{{2,6}}\s+)?(?i:table)\s+{_LABEL}(?P<title>(?:{_SEP}|\.?\s+).*)?$")
HISTORY_CAPS = re.compile(r"^(?=(?:.*?\b[A-Z]{3,}\b){2})[A-Z][A-Z0-9 ()&/,–-]{9,}$")
BODY_START = re.compile(r"^\s*(?:1\.?|I\.|A\.)\s+[A-Z][A-Za-z]{2,}")
TOC_LINE = re.compile(r"(?:\.\s?){6,}\s*\d*\s*$")
HISTORY_NUMBERED = re.compile(r"^\d{1,2}\.\s+(?P<title>[A-Za-z][^\d.]*)$")
HISTORY_VERB = re.compile(
    r"^(?:added|changed|removed|updated|revised|corrected|deleted|replaced|modified|moved|adopted|approved|retired"
    r"|inserted|clarified|renumbered|transferred|incorporated|reformatted|minor|editorial|errata|initial)\b", re.I)
PAST_PARTICIPLE = re.compile(r"[a-z]ed\b", re.I)
HEADING_SMALL = frozenset("a an and as at by for from in of on or per the this to vs with".split())
HEADING_WORD = re.compile(r"[A-Za-z0-9][^\s—–:/&()+\"“”‘’'-]*")


def heading_title(title: str | None) -> bool:
    """The words after a heading's number or label read as a heading title, not as a wrapped history cell (see the
    HISTORY_END_ANY comment); no title at all ("Table 1-1", "Attachment A") is a heading."""
    t = (title or "").strip(" —–:-")
    if not t:
        return True
    if t[-1] in ",;.-" or HISTORY_VERB.match(t) or re.search(r"\.\s", t) or t.count("(") != t.count(")"):
        return False
    words = [w for w in HEADING_WORD.findall(t) if w.lower() not in HEADING_SMALL]
    return all(w[0].isupper() or w[0].isdigit() for w in words[:3])


def history_closes(line: str, trailing: bool, after_break: bool) -> bool:
    """True when `line` is a heading that ends a (non-PJM) revision-history table; `trailing`: the table follows the
    document body; `after_break`: the line follows a page break or a blank line."""
    s = " ".join(line.split())
    if not s or len(s) > 120:
        return False
    if HISTORY_END_ANY.match(s):
        return True
    if not after_break:
        return False
    m = HISTORY_NUMBERED.match(s)
    if m:
        return heading_title(m.group("title")) and not PAST_PARTICIPLE.search(m.group("title"))
    if not trailing:
        return False
    if HISTORY_END_BREAK.match(s):
        return True
    if HISTORY_CAPS.match(s):
        return True
    m = HISTORY_ATTACH.match(s) or HISTORY_TABLE.match(s)
    return bool(m) and heading_title(m.group("title"))



def _history_table_regions(text: str) -> list[tuple[int, int]]:
    """From each history heading to the first closing heading (history_closes) or the end of the text. A history is
    trailing when a first-level body heading (BODY_START, not a contents line) precedes its heading. A page starts after
    a page-marker line or a form feed (the text after the last form feed of a line is that page's first line)."""
    regions = []
    pos, open_at, trailing, body_seen, after_break = 0, None, False, False, False
    for line in text.split("\n"):
        stripped = line.replace("\f", " ")
        cut = line.rfind("\f") + 1
        if open_at is None:
            if HISTORY_START.match(stripped):
                open_at, trailing = pos, body_seen
            elif BODY_START.match(stripped) and not TOC_LINE.search(stripped):
                body_seen = True
        elif history_closes(line[cut:], trailing, after_break or cut > 0) and not HISTORY_START.match(stripped):
            regions.append((open_at, pos + cut))
            open_at = None
        after_break = bool(PAGE_MARK.match(line)) or not line[cut:].strip()
        pos += len(line) + 1
    if open_at is not None:
        regions.append((open_at, len(text)))
    return regions


def pjm_history_regions(text: str) -> list[tuple[int, int]]:
    """Revision-history pages of a PJM manual. Form-feed text (an older text extraction): pages whose running footer
    names a revision-history section. Page-marker text (the
    unified corpus recipe, procedures.corpus.pdf_text: "=== PAGE n ===", running footers stripped, so the footer rule
    finds nothing): a page whose first line is the section heading "Current Revision" / "Revision History" opens a
    region that runs until a page opening with "Introduction" or the end of the text."""
    if "\f" in text:
        regions, pos = [], 0
        for page in text.split("\f"):
            m = PJM_FOOTER.search(page)
            if m and m.group(1).strip().lower() in PJM_HISTORY_SECTIONS:
                regions.append((pos, pos + len(page)))
            pos += len(page) + 1
        return regions
    regions, open_at, pos, first_line = [], None, 0, False
    for line in text.split("\n"):
        stripped = line.strip()
        if PAGE_MARK_LINE.match(stripped):
            first_line = True
        elif stripped and first_line:
            first_line = False
            if open_at is None and stripped.lower() in PJM_HISTORY_SECTIONS:
                open_at = pos
            elif open_at is not None and PJM_SECTION_START.match(stripped):
                regions.append((open_at, pos))
                open_at = None
        pos += len(line) + 1
    if open_at is not None:
        regions.append((open_at, len(text)))
    return regions


def history_regions(text: str, family: str) -> list[tuple[int, int]]:
    if family.startswith("PJM"):
        return pjm_history_regions(text)
    return _history_table_regions(text)

