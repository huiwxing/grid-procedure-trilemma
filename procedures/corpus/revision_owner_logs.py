"""Owner records that date revisions and define revision pairs (S1): PJM revision-history bullets and MISO
revision-history table rows.

The PJM parser reads each revision-history bullet with its revision number, date and section reference, and classifies
it as a substantive or administrative change; it also parses the history of the newest PJM files (M-13 Rev 98).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from procedures.corpus.revision_dates import parse_date

# ---------------- PJM revision history ----------------
FOOTER = re.compile(r"Revision:\s*\d+,\s*Effective Date:\s*[\d/]+\s*PJM\s*©\s*\d{4}\s*\d+")
HEAD_REV = re.compile(r"^Revision\s+(\d+)\s*\(([^)]*)\)\s*:?\s*$")
HEAD_ADMIN = re.compile(r"^Administrative Change\s*(?:\(([^)]*)\))?\s*:?\s*$")
BULLET = {"•": 1, "": 1, "◦": 2, "o": 2, "▪": 3, "": 3}
JUNK_LINE = re.compile(r"^(Revision History|Current Revision|ReRevision Hist.*|CurrCurrent Re.*|PJM Manual \d+[A-Z]?\s*:.*)$")
CTX_HEADER = re.compile(r"^(Sections?\s+\d|Attachments?\s+[A-Z]\b|Exhibits?\s+\d|Introduction\b|\d+(\.\d+)+\s+[A-Z])")
SECREF = re.compile(
    r"(Sections?\s+\d+(?:\.\d+)*(?:\s*(?:,|and|&)\s*\d+(?:\.\d+)*)*"
    r"|Attachments?\s+[A-Z](?:\s*(?:,|and|&)\s*[A-Z])*\b"
    r"|Exhibits?\s+[A-Z]?\.?\d+(?:\.\d+)*"
    r"|Introduction(?: Section)?|Throughout|Revision History|Table\s+\d+(?:\.\d+)*|Cover to Cover"
    r"|^\d+(?:\.\d+)+)", re.I)
SECREF_FALLBACK = re.compile(r"\b\d+\.\d+(?:\.\d+)*\b")
VERB = re.compile(r"\b(updat|add|remov|clarif|revis|delet|chang|replac|correct|renam|moved|fix|noted|edit|align|includ)\w*",
                  re.I)
ADMIN = [
    r"\bperiodic review\b", r"\bcover to cover\b", r"\bownership\b", r"\bowner\b(?! ?\()", r"\bcontacts?\b", r"e-?mail",
    r"\bspell", r"misspell", r"\btypos?\b", r"replace\w* mentions", r"extra bullets?", r"^Revision History\b",
    r"grammat", r"\bformatting\b", r"\bformat(ted)?\b", r"renumber", r"\bnumbering\b",
    r"cross[- ]reference", r"\breferences?\b", r"\blinks?\b", r"\burls?\b", r"hyperlink", r"\brenam", r"footnote",
    r"phone number", r"clean ?up", r"\bduplicat", r"table of contents", r"\bimage\b", r"\bstyle\b", r"editorial",
    r"administrative", r"preliminary draft", r"\bweb ?(site|page)s?\b", r"\bcorrected (the )?name\b", r"\bmoved\b",
]
OVERRIDE = [
    r"\bfrom\s+[\d.,]+\s*\S*\s+to\s+[\d.,]+",
    r"\d[\d,.]*\s*(?:%|(?:percent|MW|MVA|MVAR|kV|Hz|minutes?|hours?|days?|months?|years?|°F)\b)",
    r"\bremov\w*\b.*\b(requirements?|schemes?|RAS|limits?|ratings?|criteria|reserves?)\b",
]
ADMIN_RE = [re.compile(p, re.I) for p in ADMIN]
OVERRIDE_RE = [re.compile(p, re.I) for p in OVERRIDE]


@dataclass
class PjmBullet:
    manual: str
    revision: str               # "96" or "84-admin"
    revision_number: int | None
    entry_kind: str             # revision | administrative_change
    date_raw: str
    level: int
    index: int
    text: str
    section_ref: str
    group_header: bool
    cls: str                    # substantive-candidate | administrative
    rule_hit: str
    source_line: int


def classify_bullet(text: str) -> tuple[str, str]:
    for r in OVERRIDE_RE:
        m = r.search(text)
        if m:
            return "substantive-candidate", f"override:{m.group(0)}"
    for r in ADMIN_RE:
        m = r.search(text)
        if m:
            return "administrative", f"admin:{m.group(0)}"
    return "substantive-candidate", "no-admin-keyword"


def _clean_lines(text: str) -> list[tuple[int, str]]:
    out = []
    for i, ln in enumerate(text.split("\n"), 1):
        s = FOOTER.sub("", ln).strip()
        if not s or JUNK_LINE.match(s) or re.match(r"^=+\s*PAGE", s):
            continue
        out.append((i, s))
    return out


def _parse_entries(cl: list[tuple[int, str]], start: int, stop_at_intro: bool = False) -> list[dict]:
    entries: list[dict] = []
    cur: dict | None = None
    last: dict | None = None
    ctx = ""
    for k in range(start, len(cl)):
        ln, s = cl[k]
        if stop_at_intro and re.match(r"^(Intr.*oduction|Introduction)", s):
            break
        mr, ma = HEAD_REV.match(s), HEAD_ADMIN.match(s)
        if mr or ma:
            cur = dict(kind="revision" if mr else "administrative_change",
                       revision=int(mr.group(1)) if mr else None,
                       date_raw=(mr.group(2) if mr else (ma.group(1) or "")), line=ln, bullets=[])
            entries.append(cur)
            last, ctx = None, ""
            continue
        if cur is None:
            continue
        ch = s[0]
        if ch in BULLET and (ch != "o" or s[1:2] == " "):
            last = dict(level=BULLET[ch], text=s[1:].strip(), line=ln, ctx=ctx)
            cur["bullets"].append(last)
            continue
        nxt = cl[k + 1][1] if k + 1 < len(cl) else ""
        if CTX_HEADER.match(s) and nxt[:1] in BULLET:
            ctx = s
            continue
        if last is None:
            last = dict(level=1, text=s, line=ln, ctx=ctx)
            cur["bullets"].append(last)
        else:
            last["text"] += " " + s
    return entries


def _section_ref(b: dict, parent: dict | None) -> str:
    m = SECREF.search(b["text"])
    if m:
        return m.group(0)
    if parent and parent.get("section_ref"):
        return parent["section_ref"]
    m = SECREF.search(b.get("ctx", "")) or SECREF_FALLBACK.search(b["text"])
    return m.group(0) if m else ""


def parse_pjm_history(text: str, manual: str) -> list[PjmBullet]:
    """All bullets of the newest file's Current Revision block + cumulative Revision History (newest first).
    Administrative changes amend the revision listed immediately below them."""
    cl = _clean_lines(text)
    heads = [k for k, (_, s) in enumerate(cl) if HEAD_REV.match(s)]
    current = _parse_entries(cl, heads[0], stop_at_intro=True)[:1]
    first_hist = next(k for k in heads[1:] if cl[k][0] > cl[heads[0]][0] + 5)
    entries = current + _parse_entries(cl, first_hist)
    for i, e in enumerate(entries):
        if e["kind"] == "administrative_change":
            e["revision"] = next((x["revision"] for x in entries[i + 1:] if x["kind"] == "revision"), None)
            e["label"] = f"{e['revision']}-admin"
        else:
            e["label"] = str(e["revision"])
    out: list[PjmBullet] = []
    for e in entries:
        parent = None
        for bi, b in enumerate(e["bullets"]):
            nxt = e["bullets"][bi + 1] if bi + 1 < len(e["bullets"]) else None
            group_header = bool(nxt and nxt["level"] > b["level"] and not VERB.search(b["text"]))
            if b["level"] == 1:
                parent = None
            b["section_ref"] = _section_ref(b, parent)
            text_ = re.sub(r"\s+", " ", b["text"])
            cls, why = classify_bullet(b["text"])
            out.append(PjmBullet(manual, e["label"], e["revision"], e["kind"], e["date_raw"], b["level"], bi, text_,
                                 b["section_ref"], group_header, cls, why, b["line"]))
            if b["level"] == 1:
                parent = b
    return out


# ---------------- MISO revision history table ----------------
MISO_DOCNUM = re.compile(r"BPM-\s?0?\d\d\s?-\s?r\d+")   # BPM-021 writes its own rows as 'BPM-21-r19'
MISO_DATE = re.compile(r"\b([A-Z][A-Za-z]{2,8}\s?-?\s?\d{1,2}\s?-\s?\d{2,4}|\d{2}-\d{2}-\d{4})\b")


@dataclass
class MisoEntry:
    doc_number: str
    rev: int
    description: str
    date_raw: str

    @property
    def date(self):
        return parse_date(self.date_raw)


def parse_miso_history(text: str) -> list[MisoEntry]:
    """Front-matter 'Revision History' table: one entry per BPM doc number (newest first)."""
    i = text.find("Revision History")
    if i < 0:
        return []
    j = text.find("Table of Contents", i)
    block = re.sub(r"\s+", " ", text[i:j if j > 0 else i + 20000])
    # running page headers repeat the current doc number next to its cover date; they are not table rows
    block = re.sub(MISO_DOCNUM.pattern + r"\s*Effective Date:\s*\S+", " ", block)
    marks = list(MISO_DOCNUM.finditer(block))
    out = []
    for a, b in zip(marks, marks[1:] + [None]):
        seg = block[a.end():b.start() if b else len(block)].strip()
        dates = MISO_DATE.findall(seg)
        num, rev = (int(x) for x in re.match(r"BPM-\s?(\d+)\s?-\s?r(\d+)", a.group(0)).groups())
        out.append(MisoEntry(f"BPM-{num:03d}-r{rev}", rev, seg[:600], dates[-1] if dates else ""))
    return out
