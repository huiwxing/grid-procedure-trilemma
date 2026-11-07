"""CAISO change notices: the text recipe of the manifest's `CAISO-CHANGE-NOTICE` rows.

A notice is an HTML page on the owner's site announcing revised operating procedures. Its released text is
"Notice <date>" followed by one line per announced procedure (number, title, version, effective date, the owner's
summary), as `parse_notice` reads them off the page's visible text (`html_to_text`). scripts/corpus/rebuild.py
replays this recipe (recorded in the manifest as `procedures.corpus.caiso.html_to_text+parse_notice`).
"""
from __future__ import annotations

import html
import re
from datetime import date, datetime

def html_to_text(raw: bytes) -> str:
    """Visible text of an HTML page, whitespace-collapsed."""
    s = raw.decode("utf-8", "replace")
    s = re.sub(r"<script.*?</script>|<style.*?</style>", " ", s, flags=re.S | re.I)
    s = html.unescape(re.sub(r"<[^>]+>", " ", s))
    return re.sub(r"\s+", " ", s).strip()


_NOTICE_DATE = re.compile(r"\bNotice\s+(?P<d>\d{2}/\d{2}/\d{4}|[A-Z][a-z]{2,8}\.? \d{1,2}, \d{4})")


_BODY = re.compile(r"has revised the operating procedures? ?(?:\(s\))? referenced below[.:]?(?P<body>.*?)"
                   r"(?:The California ISO(?:’|')s operating procedures are available|The ISO(?:’|')s operating procedures are available|CONTACT INFORMATION|Contact information)",
                   re.S | re.I)


_TITLE = r"(?:(?!\.\s)[^,]){3,160}?"


_ENTRY = re.compile(
    r"(?P<proc>\b\d{4}[A-Z]?)\s+(?P<title>" + _TITLE + r")\s*,\s*version\s+(?P<ver>[\d.]+)\s*,\s*(?:effective\s+)?"
    r"(?P<date>[A-Z][a-z]{2,8}\.? \d{1,2}, \d{4})\.\s*(?P<summary>.*?)(?=\s\d{4}[A-Z]?\s+" + _TITLE + r",\s*version\s|$)",
    re.S,
)


def long_date(value: str) -> date:
    """``March 5, 2026`` / ``Oct. 26, 2023`` / ``Sept. 1, 2023`` / ``03/05/2026`` -> date."""
    v = value.replace(".", "").replace("Sept ", "Sep ")
    for fmt in ("%m/%d/%Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(v, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unparsed date {value!r}")


def parse_notice(text: str) -> tuple[str, list[dict[str, str]]]:
    """(notice date, entries) from a notice page's visible text; entries carry the owner's change summary."""
    dm = _NOTICE_DATE.search(text)
    notice_date = long_date(dm["d"]).isoformat() if dm else ""
    bm = _BODY.search(text)
    entries = []
    if bm:
        for m in _ENTRY.finditer(bm["body"].strip()):
            entries.append({
                "procedure": m["proc"], "title": m["title"].strip(), "version": m["ver"],
                "effective": long_date(m["date"]).isoformat(), "summary": m["summary"].strip(),
            })
    return notice_date, entries
