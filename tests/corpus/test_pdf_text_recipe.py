"""Offline regression tests for the single PDF -> text recipe shared by every owner (procedures.corpus.pdf_text),
the unified redline detection, and the decoding of an Internet Archive capture's payload (procedures.corpus.wayback).
Synthetic PDFs come from tests/corpus/synthetic_pdf.py."""
from __future__ import annotations

import gzip

import pypdf
from synthetic_pdf import CHAR_W, artifact, make_pdf, rule, text_op

from procedures.corpus.pdf_text import PAGE_MARKER, RECIPE_VERSION, Line, detect_redline_spans, is_red, pdf_to_pages, pdf_to_text, strip_running_lines
from procedures.corpus.wayback import decoded_payload


# ---------------------------------------------------------------- running lines: headings that must survive

def test_caiso_3100a_contingency_headings_retained(tmp_path):
    """CAISO 3100A: the section heading is the first body line of consecutive pages ("3./5./7. Pre-Contingency
    State"), and "Single Contingency)" wraps under each Post-Contingency heading; neither is running furniture
    (the old digit-masked recurrence rule deleted them). The header block here is untagged, the footer tagged."""
    pages = []
    for p in range(1, 6):
        header = (text_op(380, 745, "Procedure No. 3100A") + text_op(380, 730, "Version No. 1.4")
                  + text_op(380, 715, "Effective Date 3/05/26") + text_op(110, 698, "Acceptable Thermal Performance"))
        footer = artifact(text_op(276, 37, f"Page {p} of 5"))
        if p == 1:
            body = (text_op(77, 640.8, "1. Transmission Line Facility Rating/Thermal Example")
                    + text_op(77, 400, "2. Pre- and Post-Contingency State"))
        elif p == 5:
            body = text_op(77, 640.8, "9. References")
        else:
            dy = {2: 0.0, 3: 16.8, 4: 7.2}.get(p, 0.0)
            body = (text_op(77, 640.8, f"{2 * p - 1}. Pre-Contingency State (i.e., actual Real-time operations)")
                    + text_op(77, 374.4 + dy, f"{2 * p}. Post-Contingency State (i.e., where the system lands")
                    + text_op(96, 361.6 + dy, "Single Contingency)"))
        pages.append(header + body + footer)
    text = pdf_to_text(make_pdf(tmp_path / "3100a.pdf", pages))
    for n in (3, 5, 7):
        assert f"{n}. Pre-Contingency State (i.e., actual Real-time operations)" in text
    assert text.count("Single Contingency)") == 3
    assert "2. Pre- and Post-Contingency State" in text
    assert "Procedure No. 3100A" not in text and "Page 2 of 5" not in text


def test_pjm_actions_heading_at_top_of_consecutive_pages_retained(tmp_path):
    """M-13 Rev 92 pp. 72-73: "PJM Actions" opens the body of consecutive pages; the old slot-repeat rule
    dropped it. It must survive, while the running title and footer go."""
    pages = []
    for i in range(1, 11):
        body = ["PJM Actions"] if i in (4, 5, 6) else []
        body += [f"Unique body sentence {i}.", f"Second body sentence {i}.", f"Third body sentence {i}."]
        data = b"".join(text_op(72, 688 - 14 * k, line) for k, line in enumerate(body))
        data += (text_op(300, 742, "PJM Manual 13: Emergency Operations") + text_op(300, 730, "Section 2: Capacity")
                 + text_op(72, 45, f"Revision: 98, Effective Date: 06/24/2026 PJM (c) 2026 {i}"))
        pages.append(data)
    out = pdf_to_pages(make_pdf(tmp_path / "m13.pdf", pages))
    assert [page.split("\n")[0] for page in out[3:6]] == ["PJM Actions"] * 3
    assert all("PJM Manual 13" not in page and "Revision: 98" not in page for page in out)


def test_pjm_pseudo_bold_overprint_removed(tmp_path):
    """PJM paints headings twice 0.28 pt apart (faux bold); the text keeps one copy."""
    heading = text_op(72, 700, "Section 1: Overview") + text_op(72.28, 700, "Section 1: Overview")
    (page,) = pdf_to_pages(make_pdf(tmp_path / "bold.pdf", [heading + text_op(72, 680, "Body text.")]))
    assert page == "Section 1: Overview\nBody text."


def test_toc_listed_body_heading_kept_when_it_also_runs_as_a_title():
    """M-03: "Revision History" is the running section title on most pages and, once, the body heading that
    opens the section. The TOC lists it, so that body occurrence survives even with no body text right below."""
    def page(p: int, body: list[tuple[float, str]]) -> list[Line]:
        lines = [Line(p, 1, [("plain", "PJM Manual 03: Transmission Operations")], 742.3, 10.0),
                 Line(p, 2, [("plain", "Revision History" if p > 1 else "Table of Contents")], 729.8, 10.0)]
        return lines + [Line(p, 3 + k, [("plain", t)], y, 11.0) for k, (y, t) in enumerate(body)]
    pages = [page(1, [(688.0, "Revision History .......... 2"), (670.0, "Attachment A ........ 9")]),
             page(2, [(688.0, "Revision History"), (600.0, "Revision 71 (05/20/2026):")])]
    pages += [page(p, [(688.0, f"Revision {70 - p} (01/01/2020):")]) for p in range(3, 7)]
    kept = strip_running_lines(pages, [(0.0, 792.0)] * 6)
    assert [ln.text for ln in kept[1]] == ["Revision History", "Revision 71 (05/20/2026):"]
    assert all(ln.text != "PJM Manual 03: Transmission Operations" for p in kept for ln in p)


# ---------------------------------------------------------------- one redline detection for every family


def test_blank_lines_are_single_paragraph_breaks(tmp_path):
    """Whitespace-only lines (Word's empty paragraphs) mark paragraph breaks: kept once, never at page edges."""
    lines = [" ", "3.1 Outage Coordination", " ", "New requests must be submitted.", " ", " ", "Second paragraph.", " "]
    (page,) = pdf_to_pages(make_pdf(tmp_path / "p.pdf", [b"".join(text_op(72, 700 - 14 * k, t)
                                                                      for k, t in enumerate(lines))]))
    assert page == "3.1 Outage Coordination\n\nNew requests must be submitted.\n\nSecond paragraph."


def test_recipe_identity_and_page_marker(tmp_path):
    assert PAGE_MARKER == "=== PAGE {n} ==="
    assert RECIPE_VERSION == f"procedures.corpus.pdf_text+pypdf-{pypdf.__version__}"
    text = pdf_to_text(make_pdf(tmp_path / "t.pdf", [text_op(72, 700, "One"), text_op(72, 700, "Two")]))
    assert text == "=== PAGE 1 ===\nOne\n=== PAGE 2 ===\nTwo\n"


# ---------------------------------------------------------------- Internet Archive captures

def test_gzip_encoded_capture_payload_is_decoded():
    raw = gzip.compress(b"%PDF-1.7 body")
    assert decoded_payload(raw) == (b"%PDF-1.7 body", "payload archived gzip-encoded, decoded")
    assert decoded_payload(b"%PDF") == (b"%PDF", "")
