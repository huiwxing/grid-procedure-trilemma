"""Offline tests for offset-preserving segmentation (a page footer glued to the next page by a form feed is dropped
and the sentence it split is joined again)."""
from procedures.text.patterns import alnum
from procedures.text.sentences import lf_normalise, segment

FOOTER = "PJM Manual 03: Transmission Operations\nSection 2: Thermal Operating Guidelines\n" \
         "Revision: 69, Effective Date: 11/20/2025 PJM © 2025 {n}"


def _pjm_doc():
    return (f"Intro sentence one is here.\nFor any given dynamic rating system failure, the following\n{FOOTER.format(n=34)}"
            f"\fdispatcher actions are utilized.\nThe TO shall act within 5 minutes.\n{FOOTER.format(n=35)}"
            f"\fAnother page starts here.\n{FOOTER.format(n=36)}\f")


def test_footer_glued_by_form_feed_is_dropped_and_sentence_joined():
    text = _pjm_doc()
    sents = segment(text)
    texts = [s.text for s in sents]
    assert "For any given dynamic rating system failure, the following dispatcher actions are utilized." in texts
    assert not any("PJM Manual 03" in t or "Revision: 69" in t for t in texts)


def test_offsets_point_back_into_the_lf_text():
    text = _pjm_doc()
    for s in segment(text):
        a, b = s.raw_span(0, len(s.text))
        assert (a, b) == (s.start, s.end)
        if "\n" not in text[s.start:s.end]:
            assert text[s.start:s.end] == s.text
    s = next(x for x in segment(text) if x.text.startswith("The TO shall"))
    k = s.text.index("5 minutes")
    a, b = s.raw_span(k, k + len("5 minutes"))
    assert text[a:b] == "5 minutes"


def test_short_repeated_content_lines_survive_away_from_page_breaks():
    body = "".join(f"Rule {i}: if reducing generation or\nshedding load.\n" for i in range(4))
    sents = segment("Header line.\n" + body)
    assert sum(1 for s in sents if s.text.endswith("shedding load.")) == 4
    assert alnum("".join(s.text for s in sents)).count("sheddingload") == 4


def test_label_before_single_capital_is_not_an_initial():
    texts = [s.text for s in segment("The IR is withdrawn by the end of IC Decision Point I. If the IR is withdrawn "
                                     "later, the deposit is at risk.\n")]
    assert texts == ["The IR is withdrawn by the end of IC Decision Point I.",
                     "If the IR is withdrawn later, the deposit is at risk."]


def test_crlf_and_cr_are_normalised_to_lf():
    assert lf_normalise("a\r\nb\rc") == "a\nb\nc"
