"""procedures.items.template: the frozen headword list of the question templates."""
import hashlib

from procedures.items.template import HEADWORDS, HEADWORDS_SHA256


def test_the_headword_list_is_frozen():
    assert len(HEADWORDS) == 14 == len(set(HEADWORDS))
    assert hashlib.sha256("\n".join(HEADWORDS).encode("utf-8")).hexdigest() == HEADWORDS_SHA256
