"""Question templates, filled ONLY with phrases copied verbatim from the clause.

The item questions were generated with these templates; nothing in the released pipeline calls the builders (it uses
`revision_label` and the option constants NONE_OPTION, NOT_IN_EFFECT and SIDE_TAIL). They are kept as the executable
statement of how the questions were written.

Form names are the stored item types. S1 = revision pairs (an old and a new value of one clause), S2 = branch pairs
written by the procedure's owner, S3 = unrevised control sentences, L2 = as-of date x branch; `-value` asks for the
value, `-side` asks which of the two branch outcomes applies, and the deadline form asks for a time limit, lead time
or recurrence interval.

  S1/S3-value  Under {document} as in effect on {as_of_date}, what is the {quantity phrase} for {scope phrase}?
               If no such requirement applies on that date, answer not_in_effect.
  S2-value     Under {document} (Revision {rev}, effective {date}), {condition head}, what is the
               {outcome/quantity phrase}?
  S2-side      Under {document} (Revision {rev}, effective {date}), {condition head}, which of the following
               applies? (A) {outcome} (B) {outcome} (C) none of these.
  deadline     {S2 or S3 header}, [{condition head},] {subject phrase} {modal} {verb phrase}
               within | no later than | at least | at least once every  how many {unit}[ {verbatim tail}]?
               ("at least" only with a verbatim tail starting "prior to"; S3 adds the not_in_effect sentence)
  L2           Under {document} as in effect on {as_of_date}, {condition head}, {S2-value | S2-side | deadline
               question tail} If no such requirement applies on that date, answer not_in_effect.   (`l2_question`:
               the admitted S2 question with its header swapped; `l2_side_question`: the same with the options of a
               "changed" branch; `l2_clause_question`: the S1 clause path)
  S1/S3-value with a condition: the value's own fronted or trailing condition goes after the header as a
               condition head, exactly as in the deadline form.

Condition head (`condition_slot`): "if {condition phrase}" or "when {condition phrase}" for a subordinate clause, the
source's own subordinator deciding (if / in case / in the event (that) / provided (that) -> if; when / whenever /
where / once -> when); a condition the source states with a prepositional head ("For X, ...", "In the event of X,
...", "Under X", "After X", "Prior to X") is that verbatim phrase with its head lower-cased -- the "for {scope}" head;
the generator never writes "when for ..." / "when in the event of ...".
A verbatim scope qualifier (the enclosing label that tells two same-stem questions apart, chosen by the item
builder) goes before the condition head: "..., for {qualifier}, if {condition}, ...".

{document}, {rev} and dates come from the document registry (dates ISO yyyy-mm-dd, revisions "Revision {rev}");
{unit} is the canonical plural unit name of the verbatim quantity ("thirty (30) Calendar Days" -> "calendar days");
every other slot is a contiguous substring of the source clause. When no rule fills a slot the item is left
unfilled (None) and flagged, instead of inventing wording.

"what is the {phrase}" slot rules, in priority order:
  copula_np      "<NP> is|are|shall be|will be|must be|should be|becomes|remains|equals|is set at <ANSWER>"  (strong)
  premod_np      "<ANSWER> <NP>" where the answer pre-modifies a noun: "fifty percent (50%) refund of ..."   (strong)
  comparative_np "<NP> shall not exceed|shall be at least|... <ANSWER>"                                        (weak)
  of_np          "<NP> of <ANSWER>": "a minimum of 30 minutes" (weak unless the NP has >= 2 content words)
  headword_pp    "the {headword} {verbatim PP}": one word of the frozen HEADWORDS list (chosen by the answer's
                 dimension) heads the verbatim prepositional phrase right after the answer, "115% of the Facility
                 Rating" -> "percentage of the Facility Rating" (strong when the PP has >= 2 content words).
                 Used when no rule above fires, or when their phrase fails the number rule below.
Number rule: a quantity/scope/subject/verb/tail phrase may not contain a number (a quantity or a bare number token;
digits inside letter+digit identifiers such as "M2", "R4", "TOP-004-2" are names and pass), the gold or the other
side's value (surface, "value unit" or the spelled-out number); such a phrase falls back to headword_pp, else the form
is dropped.
Condition phrases may contain numbers (they define the branch) but never the gold or the other side's value.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import re
from dataclasses import dataclass

from procedures.scoring.score import render_lettered_options

from procedures.text.patterns import STOPWORDS, Quantity, balanced, contains_phrase, find_quantities
from procedures.text.tidy import tidy

# Frozen headword list of the question templates: exactly these 14 words, this order; HEADWORDS_SHA256 is the sha256
# of "\n".join(HEADWORDS) (UTF-8), pinned by tests/items/test_template.py.
HEADWORDS = ("limit", "time", "period", "duration", "rating", "threshold", "percentage", "amount", "value",
             "requirement", "notification", "deadline", "status", "action")
HEADWORDS_SHA256 = "6e1aa4e0535f7f5831be4a2d50c41ef58ef2aa8dcd68ea90555cf8e9c9374de9"

DETERMINERS = frozenset("the a an this that these those such its their his her our your each any all every".split())
HARD_STOP = frozenset(
    "shall will must may should can could would is are was were be been being has have had do does did if when where "
    "whenever that which who whom whose and or nor but then to by as than because while unless otherwise until not no "
    "also only therefore thus however so".split())
PREPOSITIONS = frozenset("of for in on at per via from into upon with within during under after before about over "
                         "between among through across against toward towards".split())
UNIT_WORDS = frozenset("mw mva mvar kv hz percent minutes minute mins min hours hour hrs hr seconds second secs sec days "
                       "day weeks week months month years year cycles cycle degrees degree business calendar".split())
COPULA = re.compile(
    r"\b(?:is|are|will be|shall be|must be|should be|would be|may be|becomes?|remains?|equals?|(?:is|are) equal to"
    r"|(?:is|are|shall be|will be) set (?:at|to))\s*$", re.I)
COMPARATIVE = re.compile(
    r"\b(?:(?:shall|must|will|may|should|does|do) not exceed|not to exceed|(?:shall|must|will|should) be (?:at least"
    r"|no (?:less|more|greater|later|earlier) than|not (?:less|more|greater) than|limited to|within)"
    r"|(?:is|are) (?:at least|no (?:less|more) than|limited to|capped at))\s*$", re.I)
OF_TAIL = re.compile(r"\bof\s*$", re.I)
TOKEN = re.compile(r"\S+")
MAX_NP_TOKENS = 10
MAX_OF_NP_TOKENS = 4
# words that follow a quantity without being the noun it modifies ("20 minutes prior to", "10 seconds following")
NOT_NOUN = frozenset("prior after afte before following beyond whether above below lower higher greater less more "
                     "earlier later ahead out thereafter ago whichever respectively apart long old early late total "
                     "or and in to from each per".split())
# "in excess of", "in advance of", "on behalf of": the noun before "of" is part of a complex preposition
COMPLEX_PREP_HEADS = frozenset("excess advance case event lieu terms respect addition instead spite result basis behalf "
                               "means regardless account part".split())
SECTION_NO = re.compile(r"^(?:[A-Z]\.)?\d+(?:\.\d+)*\.?$")
# a subject noun phrase may contain "to", "and", "of" ... but never an auxiliary, modal, subordinator or relative pronoun
SCOPE_STOP = frozenset("shall will must may should can could would is are was were be been being has have had do does "
                       "did if when where whenever that which who whom whose unless because while then".split())
SCOPE_BAD_START = frozenset("and or but nor of in on at to for with by from as".split())


@dataclass(frozen=True)
class Slot:
    text: str        # verbatim phrase (for headword_pp: the prepositional phrase after the answer)
    start: int       # offset within the clause string the rule was applied to
    end: int
    rule: str
    strong: bool
    headword: str | None = None   # HEADWORDS entry heading the verbatim phrase (headword_pp only)

    @property
    def phrase(self) -> str:
        """What follows "what is the" in the question."""
        return tidy(f"{self.headword} {self.text}" if self.headword else self.text)


def _bare(tok: str) -> str:
    return re.sub(r"^\W+|\W+$", "", tok).lower()


SUBJECT_STOPS = HARD_STOP - {"to"}   # "recovery time to the pre-fault value is": stopping at "to" names the wrong thing
OF_STOPS = HARD_STOP | PREPOSITIONS


def _np_backward(before: str, stops: frozenset = SUBJECT_STOPS, max_tokens: int = MAX_NP_TOKENS
                 ) -> tuple[int, int] | None:
    """Noun phrase ending at the end of `before` (offsets into `before`), walking back over at most `max_tokens`
    tokens and stopping at a stop word or clause punctuation. Leading prepositions/determiners dropped."""
    toks = list(TOKEN.finditer(before))
    picked: list[re.Match] = []
    for m in reversed(toks):
        raw = m.group(0)
        if raw[-1] in ",;:" or _bare(raw) in stops or len(picked) >= max_tokens:
            break
        picked.insert(0, m)
    while picked and (_bare(picked[0].group(0)) in PREPOSITIONS | DETERMINERS or not _bare(picked[0].group(0))):
        picked.pop(0)
    if not picked or _bare(picked[-1].group(0)) in PREPOSITIONS | DETERMINERS:
        return None
    # the phrase must start where a noun phrase can start: after punctuation, a stop word or a determiner
    return picked[0].start(), picked[-1].end()


def _verbish(raw: str) -> bool:
    """Lower-case -ing / -ly words are participles or adverbs ("45 days identifying ..."); capitalised ones are
    usually part of a defined term ("75% Spinning Reserves")."""
    return raw[:1].islower() and _bare(raw).endswith(("ing", "ly"))


def _np_forward(after: str) -> tuple[int, int] | None:
    """Noun phrase starting at the beginning of `after` (the answer pre-modifies it)."""
    toks = list(TOKEN.finditer(after))
    if not toks:
        return None
    first = _bare(toks[0].group(0))
    if (not toks[0].group(0)[0].isalpha() or first in HARD_STOP | PREPOSITIONS | DETERMINERS | UNIT_WORDS | NOT_NOUN
            or (_verbish(toks[0].group(0))) or toks[0].start() > 1):
        return None
    picked: list[re.Match] = []
    for m in toks:
        raw = m.group(0)
        b = _bare(raw)
        if (b in HARD_STOP or b in NOT_NOUN or _verbish(raw) or len(picked) >= MAX_NP_TOKENS
                or (b in PREPOSITIONS and b != "of")):
            break
        picked.append(m)
        if raw[-1] in ",;:.":
            break
    while picked and _bare(picked[-1].group(0)) in PREPOSITIONS | DETERMINERS:
        picked.pop()
    if not picked:
        return None
    return picked[0].start(), picked[-1].end()


def _content_count(phrase: str) -> int:
    return sum(1 for t in re.findall(r"[A-Za-z][\w'-]*", phrase) if t.lower() not in STOPWORDS and len(t) > 2)


def _clean(phrase: str) -> str:
    return phrase.strip().rstrip(",;:.").strip()



def find_slot(clause: str, ans_start: int, ans_end: int) -> Slot | None:
    """Verbatim noun phrase in `clause` naming the value at clause[ans_start:ans_end]."""
    before, after = clause[:ans_start], clause[ans_end:]
    for rx, rule, strong in ((COPULA, "copula_np", True), (COMPARATIVE, "comparative_np", False)):
        m = rx.search(before.rstrip())
        if m:
            span = _np_backward(before[:m.start()])
            if span and _content_count(before[span[0]:span[1]]) >= 1 and balanced(before[span[0]:span[1]]):
                a, b = span
                text = _clean(clause[a:b])
                return Slot(text, a, a + len(text), rule, strong)
    span = _np_forward(after)
    if span and _content_count(after[span[0]:span[1]]) >= 1 and balanced(after[span[0]:span[1]]):
        a, b = ans_end + span[0], ans_end + span[1]
        text = _clean(clause[a:b])
        return Slot(text, a, a + len(text), "premod_np", True)
    m = OF_TAIL.search(before.rstrip())
    if m:
        span = _np_backward(before[:m.start()], OF_STOPS, MAX_OF_NP_TOKENS)
        if span and _content_count(before[span[0]:span[1]]) >= 1 and balanced(before[span[0]:span[1]]) and                 _bare(before[span[0]:span[1]].split()[-1]) not in COMPLEX_PREP_HEADS:
            a, b = span
            text = _clean(clause[a:b])
            return Slot(text, a, a + len(text), "of_np", _content_count(text) >= 2)
    return None


SUBJECT_MODAL = re.compile(r"\b(?:shall|must|will|should|may|is required|are required|is responsible|are responsible)\b",
                           re.I)


def find_scope(main_clause: str, avoid: tuple[int, int] | None = None) -> tuple[int, int] | None:
    """Subject noun phrase of a rule clause: text before the first modal, <= 10 tokens, no clause punctuation or
    subordinators, not a bare pronoun, not overlapping `avoid` (the answer/slot span)."""
    m = SUBJECT_MODAL.search(main_clause)
    if not m:
        return None
    head = main_clause[:m.start()]
    a = len(head) - len(head.lstrip())
    b = len(head.rstrip())
    subj = main_clause[a:b]
    toks = subj.split()
    if not (1 <= len(toks) <= 10) or re.search(r"[,;:()\[\]]", subj):
        return None
    if any(_bare(t) in SCOPE_STOP for t in toks):
        return None
    if _bare(toks[0]) in {"it", "they", "this", "these", "those", "he", "she", "which", "that", "there"}:
        return None
    if _bare(toks[0]) in SCOPE_BAD_START:   # a fragment ("and recommendations on ...") or glued heading ("of SISA ...")
        return None
    if not re.match(r"[A-Za-z]", subj):
        return None
    # a section heading glued in front of the subject ("L.6 Data Retention Period The MISO")
    if any(SECTION_NO.match(t) for t in toks) or (
            len(toks) > 6 and sum(t[:1].isupper() for t in toks) >= 0.6 * len(toks)):
        return None
    if avoid and a < avoid[1] and avoid[0] < b:
        return None
    return a, b


def iso_date(value: str) -> str:
    """Template rule: every date in a question is ISO yyyy-mm-dd; anything else is a registry error."""
    try:
        return dt.date.fromisoformat(value).isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"question date must be ISO yyyy-mm-dd, got {value!r}") from exc


REV_PREFIX = re.compile(r"^(?:revision|rev\.?|r)\s*(?=\d)", re.I)


def revision_label(revision: str) -> str:
    """Template rule: revisions are written "Revision {rev}" ("r33" -> "Revision 33", "Rev. 71" -> "Revision 71")."""
    rev = REV_PREFIX.sub("", (revision or "").strip())
    if not rev:
        raise ValueError("empty revision label")
    return f"Revision {rev}"


def s2_header(document: str, revision: str, effective_date: str) -> str:
    return f"Under {document} ({revision_label(revision)}, effective {iso_date(effective_date)})"


def as_of_header(document: str, as_of_date: str) -> str:
    return f"Under {document} as in effect on {iso_date(as_of_date)}"


NOT_IN_EFFECT = "If no such requirement applies on that date, answer not_in_effect."
# first word of a condition the source states with a prepositional head (the scope and temporal heads the S2
# branch-pair builder recognises)
CONDITION_HEADS = frozenset("during under for in prior before after on beginning starting upon".split())
IF_SUBORDINATORS = frozenset(("if", "only if", "in case", "in the event", "in the event that", "provided",
                              "provided that"))
WHEN_SUBORDINATORS = frozenset(("when", "whenever", "where", "once"))


def condition_slot(condition: str, head: str | None) -> str:
    """The template's condition head (module docstring). `head` = the source's subordinator ("if", "when",
    "where", "in the event that" ...) for a subordinate-clause condition (the phrase then excludes it), None for a
    condition phrase that carries its own prepositional head ("For updates ...", "In the event of restudy")."""
    c = tidy(condition)
    if head is None:
        if c.split(" ", 1)[0].lower() not in CONDITION_HEADS:
            raise ValueError(f"condition without a subordinator must start with a prepositional head: {c!r}")
        return c[0].lower() + c[1:]
    h = " ".join(head.lower().split())
    if h in IF_SUBORDINATORS:
        return f"if {c}"
    if h in WHEN_SUBORDINATORS:
        return f"when {c}"
    raise ValueError(f"unknown condition subordinator {head!r}")


def head_word(condition: str | None, head: str | None) -> str | None:
    """The word a question's condition head starts with ("if", "when", "for", "in", "after" ...); None without one."""
    return condition_slot(condition, head).split(" ", 1)[0] if condition else None


def section_tag(section: str | None) -> str:
    """Duplicate-question disambiguator, appended to the condition phrase (else the scope / subject phrase)."""
    return f" (Section {section})" if section else ""


def lead(condition: str | None, head: str | None, section: str | None = None, qualifier: str | None = None) -> str:
    """Text between the header and the question tail: "[{qualifier}, ][{condition head}{section tag}, ]" (the section
    tag rides on the condition here; without a condition the caller puts it after the scope / subject)."""
    parts = [tidy(qualifier)] if qualifier else []
    if condition:
        parts.append(condition_slot(condition, head) + section_tag(section))
    return "".join(f"{p}, " for p in parts)


def s2_question(document: str, revision: str, effective_date: str, condition: str, slot: str,
                section: str | None = None, head: str | None = "when", qualifier: str | None = None) -> str:
    return (f"{s2_header(document, revision, effective_date)}, {lead(condition, head, section, qualifier)}"
            f"what is the {tidy(slot)}?")


def s3_question(document: str, as_of_date: str, slot: str, scope: str, section: str | None = None,
                condition: str | None = None, head: str | None = "when", qualifier: str | None = None) -> str:
    """S1/S3-value; a value stated under a condition carries it as the condition head."""
    scope_tag = "" if condition else section_tag(section)
    return (f"{as_of_header(document, as_of_date)}, {lead(condition, head, section, qualifier)}what is the "
            f"{tidy(slot)} for {tidy(scope)}{scope_tag}? {NOT_IN_EFFECT}")


NONE_OPTION = "none of these"


def option_order(pair_id: str) -> tuple[int, tuple[int, int]]:
    """S2-side option order from a seed derived from the S2 pair id: seed = first 8 hex digits of sha256(pair_id);
    even seed -> (side a outcome, side b outcome), odd -> swapped. Both sides of a pair share the order, so a pair's
    two golds are always one (A) and one (B) and the option position carries no cue. "none of these" is always
    option (C)."""
    seed = int(hashlib.sha256(pair_id.encode("utf-8")).hexdigest()[:8], 16)
    return seed, ((0, 1) if seed % 2 == 0 else (1, 0))


SIDE_TAIL = "which of the following applies?"


def side_options(options: list[str]) -> str:
    """The lettered list of the two verbatim outcomes (presentation order) + (C) none of these, rendered by the
    scorer's `render_lettered_options`, the single source of the lettered display (scorer rule N6')."""
    return render_lettered_options([tidy(o) for o in options] + [NONE_OPTION])


def s2_side_question(document: str, revision: str, effective_date: str, condition: str, options: list[str],
                     section: str | None = None, head: str | None = "when", qualifier: str | None = None) -> str:
    """`options` = the two verbatim outcomes already in presentation order; (C) none of these is appended."""
    return (f"{s2_header(document, revision, effective_date)}, {lead(condition, head, section, qualifier)}"
            f"{SIDE_TAIL} {side_options(options)}.")


def deadline_question(header: str, condition: str | None, subject: str, rest: str, variant: str, unit: str,
                      tail: str, as_of: bool, section: str | None = None, head: str | None = "when",
                      qualifier: str | None = None) -> str:
    """{header}, [{qualifier}, ][{condition head},] {subject} {modal + verb phrase} {variant} how many {unit}
    [ {tail}]? The section tag follows the condition when there is one, else the subject."""
    subj = tidy(subject) + ("" if condition else section_tag(section))
    tail_txt = f" {tidy(tail)}" if tail else ""
    q = f"{header}, {lead(condition, head, section, qualifier)}{subj} {tidy(rest)} {variant} how many {unit}{tail_txt}?"
    return f"{q} {NOT_IN_EFFECT}" if as_of else q


def _body_after_header(s2_question_text: str, s2_head: str) -> str:
    if not s2_question_text.startswith(s2_head + ", "):
        raise ValueError(f"S2 question does not start with its header {s2_head!r}")
    return s2_question_text[len(s2_head):]


def l2_question(s2_question_text: str, s2_head: str, document: str, as_of_date: str) -> str:
    """The L2 template: the S2 question (any form, section tag included) under an as-of header instead of the
    revision header, plus the not_in_effect sentence. `s2_head` = the `s2_header(...)` the S2 question starts with."""
    return f"{as_of_header(document, as_of_date)}{_body_after_header(s2_question_text, s2_head)} {NOT_IN_EFFECT}"


def l2_side_question(s2_question_text: str, s2_head: str, document: str, as_of_date: str, options: list[str]) -> str:
    """L2-side question of a "changed" branch (from the L2 item builder): the S2-side question under the as-of
    header with its option list replaced by `options` (the branch's current and earlier verbatim outcome, presentation
    order)."""
    body = _body_after_header(s2_question_text, s2_head)
    k = body.find(f" {SIDE_TAIL} ")
    if k < 0:
        raise ValueError("not an S2-side question")
    return f"{as_of_header(document, as_of_date)}{body[:k]} {SIDE_TAIL} {side_options(options)}. {NOT_IN_EFFECT}"


def l2_clause_question(document: str, as_of_date: str, condition: str, slot: str, scope: str,
                       section: str | None = None, head: str | None = "when") -> str:
    """L2 from a branch inside a revised clause (S1 path): the S2-value tail with the S1 "{slot} for {scope}" phrase."""
    return (f"{as_of_header(document, as_of_date)}, {lead(condition, head, section)}what is the "
            f"{tidy(slot)} for {tidy(scope)}? {NOT_IN_EFFECT}")


def leaked(fillers: list[str], answers: list[str]) -> list[str]:
    """Answers (gold or the other branch's value) that appear, token-bounded, inside the verbatim fillers."""
    joined = " | ".join(fillers)
    return [a for a in answers if a and contains_phrase(joined, a)]


# ------------------------------------------------------------------------------------------- number rule
ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
        "seventeen eighteen nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
NUMBER_TOKEN = re.compile(r"^[^A-Za-z]*\d[^A-Za-z]*$")   # "30", "(30)", "1.5", "3.2.1", "2026-05-20": no letter


def spelled(n: int) -> str | None:
    """English words for 0..999 ("one hundred twenty"), as the documents print spelled-out twins."""
    if not 0 <= n <= 999:
        return None
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else f" {ONES[n % 10]}")
    rest = n % 100
    return f"{ONES[n // 100]} hundred" + ("" if rest == 0 else f" {spelled(rest)}")


def answer_terms(gold: dict | None) -> list[str]:
    """Strings that must not appear in a question for this answer: the verbatim surface, "value unit-word", and the
    spelled-out number when the value is a whole number."""
    if not gold:
        return []
    surface = str(gold.get("surface") or gold.get("value") or "")
    terms = [surface]
    value = str(gold.get("value", ""))
    if re.fullmatch(r"-?\d+(?:\.\d+)?", value):
        words = surface.split()
        terms.append(f"{value} {words[-1]}" if words else value)
        if float(value).is_integer():
            terms.append(spelled(abs(int(float(value)))) or "")
    return [t for t in terms if t]


def number_free(phrase: str, banned: list[str]) -> bool:
    """Number rule for quantity / scope / subject / verb / tail phrases: no quantity ("30-minute", "100 kV", "5%"),
    no bare number token ("30", "(30)", "3.2"), no banned answer string. Digits inside a letter+digit identifier are
    names, not values, and pass ("Milestone (M2)", "P3/P6", "TOP-004-2 R4")."""
    if find_quantities(phrase) or any(NUMBER_TOKEN.match(t) for t in phrase.split()):
        return False
    return not leaked([phrase], banned)


# ------------------------------------------------------------------------------------------- headword_pp
TIME_DIMENSIONS = ("t", "t_cal")
HEADWORD_BY_DIMENSION = {"t": "time", "t_cal": "time", "pct": "percentage", "P": "amount", "E": "amount",
                         "Q": "amount", "S": "amount"}
HEADWORD_PP = re.compile(r"^\s*(?P<prep>of|for|from|after|before|prior to|following|beyond|above|below)\s", re.I)
MAX_PP_TOKENS = 8
RELATIVE = frozenset("that which who whom whose where when".split())


def headword_for(dimension: str) -> str:
    word = HEADWORD_BY_DIMENSION.get(dimension, "value")
    if word not in HEADWORDS:
        raise ValueError(f"headword {word!r} is not in the frozen HEADWORDS list")
    return word


def headword_slot(clause: str, ans_start: int, ans_end: int, dimension: str) -> Slot | None:
    """"the {headword} {verbatim PP}": the prepositional phrase right after the answer ("115% of the Facility Rating",
    "48 hours after the operating hour"), ended by clause punctuation or a stop word, <= MAX_PP_TOKENS tokens.
    Time quantities do not take "of" ("30 days of MISO's notice" names the start event, not the value)."""
    after = clause[ans_end:]
    m = HEADWORD_PP.match(after)
    if not m or (dimension in TIME_DIMENSIONS and m.group("prep").lower() == "of"):
        return None
    picked: list[re.Match] = []
    for t in TOKEN.finditer(after, m.start("prep")):
        raw = t.group(0)
        if picked and _bare(raw) in RELATIVE:   # "from the time MISO makes | that request": the PP would be cut
            return None
        if picked and (_bare(raw) in HARD_STOP or not _bare(raw)):
            break
        if len(picked) >= MAX_PP_TOKENS:
            return None
        picked.append(t)
        if raw[-1] in ",;:.":
            break
    while picked and _bare(picked[-1].group(0)) in PREPOSITIONS | DETERMINERS:
        picked.pop()
    if len(picked) < 2:
        return None
    a, b = ans_end + picked[0].start(), ans_end + picked[-1].end()
    text = _clean(clause[a:b])
    n_content = _content_count(text)
    if n_content < 1 or not balanced(text):
        return None
    return Slot(text, a, a + len(text), "headword_pp", n_content >= 2, headword_for(dimension))


def value_slot(clause: str, ans_start: int, ans_end: int, dimension: str | None, banned: list[str]
               ) -> tuple[Slot | None, str]:
    """"what is the" slot: the phrase rules (copula_np ... of_np) when their phrase passes the number rule, else
    headword_pp (numeric answers only, `dimension` given), else None. Returns (slot, note); note records why the phrase
    rules' phrase was not used (the note values name the phrase rules "v1", as the items record them in slot_note:
    no_v1_slot, v1_slot_failed_number_rule). A premod_np on a time quantity is not used: "30 Minute emergency voltage
    limits" names a kind of limit, and 30 minutes is not the value of the limits."""
    slot = find_slot(clause, ans_start, ans_end)
    if slot is not None and slot.rule == "premod_np" and dimension in TIME_DIMENSIONS:
        note = "premod_time_classifier"
    elif slot is not None and number_free(slot.text, banned):
        return slot, ""
    else:
        note = "v1_slot_failed_number_rule" if slot is not None else "no_v1_slot"
    if dimension is None:
        return None, note
    hw = headword_slot(clause, ans_start, ans_end, dimension)
    if hw is not None and number_free(hw.text, banned):
        return hw, note
    return None, note


# ------------------------------------------------------------------------------------------- sibling values
QUALIFIER_PP = re.compile(r"(?:with|without|under|during|for|in|on|at|after|before|prior to|following|upon)\s", re.I)
QUALIFIER_STOP = re.compile(r"[,;:()]|\.(?=\s|$)|\s(?:and|or|but|nor|unless|except|if|when|whenever|where|while)\b",
                            re.I)
MAX_QUALIFIER_WORDS = 6


def sibling_values(clause: str, q: Quantity, lo: int = 0) -> list[Quantity]:
    """Other quantities of the same dimension and another value in the main clause (from `lo`, never inside a
    subordinate clause): "a five (5%) percent distribution factor with system intact, or three (3%) percent under
    contingent conditions" gives one "what is the distribution factor" stem two golds."""
    return [x for x in find_quantities(clause)
            if x.start >= lo and x.start != q.start and x.dimension == q.dimension
            and (x.value, x.gold_unit) != (q.value, q.gold_unit) and not SUBORDINATOR.search(clause, lo, x.start)]


def value_qualifier(clause: str, slot: Slot, ans_end: int, banned: list[str]) -> tuple[int, int] | None:
    """Verbatim prepositional phrase that tells a value apart from a sibling value (same-stem, different-gold
    rule): the phrase right after the slot (premod_np / headword_pp follow the value) or right after the value (the
    other slot rules precede it), up to clause punctuation or a coordinator, 2..6 words, >= 1 content word, no number
    and no answer ("distribution factor | with system intact"). None when there is none."""
    a = max(slot.end, ans_end)
    a += len(clause[a:]) - len(clause[a:].lstrip())
    if not QUALIFIER_PP.match(clause, a):
        return None
    stop = QUALIFIER_STOP.search(clause, a + 1)
    b = len(clause[:stop.start() if stop else len(clause)].rstrip())
    text = clause[a:b]
    if not 2 <= len(text.split()) <= MAX_QUALIFIER_WORDS or _content_count(text) < 1 or not balanced(text):
        return None
    return (a, b) if number_free(text, banned) else None


# ------------------------------------------------------------------------------------------- deadline form
DEADLINE_LIMIT = re.compile(r"\b(?P<lim>within|no later than|not later than|at least once every|at least)\s+$", re.I)
# verbatim limit words (lower-cased, whitespace squashed) -> template words. The two "at least" variants:
# "at least how many {unit} prior to ..." is a lead time before an event, so its verbatim tail must start with
# "prior to"; "at least once every how many {unit}" is a recurrence interval
DEADLINE_VARIANT = {"within": "within", "no later than": "no later than", "not later than": "no later than",
                    "at least": "at least", "at least once every": "at least once every"}
LEAD_TIME_TAIL = re.compile(r"prior to\b", re.I)
PLURAL_UNIT = {"ms": "milliseconds", "s": "seconds", "min": "minutes", "h": "hours", "day": "days", "week": "weeks",
               "month": "months", "year": "years"}
STRICT_MODAL = re.compile(r"^(?:shall|must|is required|are required)$", re.I)
TAIL_STOP = re.compile(r"[;:,(]|\.(?=\s|$)|\s(?:and|or|but|unless|except|provided|so that|thereby|whereas|if|when"
                       r"|whenever|once|where|until|as soon as)\b", re.I)
SUBORDINATOR = re.compile(r"\b(?:if|when|whenever|unless|whether|where|once|until|because|while|provided)\b", re.I)
MAX_STEM_WORDS = 30
MAX_TAIL_WORDS = 12


@dataclass(frozen=True)
class Deadline:
    subject: tuple[int, int]      # offsets within the searched main clause
    rest: tuple[int, int]         # modal + verb phrase, up to the limit words
    modal: str
    limit: str                    # verbatim limit words ("within", "no later than", "not later than", "at least" ...)
    variant: str                  # template words: a DEADLINE_VARIANT value
    quantity: Quantity            # offsets within the main clause
    tail: tuple[int, int]         # verbatim continuation after the quantity (a == b when empty)
    tail_cut: str                 # conjunction that ended a non-empty tail ("or": "whether X or Y" lost "or Y")
    unit: str                     # canonical plural unit for the question ("business days")


def question_unit(q: Quantity) -> str:
    base = PLURAL_UNIT[q.unit]
    return f"{q.qualifier} {base}" if q.qualifier else base


def find_deadline(main: str, strict: bool) -> tuple[Deadline | None, str]:
    """Deadline form in a main clause "<subject> <modal> <verb phrase> <limit> <duration>[ <tail>]", limit = within |
    no later than (not later than) | at least (only when the tail starts "prior to") | at least once every. `strict`
    (S3) accepts only shall / must / is|are required. Returns (deadline, reason when None)."""
    subj = find_scope(main)
    if subj is None:
        return None, "no_subject"
    m = SUBJECT_MODAL.search(main)
    if strict and not STRICT_MODAL.match(m.group(0)):
        return None, "modal_not_strict"
    if re.match(r"\s+not\b", main[m.end():]):
        return None, "negated_modal"
    why = "no_deadline_limit_before_duration"
    for q in find_quantities(main):
        if q.start < m.end() or q.dimension not in TIME_DIMENSIONS or q.unit not in PLURAL_UNIT:
            continue
        lm = DEADLINE_LIMIT.search(main, m.end(), q.start)
        if lm is None:
            continue
        variant = DEADLINE_VARIANT[" ".join(lm.group("lim").lower().split())]
        if variant == "at least" and not LEAD_TIME_TAIL.match(main[q.end:].lstrip()):
            why = "at_least_without_prior_to"   # "run for at least 30 minutes": a minimum duration, not a lead time
            continue
        rest_end = len(main[:lm.start()].rstrip(" ,"))
        if rest_end <= m.end() - 1 or len(main[subj[0]:rest_end].split()) > MAX_STEM_WORDS:
            return None, "verb_phrase_too_long"
        if re.search(r"[;:]", main[subj[0]:rest_end]):
            return None, "verb_phrase_crosses_clause"
        if SUBORDINATOR.search(main, m.end(), rest_end):   # "shall be modeled if the equipment responds within 10 s"
            return None, "limit_inside_subordinate_clause"
        stop = TAIL_STOP.search(main, q.end)
        t_end = len(main[:stop.start() if stop else len(main)].rstrip())
        tail, cut = (q.end, q.end), ""
        if t_end > q.end and main[q.end:t_end].strip():
            ta = q.end + len(main[q.end:t_end]) - len(main[q.end:t_end].lstrip())
            if len(main[ta:t_end].split()) > MAX_TAIL_WORDS:
                return None, "tail_too_long"
            tail = (ta, t_end)
            word = stop.group(0).strip().lower() if stop else ""
            cut = word if word in ("and", "or", "but") else ""
        return Deadline(subj, (m.start(), rest_end), m.group(0), lm.group("lim"), variant, q, tail, cut,
                        question_unit(q)), ""
    return None, why
