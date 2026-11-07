"""Records memory (M7): typed constraint records of each as-of revision, extracted sub-block by sub-block by the
builder profile, and picked per question by BM25.

Build (independent of the question and of the budget: one request per sub-block, cap level and builder). The revision
is cut into the sections and sub-blocks of the summary memory (M5; procedures.memory.summary.sections / subblocks: the
same tiling and the same sub-block text per request). Each sub-block is one request (RECORD_PROMPT) for a JSON array
of at most k_i records with the fields RECORD_FIELDS; k_i = `record_quota` = max(MIN_RECORDS, body_i /
SOURCE_TOKENS_PER_RECORD rounded half up), body_i the sub-block's non-revision-history tokens (the revision-history
rule of procedures.text.exclusions, as in M5), so a revision-history sub-block asks for MIN_RECORDS. The requests
climb the M5 cap ladder (procedures.memory.summary.build_blocks / answers_from_records, the same code): cap level 1
caps the output at ceil(OUTPUT_CAP_FACTOR x TOKENS_PER_RECORD x k_i), a truncated answer is sent once more at level 2,
ceil(ESCALATED_CAP_FACTOR x TOKENS_PER_RECORD x k_i) + TOKENS_PER_RECORD, and the answer used is the highest level
that finished "stop", else the highest level answered. When the builder profile reasons (reasoning_expected), both
caps add its HIGH reasoning allowance, since reasoning tokens count against the output cap. The build call ledger is
the cache: the same sub-block is requested once whatever the question or the budget. The answer is read by the same
code as M5's, including its reasoning-field mapping for a builder whose panel entry sets answer_from_reasoning_field
(procedures.memory.summary.answer_text): a blank content is then parsed from the reasoning field when the record
finished "stop" / "end_turn" (any other finish leaves it empty).

Parsing (`parse_records`, deterministic): think blocks stripped; one Markdown code fence around the answer removed;
the records are, in this order of preference, the first "[" that the standard JSON decoder reads as an array holding
at least one object; else the complete elements of the first "[" that opens an object (an array that does not decode
whole, e.g. an answer cut at its cap: `cut` is 1); else the first empty array (a part without rules); else none. A
bracket in prose or a list of numbers is passed over. `array` records how: strict (the answer is the array), fenced,
embedded (text around it), salvaged, or none (no records). Each element is validated (`validate_record`): an object
whose values are strings, numbers or null and whose source_span is a non-empty string; a missing field reads as null
(counted); anything else is malformed, dropped and counted by reason. Numbers are kept as their JSON text. A
source_span longer than SPAN_MAX_CHARS characters follows SPAN_TOO_LONG, a switch whose rule was fixed before any
answer was read (see the constant): "drop" (the element is malformed, reason source_span_too_long) or "truncate" (the
setting in use: the span is cut to its first SPAN_MAX_CHARS characters and the record kept, counted as
`truncated_spans`; a prefix of a verbatim quote is still verbatim, so `locate` still places it). The switch is in
PARAMS, hence in the memory cache key, and not in the prompt: the requests, and so the build ledger, are the same under
either setting.

Provenance (`locate`): the source span (stripped) is searched in the revision's LF text, first inside its sub-block,
then anywhere; verbatim first, then with any run of whitespace matching any run of whitespace (the builder saw the raw
text, whose line breaks it may copy as spaces). A span found gives its char offsets, `located` true and `match`
verbatim | whitespace; a span not found keeps its record with `located` false and char offsets None. Every entry
carries the sub-block span it came from.

Query time (`build_m7`): every record of the as-of revisions is a candidate (the records are the memory, as the
sentences of the constraint-sentence memory, M4, are); a record line repeated in a revision is kept once (counted). A
record renders as one line, "- " and its non-null fields as "name: value" joined by " | " in RECORD_FIELDS order,
values with whitespace collapsed. Candidates are ranked by BM25 against the question (procedures.memory.chunking.rank;
the bag is the fields' values: the field names are the same on every line) and packed to the budget in rank order
(procedures.memory.chunking.pack), then shown per revision (ladder order, under its header line) in document order
(sub-block, then position in its answer).

Mechanical checks (`M7Memory.checks`, live builds): sub-blocks without a JSON array, sub-blocks whose answer used did
not finish "stop", sub-blocks with more valid records than quota + QUOTA_SLACK, malformed records, records not
located, and whether the memory exceeds its budget.

Per-sub-block, per-cap-level cache key (`record_key`): (representation, doc_id, revision, sha256_lf, builder_profile,
section_index, section_span, subblock_index, subblock_span, quota, cap_level, max_output_tokens, prompt_sha,
token_method): no budget and no question.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

from procedures.text.tokenize import tokenize
from procedures.readers import Cell
from procedures.readers.tokens import estimate_tokens, token_method
from procedures.scoring.parse import TRUNCATION_REASONS, strip_think

from procedures.memory.chunking import pack, rank
from procedures.memory.summary import CAP_LEVELS, ESCALATED_CAP_FACTOR, OUTPUT_CAP_FACTOR, QUOTA_SLACK, BlockBuild, BlockCall, BuilderLedger, build_blocks, panel, sections, subblocks
from procedures.corpus.revisions import RevisionText, in_ladder_order

RECORD_FIELDS = ("subject", "quantity", "value", "unit", "condition", "exception", "duration", "effective",
                 "source_span")
RECORD_PROMPT = """You are building a reference memory for an assistant that will later answer questions about a \
document without seeing the document itself. The memory is a list of typed constraint records, extracted part by part \
in document order; below is one part of one section of the document. Extract the rules THIS part states as a JSON \
array of at most {records} records.

Each record is a JSON object with exactly these keys (null when the part does not state it):
- "subject": who or what the rule applies to, in the document's own terms or role names
- "quantity": what the rule limits, requires or sets (for example a limit, deadline, rating, notification or action)
- "value": the value exactly as the part states it (a number or the document's wording)
- "unit": the unit of that value exactly as stated
- "condition": when the rule applies (when, if, for, in the event of)
- "exception": the stated exception (unless, except, otherwise)
- "duration": a duration, deadline or period exactly as stated
- "effective": an effective date or period the part states for the rule
- "source_span": a verbatim quote of at most {span_chars} characters from the part that states the rule

Rules:
- One record per rule, in the order the part states them; each branch of a rule (if / otherwise, normal / \
emergency) is its own record with its own condition.
- Cover the whole part: if it states more rules than you have records, give one record to each of the most \
consequential rules (thresholds, deadlines, conditions, exceptions) rather than only to the first ones.
- Copy values, units, durations and dates exactly; never invent, estimate, round or convert a value.
- source_span is copied character for character from the part (it may be shortened, never reworded).
- Skip the table of contents, page headers and footers, and background description; if the part states no rule, \
output [].
- Output the JSON array only: no prose and no code fence.

Document: {header}
Section: {section}
Part: {part}
<part>
{text}
</part>"""
SPAN_MAX_CHARS = 200
SPAN_TOO_LONG_RULES = ("drop", "truncate")
SPAN_TOO_LONG = "truncate"      # rule fixed in advance: "drop" unless it drops > 10% of elements; a development build
                                # dropped 43 of 228 (18.9%), so "truncate" applies to every full build
SOURCE_TOKENS_PER_RECORD = 125  # k_i = max(MIN_RECORDS, body_i / this rounded half up): 8 records per 1k tokens
TOKENS_PER_RECORD = 112         # a record's expected JSON length; the sub-block's token target is this x k_i
MIN_RECORDS = 1
PROMPT_SHA = hashlib.sha256(json.dumps({"template": RECORD_PROMPT, "fields": RECORD_FIELDS, "span_chars": SPAN_MAX_CHARS},
                                       sort_keys=True).encode("utf-8")).hexdigest()
STAGE = "build/M7"
CAP_ESCALATION = (f"level 1: max_output_tokens ceil({OUTPUT_CAP_FACTOR} x {TOKENS_PER_RECORD} x quota); level 2, only "
                  f"after a truncated level-1 answer: the same prompt at ceil({ESCALATED_CAP_FACTOR} x "
                  f"{TOKENS_PER_RECORD} x quota) + {TOKENS_PER_RECORD}; both + the builder's HIGH reasoning allowance "
                  "when it reasons; no level 3; the memory uses the highest level that finished stop, else the "
                  "highest level answered")
# hashed into the M7 cache key (the paper's runs recorded keys made with these exact values); the "subblocks" label
# names the module of the sub-block plan as it was called then (procedures.memory.summary)
PARAMS = {"subblocks": "tg.memory.m5 sections / subblocks", "source_tokens_per_record": SOURCE_TOKENS_PER_RECORD,
          "tokens_per_record": TOKENS_PER_RECORD, "min_records": MIN_RECORDS, "quota_slack": QUOTA_SLACK,
          "span_max_chars": SPAN_MAX_CHARS, "span_too_long": SPAN_TOO_LONG, "cap_escalation": CAP_ESCALATION,
          "parse": "one code fence removed; first array holding an object, else the complete elements of the first "
                   "array opening an object, else the first empty array; malformed elements dropped and counted",
          "locate": "in sub-block, then revision; verbatim, then whitespace-insensitive",
          "render": "one line per record: non-null fields 'name: value' joined by ' | '",
          "select": "BM25 over field values, all records candidates, packed in rank order, shown in document order",
          "dedupe": "identical record line within a revision kept once"}

FENCE_OPEN = re.compile(r"\A```[A-Za-z0-9_-]*[ \t]*\n")
FENCE_CLOSE = re.compile(r"\n?```\s*\Z")
OBJECT_ARRAY = re.compile(r"\[(?=\s*\{)")          # "[" opening an array whose first element is an object
PLACEHOLDER = "[dry-run placeholder: no records of {header} were built for a {budget}-token memory]"


@dataclass(frozen=True)
class RecordCall(BlockCall):
    """A sub-block's M7 request per cap level (procedures.memory.summary.BlockCall: `target` is the sub-block's token
    target, `quota` its record allowance k_i)."""

    @property
    def block_target(self) -> int:
        return TOKENS_PER_RECORD * self.quota


@dataclass(frozen=True)
class Parsed:
    """One answer's records (`parse_records`)."""
    records: tuple[dict, ...]           # valid records: RECORD_FIELDS -> str | None, as written
    array: str                          # strict | fenced | embedded | salvaged | none
    cut: int                            # 1 when elements after an undecodable one were lost (salvaged array)
    malformed: tuple[tuple[str, int], ...]   # (reason, count), sorted
    missing_fields: int                 # valid records with at least one field absent (read as null)
    truncated_spans: int = 0            # valid records whose source_span was cut to SPAN_MAX_CHARS (SPAN_TOO_LONG)

    @property
    def dropped(self) -> int:
        return sum(n for _, n in self.malformed)


@dataclass(frozen=True)
class Record:
    """A candidate of the memory: one valid record of one sub-block."""
    revision: RevisionText
    order: tuple[int, int]              # (sub-block position in the plan, position in its answer)
    text: str                           # the rendered line
    tokens: int
    provenance: dict
    bag: tuple[str, ...] = field(repr=False, compare=False)


@dataclass(frozen=True)
class M7Memory:
    text: str
    tokens: int
    budget: int
    provenance: list
    builds: tuple[BlockBuild, ...]
    parsed: tuple                       # Parsed per sub-block (live); None per sub-block (dry_run)
    candidates: int
    packed: int
    duplicates: int
    located: int                        # candidates whose source span was found
    flags: tuple[str, ...]

    @property
    def mode(self) -> str:
        return "dry_run" if any(b.mode == "dry_run" for b in self.builds) else "live"

    def cap_levels(self) -> dict[int, int]:
        return dict(sorted(Counter(b.level for b in self.builds).items()))

    def still_capped(self) -> int:
        return sum(b.finish_reason in TRUNCATION_REASONS for b in self.builds)

    def checks(self) -> dict[str, int]:
        """The mechanical checks of a live build (module docstring); zero everywhere = all hold."""
        return {"subblocks_without_array": sum(p.array == "none" for p in self.parsed),
                "subblocks_not_stop": sum(b.finish_reason != "stop" for b in self.builds),
                "subblocks_over_quota": sum(len(p.records) > b.call.quota + QUOTA_SLACK
                                            for b, p in zip(self.builds, self.parsed)),
                "malformed_records": sum(p.dropped for p in self.parsed),
                "unlocated_records": self.candidates - self.located,
                "over_budget": int(self.tokens > self.budget)}

    def record_counts(self) -> dict:
        """Totals over the sub-blocks' answers (live)."""
        malformed: Counter = Counter()
        for p in self.parsed:
            malformed.update(dict(p.malformed))
        return {"valid": sum(len(p.records) for p in self.parsed), "candidates": self.candidates,
                "duplicates": self.duplicates, "packed": self.packed, "located": self.located,
                "malformed": dict(sorted(malformed.items())), "cut": sum(p.cut for p in self.parsed),
                "missing_fields": sum(p.missing_fields for p in self.parsed),
                "truncated_spans": sum(p.truncated_spans for p in self.parsed),
                "arrays": dict(sorted(Counter(p.array for p in self.parsed).items()))}

    def info(self) -> list[dict]:
        """One entry per sub-block call, in plan order."""
        out = []
        for b, p in zip(self.builds, self.parsed):
            c, s, blk = b.call, b.call.section, b.call.block
            out.append({"doc_id": c.revision.doc, "revision": c.revision.revision, "call_id": c.call_id,
                        "section_index": s.index, "section_count": s.count, "section_span": [s.char_start, s.char_end],
                        "subblock_index": blk.index, "subblock_count": blk.count, "span": [blk.char_start, blk.char_end],
                        "title": blk.title, "subblock_tokens": blk.tokens, "history_tokens": blk.history_tokens,
                        "quota": c.quota, "block_target_tokens": c.block_target, "cap_level": b.level,
                        "max_output_tokens": b.max_output_tokens, "block_key": c.key, "levels": list(b.levels),
                        "mode": b.mode, "request_hash": b.request_hash, "build_tokens": b.build_tokens, "usd": b.usd,
                        "finish_reason": b.finish_reason, "output_tokens": b.output_tokens,
                        "records": None if p is None else len(p.records),
                        "malformed": None if p is None else dict(p.malformed),
                        "array": None if p is None else p.array, "cut": None if p is None else p.cut,
                        "flags": list(b.flags), "reason": b.reason})
        return out


# ------------------------------------------------------------------------------------------------ requests
def record_quota(block) -> int:
    """k_i = max(MIN_RECORDS, the sub-block's non-history tokens / SOURCE_TOKENS_PER_RECORD rounded half up)."""
    return max(MIN_RECORDS, (block.body_tokens + SOURCE_TOKENS_PER_RECORD // 2) // SOURCE_TOKENS_PER_RECORD)


def record_cap(quota: int, level: int = 1) -> int:
    """A sub-block request's output cap before the reasoning allowance: level 1 ceil(OUTPUT_CAP_FACTOR x
    TOKENS_PER_RECORD x quota), level 2 ceil(ESCALATED_CAP_FACTOR x TOKENS_PER_RECORD x quota) + TOKENS_PER_RECORD."""
    if level == 1:
        return math.ceil(OUTPUT_CAP_FACTOR * TOKENS_PER_RECORD * quota)
    if level == 2:
        return math.ceil(ESCALATED_CAP_FACTOR * TOKENS_PER_RECORD * quota) + TOKENS_PER_RECORD
    raise ValueError(f"cap level must be one of {CAP_LEVELS}, got {level!r}")


def reasoning_allowance(builder_profile: str) -> int:
    """The builder's HIGH-scenario reasoning tokens when its profile reasons, else 0."""
    pp = panel().get(builder_profile)
    return pp.expected_reasoning_tokens["high"] if pp.reasoning_expected else 0


def record_key(rev: RevisionText, section, block, builder_profile: str, quota: int, level: int, cap: int) -> str:
    fields = {"representation": "M7", "doc_id": rev.doc, "revision": rev.revision, "sha256_lf": rev.sha256_lf,
              "builder_profile": builder_profile, "section_index": section.index,
              "section_span": [section.char_start, section.char_end], "subblock_index": block.index,
              "subblock_span": [block.char_start, block.char_end], "quota": quota, "cap_level": level,
              "max_output_tokens": cap, "prompt_sha": PROMPT_SHA, "token_method": token_method()}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode("utf-8")).hexdigest()


def record_cell(rev: RevisionText, section, block, quota: int, level: int, allowance: int) -> Cell:
    """The sub-block's request at cap `level`: the same prompt at every level; a level-2 request follows an answer that
    ran to the level-1 cap, so it is estimated at that many visible output tokens; its record is labelled
    cap_level 2."""
    prompt = RECORD_PROMPT.format(records=quota, span_chars=SPAN_MAX_CHARS, header=rev.header(),
                                  section=section.label(), part=block.label(),
                                  text=rev.text()[block.char_start:block.char_end])
    target = TOKENS_PER_RECORD * quota
    return Cell(item_id=f"{rev.doc}@{rev.revision}#{section.index}.{block.index}", stage=f"{STAGE}@{target}",
                messages=({"role": "user", "content": prompt},),
                generation={"max_output_tokens": record_cap(quota, level) + allowance},
                est_visible_tokens=target if level == 1 else record_cap(quota, level - 1),
                labels={} if level == 1 else {"cap_level": level})


def plan(revisions: list[RevisionText], builder_profile: str) -> list[RecordCall]:
    """Every sub-block request of the revisions (given (ladder) order, then document order); no budget. Planned once
    per process per revision set and builder."""
    return list(_plan(tuple(revisions), builder_profile))


@lru_cache(maxsize=256)
def _plan(revisions: tuple[RevisionText, ...], builder_profile: str) -> tuple[RecordCall, ...]:
    allowance = reasoning_allowance(builder_profile)
    calls = []
    for rev in revisions:
        for s in sections(rev):
            for p in subblocks(rev, s):
                quota = record_quota(p)
                calls.append(RecordCall(
                    rev, s, p, TOKENS_PER_RECORD * quota, quota,
                    tuple(record_key(rev, s, p, builder_profile, quota, level, record_cap(quota, level) + allowance)
                          for level in CAP_LEVELS),
                    tuple(record_cell(rev, s, p, quota, level, allowance) for level in CAP_LEVELS)))
    return tuple(calls)


# ------------------------------------------------------------------------------------------------ parsing
def span_rule(too_long: str | None) -> str:
    """The span rule to apply: `too_long`, or SPAN_TOO_LONG when None; anything else than SPAN_TOO_LONG_RULES raises."""
    rule = SPAN_TOO_LONG if too_long is None else too_long
    if rule not in SPAN_TOO_LONG_RULES:
        raise ValueError(f"span rule must be one of {SPAN_TOO_LONG_RULES}, got {rule!r}")
    return rule


def validate_record(obj, too_long: str | None = None) -> tuple[dict | None, str | None, bool]:
    """(record, None, missing any field) for a valid element, (None, reason, False) for a malformed one; `too_long`
    is the span rule, default SPAN_TOO_LONG (a truncated span shows as a source_span of exactly SPAN_MAX_CHARS
    characters)."""
    too_long = span_rule(too_long)
    if not isinstance(obj, dict):
        return None, "not_object", False
    values = {}
    for name in RECORD_FIELDS:
        v = obj.get(name)
        if v is not None and (isinstance(v, bool) or not isinstance(v, (str, int, float))):
            return None, "non_scalar_field", False
        values[name] = v if v is None or isinstance(v, str) else json.dumps(v)
    span = values["source_span"]
    if span is None or not span.strip():
        return None, "no_source_span", False
    if len(span) > SPAN_MAX_CHARS:
        if too_long == "drop":
            return None, "source_span_too_long", False
        values["source_span"] = span[:SPAN_MAX_CHARS]
    return values, None, any(name not in obj for name in RECORD_FIELDS)


def _salvage(body: str, start: int) -> tuple[list, int]:
    """The elements of the array opening at `start` that decode one by one, and 1 when decoding stopped before the
    array closed (0 when it closed)."""
    decoder = json.JSONDecoder()
    out, pos = [], start + 1
    while True:
        while pos < len(body) and body[pos].isspace():
            pos += 1
        if pos < len(body) and body[pos] == "]":
            return out, 0
        try:
            value, pos = decoder.raw_decode(body, pos)
        except json.JSONDecodeError:
            return out, 1
        out.append(value)
        while pos < len(body) and body[pos].isspace():
            pos += 1
        if body.startswith(",", pos):
            pos += 1
        elif not body.startswith("]", pos):
            return out, 1


def parse_records(text: str, too_long: str | None = None) -> Parsed:
    """The records of one answer (module docstring); `too_long` = the span rule, default SPAN_TOO_LONG."""
    too_long = span_rule(too_long)
    body = strip_think(text or "")[0].strip()
    fenced = FENCE_OPEN.match(body) is not None
    if fenced:
        body = FENCE_CLOSE.sub("", FENCE_OPEN.sub("", body, count=1)).strip()
    decoder = json.JSONDecoder()
    elements, array, cut = None, "", 0
    empty = None                                        # (start, end) of the first empty array
    for m in re.finditer(r"\[", body):
        try:
            value, end = decoder.raw_decode(body, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, list) and any(isinstance(v, dict) for v in value):
            elements, span = value, (m.start(), end)
            break
        if value == [] and empty is None:
            empty = (m.start(), end)
    start = OBJECT_ARRAY.search(body)
    if elements is None and start is not None:
        elements, cut = _salvage(body, start.start())
        array = "salvaged"
    elif elements is None and empty is None:
        return Parsed((), "none", 0, (), 0)
    elif elements is None:
        elements, span = [], empty
    if not array:
        array = "fenced" if fenced else ("strict" if span[0] == 0 and not body[span[1]:].strip() else "embedded")
    records, malformed, missing, truncated = [], Counter(), 0, 0
    for obj in elements:
        record, reason, absent = validate_record(obj, too_long)
        if record is None:
            malformed[reason] += 1
        else:
            records.append(record)
            missing += absent
            truncated += len(obj["source_span"]) > SPAN_MAX_CHARS
    return Parsed(tuple(records), array, cut, tuple(sorted(malformed.items())), missing, truncated)


def locate(text: str, span: str, lo: int, hi: int) -> tuple[int, int, str] | None:
    """(start, end, verbatim | whitespace) of `span` (stripped) in `text`: inside [lo, hi) first, then anywhere;
    verbatim first, then with any whitespace run matching any whitespace run. None when not found."""
    span = span.strip()
    windows = ((lo, hi), (0, len(text)))
    for a, b in windows:
        i = text.find(span, a, b)
        if i >= 0:
            return i, i + len(span), "verbatim"
    pattern = re.compile(r"\s+".join(map(re.escape, span.split())))
    for a, b in windows:
        m = pattern.search(text, a, b)
        if m:
            return m.start(), m.end(), "whitespace"
    return None


def render_record(record: dict) -> str:
    """One line: "- " and the non-null fields "name: value" (whitespace collapsed) joined by " | "."""
    shown = [(name, " ".join(record[name].split())) for name in RECORD_FIELDS if record[name] is not None]
    return "- " + " | ".join(f"{name}: {value}" for name, value in shown if value)


@lru_cache(maxsize=4096)
def block_records(rev: RevisionText, lo: int, hi: int, answer: str, too_long: str) -> tuple[Parsed, tuple]:
    """(Parsed, per valid record its `locate` result) for one sub-block answer under span rule `too_long`; cached per
    sub-block, answer and rule."""
    parsed = parse_records(answer, too_long)
    text = rev.text()
    return parsed, tuple(locate(text, r["source_span"], lo, hi) for r in parsed.records)


def record_units(calls: list[RecordCall], builds: list[BlockBuild]) -> tuple[list[Record], list[Parsed], int]:
    """(the records of every sub-block in document order with a repeated line of a revision kept once, the parse per
    sub-block, how many repeated lines were dropped)."""
    out, parsed, seen, duplicates = [], [], set(), 0
    for position, (call, build) in enumerate(zip(calls, builds)):
        blk = call.block
        p, where = block_records(call.revision, blk.char_start, blk.char_end, build.text, SPAN_TOO_LONG)
        parsed.append(p)
        for j, (record, found) in enumerate(zip(p.records, where)):
            line = render_record(record)
            if (call.revision, line) in seen:
                duplicates += 1
                continue
            seen.add((call.revision, line))
            start, end, match = found if found else (None, None, None)
            prov = {"doc_id": call.revision.doc, "revision": call.revision.revision, "path": call.revision.source_path,
                    "char_start": start, "char_end": end, "located": found is not None, "match": match,
                    "block_span": [blk.char_start, blk.char_end]}
            values = " ".join(record[name] for name in RECORD_FIELDS if record[name] is not None)
            out.append(Record(call.revision, (position, j), line, estimate_tokens(line), prov,
                              tuple(tokenize(values))))
    return out, parsed, duplicates


def _grouped(records: list[Record]) -> list[tuple[RevisionText, list[Record]]]:
    """Records by revision in ladder order, document order inside a revision (the rendered order)."""
    by_rev: dict[RevisionText, list[Record]] = {}
    for r in records:
        by_rev.setdefault(r.revision, []).append(r)
    return [(rev, sorted(by_rev[rev], key=lambda r: r.order)) for rev in in_ladder_order(by_rev)]


def render(records: list[Record]) -> str:
    """Per revision (ladder order) its header line, then its record lines in document order."""
    return "\n\n".join(f"[{rev.header()}]\n" + "\n".join(r.text for r in body) for rev, body in _grouped(records))


# ------------------------------------------------------------------------------------------------ memory
def _check_frame(revisions: list[RevisionText], budget: int) -> None:
    frame = "\n\n".join(f"[{rev.header()}]" for rev in revisions)
    if estimate_tokens(frame) > budget:
        raise ValueError(f"budget {budget} leaves no room for records after the revision headers "
                         f"({estimate_tokens(frame)} tokens)")


def assemble(calls: list[RecordCall], builds: list[BlockBuild], question: str, budget: int) -> M7Memory:
    """The memory for one question (module docstring): live builds; a dry-run build yields the placeholder memory."""
    revisions = in_ladder_order(c.revision for c in calls)
    _check_frame(revisions, budget)
    if any(b.mode == "dry_run" for b in builds):
        text = "\n\n".join(f"[{rev.header()}]\n" + PLACEHOLDER.format(header=rev.header(), budget=budget) for rev in revisions)
        if estimate_tokens(text) > budget:
            text = "\n\n".join(f"[{rev.header()}]" for rev in revisions)
        return M7Memory(text, estimate_tokens(text), budget, [], tuple(builds), tuple(None for _ in builds), 0, 0, 0,
                        0, ())
    units, parsed, duplicates = record_units(calls, builds)
    chosen, used = pack([u for _, u in rank(units, question)], budget, render)
    text = render(chosen)
    if used > budget:
        raise RuntimeError(f"M7 memory of {used} tokens exceeds budget {budget}")
    flags = {f for b in builds for f in b.flags}
    return M7Memory(text, used, budget, [r.provenance for _, body in _grouped(chosen) for r in body], tuple(builds),
                    tuple(parsed), len(units), len(chosen), duplicates,
                    sum(u.provenance["located"] for u in units), tuple(sorted(flags)))


def build_m7(revisions: list[RevisionText], question: str, budget: int, builder_profile: str,
             ledger: BuilderLedger | None) -> tuple[list[RecordCall], M7Memory]:
    """(the plan, the memory): live into the ledger, or dry_run without one (procedures.memory.summary.build_blocks)."""
    if ledger is None:
        calls, memory = _dry_m7(tuple(revisions), budget, builder_profile)
        return list(calls), memory
    calls = plan(revisions, builder_profile)
    return calls, assemble(calls, build_blocks(calls, builder_profile, ledger), question, budget)


@lru_cache(maxsize=512)
def _dry_m7(revisions: tuple[RevisionText, ...], budget: int, builder_profile: str) -> tuple[tuple, M7Memory]:
    """build_m7 without a ledger (a dry run, nothing sent): the placeholder memory does not read the question, so it is
    the same for every item reading these revisions and planned and priced once per process."""
    calls = plan(list(revisions), builder_profile)
    return tuple(calls), assemble(calls, build_blocks(calls, builder_profile, None), "", budget)
