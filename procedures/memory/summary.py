"""Summary memory (M5): a section-wise LLM summary of each as-of revision, built sub-block by sub-block by the builder
profile.

Why sub-blocks and a cap ladder. Asked for a summary of a whole revision or section with a token target, the builder
transcribes densely from the start and ignores the target. In development builds a whole-revision summary hit its
2 x target output cap on 13 of 18 revisions and section summaries hit their 1.25 x cap on 82 of 84 sections; a
bullet-count limit (at most N bullets per section) did not help (2 of 3 sections still hit the cap, reaching ~5-36% of
the section). Breadth is therefore forced by structure: every section is cut into sub-blocks of ~PART_MAX_TOKENS
tokens and each sub-block is its own request with a small bullet quota (this reached the end of every section, but 6
of 17 sub-blocks ran to a 1.25 x 28 x k_i cap). The cap factor is 2.0 (the builder's bullets run p50 32 / p90 50
tokens), a bullet cut mid-sentence is dropped, and because 76 of 465 sub-blocks still reached the cap at 2.0 (at every
quota size, caps 56 to 784), a truncated answer climbs a cap ladder (`CAP_ESCALATION`).

Sections (`sections`, independent of the budget). Units are the sentences of procedures.text.sentences.segment (the
offset-preserving splitter the other memories use), each owning its slice of the LF text up to the next sentence
start, so no sentence is ever split. Revision history is detected with the item construction's own rule: a sentence
is history when procedures.text.exclusions.history_regions puts its start inside a region. A run of history sentences
of at least HISTORY_SPAN_MIN tokens is a span of its own; the rest of the text forms the body spans between them (a
span under HISTORY_SPAN_MIN tokens joins its predecessor, the first its successor). Each span of T tokens is cut into
n = ceil(T / SECTION_MAX_TOKENS) contiguous sections: the k-th cut goes to the section heading
(procedures.corpus.revision_text's headings, TOC-confirmed when the document has a TOC) nearest to k*T/n within
SNAP*T/n tokens of it; failing that to the nearest paragraph (layout block) start in that window; failing that to the
nearest sentence start. The sections tile the text: their spans concatenate to the whole LF text (page furniture
included, as the build prompt always carried it). A section is titled by the heading in effect at its start
("(continued)" when it starts inside that heading's section) and the last heading that starts inside it (a
table-of-contents dot leader and page number dropped from a title). A section is a revision-history section when at
least half its tokens are history.

Sub-blocks (`subblocks`, independent of the budget): the same cut rule inside each section with
n = ceil(T / PART_MAX_TOKENS), so sub-blocks tile their section, open on a heading or paragraph where one is in reach
and never split a sentence; titled like sections; the prompt calls them "Part k/n".

Targets (`plan`). The budget left after the rendered frame (revision headers + section headers) is split evenly over
the as-of revisions (one in practice). Within a revision, each revision-history section gets TARGET_FLOOR only, and
what is left is split over the other sections in proportion to their non-history tokens (history inside them gets no
weight): target = min(share, max(TARGET_FLOOR, round(rest * body_tokens / sum of body_tokens))), with
rest = share - TARGET_FLOOR * (history sections). A section's bullets N = bullet_count(target) =
max(MIN_BULLETS, target / TOKENS_PER_BULLET rounded half up) are split over its sub-blocks by `quota_split`:
k_i = max(1, N * w_i / sum w rounded half up), w_i the sub-block's non-history tokens (all its tokens in a
revision-history section, or when the section has no non-history tokens). A sub-block's token target is
TOKENS_PER_BULLET * k_i and its request caps the output at `block_cap(k_i)` =
ceil(OUTPUT_CAP_FACTOR * TOKENS_PER_BULLET * k_i), a safety net (factor 2.0: 56 tokens a bullet, see above).

Cap ladder (`CAP_ESCALATION`): that request is cap level 1. When its answer finishes truncated (`escalates`:
procedures.scoring.parse.TRUNCATION_REASONS, "length" for the DeepSeek builder), the same prompt is sent once more at
cap level 2, `block_cap(k_i, 2)` = ceil(ESCALATED_CAP_FACTOR * TOKENS_PER_BULLET * k_i) + ESCALATED_CAP_EXTRA =
ceil(140 k_i) + 112, its call-ledger record labelled `cap_level: 2`; there is no level 3. Both records stay in the
ledger. The memory uses the highest level that finished "stop", else the highest level answered (a level-2 answer
that is still truncated is kept, the cut-bullet rule applies, and it is counted as `still_capped`). Each level has its
own block key; the level-2 request is fixed by the plan, so the memory's cache key (every level's block key) is known
before anything is built.

Builds: one request per sub-block and cap level (SUBBLOCK_PROMPT: at most k_i bullets of at most BULLET_WORDS words,
one per rule or condition in order, covering the whole sub-block, under the revision header, the section label and
the part label), all through procedures.readers.executor.run_cells (never another path), one call per cap level per
memory (`build_blocks`, which may also be called on part of a plan).
  * live: a `BuilderLedger` is given and the builder profile's credential variable is set. run_cells writes the calls
    to that call ledger (append-only; resume never resends an answered request, so the ledger is the durable cache),
    level 1 for every sub-block, then level 2 for those whose level-1 answer escalates, and each summary is read back
    from its ledger record by request hash: think blocks stripped (procedures.scoring.parse.strip_think), whitespace
    trimmed (`answer_text`). A missing record, a record that is not an observed answer, or an empty summary raises
    (the build did not happen: a failure of the mechanism sanity check). A truncated finish
    (procedures.scoring.parse.TRUNCATION_REASONS) is kept and flagged `build_truncated`. build_tokens = the
    provider's input_total + output_total, summed over every level sent.
    Reasoning-field mapping (the build-side counterpart of the reader-side mapping,
    procedures.readers.parse.answer_from_reasoning): for a builder whose panel entry sets
    `answer_from_reasoning_field` (the Qwen3 profiles on nscale, whose server can put the whole answer in
    reasoning_content), an answer whose content is blank after the think strip is read from the record's reasoning
    field (`reasoning_text`, think-stripped and trimmed) when that is not blank and the record finished normally (its
    finish_reason in REASONING_FIELD_FINISH: "stop", the OpenAI shape, or "end_turn", the Anthropic shape); a
    reasoning field of a record that finished otherwise ("length" or any other value)
    is not read, so that answer is empty and takes the empty / truncated handling and the cap ladder as without the
    mapping. A mapped sub-block is flagged `text_from_reasoning_field`. The memory counts those sub-blocks and warns
    when they exceed REASONING_FIELD_WARN_SHARE of its sub-blocks (the 10% line the analysis applies to empty reader
    answers, procedures.analysis.capability.EMPTY_VOID_SHARE; here it voids nothing). Any other builder reads the
    content only.
  * dry_run: no ledger, or the credential variable is unset. run_cells(dry_run=True) prices every request of both
    levels (the level-2 one at the level-1 cap as its output: it follows an answer that reached that cap); each body
    is a placeholder; build_tokens = reference tokens of each level-1 prompt as sent + its token target, summed. The
    assembled dry-run memory ends with DRY_BUDGET_LINE naming its budget (after the budget fit: it is a pricing
    stand-in, a few tokens over the placeholders' fit at most), so a revision's four budget-ladder stand-ins are four
    distinct reader requests.
`answers_from_records` reads the builds back from a build ledger's records without sending anything (the mechanism
sanity check of a run, procedures.matrix.sanity, assembles the live memory from them).

Assembly (`assemble`): first the cut-bullet rule (`drop_cut_bullet`): an answer whose finish is a truncation
(procedures.scoring.parse.TRUNCATION_REASONS) loses its last bullet (from that bullet's line to the end) when its last
line does not end at a sentence boundary (. ! or ?, optionally followed by closing quotes or brackets); the drops are
recorded (`dropped_bullets`, flag `dropped_cut_bullets`). Then per revision (ladder order) its header line, then per
section in document order "## Section k/n: title" and the bullet lists of its sub-blocks, concatenated in order. When
the rendered text exceeds the budget, the sub-block summaries are trimmed at sentence boundaries
(procedures.text.sentences.segment) in proportion to their token targets: every sub-block is capped at the same
multiple rho of its target, rho (in thousandths) the largest whose rendered memory fits, so only the sub-blocks
longest relative to their share are cut and every section keeps its breadth. The cut is recorded (`trimmed_tokens`,
flag `trimmed_to_budget`).

Mechanical checks (`M5Memory.checks`, live builds; recorded in MemoryResult.notes and build): sub-blocks without a
bullet line (`bullet_lines`), sub-blocks whose finish_reason is not "stop", sub-blocks
with more than quota + QUOTA_SLACK bullets, and whether the memory exceeds its budget after the trim.

Per-sub-block, per-cap-level cache key (`block_key`): (doc_id, revision, sha256_lf, builder_profile, budget,
section_index, section_span, revision_history, target_tokens, subblock_index, subblock_span, quota, cap_level,
max_output_tokens, prompt_sha, token_method).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import accumulate
from pathlib import Path

from procedures.text.exclusions import history_regions
from procedures.corpus.revision_text import build_view, collapse
from procedures.text.sentences import segment
from procedures.readers import Cell, load_panel, request_hash, run_cells
from procedures.readers.tokens import estimate_message_tokens, estimate_tokens, token_method
from procedures.scoring.parse import TRUNCATION_REASONS, strip_think

from procedures.memory.chunking import sentences
from procedures.corpus.revisions import RevisionText

SUBBLOCK_PROMPT = """You are building a reference memory for an assistant that will later answer questions about a \
document without seeing the document itself. The document is summarised part by part, in document order; below is one \
part of one section of it. Summarise THIS part as a list of at most {bullets} bullets.

Format:
- Write at most {bullets} bullets, each on its own line starting with "- ", each at most {bullet_words} words.
- One bullet per rule or condition, in the order the part states them.
- Cover the whole part: if it states more rules than you have bullets, give one short bullet to each of the most \
consequential rules (thresholds, deadlines, conditions, exceptions) and combine closely related rules, rather than \
detailing only the first ones.
- If the part states no rule, write one bullet saying what it covers.
- Skip the table of contents, page headers and footers, and background description.
- No prose, no preamble, no headings and no closing remarks: the bullets only.

Content:
- Keep each rule together with its condition (when, if, unless, except, otherwise, in the event of) and each branch \
with its own outcome.
- Keep every threshold, limit, percentage and unit, every duration and deadline, every exception and every effective \
date verbatim, exactly as the part states it.
- Use the document's own defined terms and role names.
- Never invent, estimate, round or convert a value; if the part does not state a value, do not supply one.

Document: {header}
Section: {section}
Part: {part}
<part>
{text}
</part>"""
TOKENS_PER_BULLET = 28          # N = max(MIN_BULLETS, target / this rounded half up); sub-block target = this * k_i
BULLET_WORDS = 25               # words per bullet at most
MIN_BULLETS = 3
QUOTA_SLACK = 1                 # the check tolerates quota + this bullets
PROMPT_SHA = hashlib.sha256(json.dumps({"template": SUBBLOCK_PROMPT, "tokens_per_bullet": TOKENS_PER_BULLET,
                                        "bullet_words": BULLET_WORDS, "min_bullets": MIN_BULLETS},
                                       sort_keys=True).encode("utf-8")).hexdigest()
SECTION_MAX_TOKENS = 12_000     # n = ceil(T / this): sections of ~T/n tokens, at most this before snapping to a heading
PART_MAX_TOKENS = 2_000         # the same inside a section: its sub-blocks
SNAP = 0.15                     # a cut may move this share of a section's size to reach a heading / paragraph start
HISTORY_SPAN_MIN = 4_000        # a revision-history run this long gets sections of its own
TARGET_FLOOR = 120
OUTPUT_CAP_FACTOR = 2.0         # cap level 1 (1.25 was too tight): the builder's bullets run p50 32 / p90 50 tokens
ESCALATED_CAP_FACTOR = 5.0      # cap level 2: ceil(5.0 x 28 x k_i) + ESCALATED_CAP_EXTRA = ceil(140 k_i) + 112
ESCALATED_CAP_EXTRA = 112
CAP_LEVELS = (1, 2)             # no level 3
CAP_ESCALATION = (f"level 1: max_output_tokens ceil({OUTPUT_CAP_FACTOR} x {TOKENS_PER_BULLET} x quota); level 2, only "
                  f"after a truncated level-1 answer: the same prompt at ceil({ESCALATED_CAP_FACTOR} x "
                  f"{TOKENS_PER_BULLET} x quota) + {ESCALATED_CAP_EXTRA}; no level 3; the memory uses the highest "
                  "level that finished stop, else the highest level answered")
LEADER = re.compile(r"\s*(?:\.\s?){4,}\s*\d*\s*$|\s+\.\s*\d+\s*$")   # a TOC dot leader + page number ending a title
BULLET = re.compile(r"^\s*(?:[-*•◦–]|\d+[.)])\s+(.*\S)")          # a bullet line: marker, space, text
SENTENCE_END = re.compile(r"[.!?][\"'”’)\]]*\s*$")                     # a line that ends a sentence
STAGE = "build/M5"
REASONING_FIELD_FLAG = "text_from_reasoning_field"   # a sub-block / compression answered from the reasoning field
REASONING_FIELD_WARN_SHARE = 0.10   # the analysis's 10% line for empty answers, reported for builds; voids nothing
REASONING_FIELD_FINISH = frozenset({"stop", "end_turn"})   # finishes whose reasoning field is read
DRY_BUDGET_LINE = "[dry-run M5 memory for a {budget}-token budget]"   # the last line of a dry-run memory
PARAMS = {"section_max_tokens": SECTION_MAX_TOKENS, "snap": SNAP, "history_span_min": HISTORY_SPAN_MIN,
          "history_share": "target_floor", "target_floor": TARGET_FLOOR, "output_cap_factor": OUTPUT_CAP_FACTOR,
          "length_control": "subblock_quota", "part_max_tokens": PART_MAX_TOKENS,
          "tokens_per_bullet": TOKENS_PER_BULLET, "bullet_words": BULLET_WORDS, "min_bullets": MIN_BULLETS,
          "quota_slack": QUOTA_SLACK, "cut_bullet_rule": "drop_last_if_truncated_mid_sentence",
          "cap_escalation": CAP_ESCALATION}


@dataclass(frozen=True)
class BuilderLedger:
    """Where live builds go (M5 / M7 sub-blocks, M6 compressions): the run_cells call ledger file of the builder or
    compressor profile and its spend cap. `transport` and `sleep` are passed to run_cells unchanged (tests use a
    mocked transport; None = the client's own)."""
    path: Path
    cap_usd: float
    transport: object = None
    sleep: object = None


@dataclass(frozen=True)
class Section:
    index: int                  # 0-based position in the revision
    count: int                  # sections of the revision
    char_start: int             # LF-text span; a revision's sections tile its text
    char_end: int
    title: str                  # "" when no heading precedes or starts inside the section
    tokens: int                 # estimate_tokens of the span's text
    history_tokens: int = 0     # tokens of its revision-history sentences (procedures.text.exclusions rule)

    def label(self) -> str:
        return f"Section {self.index + 1}/{self.count}" + (f": {self.title}" if self.title else "")

    @property
    def revision_history(self) -> bool:
        return 2 * self.history_tokens >= self.tokens > 0

    @property
    def body_tokens(self) -> int:
        return max(0, self.tokens - self.history_tokens)


@dataclass(frozen=True)
class SubBlock(Section):
    """A sub-block of a section: index / count within the section; its span tiles the section with its siblings."""

    def label(self) -> str:
        return f"Part {self.index + 1}/{self.count}" + (f": {self.title}" if self.title else "")


@dataclass(frozen=True)
class BlockCall:
    revision: RevisionText
    section: Section
    block: SubBlock
    target: int                 # the section's target (plan)
    quota: int                  # bullets asked of this sub-block (k_i)
    keys: tuple[str, ...]       # block_key per cap level (CAP_LEVELS order)
    cells: tuple[Cell, ...]     # the request per cap level: one prompt, the level's output cap

    @property
    def key(self) -> str:
        """The sub-block's identity: its level-1 block key."""
        return self.keys[0]

    @property
    def cell(self) -> Cell:
        """The level-1 request."""
        return self.cells[0]

    @property
    def call_id(self) -> str:
        return self.cell.item_id

    @property
    def block_target(self) -> int:
        return TOKENS_PER_BULLET * self.quota


@dataclass(frozen=True)
class BlockBuild:
    call: BlockCall
    text: str                   # the summary as built (live: the answer used) or a placeholder (dry_run)
    mode: str                   # live | dry_run
    build_tokens: int           # live: summed over the levels sent; dry_run: level 1's estimate
    request_hash: str           # live: of the answer used; dry_run: of the level-1 request
    usd: float | None           # live: priced cost from the ledger, summed over the levels sent; dry_run: level 1's
                                # run_cells HIGH-scenario estimate
    finish_reason: str | None   # of the answer used
    output_tokens: int | None   # live: the provider's output_total of the answer used
    flags: tuple[str, ...]
    reason: str                 # dry_run: why no live build
    level: int | None = None    # live: cap level of the answer used
    levels: tuple = ()          # per cap level: its key, request and cap; live: each level answered, with its answer;
                                # dry_run: every planned level, with its HIGH estimate

    @property
    def bullets(self) -> int:
        return len(bullet_lines(self.text))

    @property
    def max_output_tokens(self) -> int:
        """The output cap of the answer used (dry_run: level 1's)."""
        return self.call.cells[(self.level or 1) - 1].generation["max_output_tokens"]


@dataclass(frozen=True)
class M5Memory:
    text: str
    tokens: int
    budget: int
    provenance: list
    builds: tuple[BlockBuild, ...]
    bodies: tuple[str, ...]     # each sub-block's summary after the cut-bullet rule, before fitting the budget
    kept: tuple[str, ...]       # each sub-block's summary as it stands in the memory (after fitting the budget)
    dropped: tuple[int, ...]    # bullets the cut-bullet rule dropped, per sub-block (0 or 1)
    flags: tuple[str, ...]

    @property
    def mode(self) -> str:
        return "dry_run" if any(b.mode == "dry_run" for b in self.builds) else "live"

    @property
    def section_count(self) -> int:
        return len({(b.call.revision, b.call.section.index) for b in self.builds})

    def trimmed(self) -> list[int]:
        """Tokens the budget fit cut per sub-block (the cut-bullet rule's drops are counted in `dropped`)."""
        return [estimate_tokens(b) - estimate_tokens(k) for b, k in zip(self.bodies, self.kept)]

    def revision_history(self) -> tuple[int, float]:
        """(tokens of revision-history text, share of the summed section targets given to revision-history
        sections)."""
        secs = {(b.call.revision, b.call.section.index): b.call for b in self.builds}.values()
        given = sum(c.target for c in secs if c.section.revision_history) / sum(c.target for c in secs)
        return sum(c.section.history_tokens for c in secs), round(given, 4)

    def cap_levels(self) -> dict[int, int]:
        """Sub-blocks per cap level of the answer used (live)."""
        return dict(sorted(Counter(b.level for b in self.builds).items()))

    def still_capped(self) -> int:
        """Sub-blocks whose answer used is truncated: at the top of the cap ladder (live)."""
        return sum(b.finish_reason in TRUNCATION_REASONS for b in self.builds)

    def checks(self) -> dict[str, int]:
        """The mechanical checks of a live build (module docstring); zero everywhere = all hold."""
        return {"subblocks_without_bullet": sum(b.bullets == 0 for b in self.builds),
                "subblocks_not_stop": sum(b.finish_reason != "stop" for b in self.builds),
                "subblocks_over_quota": sum(b.bullets > b.call.quota + QUOTA_SLACK for b in self.builds),
                "over_budget": int(self.tokens > self.budget)}

    def info(self) -> list[dict]:
        """One entry per sub-block call, in memory order."""
        out = []
        for b, kept, cut, dropped in zip(self.builds, self.kept, self.trimmed(), self.dropped):
            c, s, p = b.call, b.call.section, b.call.block
            out.append({"doc_id": c.revision.doc, "revision": c.revision.revision, "call_id": c.call_id,
                        "section_index": s.index, "section_count": s.count, "section_span": [s.char_start, s.char_end],
                        "section_title": s.title, "revision_history": s.revision_history, "target_tokens": c.target,
                        "subblock_index": p.index, "subblock_count": p.count, "span": [p.char_start, p.char_end],
                        "title": p.title, "subblock_tokens": p.tokens, "history_tokens": p.history_tokens,
                        "quota": c.quota, "block_target_tokens": c.block_target, "cap_level": b.level,
                        "max_output_tokens": b.max_output_tokens, "block_key": c.key, "levels": list(b.levels),
                        "mode": b.mode, "request_hash": b.request_hash, "build_tokens": b.build_tokens,
                        "usd": b.usd, "finish_reason": b.finish_reason, "output_tokens": b.output_tokens,
                        "bullets": b.bullets, "summary_tokens": estimate_tokens(b.text),
                        "dropped_bullets": dropped, "kept_tokens": estimate_tokens(kept), "trimmed_tokens": cut,
                        "flags": list(b.flags),
                        "reason": b.reason})
        return out


@lru_cache(maxsize=1)
def panel():
    """The pinned panel (procedures.readers.load_panel), loaded once per process."""
    return load_panel()


def bullet_lines(text: str) -> list[str]:
    """The bodies of the bullet lines of a summary (a line opening with -, *, •, ◦, – or 1. / 1) and a space)."""
    return [m.group(1) for m in map(BULLET.match, text.splitlines()) if m]


def drop_cut_bullet(text: str, finish_reason) -> tuple[str, int]:
    """(text, bullets dropped): a truncated answer (finish in TRUNCATION_REASONS) whose last line does not end at a
    sentence boundary loses its last bullet, from that bullet's line to the end; anything else, or an
    answer without a bullet line, is kept as is."""
    lines = text.rstrip().splitlines()
    starts = [i for i, line in enumerate(lines) if BULLET.match(line)]
    if finish_reason not in TRUNCATION_REASONS or not starts or SENTENCE_END.search(lines[-1]):
        return text, 0
    return "\n".join(lines[:starts[-1]]).rstrip(), 1


# ------------------------------------------------------------------------------------------------ sections
@dataclass(frozen=True)
class _Layout:
    """A revision's sentences as sectioning units: sentence i owns text[edges[i]:edges[i + 1]]."""
    text: str
    edges: tuple[int, ...]
    sizes: tuple[int, ...]
    history: tuple[bool, ...]
    heads: tuple[tuple[int, str], ...]
    head_at: frozenset[int]
    paras: frozenset[int]

    def cut(self, a: int, b: int, max_tokens: int) -> list[int]:
        """Sentence indices ending the pieces of sentences a..b-1 cut into ceil(T / max_tokens) (the last is b)."""
        pos = [0, *accumulate(self.sizes[a:b])][:-1]       # tokens before sentence a + i within the range
        total = sum(self.sizes[a:b])
        cuts = _cuts(pos, total, max(1, math.ceil(total / max_tokens)),
                     {i - a for i in self.head_at if a < i < b}, {i - a for i in self.paras if a < i < b})
        return [a + c for c in cuts] + [b]

    def piece(self, cls, k: int, bounds: list[int]):
        a, b = bounds[k], bounds[k + 1]
        lo, hi = self.edges[a], self.edges[b]
        return cls(k, len(bounds) - 1, lo, hi, _title(list(self.heads), a, b), estimate_tokens(self.text[lo:hi]),
                   sum(n for n, h in zip(self.sizes[a:b], self.history[a:b]) if h))


def _headings(rev: RevisionText, sents) -> list[tuple[int, str]]:
    """(index of the sentence the heading line starts or sits in, title), document order."""
    starts = [s.start for s in sents]
    out = []
    for offset, key, title in build_view(Path(rev.source_path), rev.family, rev.text()).headings:
        title = LEADER.sub("", collapse(title))
        if key == "front-matter" or not title:
            continue
        i = bisect_right(starts, offset) - 1
        if i < 0:
            i = 0
        elif sents[i].end <= offset:
            i += 1
        if i < len(sents):
            out.append((i, title))
    return out


def _cuts(pos: list[int], total: int, n: int, heads: set[int], paras: set[int]) -> list[int]:
    """Indices of the sentences that start pieces 2..n (module docstring)."""
    cuts: list[int] = []
    size = total / n
    for k in range(1, n):
        ideal, room = k * size, SNAP * size
        first = cuts[-1] + 1 if cuts else 1
        window = range(max(first, bisect_left(pos, ideal - room)), min(len(pos), bisect_right(pos, ideal + room)))
        for pool in ([i for i in window if i in heads], [i for i in window if i in paras], list(window),
                     list(range(first, len(pos)))):
            if pool:
                cuts.append(min(pool, key=lambda i: (abs(pos[i] - ideal), i)))
                break
    return cuts


def _title(heads: list[tuple[int, str]], a: int, b: int) -> str:
    before = [(i, t) for i, t in heads if i <= a]
    inside = [t for i, t in heads if a < i < b]
    first = ""
    if before:
        i, t = before[-1]
        first = t if i == a else f"{t} (continued)"
    elif inside:
        first, inside = inside[0], inside[1:]
    return first + (f" to {inside[-1]}" if inside else "")


def history_flags(rev: RevisionText) -> list[bool]:
    """Per sentence of `sentences(rev)`: inside revision history by procedures.text.exclusions' rule (the sentence start
    lies in one of history_regions' regions)."""
    regions = history_regions(rev.text(), rev.family)
    return [any(a <= s.start < b for a, b in regions) for s in sentences(rev)]


def _spans(history: list[bool], sizes: list[int]) -> list[tuple[int, int]]:
    """Sentence ranges sectioned separately: each history run of >= HISTORY_SPAN_MIN tokens, and the body between
    them; a span under HISTORY_SPAN_MIN tokens joins its predecessor (the first one its successor)."""
    spans: list[list] = []                              # [lo, hi, own history span]
    lo = 0
    for i in range(1, len(history) + 1):
        if i < len(history) and history[i] == history[lo]:
            continue
        own = history[lo] and sum(sizes[lo:i]) >= HISTORY_SPAN_MIN
        if spans and not own and not spans[-1][2]:
            spans[-1][1] = i
        else:
            spans.append([lo, i, own])
        lo = i
    k = 0
    while len(spans) > 1 and k < len(spans):
        if sum(sizes[spans[k][0]:spans[k][1]]) >= HISTORY_SPAN_MIN:
            k += 1
        elif k:
            spans[k - 1][1] = spans.pop(k)[1]
        else:
            spans[1][0] = spans.pop(0)[0]
    return [(a, b) for a, b, _ in spans]


@lru_cache(maxsize=64)
def _layout(rev: RevisionText) -> _Layout | None:
    text = rev.text()
    sents = sentences(rev)
    if not sents:
        return None
    edges = (0, *[s.start for s in sents[1:]], len(text))
    heads = _headings(rev, sents)
    return _Layout(text, edges, tuple(estimate_tokens(text[a:b]) for a, b in zip(edges, edges[1:])),
                   tuple(history_flags(rev)), tuple(heads), frozenset(i for i, _ in heads),
                   frozenset(i for i in range(1, len(sents)) if sents[i].block != sents[i - 1].block))


@lru_cache(maxsize=64)
def sections(rev: RevisionText) -> tuple[Section, ...]:
    """The revision's sections in document order (module docstring)."""
    lay = _layout(rev)
    if lay is None:
        text = rev.text()
        return (Section(0, 1, 0, len(text), "", estimate_tokens(text)),)
    bounds = [0]
    for a, b in _spans(list(lay.history), list(lay.sizes)):
        bounds += lay.cut(a, b, SECTION_MAX_TOKENS)
    return tuple(lay.piece(Section, k, bounds) for k in range(len(bounds) - 1))


@lru_cache(maxsize=1024)
def subblocks(rev: RevisionText, section: Section) -> tuple[SubBlock, ...]:
    """The section's sub-blocks in document order (module docstring)."""
    lay = _layout(rev)
    if lay is None:
        return (SubBlock(0, 1, section.char_start, section.char_end, section.title, section.tokens,
                         section.history_tokens),)
    a, b = lay.edges.index(section.char_start), lay.edges.index(section.char_end)
    bounds = [a, *lay.cut(a, b, PART_MAX_TOKENS)]
    return tuple(lay.piece(SubBlock, k, bounds) for k in range(len(bounds) - 1))


# ------------------------------------------------------------------------------------------------ requests
def bullet_count(target: int) -> int:
    """A section's bullets N for its `target` tokens: max(MIN_BULLETS, target / TOKENS_PER_BULLET rounded half up)."""
    return max(MIN_BULLETS, (target + TOKENS_PER_BULLET // 2) // TOKENS_PER_BULLET)


def quota_split(n: int, weights: list[int]) -> list[int]:
    """k_i = max(1, n * w_i / sum(w) rounded half up); all 1 when every weight is 0."""
    total = sum(weights)
    if total <= 0:
        return [1] * len(weights)
    return [max(1, (2 * n * w + total) // (2 * total)) for w in weights]


def block_weights(section: Section, blocks: tuple[SubBlock, ...]) -> list[int]:
    """The sub-blocks' non-history tokens; all their tokens in a revision-history section or a section without
    non-history tokens."""
    body = [p.body_tokens for p in blocks]
    return [p.tokens for p in blocks] if section.revision_history or not sum(body) else body


def block_cap(quota: int, level: int = 1) -> int:
    """A sub-block request's output cap at cap level 1, ceil(OUTPUT_CAP_FACTOR * TOKENS_PER_BULLET * quota), or 2,
    ceil(ESCALATED_CAP_FACTOR * TOKENS_PER_BULLET * quota) + ESCALATED_CAP_EXTRA."""
    if level == 1:
        return math.ceil(OUTPUT_CAP_FACTOR * TOKENS_PER_BULLET * quota)
    if level == 2:
        return math.ceil(ESCALATED_CAP_FACTOR * TOKENS_PER_BULLET * quota) + ESCALATED_CAP_EXTRA
    raise ValueError(f"cap level must be one of {CAP_LEVELS}, got {level!r}")


def block_key(rev: RevisionText, section: Section, block: SubBlock, builder_profile: str, budget: int, target: int,
              quota: int, level: int = 1) -> str:
    fields = {"doc_id": rev.doc, "revision": rev.revision, "sha256_lf": rev.sha256_lf,
              "builder_profile": builder_profile, "budget": budget, "section_index": section.index,
              "section_span": [section.char_start, section.char_end], "revision_history": section.revision_history,
              "target_tokens": target, "subblock_index": block.index,
              "subblock_span": [block.char_start, block.char_end], "quota": quota, "cap_level": level,
              "max_output_tokens": block_cap(quota, level), "prompt_sha": PROMPT_SHA, "token_method": token_method()}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode("utf-8")).hexdigest()


def block_cell(rev: RevisionText, section: Section, block: SubBlock, quota: int, level: int = 1) -> Cell:
    """The sub-block's request at cap `level`: the same prompt at every level. A level-2 request follows an answer that
    ran to the level-1 cap, so it is estimated at that many output tokens; its record is labelled cap_level 2."""
    prompt = SUBBLOCK_PROMPT.format(bullets=quota, bullet_words=BULLET_WORDS, header=rev.header(),
                                    section=section.label(), part=block.label(),
                                    text=rev.text()[block.char_start:block.char_end])
    target = TOKENS_PER_BULLET * quota
    return Cell(item_id=f"{rev.doc}@{rev.revision}#{section.index}.{block.index}", stage=f"{STAGE}@{target}",
                messages=({"role": "user", "content": prompt},),
                generation={"max_output_tokens": block_cap(quota, level)},
                est_visible_tokens=target if level == 1 else block_cap(quota, level - 1),
                labels={} if level == 1 else {"cap_level": level})


def parse_call_id(call_id: str) -> tuple[str, str, int, int]:
    """(doc_id, revision, section index, sub-block index) of a request's item_id "{doc}@{revision}#{section}.{part}"."""
    head, _, index = call_id.rpartition("#")
    doc, _, revision = head.rpartition("@")
    section, dot, part = index.partition(".")
    if not doc or not revision or not dot or not section.isdigit() or not part.isdigit():
        raise ValueError(f"{call_id!r} is not an M5 sub-block call id")
    return doc, revision, int(section), int(part)


def render(parts: list[tuple[RevisionText, list[tuple[Section, str]]]]) -> str:
    """The memory text: per revision its header line, then per section its header line and body."""
    return "\n\n".join(f"[{rev.header()}]\n" + "\n\n".join(f"## {s.label()}\n{body}" for s, body in secs)
                       for rev, secs in parts)


def _parts(calls: list[BlockCall], bodies) -> list[tuple[RevisionText, list[tuple[Section, str]]]]:
    """Per revision, per section: the non-empty sub-block bodies joined line by line."""
    parts: list[tuple[RevisionText, list[tuple[Section, list[str]]]]] = []
    for call, body in zip(calls, bodies):
        if not parts or parts[-1][0] != call.revision:
            parts.append((call.revision, []))
        secs = parts[-1][1]
        if not secs or secs[-1][0] != call.section:
            secs.append((call.section, []))
        if body:
            secs[-1][1].append(body)
    return [(rev, [(s, "\n".join(bodies)) for s, bodies in secs]) for rev, secs in parts]


def plan(revisions: list[RevisionText], budget: int, builder_profile: str) -> list[BlockCall]:
    """Every sub-block request of the memory: revisions in the given (ladder) order, sections and their sub-blocks in
    document order. Planned once per process per revision set, budget and builder (every item reading those
    revisions has the same plan)."""
    return list(_plan(tuple(revisions), budget, builder_profile))


@lru_cache(maxsize=512)
def _plan(revisions: tuple[RevisionText, ...], budget: int, builder_profile: str) -> tuple[BlockCall, ...]:
    secs = [(rev, sections(rev)) for rev in revisions]
    frame = estimate_tokens(render([(rev, [(s, "") for s in ss]) for rev, ss in secs]))
    share = (budget - frame) // len(revisions)
    if share < 1:
        raise ValueError(f"budget {budget} leaves no room for a summary after the revision and section headers "
                         f"({frame} tokens)")
    calls = []
    for rev, ss in secs:
        rest = share - TARGET_FLOOR * sum(s.revision_history for s in ss)     # history sections get the floor only
        body = sum(s.body_tokens for s in ss if not s.revision_history)
        for s in ss:
            weighted = round(rest * s.body_tokens / body) if body else rest
            target = min(share, TARGET_FLOOR if s.revision_history else max(TARGET_FLOOR, weighted))
            blocks = subblocks(rev, s)
            for p, quota in zip(blocks, quota_split(bullet_count(target), block_weights(s, blocks))):
                calls.append(BlockCall(rev, s, p, target, quota,
                                       tuple(block_key(rev, s, p, builder_profile, budget, target, quota, level)
                                             for level in CAP_LEVELS),
                                       tuple(block_cell(rev, s, p, quota, level) for level in CAP_LEVELS)))
    return tuple(calls)


# ------------------------------------------------------------------------------------------------ builds
def _request(pp, cell: Cell) -> tuple[str, tuple]:
    sent = pp.messages_to_send(cell.messages)
    return request_hash(pp.profile_id, pp.fingerprint, sent, cell.generation, cell.repeat_idx,
                        user_message_suffix=pp.user_message_suffix), sent


def dry_reason(pp, ledger: BuilderLedger | None) -> str:
    """Why a build of profile `pp` into `ledger` is a dry run ("" when it goes live)."""
    if ledger is None:
        return "no ledger"
    env = pp.client_profile().api_key_env
    if env and not os.environ.get(env):
        return f"credential variable {env} is unset"
    return ""


def escalates(record: dict | None) -> bool:
    """Whether an answer sends its sub-block's request up the cap ladder: an observed answer whose finish is a
    truncation (procedures.scoring.parse.TRUNCATION_REASONS)."""
    return (record is not None and record.get("outcome") == "observed"
            and record.get("finish_reason") in TRUNCATION_REASONS)


def answer_text(record: dict, pp) -> tuple[str, bool]:
    """(the answer of a build or compression record, think blocks stripped and whitespace trimmed; whether it was read
    from the reasoning field): the content, or, for a profile whose panel entry sets answer_from_reasoning_field and
    a content blank after the strip, the reasoning field `reasoning_text` stripped the same way when that is not blank
    and the record's finish_reason is in REASONING_FIELD_FINISH (module docstring); otherwise ""."""
    text = strip_think(record.get("text") or "")[0].strip()
    if text or not pp.answer_from_reasoning_field or record.get("finish_reason") not in REASONING_FIELD_FINISH:
        return text, False
    reasoning = strip_think(record.get("reasoning_text") or "")[0].strip()
    return reasoning, bool(reasoning)


def reasoning_field_report(builds) -> dict:
    """How many of `builds` (live BlockBuilds of one memory) were answered from the reasoning field, their share, and
    whether it exceeds REASONING_FIELD_WARN_SHARE (a warning; nothing is voided)."""
    mapped = sum(REASONING_FIELD_FLAG in b.flags for b in builds)
    share = round(mapped / len(builds), 4) if builds else 0.0
    return {"text_from_reasoning_field": mapped, "subblocks": len(builds), "share": share,
            "over_warn_share": share > REASONING_FIELD_WARN_SHARE}


def _level(level: int, key: str, cell: Cell, hash_: str) -> dict:
    return {"cap_level": level, "block_key": key, "request_hash": hash_,
            "max_output_tokens": cell.generation["max_output_tokens"]}


def _dry(call: BlockCall, pp, cap: float, reason: str) -> BlockBuild:
    levels = []
    for level, key, cell in zip(CAP_LEVELS, call.keys, call.cells):
        hash_, sent = _request(pp, cell)
        levels.append({**_level(level, key, cell, hash_), "mode": "dry_run",
                       "usd": run_cells([cell], pp, cap_usd=cap, dry_run=True)["usd"]})
        if level == 1:
            build_tokens = estimate_message_tokens(sent) + call.block_target
    placeholder = (f"[dry-run placeholder: no summary of {call.revision.header()}, {call.section.label()}, "
                   f"{call.block.label()} was built (target {call.block_target} tokens)]")
    return BlockBuild(call, placeholder, "dry_run", build_tokens, levels[0]["request_hash"], levels[0]["usd"], None,
                      None, (), reason, None, tuple(levels))


def answers_from_records(calls: list[BlockCall], pp, records: dict, where) -> list[BlockBuild]:
    """Each sub-block's answer read back from its ledger records (request hash -> latest record): level 1, then each
    further cap level while the previous answer escalates; the answer used is the highest level that finished "stop",
    else the highest level answered. Shared with procedures.memory.records, whose calls are BlockCall's with their own
    prompt and caps (the message names the build stage, e.g. "build/M5")."""
    out = []
    for call in calls:
        what = (f"{call.cell.stage.split('@', 1)[0]} of {call.revision.header()}, {call.section.label()}, "
                f"{call.block.label()}")
        answered = []                                   # (level entry, record, messages as sent)
        for level, key, cell in zip(CAP_LEVELS, call.keys, call.cells):
            if answered and not escalates(answered[-1][1]):
                break
            hash_, sent = _request(pp, cell)
            record = records.get(hash_)
            if record is None:
                raise RuntimeError(f"{what}: no ledger record for the cap level {level} request {hash_} in {where}")
            if record.get("outcome") != "observed":
                raise RuntimeError(f"{what} got no answer at cap level {level} ({record.get('outcome')}); see {where}")
            answered.append((_level(level, key, cell, hash_), record, sent))
        stops = [a for a in answered if a[1].get("finish_reason") == "stop"]
        used, record, _ = (stops or answered)[-1]
        text, mapped = answer_text(record, pp)
        if not text:
            raise RuntimeError(f"{what} returned an empty answer at cap level {used['cap_level']}; see {where}")
        flags = {"build_truncated"} if record.get("finish_reason") in TRUNCATION_REASONS else set()
        if mapped:
            flags.add(REASONING_FIELD_FLAG)
        levels = []
        for entry, rec, sent in answered:
            usage = rec.get("usage") or {}
            if usage.get("input_total") is None or usage.get("output_total") is None:
                flags.add("build_tokens_estimated")
                tokens = estimate_message_tokens(sent) + estimate_tokens(answer_text(rec, pp)[0])
            else:
                tokens = usage["input_total"] + usage["output_total"]
            levels.append({**entry, "mode": "live", "usd": rec.get("usd"), "finish_reason": rec.get("finish_reason"),
                           "output_tokens": usage.get("output_total"), "build_tokens": tokens,
                           "imported_from": rec.get("imported_from")})
        usd = None if any(v["usd"] is None for v in levels) else sum(v["usd"] for v in levels)
        out.append(BlockBuild(call, text, "live", sum(v["build_tokens"] for v in levels), used["request_hash"], usd,
                              record.get("finish_reason"), (record.get("usage") or {}).get("output_total"),
                              tuple(sorted(flags)), "", used["cap_level"], tuple(levels)))
    return out


def trim_to_tokens(text: str, limit: int) -> str:
    """The longest sentence-boundary prefix of `text` within `limit` tokens ("" if the first sentence is longer)."""
    if estimate_tokens(text) <= limit:
        return text
    ends = [s.end for s in segment(text)]
    lo, hi = 0, len(ends)          # invariant: prefix up to ends[lo - 1] fits (lo = 0: the empty prefix)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:ends[mid - 1]].rstrip()) <= limit:
            lo = mid
        else:
            hi = mid - 1
    return text[:ends[lo - 1]].rstrip() if lo else ""


def fit(calls: list[BlockCall], bodies: list[str], budget: int) -> list[str]:
    """The bodies as they go into the memory: unchanged when the rendered memory fits the budget; otherwise each capped
    at rho/1000 x its sub-block target at a sentence boundary, rho the largest that fits (module docstring)."""
    def fits(kept: list[str]) -> bool:
        return estimate_tokens(render(_parts(calls, kept))) <= budget

    def capped(rho: int) -> list[str]:
        return [trim_to_tokens(b, rho * c.block_target // 1000) for b, c in zip(bodies, calls)]

    if fits(bodies):
        return list(bodies)
    lo = 0                          # all bodies empty: the frame, which plan() checked fits
    hi = max(math.ceil(1000 * estimate_tokens(b) / c.block_target) for b, c in zip(bodies, calls))  # nothing cut
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(capped(mid)):
            lo = mid
        else:
            hi = mid - 1
    return capped(lo)


def assemble(calls: list[BlockCall], builds: list[BlockBuild], budget: int) -> M5Memory:
    cut = [drop_cut_bullet(b.text, b.finish_reason) for b in builds]
    bodies, dropped = [t for t, _ in cut], tuple(n for _, n in cut)
    kept = fit(calls, bodies, budget)
    text = render(_parts(calls, kept))
    tokens = estimate_tokens(text)
    if tokens > budget:
        raise RuntimeError(f"M5 memory of {tokens} tokens exceeds budget {budget}")
    flags = ({f for b in builds for f in b.flags} | ({"trimmed_to_budget"} if kept != bodies else set())
             | ({"dropped_cut_bullets"} if any(dropped) else set()))
    provenance = [{"doc_id": c.revision.doc, "revision": c.revision.revision, "path": c.revision.source_path,
                   "char_start": c.block.char_start, "char_end": c.block.char_end} for c in calls]
    return M5Memory(text, tokens, budget, provenance, tuple(builds), tuple(bodies), tuple(kept), dropped,
                    tuple(sorted(flags)))


def build_blocks(calls: list[BlockCall], builder_profile: str, ledger: BuilderLedger | None) -> list[BlockBuild]:
    """The sub-block builds of `calls` (a plan, or part of one), one run_cells call per cap level (level 1 for every
    sub-block, then level 2 for those whose level-1 answer escalates): live into the ledger, or dry_run when there is
    no ledger or the builder's credential variable is unset (module docstring)."""
    pp = panel().get(builder_profile)
    reason = dry_reason(pp, ledger)
    if reason:
        cap = ledger.cap_usd if ledger is not None else math.inf
        return [_dry(c, pp, cap, reason) for c in calls]
    records: dict[str, dict] = {}
    climbing = list(calls)
    for i in range(len(CAP_LEVELS)):
        if i:
            climbing = [c for c in climbing if escalates(records.get(_request(pp, c.cells[i - 1])[0]))]
        if not climbing:
            break
        report = run_cells([c.cells[i] for c in climbing], pp, cap_usd=ledger.cap_usd, ledger_path=Path(ledger.path),
                           transport=ledger.transport, sleep=ledger.sleep)
        records.update((r["request_hash"], r) for r in report["records"])
    return answers_from_records(calls, pp, records, ledger.path)


def build_m5(revisions: list[RevisionText], budget: int, builder_profile: str,
             ledger: BuilderLedger | None) -> M5Memory:
    if ledger is None:
        return _dry_m5(tuple(revisions), budget, builder_profile)
    calls = plan(revisions, budget, builder_profile)
    return assemble(calls, build_blocks(calls, builder_profile, ledger), budget)


@lru_cache(maxsize=512)
def _dry_m5(revisions: tuple[RevisionText, ...], budget: int, builder_profile: str) -> M5Memory:
    """build_m5 without a ledger (a dry run, nothing sent): the same for every item reading these revisions, so planned
    and priced once per process. The assembled placeholders end with DRY_BUDGET_LINE, so the dry-run memories of one
    revision at two budgets are two texts even when their plans coincide (a tiny revision gets the same sub-block
    targets at every budget), and the reader requests priced on them stay distinct."""
    calls = plan(list(revisions), budget, builder_profile)
    memory = assemble(calls, build_blocks(calls, builder_profile, None), budget)
    text = f"{memory.text}\n{DRY_BUDGET_LINE.format(budget=budget)}"
    return replace(memory, text=text, tokens=estimate_tokens(text))
