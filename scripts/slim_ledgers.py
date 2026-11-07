"""Write the released form of a run's call ledgers (the per-call API logs).

A raw call ledger (read__<reader>.jsonl, build__<builder>.jsonl, query__<compressor>.jsonl) records every API call in
full: the prompt sent (which carries the procedure text the reader or builder saw), the provider's raw response and the
model's output. The procedures' owners allow their text to be redistributed as pointers only (the corpus manifest's
`terms_flag` column), so a released ledger keeps every field the analysis and the scorer read (usage, price, status,
finish reason, timing, the hashes that tie a call to its row, and the scored part of each answer) and none of the
procedure text:

* Every call drops the prompt (request `messages`; an Anthropic request's `system`) and the provider's copy of the
  output (raw_response `choices`; an Anthropic raw_response keeps only id, model, stop_reason and usage).
* A build or compression call replaces its output, which restates the procedure at length, by its length and sha256
  (`text_chars`, `text_sha256`; `raw_text` and `reasoning_text` set to null).
* A read call keeps the reader's answer as the scorer reads it and withholds the quotes and the working around it. In
  each answer object (a JSON object the scorer finds in the answer, procedures.scoring.parse) the `value` and `unit`
  members stay and every other string member is set to "": the `evidence_span` quote the prompt asks for, and any
  member a reader added. An answer object the reader left unclosed (cut off at its token limit, or malformed) keeps
  its opening `value` and `unit` members and the rest of it becomes TEXT_WITHHELD. Any other text of the answer (a
  reader's working before or after its answer object) becomes TEXT_WITHHELD; whitespace, stray punctuation and
  markdown code fences stay. The reasoning field is treated the same way when it holds an answer object (the readers
  whose panel entry sets answer_from_reasoning_field put their answer there, and the scorer reads it); any other
  reasoning text becomes REASONING_WITHHELD, and a blank one stays as it is. What was withheld is recorded by length
  and sha256: `evidence_span_chars` / `evidence_span_sha256` (the answer object's evidence_span as the reader wrote it,
  on every record whose answer object has one), `text_chars` / `text_sha256` (the answer as returned) and
  `reasoning_chars` / `reasoning_sha256` (the reasoning text as returned), the last two pairs on the records whose
  field was changed.

Each released read record is checked against its source: the scorer's extraction of the answer (status, number of
JSON objects, think-stripping, the chosen object's members and its value and unit), its reading of the reasoning
field, and whether answer and reasoning are blank are unchanged. The header line and the record order are unchanged,
so `procedures.readers.ledger.read_ledger` reads a released ledger as it reads a raw one. A ledger is written as
<target>/<name>.gz (gzip, no timestamp), and a released ledger whose content is already the slimmed one is left as it
is.

    py -3.12 scripts/slim_ledgers.py --source <raw run dir> --target <released run dir>
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from json.decoder import WHITESPACE, scanstring
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from procedures import stored  # noqa: E402
from procedures.readers.ledger import read_ledger  # noqa: E402
from procedures.readers.parse import answer_from_reasoning  # noqa: E402
from procedures.scoring import extract_answer  # noqa: E402
from procedures.scoring.parse import strip_think  # noqa: E402
from procedures.scoring.score_path import reasoning_only  # noqa: E402

TEXT_WITHHELD = "[text withheld]"
REASONING_WITHHELD = "[reasoning text withheld]"
SCORED_MEMBERS = ("value", "unit")      # the members of an answer object the scorer reads
EVIDENCE = "evidence_span"
DROPPED_REQUEST = ("messages", "system")                    # the prompt
ANTHROPIC_RAW_RESPONSE = ("id", "model", "stop_reason", "usage")
ADAPTERS = ("openai_chat", "anthropic_messages")

_DECODER = json.JSONDecoder()
_STRING = r'"(?:[^"\\]|\\.)*"'
_TOKEN = r'[^\s,"{}][^,"{}]*?'          # a bare value: a number, null, or a malformed expression
# the opening value and unit members of an answer object that does not parse
_UNCLOSED_HEAD = re.compile(r'\{\s*"value"\s*:\s*(?:%s|%s)\s*,\s*"unit"\s*:\s*(?:%s|%s)\s*(?:,\s*|\Z)'
                            % (_STRING, _TOKEN, _STRING, _TOKEN))
_FENCE = re.compile(r"```\w*")


def _length_and_sha256(text: str) -> tuple[int, str]:
    return len(text), hashlib.sha256(text.encode("utf-8")).hexdigest()


def _objects(text: str) -> list[tuple[int, int]]:
    """(start, end) of each JSON object the scorer finds in `text`, left to right
    (procedures.scoring.parse.json_objects)."""
    found, i = [], 0
    while (j := text.find("{", i)) >= 0:
        try:
            end = _DECODER.raw_decode(text, j)[1]
        except json.JSONDecodeError:
            i = j + 1
            continue
        found.append((j, end))
        i = end
    return found


def _members(text: str, start: int) -> list[tuple[str, int, int]]:
    """(key, value start, value end) of each member of the JSON object that opens at text[start] and parses."""
    out = []
    pos = WHITESPACE.match(text, start + 1).end()
    if text[pos] == "}":
        return out
    while True:
        key, pos = scanstring(text, pos + 1)
        pos = WHITESPACE.match(text, WHITESPACE.match(text, pos).end() + 1).end()     # past the ':'
        end = _DECODER.raw_decode(text, pos)[1]
        out.append((key, pos, end))
        pos = WHITESPACE.match(text, end).end()
        if text[pos] == "}":
            return out
        pos = WHITESPACE.match(text, pos + 1).end()                                   # past the ','


def _withhold_object(text: str, start: int, end: int, evidence: list[str]) -> str:
    """The answer object text[start:end] with every string member other than value and unit set to "" (its
    evidence_span, as the object parses, appended to `evidence`)."""
    parts, done = [], start
    for key, value_start, value_end in _members(text, start):
        if key in SCORED_MEMBERS:
            continue
        value = _DECODER.raw_decode(text, value_start)[0]
        if isinstance(value, (dict, list)):
            raise ValueError(f"an answer object member {key!r} holds a {type(value).__name__}: not handled")
        if isinstance(value, str) and value:
            parts += [text[done:value_start], '""']
            done = value_end
    parts.append(text[done:end])
    quoted = _DECODER.raw_decode(text, start)[0].get(EVIDENCE)
    if isinstance(quoted, str):
        evidence.append(quoted)
    return "".join(parts)


def _withhold_words(piece: str) -> str:
    """Answer text that is no answer object, with its words replaced by TEXT_WITHHELD (whitespace kept around it);
    whitespace, stray punctuation and markdown code fences alone stay."""
    if not re.search(r"\w", _FENCE.sub("", piece)):
        return piece
    lead, tail = len(piece) - len(piece.lstrip()), len(piece.rstrip())
    return piece[:lead] + TEXT_WITHHELD + piece[tail:]


def _withhold_outside(piece: str) -> str:
    """Answer text outside the answer objects: an unclosed answer object keeps its opening value and unit members and
    the rest of it is withheld; any other words are withheld."""
    head = _UNCLOSED_HEAD.search(piece)
    if head is None:
        return _withhold_words(piece)
    rest = piece[head.end():]
    return _withhold_words(piece[:head.start()]) + head.group(0) + (TEXT_WITHHELD if rest.strip() else rest)


def withhold_answer(text: str) -> tuple[str, list[str]]:
    """(the answer as released, the evidence_span strings of its answer objects): module docstring."""
    if strip_think(text)[1]:
        raise ValueError("an answer with a think block: not handled")
    out, evidence, done = [], [], 0
    for start, end in _objects(text):
        out += [_withhold_outside(text[done:start]), _withhold_object(text, start, end, evidence)]
        done = end
    out.append(_withhold_outside(text[done:]))
    return "".join(out), evidence


def slim_read(record: dict) -> dict:
    """A read call record (prompt and provider payload already dropped) with its answer and reasoning text in the
    released form, and the length and sha256 of what was withheld."""
    if record.get("raw_text") is not None:
        raise ValueError(f"{record['request_hash']}: a read record with raw_text: not handled")
    out, evidence, added = dict(record), [], {}
    text = record.get("text")
    if text is not None:
        released, found = withhold_answer(text)
        evidence += found
        if released != text:
            out["text"] = released
            added["text_chars"], added["text_sha256"] = _length_and_sha256(text)
    reasoning = record.get("reasoning_text")
    if reasoning is not None and reasoning.strip():
        if reasoning.lstrip().startswith("{"):
            released, found = withhold_answer(reasoning)
            evidence += found
        else:
            released = REASONING_WITHHELD
        if released != reasoning:
            out["reasoning_text"] = released
            added["reasoning_chars"], added["reasoning_sha256"] = _length_and_sha256(reasoning)
    if len(evidence) > 1:
        raise ValueError(f"{record['request_hash']}: {len(evidence)} answer objects with an evidence_span")
    if evidence:
        out["evidence_span_chars"], out["evidence_span_sha256"] = _length_and_sha256(evidence[0])
    out.update(added)
    return out


def _released_payload(raw: dict, adapter: str) -> dict:
    """The provider's raw response without its copy of the output: an OpenAI-format response without `choices`, an
    Anthropic one reduced to its id, model, stop reason and usage."""
    if adapter == "openai_chat":
        return {k: v for k, v in raw.items() if k != "choices"}
    return {k: v for k, v in raw.items() if k in ANTHROPIC_RAW_RESPONSE}


def slim(record: dict, kind: str, adapter: str) -> dict:
    if adapter not in ADAPTERS:
        raise ValueError(f"ledger of adapter {adapter!r}: its provider payload is not handled")
    out = dict(record)
    request = out.get("request")
    if isinstance(request, dict):
        out["request"] = {k: v for k, v in request.items() if k not in DROPPED_REQUEST}
    raw = out.get("raw_response")
    if isinstance(raw, dict):
        out["raw_response"] = _released_payload(raw, adapter)
    if kind == "read":
        return slim_read(out)
    for key in ("raw_text", "reasoning_text"):
        if key in out:
            out[key] = None
    text = out.pop("text", None)
    out["text_chars"], out["text_sha256"] = (None, None) if text is None else _length_and_sha256(text)
    return out


def scorer_reading(record: dict) -> tuple:
    """What the scorer and the analysis read of a read call record: the scorer's extraction of its answer and of the
    answer its reasoning field holds (procedures.readers.parse.answer_from_reasoning); whether the answer and the
    reasoning text are blank; whether the reasoning text is an answer object; and whether the record is reasoning-only
    (procedures.scoring.score_path.reasoning_only)."""
    def extraction(text: str | None) -> tuple:
        found = extract_answer(text, record.get("finish_reason"))
        answer = found.answer or {}
        return (found.status, found.n_objects, found.think_stripped, sorted(answer),
                [answer.get(k) for k in SCORED_MEMBERS])
    mapped = answer_from_reasoning(record)
    reasoning = record.get("reasoning_text") or ""
    return (extraction(record.get("text")), None if mapped is None else extraction(mapped),
            not (record.get("text") or "").strip(), not reasoning.strip(), reasoning.lstrip().startswith("{"),
            reasoning_only(record))


def gzip_bytes(data: bytes, name: str) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(filename=name, mode="wb", fileobj=buffer, compresslevel=9, mtime=0) as stream:
        stream.write(data)
    return buffer.getvalue()


def slim_ledger(source: Path, target_dir: Path) -> tuple[int, int, int, bool]:
    """(calls, source bytes, released bytes, whether <target_dir>/<name>.gz was written)."""
    header, records = read_ledger(source)
    kind = source.name.split("__", 1)[0]
    adapter = header["profile_spec"]["adapter"]
    lines = [json.dumps(header, ensure_ascii=False)]
    for record in records:
        released = slim(record, kind, adapter)
        if kind == "read" and scorer_reading(released) != scorer_reading(record):
            raise ValueError(f"{source.name} {record['request_hash']}: the released answer does not read as the "
                             "original")
        lines.append(json.dumps(released, ensure_ascii=False))
    data = ("\n".join(lines) + "\n").encode("utf-8")
    target = target_dir / (source.name + ".gz")
    written = not target.exists() or stored.read_bytes(target) != data
    if written:
        target.write_bytes(gzip_bytes(data, source.name))
    _, back = read_ledger(target)
    if [r["request_hash"] for r in back] != [r["request_hash"] for r in records]:
        raise ValueError(f"{target}: the released ledger does not list the source's calls in order")
    return len(records), source.stat().st_size, len(data), written


def same_run(source: Path, target: Path) -> bool:
    """Whether two run directories hold one run, whatever they are called: their manifests agree on every field but
    the schema id and the corpus manifest's recorded path (a release may relabel both; the corpus manifest's sha256
    must agree)."""
    a, b = (json.loads((Path(d) / "manifest.json").read_text(encoding="utf-8")) for d in (source, target))
    relabelled = ("schema", "corpus_manifest")
    return ({k: v for k, v in a.items() if k not in relabelled} == {k: v for k, v in b.items() if k not in relabelled}
            and (a.get("corpus_manifest") or {}).get("sha256") == (b.get("corpus_manifest") or {}).get("sha256"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", type=Path, required=True, help="the run directory holding the raw call ledgers")
    ap.add_argument("--target", type=Path, required=True,
                    help="the released run directory (a directory holding a run manifest must hold the same run)")
    a = ap.parse_args(argv)
    if (a.target / "manifest.json").is_file() and not same_run(a.source, a.target):
        raise SystemExit(f"{a.source} and {a.target} are not the same run (their manifests differ)")
    a.target.mkdir(parents=True, exist_ok=True)
    for path in sorted(a.source.glob("*__*.jsonl")):
        n, before, after, written = slim_ledger(path, a.target)
        print(f"{path.name}: {n} calls, {before / 1e6:.1f} MB -> {after / 1e6:.1f} MB"
              f"{'' if written else ' (already released in this form)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
