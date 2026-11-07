"""Append-only JSONL call ledger with resume and a hard spend cap that holds across threads and processes.

One ledger file = one run of one profile. Line 1 is a header binding the run
identity (profile id + semantic fingerprint + price-snapshot hash, plus the panel
profile's ``user_message_suffix``, which lives outside the fingerprint); reopening
with a different identity raises. Every later line is one terminal record per request
(written with flush + fsync). Nothing is ever rewritten.

Resume: ``done(request_hash)`` returns the latest record for that hash if the
provider answered (``outcome == "observed"``: ok, empty, refused, truncated or a
malformed body — all terminal observations that are never re-sampled). Requests
that never got a 2xx answer (retry-exhausted 429/5xx, transport failures, 4xx
rejections) are sent again on resume and the new line supersedes the old one.

Spend: ``spent_usd()`` sums each record's ``usd_counted`` (the priced cost plus, as ``usd_pending_unknown``, the
bound of every attempt whose cost is unknown). A reservation is written to the ledger's send log (``<ledger>.sends``,
JSONL, fsync) before its request is sent and settled there after its record is written; ``in_flight_usd()`` = the
reservations not settled, whichever process made them, and ``committed_usd()`` = spent + in flight. ``reserve(usd)``
raises ``SpendCapExceeded`` if committed + usd would exceed the cap; ``append(record, reservation)`` refuses (raises,
the record kept in the send log as ``settle_refused``) a settlement that would carry committed over the cap. Both
run under a cross-process lock file (``<ledger>.lock``, created with O_CREAT | O_EXCL, broken when older than
LOCK_STALE_SECONDS) and re-read what other writers appended, so two holders of one ledger cannot both reserve the
same room. The cap therefore applies to the whole ledger across resumes and processes.

A live sender holds ``executor()`` (``<ledger>.executor.lock``, never broken automatically): one executor per ledger.
A reservation still open when an executor starts was made by one that died (crash, kill, power loss): its request
may have been sent and billed. ``recover_unsettled()`` lists each and settles it as a ``no_response`` record whose
whole reservation is ``usd_pending_unknown`` (or appends the record a refused settlement kept), so the request is
sent again and its possible cost stays counted.

Reopening: a process keeps its last parse of each ledger path (``_parsed``), so a Ledger opened again on the same
file parses only the lines appended since (a file replaced, truncated or rewritten is parsed in full). Call records a
Ledger hands out (``done``, ``records``) may therefore be shared with other Ledgers of the path: callers read them,
never modify them.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import threading
import time
import uuid

from procedures import stored
from procedures.readers.client.config import digest

logger = logging.getLogger(__name__)

SCHEMA = "procedures.readers.ledger.v1"         # every line of a call ledger
SENDS_SCHEMA = "procedures.readers.sends.v1"   # every line of a send log
IDENTITY_KEYS = ("profile_id", "profile_fingerprint", "price_book_sha256")
# Also compared on reopen but may be null (absent in ledgers written before the field existed).
OPTIONAL_IDENTITY_KEYS = ("user_message_suffix",)
LOCK_STALE_SECONDS = 120.0      # a short lock (reserve / settle / approval tally) older than this is a dead holder's
LOCK_WAIT_SECONDS = 300.0       # a short lock neither obtained nor broken as stale within this raises LockBusy
SETTLE_TOLERANCE_USD = 1e-9     # float slack of re-summed amounts (a sum of reservations that exactly fills a cap)


class LedgerError(RuntimeError):
    """The ledger file is malformed or belongs to a different run identity."""


class SpendCapExceeded(RuntimeError):
    """Sending the next request could exceed the run's spend cap."""


class LockBusy(LedgerError):
    """A lock file is held by another holder (an executor lock is never broken automatically)."""


def lf(text: str) -> str:
    return text.replace("\r\n", "\n")


def canonical_messages(messages) -> list:
    out = []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError("messages must be role/content dicts with text content")
        out.append({**message, "content": lf(message["content"])})
    return out


def request_hash(profile_id: str, profile_fingerprint: str, messages, params: dict, repeat_idx: int, *,
                 user_message_suffix: str | None = None) -> str:
    """sha256 over profile id + semantic profile fingerprint + LF-normalised messages as sent + params +
    repeat, plus ``user_message_suffix`` when the profile has one (left out otherwise, so hashes of
    suffix-free profiles are unchanged)."""
    identity = {"profile_id": profile_id, "profile_fingerprint": profile_fingerprint,
                "messages": canonical_messages(messages), "params": params, "repeat_idx": repeat_idx}
    if user_message_suffix is not None:
        identity["user_message_suffix"] = user_message_suffix
    return digest(identity)


def identity_clash(existing: dict, header: dict) -> dict:
    """{key: (existing, header)} for each run-identity key on which two ledger headers differ; {} = one identity."""
    return {k: (existing.get(k), header.get(k)) for k in IDENTITY_KEYS + OPTIONAL_IDENTITY_KEYS
            if existing.get(k) != header.get(k)}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sends_path(ledger_path: Path) -> Path:
    """The send log of a ledger: ``<ledger file name>.sends`` beside it (never matches ``*.jsonl``)."""
    return Path(ledger_path).with_name(Path(ledger_path).name + ".sends")


def _is_call(record) -> bool:
    return isinstance(record, dict) and record.get("kind") == "call" and "request_hash" in record\
        and "usd_counted" in record


def read_ledger(path: Path) -> tuple[dict, list[dict]]:
    """(header, call records in file order) of a ledger file. A torn final line, a non-JSON
    line, a missing header or a line that is not a call record raises LedgerError."""
    return _parse_ledger(Path(path), stored.read_bytes(path))


def _parse_ledger(path: Path, raw: bytes) -> tuple[dict, list[dict]]:
    if raw and not raw.endswith(b"\n"):
        cut = raw.rfind(b"\n") + 1
        raise LedgerError(f"{path}: torn final line at byte {cut} (process killed mid-write?); "
                          "inspect and remove the partial line before resuming")
    lines = raw.decode("utf-8").splitlines()
    if not lines:
        raise LedgerError(f"{path}: empty ledger file has no header")
    records = []
    for number, line in enumerate(lines, 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise LedgerError(f"{path}:{number}: malformed JSON ({exc})") from exc
    header = records[0]
    if header.get("kind") != "header" or header.get("schema") != SCHEMA:
        raise LedgerError(f"{path}: first line is not a {SCHEMA} header")
    for number, record in enumerate(records[1:], 2):
        if not _is_call(record):
            raise LedgerError(f"{path}:{number}: not a call record")
    return header, records[1:]


@dataclass(frozen=True)
class _Parsed:
    """What _parse_ledger made of a ledger file's first `offset` bytes, kept to reopen the file without re-parsing
    them. `ino` / `dev` name the file object; `head` and `edge` are its bytes at [0, 4096) and [offset - 4096,
    offset), re-read on reuse to catch a file replaced or rewritten in place."""
    header: dict
    calls: tuple
    offset: int
    ino: int
    dev: int
    head: bytes
    edge: bytes


_PARSED: dict[str, _Parsed] = {}        # absolute path -> the last parse of that ledger in this process
_PARSED_LOCK = threading.Lock()
_PARSED_MAX = 8                         # ledger paths kept (least recently opened dropped first)
_WINDOW = 4096


def _cache_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _window(stream, start: int, end: int) -> bytes:
    stream.seek(start)
    return stream.read(end - start)


def _reuse(path: Path, cached: _Parsed | None) -> _Parsed | None:
    """`cached` extended by the complete call lines appended since its offset, or None when it cannot be reused (no
    parse yet, another file object, the file shorter than the offset, the head or edge bytes changed, or anything
    after the offset that is not complete call lines: a torn or malformed line is left to a full parse, which
    raises the error it always did)."""
    if cached is None:
        return None
    with open(path, "rb") as stream:
        st = os.fstat(stream.fileno())
        if (st.st_ino, st.st_dev) != (cached.ino, cached.dev) or st.st_size < cached.offset:
            return None
        if _window(stream, 0, len(cached.head)) != cached.head\
                or _window(stream, cached.offset - len(cached.edge), cached.offset) != cached.edge:
            return None
        stream.seek(cached.offset)
        data = stream.read()
    if not data:
        return cached
    if not data.endswith(b"\n"):
        return None
    try:
        new = [json.loads(line) for line in data.decode("utf-8").splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not all(_is_call(record) for record in new):
        return None
    offset = cached.offset + len(data)
    edge = (cached.edge + data)[-_WINDOW:]
    head = cached.head if len(cached.head) == _WINDOW else (cached.head + data)[:_WINDOW]
    return _Parsed(cached.header, cached.calls + tuple(new), offset, cached.ino, cached.dev, head, edge)


def _parsed(path: Path) -> _Parsed:
    """The ledger file parsed as _parse_ledger does, reusing this process's last parse of the same path: only the
    lines appended since are read (the file is append-only; a file replaced, truncated or rewritten is parsed in
    full again). Correctness never rests on this cache: a Ledger still takes later lines, whoever writes them, from
    its own JsonlTail at the returned offset."""
    key = _cache_key(path)
    with _PARSED_LOCK:
        cached = _PARSED.pop(key, None)
    parsed = _reuse(path, cached)
    if parsed is None:
        with open(path, "rb") as stream:
            st = os.fstat(stream.fileno())
            raw = stream.read()
        header, calls = _parse_ledger(path, raw)
        parsed = _Parsed(header, tuple(calls), len(raw), st.st_ino, st.st_dev, raw[:_WINDOW],
                         raw[max(0, len(raw) - _WINDOW):])
    with _PARSED_LOCK:
        _PARSED[key] = parsed
        while len(_PARSED) > _PARSED_MAX:
            _PARSED.pop(next(iter(_PARSED)))
    return parsed


class FileLock:
    """Cross-process mutual exclusion through a lock file created with O_CREAT | O_EXCL (it holds the owner's pid
    and time). ``stale_seconds`` None = never broken automatically and not waited for (an executor lock: a second
    executor refuses at once); otherwise waited for up to ``wait_seconds`` and broken when its file is older than
    ``stale_seconds`` (a holder that died inside a short critical section)."""

    def __init__(self, path: Path, *, stale_seconds: float | None = LOCK_STALE_SECONDS,
                 wait_seconds: float = LOCK_WAIT_SECONDS, poll_seconds: float = 0.005):
        self.path = Path(path)
        self.stale_seconds, self.wait_seconds, self.poll_seconds = stale_seconds, wait_seconds, poll_seconds

    def _holder(self) -> str:
        try:
            return self.path.read_text(encoding="utf-8").strip() or "unknown holder"
        except FileNotFoundError:
            return "released"

    def __enter__(self):
        deadline = time.monotonic() + (0.0 if self.stale_seconds is None else self.wait_seconds)
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self.stale_seconds is None:
                    raise LockBusy(f"{self.path} is held ({self._holder()}); one executor per ledger. If that "
                                   "process is dead, delete the lock file: the next run lists and settles what it "
                                   "left in flight (procedures.readers.ledger.Ledger.recover_unsettled)") from None
                try:
                    age = time.time() - os.stat(self.path).st_mtime
                except FileNotFoundError:
                    continue
                if age > self.stale_seconds:
                    self._break(age)
                    continue
                if time.monotonic() > deadline:
                    raise LockBusy(f"{self.path} still held after {self.wait_seconds:.0f} s ({self._holder()})")\
                        from None
                time.sleep(self.poll_seconds)
                continue
            except PermissionError:          # Windows: the file is being deleted by its holder right now
                if time.monotonic() > deadline:
                    raise LockBusy(f"{self.path} could not be created (held or being released)") from None
                time.sleep(self.poll_seconds)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(f"pid {os.getpid()} since {utc_now()}\n")
            return self

    def _break(self, age: float) -> None:
        aside = self.path.with_name(f"{self.path.name}.stale-{os.getpid()}-{uuid.uuid4().hex[:8]}")
        try:
            os.replace(self.path, aside)
        except (FileNotFoundError, PermissionError):
            return
        logger.warning("broke stale lock %s (%.0f s old: %s)", self.path, age,
                       aside.read_text(encoding="utf-8").strip())
        aside.unlink()

    def __exit__(self, *exc) -> None:
        self.path.unlink()


class JsonlTail:
    """The complete lines appended to a JSONL file since the last ``read`` (a partial last line waits for the
    writer to finish it); a line that is not JSON raises LedgerError."""

    def __init__(self, path: Path, offset: int = 0):
        self.path, self.offset = Path(path), offset

    def read(self) -> list[dict]:
        try:
            with open(self.path, "rb") as stream:
                stream.seek(self.offset)
                data = stream.read()
        except FileNotFoundError:
            return []
        end = data.rfind(b"\n") + 1
        out = []
        for line in data[:end].decode("utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise LedgerError(f"{self.path}: malformed JSON after byte {self.offset} ({exc})") from exc
        self.offset += end
        return out


class SendLog:
    """The reservations of one ledger's send log: ``reserve`` opens one, ``settle`` closes it, ``settle_refused``
    keeps a record whose settlement the cap refused (the reservation stays open, counted at the larger of the two)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._tail = JsonlTail(self.path)
        self.open: dict[str, dict] = {}
        self.refused: dict[str, dict] = {}

    def refresh(self) -> None:
        for event in self._tail.read():
            kind, send_id = event.get("event"), event.get("send_id")
            if kind == "reserve":
                self.open[send_id] = event
            elif kind == "settle":
                self.open.pop(send_id, None)
                self.refused.pop(send_id, None)
            elif kind == "settle_refused":
                self.refused[send_id] = event
            else:
                raise LedgerError(f"{self.path}: unknown send-log event {event!r:.200}")

    def in_flight_usd(self) -> float:
        return sum(max(float(e["usd"]), float((self.refused.get(s) or {}).get("usd_counted", 0.0)))
                   for s, e in self.open.items())

    def write(self, event: dict) -> None:
        line = json.dumps({"schema": SENDS_SCHEMA, **event, "at": utc_now()}, ensure_ascii=False, sort_keys=True,
                          allow_nan=False) + "\n"
        with open(self.path, "a", encoding="utf-8", newline="\n") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())


@dataclass(frozen=True)
class Reservation:
    """Room held in a ledger for one request (all its allowed attempts) until its record settles it."""
    send_id: str
    usd: float
    request_hash: str | None = None


class Ledger:
    def __init__(self, path: Path, header: dict, *, cap_usd: float):
        if isinstance(cap_usd, bool) or not isinstance(cap_usd, (int, float)) or not cap_usd >= 0:
            raise ValueError("cap_usd must be a non-negative number")
        missing = [k for k in IDENTITY_KEYS if not header.get(k)]
        if missing:
            raise LedgerError(f"ledger header lacks identity fields {missing}")
        self.path = Path(path)
        self.cap_usd = float(cap_usd)
        self._lock = threading.Lock()
        self._latest: dict[str, dict] = {}
        self._send_ids: set[str] = set()
        self._spent = 0.0
        self._pending_unknown = 0.0
        self._sends = SendLog(sends_path(self.path))
        if self.path.exists():
            self.header = self._load(header)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.header = {"kind": "header", "schema": SCHEMA, "created_at": utc_now(), **header}
            self._tail = JsonlTail(self.path, self._write(self.header))
        self._sends.refresh()

    def _load(self, header: dict) -> dict:
        parsed = _parsed(self.path)     # the records are shared with other Ledgers of this path: read-only
        existing = dict(parsed.header)
        clash = identity_clash(existing, header)
        if clash:
            raise LedgerError(f"{self.path}: run identity differs from the existing ledger: {clash}")
        self._tail = JsonlTail(self.path, parsed.offset)    # later lines, whoever appends them, are read from here
        self._take(parsed.calls)
        return existing

    def _take(self, calls: list[dict]) -> None:
        for record in calls:
            if not _is_call(record):
                raise LedgerError(f"{self.path}: appended line is not a call record: {record!r:.200}")
            self._latest[record["request_hash"]] = record
            self._spent += float(record["usd_counted"])
            self._pending_unknown += float(record.get("usd_pending_unknown") or 0.0)
            if record.get("send_id"):
                self._send_ids.add(record["send_id"])

    def _refresh(self) -> None:
        """Take what any writer appended: the send log first, then the ledger, so a request settled in between is
        counted twice (its record and its reservation), never missed."""
        self._sends.refresh()
        self._take(self._tail.read())

    def _file_lock(self) -> FileLock:
        return FileLock(self.path.with_name(self.path.name + ".lock"))

    def _write(self, record: dict) -> int:
        """Append one line (flush + fsync); returns its length in bytes."""
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        with open(self.path, "a", encoding="utf-8", newline="\n") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        return len(line.encode("utf-8"))

    def records(self) -> list[dict]:
        with self._lock:
            self._refresh()
            return list(self._latest.values())

    def done(self, request_hash_: str) -> dict | None:
        """The latest record of the hash if observed, as of the last read of the file (construction or any
        reserve / append / records / *_usd call; within an executor no other process writes the ledger)."""
        with self._lock:
            record = self._latest.get(request_hash_)
        return record if record is not None and record.get("outcome") == "observed" else None

    def spent_usd(self) -> float:
        """Sum of usd_counted over every record (settled spend, its pending_unknown part included)."""
        with self._lock:
            self._refresh()
            return self._spent

    def pending_unknown_usd(self) -> float:
        """The part of spent_usd that is attempts of unknown cost (bounds, not bills), until reconciled."""
        with self._lock:
            self._refresh()
            return self._pending_unknown

    def in_flight_usd(self) -> float:
        with self._lock:
            self._refresh()
            return self._sends.in_flight_usd()

    def committed_usd(self) -> float:
        with self._lock:
            self._refresh()
            return self._spent + self._sends.in_flight_usd()

    def reserve(self, upper_bound_usd: float, *, request_hash: str | None = None, stub: dict | None = None,
                detail: dict | None = None) -> Reservation:
        """Hold `upper_bound_usd` (a request's bound over all its allowed attempts) before sending it; `stub` = the
        identifying fields a recovered record gets if the sender dies (procedures.readers.executor), `detail` = e.g.
        the bound per attempt and the attempt count."""
        if not upper_bound_usd >= 0:
            raise ValueError("reservation must be a non-negative amount")
        with self._lock, self._file_lock():
            self._refresh()
            in_flight = self._sends.in_flight_usd()
            projected = self._spent + in_flight + upper_bound_usd
            if projected > self.cap_usd + SETTLE_TOLERANCE_USD:
                raise SpendCapExceeded(
                    f"cap ${self.cap_usd:.4f}: spent ${self._spent:.4f} + in flight ${in_flight:.4f} "
                    f"+ next request bound ${upper_bound_usd:.4f} = ${projected:.4f}")
            reservation = Reservation(uuid.uuid4().hex, float(upper_bound_usd), request_hash)
            self._sends.write({"event": "reserve", "send_id": reservation.send_id, "request_hash": request_hash,
                               "usd": reservation.usd, "pid": os.getpid(), "stub": stub or {}, **(detail or {})})
            self._sends.refresh()
            return reservation

    def append(self, record: dict, reservation: Reservation | None = None) -> None:
        """Write one terminal call record and settle its reservation. A settlement that would carry committed spend
        over the cap is refused: SpendCapExceeded, nothing written to the ledger, the record kept in the send log
        (`settle_refused`) with the reservation left open, so recover_unsettled can append it once the cap allows."""
        if record.get("kind") != "call" or not record.get("request_hash"):
            raise LedgerError("only call records with a request_hash can be appended")
        counted = record.get("usd_counted")
        if isinstance(counted, bool) or not isinstance(counted, (int, float)) or not counted >= 0:
            raise LedgerError(f"usd_counted must be a non-negative number, got {counted!r}")
        written = {"schema": SCHEMA, **record, **({"send_id": reservation.send_id} if reservation else {})}
        with self._lock, self._file_lock():
            self._refresh()
            held = self._sends.open.get(reservation.send_id, {}).get("usd", 0.0) if reservation else 0.0
            others = self._sends.in_flight_usd() - float(held)
            total = self._spent + others + float(counted)
            if total > self.cap_usd + SETTLE_TOLERANCE_USD:
                if reservation is not None:
                    self._sends.write({"event": "settle_refused", "send_id": reservation.send_id,
                                       "usd_counted": float(counted), "record": written})
                raise SpendCapExceeded(
                    f"cap ${self.cap_usd:.4f}: settling {record['request_hash'][:12]} at ${float(counted):.4f} would "
                    f"make spent ${self._spent:.4f} + other in flight ${others:.4f} = ${total:.4f}; the record is "
                    f"kept in {self._sends.path.name}, its reservation stays counted")
            self._write(written)
            if reservation is not None:
                self._sends.write({"event": "settle", "send_id": reservation.send_id,
                                   "usd_counted": float(counted)})
            self._refresh()

    @contextmanager
    def executor(self):
        """The one-executor-per-ledger lock (``<ledger>.executor.lock``), held while a run sends through this
        ledger; a second executor raises LockBusy at once."""
        with FileLock(self.path.with_name(self.path.name + ".executor.lock"), stale_seconds=None):
            yield self

    def recover_unsettled(self) -> list[dict]:
        """Settle every reservation left open by an executor that died (call only while holding ``executor()``):
        a refused settlement's kept record is appended as it was (refused again if the cap still does not allow
        it); a reservation whose record reached the ledger is closed; any other becomes a ``no_response`` record,
        its whole reservation counted as usd_pending_unknown (its request may have been sent, answered and
        billed), so the request is sent again. Returns one line per reservation, in log order."""
        with self._lock:
            self._refresh()
            left = [(s, dict(e), self._sends.refused.get(s)) for s, e in self._sends.open.items()]
        out = []
        for send_id, event, refused in left:
            reservation = Reservation(send_id, float(event["usd"]), event.get("request_hash"))
            line = {"send_id": send_id, "request_hash": event.get("request_hash"), "usd": reservation.usd,
                    "pid": event.get("pid"), "reserved_at": event.get("at")}
            if send_id in self._send_ids:
                with self._lock, self._file_lock():
                    self._sends.write({"event": "settle", "send_id": send_id, "usd_counted": None,
                                       "note": "record already in the ledger"})
                    self._refresh()
                out.append({**line, "resolution": "record_present"})
                continue
            if refused is not None:
                kept = {k: v for k, v in refused["record"].items() if k not in ("schema", "send_id")}
                self.append(kept, reservation)
                out.append({**line, "resolution": "kept_record_appended", "usd_counted": kept["usd_counted"]})
                continue
            self.append({**event.get("stub", {}), "kind": "call", "request_hash": event.get("request_hash"),
                         "outcome": "no_response", "status": None, "usd": None,
                         "usd_upper_bound": event.get("attempt_bound_usd"), "usd_reserved": reservation.usd,
                         "usd_counted": reservation.usd, "usd_pending_unknown": reservation.usd, "attempts": [],
                         "error": {"kind": "unsettled_send",
                                   "message": f"reserved by pid {event.get('pid')} at {event.get('at')} and never "
                                              "settled; the sender died, the request may have been billed"},
                         "finished_at": utc_now()}, reservation)
            out.append({**line, "resolution": "pending_unknown"})
        for line in out:
            logger.warning("recovered unsettled send %s (%s): %s, $%.6f", line["send_id"],
                           (line["request_hash"] or "")[:12], line["resolution"], line["usd"])
        return out
