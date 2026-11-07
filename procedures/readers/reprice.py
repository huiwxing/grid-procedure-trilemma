"""Re-price ledger records on another price book (e.g. a corrected one) without touching the ledger.

A ledger is append-only and its header binds the run identity, including the sha256 of the price book
it was priced on (procedures.readers.ledger). Re-pricing therefore never rewrites a ledger: it reads each answered
record's normalised ``usage`` and prices it again with ``price_call`` on the given book, at the same
instant the run priced it (the answering attempt's ``started_at``, procedures.readers.executor). Request hashes, the
header and every recorded field stay as they are.
"""
from __future__ import annotations

from datetime import datetime

from procedures.readers.pricing import PriceBook, price_call


class RepriceError(ValueError):
    """A ledger record cannot be re-priced as given."""


def answering_instant(record: dict) -> datetime:
    """Start of the attempt that got the 2xx answer: the instant the run priced the call at."""
    return datetime.fromisoformat(record["attempts"][-1]["started_at"])


def reprice_record(record: dict, book: PriceBook) -> dict:
    """``price_call`` of an answered record's usage on ``book`` at its answering instant (an incomplete
    usage comes back unpriced, never 0)."""
    if record.get("outcome") != "observed":
        raise RepriceError(f"only answered records are priced, not outcome {record.get('outcome')!r}")
    key = (record.get("price") or {}).get("price_key")
    if not key:
        raise RepriceError(f"record {record.get('request_hash')} has no price_key")
    return price_call(key, record["usage"], book=book, observed_at=answering_instant(record))

