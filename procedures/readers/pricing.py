"""Price one call from normalised usage and a dated price snapshot.

    usd = [ input_uncached·p_in + input_cached·p_cached + (cache_write − cw_1h)·p_write
            + cw_1h·p_write_1h + output_total·p_out ] / 1e6            (× long-context multipliers)

output_total already contains reasoning/thinking tokens (every panel provider bills
them as output); they are never added a second time.

Rules beyond the flat rates (sources: the providers' published pricing and the snapshot's notes):
  * GPT-6 (sol/luna/astra): input > 272K tokens → whole request at 2× input-side and
    1.5× output rates.
  * Gemini 3.1 Pro preview: input > 200K → the snapshot's ``gt200k`` rates.
  * Gemini 3.8 Flash: rates in ``from_2027_01_01`` apply from 2027-01-01 (UTC).
  * DeepSeek direct: ``peak`` rates Mon–Fri 01:00–04:00 and 06:00–10:00 UTC, else ``offpeak``
    (the official rule excludes Chinese public holidays; the 2026-09-24 snapshot has no holiday
    calendar, so a holiday is priced like any other weekday).
  * A time-dependent route priced without ``observed_at`` (planning) takes the more
    expensive variant and says so in ``variant``.
  * A route without a published cached / cache-write price bills those tokens at the
    plain input price and records a flag (an upper bound; the billing reconciliation
    checks it against the provider invoice).

Unit assertions: the snapshot must declare USD per 1M tokens; each rate must be a
finite number in [1e-4, 500] (or 0); a single call above MAX_USD_PER_CALL is
treated as a unit error.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

from procedures import ROOT

DEFAULT_PRICE_BOOK = ROOT / "data" / "prices" / "prices.json"            # the snapshot of 2026-09-24
UNITS_PREFIX = "USD per 1M tokens"
MAX_RATE = 500.0          # $/Mtok; above this a per-token vs per-million mix-up is likelier
MIN_NONZERO_RATE = 1e-4   # $/Mtok; a per-token price (~1e-6) would land below this
MAX_USD_PER_CALL = 25.0   # largest planned call is ~$1.3 (128k output at $10/M)
LONG_CONTEXT = {  # price_key -> (input threshold, input-side multiplier, output multiplier)
    "gpt-6-sol": (272_000, 2.0, 1.5),
    "gpt-6-luna": (272_000, 2.0, 1.5),
    "gpt-6-astra": (272_000, 2.0, 1.5),
}
GT200K_THRESHOLD = 200_000
DATE_SWITCH = {"from_2027_01_01": date(2027, 1, 1)}
DEEPSEEK_PEAK_UTC_HOURS = ((1, 4), (6, 10))  # [start, end) hours, Monday-Friday


class PricingError(ValueError):
    """The price book or a computed price violates a unit assertion."""


@dataclass(frozen=True)
class Rates:
    input: float
    cached_input: float | None
    cache_write: float | None
    cache_write_1h: float | None
    output: float
    variant: str

    def as_dict(self) -> dict:
        return {"in": self.input, "cached_in": self.cached_input, "cache_write": self.cache_write,
                "cache_write_1h": self.cache_write_1h, "out": self.output, "variant": self.variant}


def _rate(name: str, value, *, optional: bool) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise PricingError(f"{name} must be a finite number of USD per 1M tokens, got {value!r}")
    if value < 0 or value > MAX_RATE or 0 < value < MIN_NONZERO_RATE:
        raise PricingError(f"{name}={value} is outside [{MIN_NONZERO_RATE}, {MAX_RATE}] USD per 1M tokens")
    return float(value)


def _utc(observed_at: datetime) -> datetime:
    if observed_at.tzinfo is None:
        raise PricingError("observed_at must be timezone-aware")
    return observed_at.astimezone(timezone.utc)


def is_deepseek_peak(observed_at: datetime) -> bool:
    """DeepSeek's peak window (Mon-Fri 01-04 and 06-10 UTC)."""
    t = _utc(observed_at)
    return t.weekday() < 5 and any(start <= t.hour < end for start, end in DEEPSEEK_PEAK_UTC_HOURS)


@dataclass(frozen=True)
class PriceBook:
    path: Path
    sha256: str
    fetched: str
    units: str
    models: dict

    def rates(self, key: str, *, observed_at: datetime | None = None, input_tokens: int = 0) -> Rates:
        if key not in self.models:
            raise PricingError(f"price key {key!r} is not in snapshot {self.path.name}")
        entry = self.models[key]
        base, variant = entry, "standard"
        if "offpeak" in entry and "peak" in entry:
            if observed_at is None:
                base, variant = entry["peak"], "peak (max; no observed_at)"
            else:
                base, variant = ((entry["peak"], "peak") if is_deepseek_peak(observed_at)
                                 else (entry["offpeak"], "offpeak"))
        for switch, start in DATE_SWITCH.items():
            if switch in entry:
                later = {**entry, **entry[switch]}
                if observed_at is None:
                    base, variant = later, f"{switch} (max; no observed_at)"
                elif _utc(observed_at).date() >= start:
                    base, variant = later, switch
        if "gt200k" in entry and input_tokens > GT200K_THRESHOLD:
            base, variant = {**entry, **entry["gt200k"]}, "gt200k"
        return Rates(input=_rate(f"{key}.in", base.get("in"), optional=False),
                     cached_input=_rate(f"{key}.cached_in", base.get("cached_in"), optional=True),
                     cache_write=_rate(f"{key}.cache_write", base.get("cache_write"), optional=True),
                     cache_write_1h=_rate(f"{key}.cache_write_1h", base.get("cache_write_1h"), optional=True),
                     output=_rate(f"{key}.out", base.get("out"), optional=False), variant=variant)


def file_sha256_lf(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_price_book(path: Path = DEFAULT_PRICE_BOOK) -> PriceBook:
    """A price book file: the dated snapshot, or a corrected book derived from it. Parsed and validated once per
    path and file version (modification time and size; the planners price thousands of dry-run cells with the default
    book)."""
    path = Path(path)
    stat = path.stat()
    return _parsed_book(path, stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=8)
def _parsed_book(path: Path, mtime_ns: int, size: int) -> PriceBook:
    data = path.read_bytes()
    doc = json.loads(data.decode("utf-8"))
    units = doc.get("units", "")
    if not isinstance(units, str) or not units.startswith(UNITS_PREFIX):
        raise PricingError(f"price snapshot must declare '{UNITS_PREFIX}', got {units!r}")
    models = doc.get("models")
    if not isinstance(models, dict) or not models:
        raise PricingError("price snapshot has no models")
    book = PriceBook(path=path, sha256=hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest(),
                     fetched=str(doc.get("fetched")),
                     units=units, models=models)
    for key in models:  # validate every rate once, at load time
        book.rates(key)
        book.rates(key, observed_at=datetime(2026, 1, 5, 2, tzinfo=timezone.utc))
        book.rates(key, observed_at=datetime(2026, 1, 5, 12, tzinfo=timezone.utc), input_tokens=GT200K_THRESHOLD + 1)
    return book


def price_call(profile, usage: dict, *, book: PriceBook, observed_at: datetime | None = None) -> dict:
    """Price one normalised usage record for a panel profile (or a bare price key)."""
    key = profile if isinstance(profile, str) else profile.price_key
    if not key:
        raise PricingError("profile has no price_key")
    out = {"usd": None, "status": "unpriced", "reason": None, "price_key": key,
           "price_book_sha256": book.sha256, "rates": None, "breakdown": None, "multipliers": None, "flags": []}
    if not usage.get("complete"):
        out["reason"] = f"usage incomplete: missing={usage.get('missing')} inconsistent={usage.get('inconsistent')}"
        return out
    counts = {name: usage[name] for name in ("input_total", "input_uncached", "input_cached",
                                             "cache_write", "cache_write_1h", "output_total")}
    for name, value in counts.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise PricingError(f"{name} must be a non-negative integer token count, got {value!r}")
    if counts["input_uncached"] + counts["input_cached"] + counts["cache_write"] != counts["input_total"]:
        raise PricingError(f"input parts do not sum to input_total: {counts}")
    if counts["cache_write_1h"] > counts["cache_write"]:
        raise PricingError("cache_write_1h exceeds cache_write")
    rates = book.rates(key, observed_at=observed_at, input_tokens=counts["input_total"])
    cached_rate, write_rate, write_1h_rate = rates.cached_input, rates.cache_write, rates.cache_write_1h
    if cached_rate is None:
        cached_rate = rates.input
        if counts["input_cached"]:
            out["flags"].append("cached_price_unpublished_billed_at_input")
    if write_rate is None:
        write_rate = rates.input
        if counts["cache_write"]:
            out["flags"].append("cache_write_price_unpublished_billed_at_input")
    if write_1h_rate is None:
        write_1h_rate = write_rate
        if counts["cache_write_1h"]:
            out["flags"].append("cache_write_1h_price_unpublished_billed_at_cache_write")
    in_mult, out_mult = 1.0, 1.0
    if key in LONG_CONTEXT and counts["input_total"] > LONG_CONTEXT[key][0]:
        _, in_mult, out_mult = LONG_CONTEXT[key]
        out["flags"].append("long_context_multiplier")
    breakdown = {
        "input_uncached": counts["input_uncached"] * rates.input * in_mult / 1e6,
        "input_cached": counts["input_cached"] * cached_rate * in_mult / 1e6,
        "cache_write": (counts["cache_write"] - counts["cache_write_1h"]) * write_rate * in_mult / 1e6,
        "cache_write_1h": counts["cache_write_1h"] * write_1h_rate * in_mult / 1e6,
        "output": counts["output_total"] * rates.output * out_mult / 1e6,
    }
    usd = sum(breakdown.values())
    if not math.isfinite(usd) or usd < 0 or usd > MAX_USD_PER_CALL:
        raise PricingError(f"one call priced at ${usd}; check token units and $/1M rates")
    out.update(usd=usd, status="priced", rates=rates.as_dict(), breakdown=breakdown,
               multipliers={"input_side": in_mult, "output": out_mult})
    return out
