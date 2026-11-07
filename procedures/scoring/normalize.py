"""Normalisation primitives for the frozen scorer (scoring rules N2-N8: labels, numbers, units, times and dates).

Strings   norm_str (whitespace / hyphen / underscore runs folded, trimmed, lower-cased) -> norm_label (+ terminal
          punctuation, articles) -> plural_forms.
Numbers   parse_quantity: JSON numbers, numerals with proper thousands grouping, English
          cardinals 0-100, "X (N)" in either order, a unit written inside the value.
          All numbers are Decimal; equality is exact.
Units     dimension-aware table; exact conversion only inside a dimension and only between
          units that carry a factor (month/year/business day, %, pu, cycles, temperature,
          angle and ramp rate never convert).
Time/date norm_time -> "HH:MM" (24 h), norm_date -> ISO; UNDECIDABLE when the scorer cannot
          decide (midnight, two-digit years).
Documents DocIndex: what values a source document contains (E-other subclass).
"""
from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from decimal import Decimal, localcontext

UNDECIDABLE = object()  # sentinel: well-formed but the scorer cannot pick one reading

# --------------------------------------------------------------------------- strings

_SEP = re.compile(r"[\s\-_]+")
_TERMINAL_PUNCT = " \t\r\n\f\v.,;:!?"
_ARTICLES = ("a", "an", "the")


def norm_str(s) -> str:
    """Fold whitespace/hyphen/underscore runs into one space, trim, lowercase."""
    return re.sub(r"[\s\-_]+", " ", str(s)).strip().lower()


def strip_terminal_punct(s: str) -> str:
    return s.rstrip(_TERMINAL_PUNCT)


def _is_article(tokens: list[str], i: int) -> bool:
    tok = tokens[i]
    low = tok.lower()
    if len(tokens) < 2 or low not in _ARTICLES:
        return False
    if low != "a":
        return True
    # "a": drop only in determiner position (followed by a word) and only when it cannot be
    # a label: leading, or lowercase mid-string.  "Phase A", "Category A facility" keep it.
    return i < len(tokens) - 1 and (i == 0 or tok == "a")


def norm_label(s) -> str:
    """Label form for N2 / N3 / N6 (abstention, not in effect, categorical labels): terminal punctuation off,
    articles off, then norm_str."""
    tokens = [t for t in _SEP.split(strip_terminal_punct(str(s).strip())) if t]
    return " ".join(t for i, t in enumerate(tokens) if not _is_article(tokens, i)).lower()


def _number_variants(word: str) -> set[str]:
    out = {word}
    if re.search(r"[^aeiou]ies$", word):
        out.add(word[:-3] + "y")
    elif re.search(r"(s|x|z|ch|sh)es$", word):
        out.add(word[:-2])
    elif word.endswith("s") and not re.search(r"(ss|us|is)$", word):
        out.add(word[:-1])
    elif re.search(r"[^aeiou]y$", word):
        out.add(word[:-1] + "ies")
    elif re.search(r"(s|x|z|ch|sh)$", word):
        out.add(word + "es")
    else:
        out.add(word + "s")
    return out


def plural_forms(label: str) -> frozenset[str]:
    """Singular/plural variants of the last token of an already norm_label-ed label."""
    if not label or label in ("yes", "no"):
        return frozenset({label})
    head, _, last = label.rpartition(" ")
    return frozenset(f"{head} {w}" if head else w for w in _number_variants(last))


# --------------------------------------------------------------------------- numbers

_SMALL = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * (i + 2) for i, w in enumerate(
    "twenty thirty forty fifty sixty seventy eighty ninety".split())}
_DIGIT_WORDS = "one|two|three|four|five|six|seven|eight|nine"
_TEEN_WORDS = "|".join(sorted(_SMALL, key=len, reverse=True))
_WORDNUM = (rf"(?:(?:one[\s-]+)?hundred|(?:{'|'.join(_TENS)})(?:[\s-]+(?:{_DIGIT_WORDS}))?"
            rf"|{_TEEN_WORDS})(?![a-z])")
_NUMERAL = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[-+]?\.\d+"
_NUMTOKEN = rf"(?:{_NUMERAL}|{_WORDNUM})(?![\d,.]*\d)"
_QUANTITY = re.compile(
    rf"(?P<a>{_NUMTOKEN})\s*(?:\(\s*(?P<b>{_NUMTOKEN})\s*\))?\s*(?P<rest>(?:-?(?:[^\W\d_]|[%°]).*)?)",
    re.IGNORECASE | re.DOTALL)


def words_to_int(phrase: str) -> int | None:
    """English cardinals 0-100 ("seventy-two", "one hundred"); None otherwise."""
    toks = [t for t in re.split(r"[\s\-]+", phrase.strip().lower()) if t]
    if toks in (["hundred"], ["one", "hundred"]):
        return 100
    if len(toks) == 1:
        return _SMALL.get(toks[0], _TENS.get(toks[0]))
    if len(toks) == 2 and toks[0] in _TENS and 1 <= _SMALL.get(toks[1], 0) <= 9:
        return _TENS[toks[0]] + _SMALL[toks[1]]
    return None


def _token_number(tok: str) -> Decimal:
    if re.fullmatch(_NUMERAL, tok):
        return Decimal(tok.replace(",", ""))
    return Decimal(words_to_int(tok))


def to_decimal(x) -> Decimal | None:
    """JSON number or bare numeral string -> finite Decimal; bool and everything else -> None."""
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return Decimal(x)
    if isinstance(x, float):
        d = Decimal(repr(x))
    elif isinstance(x, Decimal):
        d = x
    elif isinstance(x, str) and re.fullmatch(_NUMERAL, x.strip()):
        d = Decimal(x.strip().replace(",", ""))
    else:
        return None
    return d if d.is_finite() else None


@dataclass(frozen=True)
class Quantity:
    number: Decimal | None   # None only when note says why (e.g. word/numeral disagree)
    unit_text: str | None    # unit written inside the value string, if any
    note: str | None = None


def parse_quantity(value) -> Quantity | None:
    """N4 (numbers): a value that reads as a number (optionally with a unit), else None."""
    if not isinstance(value, str):
        d = to_decimal(value)
        return Quantity(d, None) if d is not None else None
    m = _QUANTITY.fullmatch(strip_terminal_punct(value.strip()))
    if not m:
        return None
    a = _token_number(m["a"])
    if m["b"] is not None and _token_number(m["b"]) != a:
        return Quantity(None, None, "word/numeral disagree in 'X (N)'")
    rest = m["rest"].strip()
    return Quantity(a, rest or None)


# --------------------------------------------------------------------------- units

@dataclass(frozen=True)
class Unit:
    symbol: str
    dimension: str
    factor: Decimal | None   # to the dimension's base unit; None = never converted


_D = Decimal
_UNIT_ROWS = (
    # symbol, dimension, factor, alias keys (after unit_key normalisation)
    ("W", "active_power", _D(1), "w watt watts"),
    ("kW", "active_power", _D(10) ** 3, "kw kilowatt kilowatts"),
    ("MW", "active_power", _D(10) ** 6, "mw megawatt megawatts"),
    ("GW", "active_power", _D(10) ** 9, "gw gigawatt gigawatts"),
    ("var", "reactive_power", _D(1), "var vars"),
    ("kvar", "reactive_power", _D(10) ** 3, "kvar kvars kilovar kilovars"),
    ("MVAR", "reactive_power", _D(10) ** 6, "mvar mvars megavar megavars"),
    ("VA", "apparent_power", _D(1), "va voltampere voltamperes"),
    ("kVA", "apparent_power", _D(10) ** 3, "kva kilovoltampere kilovoltamperes"),
    ("MVA", "apparent_power", _D(10) ** 6, "mva megavoltampere megavoltamperes"),
    ("V", "voltage", _D(1), "v volt volts"),
    ("kV", "voltage", _D(10) ** 3, "kv kilovolt kilovolts"),
    ("A", "current", _D(1), "a amp amps ampere amperes"),
    ("kA", "current", _D(10) ** 3, "ka kiloamp kiloamps kiloampere kiloamperes"),
    ("Wh", "energy", _D(1), "wh watthour watthours"),
    ("kWh", "energy", _D(10) ** 3, "kwh kilowatthour kilowatthours"),
    ("MWh", "energy", _D(10) ** 6, "mwh megawatthour megawatthours"),
    ("GWh", "energy", _D(10) ** 9, "gwh gigawatthour gigawatthours"),
    ("Hz", "frequency", _D(1), "hz hertz"),
    ("mHz", "frequency", _D("0.001"), "mhz millihertz"),
    ("ms", "time", _D("0.001"), "ms msec msecs millisecond milliseconds"),
    ("s", "time", _D(1), "s sec secs second seconds"),
    ("min", "time", _D(60), "min mins minute minutes"),
    ("h", "time", _D(3600), "h hr hrs hour hours"),
    ("day", "time", _D(86400), "day days calendarday calendardays"),
    ("week", "time", _D(604800), "week weeks wk wks calendarweek calendarweeks"),
    ("business_day", "time", None, "businessday businessdays workingday workingdays"),
    ("month", "time", None, "month months calendarmonth calendarmonths"),
    ("year", "time", None, "year years yr yrs calendaryear calendaryears"),
    ("cycle", "cycles", None, "cycle cycles"),
    ("%", "percent", None, "% percent pct"),
    ("pu", "per_unit", None, "pu perunit"),
    ("degF", "temperature", None, "°f degf degreef degreesf fahrenheit degreesfahrenheit"),
    ("degC", "temperature", None, "°c degc degreec degreesc celsius degreescelsius"),
    ("deg", "angle", None, "° deg degree degrees"),
    ("MW/min", "ramp_rate", None, "mw/min mw/mins mw/minute mw/minutes mwperminute"),
    ("m", "length", _D(1), "meter meters metre metres"),
    ("km", "length", _D(1000), "km kilometer kilometers kilometre kilometres"),
    ("ft", "length", _D("0.3048"), "ft foot feet"),
    ("mi", "length", _D("1609.344"), "mi mile miles"),
)


def unit_key(u) -> str:
    """Unit key: whitespace, dots, '-' and '_' dropped, lower-cased."""
    return re.sub(r"[\s.\-_]", "", str(u)).lower()


def _build_unit_aliases() -> dict[str, Unit]:
    table: dict[str, Unit] = {}
    for symbol, dim, factor, keys in _UNIT_ROWS:
        unit = Unit(symbol, dim, factor)
        for key in keys.split():
            if key in table:
                raise ValueError(f"unit alias {key!r} defined twice")
            table[key] = unit
    return table


UNIT_ALIASES = _build_unit_aliases()


def lookup_unit(u) -> Unit | None:
    return UNIT_ALIASES.get(unit_key(u))


def unit_equal(a_num: Decimal, a_unit: Unit, g_num: Decimal, g_unit: Unit) -> tuple[bool, bool]:
    """(numerically equal, a conversion was needed); exact Decimal arithmetic."""
    if a_unit == g_unit:
        return a_num == g_num, False
    if a_unit.dimension == g_unit.dimension and a_unit.factor is not None and g_unit.factor is not None:
        with localcontext() as ctx:
            ctx.prec = 80
            return a_num * a_unit.factor == g_num * g_unit.factor, True
    return False, False


# --------------------------------------------------------------------------- time of day (N8)

_TZ_WORDS = frozenset(
    "ept est edt et cpt cst cdt ct ppt pst pdt pt mpt mst mdt mt utc gmt z eastern central "
    "pacific mountain prevailing standard daylight local time hour hours hrs".split())


def norm_time(value):
    """-> "HH:MM" (24 h), UNDECIDABLE ("midnight"), or None (not a time of day)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, Decimal)):
        if value != int(value) or value < 0 or len(str(int(value))) not in (3, 4):
            return None
        s = str(int(value))
    elif isinstance(value, str):
        s = value
    else:
        return None
    toks = strip_terminal_punct(s.strip()).lower().split()
    while toks and toks[-1].strip(".,") in _TZ_WORDS:
        toks.pop()
    s = " ".join(toks)
    if s == "noon":
        return "12:00"
    if s == "midnight":
        return UNDECIDABLE
    ampm = r"(?:([ap])\.?\s*m\.?)"
    m = (re.fullmatch(rf"(\d{{1,2}}):(\d{{2}})\s*{ampm}?", s)
         or re.fullmatch(rf"(\d{{1,2}})()\s*{ampm}", s)
         or re.fullmatch(r"(\d{1,2})(\d{2})()", s))
    if not m:
        return None
    hour, minute, ap = int(m[1]), int(m[2] or 0), m[3]
    if minute > 59:
        return None
    if ap:
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if ap == "p" else 0)
    elif hour > 24 or (hour == 24 and minute):
        return None
    return f"{hour:02d}:{minute:02d}"


# --------------------------------------------------------------------------- dates (N8)

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3, "apr": 4,
    "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
_DATE_FORMS = (  # (regex, group order) — numeric forms read North-American MM/DD
    (re.compile(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})"), "ymd"),
    (re.compile(r"(\d{1,2})([/-])(\d{1,2})\2(\d{4}|\d{2})"), "m_dy"),
    (re.compile(r"([a-z]+)\.? (\d{1,2}),? (\d{4}|\d{2})"), "Mdy"),
    (re.compile(r"(\d{1,2}) ([a-z]+)\.?,? (\d{4}|\d{2})"), "dMy"),
    (re.compile(r"([a-z]{3,9})-(\d{1,2})-(\d{4}|\d{2})"), "Mdy"),
    (re.compile(r"(\d{1,2})-([a-z]{3,9})-(\d{4}|\d{2})"), "dMy"),
)


def norm_date(value):
    """-> ISO "YYYY-MM-DD", UNDECIDABLE (two-digit year), or None (not a date)."""
    if not isinstance(value, str):
        return None
    s = strip_terminal_punct(value.strip()).lower()
    s = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", s)
    s = re.sub(r"\s+", " ", s)
    for rx, order in _DATE_FORMS:
        m = rx.fullmatch(s)
        if not m:
            continue
        if order == "ymd":
            y, mo, d = m[1], m[2], m[3]
        elif order == "m_dy":
            mo, d, y = m[1], m[3], m[4]
        elif order == "Mdy":
            mo, d, y = _MONTHS.get(m[1]), m[2], m[3]
        else:
            d, mo, y = m[1], _MONTHS.get(m[2]), m[3]
        if mo is None:
            return None
        try:
            parsed = _dt.date(int(y) if len(y) == 4 else 2000, int(mo), int(d))
        except ValueError:
            return None  # impossible calendar date (e.g. 02/30/2024): not a date
        return parsed.isoformat() if len(y) == 4 else UNDECIDABLE
    return None


# --------------------------------------------------------------------------- documents

_DOC_NUMERAL = re.compile(r"(?<![\d.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\d)")
_DOC_WORDNUM = re.compile(rf"\b{_WORDNUM}", re.IGNORECASE)
_DOC_DATE = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{4}"
    r"|[a-z]{3,9}\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}|\d{1,2}\s+[a-z]{3,9}\.?,?\s+\d{4}"
    r"|[a-z]{3,9}-\d{1,2}-\d{4}|\d{1,2}-[a-z]{3,9}-\d{4})\b", re.IGNORECASE)
_DOC_TIME = re.compile(
    r"\b\d{1,2}:\d{2}(?:\s*[ap]\.?\s*m\.?)?|\b\d{1,2}\s*[ap]\.\s*m\.|\b\d{1,2}\s*[ap]m\b"
    r"|\b\d{4}\s+(?:ept|est|edt|cpt|cst|cdt|ppt|pst|pdt|hours|hrs)\b", re.IGNORECASE)


def _label_tokens(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in _ARTICLES]


@dataclass(frozen=True)
class DocIndex:
    """Values a document contains, for the E-other in-doc / not-in-doc split."""
    numbers: frozenset
    dates: frozenset
    times: frozenset
    tokens: str  # " tok tok ... " — lowercase alnum tokens, articles dropped

    @classmethod
    def from_text(cls, text: str) -> "DocIndex":
        text = text.replace("\r\n", "\n")
        numbers = {Decimal(m.group().replace(",", "")) for m in _DOC_NUMERAL.finditer(text)}
        numbers |= {Decimal(words_to_int(m.group())) for m in _DOC_WORDNUM.finditer(text)}
        dates = {norm_date(m.group()) for m in _DOC_DATE.finditer(text)}
        times = {norm_time(m.group()) for m in _DOC_TIME.finditer(text)}
        return cls(frozenset(numbers),
                   frozenset(d for d in dates if isinstance(d, str)),
                   frozenset(t for t in times if isinstance(t, str)),
                   " " + " ".join(_label_tokens(text)) + " ")

    def contains_label(self, s: str) -> bool:
        toks = _label_tokens(s)
        return bool(toks) and f" {' '.join(toks)} " in self.tokens
