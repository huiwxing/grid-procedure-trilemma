"""Offline-testable inference layer: pinned panel, usage normalisation, pricing, call ledger, runner.

The HTTP client for the provider APIs is ``procedures.readers.client``; the panel, prompts, pricing, ledger and runner
live in the sibling modules.
"""
from procedures.readers.ledger import Ledger, LedgerError, SpendCapExceeded, request_hash
from procedures.readers.pricing import PriceBook, PricingError, load_price_book, price_call
from procedures.readers.panel import Panel, PanelError, PanelProfile, TrapViolation, check_wire, load_panel
from procedures.readers.executor import Assumptions, Cell, estimate_cells, run_cells, scenario, upper_bound_usd
from procedures.readers.tokens import estimate_message_tokens, estimate_tokens, token_method
from procedures.readers.usage import normalize_usage, provider_usage

__all__ = ["Assumptions", "Cell", "Ledger", "LedgerError", "Panel", "PanelError", "PanelProfile",
           "PriceBook", "PricingError", "SpendCapExceeded", "TrapViolation", "check_wire",
           "estimate_cells", "estimate_message_tokens", "estimate_tokens", "load_panel",
           "load_price_book", "normalize_usage", "price_call", "provider_usage", "request_hash",
           "run_cells", "scenario", "token_method", "upper_bound_usd"]
