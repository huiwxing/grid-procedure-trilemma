"""The outcome of one generation request: text, reasoning, status, usage, provenance of every attempt."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class GenerationResult:
    text: str = ""
    reasoning: str | None = None
    status: str = "error"
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)
    raw_response: dict | None = None
    raw_text: str | None = None
    requested_model: str = ""
    served_model: str | None = None
    request_id: str | None = None
    attempts: list = field(default_factory=list)
    error: dict | None = None
    request: dict = field(default_factory=dict)
    identity: dict = field(default_factory=dict)
    server_request_id: str | None = None
    observed_provider: str | None = None
    system_fingerprint: str | None = None
    response_headers: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)
