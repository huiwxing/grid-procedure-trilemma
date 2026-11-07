"""The reader tests run offline (tests/conftest.py `offline`); a scripted transport and a scripted clock for the
vendored client."""
from __future__ import annotations

import json

import pytest


@pytest.fixture(autouse=True)
def _offline(offline):
    """No network and dummy credentials for every test here."""


class FakeTransport:
    """Stands in for the vendored client's send function; records every request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, headers, payload, timeout):
        self.calls.append({"url": url, "payload": payload})
        if not self.responses:
            raise AssertionError("FakeTransport ran out of scripted responses")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        code, body = response
        return code, {"x-request-id": f"req-{len(self.calls)}"}, json.dumps(body).encode("utf-8")


@pytest.fixture
def fake_transport():
    return FakeTransport


@pytest.fixture
def clock(monkeypatch):
    """Scripted start times of the vendored client's attempts, consumed one per attempt."""
    from datetime import datetime
    import procedures.readers.client.client as client_module
    times = []

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromisoformat(times.pop(0))

    monkeypatch.setattr(client_module, "datetime", Clock)
    return times
