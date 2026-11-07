"""HTTP GET for the corpus rebuild (scripts/corpus/rebuild.py): one request, the identifying User-Agent, the final URL
after redirects.
"""
from __future__ import annotations

import urllib.request

USER_AGENT = "grid-procedure-trilemma/1.0"


def http_get(url: str, accept: str | None = None, timeout: float = 120.0) -> tuple[int, str, dict[str, str], bytes]:
    """GET with the project User-Agent; returns (status, final_url, lower-cased headers, body)."""
    headers = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        return resp.status, resp.geturl(), {k.lower(): v for k, v in resp.headers.items()}, body
