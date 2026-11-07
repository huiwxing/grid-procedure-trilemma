"""Bounded HTTP requests with no cache, sampling fallback, or GPU dependency."""
from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import os
import time
import uuid
import urllib.error
import urllib.request

from .adapters import build_request, parse_response
from .config import ConfigurationError, InferenceProfile
from .result import GenerationResult


RESPONSE_HEADERS = {"content-type", "date", "retry-after", "x-request-id", "request-id",
                    "x-hf-request-id", "x-inference-provider", "x-hf-inference-provider",
                    "x-system-fingerprint"}


def response_metadata(doc, headers):
    """Only service-reported identifiers; configuration is not observation."""
    doc = doc if isinstance(doc, dict) else {}

    def first_text(*values):
        return next((value for value in values if isinstance(value, str) and value.strip()), None)

    return {
        "server_request_id": first_text(headers.get("x-request-id"), headers.get("request-id"),
            headers.get("x-hf-request-id"), doc.get("id"), doc.get("responseId")),
        "observed_provider": first_text(doc.get("provider"), headers.get("x-inference-provider"),
            headers.get("x-hf-inference-provider")),
        "system_fingerprint": first_text(doc.get("system_fingerprint"), headers.get("x-system-fingerprint")),
    }


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward credentials to an unconfigured endpoint.


def http_transport(url, headers, payload, timeout):
    request = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False,
                      allow_nan=False).encode("utf-8"), headers=headers, method="POST")
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers or {}), exc.read()


class Client:
    def __init__(self, profile, *, transport=None, sleep=None):
        self.profile = profile if isinstance(profile, InferenceProfile) else InferenceProfile.from_dict(profile)
        self._transport = transport or http_transport
        self._sleep = sleep or time.sleep

    def identity(self):
        return self.profile.identity()

    def _headers(self):
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        secrets = []
        if self.profile.adapter == "anthropic_messages":
            headers["anthropic-version"] = "2023-06-01"
        if self.profile.api_key_env:
            key = os.environ.get(self.profile.api_key_env)
            if not key:
                raise ConfigurationError(f"Set credential environment variable {self.profile.api_key_env}")
            if "\r" in key or "\n" in key:
                raise ConfigurationError("API credentials cannot contain newlines")
            secrets.append(key)
            if self.profile.adapter == "anthropic_messages":
                headers["x-api-key"] = key
            elif self.profile.adapter == "gemini_generate_content":
                headers["x-goog-api-key"] = key
            else:
                headers["Authorization"] = "Bearer " + key
        for header, env_name in self.profile.headers_env.items():
            value = os.environ.get(env_name)
            if not value:
                raise ConfigurationError(f"Set header environment variable {env_name}")
            if "\r" in value or "\n" in value:
                raise ConfigurationError("HTTP header values cannot contain newlines")
            headers[header] = value
            secrets.append(value)
        return headers, secrets

    def generate(self, messages, generation=None, request_id=None):
        endpoint, payload = build_request(self.profile, messages, generation)
        headers, secrets = self._headers()

        def redact(value):
            if isinstance(value, str):
                for secret in secrets:
                    value = value.replace(secret, "[REDACTED]")
                return value
            if isinstance(value, dict):
                return {redact(k): redact(v) for k, v in value.items()}
            if isinstance(value, list):
                return [redact(v) for v in value]
            return value

        result = GenerationResult(requested_model=self.profile.request_model,
                request_id=request_id or str(uuid.uuid4()), request=payload, identity=self.identity())
        retry_codes = {408, 425, 429, 500, 502, 503, 504}
        for attempt in range(1, self.profile.max_attempts + 1):
            started = time.monotonic()
            observed_at = datetime.now(timezone.utc).isoformat()
            response_headers = {}
            # Top-level provenance describes the terminal attempt. Earlier HTTP
            # responses remain in attempts, including before a terminal timeout.
            result.response_headers = {}
            for key, value in response_metadata(None, {}).items():
                setattr(result, key, value)
            try:
                code, response_headers, raw = self._transport(endpoint, headers, payload, self.profile.timeout)
                result.response_headers = redact({key.lower(): value for key, value in response_headers.items()
                    if key.lower() in RESPONSE_HEADERS})
                for key, value in response_metadata(None, result.response_headers).items():
                    setattr(result, key, value)
                raw_text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
                entry = {"attempt": attempt, "http_status": code, "started_at": observed_at,
                         "elapsed_seconds": time.monotonic() - started,
                         "response_headers": result.response_headers}
                if not 200 <= code < 300:
                    entry["body"] = redact(raw_text)
                    result.error = {"kind": "http", "http_status": code,
                                    "message": f"HTTP {code}", "body": redact(raw_text)}
                    retryable = code in retry_codes
                else:
                    result.attempts.append(entry)
                    result.raw_text = redact(raw_text)
                    try:
                        doc = json.loads(raw_text)
                        result.raw_response = redact(doc) if isinstance(doc, dict) else None
                        for key, value in response_metadata(doc, result.response_headers).items():
                            setattr(result, key, redact(value))
                        parsed = parse_response(self.profile.adapter, doc)
                        for key, value in parsed.items():
                            setattr(result, key, redact(value))
                    except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
                        result.error = {"kind": "protocol", "message": redact(str(exc))}
                    return result  # Semantic output and malformed envelopes are terminal observations.
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                entry = {"attempt": attempt, "started_at": observed_at,
                         "elapsed_seconds": time.monotonic() - started,
                         "error": {"kind": "transport", "message": redact(str(exc))}}
                result.error = entry["error"]
                retryable = True
            result.attempts.append(entry)
            if not retryable or attempt == self.profile.max_attempts:
                return result
            delay = self.profile.retry_backoff * 2 ** (attempt - 1)
            retry_after = next((v for k, v in response_headers.items() if k.lower() == "retry-after"), None)
            if retry_after is not None:
                try:
                    delay = max(delay, float(retry_after))
                except (TypeError, ValueError):
                    try:
                        delay = max(delay, (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds())
                    except (TypeError, ValueError, OverflowError):
                        pass
            # A server cannot turn a bounded retry into an unbounded wait.
            delay = min(max(delay, 0), self.profile.timeout)
            result.attempts[-1]["retry_after_seconds"] = delay
            self._sleep(delay)
        return result
