"""Configuration contains service choices, never a whitelist of model names."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import hashlib
import json
import math
import os
import re
from urllib.parse import urlsplit


class ConfigurationError(ValueError):
    """An explicit request cannot be represented by the selected service."""


GENERATION_KEYS = {"temperature", "top_p", "max_output_tokens", "seed", "stop", "reasoning_effort"}
ADAPTERS = {"openai_chat", "anthropic_messages", "gemini_generate_content"}
AUTH_HEADERS = {"authorization", "proxy-authorization", "x-api-key", "x-goog-api-key", "api-key", "cookie"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _expand(value):
    def replace(match):
        name = match.group(1)
        if not os.environ.get(name):
            raise ConfigurationError(f"Environment variable {name} is not set")
        return os.environ[name]
    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", replace, value)


def check_generation(generation):
    if not isinstance(generation, dict):
        raise ConfigurationError("generation must be an object")
    unknown = set(generation) - GENERATION_KEYS
    if unknown:
        raise ConfigurationError(f"Unknown generation settings: {sorted(unknown)}; use extra_body for provider options")
    for name, value in generation.items():
        if value is None:
            continue  # Explicit null unsets a profile default.
        if name in {"temperature", "top_p"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ConfigurationError(f"{name} must be finite and numeric")
            if value < 0 or (name == "top_p" and not 0 < value <= 1):
                raise ConfigurationError(f"Invalid {name}")
        if name in {"max_output_tokens", "seed"}:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigurationError(f"{name} must be an integer")
            if name == "max_output_tokens" and value <= 0:
                raise ConfigurationError("max_output_tokens must be positive")
        if name == "stop" and not (isinstance(value, str) or
                isinstance(value, list) and all(isinstance(x, str) for x in value)):
            raise ConfigurationError("stop must be a string or a list of strings")
        if name == "reasoning_effort" and not isinstance(value, str):
            raise ConfigurationError("reasoning_effort must be a string")


@dataclass(frozen=True)
class InferenceProfile:
    adapter: str
    base_url: str
    model: str
    provider: str = ""
    api_key_env: str | None = None
    generation: dict = field(default_factory=dict)
    extra_body: dict = field(default_factory=dict)
    capabilities: dict = field(default_factory=dict)
    parameter_map: dict = field(default_factory=dict)
    timeout: float = 120
    max_attempts: int = 3
    max_workers: int = 4
    retry_backoff: float = 1
    headers_env: dict = field(default_factory=dict)
    hf_provider: str | None = None

    def __post_init__(self):
        if self.adapter not in ADAPTERS:
            raise ConfigurationError(f"Unknown adapter {self.adapter!r}; available: {sorted(ADAPTERS)}")
        for key in ("base_url", "model", "provider"):
            value = getattr(self, key)
            if not isinstance(value, str):
                raise ConfigurationError(f"{key} must be a string")
            object.__setattr__(self, key, _expand(value))
        if not self.model.strip():
            raise ConfigurationError("model must not be empty")
        url = urlsplit(self.base_url)
        if url.scheme not in {"http", "https"} or not url.netloc:
            raise ConfigurationError("base_url must be an absolute http(s) URL")
        if url.username or url.password or url.query or url.fragment:
            raise ConfigurationError("base_url must not contain credentials, query parameters or a fragment")
        if self.hf_provider is not None:
            if not isinstance(self.hf_provider, str):
                raise ConfigurationError("hf_provider must be a provider identifier")
            object.__setattr__(self, "hf_provider", _expand(self.hf_provider))
            if (not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", self.hf_provider)
                    or self.hf_provider in {"auto", "fastest", "cheapest", "preferred"}):
                raise ConfigurationError("hf_provider must name a fixed provider, not an automatic routing policy")
            if (self.adapter != "openai_chat" or url.scheme != "https"
                    or url.hostname != "router.huggingface.co" or url.port not in {None, 443}
                    or url.path.rstrip("/") not in {"/v1", "/v1/chat/completions"}):
                raise ConfigurationError("hf_provider requires the Hugging Face HTTPS /v1 chat router")
            model_id, separator, suffix = self.model.partition(":")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", model_id):
                raise ConfigurationError("hf_provider requires a Hugging Face organization/model ID")
            if separator and suffix != self.hf_provider:
                raise ConfigurationError("model suffix conflicts with hf_provider; use the canonical model ID")
        if self.api_key_env is not None and (not isinstance(self.api_key_env, str) or
                not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env)):
            raise ConfigurationError("api_key_env must name an environment variable")
        for key in ("generation", "extra_body", "capabilities", "parameter_map", "headers_env"):
            if not isinstance(getattr(self, key), dict):
                raise ConfigurationError(f"{key} must be an object")
            object.__setattr__(self, key, json.loads(json.dumps(getattr(self, key), allow_nan=False)))
        check_generation(self.generation)
        if set(self.capabilities) - GENERATION_KEYS or any(type(x) is not bool for x in self.capabilities.values()):
            raise ConfigurationError("capabilities maps canonical generation names to booleans")
        if set(self.parameter_map) - GENERATION_KEYS or any(not isinstance(x, str) or not x for x in self.parameter_map.values()):
            raise ConfigurationError("parameter_map maps canonical generation names to wire names")
        forbidden = {"model", "messages", "contents", "system", "systemInstruction", "stream", "n"}
        if forbidden.intersection(self.parameter_map.values()):
            raise ConfigurationError("parameter_map cannot overwrite message/model/stream fields")
        if forbidden.intersection(self.extra_body):
            raise ConfigurationError("extra_body cannot overwrite message/model/stream fields or request multiple choices")
        for header, name in self.headers_env.items():
            if not isinstance(header, str) or not re.fullmatch(r"[A-Za-z0-9-]+", header):
                raise ConfigurationError("Invalid HTTP header name")
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ConfigurationError("headers_env values must name environment variables")
        for key in ("timeout", "retry_backoff"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ConfigurationError(f"Invalid {key}")
        if self.timeout == 0:
            raise ConfigurationError("timeout must be positive")
        for key in ("max_attempts", "max_workers"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigurationError(f"{key} must be a positive integer")

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ConfigurationError("An inference profile must be an object")
        try:
            return cls(**value)
        except TypeError as exc:
            raise ConfigurationError(f"Invalid profile fields: {exc}") from exc

    def identity(self):
        # Environment-variable names are provenance; resolved credentials are not.
        identity = asdict(self)
        # Existing profiles must keep their historical identity and fingerprints.
        if self.hf_provider is None:
            identity.pop("hf_provider")
        return identity

    @property
    def request_model(self):
        """Wire model; generic provider labels never select a service."""
        if self.hf_provider is not None and ":" not in self.model:
            return f"{self.model}:{self.hf_provider}"
        return self.model

    @property
    def fingerprint(self):
        return digest(self.semantic_identity())

    def semantic_identity(self):
        # Changing concurrency, timeout, retry policy, or credential-variable
        # names does not turn a completed scientific sample into a new sample.
        identity = {key: value for key, value in self.identity().items() if key not in
                    {"timeout", "max_attempts", "max_workers", "retry_backoff", "api_key_env", "headers_env"}}
        # Version/beta/routing headers can change model behavior. Bind their
        # effective values without persisting them or requiring credentials.
        identity["header_fingerprints"] = {
            header.lower(): digest(os.environ.get(name))
            for header, name in self.headers_env.items() if header.lower() not in AUTH_HEADERS}
        return identity
