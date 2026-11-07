"""HTTP client for the provider APIs used by the readers: adapters per API shape, retries, usage normalisation."""
from .client import Client
from .config import ConfigurationError, InferenceProfile

__all__ = ["Client", "ConfigurationError", "InferenceProfile"]
