"""Normalise each provider's usage object into one token ledger.

Output fields (all non-negative ints, or None when the provider did not report
enough to know):

    input_total     = input_uncached + input_cached + cache_write
    input_uncached  billed at the plain input price
    input_cached    cache hits / reads
    cache_write     tokens written to the prompt cache (cache_write_1h is the
                    Anthropic 1-hour-TTL subset of it)
    output_total    = output_visible + reasoning; the billed output count
    output_visible  answer tokens
    reasoning       reasoning / thinking tokens (billed as output by every panel
                    provider; never added on top of output_total)

``raw`` keeps a deep copy of the provider object. ``missing`` lists required
provider fields that were absent, ``defaulted_zero`` optional fields read as 0
because the provider omitted them, ``inconsistent`` any arithmetic or type
violation. Only ``complete`` usage (both totals known, nothing inconsistent)
is priced.

Reasoning text without a reasoning count (OpenAI-compatible shapes):
    reasoning_content_chars     len(choices[0].message.reasoning_content), or None when
                                the message carries no such string
    reasoning_tokens_estimated  True when the provider returned non-whitespace
                                reasoning_content but no completion_tokens_details.
                                reasoning_tokens (observed in the connectivity smoke run:
                                nscale, deepinfra gpt-oss)
    reasoning_tokens_estimate   tokens.estimate_tokens(reasoning_content) in that case,
                                else None; a planning figure only
``reasoning`` stays None (unreported) and nothing is billed differently: these
providers already count reasoning inside completion_tokens (= output_total).
Whitespace-only reasoning (Qwen3's empty think block) is no reasoning text.

Provider shapes [V = documented in the provider's public API reference (as of 2026-09-24;
the reference is named below); I = inferred, consistent with the connectivity smoke run]:
  openai_chat / hf_openai_compatible: prompt_tokens ⊇ prompt_tokens_details.{cached_tokens,
      cache_write_tokens}; completion_tokens ⊇ completion_tokens_details.reasoning_tokens [V openai_chat_ref;
      subset semantics of cache_write_tokens I]
  deepseek_native: prompt_tokens = prompt_cache_hit_tokens + prompt_cache_miss_tokens;
      completion_tokens ⊇ completion_tokens_details.reasoning_tokens [V deepseek_chat_ref]
  anthropic_messages: input_tokens + cache_read_input_tokens + cache_creation_input_tokens = total input;
      cache_creation.{ephemeral_5m,ephemeral_1h}_input_tokens split the writes; output_tokens ⊇
      output_tokens_details.thinking_tokens [V anthropic_prompt_caching; adaptive-thinking]
  gemini_generate_content: promptTokenCount ⊇ cachedContentTokenCount; billed output =
      candidatesTokenCount + thoughtsTokenCount [I field names; the client adapter's convention]
"""
from __future__ import annotations

from copy import deepcopy

from procedures.readers.tokens import estimate_tokens

OPENAI_COMPATIBLE = frozenset({"openai_chat", "hf_openai_compatible", "deepseek_native"})


def has_reasoning_text(text) -> bool:
    """True when ``text`` carries reasoning. Whitespace-only text (Qwen3's empty think block, returned
    by nscale as a reasoning_content of two newlines under the /no_think soft switch) is no reasoning."""
    return isinstance(text, str) and bool(text.strip())


def message_reasoning_content(adapter: str, raw_response) -> str | None:
    """``choices[0].message.reasoning_content`` of an OpenAI-compatible response, when it is a string."""
    if adapter != "openai_chat" or not isinstance(raw_response, dict):
        return None
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    value = message.get("reasoning_content") if isinstance(message, dict) else None
    return value if isinstance(value, str) else None


def usage_shape(adapter: str, provider: str, usage: dict | None) -> str:
    if adapter == "anthropic_messages":
        return "anthropic_messages"
    if adapter == "gemini_generate_content":
        return "gemini_generate_content"
    if adapter != "openai_chat":
        raise ValueError(f"No usage shape for adapter {adapter!r}")
    if isinstance(usage, dict) and ("prompt_cache_hit_tokens" in usage or "prompt_cache_miss_tokens" in usage):
        return "deepseek_native"
    return "hf_openai_compatible" if provider == "huggingface" else "openai_chat"


def provider_usage(adapter: str, raw_response: dict | None, fallback: dict | None) -> dict | None:
    """The provider's own usage object (the client's adapters add derived keys to result.usage)."""
    if isinstance(raw_response, dict):
        key = "usageMetadata" if adapter == "gemini_generate_content" else "usage"
        if key in raw_response:
            value = raw_response[key]
            return deepcopy(value) if isinstance(value, dict) else None
        return None
    return deepcopy(fallback) if isinstance(fallback, dict) and fallback else None


_ABSENT = object()


class _Reader:
    def __init__(self, usage):
        self.usage = usage if isinstance(usage, dict) else {}
        self.missing, self.defaulted, self.inconsistent = [], [], []

    def _lookup(self, path):
        node = self.usage
        for key in path:
            if not isinstance(node, dict) or node.get(key) is None:
                return _ABSENT
            node = node[key]
        return node

    def get(self, *path, required=False):
        value = self._lookup(path)
        if value is _ABSENT:
            if required:
                self.missing.append(".".join(path))
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            self.inconsistent.append(f"{'.'.join(path)} is not a non-negative integer: {value!r}")
            return None
        return value

    def zero(self, *path):
        if self._lookup(path) is _ABSENT:
            self.defaulted.append(".".join(path))
            return 0
        return self.get(*path)


def _reasoning(reader, value, basis_field, reasoning_text_present):
    if value is not None:
        return value, "provider_field"
    if reasoning_text_present is False:
        return 0, "absent_no_reasoning_text"
    reader.missing.append(basis_field)
    return None, "unreported"


def normalize_usage(adapter: str, provider: str, usage: dict | None, *,
                    reasoning_text_present: bool | None = None, reasoning_content: str | None = None) -> dict:
    shape = usage_shape(adapter, provider, usage)
    r = _Reader(usage)
    cache_write_1h = 0
    if usage is None or not isinstance(usage, dict):
        r.missing.append("usage")
        inp = uncached = cached = write = out = reasoning = None
        basis = "unreported"
    elif shape in {"openai_chat", "hf_openai_compatible"}:
        inp = r.get("prompt_tokens", required=True)
        cached = r.zero("prompt_tokens_details", "cached_tokens")
        write = r.zero("prompt_tokens_details", "cache_write_tokens")
        uncached = None if None in (inp, cached, write) else inp - cached - write
        out = r.get("completion_tokens", required=True)
        reasoning, basis = _reasoning(r, r.get("completion_tokens_details", "reasoning_tokens"),
                                      "completion_tokens_details.reasoning_tokens", reasoning_text_present)
    elif shape == "deepseek_native":
        hit = r.get("prompt_cache_hit_tokens", required=True)
        miss = r.get("prompt_cache_miss_tokens", required=True)
        reported = r.get("prompt_tokens")
        cached, uncached, write = hit, miss, 0
        inp = None if None in (hit, miss) else hit + miss
        if inp is not None and reported is not None and reported != inp:
            r.inconsistent.append(f"prompt_tokens {reported} != hit {hit} + miss {miss}")
        out = r.get("completion_tokens", required=True)
        reasoning, basis = _reasoning(r, r.get("completion_tokens_details", "reasoning_tokens"),
                                      "completion_tokens_details.reasoning_tokens", reasoning_text_present)
    elif shape == "anthropic_messages":
        uncached = r.get("input_tokens", required=True)
        cached = r.zero("cache_read_input_tokens")
        write = r.zero("cache_creation_input_tokens")
        split = usage.get("cache_creation")
        if isinstance(split, dict):
            cache_write_1h = r.zero("cache_creation", "ephemeral_1h_input_tokens")
            five = r.zero("cache_creation", "ephemeral_5m_input_tokens")
            if None not in (write, cache_write_1h, five) and five + cache_write_1h != write:
                r.inconsistent.append(f"cache_creation split {five}+{cache_write_1h} != cache_creation_input_tokens {write}")
        inp = None if None in (uncached, cached, write) else uncached + cached + write
        out = r.get("output_tokens", required=True)
        reasoning, basis = _reasoning(r, r.get("output_tokens_details", "thinking_tokens"),
                                      "output_tokens_details.thinking_tokens", reasoning_text_present)
    else:  # gemini_generate_content
        inp = r.get("promptTokenCount", required=True)
        cached = r.zero("cachedContentTokenCount")
        write = 0
        uncached = None if None in (inp, cached) else inp - cached
        visible = r.zero("candidatesTokenCount")
        # Gemini omits zero-valued counters; an absent thoughtsTokenCount means no thinking tokens.
        reasoning = r.zero("thoughtsTokenCount")
        basis = "provider_omitted_zero" if "thoughtsTokenCount" in r.defaulted else "provider_field"
        out = None if None in (visible, reasoning) else visible + reasoning
        total = r.get("totalTokenCount")
        tool = r.zero("toolUsePromptTokenCount")
        if None not in (total, inp, out, tool) and total != inp + out + tool:
            r.inconsistent.append(f"totalTokenCount {total} != prompt {inp} + candidates+thoughts {out} + toolUse {tool}")
    if uncached is not None and uncached < 0:
        r.inconsistent.append(f"cached {cached} + cache_write {write} exceed prompt tokens {inp}")
    if reasoning is not None and out is not None and reasoning > out:
        r.inconsistent.append(f"reasoning {reasoning} exceeds billed output {out}")
    visible = None if None in (out, reasoning) else out - reasoning
    complete = inp is not None and out is not None and uncached is not None and not r.inconsistent
    content = reasoning_content if isinstance(reasoning_content, str) else None
    estimated = shape in OPENAI_COMPATIBLE and basis == "unreported" and has_reasoning_text(content)
    return {"shape": shape, "input_total": inp, "input_uncached": uncached, "input_cached": cached,
            "cache_write": write, "cache_write_1h": cache_write_1h, "output_total": out,
            "output_visible": visible, "reasoning": reasoning, "reasoning_basis": basis,
            "reasoning_content_chars": None if content is None else len(content),
            "reasoning_tokens_estimated": estimated,
            "reasoning_tokens_estimate": estimate_tokens(content) if estimated else None,
            "complete": complete, "missing": r.missing, "defaulted_zero": r.defaulted,
            "inconsistent": r.inconsistent, "raw": deepcopy(usage) if isinstance(usage, dict) else None}


def synthetic_usage(*, input_uncached: int, input_cached: int = 0, cache_write: int = 0,
                    output_visible: int, reasoning: int = 0, shape: str = "estimate") -> dict:
    """A normalised usage record for planning; priced by the same price_call as real calls."""
    values = dict(input_uncached=input_uncached, input_cached=input_cached, cache_write=cache_write,
                  output_visible=output_visible, reasoning=reasoning)
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return {"shape": shape, "input_total": input_uncached + input_cached + cache_write,
            "input_uncached": input_uncached, "input_cached": input_cached, "cache_write": cache_write,
            "cache_write_1h": 0, "output_total": output_visible + reasoning, "output_visible": output_visible,
            "reasoning": reasoning, "reasoning_basis": "assumption", "complete": True, "missing": [],
            "defaulted_zero": [], "inconsistent": [], "raw": None}
