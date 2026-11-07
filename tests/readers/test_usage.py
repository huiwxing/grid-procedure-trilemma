"""Usage normalisation for every provider shape the panel produces."""
import pytest

from procedures.readers import normalize_usage, price_call, provider_usage
from procedures.readers.tokens import estimate_tokens
from procedures.readers.usage import has_reasoning_text, message_reasoning_content, synthetic_usage


def parts(u):
    return {k: u[k] for k in ("input_total", "input_uncached", "input_cached", "cache_write", "cache_write_1h",
                              "output_total", "output_visible", "reasoning")}


def test_openai_cached_cache_write_and_reasoning_details():
    raw = {"prompt_tokens": 10_000, "completion_tokens": 1_500, "total_tokens": 11_500,
           "prompt_tokens_details": {"cached_tokens": 4_000, "cache_write_tokens": 1_000, "audio_tokens": 0},
           "completion_tokens_details": {"reasoning_tokens": 1_000, "audio_tokens": 0}}
    u = normalize_usage("openai_chat", "openai", raw)
    assert u["shape"] == "openai_chat" and u["complete"] and u["reasoning_basis"] == "provider_field"
    assert parts(u) == dict(input_total=10_000, input_uncached=5_000, input_cached=4_000, cache_write=1_000,
                            cache_write_1h=0, output_total=1_500, output_visible=500, reasoning=1_000)
    assert u["raw"] == raw and u["raw"] is not raw


def test_anthropic_cache_creation_read_1h_split_and_thinking():
    raw = {"input_tokens": 1_000, "cache_read_input_tokens": 50_000, "cache_creation_input_tokens": 20_000,
           "cache_creation": {"ephemeral_5m_input_tokens": 15_000, "ephemeral_1h_input_tokens": 5_000},
           "output_tokens": 800, "output_tokens_details": {"thinking_tokens": 312}}
    u = normalize_usage("anthropic_messages", "anthropic", raw)
    assert u["complete"]
    assert parts(u) == dict(input_total=71_000, input_uncached=1_000, input_cached=50_000, cache_write=20_000,
                            cache_write_1h=5_000, output_total=800, output_visible=488, reasoning=312)


def test_anthropic_without_thinking_breakdown_uses_absence_of_thinking_text():
    raw = {"input_tokens": 30, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "output_tokens": 5}
    disabled = normalize_usage("anthropic_messages", "anthropic", raw, reasoning_text_present=False)
    assert disabled["reasoning"] == 0 and disabled["reasoning_basis"] == "absent_no_reasoning_text"
    unknown = normalize_usage("anthropic_messages", "anthropic", raw, reasoning_text_present=True)
    assert unknown["reasoning"] is None and unknown["output_visible"] is None
    assert "output_tokens_details.thinking_tokens" in unknown["missing"]
    assert unknown["complete"]  # billed output is still known, so the call can be priced


def test_anthropic_split_mismatch_is_inconsistent():
    raw = {"input_tokens": 1, "cache_creation_input_tokens": 10, "output_tokens": 1,
           "cache_creation": {"ephemeral_5m_input_tokens": 3, "ephemeral_1h_input_tokens": 3}}
    u = normalize_usage("anthropic_messages", "anthropic", raw)
    assert not u["complete"] and "cache_creation split" in u["inconsistent"][0]


def test_gemini_thoughts_and_cached_content():
    raw = {"promptTokenCount": 12_000, "cachedContentTokenCount": 8_000, "candidatesTokenCount": 400,
           "thoughtsTokenCount": 600, "totalTokenCount": 13_000}
    u = normalize_usage("gemini_generate_content", "google-ai-studio", raw)
    assert u["complete"] and u["reasoning_basis"] == "provider_field"
    assert parts(u) == dict(input_total=12_000, input_uncached=4_000, input_cached=8_000, cache_write=0,
                            cache_write_1h=0, output_total=1_000, output_visible=400, reasoning=600)


def test_gemini_total_mismatch_is_inconsistent():
    raw = {"promptTokenCount": 10, "candidatesTokenCount": 5, "thoughtsTokenCount": 5, "totalTokenCount": 99}
    assert not normalize_usage("gemini_generate_content", "google-ai-studio", raw)["complete"]


def test_deepseek_prompt_cache_hit_and_miss():
    raw = {"prompt_tokens": 32_000, "prompt_cache_hit_tokens": 30_000, "prompt_cache_miss_tokens": 2_000,
           "completion_tokens": 500, "completion_tokens_details": {"reasoning_tokens": 200},
           "prompt_tokens_details": {"cached_tokens": 30_000}}
    u = normalize_usage("openai_chat", "deepseek", raw)
    assert u["shape"] == "deepseek_native" and u["complete"]
    assert parts(u) == dict(input_total=32_000, input_uncached=2_000, input_cached=30_000, cache_write=0,
                            cache_write_1h=0, output_total=500, output_visible=300, reasoning=200)
    bad = normalize_usage("openai_chat", "deepseek", {**raw, "prompt_tokens": 31_999})
    assert not bad["complete"] and "prompt_tokens" in bad["inconsistent"][0]


def test_hf_openai_compatible_minimal_and_reasoning_route():
    llama = normalize_usage("openai_chat", "huggingface",
                            {"prompt_tokens": 4_300, "completion_tokens": 300, "total_tokens": 4_600,
                             "prompt_tokens_details": None}, reasoning_text_present=False)
    assert llama["shape"] == "hf_openai_compatible" and llama["complete"]
    assert parts(llama)["input_uncached"] == 4_300 and llama["output_visible"] == 300 and llama["reasoning"] == 0
    assert "prompt_tokens_details.cached_tokens" in llama["defaulted_zero"]
    oss = normalize_usage("openai_chat", "huggingface",
                          {"prompt_tokens": 5_000, "completion_tokens": 1_300,
                           "prompt_tokens_details": {"cached_tokens": 1_000},
                           "completion_tokens_details": {"reasoning_tokens": 1_000}})
    assert parts(oss) == dict(input_total=5_000, input_uncached=4_000, input_cached=1_000, cache_write=0,
                              cache_write_1h=0, output_total=1_300, output_visible=300, reasoning=1_000)


@pytest.mark.parametrize("raw, problem", [
    (None, "usage"),
    ({"completion_tokens": 3}, "prompt_tokens"),
    ({"prompt_tokens": "12", "completion_tokens": 3}, "not a non-negative integer"),
    ({"prompt_tokens": 10, "completion_tokens": 3, "completion_tokens_details": {"reasoning_tokens": 4}},
     "exceeds billed output"),
    ({"prompt_tokens": 10, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 11}}, "exceed prompt"),
])
def test_missing_or_invalid_usage_is_never_complete(raw, problem):
    u = normalize_usage("openai_chat", "openai", raw)
    assert not u["complete"]
    assert any(problem in m for m in u["missing"] + u["inconsistent"])


def test_provider_usage_reads_the_native_object_not_the_adapter_derived_copy():
    raw_response = {"usage": {"input_tokens": 5, "output_tokens": 2}}
    derived = {"input_tokens": 5, "output_tokens": 2, "prompt_tokens": 5, "completion_tokens": 2}
    assert provider_usage("anthropic_messages", raw_response, derived) == {"input_tokens": 5, "output_tokens": 2}
    assert provider_usage("gemini_generate_content", {"usageMetadata": {"promptTokenCount": 3}}, {}) == {
        "promptTokenCount": 3}
    assert provider_usage("openai_chat", {"choices": []}, {"prompt_tokens": 1}) is None
    assert provider_usage("openai_chat", None, {"prompt_tokens": 1}) == {"prompt_tokens": 1}


def test_synthetic_usage_rejects_non_integer_tokens():
    with pytest.raises(ValueError):
        synthetic_usage(input_uncached=1.5, output_visible=1)


# ---- reasoning_content without a reasoning-token count (connectivity smoke run: Qwen3-8B on nscale) -------------

LEAKED_THOUGHT = ("Okay, the user wants me to return only the word READY, without any explanation. "
                  "Let me make sure I follow that exactly and do not add anything else.")


def nscale_response(reasoning_content, completion_tokens, prompt_tokens=18):
    """Shape of the Qwen/Qwen3-8B:nscale (vLLM) answers of the connectivity smoke (its call ledgers are not in this
    repository): message.reasoning_content, usage without completion_tokens_details, prompt_tokens_details null."""
    return {"id": "chatcmpl-synthetic", "object": "chat.completion", "created": 0, "model": "Qwen/Qwen3-8B",
            "system_fingerprint": "vllm-0.21.0-9aa9c829", "prompt_logprobs": None,
            "choices": [{"index": 0, "finish_reason": "stop", "stop_reason": None,
                         "message": {"role": "assistant", "content": "\n\nREADY",
                                     "reasoning_content": reasoning_content, "tool_calls": []}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens, "prompt_tokens_details": None}}


def normalise(raw):
    content = message_reasoning_content("openai_chat", raw)
    return normalize_usage("openai_chat", "huggingface", provider_usage("openai_chat", raw, None),
                           reasoning_text_present=has_reasoning_text(content), reasoning_content=content)


def test_reasoning_content_without_reasoning_tokens_is_estimated_but_not_billed(panel, book):
    raw = nscale_response(LEAKED_THOUGHT, 104)
    u = normalise(raw)
    assert u["reasoning_tokens_estimated"] is True and u["reasoning_content_chars"] == len(LEAKED_THOUGHT)
    assert u["reasoning_tokens_estimate"] == estimate_tokens(LEAKED_THOUGHT) > 0
    # the provider did not report a count, so the normalised reasoning stays unknown
    assert u["reasoning"] is None and u["output_visible"] is None and u["reasoning_basis"] == "unreported"
    assert "completion_tokens_details.reasoning_tokens" in u["missing"] and u["complete"]
    assert u["output_total"] == 104 and u["input_total"] == 18
    without = normalize_usage("openai_chat", "huggingface", provider_usage("openai_chat", raw, None),
                              reasoning_text_present=True)
    assert without["reasoning_tokens_estimated"] is False and without["reasoning_content_chars"] is None
    assert without["reasoning_tokens_estimate"] is None
    pp = panel.get("C2_qwen3_8b_hf_nscale_nothink")
    billed = (18 * 0.07 + 104 * 0.18) / 1e6  # reasoning is already inside completion_tokens
    assert price_call(pp, u, book=book)["usd"] == price_call(pp, without, book=book)["usd"] == pytest.approx(billed)


def test_whitespace_only_reasoning_content_is_no_reasoning():
    u = normalise(nscale_response("\n\n", 6, prompt_tokens=22))  # the /no_think shape of the connectivity smoke
    assert u["reasoning"] == 0 and u["reasoning_basis"] == "absent_no_reasoning_text" and u["output_visible"] == 6
    assert u["reasoning_content_chars"] == 2
    assert u["reasoning_tokens_estimated"] is False and u["reasoning_tokens_estimate"] is None


def test_a_reported_reasoning_count_is_never_replaced_by_an_estimate():
    raw = {"prompt_tokens": 41, "completion_tokens": 19, "completion_tokens_details": {"reasoning_tokens": 16},
           "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 41}  # DeepSeek thinking shape (connectivity smoke)
    u = normalize_usage("openai_chat", "deepseek", raw, reasoning_text_present=True, reasoning_content="x" * 62)
    assert u["reasoning"] == 16 and u["reasoning_basis"] == "provider_field"
    assert u["reasoning_content_chars"] == 62 and u["reasoning_tokens_estimated"] is False
    anthropic = normalize_usage("anthropic_messages", "anthropic", {"input_tokens": 3, "output_tokens": 2},
                                reasoning_text_present=True, reasoning_content="thinking")
    assert anthropic["reasoning_tokens_estimated"] is False  # only OpenAI-compatible shapes carry reasoning_content


@pytest.mark.parametrize("text, expected", [
    (None, False), ("", False), ("\n\n", False), (" \t\r\n", False), (42, False), ("\n\nhmm", True)])
def test_has_reasoning_text_treats_whitespace_as_none(text, expected):
    assert has_reasoning_text(text) is expected


def test_message_reasoning_content_reads_only_openai_compatible_strings():
    assert message_reasoning_content("openai_chat", nscale_response("t", 1)) == "t"
    assert message_reasoning_content("openai_chat", nscale_response(None, 1)) is None
    assert message_reasoning_content("anthropic_messages", nscale_response("t", 1)) is None
    assert message_reasoning_content("openai_chat", {"choices": []}) is None
    assert message_reasoning_content("openai_chat", None) is None
