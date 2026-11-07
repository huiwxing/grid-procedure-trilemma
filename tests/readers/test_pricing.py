"""price_call against hand-computed cases (rates from data/prices/prices.json, the 2026-09-24 snapshot, $/1M tokens)."""
from datetime import datetime, timezone
import json

import pytest

from procedures.readers import PricingError, load_price_book, normalize_usage, price_call

UTC = timezone.utc


def priced(book, key, adapter, provider, raw, when=None, **kw):
    usage = normalize_usage(adapter, provider, raw, **kw)
    return price_call(key, usage, book=book, observed_at=when)


def test_case1_gpt6_sol_cached_write_and_reasoning(book):
    # 5,000 uncached x 2.00 + 4,000 cached x 0.20 + 1,000 write x 2.50 + 1,500 output (incl. 1,000 reasoning) x 10.00
    #   = 10,000 + 800 + 2,500 + 15,000 = 28,300 / 1e6
    out = priced(book, "gpt-6-sol", "openai_chat", "openai",
                 {"prompt_tokens": 10_000, "completion_tokens": 1_500,
                  "prompt_tokens_details": {"cached_tokens": 4_000, "cache_write_tokens": 1_000},
                  "completion_tokens_details": {"reasoning_tokens": 1_000}})
    assert out["status"] == "priced" and out["usd"] == pytest.approx(0.0283, abs=1e-12)
    assert out["breakdown"]["output"] == pytest.approx(0.015, abs=1e-12)  # reasoning is not billed twice


def test_case2_sonnet5_cache_read_write_5m_and_1h(book):
    # 1,000 x 2.00 + 50,000 x 0.20 + 15,000 x 2.50 (5m) + 5,000 x 4.00 (1h) + 800 x 10.00
    #   = 2,000 + 10,000 + 37,500 + 20,000 + 8,000 = 77,500 / 1e6
    out = priced(book, "claude-sonnet-5", "anthropic_messages", "anthropic",
                 {"input_tokens": 1_000, "cache_read_input_tokens": 50_000, "cache_creation_input_tokens": 20_000,
                  "cache_creation": {"ephemeral_5m_input_tokens": 15_000, "ephemeral_1h_input_tokens": 5_000},
                  "output_tokens": 800})
    assert out["usd"] == pytest.approx(0.0775, abs=1e-12)


def test_case3_gemini_flash_thoughts_and_2027_price_switch(book):
    raw = {"promptTokenCount": 12_000, "cachedContentTokenCount": 8_000, "candidatesTokenCount": 400,
           "thoughtsTokenCount": 600}
    # 2026: 4,000 x 0.75 + 8,000 x 0.075 + 1,000 x 3.75 = 3,000 + 600 + 3,750 = 7,350 / 1e6
    before = priced(book, "gemini-3.8-flash", "gemini_generate_content", "google-ai-studio", raw,
                    datetime(2026, 10, 1, 12, tzinfo=UTC))
    assert before["usd"] == pytest.approx(0.00735, abs=1e-12) and before["rates"]["variant"] == "standard"
    # 2027: 4,000 x 1.50 + 8,000 x 0.15 + 1,000 x 7.50 = 6,000 + 1,200 + 7,500 = 14,700 / 1e6
    after = priced(book, "gemini-3.8-flash", "gemini_generate_content", "google-ai-studio", raw,
                   datetime(2027, 1, 2, 12, tzinfo=UTC))
    assert after["usd"] == pytest.approx(0.0147, abs=1e-12) and after["rates"]["variant"] == "from_2027_01_01"


def test_case4_deepseek_direct_peak_offpeak_and_weekend(book):
    raw = {"prompt_tokens": 32_000, "prompt_cache_hit_tokens": 30_000, "prompt_cache_miss_tokens": 2_000,
           "completion_tokens": 500, "completion_tokens_details": {"reasoning_tokens": 200}}
    key = "deepseek-flash(V4.1-Flash)@direct"
    # Tue 2026-09-29 02:30 UTC is peak: 2,000 x 0.30 + 30,000 x 0.006 + 500 x 1.20 = 600 + 180 + 600 = 1,380 / 1e6
    peak = priced(book, key, "openai_chat", "deepseek", raw, datetime(2026, 9, 29, 2, 30, tzinfo=UTC))
    assert peak["usd"] == pytest.approx(0.00138, abs=1e-12) and peak["rates"]["variant"] == "peak"
    # Tue 12:00 UTC is off-peak: 2,000 x 0.15 + 30,000 x 0.003 + 500 x 0.60 = 300 + 90 + 300 = 690 / 1e6
    off = priced(book, key, "openai_chat", "deepseek", raw, datetime(2026, 9, 29, 12, tzinfo=UTC))
    assert off["usd"] == pytest.approx(0.00069, abs=1e-12)
    weekend = priced(book, key, "openai_chat", "deepseek", raw, datetime(2026, 9, 26, 2, 30, tzinfo=UTC))
    assert weekend["rates"]["variant"] == "offpeak"
    planning = priced(book, key, "openai_chat", "deepseek", raw)
    assert planning["usd"] == pytest.approx(0.00138, abs=1e-12) and "max" in planning["rates"]["variant"]


def test_case5_hf_routes_flat_and_unpublished_cache_price(book):
    # Llama-3.1-8B @deepinfra: 4,300 x 0.02 + 300 x 0.05 = 86 + 15 = 101 / 1e6
    llama = priced(book, "meta-llama/Llama-3.1-8B-Instruct:deepinfra", "openai_chat", "huggingface",
                   {"prompt_tokens": 4_300, "completion_tokens": 300}, reasoning_text_present=False)
    assert llama["usd"] == pytest.approx(0.000101, abs=1e-12) and llama["flags"] == []
    # gpt-oss-20b @deepinfra reports 1,000 cached tokens but HF publishes no cached price -> input price
    #   5,000 x 0.03 + 1,300 x 0.14 = 150 + 182 = 332 / 1e6
    oss = priced(book, "openai/gpt-oss-20b:deepinfra", "openai_chat", "huggingface",
                 {"prompt_tokens": 5_000, "completion_tokens": 1_300, "prompt_tokens_details": {"cached_tokens": 1_000},
                  "completion_tokens_details": {"reasoning_tokens": 1_000}})
    assert oss["usd"] == pytest.approx(0.000332, abs=1e-12)
    assert oss["flags"] == ["cached_price_unpublished_billed_at_input"]


def test_case6_gpt6_long_context_multiplier(book):
    # 300,000 > 272K: 300,000 x 2.00 x 2 + 1,000 x 10.00 x 1.5 = 1,200,000 + 15,000 = 1,215,000 / 1e6
    out = priced(book, "gpt-6-sol", "openai_chat", "openai",
                 {"prompt_tokens": 300_000, "completion_tokens": 1_000,
                  "completion_tokens_details": {"reasoning_tokens": 0}})
    assert out["usd"] == pytest.approx(1.215, abs=1e-12) and "long_context_multiplier" in out["flags"]


def test_incomplete_usage_is_unpriced_not_zero(book):
    out = priced(book, "gpt-6-sol", "openai_chat", "openai", {"completion_tokens": 10})
    assert out["usd"] is None and out["status"] == "unpriced" and "prompt_tokens" in out["reason"]


def test_every_panel_price_key_resolves(panel, book):
    for pp in panel.profiles.values():
        if pp.role in ("negative", "candidate"):
            assert pp.price_key is None   # never sent, never priced
        else:
            assert book.rates(pp.price_key).output > 0


def _book_with(tmp_path, **overrides):
    doc = {"units": "USD per 1M tokens, standard (synchronous) tier, pre-tax", "fetched": "test",
           "models": {"m": {"in": 2.0, "out": 10.0}}}
    doc.update(overrides)
    path = tmp_path / "prices.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_unit_assertions_reject_per_token_prices_and_wrong_units(tmp_path):
    with pytest.raises(PricingError, match="outside"):
        load_price_book(_book_with(tmp_path, models={"m": {"in": 2e-6, "out": 1e-5}}))
    with pytest.raises(PricingError, match="USD per 1M tokens"):
        load_price_book(_book_with(tmp_path, units="USD per token"))
    with pytest.raises(PricingError, match="finite number"):
        load_price_book(_book_with(tmp_path, models={"m": {"in": "2.00", "out": 10.0}}))


def test_unit_assertion_rejects_an_implausible_single_call(tmp_path):
    book = load_price_book(_book_with(tmp_path))
    usage = normalize_usage("openai_chat", "openai", {"prompt_tokens": 20_000_000, "completion_tokens": 0,
                                                      "completion_tokens_details": {"reasoning_tokens": 0}})
    with pytest.raises(PricingError, match="token units"):
        price_call("m", usage, book=book)


def test_observed_at_must_be_timezone_aware(book):
    usage = normalize_usage("openai_chat", "deepseek", {"prompt_cache_hit_tokens": 1, "prompt_cache_miss_tokens": 1,
                                                        "completion_tokens": 1})
    with pytest.raises(PricingError, match="timezone-aware"):
        price_call("deepseek-flash(V4.1-Flash)@direct", usage, book=book, observed_at=datetime(2026, 9, 29, 2))
