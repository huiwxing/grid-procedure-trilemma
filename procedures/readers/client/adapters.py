"""Wire protocols. No provider or model-name inference happens here."""
from __future__ import annotations

from copy import deepcopy
from urllib.parse import quote

from .config import ConfigurationError, check_generation


def _merge(left, right):
    out = deepcopy(left)
    for key, value in right.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _merge(out[key], value)
        elif key in out and out[key] != value:
            raise ConfigurationError(f"extra_body conflicts with explicit request field {key}")
        else:
            out[key] = deepcopy(value)
    return out


def _native_messages(messages):
    system, turns = [], []
    for msg in messages:
        role, content = msg["role"], msg["content"]
        if set(msg) - {"role", "content"} or not isinstance(content, str):
            raise ConfigurationError("Native text adapters accept only role/content text messages; use a suitable adapter for tool or multimodal messages")
        if role == "system":
            if turns:
                raise ConfigurationError("Native adapters require system messages before conversation turns")
            system.append(content)
        elif role in {"user", "assistant"}:
            turns.append({"role": role, "content": content})
        else:
            raise ConfigurationError(f"Native text adapter cannot represent role {role!r}")
    if not turns:
        raise ConfigurationError("At least one non-system message is required")
    return "\n\n".join(system), turns


def build_request(profile, messages, overrides=None):
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ConfigurationError("messages must be a nonempty list")
    for msg in messages:
        if not isinstance(msg, dict) or not isinstance(msg.get("role"), str) or "content" not in msg:
            raise ConfigurationError("Every message needs role and content")
    generation = dict(profile.generation)
    if overrides is not None:
        check_generation(overrides)
        generation.update(overrides)
    check_generation(generation)
    generation = {k: v for k, v in generation.items() if v is not None}
    for key in generation:
        if profile.capabilities.get(key) is False:
            raise ConfigurationError(f"Profile declares {key} unsupported; explicitly unset or change the experiment setting")
    base = profile.base_url.rstrip("/")
    if profile.adapter == "openai_chat":
        endpoint = base if base.endswith("/chat/completions") else base + "/chat/completions"
        payload = {"model": profile.request_model, "messages": deepcopy(list(messages)), "stream": False}
        mapping = {key: key for key in generation}
        mapping["max_output_tokens"] = "max_tokens"
        destination = payload
    elif profile.adapter == "anthropic_messages":
        system, turns = _native_messages(messages)
        endpoint = base if base.endswith("/messages") else base + "/messages"
        payload = {"model": profile.model, "messages": turns, "stream": False}
        if system:
            payload["system"] = system
        mapping = {"temperature": "temperature", "top_p": "top_p",
                   "max_output_tokens": "max_tokens", "stop": "stop_sequences"}
        destination = payload
    else:
        system, turns = _native_messages(messages)
        model_path = profile.model.removeprefix("models/")
        endpoint = base + "/models/" + quote(model_path, safe="") + ":generateContent"
        payload = {"contents": [{"role": "model" if x["role"] == "assistant" else "user",
                                  "parts": [{"text": x["content"]}]} for x in turns],
                   "generationConfig": {}}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        mapping = {"temperature": "temperature", "top_p": "topP",
                   "max_output_tokens": "maxOutputTokens", "seed": "seed", "stop": "stopSequences"}
        destination = payload["generationConfig"]
    mapping.update(profile.parameter_map)
    used = set()
    for key, value in generation.items():
        target = mapping.get(key)
        if target is None:
            raise ConfigurationError(f"{profile.adapter} cannot translate {key}; use documented extra_body options instead")
        if target in used or target in destination:
            raise ConfigurationError(f"Two settings map to {target}")
        used.add(target)
        if key == "stop" and profile.adapter != "openai_chat" and isinstance(value, str):
            value = [value]
        destination[target] = deepcopy(value)
    payload = _merge(payload, profile.extra_body)
    if profile.adapter == "anthropic_messages" and "max_tokens" not in payload:
        raise ConfigurationError("anthropic_messages requires an explicit max_output_tokens budget")
    if payload.get("n", 1) != 1 or payload.get("generationConfig", {}).get("candidateCount", 1) != 1:
        raise ConfigurationError("One result per request is required; use generate_batch for independent samples")
    return endpoint, payload


def _text_blocks(content):
    if isinstance(content, str):
        return content, None, False
    if content is None:
        return "", None, False
    if not isinstance(content, list):
        raise ValueError("Response content is neither text nor a block list")
    texts, thoughts, refused = [], [], False
    for block in content:
        if not isinstance(block, dict):
            raise ValueError("Invalid content block")
        if block.get("type") in {"text", "output_text"}:
            if not isinstance(block.get("text"), str):
                raise ValueError("Invalid text block")
            texts.append(block["text"])
        elif block.get("type") == "thinking":
            thoughts.append(block.get("thinking", ""))
        elif block.get("type") == "refusal":
            refused = True
            texts.append(block.get("refusal", ""))
        elif block.get("type") not in {"redacted_thinking"}:
            raise ValueError(f"Unsupported response block {block.get('type')!r}")
    return "".join(texts), "".join(thoughts) or None, refused


def parse_response(adapter, doc):
    if not isinstance(doc, dict):
        raise ValueError("Expected an object response")
    if "error" in doc:
        raise ValueError("Service returned an error envelope")
    refused = False
    if adapter == "openai_chat":
        choices = doc.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("Expected exactly one choice")
        choice = choices[0]
        message = choice["message"]
        if isinstance(message, dict) and (message.get("tool_calls") or message.get("function_call")):
            raise ValueError("Tool calls are not final text answers; this adapter requires a completed text response")
        if not isinstance(message, dict) or "content" not in message and not message.get("refusal"):
            raise ValueError("Missing message.content")
        text, reasoning, refused = _text_blocks(message.get("content"))
        candidate = message.get("reasoning_content", message.get("reasoning"))
        if isinstance(candidate, str):
            reasoning = candidate
        refused = refused or bool(message.get("refusal"))
        if not text and message.get("refusal"):
            text = str(message["refusal"])
        reason = choice.get("finish_reason")
        usage = deepcopy(doc.get("usage") or {})
        served = doc.get("model")
    elif adapter == "anthropic_messages":
        if not isinstance(doc.get("content"), list):
            raise ValueError("Missing content blocks")
        text, reasoning, refused = _text_blocks(doc["content"])
        reason, served = doc.get("stop_reason"), doc.get("model")
        usage = deepcopy(doc.get("usage") or {})
        if "input_tokens" in usage:
            usage["prompt_tokens"] = sum(usage.get(x, 0) for x in
                    ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        if "output_tokens" in usage:
            usage["completion_tokens"] = usage["output_tokens"]
    else:
        candidates = doc.get("candidates", [])
        blocked = (doc.get("promptFeedback") or {}).get("blockReason")
        if not candidates and blocked:
            text, reasoning, reason, refused = "", None, str(blocked), True
        else:
            if not isinstance(candidates, list) or len(candidates) != 1:
                raise ValueError("Expected exactly one candidate")
            candidate = candidates[0]
            reason = candidate.get("finishReason")
            texts, thoughts = [], []
            for part in candidate.get("content", {}).get("parts", []):
                if set(part) - {"text", "thought", "thoughtSignature"}:
                    raise ValueError("Unsupported non-text response part")
                if "text" in part:
                    if not isinstance(part["text"], str):
                        raise ValueError("Invalid text part")
                    (thoughts if part.get("thought") else texts).append(part["text"])
            text, reasoning = "".join(texts), "".join(thoughts) or None
        served = doc.get("modelVersion")
        usage = deepcopy(doc.get("usageMetadata") or {})
        if "promptTokenCount" in usage:
            usage["prompt_tokens"] = usage["promptTokenCount"]
        if "candidatesTokenCount" in usage or "thoughtsTokenCount" in usage:
            usage["completion_tokens"] = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
        if "thoughtsTokenCount" in usage:
            usage["reasoning_tokens"] = usage["thoughtsTokenCount"]
        if "totalTokenCount" in usage:
            usage["total_tokens"] = usage["totalTokenCount"]
    if not isinstance(usage, dict):
        raise ValueError("Invalid usage object")
    if "total_tokens" not in usage and all(isinstance(usage.get(x), int) for x in ("prompt_tokens", "completion_tokens")):
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    if reason in {"tool_calls", "function_call", "tool_use", "pause_turn", "MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL", "OTHER"}:
        raise ValueError(f"Response did not complete a final text answer: {reason}")
    error = None
    if refused or reason in {"content_filter", "refusal", "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}:
        status = "refused"
    elif reason in {"length", "max_tokens", "MAX_TOKENS", "model_context_window_exceeded",
                    "insufficient_system_resource", "aborted"}:
        status = "truncated"
    elif reason not in {"openai_chat": {"stop"}, "anthropic_messages": {"end_turn", "stop_sequence"},
                        "gemini_generate_content": {"STOP"}}[adapter]:
        # A new or absent service finish reason is not evidence of completion.
        # Keep parsed partial text and accounting instead of raising them away.
        status = "error"
        error = {"kind": "protocol", "message": f"Unrecognized finish_reason: {reason!r}"}
    else:
        status = "ok" if text.strip() else "empty"
    return dict(text=text, reasoning=reasoning, status=status, finish_reason=reason,
                served_model=served, usage=usage, error=error)
