import json
import os
from dataclasses import dataclass
from typing import Any

import httpx

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")


class LLMError(Exception):
    pass


@dataclass
class LLMResponse:
    content: str
    tool_call: dict[str, Any] | None = None


def _parse_tool_call(message: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the first tool call (if any) as {"name": str, "arguments": dict}.

    Tolerant of malformed/unexpected shapes from the LLM: returns None rather
    than raising, since a tool call is optional and this is untrusted model
    output that gets re-validated by core.actions before use.
    """
    tool_calls = message.get("tool_calls")
    if not tool_calls or not isinstance(tool_calls, list):
        return None

    function = tool_calls[0].get("function")
    if not isinstance(function, dict):
        return None

    name = function.get("name")
    raw_arguments = function.get("arguments", function.get("parameters"))

    arguments: Any = raw_arguments
    if isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return None

    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None

    return {"name": name, "arguments": arguments}


def _find_json_object(text: str) -> dict[str, Any] | None:
    """Find and parse the first balanced {...} object in text, tolerating
    surrounding junk (e.g. a stray leading brace from a truncated earlier
    attempt). Returns None if no balanced, parseable object is found.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def _extract_leaked_tool_call(content: str) -> dict[str, Any] | None:
    """Some local models (observed with Ollama + llama3.2) don't reliably use
    the tool_calls field and instead dump a tool-call-shaped JSON object in
    the message content, e.g.
    '{"name":"navigate_to_section","parameters":{"target":"projects"}}',
    sometimes with stray leading/trailing junk around it.

    Detect that shape here so the caller can treat it like any other tool
    call (and validate it the same way) — this content must never be shown
    to the user as-is, even when the function name turns out to be
    unrecognized (rejected downstream, but still not displayable JSON).
    """
    parsed = _find_json_object(content)
    if parsed is None:
        return None

    name = parsed.get("name")
    arguments = parsed.get("arguments", parsed.get("parameters"))
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None

    return {"name": name, "arguments": arguments}


async def get_chat_completion(
    message: str,
    context: str | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> LLMResponse:
    """Send a user message to an OpenAI-compatible chat completions endpoint
    and return the assistant's reply text plus any tool call it made.

    If `context` is given, it's sent as a system message ahead of the user
    message (used to ground replies in retrieved portfolio content).

    If `tools` is given, it's passed through as the OpenAI-compatible `tools`
    parameter, letting the model optionally request one of them. The caller
    is responsible for validating any returned tool call against a trusted
    allow-list before acting on it.

    Swapping LLM providers (Ollama -> Groq, etc.) only requires changing the
    LLM_BASE_URL / LLM_API_KEY / LLM_MODEL environment variables.
    """
    url = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"}
    messages = []
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": message})
    payload: dict[str, Any] = {
        "model": LLM_MODEL,
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

    data = response.json()
    try:
        message_data = data["choices"][0]["message"]
        content = message_data.get("content") or ""
    except (KeyError, IndexError) as exc:
        raise LLMError(f"Unexpected LLM response shape: {data}") from exc

    tool_call = _parse_tool_call(message_data)
    visible_content = content
    leaked = _extract_leaked_tool_call(content)
    if leaked is not None:
        # A model can emit a real tool_calls entry AND separately dump
        # JSON-shaped text in content. Content is never shown verbatim in
        # that case, whether or not a real tool call already exists — but
        # the real tool_calls entry (if any) still wins as the action
        # source, since it's the more structured, intended signal.
        if tool_call is None:
            tool_call = leaked
        visible_content = ""

    return LLMResponse(content=visible_content, tool_call=tool_call)
