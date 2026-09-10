import os

import httpx

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")


class LLMError(Exception):
    pass


async def get_chat_completion(message: str, context: str | None = None) -> str:
    """Send a user message to an OpenAI-compatible chat completions endpoint
    and return the assistant's reply text.

    If `context` is given, it's sent as a system message ahead of the user
    message (used to ground replies in retrieved portfolio content).

    Swapping LLM providers (Ollama -> Groq, etc.) only requires changing the
    LLM_BASE_URL / LLM_API_KEY / LLM_MODEL environment variables.
    """
    url = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"}
    messages = []
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": message})
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise LLMError(f"Unexpected LLM response shape: {data}") from exc
