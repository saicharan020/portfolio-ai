import os

import httpx

EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")


class EmbeddingError(Exception):
    pass


async def get_embedding(text: str) -> list[float]:
    """Embed a single piece of text using Ollama's native embeddings API.

    Kept separate from LLM_BASE_URL/LLM_MODEL (used for chat) so the
    embedding provider can be swapped independently later.
    """
    url = f"{EMBEDDING_BASE_URL.rstrip('/')}/api/embeddings"
    payload = {"model": EMBEDDING_MODEL, "prompt": text}

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"Embedding request failed: {exc}") from exc

    data = response.json()
    try:
        return data["embedding"]
    except KeyError as exc:
        raise EmbeddingError(f"Unexpected embedding response shape: {data}") from exc
