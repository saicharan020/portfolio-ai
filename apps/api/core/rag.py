import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.embeddings import EmbeddingError, get_embedding

logger = logging.getLogger(__name__)

CONTENT_DIR = Path(__file__).resolve().parent.parent / "content"


@dataclass
class Chunk:
    id: str
    title: str
    text: str
    vector: np.ndarray


_index: list[Chunk] = []


def _title_from_filename(filename: str) -> str:
    return filename.removesuffix(".md").replace("_", " ").title()


async def build_index() -> None:
    """Load content/*.md, embed each file as a single chunk, and hold the
    resulting vectors in memory. Called once at FastAPI startup.
    """
    global _index
    loaded: list[Chunk] = []

    if not CONTENT_DIR.exists():
        logger.warning("Content directory %s does not exist; RAG index is empty.", CONTENT_DIR)
        _index = loaded
        return

    for path in sorted(CONTENT_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        try:
            vector = await get_embedding(text)
        except EmbeddingError as exc:
            logger.warning("Skipping %s: %s", path.name, exc)
            continue
        loaded.append(
            Chunk(
                id=path.stem,
                title=_title_from_filename(path.name),
                text=text,
                vector=np.array(vector, dtype=np.float32),
            )
        )

    _index = loaded
    logger.info("RAG index built with %d chunk(s).", len(_index))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


async def retrieve(query: str, k: int = 3) -> list[Chunk]:
    """Return the top-k most relevant chunks for a query.

    Returns an empty list if the index hasn't been built or the query
    embedding call fails, so callers can fall back to an ungrounded reply
    instead of erroring out.
    """
    if not _index:
        return []

    try:
        query_vector = np.array(await get_embedding(query), dtype=np.float32)
    except EmbeddingError:
        return []

    scored = [(chunk, _cosine_similarity(query_vector, chunk.vector)) for chunk in _index]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [chunk for chunk, _ in scored[:k]]
