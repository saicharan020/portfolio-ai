import numpy as np
import pytest

from core import rag


@pytest.fixture(autouse=True)
def restore_index():
    original = rag._index
    yield
    rag._index = original


def _chunk(id_: str, vector: list[float]) -> rag.Chunk:
    return rag.Chunk(id=id_, title=id_.title(), text=f"{id_} text", vector=np.array(vector, dtype=np.float32))


async def test_retrieve_scored_orders_by_similarity_descending(monkeypatch):
    rag._index = [
        _chunk("about", [1.0, 0.0]),
        _chunk("experience", [0.0, 1.0]),
    ]

    async def fake_embedding(_text):
        return [0.0, 1.0]

    monkeypatch.setattr(rag, "get_embedding", fake_embedding)

    results = await rag.retrieve_scored("tell me about your experience", k=2)

    assert [chunk.id for chunk, _score in results] == ["experience", "about"]
    assert results[0][1] > results[1][1]


async def test_retrieve_returns_chunks_without_scores(monkeypatch):
    rag._index = [_chunk("skills", [1.0, 0.0])]

    async def fake_embedding(_text):
        return [1.0, 0.0]

    monkeypatch.setattr(rag, "get_embedding", fake_embedding)

    results = await rag.retrieve("what are your skills?")

    assert results == [rag._index[0]]


async def test_empty_index_returns_empty_results():
    rag._index = []

    assert await rag.retrieve_scored("anything") == []
    assert await rag.retrieve("anything") == []
