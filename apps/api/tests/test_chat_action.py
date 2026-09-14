import numpy as np
import pytest
from fastapi.testclient import TestClient

import main
from core.llm_client import LLMResponse
from core.rag import Chunk


def _chunk(id_: str, text: str = "") -> Chunk:
    return Chunk(id=id_, title=id_.title(), text=text or f"{id_} text", vector=np.zeros(1, dtype=np.float32))


@pytest.fixture
def client(monkeypatch):
    # Skip the real RAG index build (which hits the embedding endpoint over
    # the network) and default retrieval to empty so tests are hermetic
    # unless a test overrides it explicitly.
    async def _noop_build_index():
        return None

    async def _empty_retrieve_scored(_query, k=3):
        return []

    monkeypatch.setattr(main, "build_index", _noop_build_index)
    monkeypatch.setattr(main, "retrieve_scored", _empty_retrieve_scored)

    with TestClient(main.app) as test_client:
        yield test_client


def _mock_llm(monkeypatch, content: str, tool_call: dict | None = None, retry_content: str | None = None):
    """Mock get_chat_completion. The first call (with tools, for navigation
    + an initial answer attempt) returns (content, tool_call). If content is
    empty, main.py retries once without tools purely for natural language;
    that retry returns retry_content (defaulting to content) with no tool_call,
    matching the real client's behavior of only offering tools on request.
    """

    async def _fake_get_chat_completion(message, context=None, tools=None):
        if tools:
            return LLMResponse(content=content, tool_call=tool_call)
        return LLMResponse(content=retry_content if retry_content is not None else content, tool_call=None)

    monkeypatch.setattr(main, "get_chat_completion", _fake_get_chat_completion)


def _mock_retrieval(monkeypatch, scored: list[tuple[Chunk, float]]):
    async def _fake_retrieve_scored(_query, k=3):
        return scored

    monkeypatch.setattr(main, "retrieve_scored", _fake_retrieve_scored)


def test_valid_tool_call_returns_action(client, monkeypatch):
    _mock_llm(
        monkeypatch,
        content="I have experience as an AI/ML Engineer.",
        tool_call={"name": "navigate_to_section", "arguments": {"target": "experience"}},
    )

    res = client.post("/api/chat", json={"message": "take me to experience"})

    assert res.status_code == 200
    body = res.json()
    assert body["reply"] == "I have experience as an AI/ML Engineer."
    assert body["action"] == {"type": "navigate_to_section", "target": "experience"}


def test_tool_call_synonym_is_normalized_to_canonical_section(client, monkeypatch):
    _mock_llm(
        monkeypatch,
        content="Sure, here are my projects.",
        tool_call={"name": "navigate_to_section", "arguments": {"target": "projects"}},
    )

    res = client.post("/api/chat", json={"message": "tell me about your projects"})

    assert res.status_code == 200
    assert res.json()["action"] == {"type": "navigate_to_section", "target": "ai_projects"}


def test_invalid_target_drops_action_but_keeps_reply(client, monkeypatch):
    _mock_llm(
        monkeypatch,
        content="I can help with that.",
        tool_call={"name": "navigate_to_section", "arguments": {"target": "skin"}},
    )

    res = client.post("/api/chat", json={"message": "hello"})

    assert res.status_code == 200
    body = res.json()
    assert body["action"] is None
    assert body["reply"] == "I can help with that."


def test_unknown_function_name_is_ignored(client, monkeypatch):
    _mock_llm(
        monkeypatch,
        content="Sure.",
        tool_call={"name": "delete_everything", "arguments": {"target": "home"}},
    )

    res = client.post("/api/chat", json={"message": "hello"})

    assert res.status_code == 200
    assert res.json()["action"] is None


def test_no_tool_call_and_no_relevant_content_does_not_navigate(client, monkeypatch):
    _mock_llm(monkeypatch, content="Plain reply, no navigation intent.", tool_call=None)

    res = client.post("/api/chat", json={"message": "what's your experience with RAG?"})

    assert res.status_code == 200
    body = res.json()
    assert body["reply"] == "Plain reply, no navigation intent."
    assert body["action"] is None


def test_knowledge_question_navigates_via_rag_confidence_without_any_tool_call(client, monkeypatch):
    # This is the core Phase 4 revision: a plain knowledge question with no
    # "show"/"go"/"take me" wording, and no tool call from the model at all,
    # should still navigate when the question is clearly about a section.
    _mock_retrieval(monkeypatch, [(_chunk("experience", "Experience details."), 0.82)])
    _mock_llm(monkeypatch, content="I have two years of experience.", tool_call=None)

    res = client.post("/api/chat", json={"message": "what experience do you have?"})

    assert res.status_code == 200
    body = res.json()
    assert body["reply"] == "I have two years of experience."
    assert body["action"] == {"type": "navigate_to_section", "target": "experience"}


def test_low_similarity_does_not_trigger_navigation(client, monkeypatch):
    _mock_retrieval(monkeypatch, [(_chunk("skills", "Skills details."), 0.12)])
    _mock_llm(monkeypatch, content="I don't have that information in my portfolio.", tool_call=None)

    res = client.post("/api/chat", json={"message": "what is the capital of france?"})

    assert res.status_code == 200
    assert res.json()["action"] is None


def test_explicit_tool_call_takes_precedence_over_rag_match(client, monkeypatch):
    # RAG's top match is "about", but the model made an explicit, valid
    # tool call for a different (non-RAG-backed) section - that should win.
    _mock_retrieval(monkeypatch, [(_chunk("about", "About details."), 0.9)])
    _mock_llm(
        monkeypatch,
        content="Here's my resume.",
        tool_call={"name": "navigate_to_section", "arguments": {"target": "resume"}},
    )

    res = client.post("/api/chat", json={"message": "show me your resume"})

    assert res.status_code == 200
    assert res.json()["action"] == {"type": "navigate_to_section", "target": "resume"}


def test_tool_call_with_implausibly_low_rag_score_for_its_own_target_is_distrusted(
    client, monkeypatch
):
    # Regression test for a real failure found live: the model spontaneously
    # called navigate_to_section for a clearly unrelated general-knowledge
    # question. Its claimed target has a very low RAG score for the actual
    # RAG-backed section it named - that combination should be distrusted.
    _mock_retrieval(
        monkeypatch,
        [
            (_chunk("certifications", "Certs."), 0.42),
            (_chunk("contact", "Contact."), 0.41),
            (_chunk("other_projects", "Nothing here."), 0.36),
        ],
    )
    _mock_llm(
        monkeypatch,
        content="Paris is the capital of France.",
        tool_call={"name": "navigate_to_section", "arguments": {"target": "other_projects"}},
    )

    res = client.post("/api/chat", json={"message": "what is the capital of france?"})

    assert res.status_code == 200
    body = res.json()
    # Distrusted tool call falls through to the RAG-confidence fallback,
    # whose own top score (0.42) is still below NAV_SIMILARITY_THRESHOLD
    # (0.55), so no navigation happens at all - never the hallucinated one.
    assert body["action"] is None
    assert body["reply"] == "Paris is the capital of France."


def test_tool_call_with_modest_but_adequate_rag_score_is_still_trusted(client, monkeypatch):
    # An explicit command can legitimately have a modest (not high) RAG
    # score for its own target - e.g. "take me to experience" scored ~0.42
    # in live testing. The confirmation floor (0.4) must not reject that.
    _mock_retrieval(
        monkeypatch,
        [
            (_chunk("ai_projects", "Projects."), 0.43),
            (_chunk("skills", "Skills."), 0.42),
            (_chunk("experience", "Experience."), 0.42),
        ],
    )
    _mock_llm(
        monkeypatch,
        content="I have experience as an AI/ML Engineer.",
        tool_call={"name": "navigate_to_section", "arguments": {"target": "experience"}},
    )

    res = client.post("/api/chat", json={"message": "take me to experience"})

    assert res.status_code == 200
    assert res.json()["action"] == {"type": "navigate_to_section", "target": "experience"}


def test_tool_call_for_section_with_no_rag_backing_is_always_trusted(client, monkeypatch):
    # "resume"/"github"/"home"/"contact" style sections may have no chunk in
    # the retrieved results at all for a given query - there's no RAG
    # evidence to check the claim against, so it's trusted outright. This
    # is what keeps explicit commands to non-RAG-backed sections working.
    _mock_retrieval(monkeypatch, [(_chunk("about", "About."), 0.9)])
    _mock_llm(
        monkeypatch,
        content="Here's my GitHub.",
        tool_call={"name": "navigate_to_section", "arguments": {"target": "github"}},
    )

    res = client.post("/api/chat", json={"message": "take me to your github"})

    assert res.status_code == 200
    assert res.json()["action"] == {"type": "navigate_to_section", "target": "github"}


def test_leaked_json_style_tool_call_never_reaches_the_visible_reply(client, monkeypatch):
    # llm_client is responsible for stripping leaked JSON from `content`
    # before main.py ever sees it; when that leaves the first call's content
    # empty, main.py retries without tools to get a natural-language answer.
    _mock_retrieval(monkeypatch, [(_chunk("ai_projects", "My RAG chatbot project."), 0.77)])
    _mock_llm(
        monkeypatch,
        content="",
        tool_call={"name": "navigate_to_section", "arguments": {"target": "projects"}},
        retry_content="My Voice-First AI Portfolio uses RAG to ground its answers.",
    )

    res = client.post("/api/chat", json={"message": "tell me about your rag project"})

    assert res.status_code == 200
    body = res.json()
    assert body["reply"] == "My Voice-First AI Portfolio uses RAG to ground its answers."
    assert "{" not in body["reply"]
    assert body["action"] == {"type": "navigate_to_section", "target": "ai_projects"}


def test_empty_first_call_retries_without_tools_for_natural_language(client, monkeypatch):
    # Core of this revision: an empty first-call response (common when a
    # small local model is offered a tool) must never surface raw retrieved
    # Markdown - it should retry once, tools-free, for a natural answer.
    _mock_retrieval(monkeypatch, [(_chunk("experience", "# Experience\n\n- bullet one"), 0.6)])
    _mock_llm(
        monkeypatch,
        content="",
        tool_call=None,
        retry_content="I have experience as an AI/ML Engineer at Chase.",
    )

    res = client.post("/api/chat", json={"message": "tell me about your experience"})

    assert res.status_code == 200
    assert res.json()["reply"] == "I have experience as an AI/ML Engineer at Chase."
    assert "#" not in res.json()["reply"]


def test_malformed_multi_fragment_json_leak_never_shown_retries_for_prose(client, monkeypatch):
    # Real shape observed live: the model emits several broken,
    # semicolon-joined JSON fragments (not one clean parseable object, so
    # llm_client's balanced-brace scanner can't catch it). Must never be
    # shown, even partially - main.py retries without tools instead.
    leaked = (
        '}; {"name": ".navigate\\_to\\_section","parameters": {"target":"Engineering"}}; '
        '{"name": "navigate_to_section", "parameters": {"target": "some fragment"}}'
    )
    _mock_llm(
        monkeypatch,
        content=leaked,
        tool_call=None,
        retry_content="I worked on LLM applications and MLOps workflows at Chase.",
    )

    res = client.post("/api/chat", json={"message": "what did you do at chase?"})

    assert res.status_code == 200
    body = res.json()
    assert body["reply"] == "I worked on LLM applications and MLOps workflows at Chase."
    assert "{" not in body["reply"]
    assert "navigate_to_section" not in body["reply"]


def test_leaked_json_on_both_calls_falls_back_to_apology_not_raw_json(client, monkeypatch):
    leaked = '{"name": "navigate_to_section", "parameters": {"target": "skills"}}'
    _mock_llm(monkeypatch, content=leaked, tool_call=None, retry_content=leaked)

    res = client.post("/api/chat", json={"message": "hello"})

    assert res.status_code == 200
    body = res.json()
    assert body["reply"] == "I don't have that information in my portfolio."


def test_both_calls_empty_falls_back_to_grounding_apology(client, monkeypatch):
    _mock_llm(monkeypatch, content="", tool_call=None, retry_content="")

    res = client.post("/api/chat", json={"message": "hello"})

    assert res.status_code == 200
    body = res.json()
    assert body["reply"] == "I don't have that information in my portfolio."
    assert body["action"] is None


def test_markdown_is_stripped_from_the_visible_reply(client, monkeypatch):
    messy = "# About\n\n- I build **AI** systems\n- I use `Python` a lot\n\n## More\nSome *italic* text."
    _mock_llm(monkeypatch, content=messy, tool_call=None)

    res = client.post("/api/chat", json={"message": "tell me about yourself"})

    assert res.status_code == 200
    reply = res.json()["reply"]
    for forbidden in ("#", "**", "`", "- I build"):
        assert forbidden not in reply
    assert "I build AI systems" in reply
    assert "I use Python a lot" in reply
    assert "italic" in reply


def test_empty_message_is_rejected(client):
    res = client.post("/api/chat", json={"message": "   "})
    assert res.status_code == 400
