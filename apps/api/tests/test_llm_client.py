import json

import httpx
import pytest

from core import llm_client
from core.llm_client import LLMConfigError, get_chat_completion


def _mock_client(monkeypatch, message_data: dict, captured_payloads: list[dict] | None = None):
    """Patch httpx.AsyncClient.post to return a canned OpenAI-shaped
    chat-completion response, without hitting the network.

    If `captured_payloads` is given, every outgoing request body is recorded
    into it (in order), so a test can assert on the exact `messages` array
    sent to the provider.
    """

    async def fake_post(self, url, json=None, headers=None):
        if captured_payloads is not None:
            captured_payloads.append(json)
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            request=request,
            json={"choices": [{"message": message_data}]},
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)


async def test_normal_reply_with_no_tool_call_is_passed_through(monkeypatch):
    _mock_client(monkeypatch, {"content": "Here's my experience.", "tool_calls": None})

    result = await get_chat_completion("hi")

    assert result.content == "Here's my experience."
    assert result.tool_call is None


async def test_real_tool_call_is_parsed(monkeypatch):
    _mock_client(
        monkeypatch,
        {
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "navigate_to_section",
                        "arguments": json.dumps({"target": "experience"}),
                    }
                }
            ],
        },
    )

    result = await get_chat_completion("take me to experience")

    assert result.tool_call == {"name": "navigate_to_section", "arguments": {"target": "experience"}}


async def test_leaked_json_tool_call_is_stripped_from_visible_content(monkeypatch):
    # Exact shape reported as a bug: the model dumps the tool call as plain
    # JSON text in `content` instead of using tool_calls, using the schema's
    # "parameters" key name rather than "arguments".
    leaked = '{"name":"navigate_to_section","parameters":{"target":"projects"}}'
    _mock_client(monkeypatch, {"content": leaked, "tool_calls": None})

    result = await get_chat_completion("tell me about your projects")

    assert result.content == ""
    assert result.tool_call == {"name": "navigate_to_section", "arguments": {"target": "projects"}}


async def test_leaked_json_with_arguments_key_is_also_recovered(monkeypatch):
    leaked = '{"name":"navigate_to_section","arguments":{"target":"skills"}}'
    _mock_client(monkeypatch, {"content": leaked, "tool_calls": None})

    result = await get_chat_completion("what are your skills?")

    assert result.content == ""
    assert result.tool_call == {"name": "navigate_to_section", "arguments": {"target": "skills"}}


async def test_ordinary_prose_is_never_mistaken_for_a_tool_call(monkeypatch):
    _mock_client(monkeypatch, {"content": "Paris is the capital of France.", "tool_calls": None})

    result = await get_chat_completion("what is the capital of france?")

    assert result.content == "Paris is the capital of France."
    assert result.tool_call is None


async def test_malformed_json_looking_content_is_left_as_is(monkeypatch):
    _mock_client(monkeypatch, {"content": "{not valid json}", "tool_calls": None})

    result = await get_chat_completion("hi")

    assert result.content == "{not valid json}"
    assert result.tool_call is None


async def test_stray_brace_before_leaked_json_is_still_stripped(monkeypatch):
    # Real shape observed live: a stray leftover "}" before a full JSON
    # object for an unrelated, unrecognized pseudo-function ("print"). Must
    # never be shown to the user, even though it doesn't match our tool
    # schema and won't resolve to a valid action.
    leaked = (
        '} \n\n{"name": "print", "parameters": '
        '{"value": "Python dictionaries are unordered collections."}}'
    )
    _mock_client(monkeypatch, {"content": leaked, "tool_calls": None})

    result = await get_chat_completion("explain python dictionaries")

    assert result.content == ""
    assert result.tool_call == {
        "name": "print",
        "arguments": {"value": "Python dictionaries are unordered collections."},
    }


async def test_empty_api_key_is_rejected_before_any_network_call(monkeypatch):
    monkeypatch.setattr(llm_client, "LLM_API_KEY", "")

    async def _fail_if_called(self, url, json=None, headers=None):
        raise AssertionError("should not make a network call with no API key")

    monkeypatch.setattr(httpx.AsyncClient, "post", _fail_if_called)

    with pytest.raises(LLMConfigError):
        await get_chat_completion("hi")


async def test_hosted_provider_with_leftover_ollama_key_is_rejected(monkeypatch):
    monkeypatch.setattr(llm_client, "LLM_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setattr(llm_client, "LLM_API_KEY", "ollama")

    with pytest.raises(LLMConfigError):
        await get_chat_completion("hi")


async def test_hosted_provider_with_unfilled_placeholder_key_is_rejected(monkeypatch):
    monkeypatch.setattr(llm_client, "LLM_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setattr(llm_client, "LLM_API_KEY", "your-groq-api-key-here")

    with pytest.raises(LLMConfigError):
        await get_chat_completion("hi")


async def test_local_provider_with_ollama_default_key_is_accepted(monkeypatch):
    # "ollama" is only rejected for a hosted (non-localhost) base URL.
    monkeypatch.setattr(llm_client, "LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setattr(llm_client, "LLM_API_KEY", "ollama")
    _mock_client(monkeypatch, {"content": "hi there", "tool_calls": None})

    result = await get_chat_completion("hi")

    assert result.content == "hi there"


async def test_hosted_provider_with_real_looking_key_is_accepted(monkeypatch):
    monkeypatch.setattr(llm_client, "LLM_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setattr(llm_client, "LLM_API_KEY", "gsk_realkeylookingvalue")
    _mock_client(monkeypatch, {"content": "hi there", "tool_calls": None})

    result = await get_chat_completion("hi")

    assert result.content == "hi there"


async def test_401_from_provider_raises_clear_config_error(monkeypatch):
    monkeypatch.setattr(llm_client, "LLM_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setattr(llm_client, "LLM_API_KEY", "gsk_invalidkey")

    async def fake_post(self, url, json=None, headers=None):
        request = httpx.Request("POST", url)
        return httpx.Response(401, request=request, json={"error": {"message": "Invalid API Key"}})

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    with pytest.raises(LLMConfigError):
        await get_chat_completion("hi")


async def test_no_history_sends_only_the_user_message(monkeypatch):
    payloads: list[dict] = []
    _mock_client(monkeypatch, {"content": "hi there", "tool_calls": None}, payloads)

    await get_chat_completion("hello")

    assert payloads[0]["messages"] == [{"role": "user", "content": "hello"}]


async def test_history_is_sent_between_context_and_the_new_message_in_order(monkeypatch):
    payloads: list[dict] = []
    _mock_client(monkeypatch, {"content": "I completed it in May 2025.", "tool_calls": None}, payloads)
    history = [
        {"role": "user", "text": "What is your education?"},
        {
            "role": "assistant",
            "text": "I earned my Master's from Concordia University Wisconsin.",
        },
    ]

    await get_chat_completion(
        "In which year?", context="Portfolio information:\n## Education\n...", history=history
    )

    assert payloads[0]["messages"] == [
        {"role": "system", "content": "Portfolio information:\n## Education\n..."},
        {"role": "user", "content": "What is your education?"},
        {"role": "assistant", "content": "I earned my Master's from Concordia University Wisconsin."},
        {"role": "user", "content": "In which year?"},
    ]


async def test_history_without_context_still_precedes_the_new_message(monkeypatch):
    payloads: list[dict] = []
    _mock_client(monkeypatch, {"content": "ok", "tool_calls": None}, payloads)
    history = [{"role": "user", "text": "earlier turn"}]

    await get_chat_completion("latest message", history=history)

    assert payloads[0]["messages"] == [
        {"role": "user", "content": "earlier turn"},
        {"role": "user", "content": "latest message"},
    ]


async def test_empty_history_list_is_equivalent_to_no_history(monkeypatch):
    payloads: list[dict] = []
    _mock_client(monkeypatch, {"content": "hi there", "tool_calls": None}, payloads)

    await get_chat_completion("hello", history=[])

    assert payloads[0]["messages"] == [{"role": "user", "content": "hello"}]


async def test_real_tool_call_wins_over_separately_leaked_junk_in_content(monkeypatch):
    # A model can emit a genuine tool_calls entry AND separately dump
    # unrelated JSON-shaped junk in content. The junk must never be shown,
    # but the real tool_calls entry (not the junk) is the action source.
    _mock_client(
        monkeypatch,
        {
            "content": '{"name": "print", "parameters": {"value": "some junk"}}',
            "tool_calls": [
                {
                    "function": {
                        "name": "navigate_to_section",
                        "arguments": json.dumps({"target": "about"}),
                    }
                }
            ],
        },
    )

    result = await get_chat_completion("tell me about yourself")

    assert result.content == ""
    assert result.tool_call == {"name": "navigate_to_section", "arguments": {"target": "about"}}
