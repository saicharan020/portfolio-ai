import os
import re
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv

# Must run before importing core.* modules below: several of them read
# provider config (LLM_BASE_URL, LLM_API_KEY, EMBEDDING_BASE_URL, ...) into
# module-level constants at import time, so .env has to be loaded first or
# those values silently fall back to hardcoded defaults.
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.actions import NAVIGATE_TOOL_NAME, TOOL_SCHEMA, NavigateAction, validate_action
from core.llm_client import LLMError, get_chat_completion
from core.rag import Chunk, build_index, retrieve, retrieve_scored

# Minimum cosine similarity between the user's message and a portfolio
# content chunk before we treat the question as "clearly about" that
# section and navigate to it. Tunable via env var without a code change —
# the right value depends on the embedding model and content in use.
#
# Calibrated against nomic-embed-text over the real Phase 5 content: with
# short, single-chunk-per-file documents, on-topic and off-topic scores
# overlap (e.g. "Explain Python dictionaries" scored 0.537, higher than
# several genuinely on-topic queries). 0.55 was chosen to keep clearly
# off-topic/general-knowledge questions (which must never navigate) below
# the bar, at the cost of missing navigation for a few softer knowledge
# questions whose top score falls just under it. Explicit commands
# ("take me to experience") aren't affected by this threshold — they're
# handled by the LLM's own tool call, validated the same way, independent
# of this fallback path.
NAV_SIMILARITY_THRESHOLD = float(os.getenv("NAV_SIMILARITY_THRESHOLD", "0.55"))

# Minimum cosine similarity a tool call's OWN claimed target must reach
# against the RAG index before it's trusted, when that target is a
# RAG-backed section. This is a lower bar than NAV_SIMILARITY_THRESHOLD -
# it exists only to catch clearly implausible tool calls (e.g. the model
# spontaneously calling navigate_to_section for an unrelated general
# knowledge question), not to gate legitimate explicit commands, which can
# have modest scores. Sections with no RAG content (home, github) have
# nothing to check against, so their tool calls are always trusted -
# explicit commands for those keep working regardless of this floor.
#
# This can't fully prevent a small local model from hallucinating a
# plausible-looking-but-wrong tool call (e.g. it may still pick a section
# whose content happens to share vocabulary with an unrelated question -
# "Python" appears in both a general-knowledge question and this
# portfolio's Skills/Experience content), but it does catch the more
# obviously spurious cases.
TOOL_CALL_CONFIRMATION_FLOOR = float(os.getenv("TOOL_CALL_CONFIRMATION_FLOOR", "0.4"))

# Large enough to cover the whole content index (currently 9 files), so the
# tool-call confirmation check above can look up any claimed target's own
# score. retrieve_scored already computes similarity against every chunk
# internally - this only changes how much of that already-computed, sorted
# list is returned, not what's retrieved or how. Context building below
# still only uses the top 3, unchanged from before.
_RAG_FULL_INDEX_K = 20

SYSTEM_PROMPT_PREFIX = (
    "You are Charan, answering questions about your own background as "
    "yourself, in first person, the way you would in an interview. Use "
    "only the portfolio information below as your factual source. "
    "Paraphrase it into natural, spoken sentences - do not quote it, "
    "list it, or use Markdown formatting. Usually 2-3 sentences is "
    "enough. If the information below does not cover the question, say "
    "you do not have that information in your portfolio rather than "
    "guessing. Questions unrelated to your background should be "
    "answered normally, as general knowledge.\n\n"
    "Portfolio information:\n"
)

_NO_PORTFOLIO_INFO_REPLY = "I don't have that information in my portfolio."

# Conversation history is untrusted client input (Phase 6: conversational
# context) - bounded to the same "last 6-10 messages" the frontend is
# expected to send, so a misbehaving/stale client can't force an
# ever-growing prompt. Applied after sanitization, oldest-first, so the LLM
# always sees a fixed-size, chronologically-ordered window.
_MAX_HISTORY_TURNS = 8

# How many of the user's own recent turns (never the assistant's, see
# _build_retrieval_query) get folded into the RAG search query so a short,
# context-dependent follow-up can still retrieve the right section. Smaller
# than _MAX_HISTORY_TURNS on purpose: the LLM benefits from the fuller
# conversational memory, but the retrieval query should stay focused on
# what was JUST being discussed, not topics from several turns ago.
_RETRIEVAL_HISTORY_USER_TURNS = 2


def _sanitize_history(raw: Any) -> list[dict[str, str]]:
    """Defensively validate client-supplied conversation history.

    This is untrusted input from the browser, so nothing here is trusted by
    shape: anything that isn't a well-formed {"role": "user"|"assistant",
    "text": <non-empty str>} turn is silently dropped rather than rejected -
    one malformed entry (e.g. from a stale frontend build) must not break
    the whole request, matching how core.llm_client already treats
    malformed model output as "tolerant, drop and continue" rather than a
    hard error. Returns at most the most recent _MAX_HISTORY_TURNS turns,
    oldest first.
    """
    if not isinstance(raw, list):
        return []

    sanitized: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        text = item.get("text")
        if role not in ("user", "assistant"):
            continue
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
        sanitized.append({"role": role, "text": text})

    return sanitized[-_MAX_HISTORY_TURNS:]


def _build_retrieval_query(message: str, history: list[dict[str, str]]) -> str:
    """Fold the user's own recent wording into the RAG search query so a
    short, context-dependent follow-up ("In which year?", "What about
    Chase?", "How long did you work there?") can still retrieve the right
    portfolio section, instead of searching on a near-empty/pronoun-only
    string that matches nothing.

    Deliberately uses only USER turns, never the assistant's prior reply:
    an assistant reply already contains near-verbatim RAG chunk text, and
    mixing that in would bias retrieval toward whatever topic was just
    discussed even for a genuinely new, unrelated question (e.g. general
    knowledge asked right after a portfolio question) - the same failure
    mode NAV_SIMILARITY_THRESHOLD's calibration note guards against, just
    from the history side instead of the query side.

    With no history (or no prior user turns), this returns `message`
    unchanged, so a request with no history behaves exactly as it did
    before conversation history existed.
    """
    recent_user_turns = [turn["text"] for turn in history if turn["role"] == "user"]
    recent_user_turns = recent_user_turns[-_RETRIEVAL_HISTORY_USER_TURNS:]
    if not recent_user_turns:
        return message
    return "\n".join([*recent_user_turns, message])

# Substrings that only show up when a model has leaked tool-call-shaped
# text into its answer instead of (or alongside) real prose - including
# malformed/multi-fragment JSON soup that _find_json_object can't cleanly
# parse. Content matching this is never shown to the user, even partially.
_LEAKED_CALL_MARKERS = ('"name"', '"parameters"', '"arguments"', "navigate_to_section", "navigate\\_to\\_section")

_HEADING_RE = re.compile(r"(?m)^#{1,6}\s*")
_BULLET_RE = re.compile(r"(?m)^\s*[-*+]\s+")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_ITALIC_RE = re.compile(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])")
_CODE_RE = re.compile(r"`([^`]*)`")


def _build_context(chunks: list[Chunk]) -> str:
    if not chunks:
        return ""
    sections = "\n\n".join(f"## {c.title}\n{c.text}" for c in chunks)
    return SYSTEM_PROMPT_PREFIX + sections


def _strip_markdown(text: str) -> str:
    """Strip Markdown syntax (headings, bullet markers, emphasis, code
    spans) from a reply so raw document formatting never reaches the user.

    This is a deterministic safety net on top of the system prompt's
    plain-prose instruction: the model is told not to use Markdown, but a
    small local model can still slip up, so this guarantees it regardless.
    """
    if not text:
        return text

    text = _HEADING_RE.sub("", text)
    text = _BULLET_RE.sub("", text)
    text = _BOLD_RE.sub(lambda m: m.group(1) or m.group(2), text)
    text = _ITALIC_RE.sub(r"\1", text)
    text = _CODE_RE.sub(r"\1", text)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return re.sub(r"\s{2,}", " ", " ".join(lines)).strip()


def _looks_like_leaked_tool_call(text: str) -> bool:
    """Detect tool-call-shaped text that survived cleanup - including
    malformed/multi-fragment JSON soup that isn't one clean parseable
    object, so _find_json_object in llm_client can't catch it. Natural
    prose about a portfolio essentially never contains a brace alongside
    one of these exact quoted JSON key names, so this is a safe check.
    """
    if "{" not in text and "}" not in text:
        return False
    return any(marker in text for marker in _LEAKED_CALL_MARKERS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await build_index()
    yield


app = FastAPI(title="Portfolio AI API", lifespan=lifespan)

cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    # Intentionally untyped/unvalidated by pydantic (Any, not
    # list[ChatTurn]): malformed history must never 422 the whole request -
    # see _sanitize_history, which does the real, defensive validation.
    history: Any = None


class ChatResponse(BaseModel):
    reply: str
    action: NavigateAction | None = None


class RetrievedChunk(BaseModel):
    id: str
    title: str
    text: str


class RetrieveResponse(BaseModel):
    query: str
    results: list[RetrievedChunk]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    history = _sanitize_history(request.history)

    retrieval_query = _build_retrieval_query(request.message, history)
    scored_chunks = await retrieve_scored(retrieval_query, k=_RAG_FULL_INDEX_K)
    chunks = [chunk for chunk, _score in scored_chunks[:3]]
    context = _build_context(chunks)

    try:
        result = await get_chat_completion(
            request.message, context=context or None, tools=[TOOL_SCHEMA], history=history
        )
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # An explicit, well-formed tool call (real or recovered from a leaked
    # JSON blob) takes precedence — but only if RAG doesn't clearly
    # contradict it (see TOOL_CALL_CONFIRMATION_FLOOR above). Otherwise,
    # fall back to whichever portfolio section the question is most
    # semantically about — this is what makes an ordinary knowledge
    # question ("tell me about your experience") navigate too, not just an
    # explicit command ("show me...").
    action = validate_action(result.tool_call)
    if action is not None:
        target_score = next(
            (score for chunk, score in scored_chunks if chunk.id == action.target), None
        )
        if target_score is not None and target_score < TOOL_CALL_CONFIRMATION_FLOOR:
            action = None

    if action is None and scored_chunks:
        top_chunk, top_score = scored_chunks[0]
        if top_score >= NAV_SIMILARITY_THRESHOLD:
            action = validate_action(
                {"name": NAVIGATE_TOOL_NAME, "arguments": {"target": top_chunk.id}}
            )

    reply_text = result.content.strip()
    if not reply_text or _looks_like_leaked_tool_call(reply_text):
        # The first call (offered the navigation tool) sometimes produces no
        # usable prose at all, or leaks tool-call-shaped/malformed JSON text
        # instead of an answer - small local models do this more than
        # larger ones. Retry once without tools, purely to generate natural
        # language; the navigation decision above already happened and is
        # untouched by this retry (it never runs again).
        try:
            retry = await get_chat_completion(
                request.message, context=context or None, history=history
            )
            candidate = retry.content.strip()
            reply_text = candidate if candidate and not _looks_like_leaked_tool_call(candidate) else ""
        except LLMError:
            reply_text = ""

    if not reply_text:
        # Never fall back to showing the raw retrieved Markdown - if there's
        # still no natural-language answer, say so plainly instead.
        reply_text = _NO_PORTFOLIO_INFO_REPLY

    return ChatResponse(reply=_strip_markdown(reply_text), action=action)


@app.get("/api/retrieve", response_model=RetrieveResponse)
async def debug_retrieve(q: str) -> RetrieveResponse:
    """Debug endpoint: returns raw retrieved chunks with no LLM call, for
    testing retrieval quality in isolation.
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="q must not be empty")

    chunks = await retrieve(q)
    return RetrieveResponse(
        query=q,
        results=[RetrievedChunk(id=c.id, title=c.title, text=c.text) for c in chunks],
    )
