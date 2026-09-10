import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.llm_client import LLMError, get_chat_completion
from core.rag import Chunk, build_index, retrieve

load_dotenv()

SYSTEM_PROMPT_PREFIX = (
    "You are an assistant answering questions about a portfolio, using only "
    "the portfolio information below. If the answer isn't in this "
    "information, say you don't have that information rather than "
    "guessing.\n\n"
)


def _build_context(chunks: list[Chunk]) -> str:
    if not chunks:
        return ""
    sections = "\n\n".join(f"## {c.title}\n{c.text}" for c in chunks)
    return SYSTEM_PROMPT_PREFIX + sections


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


class ChatResponse(BaseModel):
    reply: str


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

    chunks = await retrieve(request.message)
    context = _build_context(chunks)

    try:
        reply = await get_chat_completion(request.message, context=context or None)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ChatResponse(reply=reply)


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
