# Voice AI Portfolio

Phase 1: Next.js frontend + FastAPI backend + local Ollama LLM, connected end to end. No voice, RAG, embeddings, tool calling, or production deployment yet.

## Prerequisites

- Node.js 20+
- Python 3.11+
- [Ollama](https://ollama.com) installed locally, with a model pulled (default expected: `llama3.2`)

## Backend (FastAPI)

```bash
cd apps/api
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env

# In a separate terminal, make sure Ollama is running:
ollama serve
ollama pull llama3.2   # if not already pulled

uvicorn main:app --reload --port 8000
```

Backend runs at http://localhost:8000. Check `GET /health` for a quick sanity check.

## Frontend (Next.js)

```bash
cd apps/web
npm install
cp .env.example .env.local
npm run dev
```

Frontend runs at http://localhost:3000. Type a message and click Send — it calls the FastAPI backend, which forwards it to Ollama and returns the reply.

## Configuration

- `apps/api/.env` — `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `CORS_ORIGINS`. Swapping to a production provider (e.g. Groq) later is just changing these values.
- `apps/web/.env.local` — `NEXT_PUBLIC_API_URL` (points the frontend at the backend).

Both `.env.example` files are committed; actual `.env`/`.env.local` files are gitignored.
