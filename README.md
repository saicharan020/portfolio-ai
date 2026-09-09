# Voice AI Portfolio

My personal portfolio with a voice-first AI assistant. Visitors will be able to talk to it naturally and ask about my experience, projects, skills, education, and achievements. The assistant will answer using information from my portfolio and will eventually navigate the website through voice commands.

## Current Status

Phase 1 is complete. The basic system works end to end:

Next.js → FastAPI → Ollama → FastAPI → Next.js

For now, interaction is through text. Voice, RAG, navigation tools, and production deployment will be added in later phases.

## Requirements

- Node.js 20+
- Python 3.11+
- Ollama
- Git
- Ollama llama3.2 model

## Backend

```bash
cd apps/api
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Create `.env` from `.env.example`.

Start Ollama:

```bash
ollama serve
```

If needed:

```bash
ollama pull llama3.2
```

Start FastAPI:

```bash
uvicorn main:app --reload --port 8000
```

API: [http://localhost:8000](http://localhost:8000)

Health check: [http://localhost:8000/health](http://localhost:8000/health)

## Frontend

Open another terminal:

```bash
cd apps/web
npm install
npm run dev
```

Create `.env.local` from `.env.example`.

Portfolio: [http://localhost:3000](http://localhost:3000)

## Architecture

```text
User
 ↓
Next.js Frontend
 ↓
FastAPI Backend
 ↓
Ollama
 ↓
Local LLM
```

The frontend communicates with FastAPI instead of calling the LLM directly. The backend uses an OpenAI-compatible interface so the LLM provider can be changed later without redesigning the application.

## Future Plans

- Voice interaction using speech-to-text and text-to-speech
- RAG using verified portfolio content
- AI tool calling for website navigation
- Hosted LLM for production
- Frontend deployment on Vercel
- Backend deployment on a free-tier hosting service
