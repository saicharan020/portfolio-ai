# Voice AI Portfolio

Build a personal portfolio for Charan focused on AI Engineering roles.

## Core Idea

The portfolio is voice-first.

A recruiter should be able to speak naturally to an AI assistant and ask about:

- Experience
- AI projects
- Skills
- Education
- Certifications
- Achievements
- Resume
- GitHub

The assistant should answer using only verified portfolio information.

It should also control portfolio navigation through voice.

Examples:
- "Show me your projects."
- "Tell me about your RAG project."
- "Show me the architecture."
- "Take me to experience."
- "Show me the resume."

There should be no traditional chatbot UI. Voice is the primary AI interaction.

Normal website navigation must still be available.

## AI

Development: Ollama + local LLM

Production: Free-tier hosted LLM

The LLM provider must be replaceable without rewriting the application.

Use controlled tools/functions for website actions rather than unrestricted browser control.

## Voice

Voice → Speech-to-Text → AI → Text-to-Speech → Voice

Prioritize low latency and natural interaction.

## Portfolio

Include:

Home, About, Experience, AI Projects, Other Projects, Skills, Education, Certifications, Resume, GitHub, Contact.

## Technology

Preferred:

- React/Next.js
- TypeScript
- Python/FastAPI when needed
- RAG
- Tool/function calling
- GitHub
- GitHub Actions
- Docker when useful

Choose the simplest appropriate architecture.

## Hosting

Target $0/month using free-tier services.

Prefer Vercel for the frontend.

Never expose API keys in frontend code.

## Development Rule

Build incrementally.

Before coding, analyze this specification and propose:

1. Architecture
2. Technology stack
3. Repository structure
4. Voice architecture
5. AI/tool architecture
6. RAG approach
7. Free-tier deployment plan

Do not implement until I approve the architecture.