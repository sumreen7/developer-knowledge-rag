<!-- README.md -->
# Developer Docs AI Assistant

A production-quality RAG system that lets developers ask natural language questions
about documentation and receive grounded, source-cited answers.

## Architecture
```
Documentation files → Ingestion → Chunking → Embeddings → ChromaDB
                                                               ↓
User query → FastAPI → RAG pipeline → Ollama (Mistral) + Retrieved chunks → Answer
```

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangChain |
| Local LLM | Ollama (Mistral / Phi-3) |
| Vector DB | ChromaDB |
| Embeddings | SentenceTransformers |
| API | FastAPI |
| UI | Streamlit |
| Deployment | Docker |

## Getting Started
```bash
# Clone and set up
git clone https://github.com/YOUR_USERNAME/developer-docs-ai-assistant.git
cd developer-docs-ai-assistant
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env

# Start Ollama (separate terminal)
ollama serve
ollama pull mistral

# Run health check
uvicorn src.api.main:app --reload
curl http://localhost:8000/health
```

## Development
```bash
pytest tests/ -v          # Run tests
ruff check src/           # Lint
ruff format src/          # Format
```

## Build Phases

- [x] Phase 1 — Project setup
- [ ] Phase 2 — Document ingestion
- [ ] Phase 3 — Chunking
- [ ] Phase 4 — Embeddings + ChromaDB
- [ ] Phase 5 — Retrieval pipeline
- [ ] Phase 6 — Ollama LLM integration
- [ ] Phase 7 — RAG pipeline
- [ ] Phase 8 — FastAPI inference endpoint
- [ ] Phase 9 — Chat UI
- [ ] Phase 10 — Evaluation
- [ ] Phase 11 — Docker deployment