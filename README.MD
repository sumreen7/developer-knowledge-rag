# Developer Knowledge RAG

A production-quality RAG (Retrieval-Augmented Generation) system that lets
developers ask natural language questions about documentation and receive
grounded, source-cited answers — powered by a local LLM running on your machine.

---

## How it works
```
Your docs (Markdown, PDF, text)
        ↓
  Ingestion pipeline       ← loads and labels every document
        ↓
  Chunker                  ← splits docs into searchable pieces
        ↓
  Embeddings + ChromaDB    ← converts text to vectors, stores them
        ↓
  Developer asks a question
        ↓
  Retrieval pipeline       ← finds the most relevant chunks
        ↓
  Ollama (Mistral/Phi-3)   ← generates an answer using those chunks
        ↓
  Answer + source citation ← "Use Bearer tokens... [source: authentication.md]"
```

---

## Tech stack

| Component      | Technology                        |
|----------------|-----------------------------------|
| Orchestration  | LangChain                         |
| Local LLM      | Ollama (Mistral 7B / Phi-3 Mini)  |
| Vector DB      | ChromaDB                          |
| Embeddings     | SentenceTransformers              |
| API            | FastAPI                           |
| UI             | Streamlit                         |
| Deployment     | Docker                            |

---

## Getting started
```bash
# Clone and set up
git clone https://github.com/sumreen7/developer-knowledge-rag.git
cd developer-knowledge-rag
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env

# Start Ollama in a separate terminal
ollama serve
ollama pull mistral

# Run the health check
.venv/bin/uvicorn src.api.main:app --reload
curl http://localhost:8000/health
```

---

## Running the ingestion pipeline

Add your documentation files (`.md`, `.txt`, `.pdf`) to `data/raw/`, then run:
```bash
python -m src.ingestion.run_ingestion
```

Example output:
```
INGESTION REPORT — 3 total documents
  api_endpoints.md: 1 document(s)
  authentication.md: 1 document(s)
  getting_started.txt: 1 document(s)
```

---

## Running tests
```bash
.venv/bin/pytest tests/ -v
```

---

## Build phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Project setup, config, health check API | ✅ Done |
| 2 | Document ingestion pipeline | ✅ Done |
| 3 | Document chunking | 🔄 Next |
| 4 | Embeddings + ChromaDB vector store | Pending |
| 5 | Retrieval pipeline | Pending |
| 6 | Ollama LLM integration | Pending |
| 7 | RAG pipeline | Pending |
| 8 | FastAPI inference endpoint | Pending |
| 9 | Chat UI | Pending |
| 10 | Evaluation | Pending |
| 11 | Docker deployment | Pending |

---

## Project structure
```
developer-knowledge-rag/
├── src/
│   ├── ingestion/       # document loading pipeline
│   ├── embeddings/      # embedding generation
│   ├── retrieval/       # vector search
│   ├── llm/             # Ollama integration
│   ├── rag/             # full RAG pipeline
│   ├── api/             # FastAPI endpoints
│   └── config.py        # centralized settings
├── tests/               # mirrors src/ structure
├── data/raw/            # your documentation files go here
├── vectorstore/         # ChromaDB files (auto-generated)
├── ui/                  # Streamlit chat interface
└── pyproject.toml       # dependencies and project config
```

---

## Adding your own documentation

Drop any `.md`, `.txt`, or `.pdf` file into `data/raw/` and re-run the
ingestion pipeline. The system will automatically load, chunk, embed, and
index it — no code changes needed.