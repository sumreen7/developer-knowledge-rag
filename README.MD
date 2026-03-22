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

| Component     | Technology                              |
|---------------|-----------------------------------------|
| Orchestration | LangChain                               |
| Local LLM     | Ollama (Mistral 7B / Phi-3 Mini)        |
| Vector DB     | ChromaDB                                |
| Embeddings    | SentenceTransformers (all-MiniLM-L6-v2) |
| API           | FastAPI                                 |
| UI            | Streamlit                               |
| Deployment    | Docker                                  |

---

## Getting started

```bash
# Clone and set up
git clone https://github.com/sumreen7/developer-knowledge-rag.git
cd developer-knowledge-rag
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install langchain-chroma langchain-huggingface langchain-text-splitters markdown
cp .env.example .env

# Start Ollama in a separate terminal
ollama serve
ollama pull mistral

# Run the health check
.venv/bin/uvicorn src.api.main:app --reload
curl http://localhost:8000/health
```

---

## Running the full ingestion pipeline

Add your documentation files (`.md`, `.txt`, `.pdf`) to `data/raw/`, then run:

```bash
python -m src.ingestion.run_ingestion
```

This runs the complete pipeline in one command:

```
1. Load    — reads all files from data/raw/
2. Chunk   — splits into 512-char overlapping pieces
3. Embed   — converts each chunk to a 384-number vector
4. Store   — saves vectors + text + metadata to ChromaDB
```

Example output:

```
========================================
FULL PIPELINE REPORT
========================================
Documents loaded:   3
Chunks created:     4
Chunks stored:      4
Embedding model:    all-MiniLM-L6-v2
Vector store:       ./vectorstore
========================================
Pipeline complete. Ready for Phase 5 — retrieval.
```

---

## How embeddings work

Every chunk of text is converted into 384 numbers by the
`all-MiniLM-L6-v2` model. Chunks with similar meaning produce
similar numbers. When a developer asks a question, the question
is also converted to 384 numbers — and ChromaDB finds the chunks
whose numbers are closest.

This is why the system can answer "how do I log in?" even though
the docs say "Bearer token authentication" — same meaning,
similar numbers, matched correctly.

---

## Running tests

```bash
.venv/bin/pytest tests/ -v
```

Current test count: **26 passing**

---

## Build phases

| Phase | Description                        | Status  |
|-------|------------------------------------|---------|
| 1     | Project setup, config, health API  | ✅ Done |
| 2     | Document ingestion pipeline        | ✅ Done |
| 3     | Document chunking                  | ✅ Done |
| 4     | Embeddings + ChromaDB vector store | ✅ Done |
| 5     | Retrieval pipeline                 | 🔄 Next |
| 6     | Ollama LLM integration             | Pending |
| 7     | RAG pipeline                       | Pending |
| 8     | FastAPI inference endpoint         | Pending |
| 9     | Chat UI                            | Pending |
| 10    | Evaluation                         | Pending |
| 11    | Docker deployment                  | Pending |

---

## Project structure

```
developer-knowledge-rag/
├── src/
│   ├── ingestion/
│   │   ├── loader.py          # loads .md .txt .pdf → Documents
│   │   ├── chunker.py         # splits Documents into chunks
│   │   └── run_ingestion.py   # entry point: load → chunk → embed → store
│   ├── embeddings/
│   │   └── embedder.py        # converts chunks to vectors, manages ChromaDB
│   ├── retrieval/             # Phase 5 — search ChromaDB by meaning
│   ├── llm/                   # Phase 6 — Ollama integration
│   ├── rag/                   # Phase 7 — full RAG pipeline
│   ├── api/                   # Phase 8 — FastAPI endpoints
│   └── config.py              # centralized settings from .env
├── tests/
│   ├── embeddings/
│   │   └── test_embedder.py   # 6 tests
│   └── ingestion/
│       ├── test_loader.py     # 7 tests
│       └── test_chunker.py    # 9 tests
├── data/raw/                  # put your documentation files here
├── vectorstore/               # ChromaDB files (auto-generated)
├── ui/                        # Streamlit chat interface (Phase 9)
└── pyproject.toml             # dependencies and project config
```

---

## Adding your own documentation

Drop any `.md`, `.txt`, or `.pdf` file into `data/raw/` and re-run:

```bash
python -m src.ingestion.run_ingestion
```

The system will load, chunk, embed, and index it automatically.
No code changes needed.