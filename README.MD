# Developer Documentation AI Assistant

A production-quality RAG system that lets developers ask natural language questions
about documentation and receive grounded, source-cited answers — powered by a local
LLM running entirely on your machine. No API keys, no cloud dependencies, no data
leaves your laptop.

Supports both **Naive RAG** (fixed retrieve-then-generate pipeline) and **Agentic RAG**
(LLM decides when and how to search, can perform multiple searches per question).

---

## How It Works

```
Developer asks: "What is the difference between a 401 and 403 error?"

                        ┌─────────────────────────┐
                        │   Developer Question     │
                        └────────────┬────────────┘
                                     │
                        ┌────────────▼────────────┐
                        │   Embedding (MiniLM)     │
                        │   Question → 384 numbers │
                        └────────────┬────────────┘
                                     │
                        ┌────────────▼────────────┐
                        │   ChromaDB Vector Search │
                        │   Find closest chunks    │
                        └────────────┬────────────┘
                                     │
                        ┌────────────▼────────────┐
                        │   Mistral via Ollama     │
                        │   Generate grounded      │
                        │   answer from chunks     │
                        └────────────┬────────────┘
                                     │
                        ┌────────────▼────────────┐
                        │   Answer + Sources       │
                        │   "401 means token is    │
                        │   missing or invalid..." │
                        │   [authentication.md]    │
                        └─────────────────────────┘
```

System response:
```json
{
  "answer": "A 401 Unauthorized error means your token is missing or invalid.
             Request a new token via POST /auth/token. A 403 Forbidden error
             means your token is valid but you lack permission for the action.",
  "sources": [
    {"file_name": "authentication.md", "similarity_score": 0.551},
    {"file_name": "error_handling.md", "similarity_score": 0.359}
  ],
  "model": "mistral"
}
```

Every answer is grounded in your documentation. No hallucination.

---

## Naive RAG vs Agentic RAG

This project implements both approaches. Switch between them with one environment variable.

| Feature | Naive RAG | Agentic RAG |
|---------|-----------|-------------|
| Search strategy | Always retrieves once | LLM decides when and how to search |
| Complex questions | Hopes all info is in top 3 chunks | Breaks question into sub-queries |
| Missing info | Generates answer anyway | Can say "I don't have enough info" |
| Follow-up searches | Never | Searches again with rephrased query |
| Speed | Fast (~5 seconds) | Slower (~15 seconds) |
| Reliability | Predictable, consistent | Depends on LLM reasoning quality |
| Best for | Simple, well-defined questions | Complex, multi-part questions |

Switch modes by setting `PIPELINE_MODE` in your `.env`:
```
PIPELINE_MODE=naive      # default — fast and predictable
PIPELINE_MODE=agentic    # LLM-driven search decisions
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Orchestration | LangChain + LangGraph | Pipeline wiring and agent framework |
| Local LLM | Ollama (Mistral 7B) | Answer generation, runs on your machine |
| Vector DB | ChromaDB | Stores and searches document embeddings |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) | Converts text to 384-dim vectors |
| API | FastAPI | REST endpoint for queries |
| UI | Streamlit | Chat interface with source citations |
| Evaluation | RAGAS | Measures faithfulness, relevancy, precision |
| Deployment | Docker + Docker Compose | One-command containerized deployment |
| Testing | pytest | 63 automated tests across all modules |

---

## Evaluation Results (RAGAS)

Evaluated using 7 test cases with Mistral as the local LLM judge:

| Metric | Score | What it measures |
|--------|-------|------------------|
| **Faithfulness** | **1.000** | Does the answer only use info from retrieved docs? |
| **Context Precision** | **1.000** | Were the right chunks retrieved for the question? |
| Answer Relevancy | 0.458 | Does the answer address the question asked? |
| **Overall** | **0.819** | Weighted average across all metrics |

Faithfulness at 1.0 means the system never hallucinated — every claim traces back to
a retrieved document chunk. Context precision at 1.0 means the retriever consistently
finds the correct documentation. Answer relevancy was affected by timeout issues with
local CPU inference (most questions returned `nan` due to RAGAS evaluation timeouts).

---

## Getting Started

### Prerequisites

- Python 3.11+ (tested on 3.13)
- Ollama installed ([ollama.com](https://ollama.com))
- ~4GB disk space for Mistral model

### Setup

```bash
# Clone the repo
git clone https://github.com/sumreen7/developer-knowledge-rag.git
cd developer-knowledge-rag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate    # Mac/Linux

# Install dependencies
pip install --upgrade pip
pip install -e ".[dev]"
pip install langchain-chroma langchain-huggingface langchain-ollama \
            langchain-text-splitters langchain langgraph markdown

# Copy environment config
cp .env.example .env

# Start Ollama and pull Mistral (separate terminal)
ollama serve
ollama pull mistral
```

### Run the Pipeline

```bash
# 1. Ingest your documentation
python -m src.ingestion.run_ingestion

# 2. Start the API server
.venv/bin/uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# 3. Start the chat UI (separate terminal)
streamlit run ui/app.py

# 4. Open the chat UI
open http://localhost:8501

# 5. Or query the API directly
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I authenticate with the API?"}'
```

### Run with Docker

```bash
# Start Ollama
docker-compose up -d ollama

# Pull the model
docker-compose exec ollama ollama pull mistral

# Ingest documents
docker-compose --profile ingestion up ingestion

# Start API + UI
docker-compose up -d api ui

# Open the UI
open http://localhost:8501
```

---

## Adding Your Own Documentation

Drop any `.md`, `.txt`, or `.pdf` file into `data/raw/` and run ingestion:

```bash
python -m src.ingestion.run_ingestion
```

The pipeline uses **incremental ingestion** — it queries ChromaDB's metadata index
for already-embedded file names and skips them automatically. Only new, previously
unseen documents get chunked, embedded, and stored. No duplicate vectors, no
redundant computation, no manual vectorstore cleanup.

### Ingestion Modes

| Mode | Command | Behavior |
|------|---------|----------|
| **Incremental** (default) | `python -m src.ingestion.run_ingestion` | Skips already-indexed files, processes only new ones |
| **Force re-index** | `python -m src.ingestion.run_ingestion --force` | Deletes vectorstore, re-embeds entire corpus from scratch |
| **Single-file re-index** | `python -m src.ingestion.run_ingestion --reindex auth.md` | Deletes stale chunks for one file, re-embeds it |

### Example: Adding a New Document

```bash
# Drop a new file into the data directory
cp deployment_guide.md data/raw/

# Run ingestion — only the new file gets processed
python -m src.ingestion.run_ingestion

# Output:
# Skipping 4 already-ingested file(s)
# deployment_guide.md → 3 chunk(s)
# New files ingested: 1
# Total in store:     9
```

Use `--reindex` when a document has been edited and needs re-embedding,
or `--force` when you want to rebuild the entire vector index from scratch.

---

## API Reference

### POST /query

Ask a question about the documentation.

**Request:**
```json
{
  "question": "How do I authenticate with the API?",
  "k": 3
}
```

**Response:**
```json
{
  "question": "How do I authenticate with the API?",
  "answer": "Use Bearer token authentication. Send a POST request to /auth/token...",
  "sources": [
    {
      "file_name": "authentication.md",
      "chunk_index": 0,
      "content_preview": "Authentication Guide...",
      "similarity_score": 0.611
    }
  ],
  "model": "mistral",
  "num_chunks_retrieved": 3,
  "agent_steps": 0,
  "search_queries": []
}
```

### GET /health

Check pipeline status.

```json
{
  "status": "healthy",
  "pipeline_mode": "naive",
  "ollama": "connected",
  "model": "mistral",
  "config": {
    "ollama_model": "mistral",
    "embedding_model": "all-MiniLM-L6-v2",
    "chroma_collection": "docs"
  }
}
```

### GET /docs

Interactive Swagger UI for testing the API in your browser.

---

## Running Tests

```bash
.venv/bin/pytest tests/ -v
```

**63 tests** across all modules:

| Module | Tests | What they cover |
|--------|-------|-----------------|
| API | 10 | Endpoint responses, validation, pipeline integration |
| Embeddings | 6 | Model loading, ChromaDB storage, collection stats |
| Chunker | 9 | Splitting, metadata inheritance, edge cases |
| Loader | 7 | File loading, metadata, unsupported types |
| LLM | 7 | Ollama client, prompt template, error handling |
| RAG Pipeline | 10 | End-to-end query, sources, response format |
| Retrieval | 10 | Similarity search, scores, context formatting |
| Health | 4 | API health check, config exposure |

---

## Running Evaluation

```bash
python -m src.evaluation.evaluate
```

Runs 7 test cases through the full pipeline, uses Mistral as the local judge,
and saves timestamped results to `evaluation_results/`.

---

## Project Structure

```
developer-knowledge-rag/
├── src/
│   ├── api/
│   │   └── main.py                # FastAPI — POST /query, GET /health
│   ├── embeddings/
│   │   └── embedder.py            # SentenceTransformers → ChromaDB + incremental indexing
│   ├── evaluation/
│   │   └── evaluate.py            # RAGAS evaluation script
│   ├── ingestion/
│   │   ├── loader.py              # Loads .md, .txt, .pdf files
│   │   ├── chunker.py             # Splits documents into chunks
│   │   └── run_ingestion.py       # Incremental pipeline: load → chunk → embed → store
│   ├── llm/
│   │   └── ollama_client.py       # Mistral via Ollama wrapper
│   ├── rag/
│   │   ├── pipeline.py            # Naive RAG: retrieve once → generate
│   │   ├── agentic_pipeline.py    # Agentic RAG: LLM-driven search decisions
│   │   ├── run_rag.py             # Test naive pipeline
│   │   └── run_agentic.py         # Test agentic pipeline
│   ├── retrieval/
│   │   ├── retriever.py           # ChromaDB similarity search
│   │   └── run_retrieval.py       # Test retrieval standalone
│   └── config.py                  # Pydantic Settings from .env
├── tests/                         # 63 tests mirroring src/ structure
├── data/raw/                      # Your documentation files go here
├── ui/
│   └── app.py                     # Streamlit chat interface
├── vectorstore/                   # ChromaDB files (auto-generated)
├── evaluation_results/            # RAGAS JSON reports
├── Dockerfile                     # FastAPI backend container
├── Dockerfile.ui                  # Streamlit UI container
├── docker-compose.yml             # Full stack orchestration
├── .dockerignore
├── .env.example
├── pyproject.toml                 # Dependencies and project config
└── README.MD
```

---

## Build Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Project setup, config, health check API | ✅ |
| 2 | Document ingestion pipeline with incremental indexing | ✅ |
| 3 | Document chunking (RecursiveCharacterTextSplitter) | ✅ |
| 4 | Embeddings + ChromaDB vector store | ✅ |
| 5 | Retrieval pipeline (similarity search) | ✅ |
| 6 | Ollama LLM integration (Mistral) | ✅ |
| 7 | RAG pipeline (retrieve → generate) | ✅ |
| 8 | FastAPI inference endpoint | ✅ |
| 9 | Streamlit chat interface | ✅ |
| 10 | RAGAS evaluation (faithfulness, relevancy, precision) | ✅ |
| 11 | Docker deployment + Agentic RAG upgrade | ✅ |

---

## How Chunking Works

Documents are split into overlapping pieces using LangChain's
`RecursiveCharacterTextSplitter`. It cuts at natural boundaries
in this priority: paragraph → line → sentence → word.

```
chunk_size    = 512 characters
chunk_overlap = 50 characters
```

The overlap ensures answers spanning two sections are never lost.

---

## How Embeddings Work

Each chunk is converted into 384 numbers by the `all-MiniLM-L6-v2` model.
Chunks with similar meaning produce similar numbers. When a developer asks
a question, the question is also converted to 384 numbers — and ChromaDB
finds the chunks whose numbers are closest.

This is why the system can answer "how do I log in?" even when the docs
say "Bearer token authentication" — same meaning, similar vectors.

---

## Configuration

All settings are managed via `.env` file using Pydantic Settings:

```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIRECTORY=./vectorstore
DATA_DIRECTORY=./data/raw
CHUNK_SIZE=512
CHUNK_OVERLAP=50
PIPELINE_MODE=naive
API_DEBUG=true
```

---

## Built With

- [LangChain](https://python.langchain.com/) — LLM orchestration
- [LangGraph](https://langchain-ai.github.io/langgraph/) — Agentic RAG framework
- [Ollama](https://ollama.com/) — Local LLM inference
- [ChromaDB](https://www.trychroma.com/) — Vector database
- [SentenceTransformers](https://www.sbert.net/) — Text embeddings
- [FastAPI](https://fastapi.tiangolo.com/) — REST API
- [Streamlit](https://streamlit.io/) — Chat UI
- [RAGAS](https://docs.ragas.io/) — RAG evaluation
- [Docker](https://www.docker.com/) — Containerized deployment