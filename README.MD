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

## Running the ingestion + chunking pipeline

Add your documentation files (`.md`, `.txt`, `.pdf`) to `data/raw/`, then run:
```bash
python -m src.ingestion.run_ingestion
```

Example output:
```
========================================
INGESTION + CHUNKING REPORT
========================================
Documents loaded:  3
Total chunks:      4
Avg chunk length:  317 chars
Min chunk length:  99 chars
Max chunk length:  459 chars
========================================
Chunk 1 — api_endpoints.md (0 of 1)
Chunk 2 — authentication.md (0 of 2)
Chunk 3 — authentication.md (1 of 2)
Chunk 4 — getting_started.txt (0 of 1)
```

Each chunk inherits the source document's metadata so every answer
can be traced back to the exact file it came from.

---

## Running tests
```bash
.venv/bin/pytest tests/ -v
```

Current test count: **20 passing**

---

## Build phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Project setup, config, health check API | ✅ Done |
| 2 | Document ingestion pipeline | ✅ Done |
| 3 | Document chunking | ✅ Done |
| 4 | Embeddings + ChromaDB vector store | 🔄 Next |
| 5 | Retrieval pipeline | Pending |
| 6 | Ollama LLM integration | Pending |
| 7 | RAG pipeline | Pending |
| 8 | FastAPI inference endpoint | Pending |
| 9 | Chat UI | Pending |
| 10 | Evaluation | Pending |
| 11 | Docker deployment | Pending |

---

## How chunking works

Each document is split into overlapping pieces using LangChain's
`RecursiveCharacterTextSplitter`. It cuts at natural boundaries in
this order: paragraph → line → sentence → word. The overlap ensures
answers that span two sections are never lost.
```
chunk_size    = 512 characters   (set in .env)
chunk_overlap = 50 characters    (set in .env)
```

---

## Project structure
```
developer-knowledge-rag/
├── src/
│   ├── ingestion/
│   │   ├── loader.py        # loads .md .txt .pdf → Document objects
│   │   ├── chunker.py       # splits Documents into chunks
│   │   └── run_ingestion.py # entry point: load + chunk + report
│   ├── embeddings/          # Phase 4
│   ├── retrieval/           # Phase 5
│   ├── llm/                 # Phase 6
│   ├── rag/                 # Phase 7
│   ├── api/                 # Phase 8
│   └── config.py            # centralized settings
├── tests/
│   └── ingestion/
│       ├── test_loader.py   # 7 tests
│       └── test_chunker.py  # 9 tests
├── data/raw/                # your documentation files go here
├── vectorstore/             # ChromaDB files (auto-generated, Phase 4)
├── ui/                      # Streamlit chat interface (Phase 9)
└── pyproject.toml           # dependencies and project config
```

---

## Adding your own documentation

Drop any `.md`, `.txt`, or `.pdf` file into `data/raw/` and re-run the
ingestion pipeline. The system will automatically load, chunk, embed,
and index it — no code changes needed.