# Developer Knowledge RAG

A production-quality RAG (Retrieval-Augmented Generation) system that lets
developers ask natural language questions about documentation and receive
grounded, source-cited answers — powered by a local LLM running entirely
on your machine. No API keys. No internet required after setup.

---

## Live demo flow

```
Developer types: "How do I authenticate with the API?"
        ↓
System searches ChromaDB for relevant doc chunks
        ↓
Mistral reads the chunks and writes a grounded answer
        ↓
UI shows the answer + source cards with relevance scores
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
| Chat UI       | Streamlit                               |
| Deployment    | Docker (Phase 11)                       |

---

## Getting started

```bash
# 1. Clone and set up
git clone https://github.com/sumreen7/developer-knowledge-rag.git
cd developer-knowledge-rag
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install langchain-chroma langchain-huggingface langchain-ollama langchain-text-splitters markdown pytest-mock
cp .env.example .env

# 2. Pull a local model (one time)
ollama pull mistral

# 3. Add your documentation files to data/raw/
# Supported: .md  .txt  .pdf

# 4. Run the ingestion pipeline
python -m src.ingestion.run_ingestion

# 5. Start the API (Terminal 1)
.venv/bin/uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# 6. Start the chat UI (Terminal 2)
streamlit run ui/app.py
```

Open `http://localhost:8501` in your browser.

---

## Adding your own documentation

Drop any `.md`, `.txt`, or `.pdf` file into `data/raw/` then re-run:

```bash
# Clean re-index (recommended when editing existing docs)
rm -rf vectorstore/
python -m src.ingestion.run_ingestion

# Then restart the API
pkill -f uvicorn
.venv/bin/uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

---

## API endpoints

| Method | Path      | Description                          |
|--------|-----------|--------------------------------------|
| GET    | `/health` | Pipeline status + model info         |
| POST   | `/query`  | Ask a question, get answer + sources |
| GET    | `/docs`   | Interactive Swagger UI               |

Example query:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I authenticate with the API?"}'
```

Example response:

```json
{
  "question": "How do I authenticate with the API?",
  "answer": "Use Bearer tokens. Send POST /auth/token...",
  "sources": [
    {
      "file_name": "authentication.md",
      "chunk_index": 0,
      "content_preview": "Authentication Guide...",
      "similarity_score": 0.611
    }
  ],
  "model": "mistral",
  "num_chunks_retrieved": 3
}
```

---

## Verifying answers

Every answer includes source cards showing which document and chunk it
came from. To verify manually:

```bash
# See exact chunks retrieved for any question
python -m src.retrieval.run_retrieval "your question here"

# Check the raw document
cat data/raw/authentication.md
```

---

## Running tests

```bash
.venv/bin/pytest tests/ -v
```

Current test count: **63 passing**

---

## Build phases

| Phase | Description                        | Status     |
|-------|------------------------------------|------------|
| 1     | Project setup, config, health API  | ✅ Done    |
| 2     | Document ingestion pipeline        | ✅ Done    |
| 3     | Document chunking                  | ✅ Done    |
| 4     | Embeddings + ChromaDB vector store | ✅ Done    |
| 5     | Retrieval pipeline                 | ✅ Done    |
| 6     | Ollama LLM integration             | ✅ Done    |
| 7     | RAG pipeline                       | ✅ Done    |
| 8     | FastAPI inference endpoint         | ✅ Done    |
| 9     | Chat UI (Streamlit)                | ✅ Done    |
| 10    | Evaluation (RAG metrics)           | 🔄 Next    |
| 11    | Docker deployment                  | Pending    |

---

## Project structure

```
developer-knowledge-rag/
├── src/
│   ├── ingestion/
│   │   ├── loader.py          # loads .md .txt .pdf → Documents
│   │   ├── chunker.py         # splits Documents into chunks
│   │   └── run_ingestion.py   # load → chunk → embed → store
│   ├── embeddings/
│   │   └── embedder.py        # SentenceTransformers + ChromaDB
│   ├── retrieval/
│   │   ├── retriever.py       # similarity search
│   │   └── run_retrieval.py   # interactive retrieval test
│   ├── llm/
│   │   ├── ollama_client.py   # Ollama / Mistral integration
│   │   └── run_llm_test.py    # LLM connection test
│   ├── rag/
│   │   ├── pipeline.py        # full RAG: retrieve → generate
│   │   └── run_rag.py         # end-to-end test script
│   ├── api/
│   │   └── main.py            # FastAPI endpoints
│   └── config.py              # settings from .env
├── tests/                     # 63 passing tests
├── ui/
│   └── app.py                 # Streamlit chat interface
├── data/raw/                  # put your docs here
├── vectorstore/               # ChromaDB (auto-generated)
└── pyproject.toml
```

---

## How chunking works

Documents are split into 512-character overlapping pieces using
`RecursiveCharacterTextSplitter`. It cuts at natural boundaries:
paragraph → line → sentence → word. The 50-character overlap ensures
answers spanning section boundaries are never lost.

---

## How embeddings work

Each chunk is converted to a 384-number vector by `all-MiniLM-L6-v2`.
Chunks with similar meaning produce similar vectors. At query time the
question is embedded using the same model and ChromaDB returns the
closest matching chunks by cosine similarity.