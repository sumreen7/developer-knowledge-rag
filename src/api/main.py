# src/api/main.py
# FastAPI application with two endpoints:
#   POST /query  — takes a question, returns a RAG answer with sources
#   GET  /health — checks all pipeline components are running
#
# The RAGPipeline is initialized ONCE at startup and reused for all requests.
# This is critical for performance — loading the embedding model and
# connecting to Ollama takes several seconds. We don't want that per request.

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import settings

logger = logging.getLogger(__name__)

# ── Request / Response models ─────────────────────────────────────────────────
# Pydantic models define the shape of data coming in and going out.
# FastAPI uses these for automatic validation and the /docs UI.

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The developer's question about the documentation",
        json_schema_extra={"example": "How do I authenticate with the API?"},
    )
    k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of document chunks to retrieve",
    )


class SourceModel(BaseModel):
    """A single source document cited in the answer."""
    file_name: str
    chunk_index: int
    content_preview: str
    similarity_score: float


class QueryResponse(BaseModel):
    """What the API returns from /query."""
    question: str
    answer: str
    sources: list[SourceModel]
    model: str
    num_chunks_retrieved: int


# ── App lifecycle ─────────────────────────────────────────────────────────────
# The lifespan context manager runs startup/shutdown logic.
# We initialize the RAG pipeline here so it's ready before the first request.

pipeline = None   # module-level — shared across all requests


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the RAG pipeline on startup, clean up on shutdown."""
    global pipeline

    logger.info("Starting up — initializing RAG pipeline...")
    try:
        from src.rag.pipeline import RAGPipeline
        pipeline = RAGPipeline(k=settings.chroma_collection_name and 3)
        logger.info("RAG pipeline ready")
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        # Don't crash on startup — health endpoint will report degraded state

    yield   # application runs here

    logger.info("Shutting down")
    pipeline = None


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Developer Docs AI Assistant",
    description=(
        "Ask natural language questions about developer documentation. "
        "Answers are grounded in your docs and include source citations."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Root endpoint — points developers to the docs and query endpoint."""
    return {
        "message": "Developer Docs AI Assistant",
        "docs": "/docs",
        "query": "POST /query",
        "health": "/health",
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Returns status of all pipeline components.
    Used by Docker HEALTHCHECK and monitoring systems.
    """
    if pipeline is None:
        return {
            "status": "degraded",
            "pipeline": "not initialized",
            "ollama": "unknown",
            "config": {
                "ollama_model": settings.ollama_model,
                "embedding_model": settings.embedding_model,
            },
        }

    pipeline_health = pipeline.check_health()

    return {
        "status": pipeline_health.get("pipeline", "unknown"),
        **pipeline_health,
        "config": {
            "ollama_model": settings.ollama_model,
            "embedding_model": settings.embedding_model,
            "chroma_collection": settings.chroma_collection_name,
        },
    }


@app.post("/query", response_model=QueryResponse)
async def query_docs(request: QueryRequest):
    """
    Answer a developer's question using the RAG pipeline.

    Retrieves relevant documentation chunks, passes them to the local LLM,
    and returns a grounded answer with source citations.

    Example request:
        POST /query
        {"question": "How do I authenticate with the API?"}

    Example response:
        {
            "question": "How do I authenticate with the API?",
            "answer": "Use Bearer tokens...",
            "sources": [{"file_name": "authentication.md", ...}],
            "model": "mistral",
            "num_chunks_retrieved": 3
        }
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized. Check /health for details.",
        )

    try:
        response = pipeline.query(request.question)
        return QueryResponse(**response.to_dict())

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}",
        ) 