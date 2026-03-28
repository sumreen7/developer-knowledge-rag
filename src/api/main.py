# src/api/main.py
# Supports both naive and agentic RAG modes via PIPELINE_MODE env var.
# Set PIPELINE_MODE=agentic in .env to use agentic mode.

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import settings

logger = logging.getLogger(__name__)


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
    file_name: str
    chunk_index: int
    content_preview: str
    similarity_score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceModel]
    model: str
    num_chunks_retrieved: int
    agent_steps: int = 0
    search_queries: list[str] = []


pipeline = None
pipeline_mode = os.environ.get("PIPELINE_MODE", "naive")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, pipeline_mode

    logger.info(f"Starting up — pipeline mode: {pipeline_mode}")
    try:
        if pipeline_mode == "agentic":
            from src.rag.agentic_pipeline import AgenticRAGPipeline
            pipeline = AgenticRAGPipeline(k=3, max_iterations=4)
            logger.info("Agentic RAG pipeline ready")
        else:
            from src.rag.pipeline import RAGPipeline
            pipeline = RAGPipeline(k=3)
            logger.info("Naive RAG pipeline ready")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")

    yield
    logger.info("Shutting down")
    pipeline = None


app = FastAPI(
    title="Developer Docs AI Assistant",
    description=(
        "Ask natural language questions about developer documentation. "
        "Supports both naive and agentic RAG modes."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "Developer Docs AI Assistant",
        "pipeline_mode": pipeline_mode,
        "docs": "/docs",
        "query": "POST /query",
        "health": "/health",
    }


@app.get("/health")
async def health_check():
    if pipeline is None:
        return {
            "status": "degraded",
            "pipeline": "not initialized",
            "pipeline_mode": pipeline_mode,
            "config": {
                "ollama_model": settings.ollama_model,
                "embedding_model": settings.embedding_model,
            },
        }

    pipeline_health = pipeline.check_health()

    return {
        "status": pipeline_health.get("pipeline", "unknown"),
        **pipeline_health,
        "pipeline_mode": pipeline_mode,
        "config": {
            "ollama_model": settings.ollama_model,
            "embedding_model": settings.embedding_model,
            "chroma_collection": settings.chroma_collection_name,
        },
    }


@app.post("/query", response_model=QueryResponse)
async def query_docs(request: QueryRequest):
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized. Check /health.",
        )

    try:
        response = pipeline.query(request.question)
        result = response.to_dict()

        if "agent_steps" not in result:
            result["agent_steps"] = 0
        if "search_queries" not in result:
            result["search_queries"] = []

        return QueryResponse(**result)

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}",
        )
