# src/api/main.py
# The entry point for our FastAPI application.
# Right now it only serves a health check endpoint.
# We'll add the RAG inference endpoint in Phase 8.

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys

# Import our settings singleton
from src.config import settings

# Create the FastAPI application instance
# title and version show up in the auto-generated /docs page
app = FastAPI(
    title="Developer Docs AI Assistant",
    description="RAG-powered assistant for querying developer documentation",
    version="0.1.0",
)

# CORS middleware — allows the Streamlit UI (on a different port) to call the API
# In production you'd restrict allow_origins to specific domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict:
    """
    Health check endpoint.
    
    Returns the status of the API and which models/config are loaded.
    CI/CD pipelines and Docker HEALTHCHECK commands poll this endpoint.
    If it returns 200, the service is considered healthy.
    """
    return {
        "status": "healthy",
        "version": "0.1.0",
        "config": {
            "ollama_model": settings.ollama_model,
            "embedding_model": settings.embedding_model,
            "chroma_collection": settings.chroma_collection_name,
        },
        "python_version": sys.version,
    }


@app.get("/")
async def root() -> dict:
    """Root redirect — tells developers where to find the API docs."""
    return {
        "message": "Developer Docs AI Assistant API",
        "docs": "/docs",           # Swagger UI
        "health": "/health",
    }