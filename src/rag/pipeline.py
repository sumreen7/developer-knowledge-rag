# src/rag/pipeline.py
# The complete RAG pipeline — wires retrieval + LLM into one class.
# This is what the FastAPI endpoint and Chat UI call directly.

import logging
from dataclasses import dataclass, field
from typing import List

from src.retrieval.retriever import DocumentRetriever
from src.llm.ollama_client import OllamaClient
from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SourceDocument:
    """
    A single source document used to generate the answer.
    Returned alongside the answer so the UI can show citations.
    """
    file_name: str
    chunk_index: int
    content_preview: str    # first 200 chars — enough to show in UI
    similarity_score: float = 0.0


@dataclass
class RAGResponse:
    """
    The complete response from the RAG pipeline.

    Why a dataclass instead of a plain dict?
    - Type safety — IDE autocomplete works
    - Easy to convert to JSON for the FastAPI endpoint
    - Clear contract between pipeline and API layer
    """
    question: str
    answer: str
    sources: List[SourceDocument]
    model: str
    num_chunks_retrieved: int

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization in FastAPI."""
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [
                {
                    "file_name": s.file_name,
                    "chunk_index": s.chunk_index,
                    "content_preview": s.content_preview,
                    "similarity_score": round(s.similarity_score, 3),
                }
                for s in self.sources
            ],
            "model": self.model,
            "num_chunks_retrieved": self.num_chunks_retrieved,
        }


class RAGPipeline:
    """
    Orchestrates the full RAG flow: retrieve → format → generate.

    This is the single entry point for answering developer questions.
    The API endpoint in Phase 8 creates one instance of this class
    at startup and reuses it for all requests.

    Why initialize retriever and LLM client once?
    Both involve loading models into memory — the embedding model
    and the Ollama connection. Loading them once at startup and
    reusing them per request is much faster than reloading per query.
    """

    def __init__(self, k: int = 3):
        """
        Args:
            k: number of chunks to retrieve per question.
               3 is the default — enough context without overwhelming the LLM.
        """
        logger.info("Initializing RAG pipeline...")

        # Initialize retriever — loads embedding model + ChromaDB connection
        self.retriever = DocumentRetriever(k=k)

        # Initialize LLM client — connects to Ollama
        self.llm_client = OllamaClient()

        self.k = k
        logger.info(
            f"RAG pipeline ready — "
            f"model: {self.llm_client.model}, "
            f"retrieving top {self.k} chunks per query"
        )

    def query(self, question: str) -> RAGResponse:
        """
        Answer a developer's question using retrieved documentation.

        Full flow:
        1. Retrieve top-k relevant chunks from ChromaDB
        2. Format chunks into labelled context string
        3. Send context + question to Mistral via Ollama
        4. Return structured response with answer + sources

        Args:
            question: the developer's natural language question

        Returns:
            RAGResponse with answer, sources, and metadata
        """
        question = question.strip()

        if not question:
            return RAGResponse(
                question=question,
                answer="Please provide a question.",
                sources=[],
                model=self.llm_client.model,
                num_chunks_retrieved=0,
            )

        logger.info(f"RAG query: '{question}'")

        # Step 1 — Retrieve relevant chunks
        retrieved = self.retriever.retrieve_with_scores(question)

        if not retrieved:
            return RAGResponse(
                question=question,
                answer="I couldn't find any relevant documentation to answer your question.",
                sources=[],
                model=self.llm_client.model,
                num_chunks_retrieved=0,
            )

        # Step 2 — Format context for the LLM
        docs = [doc for doc, _ in retrieved]
        context = self.retriever.format_retrieved_context(docs)

        # Step 3 — Generate answer
        answer = self.llm_client.generate(
            context=context,
            question=question,
        )

        # Step 4 — Build structured response with source citations
        sources = [
            SourceDocument(
                file_name=doc.metadata.get("file_name", "unknown"),
                chunk_index=doc.metadata.get("chunk_index", 0),
                content_preview=doc.page_content[:200],
                similarity_score=score,
            )
            for doc, score in retrieved
        ]

        response = RAGResponse(
            question=question,
            answer=answer,
            sources=sources,
            model=self.llm_client.model,
            num_chunks_retrieved=len(retrieved),
        )

        logger.info(
            f"Answer generated — "
            f"{len(retrieved)} sources, "
            f"model: {self.llm_client.model}"
        )

        return response

    def check_health(self) -> dict:
        """
        Check that all pipeline components are working.
        Called by the FastAPI /health endpoint in Phase 8.
        """
        ollama_ok = self.llm_client.check_connection()
        store_stats = self.llm_client.get_model_info()

        return {
            "pipeline": "healthy" if ollama_ok else "degraded",
            "ollama": "connected" if ollama_ok else "unreachable",
            "model": self.llm_client.model,
            "retriever_k": self.k,
        }