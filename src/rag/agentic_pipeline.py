# src/rag/agentic_pipeline.py
# Agentic RAG using LangGraph's prebuilt ReAct agent.
# Forces initial search, then lets agent decide on follow-up searches.

import logging
from dataclasses import dataclass, field
from typing import List

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from src.retrieval.retriever import DocumentRetriever
from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SourceDocument:
    file_name: str
    chunk_index: int
    content_preview: str
    similarity_score: float = 0.0


@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: List[SourceDocument]
    model: str
    num_chunks_retrieved: int
    agent_steps: int = 0
    search_queries: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
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
            "agent_steps": self.agent_steps,
            "search_queries": self.search_queries,
        }


_current_sources: List[SourceDocument] = []
_current_search_queries: List[str] = []
_retriever: DocumentRetriever = None


@tool
def search_documentation(query: str) -> str:
    """Search the developer documentation for information.
    Use this to find answers about the API, authentication,
    error codes, rate limits, endpoints, and getting started.
    Input should be a specific search query."""
    global _current_sources, _current_search_queries, _retriever

    logger.info(f"Agent searching: '{query}'")
    _current_search_queries.append(query)

    results = _retriever.retrieve_with_scores(query)

    if not results:
        return "No relevant documentation found for this query."

    for doc, score in results:
        _current_sources.append(
            SourceDocument(
                file_name=doc.metadata.get("file_name", "unknown"),
                chunk_index=doc.metadata.get("chunk_index", 0),
                content_preview=doc.page_content[:200],
                similarity_score=score,
            )
        )

    docs = [doc for doc, _ in results]
    return _retriever.format_retrieved_context(docs)


SYSTEM_PROMPT = """You are a developer documentation assistant. You MUST use the search_documentation tool to find answers. NEVER answer from your own knowledge.

RULES:
- You MUST call search_documentation at least once before answering ANY question.
- ONLY use information returned by search_documentation. Never use your training data.
- If search results don't contain the answer, say "I don't have enough information in the documentation."
- Always cite which document (file name) your answer comes from.
- For complex questions, search multiple times with different queries.
- Maximum 4 searches per question."""


class AgenticRAGPipeline:
    def __init__(self, k: int = 3, max_iterations: int = 4):
        global _retriever

        logger.info("Initializing Agentic RAG pipeline...")

        self.k = k
        self.max_iterations = max_iterations

        _retriever = DocumentRetriever(k=k)
        self.retriever = _retriever

        self.llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.1,
        )
        self.model_name = settings.ollama_model

        self.agent = create_react_agent(
            model=self.llm,
            tools=[search_documentation],
            prompt=SYSTEM_PROMPT,
        )

        logger.info(
            f"Agentic RAG pipeline ready — "
            f"model: {self.model_name}, "
            f"max_iterations: {max_iterations}, k: {k}"
        )

    def query(self, question: str) -> RAGResponse:
        global _current_sources, _current_search_queries

        question = question.strip()

        if not question:
            return RAGResponse(
                question=question,
                answer="Please provide a question.",
                sources=[],
                model=self.model_name,
                num_chunks_retrieved=0,
            )

        _current_sources = []
        _current_search_queries = []

        logger.info(f"Agentic RAG query: '{question}'")

        try:
            # FORCE the first search by doing it ourselves
            # This guarantees the agent always has documentation context
            initial_context = search_documentation.invoke(question)

            # Now let the agent reason with the pre-fetched context
            # and decide if it needs additional searches
            augmented_prompt = (
                f"I already searched the documentation for your question. "
                f"Here are the results:\n\n{initial_context}\n\n"
                f"Based on these search results, answer the following question. "
                f"If you need MORE information, use search_documentation with a different query. "
                f"If you have enough, provide your answer citing the source documents.\n\n"
                f"Question: {question}"
            )

            result = self.agent.invoke(
                {"messages": [{"role": "user", "content": augmented_prompt}]}
            )

            messages = result.get("messages", [])
            answer = "I couldn't generate an answer."
            for msg in reversed(messages):
                if hasattr(msg, "content") and msg.content:
                    if not hasattr(msg, "tool_calls") or not msg.tool_calls:
                        answer = msg.content
                        break

            agent_steps = len(_current_search_queries)

            seen = set()
            unique_sources = []
            for s in _current_sources:
                key = (s.file_name, s.chunk_index)
                if key not in seen:
                    seen.add(key)
                    unique_sources.append(s)

            response = RAGResponse(
                question=question,
                answer=answer,
                sources=unique_sources,
                model=self.model_name,
                num_chunks_retrieved=len(unique_sources),
                agent_steps=agent_steps,
                search_queries=list(_current_search_queries),
            )

            logger.info(
                f"Agentic answer generated — "
                f"{agent_steps} searches, "
                f"{len(unique_sources)} unique sources"
            )

            return response

        except Exception as e:
            logger.error(f"Agent failed: {e}")
            logger.info("Falling back to direct retrieval...")
            return self._fallback_query(question)

    def _fallback_query(self, question: str) -> RAGResponse:
        try:
            results = self.retriever.retrieve_with_scores(question)
            docs = [doc for doc, _ in results]
            context = self.retriever.format_retrieved_context(docs)

            from langchain_ollama import OllamaLLM
            fallback_llm = OllamaLLM(
                model=settings.ollama_model,
                base_url=settings.ollama_base_url,
                temperature=0.1,
            )

            prompt = (
                f"Answer this question using ONLY the context below. "
                f"If the answer isn't in the context, say you don't have enough information.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            )

            answer = fallback_llm.invoke(prompt)

            sources = [
                SourceDocument(
                    file_name=doc.metadata.get("file_name", "unknown"),
                    chunk_index=doc.metadata.get("chunk_index", 0),
                    content_preview=doc.page_content[:200],
                    similarity_score=score,
                )
                for doc, score in results
            ]

            return RAGResponse(
                question=question,
                answer=answer.strip(),
                sources=sources,
                model=self.model_name,
                num_chunks_retrieved=len(sources),
                agent_steps=0,
                search_queries=[question],
            )
        except Exception as e:
            return RAGResponse(
                question=question,
                answer=f"System error: {str(e)}",
                sources=[],
                model=self.model_name,
                num_chunks_retrieved=0,
            )

    def check_health(self) -> dict:
        try:
            self.llm.invoke("Say 'ok'.")
            ollama_ok = True
        except Exception:
            ollama_ok = False

        return {
            "pipeline": "healthy" if ollama_ok else "degraded",
            "pipeline_type": "agentic",
            "ollama": "connected" if ollama_ok else "unreachable",
            "model": self.model_name,
            "retriever_k": self.k,
            "max_iterations": self.max_iterations,
        }