# src/retrieval/retriever.py
# Searches ChromaDB for chunks relevant to a query.
# Takes a question string, returns a list of Documents with scores.

import logging
from typing import List, Tuple

from langchain_core.documents import Document

from src.embeddings.embedder import EmbeddingManager
from src.config import settings

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """
    Searches the vector store for chunks relevant to a query.

    Why a separate class from EmbeddingManager?
    EmbeddingManager owns storage — writing vectors to ChromaDB.
    DocumentRetriever owns search — reading vectors from ChromaDB.
    Single responsibility: each class does one thing.

    At query time, this class:
    1. Loads the existing ChromaDB collection from disk
    2. Converts the query to a vector using the same embedding model
    3. Finds the k most similar chunks by cosine similarity
    4. Returns them as Documents with their metadata intact
    """

    def __init__(self, k: int = 3):
        """
        Args:
            k: number of chunks to retrieve per query.
               3 is a good default — enough context, not too noisy.
               Increase to 5 for longer documents.
        """
        self.k = k

        # Load the embedding manager — this loads the model and
        # connects to the existing ChromaDB collection on disk
        self.embedding_manager = EmbeddingManager()
        self.vector_store = self.embedding_manager.get_vector_store()

        logger.info(f"DocumentRetriever ready — returning top {self.k} chunks per query")

    def retrieve(self, query: str) -> List[Document]:
        """
        Find the most relevant chunks for a query.

        This is the simplest retrieval method — pure similarity search.
        The query is embedded and compared to all stored chunk vectors.
        The k closest chunks are returned.

        Args:
            query: the developer's question as a plain string

        Returns:
            List of the k most relevant Document chunks,
            ordered by relevance (most relevant first)
        """
        if not query or not query.strip():
            logger.warning("Empty query received")
            return []

        logger.info(f"Retrieving top {self.k} chunks for query: '{query}'")

        # similarity_search embeds the query and finds nearest neighbors
        # Returns List[Document] ordered by similarity (closest first)
        results = self.vector_store.similarity_search(
            query=query,
            k=self.k,
        )

        logger.info(f"Retrieved {len(results)} chunks")
        for i, doc in enumerate(results):
            logger.info(
                f"  [{i+1}] {doc.metadata.get('file_name', 'unknown')} "
                f"(chunk {doc.metadata.get('chunk_index', '?')})"
            )

        return results

    def retrieve_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """
        Same as retrieve() but also returns the similarity score for each chunk.

        Score is a float between 0 and 1 — higher means more similar.
        Useful for debugging and for Phase 10 evaluation.

        Args:
            query: the developer's question

        Returns:
            List of (Document, score) tuples, ordered by score descending
        """
        if not query or not query.strip():
            return []

        results = self.vector_store.similarity_search_with_relevance_scores(
            query=query,
            k=self.k,
        )

        logger.info(f"Retrieved {len(results)} chunks with scores:")
        for doc, score in results:
            logger.info(
                f"  score={score:.3f} | "
                f"{doc.metadata.get('file_name', 'unknown')} "
                f"chunk {doc.metadata.get('chunk_index', '?')}"
            )

        return results

    def format_retrieved_context(self, docs: List[Document]) -> str:
        """
        Formats retrieved chunks into a single context string for the LLM.

        This is what gets injected into the prompt in Phase 7.
        Each chunk is labelled with its source so the LLM can cite it.

        Args:
            docs: list of retrieved Document chunks

        Returns:
            formatted string ready to inject into an LLM prompt
        """
        if not docs:
            return "No relevant documentation found."

        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("file_name", "unknown")
            chunk_idx = doc.metadata.get("chunk_index", "?")
            context_parts.append(
                f"[Source {i}: {source} — chunk {chunk_idx}]\n"
                f"{doc.page_content}"
            )

        return "\n\n---\n\n".join(context_parts)