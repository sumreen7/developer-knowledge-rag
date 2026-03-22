# src/retrieval/run_retrieval.py
# Quick test script — ask a question, see what gets retrieved.
# Run with: python -m src.retrieval.run_retrieval

import logging
import sys

from src.retrieval.retriever import DocumentRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_retrieval(query: str = None):
    query = query or "How do I authenticate with the API?"

    logger.info(f"Query: '{query}'")
    logger.info("=" * 50)

    retriever = DocumentRetriever(k=3)

    # Retrieve with scores so we can see how confident the system is
    results = retriever.retrieve_with_scores(query)

    logger.info("\nRETRIEVAL RESULTS")
    logger.info("=" * 50)

    for i, (doc, score) in enumerate(results, 1):
        logger.info(f"\nResult {i} — similarity score: {score:.3f}")
        logger.info(f"  Source:  {doc.metadata.get('file_name')}")
        logger.info(f"  Chunk:   {doc.metadata.get('chunk_index')} of {doc.metadata.get('total_chunks')}")
        logger.info(f"  Content: {doc.page_content[:200]}...")

    logger.info("\n" + "=" * 50)
    logger.info("FORMATTED CONTEXT (what the LLM will see in Phase 7):")
    logger.info("=" * 50)

    docs = [doc for doc, _ in results]
    context = retriever.format_retrieved_context(docs)
    logger.info(f"\n{context}")


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    run_retrieval(query)