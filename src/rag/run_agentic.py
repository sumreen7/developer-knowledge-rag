# src/rag/run_agentic.py
# Interactive test for agentic RAG.
# Run with: python -m src.rag.run_agentic
# Or:       python -m src.rag.run_agentic "your question here"

import logging
import sys

from src.rag.agentic_pipeline import AgenticRAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_agentic(question: str = None):
    questions = [
        question or "What is the difference between a 401 and 403 error, and how do I fix each one?",
    ]

    logger.info("Initializing Agentic RAG pipeline...")
    pipeline = AgenticRAGPipeline(k=3, max_iterations=4)

    for q in questions:
        logger.info(f"\n{'='*60}")
        logger.info(f"QUESTION: {q}")
        logger.info(f"{'='*60}")

        response = pipeline.query(q)

        print(f"\nANSWER:\n{response.answer}")
        print(f"\nAGENT REASONING:")
        print(f"  Steps taken: {response.agent_steps}")
        print(f"  Searches made: {len(response.search_queries)}")
        for i, sq in enumerate(response.search_queries, 1):
            print(f"    [{i}] '{sq}'")

        print(f"\nSOURCES ({response.num_chunks_retrieved}):")
        for i, source in enumerate(response.sources, 1):
            print(f"  [{i}] {source.file_name} "
                  f"(chunk {source.chunk_index}, "
                  f"score: {source.similarity_score:.3f})")
            print(f"      {source.content_preview[:100]}...")

        print(f"\nModel: {response.model}")
        logger.info(f"{'='*60}")


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    run_agentic(question)