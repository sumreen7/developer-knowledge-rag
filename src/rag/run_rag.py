# src/rag/run_rag.py
# Interactive test — ask any question and get a full RAG answer.
# Run with: python -m src.rag.run_rag
# Or:       python -m src.rag.run_rag "your question here"

import logging
import sys

from src.rag.pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_rag(question: str = None):
    questions = [
        question or "How do I authenticate with the API?",
    ]

    logger.info("Initializing RAG pipeline...")
    pipeline = RAGPipeline(k=3)

    for q in questions:
        logger.info(f"\n{'='*60}")
        logger.info(f"QUESTION: {q}")
        logger.info(f"{'='*60}")

        response = pipeline.query(q)

        print(f"\nANSWER:\n{response.answer}")
        print(f"\nSOURCES:")
        for i, source in enumerate(response.sources, 1):
            print(f"  [{i}] {source.file_name} "
                  f"(chunk {source.chunk_index}, "
                  f"score: {source.similarity_score:.3f})")
            print(f"      {source.content_preview[:100]}...")

        print(f"\nModel: {response.model}")
        print(f"Chunks retrieved: {response.num_chunks_retrieved}")
        logger.info(f"{'='*60}")


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    run_rag(question)