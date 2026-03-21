# src/ingestion/run_ingestion.py
# Runs the full ingestion + chunking pipeline.
# This is what you run whenever you add new docs to data/raw/

import logging
import sys
from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import DocumentChunker
from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_ingestion(data_dir: str = None):
    directory = data_dir or settings.data_directory
    logger.info(f"Starting ingestion from: {directory}")

    # Step 1: Load documents
    loader = DocumentLoader(data_directory=directory)
    documents = loader.load_directory()

    if not documents:
        logger.warning("No documents loaded. Check your data/raw/ folder.")
        return

    # Step 2: Chunk documents
    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(documents)

    # Step 3: Print report
    stats = chunker.get_stats(chunks)

    logger.info(f"\n{'='*40}")
    logger.info(f"INGESTION + CHUNKING REPORT")
    logger.info(f"{'='*40}")
    logger.info(f"Documents loaded:  {len(documents)}")
    logger.info(f"Total chunks:      {stats['total_chunks']}")
    logger.info(f"Avg chunk length:  {stats['avg_chunk_length']} chars")
    logger.info(f"Min chunk length:  {stats['min_chunk_length']} chars")
    logger.info(f"Max chunk length:  {stats['max_chunk_length']} chars")
    logger.info(f"{'='*40}")

    # Show a preview of the first 3 chunks
    logger.info("Preview — first 3 chunks:")
    for i, chunk in enumerate(chunks[:3]):
        logger.info(f"\n  Chunk {i+1}:")
        logger.info(f"  Source:  {chunk.metadata['file_name']}")
        logger.info(f"  Index:   {chunk.metadata['chunk_index']} of {chunk.metadata['total_chunks']}")
        logger.info(f"  Length:  {len(chunk.page_content)} chars")
        logger.info(f"  Content: {chunk.page_content[:120]}...")

    logger.info(f"\n{'='*40}")
    logger.info("Ready for Phase 4 — embeddings and vector store.")


if __name__ == "__main__":
    custom_dir = sys.argv[1] if len(sys.argv) > 1 else None
    run_ingestion(custom_dir)