# src/ingestion/run_ingestion.py
# Full pipeline: load → chunk → embed → store in ChromaDB

import logging
import sys
from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import DocumentChunker
from src.embeddings.embedder import EmbeddingManager
from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_ingestion(data_dir: str = None):
    directory = data_dir or settings.data_directory
    logger.info(f"Starting full ingestion pipeline from: {directory}")

    # Step 1 — Load
    loader = DocumentLoader(data_directory=directory)
    documents = loader.load_directory()

    if not documents:
        logger.warning("No documents loaded. Check your data/raw/ folder.")
        return

    # Step 2 — Chunk
    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(documents)
    stats = chunker.get_stats(chunks)

    # Step 3 — Embed and store
    logger.info("Starting embedding and storage in ChromaDB...")
    embedder = EmbeddingManager()
    vector_store = embedder.embed_and_store(chunks)

    # Step 4 — Report
    collection_stats = embedder.get_collection_stats()

    logger.info(f"\n{'='*40}")
    logger.info("FULL PIPELINE REPORT")
    logger.info(f"{'='*40}")
    logger.info(f"Documents loaded:   {len(documents)}")
    logger.info(f"Chunks created:     {stats['total_chunks']}")
    logger.info(f"Chunks stored:      {collection_stats.get('total_chunks', 'N/A')}")
    logger.info(f"Embedding model:    {collection_stats.get('embedding_model')}")
    logger.info(f"Vector store:       {collection_stats.get('persist_directory')}")
    logger.info(f"{'='*40}")
    logger.info("Pipeline complete. Ready for Phase 5 — retrieval.")


if __name__ == "__main__":
    custom_dir = sys.argv[1] if len(sys.argv) > 1 else None
    run_ingestion(custom_dir)