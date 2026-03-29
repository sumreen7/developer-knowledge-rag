# src/ingestion/run_ingestion.py
# Incremental ingestion: only processes new or updated files.
# Already-ingested files are skipped automatically.
#
# Usage:
#   python -m src.ingestion.run_ingestion              # ingest new files only
#   python -m src.ingestion.run_ingestion --force       # re-ingest everything
#   python -m src.ingestion.run_ingestion --reindex auth.md  # re-ingest one file

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


def run_ingestion(data_dir: str = None, force: bool = False, reindex_file: str = None):
    directory = data_dir or settings.data_directory
    logger.info(f"Starting ingestion from: {directory}")

    # Step 1 — Load all documents from disk
    loader = DocumentLoader(data_directory=directory)
    all_documents = loader.load_directory()

    if not all_documents:
        logger.warning("No documents found in data/raw/. Nothing to ingest.")
        return

    # Step 2 — Initialize embedder and check what's already ingested
    embedder = EmbeddingManager()

    if force:
        logger.info("FORCE mode — re-ingesting all documents")
        import shutil
        import os
        if os.path.exists(settings.chroma_persist_directory):
            shutil.rmtree(settings.chroma_persist_directory)
            logger.info("Deleted existing vectorstore")
        # Re-initialize embedder after deleting vectorstore
        embedder = EmbeddingManager()
        documents_to_process = all_documents

    elif reindex_file:
        logger.info(f"REINDEX mode — re-ingesting '{reindex_file}'")
        # Delete old chunks for this file, then re-ingest it
        embedder.delete_file_chunks(reindex_file)
        documents_to_process = [
            doc for doc in all_documents
            if doc.metadata.get("file_name") == reindex_file
        ]
        if not documents_to_process:
            logger.warning(f"File '{reindex_file}' not found in data/raw/")
            return

    else:
        # Normal mode — only ingest new files
        already_ingested = embedder.get_ingested_files()
        documents_to_process = [
            doc for doc in all_documents
            if doc.metadata.get("file_name") not in already_ingested
        ]

        if not documents_to_process:
            logger.info("All files are already ingested. Nothing new to process.")
            logger.info(f"Files in vectorstore: {sorted(already_ingested)}")
            stats = embedder.get_collection_stats()
            logger.info(f"Total chunks stored: {stats.get('total_chunks', 'N/A')}")
            return

        skipped = len(all_documents) - len(documents_to_process)
        if skipped > 0:
            logger.info(f"Skipping {skipped} already-ingested file(s)")

    # Step 3 — Chunk new documents
    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(documents_to_process)
    chunk_stats = chunker.get_stats(chunks)

    # Step 4 — Embed and store
    logger.info("Embedding and storing in ChromaDB...")
    embedder.embed_and_store(chunks)

    # Step 5 — Report
    collection_stats = embedder.get_collection_stats()

    logger.info(f"\n{'='*40}")
    logger.info("INGESTION REPORT")
    logger.info(f"{'='*40}")
    logger.info(f"Files on disk:      {len(all_documents)}")
    logger.info(f"New files ingested: {len(documents_to_process)}")
    logger.info(f"Chunks created:     {chunk_stats.get('total_chunks', 0)}")
    logger.info(f"Total in store:     {collection_stats.get('total_chunks', 'N/A')}")
    logger.info(f"Embedding model:    {collection_stats.get('embedding_model')}")
    logger.info(f"{'='*40}")

    # List what's now in the vectorstore
    all_ingested = embedder.get_ingested_files()
    logger.info(f"Files in vectorstore: {sorted(all_ingested)}")
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    force = "--force" in sys.argv
    reindex_file = None

    if "--reindex" in sys.argv:
        idx = sys.argv.index("--reindex")
        if idx + 1 < len(sys.argv):
            reindex_file = sys.argv[idx + 1]

    custom_dir = None
    for arg in sys.argv[1:]:
        if not arg.startswith("--") and arg != reindex_file:
            custom_dir = arg
            break

    run_ingestion(data_dir=custom_dir, force=force, reindex_file=reindex_file)