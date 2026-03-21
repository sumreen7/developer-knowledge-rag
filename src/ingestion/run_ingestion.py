# src/ingestion/run_ingestion.py

import logging
import sys
from src.ingestion.loader import DocumentLoader
from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_ingestion(data_dir: str = None):
    directory = data_dir or settings.data_directory
    logger.info(f"Starting ingestion from: {directory}")

    loader = DocumentLoader(data_directory=directory)
    documents = loader.load_directory()

    if not documents:
        logger.warning("No documents loaded.")
        return

    # Summary
    files_seen = {}
    for doc in documents:
        name = doc.metadata.get("file_name", "unknown")
        files_seen[name] = files_seen.get(name, 0) + 1

    logger.info(f"\n{'='*40}")
    logger.info(f"INGESTION REPORT — {len(documents)} total documents")
    logger.info(f"{'='*40}")
    for name, count in sorted(files_seen.items()):
        logger.info(f"  {name}: {count} document(s)")

    logger.info(f"\nPreview of first document:")
    logger.info(f"  File:    {documents[0].metadata['file_name']}")
    logger.info(f"  Type:    {documents[0].metadata['file_type']}")
    logger.info(f"  Content: {documents[0].page_content[:150]}...")
    logger.info(f"{'='*40}")


if __name__ == "__main__":
    custom_dir = sys.argv[1] if len(sys.argv) > 1 else None
    run_ingestion(custom_dir)