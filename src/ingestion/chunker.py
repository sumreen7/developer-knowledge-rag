# src/ingestion/chunker.py
# Splits loaded Documents into smaller chunks for embedding and retrieval.
# Takes List[Document] from the loader and returns List[Document] —
# the same type, just more of them, each smaller.

import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Splits documents into overlapping chunks.

    Why RecursiveCharacterTextSplitter?
    It tries to split on natural boundaries in this order:
      1. Paragraphs (\n\n)
      2. Lines (\n)
      3. Sentences (". ")
      4. Words (" ")
      5. Characters (last resort)

    This means it never cuts mid-word if it can avoid it, and prefers
    to cut at paragraph boundaries — which keeps semantic meaning intact.
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        # Use values from config if not explicitly passed
        # This allows tests to override with small values
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        # Initialize the splitter once — reused for all documents
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            # Keep track of character position within the original document
            add_start_index=True,
            # These are the separators tried in order
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        logger.info(
            f"DocumentChunker ready — "
            f"chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}"
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of Documents into chunks.

        Each chunk:
        - Inherits ALL metadata from its parent document
        - Gets two new metadata fields added:
            chunk_index: position of this chunk within its source document
            total_chunks: how many chunks the source was split into
        - Has page_content trimmed of leading/trailing whitespace

        Args:
            documents: list of Documents from the loader

        Returns:
            list of chunk Documents, ready for embedding
        """
        if not documents:
            logger.warning("No documents provided to chunker")
            return []

        all_chunks: List[Document] = []

        for doc in documents:
            # Skip documents with no meaningful content
            if not doc.page_content.strip():
                logger.warning(
                    f"Skipping empty document: "
                    f"{doc.metadata.get('file_name', 'unknown')}"
                )
                continue

            # Split this document into chunks
            # split_documents preserves metadata from the parent
            raw_chunks = self.splitter.split_documents([doc])

            # Enrich each chunk with position metadata
            total = len(raw_chunks)
            for idx, chunk in enumerate(raw_chunks):
                chunk.metadata["chunk_index"] = idx
                chunk.metadata["total_chunks"] = total
                # Clean up whitespace that sometimes appears at boundaries
                chunk.page_content = chunk.page_content.strip()

            all_chunks.extend(raw_chunks)

            logger.info(
                f"{doc.metadata.get('file_name', 'unknown')} → "
                f"{total} chunk(s)"
            )

        logger.info(
            f"Chunking complete: {len(documents)} documents → "
            f"{len(all_chunks)} chunks"
        )
        return all_chunks

    def get_stats(self, chunks: List[Document]) -> dict:
        """
        Returns a summary of chunk statistics.
        Useful for debugging — helps you tune chunk_size.
        """
        if not chunks:
            return {}

        lengths = [len(c.page_content) for c in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_chunk_length": round(sum(lengths) / len(lengths)),
            "min_chunk_length": min(lengths),
            "max_chunk_length": max(lengths),
            "chunk_size_setting": self.chunk_size,
            "chunk_overlap_setting": self.chunk_overlap,
        }