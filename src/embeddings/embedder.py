# src/embeddings/embedder.py
# Converts text chunks into embedding vectors using SentenceTransformers.
# Also handles storing and loading from ChromaDB.

import logging
from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages the embedding model and ChromaDB vector store.

    Responsibilities:
    - Load the SentenceTransformers embedding model
    - Convert chunks into vectors and store in ChromaDB
    - Provide the vector store for retrieval in Phase 5

    Why a class?
    The embedding model takes a few seconds to load.
    By wrapping it in a class, we load it once and reuse it
    for all chunks — rather than reloading per chunk.
    """

    def __init__(
        self,
        model_name: str = None,
        persist_directory: str = None,
        collection_name: str = None,
    ):
        self.model_name = model_name or settings.embedding_model
        self.persist_directory = persist_directory or settings.chroma_persist_directory
        self.collection_name = collection_name or settings.chroma_collection_name

        logger.info(f"Loading embedding model: {self.model_name}")

        # HuggingFaceEmbeddings wraps SentenceTransformers in LangChain's interface
        # model_kwargs: run on CPU (change to 'cuda' if you have a GPU)
        # encode_kwargs: normalize_embeddings=True improves similarity search accuracy
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info(f"Embedding model loaded: {self.model_name}")
        logger.info(f"Vector store directory: {self.persist_directory}")
        logger.info(f"Collection name: {self.collection_name}")

    def embed_and_store(self, chunks: List[Document]) -> Chroma:
        """
        Convert chunks to vectors and store in ChromaDB.

        This is the main method called during ingestion.
        It does two things in one step:
        1. Calls the embedding model on each chunk's page_content
        2. Stores the vector + metadata + original text in ChromaDB

        ChromaDB persists to disk automatically at self.persist_directory.
        Next time you call get_vector_store(), it loads from disk —
        no need to re-embed.

        Args:
            chunks: list of Document chunks from the chunker

        Returns:
            Chroma vector store object (used by retriever in Phase 5)
        """
        if not chunks:
            raise ValueError("No chunks provided to embed")

        logger.info(f"Embedding {len(chunks)} chunks into ChromaDB...")

        # Chroma.from_documents does everything in one call:
        # - calls embedding_function on each chunk's page_content
        # - stores the vector, metadata, and text in the collection
        # - persists to disk at persist_directory
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_function,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
        )

        logger.info(
            f"Successfully stored {len(chunks)} chunks in ChromaDB "
            f"at {self.persist_directory}"
        )

        return vector_store

    def get_vector_store(self) -> Chroma:
        """
        Load an existing ChromaDB collection from disk.

        Called during query time (Phase 5 retrieval) — loads the
        already-embedded chunks without re-embedding anything.

        Returns:
            Chroma vector store object ready for similarity search
        """
        logger.info(
            f"Loading existing vector store from: {self.persist_directory}"
        )

        vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory,
        )

        # Quick check — how many chunks are stored?
        count = vector_store._collection.count()
        logger.info(f"Vector store loaded: {count} chunks available")

        return vector_store

    def get_collection_stats(self) -> dict:
        """
        Returns basic stats about what's stored in ChromaDB.
        Useful for debugging — tells you if ingestion worked.
        """
        try:
            store = self.get_vector_store()
            count = store._collection.count()
            return {
                "collection_name": self.collection_name,
                "total_chunks": count,
                "persist_directory": self.persist_directory,
                "embedding_model": self.model_name,
            }
        except Exception as e:
            return {"error": str(e)}