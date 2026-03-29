# src/embeddings/embedder.py

import logging
from typing import List, Set

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings

logger = logging.getLogger(__name__)


class EmbeddingManager:
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

        self.embedding_function = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info(f"Embedding model loaded: {self.model_name}")
        logger.info(f"Vector store directory: {self.persist_directory}")
        logger.info(f"Collection name: {self.collection_name}")

    def embed_and_store(self, chunks: List[Document]) -> Chroma:
        if not chunks:
            raise ValueError("No chunks provided to embed")

        logger.info(f"Embedding {len(chunks)} chunks into ChromaDB...")

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
        logger.info(
            f"Loading existing vector store from: {self.persist_directory}"
        )

        vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory,
        )

        count = vector_store._collection.count()
        logger.info(f"Vector store loaded: {count} chunks available")

        return vector_store

    def get_ingested_files(self) -> Set[str]:
        """Returns the set of file names already stored in ChromaDB."""
        try:
            store = self.get_vector_store()
            collection = store._collection
            results = collection.get(include=["metadatas"])

            ingested = set()
            for m in results["metadatas"]:
                fname = m.get("file_name")
                if fname:
                    ingested.add(fname)

            logger.info(f"Already ingested files: {ingested}")
            return ingested

        except Exception as e:
            logger.warning(f"Could not read existing collection: {e}")
            return set()

    def delete_file_chunks(self, file_name: str) -> int:
        """Delete all chunks belonging to a specific file.
        Use this when a file has been updated and needs re-ingestion."""
        try:
            store = self.get_vector_store()
            collection = store._collection

            # Find all chunk IDs for this file
            results = collection.get(
                where={"file_name": file_name},
                include=["metadatas"],
            )

            ids_to_delete = results["ids"]

            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} chunks from '{file_name}'")
            else:
                logger.info(f"No existing chunks found for '{file_name}'")

            return len(ids_to_delete)

        except Exception as e:
            logger.warning(f"Could not delete chunks for '{file_name}': {e}")
            return 0

    def get_collection_stats(self) -> dict:
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