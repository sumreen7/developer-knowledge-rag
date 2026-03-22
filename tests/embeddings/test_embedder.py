import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from src.embeddings.embedder import EmbeddingManager


@pytest.fixture
def sample_chunks():
    return [
        Document(
            page_content="Use Bearer tokens for authentication.",
            metadata={
                "source": "data/raw/authentication.md",
                "file_name": "authentication.md",
                "file_type": "md",
                "chunk_index": 0,
                "total_chunks": 1,
            }
        ),
        Document(
            page_content="GET /users returns a list of all users.",
            metadata={
                "source": "data/raw/api_endpoints.md",
                "file_name": "api_endpoints.md",
                "file_type": "md",
                "chunk_index": 0,
                "total_chunks": 1,
            }
        ),
    ]


def test_embedder_initializes(tmp_path):
    embedder = EmbeddingManager(
        model_name="all-MiniLM-L6-v2",
        persist_directory=str(tmp_path),
        collection_name="test_collection",
    )
    assert embedder.model_name == "all-MiniLM-L6-v2"
    assert embedder.collection_name == "test_collection"


def test_embed_and_store_returns_vector_store(tmp_path, sample_chunks):
    embedder = EmbeddingManager(
        persist_directory=str(tmp_path),
        collection_name="test_collection",
    )
    vector_store = embedder.embed_and_store(sample_chunks)
    assert vector_store is not None


def test_embed_and_store_correct_count(tmp_path, sample_chunks):
    embedder = EmbeddingManager(
        persist_directory=str(tmp_path),
        collection_name="test_collection",
    )
    embedder.embed_and_store(sample_chunks)
    stats = embedder.get_collection_stats()
    assert stats["total_chunks"] == len(sample_chunks)


def test_embed_empty_chunks_raises(tmp_path):
    embedder = EmbeddingManager(
        persist_directory=str(tmp_path),
        collection_name="test_collection",
    )
    with pytest.raises(ValueError, match="No chunks"):
        embedder.embed_and_store([])


def test_get_vector_store_loads_existing(tmp_path, sample_chunks):
    embedder = EmbeddingManager(
        persist_directory=str(tmp_path),
        collection_name="test_collection",
    )
    embedder.embed_and_store(sample_chunks)

    # Load it back — should work without re-embedding
    store = embedder.get_vector_store()
    assert store is not None


def test_collection_stats_returns_correct_keys(tmp_path, sample_chunks):
    embedder = EmbeddingManager(
        persist_directory=str(tmp_path),
        collection_name="test_collection",
    )
    embedder.embed_and_store(sample_chunks)
    stats = embedder.get_collection_stats()

    assert "collection_name" in stats
    assert "total_chunks" in stats
    assert "embedding_model" in stats
    assert "persist_directory" in stats
