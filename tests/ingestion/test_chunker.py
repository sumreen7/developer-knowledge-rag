import pytest
from langchain_core.documents import Document
from src.ingestion.chunker import DocumentChunker


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="Bearer tokens are used for authentication. "
                         "Send a POST request to /auth/token with your credentials. "
                         "The response includes an access_token valid for 24 hours. "
                         "Include the token in every request as Authorization: Bearer <token>. "
                         "Tokens expire after 24 hours. Use /auth/refresh to renew.",
            metadata={
                "source": "data/raw/authentication.md",
                "file_name": "authentication.md",
                "file_type": "md",
                "load_timestamp": "2026-01-01T00:00:00",
            }
        ),
        Document(
            page_content="GET /users returns a list of users. "
                         "POST /users creates a new user. "
                         "DELETE /users/{id} removes a user permanently.",
            metadata={
                "source": "data/raw/api_endpoints.md",
                "file_name": "api_endpoints.md",
                "file_type": "md",
                "load_timestamp": "2026-01-01T00:00:00",
            }
        ),
    ]


def test_chunker_initializes():
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
    assert chunker.chunk_size == 100
    assert chunker.chunk_overlap == 10


def test_chunking_returns_documents(sample_documents):
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk_documents(sample_documents)
    assert len(chunks) > 0
    assert all(isinstance(c, Document) for c in chunks)


def test_chunks_are_smaller_than_originals(sample_documents):
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk_documents(sample_documents)
    for chunk in chunks:
        assert len(chunk.page_content) <= 150


def test_chunks_inherit_parent_metadata(sample_documents):
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk_documents(sample_documents)
    for chunk in chunks:
        assert "file_name" in chunk.metadata
        assert "file_type" in chunk.metadata
        assert "source" in chunk.metadata


def test_chunks_have_position_metadata(sample_documents):
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk_documents(sample_documents)
    for chunk in chunks:
        assert "chunk_index" in chunk.metadata
        assert "total_chunks" in chunk.metadata


def test_chunk_index_starts_at_zero(sample_documents):
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk_documents(sample_documents)
    # First chunk of each document should have index 0
    first_chunks = [c for c in chunks if c.metadata["chunk_index"] == 0]
    assert len(first_chunks) == len(sample_documents)


def test_empty_input_returns_empty_list():
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk_documents([])
    assert chunks == []


def test_get_stats_returns_correct_keys(sample_documents):
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk_documents(sample_documents)
    stats = chunker.get_stats(chunks)
    assert "total_chunks" in stats
    assert "avg_chunk_length" in stats
    assert "min_chunk_length" in stats
    assert "max_chunk_length" in stats


def test_get_stats_empty_returns_empty_dict():
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
    stats = chunker.get_stats([])
    assert stats == {}
