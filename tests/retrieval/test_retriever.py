import pytest
from langchain_core.documents import Document
from src.retrieval.retriever import DocumentRetriever


@pytest.fixture
def retriever():
    return DocumentRetriever(k=3)


def test_retriever_initializes(retriever):
    assert retriever.k == 3
    assert retriever.vector_store is not None


def test_retrieve_returns_documents(retriever):
    docs = retriever.retrieve("How do I authenticate?")
    assert len(docs) > 0
    assert all(isinstance(d, Document) for d in docs)


def test_retrieve_respects_k_limit(retriever):
    docs = retriever.retrieve("authentication bearer token")
    assert len(docs) <= 3


def test_retrieve_returns_metadata(retriever):
    docs = retriever.retrieve("How do I authenticate?")
    for doc in docs:
        assert "file_name" in doc.metadata
        assert "source" in doc.metadata


def test_retrieve_empty_query_returns_empty(retriever):
    docs = retriever.retrieve("")
    assert docs == []


def test_retrieve_with_scores_returns_tuples(retriever):
    results = retriever.retrieve_with_scores("authentication")
    assert len(results) > 0
    for doc, score in results:
        assert isinstance(doc, Document)
        assert isinstance(score, float)


def test_retrieve_auth_query_finds_auth_doc(retriever):
    docs = retriever.retrieve("How do I authenticate with the API?")
    sources = [d.metadata.get("file_name") for d in docs]
    assert "authentication.md" in sources


def test_format_context_returns_string(retriever):
    docs = retriever.retrieve("authentication")
    context = retriever.format_retrieved_context(docs)
    assert isinstance(context, str)
    assert len(context) > 0


def test_format_context_includes_source(retriever):
    docs = retriever.retrieve("authentication")
    context = retriever.format_retrieved_context(docs)
    assert "Source" in context


def test_format_context_empty_returns_message(retriever):
    context = retriever.format_retrieved_context([])
    assert "No relevant" in context
