import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def client_with_pipeline():
    """Create test client with a mocked RAG pipeline."""
    mock_response = MagicMock()
    mock_response.to_dict.return_value = {
        "question": "How do I authenticate?",
        "answer": "Use Bearer tokens.",
        "sources": [
            {
                "file_name": "authentication.md",
                "chunk_index": 0,
                "content_preview": "Use Bearer tokens for authentication.",
                "similarity_score": 0.85,
            }
        ],
        "model": "mistral",
        "num_chunks_retrieved": 1,
    }

    mock_pipeline = MagicMock()
    mock_pipeline.query.return_value = mock_response
    mock_pipeline.check_health.return_value = {
        "pipeline": "healthy",
        "ollama": "connected",
        "model": "mistral",
        "retriever_k": 3,
    }

    with patch("src.api.main.pipeline", mock_pipeline):
        from src.api.main import app
        client = TestClient(app)
        yield client, mock_pipeline


def test_root_returns_200():
    from src.api.main import app
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200


def test_root_contains_query_path():
    from src.api.main import app
    client = TestClient(app)
    data = client.get("/").json()
    assert "query" in data


def test_health_returns_200():
    from src.api.main import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200


def test_query_returns_200(client_with_pipeline):
    client, _ = client_with_pipeline
    response = client.post("/query", json={"question": "How do I authenticate?"})
    assert response.status_code == 200


def test_query_returns_answer(client_with_pipeline):
    client, _ = client_with_pipeline
    data = client.post(
        "/query", json={"question": "How do I authenticate?"}
    ).json()
    assert "answer" in data
    assert len(data["answer"]) > 0


def test_query_returns_sources(client_with_pipeline):
    client, _ = client_with_pipeline
    data = client.post(
        "/query", json={"question": "How do I authenticate?"}
    ).json()
    assert "sources" in data
    assert len(data["sources"]) > 0


def test_query_source_has_file_name(client_with_pipeline):
    client, _ = client_with_pipeline
    data = client.post(
        "/query", json={"question": "How do I authenticate?"}
    ).json()
    assert "file_name" in data["sources"][0]


def test_query_empty_question_returns_422():
    from src.api.main import app
    client = TestClient(app)
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422


def test_query_missing_question_returns_422():
    from src.api.main import app
    client = TestClient(app)
    response = client.post("/query", json={})
    assert response.status_code == 422


def test_query_calls_pipeline(client_with_pipeline):
    client, mock_pipeline = client_with_pipeline
    client.post("/query", json={"question": "How do I authenticate?"})
    mock_pipeline.query.assert_called_once_with("How do I authenticate?")
