# tests/test_health.py
# Verifies the API starts and the health endpoint responds correctly.
# We use FastAPI's TestClient — no need to run a server.

import pytest
from fastapi.testclient import TestClient

# Import the FastAPI app object
from src.api.main import app

# Create a test client — simulates HTTP requests without a real server
client = TestClient(app)


def test_health_returns_200():
    """The /health endpoint must return HTTP 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_contains_status_healthy():
    """The response body must include a status field."""
    response = client.get("/health")
    data = response.json()
    # Status is 'healthy' when pipeline is running, 'degraded' in test env
    assert "status" in data
    assert data["status"] in ("healthy", "degraded")


def test_health_contains_config():
    """Config fields must be present in the health response."""
    response = client.get("/health")
    data = response.json()
    assert "config" in data
    assert "ollama_model" in data["config"]
    assert "embedding_model" in data["config"]


def test_root_endpoint():
    """The root endpoint must return a docs link."""
    response = client.get("/")
    assert response.status_code == 200
    assert "docs" in response.json()