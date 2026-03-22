import pytest
from unittest.mock import MagicMock, patch
from src.rag.pipeline import RAGPipeline, RAGResponse, SourceDocument
from langchain_core.documents import Document


@pytest.fixture
def mock_pipeline():
    with patch("src.rag.pipeline.DocumentRetriever") as mock_retriever_cls, \
         patch("src.rag.pipeline.OllamaClient") as mock_llm_cls:

        mock_retriever = MagicMock()
        mock_retriever_cls.return_value = mock_retriever

        mock_retriever.retrieve_with_scores.return_value = [
            (
                Document(
                    page_content="Use Bearer tokens for authentication.",
                    metadata={
                        "file_name": "authentication.md",
                        "chunk_index": 0,
                        "total_chunks": 2,
                    }
                ),
                0.85,
            )
        ]
        mock_retriever.format_retrieved_context.return_value = (
            "[Source 1: authentication.md] Use Bearer tokens."
        )

        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm
        mock_llm.model = "mistral"
        mock_llm.generate.return_value = (
            "Use Bearer tokens. [Source: authentication.md]"
        )
        mock_llm.check_connection.return_value = True
        mock_llm.get_model_info.return_value = {
            "model": "mistral",
            "base_url": "http://localhost:11434",
            "temperature": 0.1,
        }

        pipeline = RAGPipeline(k=3)
        yield pipeline, mock_retriever, mock_llm


def test_pipeline_initializes(mock_pipeline):
    pipeline, _, _ = mock_pipeline
    assert pipeline.k == 3
    assert pipeline.retriever is not None
    assert pipeline.llm_client is not None


def test_query_returns_rag_response(mock_pipeline):
    pipeline, _, _ = mock_pipeline
    response = pipeline.query("How do I authenticate?")
    assert isinstance(response, RAGResponse)


def test_query_contains_answer(mock_pipeline):
    pipeline, _, _ = mock_pipeline
    response = pipeline.query("How do I authenticate?")
    assert len(response.answer) > 0


def test_query_contains_sources(mock_pipeline):
    pipeline, _, _ = mock_pipeline
    response = pipeline.query("How do I authenticate?")
    assert len(response.sources) > 0
    assert isinstance(response.sources[0], SourceDocument)


def test_query_source_has_file_name(mock_pipeline):
    pipeline, _, _ = mock_pipeline
    response = pipeline.query("How do I authenticate?")
    assert response.sources[0].file_name == "authentication.md"


def test_query_empty_question_returns_message(mock_pipeline):
    pipeline, _, _ = mock_pipeline
    response = pipeline.query("")
    assert "Please provide" in response.answer
    assert len(response.sources) == 0


def test_query_calls_retriever(mock_pipeline):
    pipeline, mock_retriever, _ = mock_pipeline
    pipeline.query("How do I authenticate?")
    mock_retriever.retrieve_with_scores.assert_called_once()


def test_query_calls_llm(mock_pipeline):
    pipeline, _, mock_llm = mock_pipeline
    pipeline.query("How do I authenticate?")
    mock_llm.generate.assert_called_once()


def test_rag_response_to_dict(mock_pipeline):
    pipeline, _, _ = mock_pipeline
    response = pipeline.query("How do I authenticate?")
    d = response.to_dict()
    assert "question" in d
    assert "answer" in d
    assert "sources" in d
    assert "model" in d
    assert "num_chunks_retrieved" in d


def test_check_health_returns_dict(mock_pipeline):
    pipeline, _, _ = mock_pipeline
    health = pipeline.check_health()
    assert "pipeline" in health
    assert "ollama" in health
    assert "model" in health
