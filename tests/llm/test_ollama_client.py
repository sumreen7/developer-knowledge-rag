import pytest
from unittest.mock import patch, MagicMock
from src.llm.ollama_client import OllamaClient, RAG_PROMPT_TEMPLATE


def test_client_initializes():
    client = OllamaClient(
        model="mistral",
        base_url="http://localhost:11434",
    )
    assert client.model == "mistral"
    assert client.base_url == "http://localhost:11434"
    assert client.temperature == 0.1


def test_prompt_template_has_required_variables():
    assert "context" in RAG_PROMPT_TEMPLATE
    assert "question" in RAG_PROMPT_TEMPLATE


def test_prompt_template_has_grounding_instruction():
    assert "only" in RAG_PROMPT_TEMPLATE.lower()


def test_generate_empty_question_returns_message():
    client = OllamaClient()
    result = client.generate(context="some context", question="")
    assert "Please provide" in result


def test_generate_empty_context_returns_message():
    client = OllamaClient()
    result = client.generate(context="", question="How do I auth?")
    assert "No relevant" in result


def test_generate_calls_ollama_with_context(mocker):
    mock_invoke = mocker.patch.object(
        OllamaClient,
        "__init__",
        lambda self, **kwargs: None
    )
    client = OllamaClient.__new__(OllamaClient)
    client.model = "mistral"
    client.base_url = "http://localhost:11434"
    client.temperature = 0.1

    from langchain_core.prompts import PromptTemplate
    client.prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=RAG_PROMPT_TEMPLATE,
    )

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "Use Bearer tokens for authentication."
    client.llm = mock_llm

    result = client.generate(
        context="Use Bearer tokens.",
        question="How do I authenticate?"
    )

    assert mock_llm.invoke.called
    assert isinstance(result, str)
    assert len(result) > 0


def test_get_model_info_returns_dict():
    client = OllamaClient(model="mistral")
    info = client.get_model_info()
    assert "model" in info
    assert "base_url" in info
    assert "temperature" in info
    assert info["model"] == "mistral"
