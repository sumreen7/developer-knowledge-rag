# src/llm/ollama_client.py
# Wraps the Ollama LLM in a clean interface.
# Phase 7 (RAG pipeline) uses this to generate answers from retrieved context.

import logging
from typing import Optional

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

from src.config import settings

logger = logging.getLogger(__name__)


# This is the prompt template the LLM receives.
# {context} = the retrieved chunks from ChromaDB (injected by the RAG pipeline)
# {question} = the developer's original question
#
# The instructions are deliberately explicit:
# - "only use the provided context" prevents hallucination
# - "cite the source" ensures answers are traceable
# - "say you don't know" is better than a confident wrong answer
RAG_PROMPT_TEMPLATE = """You are a helpful developer documentation assistant.
Answer the developer's question using ONLY the information provided in the context below.
If the answer is not in the context, say "I don't have enough information in the documentation to answer this question."
Always cite which document your answer comes from.

Context:
{context}

Question: {question}

Answer:"""


class OllamaClient:
    """
    Manages the connection to a locally running Ollama LLM.

    Why wrap OllamaLLM in our own class?
    - Centralizes configuration (model name, temperature, timeout)
    - Makes it easy to swap models by changing .env
    - Gives us a single place to add retries, logging, or fallbacks
    - Makes testing easier — we can mock this class instead of Ollama
    """

    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        temperature: float = 0.1,
    ):
        self.model = model or settings.ollama_model
        self.base_url = base_url or settings.ollama_base_url

        # temperature controls creativity vs accuracy
        # 0.0 = fully deterministic (same answer every time)
        # 1.0 = creative/random
        # 0.1 = slightly flexible but mostly factual — ideal for docs QA
        self.temperature = temperature

        logger.info(f"Initializing Ollama client — model: {self.model}")
        logger.info(f"Ollama base URL: {self.base_url}")

        # OllamaLLM connects to the running Ollama server
        # It doesn't load the model here — connection is lazy
        self.llm = OllamaLLM(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
        )

        # PromptTemplate formats context + question into the final prompt string
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=RAG_PROMPT_TEMPLATE,
        )

        logger.info("Ollama client ready")

    def generate(self, context: str, question: str) -> str:
        """
        Generate an answer from context and a question.

        This is the main method called by the RAG pipeline in Phase 7.

        Args:
            context: formatted retrieved chunks (from retriever.format_retrieved_context)
            question: the developer's original question

        Returns:
            the LLM's answer as a plain string
        """
        if not question.strip():
            return "Please provide a question."

        if not context.strip():
            return "No relevant documentation found to answer your question."

        # Format the prompt — fills {context} and {question} placeholders
        prompt = self.prompt_template.format(
            context=context,
            question=question,
        )

        logger.info(f"Sending prompt to {self.model}...")
        logger.debug(f"Prompt length: {len(prompt)} chars")

        try:
            # .invoke() sends the prompt to Ollama and returns the response
            response = self.llm.invoke(prompt)
            logger.info("Response received from Ollama")
            return response.strip()

        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            raise RuntimeError(
                f"Failed to get response from Ollama. "
                f"Is Ollama running? Try: ollama serve\n"
                f"Error: {e}"
            ) from e

    def check_connection(self) -> bool:
        """
        Verify Ollama is running and the model is available.
        Called by the health check endpoint in Phase 8.

        Returns:
            True if Ollama responds, False otherwise
        """
        try:
            # Send a minimal test prompt
            response = self.llm.invoke("Say 'ok' and nothing else.")
            logger.info(f"Ollama health check passed — model: {self.model}")
            return True
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False

    def get_model_info(self) -> dict:
        """Returns current model configuration."""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
        }