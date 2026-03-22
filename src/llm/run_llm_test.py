# src/llm/run_llm_test.py
# Quick test — sends a real question to Ollama and prints the answer.
# Run with: python -m src.llm.run_llm_test

import logging
import sys

from src.llm.ollama_client import OllamaClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_llm_test():
    logger.info("Testing Ollama connection...")

    client = OllamaClient()

    # First check the connection
    if not client.check_connection():
        logger.error("Ollama is not running! Start it with: ollama serve")
        sys.exit(1)

    logger.info("Connection OK. Testing with sample context...")
    logger.info("=" * 50)

    # Simulate what the RAG pipeline will send
    sample_context = """[Source 1: authentication.md — chunk 0]
Authentication Guide

This API uses Bearer token authentication. All requests must include 
a valid token in the Authorization header.

To get a token, send a POST request to /auth/token with your credentials.
The response includes an access_token valid for 24 hours.

Include the token in every request:
Authorization: Bearer <your_access_token>

Error Codes:
401 Unauthorized: token missing or invalid
403 Forbidden: valid token but insufficient permissions"""

    question = "How do I authenticate with the API?"

    logger.info(f"Question: {question}")
    logger.info("Generating answer...")
    logger.info("=" * 50)

    answer = client.generate(
        context=sample_context,
        question=question,
    )

    logger.info("ANSWER FROM MISTRAL:")
    logger.info("=" * 50)
    print(f"\n{answer}\n")
    logger.info("=" * 50)
    logger.info(f"Model info: {client.get_model_info()}")


if __name__ == "__main__":
    run_llm_test()