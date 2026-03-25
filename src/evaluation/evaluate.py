# src/evaluation/evaluate.py
# Evaluates the RAG pipeline using RAGAS metrics.
# Runs a test set through the full pipeline and scores:
#   - Faithfulness: does the answer stay within the retrieved docs?
#   - Answer relevancy: does the answer address the question?
#   - Context precision: were the right chunks retrieved?
#
# Run with: python -m src.evaluation.evaluate

import logging
import json
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

from src.rag.pipeline import RAGPipeline
from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Test set ──────────────────────────────────────────────────────────────────
# These are the questions + expected answers we evaluate against.
# ground_truth is what a correct answer should contain.
# Add more test cases as you add more documentation.

TEST_CASES = [
    {
        "question": "How do I authenticate with the API?",
        "ground_truth": (
            "Use Bearer token authentication. Send a POST request to "
            "/auth/token with your credentials to receive an access_token. "
            "Include the token in every request as: Authorization: Bearer <token>. "
            "Tokens are valid for 24 hours."
        ),
    },
    {
        "question": "What does a 401 error mean?",
        "ground_truth": (
            "A 401 Unauthorized error means your token is missing or invalid. "
            "You need to get a new token by sending a POST request to /auth/token."
        ),
    },
    {
        "question": "What is the difference between a 401 and 403 error?",
        "ground_truth": (
            "A 401 Unauthorized error means the token is missing or invalid. "
            "A 403 Forbidden error means the token is valid but the user "
            "does not have sufficient permissions for the requested action."
        ),
    },
    {
        "question": "How do I get started with the API?",
        "ground_truth": (
            "Create an account, navigate to API Keys in your dashboard, "
            "generate a new API key, install the SDK with pip install example-sdk, "
            "and initialize the client with your API key."
        ),
    },
    {
        "question": "What are the API rate limits?",
        "ground_truth": (
            "The free tier allows 1000 requests per hour. "
            "The paid plan allows 10000 requests per hour."
        ),
    },
    {
        "question": "How do I create a new user?",
        "ground_truth": (
            "Send a POST request to /users endpoint to create a new user account."
        ),
    },
    {
        "question": "What happens if I exceed the rate limit?",
        "ground_truth": (
            "You will receive a 429 Too Many Requests error. "
            "Wait for the rate limit window to reset or upgrade your plan."
        ),
    },
]


def run_evaluation():
    logger.info("=" * 60)
    logger.info("RAGAS EVALUATION — Developer Docs AI Assistant")
    logger.info("=" * 60)
    logger.info(f"Test cases: {len(TEST_CASES)}")
    logger.info(f"Model: {settings.ollama_model}")

    # Step 1 — Initialize the RAG pipeline
    logger.info("\nInitializing RAG pipeline...")
    pipeline = RAGPipeline(k=3)

    # Step 2 — Run each test case through the pipeline
    logger.info("\nRunning test cases through pipeline...")
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, case in enumerate(TEST_CASES, 1):
        logger.info(f"  [{i}/{len(TEST_CASES)}] {case['question'][:60]}...")
        try:
            response = pipeline.query(case["question"])

            questions.append(case["question"])
            answers.append(response.answer)
            # RAGAS expects contexts as a list of strings per question
            contexts.append([s.content_preview for s in response.sources])
            ground_truths.append(case["ground_truth"])

            logger.info(f"         → {len(response.sources)} sources retrieved")

        except Exception as e:
            logger.error(f"  Failed: {e}")
            # Add placeholder so dataset stays aligned
            questions.append(case["question"])
            answers.append("Error generating answer")
            contexts.append([""])
            ground_truths.append(case["ground_truth"])

    # Step 3 — Build RAGAS dataset
    logger.info("\nBuilding evaluation dataset...")
    eval_dataset = Dataset.from_dict({
        "question":    questions,
        "answer":      answers,
        "contexts":    contexts,
        "ground_truth": ground_truths,
    })

    # Step 4 — Configure RAGAS to use local Ollama + HuggingFace
    # We wrap our existing models so RAGAS uses them as judges
    # No OpenAI API key needed
    logger.info("Configuring local LLM judge (Mistral via Ollama)...")

    ollama_llm = OllamaLLM(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.1,
    )
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    ragas_llm = LangchainLLMWrapper(ollama_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    # Step 5 — Run RAGAS evaluation
    logger.info("\nRunning RAGAS evaluation (this takes a few minutes)...")
    logger.info("Ollama is judging each answer — be patient.\n")

    metrics = [faithfulness, answer_relevancy, context_precision]

    # Configure each metric to use our local models
    for metric in metrics:
        metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_embeddings

    results = evaluate(
    dataset=eval_dataset,
    metrics=metrics,
    raise_exceptions=False,
    )

    # Step 6 — Print report
    scores = results.to_pandas()

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)

    # Per-question breakdown
    logger.info("\nPer-question scores:")
    for i, row in scores.iterrows():
        logger.info(f"\n  Q{i+1}: {questions[i][:55]}...")
        logger.info(f"       Faithfulness:     {row.get('faithfulness', 'N/A'):.3f}")
        logger.info(f"       Answer relevancy: {row.get('answer_relevancy', 'N/A'):.3f}")
        logger.info(f"       Context precision:{row.get('context_precision', 'N/A'):.3f}")

    # Overall averages
    logger.info("\n" + "-" * 40)
    logger.info("OVERALL AVERAGES:")
    faith_avg = scores["faithfulness"].mean()
    rel_avg   = scores["answer_relevancy"].mean()
    prec_avg  = scores["context_precision"].mean()

    logger.info(f"  Faithfulness:      {faith_avg:.3f}  {'✓' if faith_avg > 0.7 else '✗ needs improvement'}")
    logger.info(f"  Answer relevancy:  {rel_avg:.3f}  {'✓' if rel_avg > 0.7 else '✗ needs improvement'}")
    logger.info(f"  Context precision: {prec_avg:.3f}  {'✓' if prec_avg > 0.7 else '✗ needs improvement'}")
    logger.info("-" * 40)

    overall = (faith_avg + rel_avg + prec_avg) / 3
    logger.info(f"  Overall score:     {overall:.3f}")
    logger.info("=" * 60)

    # Step 7 — Save results to JSON
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"eval_{timestamp}.json"

    report = {
        "timestamp": timestamp,
        "model": settings.ollama_model,
        "embedding_model": settings.embedding_model,
        "num_test_cases": len(TEST_CASES),
        "scores": {
            "faithfulness":      round(faith_avg, 4),
            "answer_relevancy":  round(rel_avg, 4),
            "context_precision": round(prec_avg, 4),
            "overall":           round(overall, 4),
        },
        "per_question": [
            {
                "question": questions[i],
                "faithfulness":      round(scores["faithfulness"].iloc[i], 4),
                "answer_relevancy":  round(scores["answer_relevancy"].iloc[i], 4),
                "context_precision": round(scores["context_precision"].iloc[i], 4),
            }
            for i in range(len(questions))
        ],
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")
    logger.info("Run again after changes to track improvement over time.")

    return report


if __name__ == "__main__":
    run_evaluation()