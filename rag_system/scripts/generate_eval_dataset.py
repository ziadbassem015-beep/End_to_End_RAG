"""
Script to generate the corrected evaluation dataset.
Run this after `python cli.py chunk` has completed to get valid chunk IDs.

This script:
1. Loads the generated chunks
2. For each eval query, finds the top-5 semantically similar chunks
3. Writes rag_system/eval/eval_dataset.json

Usage:
    uv run python rag_system/scripts/generate_eval_dataset.py
    OR
    python cli.py validate --correct --eval-dataset rag_system/eval/eval_queries_seed.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


SEED_QUERIES = [
    {
        "query_id": "q_001",
        "query": "What is machine learning and why does it matter?",
    },
    {
        "query_id": "q_002",
        "query": "How should I use this book to help my team?",
    },
    {
        "query_id": "q_003",
        "query": "What are the prerequisites and notation used in this book?",
    },
    {
        "query_id": "q_004",
        "query": "How does scale affect machine learning performance?",
    },
    {
        "query_id": "q_005",
        "query": "Who is Andrew Ng and what is Machine Learning Yearning?",
    },
    {
        "query_id": "q_006",
        "query": "Why is machine learning strategy important for AI teams?",
    },
    {
        "query_id": "q_007",
        "query": "What are best practices for deep learning projects?",
    },
    {
        "query_id": "q_008",
        "query": "How does dataset size affect model accuracy?",
    },
    {
        "query_id": "q_009",
        "query": "How do you set up a train, dev and test set?",
    },
    {
        "query_id": "q_010",
        "query": "How to build AI systems with machine learning components?",
    },
    {
        "query_id": "q_011",
        "query": "What is error analysis and how do you do it?",
    },
    {
        "query_id": "q_012",
        "query": "How do you diagnose bias and variance in ML models?",
    },
    {
        "query_id": "q_013",
        "query": "What is end-to-end learning in deep learning?",
    },
    {
        "query_id": "q_014",
        "query": "How do you handle mismatched train and test distributions?",
    },
    {
        "query_id": "q_015",
        "query": "What is transfer learning and when should you use it?",
    },
]


def generate_dataset(
    embeddings_path: str = "rag_system/output/embeddings.json",
    output_path: str = "rag_system/eval/eval_dataset.json",
    top_k: int = 5,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> None:
    from rag_system.evaluation.validator import EvalDatasetValidator

    # Write seed queries to temp file
    seed_path = Path("rag_system/eval/eval_queries_seed.json")
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    seed_path.write_text(
        json.dumps(
            [{"query_id": q["query_id"], "query": q["query"], "relevant_chunk_ids": []}
             for q in SEED_QUERIES],
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Loading validator with embeddings from {embeddings_path}...")
    validator = EvalDatasetValidator(
        embeddings_path=embeddings_path,
        model_name=model_name,
        suggestion_top_k=top_k,
    )

    print(f"Generating corrected eval dataset with top_k={top_k}...")
    out = validator.generate_corrected_dataset(
        eval_dataset_path=seed_path,
        output_path=output_path,
        top_k=top_k,
    )
    print(f"\nEval dataset saved -> {out}")

    # Show a preview
    data = json.loads(Path(output_path).read_text(encoding="utf-8"))
    for item in data[:3]:
        print(f"\n  [{item['query_id']}] {item['query']!r}")
        print(f"  relevant_chunk_ids: {item['relevant_chunk_ids']}")


if __name__ == "__main__":
    generate_dataset()
