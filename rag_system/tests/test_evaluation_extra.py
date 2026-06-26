"""
tests/test_evaluation_extra.py
==============================
Tests for PDFLoader, EvalDatasetValidator, RAGEvaluator, ReportGenerator, and RAGBenchmarker.
Uses mocking to execute without downloading real models.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from langchain_core.documents import Document

from rag_system.ingestion.loader import PDFLoader
from rag_system.evaluation.reports import ReportGenerator
from rag_system.evaluation.evaluator import RAGEvaluator, EvaluationResults, AggregatedMetrics, QueryResult
from rag_system.evaluation.validator import EvalDatasetValidator
from rag_system.benchmark.benchmarker import RAGBenchmarker


# ─── Mock SentenceTransformer ───────────────────────────────────────────────

class MockSentenceTransformer:
    def __init__(self, model_name: str = None) -> None:
        self.model_name = model_name

    def encode(self, texts: Any, **kwargs: Any) -> Any:
        if isinstance(texts, str):
            input_list = [texts]
            is_single = True
        else:
            input_list = list(texts)
            is_single = False

        embeddings = []
        for text in input_list:
            # Deterministic pseudo-random generation based on text hash
            val = sum(ord(c) for c in text) % 256
            rng = np.random.default_rng(seed=val)
            vec = rng.random(384).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            embeddings.append(vec)

        arr = np.vstack(embeddings)
        
        convert_to_numpy = kwargs.get("convert_to_numpy", True)
        if is_single and isinstance(texts, str):
            return arr[0] if convert_to_numpy else arr[0].tolist()
        return arr if convert_to_numpy else arr.tolist()


# ─── Test PDFLoader ─────────────────────────────────────────────────────────

def test_pdf_loader_standard():
    with tempfile.TemporaryDirectory() as tmpdir:
        dummy_pdf = Path(tmpdir) / "dummy.pdf"
        dummy_pdf.touch()

        with patch("rag_system.ingestion.loader.PyPDFLoader") as mock_class:
            mock_loader_instance = mock_class.return_value
            mock_loader_instance.load.return_value = [
                Document(page_content="Page one content. Very long sentence.", metadata={"page": 0}),
                Document(page_content="Page two content. Another long sentence.", metadata={"page": 1}),
            ]

            loader = PDFLoader(dummy_pdf, ocr_fallback=False)
            docs = loader.load()

            assert len(docs) == 2
            assert docs[0].metadata["page_number"] == 1
            assert docs[0].metadata["source"] == "dummy.pdf"
            assert docs[1].metadata["page_number"] == 2
            assert docs[0].metadata["loader"] == "PyPDFLoader"


def test_pdf_loader_needs_ocr_detection():
    with tempfile.TemporaryDirectory() as tmpdir:
        dummy_pdf = Path(tmpdir) / "dummy.pdf"
        dummy_pdf.touch()

        with patch("rag_system.ingestion.loader.PyPDFLoader") as mock_class:
            mock_loader_instance = mock_class.return_value
            mock_loader_instance.load.return_value = [
                Document(page_content="Short", metadata={"page": 0}),
                Document(page_content="Short", metadata={"page": 1}),
            ]

            loader = PDFLoader(dummy_pdf, ocr_fallback=True)
            assert loader._needs_ocr(mock_loader_instance.load.return_value) is True


# ─── Test ReportGenerator ───────────────────────────────────────────────────

def test_report_generator():
    agg = AggregatedMetrics(
        avg_precision_at_k=0.8,
        avg_recall_at_k=0.7,
        avg_hit_rate_at_k=0.9,
        avg_mrr=0.85,
        avg_map=0.75,
        avg_ndcg=0.82,
        median_precision_at_k=0.8,
        median_recall_at_k=0.7,
        median_ndcg=0.82,
        std_precision_at_k=0.1,
        std_recall_at_k=0.1,
        std_ndcg=0.1
    )

    per_query = [
        QueryResult(
            query_id="q01",
            query="test query",
            retrieved_ids=["chunk_001"],
            relevant_ids=["chunk_001"],
            precision_at_k=1.0,
            recall_at_k=1.0,
            hit_rate_at_k=1.0,
            mrr=1.0,
            average_precision=1.0,
            ndcg_at_k=1.0,
            retrieved_count=1,
            relevant_count=1,
            relevant_retrieved_count=1,
            missed_ids=[],
            retrieval_time_ms=5.2
        )
    ]

    results = EvaluationResults(
        total_queries=1,
        top_k=10,
        model_name="test-model",
        alpha=0.7,
        aggregated=agg,
        per_query=per_query,
        best_queries=per_query,
        worst_queries=per_query,
        failing_queries=[]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        report_gen = ReportGenerator(output_dir=tmpdir)
        paths = report_gen.generate_all(results)

        assert Path(paths["json"]).exists()
        assert Path(paths["markdown"]).exists()

        with open(paths["json"], encoding="utf-8") as f:
            data = json.load(f)
            assert data["metadata"]["model"] == "test-model"
            assert data["summary"]["avg_map"] == 0.75


# ─── Test EvalDatasetValidator ──────────────────────────────────────────────

@patch("sentence_transformers.SentenceTransformer", MockSentenceTransformer)
def test_eval_dataset_validator():
    rng = np.random.default_rng(42)
    mock_embeddings = [
        {
            "chunk_id": "MLY_P001_C001",
            "content": "Copyright deeplearning.ai. All rights reserved.",
            "source": "mly.pdf",
            "page": 1,
            "section": "Copyright",
            "embedding": rng.random(384).tolist()
        },
        {
            "chunk_id": "MLY_P002_C001",
            "content": "Table of Contents. Chapter 1 is Machine Learning.",
            "source": "mly.pdf",
            "page": 2,
            "section": "TOC",
            "embedding": rng.random(384).tolist()
        },
        {
            "chunk_id": "MLY_P012_C001",
            "content": "Deep learning models are trained on large scale datasets. Machine learning works.",
            "source": "mly.pdf",
            "page": 12,
            "section": "Content",
            "embedding": rng.random(384).tolist()
        }
    ]

    mock_eval_dataset = [
        {
            "query_id": "q1",
            "query": "what is machine learning",
            "relevant_chunk_ids": ["MLY_P002_C001", "MLY_P012_C001"]
        },
        {
            "query_id": "q2",
            "query": "something else",
            "relevant_chunk_ids": ["MLY_P999_C999"]
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        emb_path = Path(tmpdir) / "embeddings.json"
        emb_path.write_text(json.dumps(mock_embeddings))

        eval_path = Path(tmpdir) / "eval.json"
        eval_path.write_text(json.dumps(mock_eval_dataset))

        validator = EvalDatasetValidator(embeddings_path=emb_path, low_similarity_threshold=0.1)
        report = validator.validate(eval_path)

        assert report.queries_with_irrelevant_chunks >= 1
        q1_val = next(q for q in report.per_query if q.query_id == "q1")
        assert len(q1_val.irrelevant_chunks) >= 1
        assert "TOC" in q1_val.irrelevant_chunks[0].reason

        assert report.queries_with_missing_ids == 1
        q2_val = next(q for q in report.per_query if q.query_id == "q2")
        assert "MLY_P999_C999" in q2_val.missing_chunk_ids

        corrected_path = Path(tmpdir) / "corrected_eval.json"
        validator.generate_corrected_dataset(eval_path, corrected_path, top_k=1)
        
        corrected_data = json.loads(corrected_path.read_text(encoding="utf-8"))
        assert len(corrected_data) == 2
        assert "MLY_P012_C001" in corrected_data[0]["relevant_chunk_ids"]


# ─── Test RAGEvaluator ──────────────────────────────────────────────────────

@patch("sentence_transformers.SentenceTransformer", MockSentenceTransformer)
def test_rag_evaluator():
    rng = np.random.default_rng(42)
    mock_embeddings = [
        {
            "chunk_id": "MLY_P012_C001",
            "content": "Deep learning models are trained on large scale datasets.",
            "source": "mly.pdf",
            "page": 12,
            "section": "Content",
            "embedding": rng.random(384).tolist()
        }
    ]

    mock_eval_dataset = [
        {
            "query_id": "q1",
            "query": "what is deep learning scale",
            "relevant_chunk_ids": ["MLY_P012_C001"]
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        emb_path = Path(tmpdir) / "embeddings.json"
        emb_path.write_text(json.dumps(mock_embeddings))

        eval_path = Path(tmpdir) / "eval.json"
        eval_path.write_text(json.dumps(mock_eval_dataset))

        evaluator = RAGEvaluator(
            vector_store_path=emb_path,
            eval_dataset_path=eval_path,
            top_k=1,
            hybrid=True
        )
        results = evaluator.evaluate()

        assert results.total_queries == 1
        assert results.aggregated.avg_hit_rate_at_k == 1.0


# ─── Test RAGBenchmarker ────────────────────────────────────────────────────

@patch("rag_system.benchmark.benchmarker.PDFLoader")
@patch("sentence_transformers.SentenceTransformer", MockSentenceTransformer)
def test_rag_benchmarker(mock_loader_class):
    mock_loader_instance = mock_loader_class.return_value
    mock_loader_instance.load.return_value = [
        Document(page_content="Deep learning and machine learning systems learn from data.", metadata={"source": "mly.pdf", "page": 1})
    ]

    mock_eval = [
        {
            "query_id": "q1",
            "query": "machine learning",
            "relevant_chunk_ids": ["MLY_P001_C001"]
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        eval_path = Path(tmpdir) / "eval.json"
        eval_path.write_text(json.dumps(mock_eval))

        benchmarker = RAGBenchmarker(
            pdf_path="dummy.pdf",
            eval_dataset_path=eval_path,
            output_dir=tmpdir,
            cache_dir=tmpdir
        )
        
        benchmarker._pages = mock_loader_instance.load()

        results = benchmarker.run_all()
        assert "chunking" in results
        assert "fusion" in results
        assert "reranker" in results
        assert "expansion" in results
        assert "hyde" in results

        report_file = Path(tmpdir) / "benchmark_report.md"
        assert report_file.exists()
