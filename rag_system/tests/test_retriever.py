"""
tests/test_retriever.py
=======================
Tests for the hybrid retriever and score fusion.
"""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
import pytest

from rag_system.vectorstore.faiss_store import FAISSStore
from rag_system.retrieval.retriever import BM25Index, HybridRetriever, RetrievalResult
from rag_system.retrieval.reranker import ScoreFusion, CrossEncoderReranker, BGEReranker


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_embedded_chunks(n: int = 10, dim: int = 8) -> List[Dict[str, Any]]:
    """Create fake embedded chunks with deterministic random embeddings."""
    rng = np.random.default_rng(seed=42)

    # Use realistic text that contains query terms for BM25 to work
    TEXTS = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Deep learning uses neural networks with multiple layers to model complex patterns in data.",
        "Natural language processing allows machines to understand and generate human language.",
        "Data science combines statistics, machine learning, and domain expertise to extract insights.",
        "Supervised learning trains models on labeled examples to make predictions on new data.",
        "Unsupervised learning discovers hidden structure in unlabeled datasets automatically.",
        "Reinforcement learning agents learn by interacting with an environment and receiving rewards.",
        "Transfer learning reuses knowledge from one task to improve performance on another task.",
        "Feature engineering involves transforming raw data into informative features for machine learning.",
        "Model evaluation metrics include precision, recall, F1-score, and area under the ROC curve.",
    ]

    chunks = []
    for i in range(min(n, len(TEXTS))):
        vec = rng.random(dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        chunks.append({
            "chunk_id": f"chunk_{i:04d}",
            "content": TEXTS[i % len(TEXTS)],
            "source": "test.pdf",
            "page": i + 1,
            "section": f"Section {i}",
            "embedding": vec.tolist(),
        })
    return chunks


# ─── BM25Index Tests ─────────────────────────────────────────────────────────

class TestBM25Index:
    @pytest.fixture
    def index(self):
        return BM25Index(make_embedded_chunks(10, dim=8))

    def test_search_returns_results(self, index):
        results = index.search("machine learning", top_k=5)
        assert len(results) > 0

    def test_search_chunk_ids_in_corpus(self, index):
        chunks = make_embedded_chunks(10, dim=8)
        all_ids = {c["chunk_id"] for c in chunks}
        results = index.search("data science", top_k=5)
        for r in results:
            assert r["chunk_id"] in all_ids

    def test_search_relevant_term_scores_high(self, index):
        results = index.search("machine learning", top_k=10)
        assert len(results) > 0
        assert all(r["score"] > 0 for r in results)

    def test_zero_score_results_filtered(self, index):
        results = index.search("xyzzy-nonexistent-term", top_k=5)
        assert len(results) == 0


# ─── ScoreFusion Tests ───────────────────────────────────────────────────────

class TestScoreFusion:
    def _make_result(self, chunk_id: str, sem: float = 0.0, bm25: float = 0.0) -> dict:
        return {
            "chunk_id": chunk_id,
            "content": f"Content for {chunk_id}",
            "source": "test.pdf",
            "page": 1,
            "section": "",
            "score": sem if sem > 0.0 else bm25,
            "semantic_score": sem,
            "bm25_score": bm25,
        }

    def test_pure_semantic_alpha_1(self):
        fusion = ScoreFusion(alpha=1.0)
        sem = [
            self._make_result("a", sem=0.9),
            self._make_result("b", sem=0.5),
            self._make_result("c", sem=0.3),
        ]
        results = fusion.fuse(sem, [], top_k=3)
        ids = [r["chunk_id"] for r in results]
        assert ids[0] == "a"

    def test_pure_bm25_alpha_0(self):
        fusion = ScoreFusion(alpha=0.0)
        bm25 = [
            self._make_result("x", bm25=10.0),
            self._make_result("y", bm25=6.0),
            self._make_result("z", bm25=2.0),
        ]
        results = fusion.fuse([], bm25, top_k=3)
        assert results[0]["chunk_id"] == "x"

    def test_top_k_respected(self):
        fusion = ScoreFusion(alpha=0.7)
        sem = [self._make_result(f"s{i}", sem=float(i)) for i in range(20)]
        results = fusion.fuse(sem, [], top_k=5)
        assert len(results) <= 5

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            ScoreFusion(alpha=1.5)
        with pytest.raises(ValueError):
            ScoreFusion(alpha=-0.1)

    def test_combines_unique_chunks(self):
        fusion = ScoreFusion(alpha=0.5)
        sem = [self._make_result("shared", sem=0.8), self._make_result("sem_only", sem=0.5)]
        bm25 = [self._make_result("shared", bm25=5.0), self._make_result("bm25_only", bm25=8.0)]
        results = fusion.fuse(sem, bm25, top_k=10)
        ids = {r["chunk_id"] for r in results}
        assert "shared" in ids
        assert "sem_only" in ids
        assert "bm25_only" in ids


# ─── HybridRetriever Tests ───────────────────────────────────────────────────

class TestHybridRetriever:
    @pytest.fixture
    def retriever(self):
        chunks = make_embedded_chunks(10, dim=384)  # 384 matches encoder default dimension
        store = FAISSStore(dimension=384)
        store.add_batch(chunks)
        bm25 = BM25Index(chunks)
        return HybridRetriever(
            vector_store=store,
            bm25_index=bm25,
            model_name="BAAI/bge-small-en-v1.5",
            alpha=0.7
        )

    def test_retrieve_returns_results(self, retriever):
        results = retriever.retrieve("machine learning", top_k=5)
        assert len(results) > 0
        assert isinstance(results[0], RetrievalResult)
        assert results[0].chunk_id.startswith("chunk_")
        assert results[0].rank == 1

    def test_retrieve_batch(self, retriever):
        queries = ["deep learning", "natural language"]
        results = retriever.retrieve_batch(queries, top_k=3)
        assert len(results) == 2
        assert len(results[0]) > 0
