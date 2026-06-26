"""
tests/test_embedder.py
======================
Tests for the Embedder.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from rag_system.embeddings.embedder import Embedder


def make_chunks(n: int = 5) -> List[Dict[str, Any]]:
    """Make n fake chunk dicts."""
    return [
        {
            "chunk_id": f"chunk_{i:04d}",
            "content": f"This is sample text for chunk number {i}. "
                       f"It contains information about machine learning topic {i}.",
            "metadata": {"source": "test.pdf", "page": i, "section": f"Section {i}"},
        }
        for i in range(n)
    ]


class TestEmbedder:
    @pytest.fixture(scope="class")
    def embedder(self):
        """Single model instance shared across tests in this class."""
        return Embedder(
            model_name="BAAI/bge-small-en-v1.5",
            batch_size=8,
            normalize=True,
        )

    def test_output_count_matches_input(self, embedder):
        chunks = make_chunks(5)
        result = embedder.embed(chunks)
        assert len(result) == 5

    def test_embedding_dimensions_correct(self, embedder):
        """BGE-small produces 384-dimensional embeddings."""
        chunks = make_chunks(3)
        result = embedder.embed(chunks)
        for item in result:
            assert item["embedding_dimensions"] == 384
            assert len(item["embedding"]) == 384

    def test_chunk_ids_preserved_in_order(self, embedder):
        chunks = make_chunks(8)
        input_ids = [c["chunk_id"] for c in chunks]
        result = embedder.embed(chunks)
        output_ids = [r["chunk_id"] for r in result]
        assert input_ids == output_ids, "chunk_id order drifted after embedding!"

    def test_normalized_embeddings_have_unit_norm(self, embedder):
        chunks = make_chunks(4)
        result = embedder.embed(chunks)
        for item in result:
            vec = np.array(item["embedding"])
            norm = np.linalg.norm(vec)
            assert abs(norm - 1.0) < 1e-5

    def test_all_required_fields_present(self, embedder):
        chunks = make_chunks(2)
        result = embedder.embed(chunks)
        required = {"chunk_id", "content", "source", "page", "section", "metadata", "embedding",
                    "embedding_model", "embedding_dimensions"}
        for item in result:
            assert required.issubset(set(item.keys())), (
                f"Missing fields: {required - set(item.keys())}"
            )

    def test_different_texts_produce_different_embeddings(self, embedder):
        chunks = [
            {"chunk_id": "c1", "content": "Machine learning is great.", "metadata": {"source": "doc", "page": 1, "section": ""}},
            {"chunk_id": "c2", "content": "The weather today is sunny.", "metadata": {"source": "doc", "page": 1, "section": ""}},
        ]
        result = embedder.embed(chunks)
        e1 = np.array(result[0]["embedding"])
        e2 = np.array(result[1]["embedding"])
        cosine_sim = float(np.dot(e1, e2))
        assert cosine_sim < 0.90

    def test_similar_texts_produce_similar_embeddings(self, embedder):
        chunks = [
            {"chunk_id": "c1", "content": "Machine learning algorithms learn from data.", "metadata": {"source": "doc", "page": 1, "section": ""}},
            {"chunk_id": "c2", "content": "ML models are trained using datasets.", "metadata": {"source": "doc", "page": 1, "section": ""}},
        ]
        result = embedder.embed(chunks)
        e1 = np.array(result[0]["embedding"])
        e2 = np.array(result[1]["embedding"])
        cosine_sim = float(np.dot(e1, e2))
        assert cosine_sim > 0.6, f"Similar texts have low similarity: {cosine_sim:.4f}"


class TestEmbedderValidation:
    @pytest.fixture(scope="class")
    def embedder(self):
        return Embedder("BAAI/bge-small-en-v1.5", normalize=True)

    def test_missing_chunk_id_raises(self, embedder):
        chunks = [{"content": "No chunk_id here", "metadata": {}}]
        with pytest.raises(ValueError, match="chunk_id"):
            embedder.embed(chunks)

    def test_missing_content_raises(self, embedder):
        chunks = [{"chunk_id": "c1", "metadata": {}}]
        with pytest.raises(ValueError, match="content"):
            embedder.embed(chunks)

    def test_duplicate_chunk_ids_raise(self, embedder):
        chunks = [
            {"chunk_id": "same", "content": "Text one.", "metadata": {}},
            {"chunk_id": "same", "content": "Text two.", "metadata": {}},
        ]
        with pytest.raises(ValueError, match="Duplicate"):
            embedder.embed(chunks)


class TestEmbedderCache:
    def test_cache_produces_same_embeddings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            embedder_no_cache = Embedder(
                "BAAI/bge-small-en-v1.5",
                normalize=True,
                cache_dir=None,
            )
            embedder_with_cache = Embedder(
                "BAAI/bge-small-en-v1.5",
                normalize=True,
                cache_dir=tmpdir,
            )

            chunks = make_chunks(3)

            result_no_cache = embedder_no_cache.embed(chunks)
            result_cached_1 = embedder_with_cache.embed(chunks)
            result_cached_2 = embedder_with_cache.embed(chunks)

            for r1, r2 in zip(result_cached_1, result_cached_2):
                e1 = np.array(r1["embedding"])
                e2 = np.array(r2["embedding"])
                np.testing.assert_array_almost_equal(e1, e2, decimal=6)

    def test_save_and_load_round_trip(self):
        embedder = Embedder("BAAI/bge-small-en-v1.5", normalize=True)
        chunks = make_chunks(3)
        result = embedder.embed(chunks)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "embeddings.json"
            embedder.save(result, out)

            loaded = Embedder.load(out)
            assert len(loaded) == len(result)
            for orig, load in zip(result, loaded):
                assert orig["chunk_id"] == load["chunk_id"]
                np.testing.assert_array_almost_equal(
                    orig["embedding"], load["embedding"], decimal=6
                )
