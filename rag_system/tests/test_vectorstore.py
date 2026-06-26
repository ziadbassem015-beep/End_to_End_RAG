"""
tests/test_vectorstore.py
==========================
Unit tests for the VectorStore interface and FAISSStore implementation.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List
import numpy as np
import pytest

from rag_system.vectorstore.factory import VectorStoreFactory
from rag_system.vectorstore.faiss_store import FAISSStore


def make_vector_records(n: int = 5, dim: int = 8) -> List[Dict[str, Any]]:
    """Create mock vector records for testing."""
    rng = np.random.default_rng(seed=42)
    records = []
    for i in range(n):
        vec = rng.random(dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        records.append({
            "chunk_id": f"chunk_{i:04d}",
            "content": f"Sample chunk content for chunk {i}",
            "source": "test_doc.pdf",
            "page": i + 1,
            "section": f"Section {i}",
            "embedding": vec.tolist()
        })
    return records


class TestFAISSStore:
    def test_factory_returns_faiss(self):
        store = VectorStoreFactory.get_vector_store("faiss", dimension=8)
        assert isinstance(store, FAISSStore)
        assert store.dimension == 8

    def test_add_and_count(self):
        store = FAISSStore(dimension=8)
        records = make_vector_records(3, dim=8)
        
        # Test empty state
        assert store.count() == 0
        
        # Add single
        store.add(records[0])
        assert store.count() == 1
        
        # Add remaining batch
        store.add_batch(records[1:])
        assert store.count() == 3

    def test_dimension_mismatch_raises(self):
        store = FAISSStore(dimension=8)
        record_mismatch = make_vector_records(1, dim=16)[0]
        
        # If empty, FAISSStore auto-adjusts to the first added vector's dimension
        store.add(record_mismatch)
        assert store.dimension == 16
        
        # Subsequent mismatches should raise ValueError
        record_another = make_vector_records(1, dim=8)[0]
        with pytest.raises(ValueError, match="Dimension mismatch"):
            store.add(record_another)

    def test_search_similarity(self):
        store = FAISSStore(dimension=8)
        records = make_vector_records(5, dim=8)
        store.add_batch(records)

        # Search with the first vector as query
        query_vec = records[0]["embedding"]
        results = store.search(query_vec, top_k=3)
        
        assert len(results) == 3
        # Closest match should be records[0] (score ~1.0 since it's identical and L2 normalized)
        assert results[0]["chunk_id"] == "chunk_0000"
        assert abs(results[0]["score"] - 1.0) < 1e-4

    def test_delete(self):
        store = FAISSStore(dimension=8)
        records = make_vector_records(3, dim=8)
        store.add_batch(records)
        
        assert store.count() == 3
        
        # Delete existing
        success = store.delete("chunk_0001")
        assert success is True
        assert store.count() == 2
        
        # Check that it's no longer searchable
        results = store.search(records[1]["embedding"], top_k=5)
        for r in results:
            assert r["chunk_id"] != "chunk_0001"
            
        # Delete non-existent
        success_fail = store.delete("chunk_non_existent")
        assert success_fail is False

    def test_update(self):
        store = FAISSStore(dimension=8)
        records = make_vector_records(2, dim=8)
        store.add_batch(records)
        
        # Update existing record content
        updated_record = dict(records[0])
        updated_record["content"] = "This content has been updated."
        
        success = store.update("chunk_0000", updated_record)
        assert success is True
        assert store.count() == 2
        
        # Verify update was recorded in metadata
        assert store.metadata["chunk_0000"]["content"] == "This content has been updated."
        
        # Update non-existent
        success_fail = store.update("chunk_non_existent", updated_record)
        assert success_fail is False

    def test_save_and_load_roundtrip(self):
        store = FAISSStore(dimension=8)
        records = make_vector_records(4, dim=8)
        store.add_batch(records)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store.save(tmpdir)
            
            # Index file and metadata JSON should exist
            assert os.path.exists(os.path.join(tmpdir, "index.faiss"))
            assert os.path.exists(os.path.join(tmpdir, "metadata.json"))
            
            # Load into a new store
            loaded_store = FAISSStore(dimension=8)
            loaded_store.load(tmpdir)
            
            assert loaded_store.count() == 4
            assert loaded_store.dimension == 8
            assert loaded_store.metadata["chunk_0002"]["content"] == records[2]["content"]
            
            # Verify searching behaves identical
            results = loaded_store.search(records[1]["embedding"], top_k=2)
            assert results[0]["chunk_id"] == "chunk_0001"

    def test_health_check(self):
        store = FAISSStore(dimension=8)
        assert store.health_check() is True
        
        records = make_vector_records(3, dim=8)
        store.add_batch(records)
        assert store.health_check() is True
        
        # Corrupt count metadata intentionally to simulate failure
        store.metadata["rogue_chunk"] = {"dummy": "data"}
        assert store.health_check() is False
