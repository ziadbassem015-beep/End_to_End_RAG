"""
tests/test_next_gen_retrieval.py
================================
Unit tests for the redesigned next-generation retrieval pipeline.
"""

from __future__ import annotations

import pytest
from typing import Any, Dict, List
from langchain_core.documents import Document

from rag_system.ingestion.chunker import DocumentChunker, check_front_matter
from rag_system.retrieval.reranker import RRFFusion
from rag_system.retrieval.query_expansion import QueryExpansion, HyDERetriever
from rag_system.retrieval.retriever import HybridRetriever, BM25Index
from rag_system.vectorstore.faiss_store import FAISSStore


# ─── Parent-Child Mapping Tests ──────────────────────────────────────────────

def test_parent_child_chunking():
    """Verify parent-child chunking splits text correctly and stores parent_id."""
    chunker = DocumentChunker(
        use_parent_child=True,
        parent_size=200,
        parent_overlap=50,
        child_size=50,
        child_overlap=10,
        min_chunk_length=10
    )

    doc = Document(
        page_content="This is page one of our test document. It contains some text that we want to chunk using parent-child strategies. The parent chunks are larger and the child chunks are smaller.",
        metadata={"source": "test_doc.pdf", "page": 1}
    )

    chunks = chunker.chunk([doc])
    
    assert len(chunks) > 0
    for chunk in chunks:
        # Every child chunk must have metadata specifying parent_id, child_id, page, source
        meta = chunk.metadata
        assert meta.parent_id is not None
        assert meta.child_id is not None
        assert meta.page == 1
        assert meta.source == "test_doc.pdf"
        assert chunk.child_content is not None
        assert chunk.content == doc.page_content  # Parent content is full text


# ─── RRF Fusion Tests ────────────────────────────────────────────────────────

def test_rrf_fusion_correctness():
    """Verify RRF fusion computes scores and ranks results correctly."""
    fusion = RRFFusion(rrf_k=60)

    # Mock candidate search lists
    sem_results = [
        {"chunk_id": "doc_A", "content": "Content A", "source": "test.pdf", "page": 1, "section": "", "score": 0.9},
        {"chunk_id": "doc_B", "content": "Content B", "source": "test.pdf", "page": 1, "section": "", "score": 0.8},
        {"chunk_id": "doc_C", "content": "Content C", "source": "test.pdf", "page": 1, "section": "", "score": 0.7},
    ]

    bm25_results = [
        {"chunk_id": "doc_B", "content": "Content B", "source": "test.pdf", "page": 1, "section": "", "score": 12.0},
        {"chunk_id": "doc_C", "content": "Content C", "source": "test.pdf", "page": 1, "section": "", "score": 8.0},
        {"chunk_id": "doc_A", "content": "Content A", "source": "test.pdf", "page": 1, "section": "", "score": 4.0},
    ]

    fused = fusion.fuse(sem_results, bm25_results, top_k=3, alpha=0.5)

    assert len(fused) == 3
    # Check that doc_B is ranked 1st since it was rank 2 in sem and rank 1 in bm25
    # score_B = 0.5 * (1 / (60 + 2)) + 0.5 * (1 / (60 + 1))
    # score_A = 0.5 * (1 / (60 + 1)) + 0.5 * (1 / (60 + 3))
    # score_B is larger than score_A
    assert fused[0]["chunk_id"] == "doc_B"
    assert fused[1]["chunk_id"] == "doc_A"
    assert fused[2]["chunk_id"] == "doc_C"


# ─── Query Expansion Tests ───────────────────────────────────────────────────

def test_query_expansion_caching():
    """Verify QueryExpansion caches query variants correctly."""
    expansion = QueryExpansion(generator=None)  # None generator forces fallback/offline mode
    query = "How does scale affect model size?"

    # Should fall back to list containing original query since generator is mock/None
    variants = expansion.expand_query(query, num_variants=4)
    assert len(variants) == 1
    assert variants[0] == query

    # Artificially inject into cache
    cached_variants = [query, "model size vs scale", "neural net parameters scaling"]
    expansion._cache[f"{query}||4"] = cached_variants

    # Fetch again to check cache hit
    variants_hit = expansion.expand_query(query, num_variants=4)
    assert len(variants_hit) == 3
    assert variants_hit[1] == "model size vs scale"


# ─── HyDE Retrieval Tests ────────────────────────────────────────────────────

def test_hyde_retriever_caching():
    """Verify HyDERetriever caches generated documents in memory."""
    hyde = HyDERetriever(generator=None)
    query = "What are the rules of overfitting?"

    doc = hyde.generate_hypothetical_document(query)
    assert doc == query  # Offline fallback returns original query

    # Inject cached document
    hypothetical_doc = "Overfitting occurs when a machine learning model fits noise in training data..."
    hyde._cache[query] = hypothetical_doc

    doc_hit = hyde.generate_hypothetical_document(query)
    assert doc_hit == hypothetical_doc


# ─── Metadata Filtering Tests ────────────────────────────────────────────────

def test_metadata_filtering_front_matter():
    """Verify check_front_matter detects table of contents and copyright pages."""
    # TOC detection
    toc_text = "Table of Contents\n1. Introduction............Page 1\n2. Background............Page 2"
    assert check_front_matter(toc_text, page=2) is True

    # Copyright detection on early page
    copyright_text = "Copyright © 2018 Andrew Ng. All Rights Reserved. ISBN 12345"
    assert check_front_matter(copyright_text, page=2) is True

    # Standard body content should not be front matter
    body_text = "Machine learning is the foundation of countless applications, including web search and email anti-spam."
    assert check_front_matter(body_text, page=12) is False

    # Standard body content on early page should not be front matter if no copyright/TOC words are present
    assert check_front_matter(body_text, page=2) is False


def test_metadata_filtering_retrieval():
    """Verify retriever filters out front matter chunks during search."""
    chunks = [
        {
            "chunk_id": "chunk_toc",
            "content": "Table of Contents Page 3",
            "source": "test.pdf",
            "page": 3,
            "section": "TOC",
            "embedding": [0.1] * 384,
            "is_front_matter": True
        },
        {
            "chunk_id": "chunk_body",
            "content": "Deep learning is a subset of machine learning.",
            "source": "test.pdf",
            "page": 12,
            "section": "Intro",
            "embedding": [0.9] * 384,
            "is_front_matter": False
        }
    ]

    store = FAISSStore(dimension=384)
    store.add_batch(chunks)
    bm25 = BM25Index(chunks)

    # Retriever with filtering enabled
    retriever_filtered = HybridRetriever(
        vector_store=store,
        bm25_index=bm25,
        model_name="BAAI/bge-small-en-v1.5",
        alpha=0.7,
        use_metadata_filtering=True
    )

    results = retriever_filtered.retrieve("deep learning", top_k=5)
    # The TOC chunk should be excluded
    assert len(results) == 1
    assert results[0].chunk_id == "chunk_body"

    # Retriever with filtering disabled
    retriever_unfiltered = HybridRetriever(
        vector_store=store,
        bm25_index=bm25,
        model_name="BAAI/bge-small-en-v1.5",
        alpha=0.7,
        use_metadata_filtering=False
    )
    results_unfiltered = retriever_unfiltered.retrieve("deep learning", top_k=5)
    assert len(results_unfiltered) == 2
