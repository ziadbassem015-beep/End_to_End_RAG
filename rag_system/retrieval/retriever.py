"""
retrieval/retriever.py
======================
Production hybrid retriever.
Uses abstract VectorStore interface for semantic search and BM25 for keyword search.
Supports linear fusion and optional reranking.
"""

from __future__ import annotations

import logging
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_system.vectorstore.base import VectorStore
from rag_system.vectorstore.faiss_store import FAISSStore
from rag_system.retrieval.reranker import Reranker, ScoreFusion, RRFFusion, get_reranker
from rag_system.retrieval.query_expansion import QueryExpansion, HyDERetriever
import os

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Standardized retrieval result returned by the retriever."""
    chunk_id: str
    rank: int
    score: float
    content: str
    source: str
    page: int
    section: str
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0
    child_content: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "rank": self.rank,
            "score": self.score,
            "content": self.content,
            "source": self.source,
            "page": self.page,
            "section": self.section,
            "semantic_score": self.semantic_score,
            "bm25_score": self.bm25_score,
            "rerank_score": self.rerank_score,
            "child_content": self.child_content,
        }


class BM25Index:
    """BM25 Index over chunk content."""

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "BM25 requires rank-bm25 package. Install it with pip install rank-bm25"
            )

        self._chunk_ids = [c["chunk_id"] for c in chunks]
        self._chunks = {c["chunk_id"]: c for c in chunks}

        tokenized = [self._tokenize(c["content"]) for c in chunks]
        self._bm25 = BM25Okapi(tokenized, k1=k1, b=b)
        logger.info("BM25Index: indexed %d chunks", len(chunks))

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search BM25 index and return results."""
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)

        k = min(top_k, len(self._chunk_ids))
        if k == 0:
            return []

        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            idx = int(idx)
            if scores[idx] <= 0:
                continue
            chunk_id = self._chunk_ids[idx]
            chunk = self._chunks[chunk_id]
            
            # Resolve metadata fields
            meta = chunk.get("metadata", {})
            source = chunk.get("source", meta.get("source", ""))
            page = int(chunk.get("page", meta.get("page", 1)))
            section = chunk.get("section", meta.get("section", ""))
            is_front = chunk.get("is_front_matter", meta.get("is_front_matter", False))

            results.append({
                "chunk_id": chunk_id,
                "content": chunk["content"],
                "source": source,
                "page": page,
                "section": section,
                "is_front_matter": is_front,
                "score": float(scores[idx]),
            })
        return results

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple tokenization logic."""
        text = text.lower()
        text = re.sub(r"[" + re.escape(string.punctuation) + r"]", " ", text)
        return [t for t in text.split() if len(t) > 1]


class HybridRetriever:
    """
    Modular Hybrid Retriever relying on VectorStore abstraction and BM25 index.
    Supports reciprocal rank fusion, parent-child mapping, query expansion, and reranking.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: Optional[BM25Index] = None,
        model_name: str = "BAAI/bge-small-en-v1.5",
        alpha: float = 0.7,
        similarity_threshold: float = 0.0,
        reranker: Optional[Reranker] = None,
        fusion_type: str = "rrf",
        rrf_k: int = 60,
        use_parent_child: bool = False,
        retrieval_pool_multiplier: int = 10,
        use_query_expansion: bool = False,
        query_variants: int = 4,
        use_hyde: bool = False,
        use_metadata_filtering: bool = True,
        # Allow old parameters for backward compatibility
        expander: Optional[Any] = None,
        expansion_method: str = "none",
    ) -> None:
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.model_name = model_name
        self.alpha = alpha
        self.similarity_threshold = similarity_threshold
        self.reranker = reranker
        self.fusion_type = fusion_type.lower()
        self.rrf_k = rrf_k
        self.use_parent_child = use_parent_child
        self.retrieval_pool_multiplier = retrieval_pool_multiplier
        self.use_query_expansion = use_query_expansion or (expansion_method == "multi_query" or expansion_method == "mq")
        self.query_variants = query_variants
        self.use_hyde = use_hyde or (expansion_method == "hyde")
        self.use_metadata_filtering = use_metadata_filtering

        self._fusion = ScoreFusion(alpha=alpha, fusion_type=self.fusion_type, rrf_k=rrf_k)
        self._rrf_fusion = RRFFusion(rrf_k=rrf_k)

        logger.info("Loading query encoder: %s", model_name)
        self._encoder = SentenceTransformer(model_name)

        # Initialize query expansion and HyDE modules using active env keys
        api_key = os.getenv("GITHUB_TOKEN") or os.getenv("OPENAI_API_KEY") or "mock"
        provider = "mock"
        if api_key != "mock":
            if api_key.startswith("ghp_") or api_key.startswith("github_pat_") or len(api_key) > 40:
                provider = "github"
            else:
                provider = "openai"

        self._llm_gen = None
        try:
            from rag_system.llm.generator import LLMGenerator
            self._llm_gen = LLMGenerator(api_key=api_key, provider=provider)
        except Exception as e:
            logger.warning("Could not initialize LLMGenerator for retriever expansions: %s", e)

        self.query_expander = QueryExpansion(self._llm_gen)
        self.hyde_retriever = HyDERetriever(self._llm_gen)

        logger.info("HybridRetriever initialized successfully. fusion_type=%s, rrf_k=%d, use_hyde=%s, use_query_expansion=%s", 
                    self.fusion_type, rrf_k, self.use_hyde, self.use_query_expansion)

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve chunks matching query.
        """
        # Determine candidate pool size
        pool_size = top_k * self.retrieval_pool_multiplier

        # Generate query variations for expansion
        queries_to_embed = [query]
        if self.use_query_expansion:
            queries_to_embed = self.query_expander.expand_query(query, num_variants=self.query_variants)

        # Generate HyDE document if enabled
        if self.use_hyde:
            hyde_doc = self.hyde_retriever.generate_hypothetical_document(query)
            if hyde_doc not in queries_to_embed:
                queries_to_embed.append(hyde_doc)

        # Run semantic retrieval over all query variations
        semantic_candidates_map = {}
        for q in queries_to_embed:
            query_vec = self._encoder.encode(
                q,
                normalize_embeddings=True,
                convert_to_numpy=True
            ).tolist()
            res = self.vector_store.search(query_vec, top_k=pool_size)
            for r in res:
                cid = r["chunk_id"]
                # Apply metadata filtering: exclude front matter if configured
                is_front = r.get("is_front_matter", r.get("metadata", {}).get("is_front_matter", False))
                if self.use_metadata_filtering and is_front:
                    continue

                if cid not in semantic_candidates_map or r.get("score", 0.0) > semantic_candidates_map[cid].get("score", 0.0):
                    semantic_candidates_map[cid] = r

        semantic_candidates = list(semantic_candidates_map.values())
        semantic_candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        # Run BM25 sparse search on original query
        bm25_candidates = []
        if self.bm25_index and self.alpha < 1.0:
            raw_bm25 = self.bm25_index.search(query, top_k=pool_size)
            for r in raw_bm25:
                is_front = r.get("is_front_matter", r.get("metadata", {}).get("is_front_matter", False))
                if self.use_metadata_filtering and is_front:
                    continue
                bm25_candidates.append(r)

        # Merge candidate lists
        if self.bm25_index and self.alpha < 1.0:
            if self.fusion_type == "rrf":
                fused_candidates = self._rrf_fusion.fuse(
                    semantic_results=semantic_candidates,
                    bm25_results=bm25_candidates,
                    top_k=pool_size,
                    alpha=self.alpha
                )
            else:
                fused_candidates = self._fusion.fuse(
                    semantic_results=semantic_candidates,
                    bm25_results=bm25_candidates,
                    top_k=pool_size,
                    normalize=True
                )
        else:
            fused_candidates = []
            for r in semantic_candidates:
                rc = dict(r)
                rc["semantic_score"] = r.get("score", 0.0)
                rc["bm25_score"] = 0.0
                fused_candidates.append(rc)

        # Rerank candidates if a reranker is active
        if self.reranker:
            fused_candidates = self.reranker.rerank(query, fused_candidates[:pool_size])

        # Apply similarity threshold
        if self.similarity_threshold > 0.0:
            fused_candidates = [
                c for c in fused_candidates if c.get("score", 0.0) >= self.similarity_threshold
            ]

        # Convert to final RetrievalResult list
        results = []
        for rank, c in enumerate(fused_candidates[:top_k], 1):
            # If child chunk metadata contains parent content, use it to populate content
            content_text = c.get("content", "")
            child_text = c.get("child_content")

            results.append(
                RetrievalResult(
                    chunk_id=c["chunk_id"],
                    rank=rank,
                    score=c.get("score", 0.0),
                    content=content_text,
                    source=c.get("source", ""),
                    page=int(c.get("page", 1)),
                    section=c.get("section", ""),
                    semantic_score=c.get("semantic_score", 0.0),
                    bm25_score=c.get("bm25_score", 0.0),
                    rerank_score=c.get("rerank_score", 0.0),
                    child_content=child_text,
                )
            )

        return results

    def retrieve_batch(
        self,
        queries: List[str],
        top_k: int = 10,
    ) -> List[List[RetrievalResult]]:
        """Retrieve for a batch of queries."""
        return [self.retrieve(q, top_k=top_k) for q in queries]

    @property
    def chunk_ids(self) -> List[str]:
        """Get all chunk IDs currently in vector store."""
        if hasattr(self.vector_store, "metadata"):
            return list(self.vector_store.metadata.keys())
        return []

