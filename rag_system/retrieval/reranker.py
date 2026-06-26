"""
retrieval/reranker.py
=====================
Abstract Reranker interface and concrete implementations:
- CrossEncoderReranker (MiniLM-L-6)
- BGEReranker (bge-reranker-base)
"""

from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class Reranker(ABC):
    """
    Abstract interface for rerankers.
    """

    @abstractmethod
    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank candidates for the given query.

        Args:
            query: Natural language query string.
            candidates: List of candidate chunk records.

        Returns:
            List of candidate chunk records, sorted by rerank score descending.
        """
        pass


class CrossEncoderReranker(Reranker):
    """
    Reranker based on CrossEncoder models from sentence-transformers.
    Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        logger.info("Loading CrossEncoder reranker: %s", model_name)
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
        logger.info("CrossEncoder reranker loaded successfully.")

    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        pairs = [[query, c["content"]] for c in candidates]
        scores = self.model.predict(pairs)

        # Update candidate scores
        for c, score in zip(candidates, scores):
            c["rerank_score"] = float(score)
            c["score"] = float(score)  # Overwrite primary retrieval score with reranked score

        # Sort descending
        reranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
        return reranked


class BGEReranker(Reranker):
    """
    Reranker based on BAAI/bge-reranker-large or base.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-large") -> None:
        self.model_name = model_name
        logger.info("Loading BGE reranker: %s", model_name)
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
        logger.info("BGE reranker loaded successfully.")

    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        pairs = [[query, c["content"]] for c in candidates]
        scores = self.model.predict(pairs)

        for c, score in zip(candidates, scores):
            c["rerank_score"] = float(score)
            c["score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
        return reranked


def get_reranker(model_name: str = "BAAI/bge-reranker-large") -> Reranker:
    """
    Reranker factory and model abstraction.
    """
    model_name_lower = model_name.lower()
    if "minilm" in model_name_lower or "ms-marco" in model_name_lower:
        path = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        if "/" in model_name:
            path = model_name
        return CrossEncoderReranker(path)
    elif "bge-reranker-base" in model_name_lower:
        return BGEReranker("BAAI/bge-reranker-base")
    else:
        path = "BAAI/bge-reranker-large"
        if "/" in model_name:
            path = model_name
        return BGEReranker(path)


class RRFFusion:
    """
    Reciprocal Rank Fusion (RRF) for merging candidate search lists.
    RRF(d) = Σ 1/(k + rank)
    """

    def __init__(self, rrf_k: int = 60) -> None:
        self.rrf_k = rrf_k

    def fuse(
        self,
        semantic_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int = 10,
        alpha: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Fuse lists using Reciprocal Rank Fusion.
        """
        rrf_scores: Dict[str, float] = {}
        chunk_lookup: Dict[str, Dict[str, Any]] = {}

        # Alpha weight is optionally used to scale dense vs sparse lists
        for rank, r in enumerate(semantic_results, 1):
            cid = r["chunk_id"]
            if cid not in chunk_lookup:
                chunk_lookup[cid] = r
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + alpha / (self.rrf_k + rank)

        for rank, r in enumerate(bm25_results, 1):
            cid = r["chunk_id"]
            if cid not in chunk_lookup:
                chunk_lookup[cid] = r
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 - alpha) / (self.rrf_k + rank)

        results = []
        for cid, score in rrf_scores.items():
            r = dict(chunk_lookup[cid])
            r["score"] = score
            r["semantic_score"] = next((x.get("score", x.get("semantic_score", 0.0)) for x in semantic_results if x["chunk_id"] == cid), 0.0)
            r["bm25_score"] = next((x.get("score", x.get("bm25_score", 0.0)) for x in bm25_results if x["chunk_id"] == cid), 0.0)
            results.append(r)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]



class ScoreFusion:
    """
    Combines semantic and BM25 scores. Supports Linear score fusion and Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, alpha: float = 0.7, fusion_type: str = "rrf", rrf_k: int = 60) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self.fusion_type = fusion_type.lower()
        self.rrf_k = rrf_k

    def fuse(
        self,
        semantic_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int = 10,
        normalize: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Fuse semantic and BM25 matches.
        """
        if self.fusion_type == "rrf":
            return self._fuse_rrf(semantic_results, bm25_results, top_k)

        all_chunks: Dict[str, Dict[str, Any]] = {}

        # Load semantic results
        for r in semantic_results:
            cid = r["chunk_id"]
            all_chunks[cid] = {
                "chunk_id": cid,
                "content": r["content"],
                "source": r["source"],
                "page": r["page"],
                "section": r["section"],
                "semantic_score": r.get("score", r.get("semantic_score", 0.0)),
                "bm25_score": 0.0,
            }
            if "child_content" in r:
                all_chunks[cid]["child_content"] = r["child_content"]

        # Load BM25 results
        for r in bm25_results:
            cid = r["chunk_id"]
            if cid in all_chunks:
                all_chunks[cid]["bm25_score"] = r.get("score", r.get("bm25_score", 0.0))
            else:
                all_chunks[cid] = {
                    "chunk_id": cid,
                    "content": r["content"],
                    "source": r["source"],
                    "page": r["page"],
                    "section": r["section"],
                    "semantic_score": 0.0,
                    "bm25_score": r.get("score", r.get("bm25_score", 0.0)),
                }
                if "child_content" in r:
                    all_chunks[cid]["child_content"] = r["child_content"]

        results = list(all_chunks.values())
        if not results:
            return []

        # Extract scores
        sem_scores = np.array([r["semantic_score"] for r in results], dtype=np.float32)
        bm25_scores = np.array([r["bm25_score"] for r in results], dtype=np.float32)

        if normalize:
            # Min-Max normalize
            sem_norm = self._minmax_normalize(sem_scores)
            bm25_norm = self._minmax_normalize(bm25_scores)
        else:
            sem_norm = sem_scores
            bm25_norm = bm25_scores

        # Compute final fused scores
        for i, r in enumerate(results):
            r["score"] = float(self.alpha * sem_norm[i] + (1.0 - self.alpha) * bm25_norm[i])

        # Sort and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _fuse_rrf(
        self,
        semantic_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) for combining search results from multiple lists.
        """
        rrf_scores: Dict[str, float] = {}
        chunk_lookup: Dict[str, Dict[str, Any]] = {}

        # Helper to index ranks
        for rank, r in enumerate(semantic_results, 1):
            cid = r["chunk_id"]
            if cid not in chunk_lookup:
                chunk_lookup[cid] = r
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + self.alpha / (self.rrf_k + rank)

        for rank, r in enumerate(bm25_results, 1):
            cid = r["chunk_id"]
            if cid not in chunk_lookup:
                chunk_lookup[cid] = r
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 - self.alpha) / (self.rrf_k + rank)

        results = []
        for cid, score in rrf_scores.items():
            r = dict(chunk_lookup[cid])
            r["score"] = score
            # Preserve child/semantic/bm25 metrics if needed
            r["semantic_score"] = next((x.get("score", x.get("semantic_score", 0.0)) for x in semantic_results if x["chunk_id"] == cid), 0.0)
            r["bm25_score"] = next((x.get("score", x.get("bm25_score", 0.0)) for x in bm25_results if x["chunk_id"] == cid), 0.0)
            results.append(r)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    @staticmethod
    def _minmax_normalize(scores: np.ndarray) -> np.ndarray:
        mn, mx = scores.min(), scores.max()
        if mx - mn < 1e-10:
            return np.zeros_like(scores, dtype=np.float32)
        return (scores - mn) / (mx - mn)
