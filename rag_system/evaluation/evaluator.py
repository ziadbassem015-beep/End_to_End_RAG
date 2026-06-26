"""
evaluation/evaluator.py
========================
Evaluation orchestrator that loads the VectorStore and evaluation dataset,
retrieves matches for queries, and aggregates mathematical metrics.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from rag_system.evaluation.metrics import compute_all_metrics
from rag_system.vectorstore.faiss_store import FAISSStore
from rag_system.retrieval.retriever import HybridRetriever, BM25Index
from rag_system.retrieval.reranker import Reranker
from rag_system.retrieval.expansion import QueryExpander

logger = logging.getLogger(__name__)


@dataclass
class RelevanceJudgment:
    """Ground truth for a single query."""
    query_id: str
    query: str
    relevant_chunk_ids: List[str]


@dataclass
class QueryResult:
    """Full per-query evaluation result."""
    query_id: str
    query: str
    retrieved_ids: List[str]
    relevant_ids: List[str]

    # Metrics
    precision_at_k: float
    recall_at_k: float
    hit_rate_at_k: float
    mrr: float
    average_precision: float
    ndcg_at_k: float

    # Counts
    retrieved_count: int
    relevant_count: int
    relevant_retrieved_count: int
    missed_ids: List[str] = field(default_factory=list)

    # Timing
    retrieval_time_ms: float = 0.0

    def __post_init__(self) -> None:
        self.missed_ids = [
            cid for cid in self.relevant_ids if cid not in self.retrieved_ids
        ]


@dataclass
class AggregatedMetrics:
    """Dataset-level aggregated metrics."""
    avg_precision_at_k: float
    avg_recall_at_k: float
    avg_hit_rate_at_k: float
    avg_mrr: float
    avg_map: float
    avg_ndcg: float

    median_precision_at_k: float
    median_recall_at_k: float
    median_ndcg: float

    std_precision_at_k: float
    std_recall_at_k: float
    std_ndcg: float


@dataclass
class EvaluationResults:
    """Complete evaluation output."""
    total_queries: int
    top_k: int
    model_name: str
    alpha: float
    aggregated: AggregatedMetrics
    per_query: List[QueryResult] = field(default_factory=list)
    best_queries: List[QueryResult] = field(default_factory=list)
    worst_queries: List[QueryResult] = field(default_factory=list)
    failing_queries: List[QueryResult] = field(default_factory=list)


class RAGEvaluator:
    """
    Orchestrate retrieval evaluation over a dataset of queries.
    """

    def __init__(
        self,
        vector_store_path: str | Path,
        eval_dataset_path: str | Path,
        model_name: str = "BAAI/bge-small-en-v1.5",
        top_k: int = 10,
        alpha: float = 0.7,
        similarity_threshold: float = 0.0,
        hybrid: bool = True,
        fusion_type: str = "rrf",
        rrf_k: int = 60,
        reranker: Optional[Reranker] = None,
        expander: Optional[Any] = None,
        expansion_method: str = "none",
        use_parent_child: bool = False,
        retrieval_pool_multiplier: int = 10,
        use_query_expansion: bool = False,
        query_variants: int = 4,
        use_hyde: bool = False,
        use_metadata_filtering: bool = True,
    ) -> None:
        self.top_k = top_k
        self.model_name = model_name
        self.alpha = alpha if hybrid else 1.0

        logger.info("RAGEvaluator: Loading VectorStore from %s", vector_store_path)
        path_str = str(vector_store_path)

        # 1. Load/Reconstruct VectorStore and Chunk list for BM25
        if os.path.isdir(path_str):
            # Load metadata to extract dimension
            metadata_file = os.path.join(path_str, "metadata.json")
            dimension = 384
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        dimension = meta.get("dimension", 384)
                except Exception as e:
                    logger.warning("Failed to parse dimension from FAISS metadata: %s", e)

            vector_store = FAISSStore(dimension=dimension)
            vector_store.load(path_str)
            chunks = list(vector_store.metadata.values())
        else:
            # Fall back to loading JSON embeddings file and indexing them
            from rag_system.embeddings.embedder import Embedder
            chunks = Embedder.load(path_str)
            if not chunks:
                raise ValueError(f"No chunks found in embeddings file: {path_str}")
            dimension = len(chunks[0]["embedding"])
            vector_store = FAISSStore(dimension=dimension)
            vector_store.add_batch(chunks)

        # 2. Build BM25 index
        bm25_index = None
        if hybrid and chunks:
            bm25_index = BM25Index(chunks)

        # 3. Create Retriever
        self._retriever = HybridRetriever(
            vector_store=vector_store,
            bm25_index=bm25_index,
            model_name=model_name,
            alpha=self.alpha,
            similarity_threshold=similarity_threshold,
            fusion_type=fusion_type,
            rrf_k=rrf_k,
            reranker=reranker,
            use_parent_child=use_parent_child,
            retrieval_pool_multiplier=retrieval_pool_multiplier,
            use_query_expansion=use_query_expansion,
            query_variants=query_variants,
            use_hyde=use_hyde,
            use_metadata_filtering=use_metadata_filtering,
            expander=expander,
            expansion_method=expansion_method,
        )


        self._judgments = self._load_eval_dataset(eval_dataset_path)
        logger.info("Loaded %d evaluation queries.", len(self._judgments))

    def evaluate(self) -> EvaluationResults:
        """Run evaluation over all queries."""
        logger.info("Starting evaluation of %d queries...", len(self._judgments))

        per_query_results: List[QueryResult] = []

        for judgment in self._judgments:
            result = self._evaluate_single(judgment)
            per_query_results.append(result)

            logger.debug(
                "[%s] NDCG=%.3f P@%d=%.3f R@%d=%.3f MRR=%.3f | %r",
                result.query_id,
                result.ndcg_at_k,
                self.top_k, result.precision_at_k,
                self.top_k, result.recall_at_k,
                result.mrr,
                result.query[:50],
            )

        aggregated = self._aggregate(per_query_results)

        # Sort by NDCG descending to get best/worst/failing
        sorted_by_ndcg = sorted(per_query_results, key=lambda r: r.ndcg_at_k, reverse=True)
        failing_threshold = 0.5

        results = EvaluationResults(
            total_queries=len(per_query_results),
            top_k=self.top_k,
            model_name=self.model_name,
            alpha=self.alpha,
            aggregated=aggregated,
            per_query=per_query_results,
            best_queries=sorted_by_ndcg[:3],
            worst_queries=sorted_by_ndcg[-3:],
            failing_queries=[r for r in per_query_results if r.ndcg_at_k < failing_threshold],
        )

        logger.info(
            "Evaluation complete | NDCG=%.4f P@%d=%.4f R@%d=%.4f MRR=%.4f MAP=%.4f",
            aggregated.avg_ndcg,
            self.top_k, aggregated.avg_precision_at_k,
            self.top_k, aggregated.avg_recall_at_k,
            aggregated.avg_mrr,
            aggregated.avg_map,
        )
        return results

    def _evaluate_single(self, judgment: RelevanceJudgment) -> QueryResult:
        """Evaluate a single query."""
        t0 = time.perf_counter()
        retrieval = self._retriever.retrieve(judgment.query, top_k=self.top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        retrieved_ids = [r.chunk_id for r in retrieval]
        relevant_ids = judgment.relevant_chunk_ids

        metrics = compute_all_metrics(retrieved_ids, relevant_ids, k=self.top_k)

        return QueryResult(
            query_id=judgment.query_id,
            query=judgment.query,
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            precision_at_k=metrics["precision_at_k"],
            recall_at_k=metrics["recall_at_k"],
            hit_rate_at_k=metrics["hit_rate_at_k"],
            mrr=metrics["mrr"],
            average_precision=metrics["average_precision"],
            ndcg_at_k=metrics["ndcg_at_k"],
            retrieved_count=len(retrieved_ids),
            relevant_count=len(relevant_ids),
            relevant_retrieved_count=sum(
                1 for cid in retrieved_ids if cid in set(relevant_ids)
            ),
            retrieval_time_ms=round(elapsed_ms, 2),
        )

    @staticmethod
    def _aggregate(results: List[QueryResult]) -> AggregatedMetrics:
        """Compute statistics across query results."""
        def _stats(values: List[float]):
            arr = np.array(values)
            return float(np.mean(arr)), float(np.median(arr)), float(np.std(arr))

        p_mean, p_med, p_std = _stats([r.precision_at_k for r in results])
        r_mean, r_med, r_std = _stats([r.recall_at_k for r in results])
        n_mean, n_med, n_std = _stats([r.ndcg_at_k for r in results])

        return AggregatedMetrics(
            avg_precision_at_k=p_mean,
            avg_recall_at_k=r_mean,
            avg_hit_rate_at_k=float(np.mean([r.hit_rate_at_k for r in results])),
            avg_mrr=float(np.mean([r.mrr for r in results])),
            avg_map=float(np.mean([r.average_precision for r in results])),
            avg_ndcg=n_mean,
            median_precision_at_k=p_med,
            median_recall_at_k=r_med,
            median_ndcg=n_med,
            std_precision_at_k=p_std,
            std_recall_at_k=r_std,
            std_ndcg=n_std,
        )

    @staticmethod
    def _load_eval_dataset(path: str | Path) -> List[RelevanceJudgment]:
        """Load and parse evaluation dataset."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Eval dataset not found: {path}")

        data = json.loads(path.read_text(encoding="utf-8"))
        judgments = []
        for idx, item in enumerate(data):
            judgments.append(
                RelevanceJudgment(
                    query_id=str(item.get("query_id", f"q_{idx:03d}")),
                    query=item["query"],
                    relevant_chunk_ids=[
                        str(cid) for cid in item.get("relevant_chunk_ids", [])
                    ],
                )
            )
        return judgments
