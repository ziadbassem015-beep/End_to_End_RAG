"""
evaluation/metrics.py
======================
Mathematically correct implementations of all retrieval metrics.

Metrics implemented:
    - Precision@K
    - Recall@K
    - Hit Rate@K (HR@K)
    - MRR (Mean Reciprocal Rank)
    - MAP (Mean Average Precision)   ← Fixed from original (was computing recall)
    - NDCG (Normalized Discounted Cumulative Gain)

Every function is:
    - Pure (no side effects)
    - Type-annotated
    - Documented with the exact formula
    - Unit-tested in tests/test_metrics.py
"""

from __future__ import annotations

import math
from typing import List, Set


# ─── Type alias ──────────────────────────────────────────────────────────────

ChunkIds = List[str]   # List of chunk ID strings in ranked order


# ─── Precision@K ─────────────────────────────────────────────────────────────

def precision_at_k(retrieved: ChunkIds, relevant: ChunkIds, k: int) -> float:
    """
    P@K = |relevant ∩ retrieved[:K]| / K

    The fraction of the top-K retrieved items that are relevant.
    Denominator is always K (not the number of retrieved items).

    Args:
        retrieved: Ranked list of retrieved chunk IDs.
        relevant:  Ground truth relevant chunk IDs.
        k:         Cutoff rank.

    Returns:
        Float in [0, 1].
    """
    if k <= 0:
        return 0.0
    relevant_set: Set[str] = set(relevant)
    hits = sum(1 for cid in retrieved[:k] if cid in relevant_set)
    return hits / k


# ─── Recall@K ────────────────────────────────────────────────────────────────

def recall_at_k(retrieved: ChunkIds, relevant: ChunkIds, k: int) -> float:
    """
    R@K = |relevant ∩ retrieved[:K]| / |relevant|

    The fraction of all relevant items that appear in the top-K results.

    Args:
        retrieved: Ranked list of retrieved chunk IDs.
        relevant:  Ground truth relevant chunk IDs.
        k:         Cutoff rank.

    Returns:
        Float in [0, 1]. Returns 0.0 if relevant is empty.
    """
    if not relevant:
        return 0.0
    relevant_set: Set[str] = set(relevant)
    hits = sum(1 for cid in retrieved[:k] if cid in relevant_set)
    return hits / len(relevant_set)


# ─── Hit Rate@K ──────────────────────────────────────────────────────────────

def hit_rate_at_k(retrieved: ChunkIds, relevant: ChunkIds, k: int) -> float:
    """
    HR@K = 1 if any relevant item in retrieved[:K] else 0

    Binary: did we retrieve at least one relevant document?

    Args:
        retrieved: Ranked list of retrieved chunk IDs.
        relevant:  Ground truth relevant chunk IDs.
        k:         Cutoff rank.

    Returns:
        1.0 or 0.0.
    """
    relevant_set: Set[str] = set(relevant)
    return 1.0 if any(cid in relevant_set for cid in retrieved[:k]) else 0.0


# ─── MRR ─────────────────────────────────────────────────────────────────────

def mrr(retrieved: ChunkIds, relevant: ChunkIds) -> float:
    """
    MRR = 1 / rank_of_first_relevant_item

    Mean Reciprocal Rank over the full retrieved list.
    Returns 0 if no relevant item is found.

    Args:
        retrieved: Ranked list of retrieved chunk IDs (full list, no cutoff).
        relevant:  Ground truth relevant chunk IDs.

    Returns:
        Float in [0, 1].
    """
    relevant_set: Set[str] = set(relevant)
    for rank, cid in enumerate(retrieved, 1):
        if cid in relevant_set:
            return 1.0 / rank
    return 0.0


# ─── MAP ─────────────────────────────────────────────────────────────────────

def average_precision(retrieved: ChunkIds, relevant: ChunkIds, k: int) -> float:
    """
    AP@K = (1/|relevant|) * Σ P@i  for each i in 1..K where retrieved[i] is relevant

    Average Precision at K. Rewards finding relevant items early in the ranking.

    IMPORTANT: P@i is the running precision at rank i, not 1.0.
    This was the bug in the original code (it used `idx/idx = 1.0`).

    Args:
        retrieved: Ranked list of retrieved chunk IDs.
        relevant:  Ground truth relevant chunk IDs.
        k:         Cutoff rank.

    Returns:
        Float in [0, 1]. Returns 0.0 if relevant is empty.
    """
    if not relevant:
        return 0.0

    relevant_set: Set[str] = set(relevant)
    hits = 0
    precision_sum = 0.0

    for rank, cid in enumerate(retrieved[:k], 1):
        if cid in relevant_set:
            hits += 1
            # Running precision at rank i: hits_so_far / rank
            precision_sum += hits / rank

    return precision_sum / len(relevant_set)


# ─── NDCG ────────────────────────────────────────────────────────────────────

def ndcg_at_k(retrieved: ChunkIds, relevant: ChunkIds, k: int) -> float:
    """
    NDCG@K = DCG@K / IDCG@K

    DCG@K  = Σ rel_i / log2(i + 1)  for i in 1..K
    IDCG@K = DCG of the ideal ranking (all relevant docs first)

    Binary relevance: rel_i = 1 if retrieved[i] is relevant, else 0.

    Args:
        retrieved: Ranked list of retrieved chunk IDs.
        relevant:  Ground truth relevant chunk IDs.
        k:         Cutoff rank.

    Returns:
        Float in [0, 1].
    """
    relevant_set: Set[str] = set(relevant)

    # DCG
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, cid in enumerate(retrieved[:k], 1)
        if cid in relevant_set
    )

    # IDCG: ideal case — all relevant docs appear first
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


# ─── All-in-one helper ───────────────────────────────────────────────────────

def compute_all_metrics(
    retrieved: ChunkIds,
    relevant: ChunkIds,
    k: int = 10,
) -> dict:
    """
    Compute all retrieval metrics for a single query.

    Args:
        retrieved: Ranked list of retrieved chunk IDs.
        relevant:  Ground truth relevant chunk IDs.
        k:         Cutoff rank for @K metrics.

    Returns:
        Dict with keys: precision_at_k, recall_at_k, hit_rate_at_k,
                         mrr, average_precision, ndcg_at_k
    """
    return {
        "precision_at_k":    precision_at_k(retrieved, relevant, k),
        "recall_at_k":       recall_at_k(retrieved, relevant, k),
        "hit_rate_at_k":     hit_rate_at_k(retrieved, relevant, k),
        "mrr":               mrr(retrieved, relevant),
        "average_precision": average_precision(retrieved, relevant, k),
        "ndcg_at_k":         ndcg_at_k(retrieved, relevant, k),
    }
