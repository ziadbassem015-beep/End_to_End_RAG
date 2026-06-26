"""
tests/test_metrics.py
=====================
Unit tests for every metric in evaluation/metrics.py.

Tests verify:
- Correct formula implementation (especially MAP which was wrong in v1)
- Edge cases: empty retrieved, empty relevant, no overlap
- Known-answer cases with manually computed expected values
"""

from __future__ import annotations

import math
import pytest

from rag_system.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    mrr,
    average_precision,
    ndcg_at_k,
    compute_all_metrics,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

RETRIEVED_PERFECT = ["a", "b", "c", "d", "e"]  # All 3 relevant in top-3
RETRIEVED_PARTIAL = ["x", "a", "y", "b", "z"]  # Relevant at positions 2, 4
RETRIEVED_NONE    = ["x", "y", "z", "w", "v"]  # No relevant items
RELEVANT          = ["a", "b", "c"]


# ─── Precision@K ─────────────────────────────────────────────────────────────

class TestPrecisionAtK:
    def test_perfect(self):
        # 3 relevant in top-3 → P@3 = 3/3 = 1.0
        assert precision_at_k(RETRIEVED_PERFECT, RELEVANT, k=3) == pytest.approx(1.0)

    def test_partial(self):
        # retrieved[:3] = [x, a, y] → 1 hit in 3 → P@3 = 1/3
        assert precision_at_k(RETRIEVED_PARTIAL, RELEVANT, k=3) == pytest.approx(1 / 3)

    def test_no_overlap(self):
        assert precision_at_k(RETRIEVED_NONE, RELEVANT, k=5) == pytest.approx(0.0)

    def test_empty_retrieved(self):
        assert precision_at_k([], RELEVANT, k=10) == pytest.approx(0.0)

    def test_k_zero(self):
        assert precision_at_k(RETRIEVED_PERFECT, RELEVANT, k=0) == pytest.approx(0.0)

    def test_denominator_is_always_k(self):
        # P@10 with only 2 relevant items found = 2/10, not 2/5
        assert precision_at_k(["a", "b", "x", "y", "z"], RELEVANT, k=10) == pytest.approx(2 / 10)

    def test_k_larger_than_retrieved(self):
        # If k > len(retrieved), still use k as denominator
        result = precision_at_k(["a", "b"], RELEVANT, k=10)
        assert result == pytest.approx(2 / 10)


# ─── Recall@K ────────────────────────────────────────────────────────────────

class TestRecallAtK:
    def test_perfect(self):
        # All 3 relevant found → R@5 = 3/3 = 1.0
        assert recall_at_k(RETRIEVED_PERFECT, RELEVANT, k=5) == pytest.approx(1.0)

    def test_partial(self):
        # retrieved[:5] = [x, a, y, b, z] → 2 relevant found → R@5 = 2/3
        assert recall_at_k(RETRIEVED_PARTIAL, RELEVANT, k=5) == pytest.approx(2 / 3)

    def test_no_overlap(self):
        assert recall_at_k(RETRIEVED_NONE, RELEVANT, k=5) == pytest.approx(0.0)

    def test_empty_relevant(self):
        assert recall_at_k(RETRIEVED_PERFECT, [], k=5) == pytest.approx(0.0)

    def test_empty_retrieved(self):
        assert recall_at_k([], RELEVANT, k=10) == pytest.approx(0.0)


# ─── Hit Rate@K ──────────────────────────────────────────────────────────────

class TestHitRateAtK:
    def test_hit(self):
        assert hit_rate_at_k(RETRIEVED_PARTIAL, RELEVANT, k=5) == 1.0

    def test_miss(self):
        assert hit_rate_at_k(RETRIEVED_NONE, RELEVANT, k=5) == 0.0

    def test_k_too_small(self):
        # Retrieved = [x, a, ...], k=1 → only "x" considered → miss
        assert hit_rate_at_k(RETRIEVED_PARTIAL, RELEVANT, k=1) == 0.0

    def test_k_reaches_first_relevant(self):
        # Retrieved = [x, a, ...], k=2 → "a" is at position 2 → hit
        assert hit_rate_at_k(RETRIEVED_PARTIAL, RELEVANT, k=2) == 1.0


# ─── MRR ─────────────────────────────────────────────────────────────────────

class TestMRR:
    def test_first_position(self):
        # "a" is at rank 1 → MRR = 1.0
        assert mrr(["a", "x", "y"], RELEVANT) == pytest.approx(1.0)

    def test_second_position(self):
        # "a" is at rank 2 → MRR = 0.5
        assert mrr(["x", "a", "y"], RELEVANT) == pytest.approx(0.5)

    def test_third_position(self):
        assert mrr(["x", "y", "a"], RELEVANT) == pytest.approx(1 / 3)

    def test_no_relevant(self):
        assert mrr(RETRIEVED_NONE, RELEVANT) == pytest.approx(0.0)

    def test_empty_retrieved(self):
        assert mrr([], RELEVANT) == pytest.approx(0.0)

    def test_returns_first_hit_only(self):
        # "b" at rank 2, "a" at rank 4 → MRR = 1/2 (first hit is b)
        assert mrr(["x", "b", "y", "a"], RELEVANT) == pytest.approx(0.5)


# ─── MAP ─────────────────────────────────────────────────────────────────────

class TestAveragePrecision:
    """
    Tests specifically designed to catch the original bug where
    every hit contributed 1.0 (idx/idx) instead of running precision.
    """

    def test_perfect_ranking(self):
        # retrieved = [a, b, c], relevant = [a, b, c]
        # P@1 = 1/1, P@2 = 2/2, P@3 = 3/3 → AP = (1.0 + 1.0 + 1.0) / 3 = 1.0
        assert average_precision(["a", "b", "c"], ["a", "b", "c"], k=3) == pytest.approx(1.0)

    def test_delayed_hits(self):
        # retrieved = [x, a, y, b], relevant = [a, b]
        # Hit at rank 2: P@2 = 1/2
        # Hit at rank 4: P@4 = 2/4 = 0.5
        # AP = (0.5 + 0.5) / 2 = 0.5
        assert average_precision(["x", "a", "y", "b"], ["a", "b"], k=4) == pytest.approx(0.5)

    def test_single_hit_early(self):
        # retrieved = [a, x, x, x], relevant = [a, b, c]
        # Only 1 hit at rank 1: P@1 = 1/1 = 1.0
        # AP = 1.0 / 3 (denominator is |relevant| = 3)
        assert average_precision(["a", "x", "x", "x"], ["a", "b", "c"], k=4) == pytest.approx(1 / 3)

    def test_no_hits(self):
        assert average_precision(RETRIEVED_NONE, RELEVANT, k=5) == pytest.approx(0.0)

    def test_empty_relevant(self):
        assert average_precision(RETRIEVED_PERFECT, [], k=5) == pytest.approx(0.0)

    def test_not_inflated_like_original_bug(self):
        """
        The original rag_metrics.py computed idx/idx = 1.0 for every hit.
        This test FAILS with the old code and PASSES with the fixed code.

        retrieved = [x, a, y, b] — hits at rank 2 and 4
        OLD (buggy): AP = (1.0 + 1.0) / 2 = 1.0   ← WRONG
        NEW (fixed): AP = (0.5 + 0.5) / 2 = 0.5   ← CORRECT
        """
        result = average_precision(["x", "a", "y", "b"], ["a", "b"], k=4)
        assert result == pytest.approx(0.5), (
            f"MAP bug detected! Got {result:.4f}, expected 0.5. "
            "The original code was computing idx/idx=1.0 for every hit."
        )


# ─── NDCG ────────────────────────────────────────────────────────────────────

class TestNDCGAtK:
    def test_perfect(self):
        # Ideal ranking → NDCG = 1.0
        assert ndcg_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert ndcg_at_k(RETRIEVED_NONE, RELEVANT, k=5) == pytest.approx(0.0)

    def test_single_hit_at_rank_1(self):
        # DCG = 1/log2(2) = 1.0, IDCG (1 relevant) = 1.0
        assert ndcg_at_k(["a", "x", "y"], ["a"], k=3) == pytest.approx(1.0)

    def test_single_hit_at_rank_2(self):
        # DCG = 1/log2(3), IDCG = 1/log2(2) = 1.0
        expected = (1 / math.log2(3)) / (1 / math.log2(2))
        assert ndcg_at_k(["x", "a", "y"], ["a"], k=3) == pytest.approx(expected)

    def test_partial_hit(self):
        # retrieved = [a, x, b], relevant = [a, b, c]
        # DCG = 1/log2(2) + 1/log2(4)
        # IDCG (3 relevant, k=3): 1/log2(2) + 1/log2(3) + 1/log2(4)
        dcg = 1 / math.log2(2) + 1 / math.log2(4)
        idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
        expected = dcg / idcg
        assert ndcg_at_k(["a", "x", "b"], ["a", "b", "c"], k=3) == pytest.approx(expected, rel=1e-4)

    def test_empty_relevant(self):
        assert ndcg_at_k(RETRIEVED_PERFECT, [], k=5) == pytest.approx(0.0)


# ─── compute_all_metrics ─────────────────────────────────────────────────────

class TestComputeAllMetrics:
    def test_returns_all_keys(self):
        result = compute_all_metrics(["a", "b"], ["a"], k=5)
        expected_keys = {
            "precision_at_k", "recall_at_k", "hit_rate_at_k",
            "mrr", "average_precision", "ndcg_at_k",
        }
        assert set(result.keys()) == expected_keys

    def test_values_are_floats(self):
        result = compute_all_metrics(["a", "b"], ["a"], k=5)
        for key, val in result.items():
            assert isinstance(val, float), f"{key} is not a float"

    def test_all_in_range(self):
        result = compute_all_metrics(RETRIEVED_PERFECT, RELEVANT, k=5)
        for key, val in result.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0, 1]"
