"""
evaluation/reports.py
======================
Generate evaluation reports in JSON and Markdown formats.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

from rag_system.evaluation.evaluator import EvaluationResults

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate evaluation reports from EvaluationResults."""

    def __init__(self, output_dir: str | Path = "data/reports/evaluation") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ReportGenerator: output dir = %s", self.output_dir)

    # ── JSON Report ──────────────────────────────────────────────────────────

    def generate_json(
        self,
        results: EvaluationResults,
        filename: str = "evaluation_report.json",
    ) -> Path:
        """Generate full JSON evaluation report."""
        agg = results.aggregated

        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": results.model_name,
                "top_k": results.top_k,
                "hybrid_alpha": results.alpha,
                "total_queries": results.total_queries,
            },
            "summary": {
                f"avg_precision@{results.top_k}":    round(agg.avg_precision_at_k, 4),
                f"avg_recall@{results.top_k}":       round(agg.avg_recall_at_k, 4),
                f"avg_hit_rate@{results.top_k}":     round(agg.avg_hit_rate_at_k, 4),
                "avg_mrr":                           round(agg.avg_mrr, 4),
                "avg_map":                           round(agg.avg_map, 4),
                f"avg_ndcg@{results.top_k}":         round(agg.avg_ndcg, 4),
                f"median_precision@{results.top_k}": round(agg.median_precision_at_k, 4),
                f"median_recall@{results.top_k}":    round(agg.median_recall_at_k, 4),
                f"median_ndcg@{results.top_k}":      round(agg.median_ndcg, 4),
                f"std_ndcg@{results.top_k}":         round(agg.std_ndcg, 4),
            },
            "per_query": [
                {
                    "query_id":              q.query_id,
                    "query":                 q.query,
                    f"precision@{results.top_k}": round(q.precision_at_k, 4),
                    f"recall@{results.top_k}":    round(q.recall_at_k, 4),
                    f"hit_rate@{results.top_k}":  round(q.hit_rate_at_k, 4),
                    "mrr":                   round(q.mrr, 4),
                    "average_precision":     round(q.average_precision, 4),
                    f"ndcg@{results.top_k}": round(q.ndcg_at_k, 4),
                    "relevant_count":        q.relevant_count,
                    "retrieved_count":       q.retrieved_count,
                    "relevant_retrieved":    q.relevant_retrieved_count,
                    "missed_ids":            q.missed_ids,
                    "retrieved_ids":         q.retrieved_ids,
                    "retrieval_time_ms":     q.retrieval_time_ms,
                }
                for q in results.per_query
            ],
            "analysis": {
                "best_queries": [
                    {"query_id": q.query_id, "query": q.query, "ndcg": round(q.ndcg_at_k, 4)}
                    for q in results.best_queries
                ],
                "worst_queries": [
                    {"query_id": q.query_id, "query": q.query, "ndcg": round(q.ndcg_at_k, 4)}
                    for q in results.worst_queries
                ],
                "failing_queries_count": len(results.failing_queries),
                "failing_queries": [
                    {"query_id": q.query_id, "query": q.query, "ndcg": round(q.ndcg_at_k, 4)}
                    for q in results.failing_queries
                ],
            },
            "recommendations": self._build_recommendations(results),
        }

        out = self.output_dir / filename
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("JSON report -> %s", out)
        return out

    # ── Markdown Report ──────────────────────────────────────────────────────

    def generate_markdown(
        self,
        results: EvaluationResults,
        filename: str = "evaluation_report.md",
    ) -> Path:
        """Generate rich Markdown evaluation report."""
        agg = results.aggregated
        k = results.top_k
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            "# RAG Evaluation Report\n",
            f"> **Generated:** {ts}  \n",
            f"> **Model:** `{results.model_name}`  \n",
            f"> **Top-K:** {k}  |  **Hybrid α:** {results.alpha}  \n\n",

            "---\n\n",
            "## 📊 Summary Metrics\n\n",
            f"| Metric | Value |\n|--------|-------|\n",
            f"| **Precision@{k}** | `{agg.avg_precision_at_k:.4f}` |\n",
            f"| **Recall@{k}** | `{agg.avg_recall_at_k:.4f}` |\n",
            f"| **Hit Rate@{k}** | `{agg.avg_hit_rate_at_k:.4f}` |\n",
            f"| **MRR** | `{agg.avg_mrr:.4f}` |\n",
            f"| **MAP** | `{agg.avg_map:.4f}` |\n",
            f"| **NDCG@{k}** | `{agg.avg_ndcg:.4f}` |\n",
            f"| Median NDCG | `{agg.median_ndcg:.4f}` |\n",
            f"| Std NDCG | `{agg.std_ndcg:.4f}` |\n\n",

            "---\n\n",
            "## 🏆 Best Performing Queries\n\n",
        ]

        for q in results.best_queries:
            lines.append(
                f"- `{q.query_id}` — **NDCG={q.ndcg_at_k:.3f}** — *{q.query}*\n"
            )

        lines += [
            "\n## ⚠️ Worst Performing Queries\n\n",
        ]
        for q in results.worst_queries:
            lines.append(
                f"- `{q.query_id}` — **NDCG={q.ndcg_at_k:.3f}** — *{q.query}*\n"
            )

        if results.failing_queries:
            n_fail = len(results.failing_queries)
            lines += [
                f"\n## ❌ Failing Queries (NDCG < 0.5) — {n_fail} / {results.total_queries}\n\n",
            ]
            for q in results.failing_queries:
                lines.append(f"- `{q.query_id}` NDCG={q.ndcg_at_k:.3f}: *{q.query}*\n")

        lines += [
            "\n---\n\n",
            "## 📋 Per-Query Detailed Metrics\n\n",
            f"| Query ID | Query | P@{k} | R@{k} | HR@{k} | MRR | MAP | NDCG | Missed | Time(ms) |\n",
            f"|----------|-------|------|------|-------|-----|-----|------|--------|----------|\n",
        ]

        for q in results.per_query:
            query_short = q.query[:35] + "..." if len(q.query) > 35 else q.query
            lines.append(
                f"| `{q.query_id}` | {query_short} | "
                f"{q.precision_at_k:.3f} | {q.recall_at_k:.3f} | "
                f"{q.hit_rate_at_k:.3f} | {q.mrr:.3f} | "
                f"{q.average_precision:.3f} | {q.ndcg_at_k:.3f} | "
                f"{len(q.missed_ids)} | {q.retrieval_time_ms:.1f} |\n"
            )

        lines += ["\n---\n\n", "## 💡 Recommendations\n\n"]
        for rec in self._build_recommendations(results):
            lines.append(f"- {rec}\n")

        out = self.output_dir / filename
        out.write_text("".join(lines), encoding="utf-8")
        logger.info("Markdown report -> %s", out)
        return out

    def generate_all(
        self,
        results: EvaluationResults,
        json_filename: str = "evaluation_report.json",
        md_filename: str = "evaluation_report.md",
    ) -> Dict[str, Path]:
        """Generate both JSON and Markdown reports."""
        return {
            "json": self.generate_json(results, json_filename),
            "markdown": self.generate_markdown(results, md_filename),
        }

    # ── Recommendations ──────────────────────────────────────────────────────

    @staticmethod
    def _build_recommendations(results: EvaluationResults) -> list:
        """Generate actionable recommendations based on metrics."""
        agg = results.aggregated
        k = results.top_k
        recs = []

        if agg.avg_ndcg < 0.4:
            recs.append(
                f"⚠️ NDCG@{k}={agg.avg_ndcg:.3f} is low. "
                "Run `python cli.py validate` to check eval dataset quality. "
                "Consider re-annotating relevant_chunk_ids using the corrected dataset."
            )

        if agg.avg_recall_at_k < 0.3:
            recs.append(
                f"⚠️ Recall@{k}={agg.avg_recall_at_k:.3f} is low. "
                "Try increasing `top_k`, reducing `chunk_size`, or adjusting BM25 weight (lower alpha)."
            )

        if agg.avg_precision_at_k < 0.1:
            recs.append(
                f"⚠️ Precision@{k}={agg.avg_precision_at_k:.3f} is very low. "
                "Consider adding a similarity threshold filter or cross-encoder reranking."
            )

        if agg.avg_hit_rate_at_k > 0.7 and agg.avg_precision_at_k < 0.15:
            recs.append(
                "ℹ️ High Hit Rate but low Precision suggests the system finds relevant docs "
                "but ranks them poorly. Increase semantic weight (higher alpha) or use reranking."
            )

        if len(results.failing_queries) > results.total_queries * 0.5:
            recs.append(
                f"🔴 {len(results.failing_queries)}/{results.total_queries} queries are failing. "
                "This strongly suggests an eval dataset annotation problem. "
                "Run `python cli.py validate --correct` to auto-correct relevant IDs."
            )

        if not recs:
            recs.append("✅ Metrics look healthy. Consider adding more diverse eval queries.")

        return recs
