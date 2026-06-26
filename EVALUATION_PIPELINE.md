# RAG Evaluation Pipeline - Production Quality

A comprehensive, modular RAG (Retrieval-Augmented Generation) evaluation system with professional-grade metrics, reporting, and analysis.

## 📋 Overview

This pipeline evaluates the retrieval quality of a RAG system by:
- Computing comprehensive retrieval metrics (Precision, Recall, MRR, MAP, NDCG)
- Analyzing per-query performance
- Identifying failure cases and best practices
- Generating detailed JSON and Markdown reports
- Providing both CLI and programmatic interfaces

## 🏗️ Architecture

```
rag_models.py          → Data models (Dataclasses)
rag_metrics.py         → Metric calculations
rag_retriever.py       → Optimized semantic search
rag_evaluator.py       → Evaluation orchestration
rag_report_generator.py → Report generation
rag_cli.py            → Command-line interface
```

## 🚀 Quick Start

### 1. Run Full Evaluation

```bash
uv run python rag_cli.py evaluate \
  --embeddings output_embeddings.json \
  --eval-queries eval_queries_comprehensive.json \
  --top-k 10
```

### 2. Single Query Retrieval

```bash
uv run python rag_cli.py query \
  --query "What is machine learning?" \
  --top-k 5
```

### 3. Batch Query Retrieval

```bash
uv run python rag_cli.py batch \
  --queries-file eval_queries_comprehensive.json \
  --top-k 5
```

## 📊 Metrics Explained

### Retrieval Metrics

- **Precision@K**: What fraction of top-K retrieved results are relevant?
  - Formula: `|relevant_retrieved| / K`
  - Range: [0, 1] (higher is better)

- **Recall@K**: What fraction of relevant items are in top-K?
  - Formula: `|relevant_retrieved| / |relevant|`
  - Range: [0, 1] (higher is better)

- **Hit Rate@K**: Is at least one relevant item in top-K?
  - Formula: 1 if any relevant, else 0
  - Range: [0, 1] (higher is better)

- **MRR (Mean Reciprocal Rank)**: Rank position of first relevant item
  - Formula: `1 / rank_of_first_relevant`
  - Range: [0, 1] (higher is better)

- **MAP (Mean Average Precision)**: Average of precisions at each relevant position
  - Formula: `(1/|relevant|) * sum(P@i for each relevant)`
  - Range: [0, 1] (higher is better)

- **NDCG (Normalized Discounted Cumulative Gain)**: Ranking quality metric
  - Formula: `DCG / IDCG`
  - Range: [0, 1] (1 = perfect ranking)

## 📁 Output

Generated reports in `evaluation_reports/`:

- `evaluation_report.json` - Machine-readable metrics
- `evaluation_report.md` - Human-readable analysis

### Report Contents

```json
{
  "metadata": {
    "timestamp": "2026-06-19T10:30:00",
    "total_queries": 10
  },
  "summary": {
    "avg_precision@10": 0.65,
    "avg_recall@10": 0.58,
    "avg_hit_rate@10": 0.90,
    "avg_mrr": 0.72,
    "avg_map": 0.61,
    "avg_ndcg": 0.68
  },
  "per_query_metrics": [...],
  "analysis": {
    "best_performing_queries": [...],
    "worst_performing_queries": [...],
    "failing_queries_count": 2
  }
}
```

## 📈 Evaluation Dataset Format

Create `eval_queries.json`:

```json
[
  {
    "query_id": "q_001",
    "query": "What is machine learning?",
    "relevant_chunk_ids": [1, 2, 5]
  },
  ...
]
```

## 🔧 Programmatic Usage

```python
from rag_evaluator import RAGEvaluator
from rag_report_generator import ReportGenerator

# Initialize evaluator
evaluator = RAGEvaluator(
    embeddings_path="output_embeddings.json",
    eval_dataset_path="eval_queries.json",
    top_k=10
)

# Run evaluation
results = evaluator.evaluate_dataset()

# Generate reports
report_gen = ReportGenerator()
report_gen.generate_all_reports(results)
```

## 📊 Single Query Evaluation

```python
from rag_retriever import Retriever

retriever = Retriever("output_embeddings.json")
results = retriever.retrieve("Your query here", top_k=5)

for result in results:
    print(f"Rank {result.rank}: {result.chunk.content[:100]}")
    print(f"Score: {result.score:.4f}\n")
```

## 🎯 Interpretation Guide

| Metric | Excellent | Good | Fair | Poor |
|--------|-----------|------|------|------|
| NDCG   | > 0.8    | 0.6-0.8 | 0.4-0.6 | < 0.4 |
| Precision@10 | > 0.7 | 0.5-0.7 | 0.3-0.5 | < 0.3 |
| Recall@10 | > 0.7 | 0.5-0.7 | 0.3-0.5 | < 0.3 |
| MRR | > 0.8 | 0.6-0.8 | 0.4-0.6 | < 0.4 |

## 🚨 Troubleshooting

### Low NDCG / Precision
- Check chunk quality and splitting
- Verify embedding model adequacy
- Increase chunk size if too small
- Review evaluation dataset relevance

### High Hit Rate but Low Precision
- Too many irrelevant results in top-K
- Consider stricter similarity threshold
- Use larger K for recall calculation

### Inconsistent Results
- Check embedding consistency
- Verify similarity computation
- Ensure query dataset diversity

## 📦 Dependencies

```
sentence-transformers>=3.0.0
torch>=2.0.0
numpy>=1.26.0
langchain>=0.2.0
```

## 📝 Best Practices

1. **Dataset**: Create balanced evaluation with 10-50 queries
2. **Relevance**: Ground truth must be accurate and complete
3. **Metrics**: Use NDCG as primary metric (captures ranking quality)
4. **Analysis**: Always review failure cases
5. **Iteration**: Use insights to improve chunks/embeddings

## 🎓 Graduation Project Highlights

- ✅ Production-quality code (type hints, logging, OOP)
- ✅ Comprehensive metrics (6+ evaluation metrics)
- ✅ Modular architecture (5 independent modules)
- ✅ Professional reporting (JSON + Markdown)
- ✅ CLI and programmatic interfaces
- ✅ Error handling and validation
- ✅ Qualitative analysis (failure cases)
- ✅ Scalable design

## 📜 License

MIT
