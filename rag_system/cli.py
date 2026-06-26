"""
cli.py
======
Unified CLI for the production RAG framework.
Commands:
    python cli.py chunk      - Load PDF and chunk into JSON
    python cli.py embed      - Generate embeddings and index into FAISS
    python cli.py validate   - Validate eval dataset (TOC, copyright, missing IDs)
    python cli.py evaluate   - Run evaluation from FAISS index and report metrics
    python cli.py query      - Run test query using FAISS vector store
    python cli.py benchmark  - Compare embeddings, chunking, and retrieval strategies
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import click
import yaml

# Add root folder to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(config_path: str = "rag_system/configs/default.yaml") -> dict:
    """Load YAML config file."""
    path = Path(config_path)
    if not path.exists():
        click.echo(f"Warning: Config not found: {path}, using defaults.", err=True)
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@click.group()
@click.option(
    "--config",
    default="rag_system/configs/default.yaml",
    show_default=True,
    help="Path to YAML configuration file.",
)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Logging verbosity.",
)
@click.pass_context
def cli(ctx: click.Context, config: str, log_level: str) -> None:
    """
    RAG System Production CLI
    """
    setup_logging(log_level)
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)


# ─── chunk ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--pdf",    default=None, help="Path to PDF file.")
@click.option("--output", default=None, help="Output chunks JSON path.")
@click.option("--chunk-size",    type=int, default=None, help="Max chars per chunk.")
@click.option("--chunk-overlap", type=int, default=None, help="Overlap chars.")
@click.option("--min-length",    type=int, default=None, help="Min chunk length.")
@click.pass_context
def chunk(ctx, pdf, output, chunk_size, chunk_overlap, min_length):
    """Load a PDF and split it into validated, ID-stable chunks."""
    cfg = ctx.obj["config"]
    ing = cfg.get("ingestion", {})
    chk = cfg.get("chunking", {})

    pdf_path  = pdf         or ing.get("pdf_path",   "data/row/andrew-ng-machine-learning-yearning.pdf")
    out_path  = output      or str(Path(ing.get("output_dir", "data/output")) / ing.get("chunks_file", "chunks.json"))
    c_size    = chunk_size  or chk.get("chunk_size",   800)
    c_overlap = chunk_overlap or chk.get("chunk_overlap", 150)
    c_min     = min_length  or chk.get("min_chunk_length", 50)

    click.echo(f"\n[PDF] Loading PDF: {pdf_path}")
    click.echo(f"   chunk_size={c_size}, overlap={c_overlap}, min_length={c_min}\n")

    from rag_system.ingestion.loader import PDFLoader
    from rag_system.ingestion.chunker import DocumentChunker

    loader = PDFLoader(pdf_path)
    docs = loader.load()
    click.echo(f"[SUCCESS] Loaded {len(docs)} pages.")

    c_child_size = chk.get("child_chunk_size", 0)
    c_child_overlap = chk.get("child_chunk_overlap", 0)

    chunker = DocumentChunker(
        chunk_size=c_size,
        chunk_overlap=c_overlap,
        min_chunk_length=c_min,
        separators=chk.get("separators"),
        child_chunk_size=c_child_size,
        child_chunk_overlap=c_child_overlap,
        use_parent_child=chk.get("use_parent_child", False),
        parent_size=chk.get("parent_size", 1600),
        parent_overlap=chk.get("parent_overlap", 300),
        child_size=chk.get("child_size", 400),
        child_overlap=chk.get("child_overlap", 100),
    )

    chunks = chunker.chunk(docs)
    chunker.save(chunks, out_path)

    click.echo(f"\n[SUCCESS] Saved {len(chunks)} chunks -> {out_path}")
    if chunks:
        click.echo(f"   First chunk ID: {chunks[0].chunk_id}")
        click.echo(f"   Last  chunk ID: {chunks[-1].chunk_id}")


# ─── embed ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--input",   default=None, help="Input chunks JSON path.")
@click.option("--output",  default=None, help="Output embeddings JSON path.")
@click.option("--faiss-dir", default=None, help="Output FAISS index directory.")
@click.option("--model",   default=None, help="SentenceTransformer model name.")
@click.option("--batch-size", type=int, default=None, help="Batch size.")
@click.option("--no-normalize", is_flag=True, default=False, help="Disable L2 normalization.")
@click.option("--cache-dir", default=None, help="Embedding cache directory.")
@click.pass_context
def embed(ctx, input, output, faiss_dir, model, batch_size, no_normalize, cache_dir):
    """Generate embeddings and index them directly into FAISS."""
    cfg = ctx.obj["config"]
    emb = cfg.get("embeddings", {})
    ing = cfg.get("ingestion", {})
    vs  = cfg.get("vectorstore", {})

    output_dir  = ing.get("output_dir", "data/output")
    input_path  = input      or str(Path(output_dir) / ing.get("chunks_file", "chunks.json"))
    output_path = output     or emb.get("output_file", "data/output/embeddings.json")
    faiss_path  = faiss_dir  or vs.get("index_dir", "data/indices/faiss")
    model_name  = model      or emb.get("model", "BAAI/bge-small-en-v1.5")
    bs          = batch_size or emb.get("batch_size", 32)
    normalize   = not no_normalize and emb.get("normalize", True)
    cache       = cache_dir  or emb.get("cache_dir")

    click.echo(f"\n[EMBED] Generating embeddings from: {input_path}")
    click.echo(f"   Model:        {model_name}")
    click.echo(f"   Normalize:    {normalize}")
    click.echo(f"   Batch Size:   {bs}\n")

    from rag_system.ingestion.chunker import DocumentChunker
    from rag_system.embeddings.embedder import Embedder
    from rag_system.vectorstore.faiss_store import FAISSStore

    chunks = DocumentChunker.load(input_path)
    embedder = Embedder(
        model_name=model_name,
        batch_size=bs,
        normalize=normalize,
        cache_dir=cache,
    )
    embedded = embedder.embed(chunks)
    embedder.save(embedded, output_path)

    # Index directly into FAISS Store
    click.echo(f"\nIndexing {len(embedded)} vectors into FAISS Vector Store...")
    dim = embedded[0]["embedding_dimensions"] if embedded else 384
    store = FAISSStore(dimension=dim)
    store.add_batch(embedded)
    store.save(faiss_path)

    click.echo(f"[SUCCESS] FAISS Store successfully saved to: {faiss_path}")
    click.echo(f"[SUCCESS] JSON embeddings saved to: {output_path}")


# ─── validate ────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--eval-dataset",  default=None, help="Eval dataset JSON path.")
@click.option("--embeddings",    default=None, help="Embeddings JSON path.")
@click.option("--output",        default="data/reports/evaluation/evaluation_validation_report.json",
              show_default=True, help="Validation report output path.")
@click.option("--model",         default=None, help="Encoder model.")
@click.option("--correct",       is_flag=True, default=False,
              help="Also generate a corrected eval dataset using similarity-based annotation.")
@click.option("--corrected-output", default="rag_system/eval/eval_dataset.json",
              show_default=True, help="Path for corrected eval dataset.")
@click.option("--top-k", type=int, default=5, help="Relevant chunks to suggest per query.")
@click.pass_context
def validate(ctx, eval_dataset, embeddings, output, model, correct, corrected_output, top_k):
    """Validate eval dataset: checks missing IDs, semantic match, and irrelevant front matter."""
    cfg = ctx.obj["config"]
    ev = cfg.get("evaluation", {})
    emb = cfg.get("embeddings", {})

    eval_path = eval_dataset or ev.get("eval_dataset", "rag_system/eval/eval_dataset.json")
    emb_path  = embeddings   or emb.get("output_file", "data/output/embeddings.json")
    model_name = model       or emb.get("model", "BAAI/bge-small-en-v1.5")

    click.echo(f"\n[VALIDATE] Validating evaluation dataset: {eval_path}")
    click.echo(f"   Against embeddings: {emb_path}\n")

    from rag_system.evaluation.validator import EvalDatasetValidator

    validator = EvalDatasetValidator(
        embeddings_path=emb_path,
        model_name=model_name,
    )
    report = validator.validate(eval_path)
    validator.save_report(report, output)

    click.echo(f"Validation Summary:")
    click.echo(f"   Total queries:                  {report.total_queries}")
    click.echo(f"   Valid queries:                  {report.valid_queries}")
    click.echo(f"   Queries with missing IDs:       {report.queries_with_missing_ids}")
    click.echo(f"   Queries with low similarity:    {report.queries_with_low_similarity}")
    click.echo(f"   Queries with irrelevant chunks: {report.queries_with_irrelevant_chunks}")
    click.echo(f"   Coverage score:                 {report.coverage_score:.2%}")
    
    if report.missing_chunk_ids:
        click.echo(f"   Warning: Missing Chunk IDs: {sorted(set(report.missing_chunk_ids))[:10]}")

    # Flag irrelevant chunks found
    for qv in report.per_query:
        if qv.irrelevant_chunks:
            for ic in qv.irrelevant_chunks:
                click.echo(f"   Warning: Irrelevant Chunk: {ic.chunk_id} on page {ic.page} (Reason: {ic.reason})")

    click.echo(f"\n[SUCCESS] Validation report saved -> {output}")

    if correct:
        click.echo(f"\n[CORRECT] Generating corrected eval dataset (top_k={top_k})...")
        out = validator.generate_corrected_dataset(eval_path, corrected_output, top_k=top_k)
        click.echo(f"[SUCCESS] Corrected dataset saved -> {out}")


# ─── evaluate ────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--eval-dataset", default=None, help="Eval dataset JSON path.")
@click.option("--faiss-dir",   default=None, help="FAISS Index directory.")
@click.option("--model",        default=None, help="SentenceTransformer model.")
@click.option("--top-k",   type=int,   default=None, help="Top-K for retrieval.")
@click.option("--alpha",   type=float, default=None, help="Hybrid alpha weight (0=BM25, 1=semantic).")
@click.option("--no-hybrid", is_flag=True, default=False, help="Disable BM25 (pure semantic).")
@click.option("--output-dir",   default=None, help="Report output directory.")
@click.pass_context
def evaluate(ctx, eval_dataset, faiss_dir, model, top_k, alpha, no_hybrid, output_dir):
    """Run evaluation and generate JSON + Markdown reports from FAISS."""
    cfg = ctx.obj["config"]
    ev  = cfg.get("evaluation", {})
    emb = cfg.get("embeddings", {})
    ret = cfg.get("retrieval",  {})
    vs  = cfg.get("vectorstore", {})
    hyb = ret.get("hybrid", {})

    eval_path  = eval_dataset or ev.get("eval_dataset",  "rag_system/eval/eval_dataset.json")
    faiss_path = faiss_dir    or vs.get("index_dir",     "data/indices/faiss")
    model_name = model        or emb.get("model",        "BAAI/bge-small-en-v1.5")
    k          = top_k        or ev.get("top_k",         10)
    a          = alpha        or hyb.get("alpha",        0.7)
    out_dir    = output_dir   or ev.get("output_dir",    "data/reports/evaluation")
    use_hybrid = not no_hybrid and hyb.get("enabled",    True)

    click.echo(f"\n[EVALUATE] Starting RAG Evaluation")
    click.echo(f"   Dataset:     {eval_path}")
    click.echo(f"   FAISS Index: {faiss_path}")
    click.echo(f"   Model:       {model_name}")
    click.echo(f"   Top-K:       {k}")
    click.echo(f"   Hybrid:      {use_hybrid} (alpha={a})\n")

    from rag_system.evaluation.evaluator import RAGEvaluator
    from rag_system.evaluation.reports import ReportGenerator

    fusion_type = hyb.get("fusion_type", "rrf")
    rrf_k       = hyb.get("rrf_k", 60)

    reranker_cfg = ret.get("reranker", {})
    reranker_obj = None
    if reranker_cfg.get("enabled", False):
        rmodel = reranker_cfg.get("model", "BAAI/bge-reranker-large")
        from rag_system.retrieval.reranker import get_reranker
        reranker_obj = get_reranker(model_name=rmodel)

    expander_cfg = ret.get("query_expansion", {})
    expander_obj = None
    expansion_method = "none"
    if expander_cfg.get("enabled", False):
        expansion_method = expander_cfg.get("method", "hyde")
        from rag_system.llm.generator import LLMGenerator
        from rag_system.retrieval.expansion import QueryExpander
        try:
            llm_gen = LLMGenerator()
            expander_obj = QueryExpander(generator=llm_gen)
        except Exception as e:
            click.echo(f"Warning: Failed to initialize LLM generator for query expansion: {e}", err=True)

    use_parent_child = ret.get("use_parent_child", False)
    retrieval_pool_multiplier = ret.get("retrieval_pool_multiplier", 10)
    use_query_expansion = ret.get("use_query_expansion", False)
    query_variants = ret.get("query_variants", 4)
    use_hyde = ret.get("use_hyde", False)
    use_metadata_filtering = ret.get("use_metadata_filtering", True)

    evaluator = RAGEvaluator(
        vector_store_path=faiss_path,
        eval_dataset_path=eval_path,
        model_name=model_name,
        top_k=k,
        alpha=a,
        hybrid=use_hybrid,
        fusion_type=fusion_type,
        rrf_k=rrf_k,
        reranker=reranker_obj,
        expander=expander_obj,
        expansion_method=expansion_method,
        use_parent_child=use_parent_child,
        retrieval_pool_multiplier=retrieval_pool_multiplier,
        use_query_expansion=use_query_expansion,
        query_variants=query_variants,
        use_hyde=use_hyde,
        use_metadata_filtering=use_metadata_filtering,
    )

    results = evaluator.evaluate()

    report_gen = ReportGenerator(output_dir=out_dir)
    paths = report_gen.generate_all(results)

    agg = results.aggregated
    click.echo("\n" + "=" * 54)
    click.echo("  EVALUATION RESULTS")
    click.echo("=" * 54)
    click.echo(f"  Total Queries:      {results.total_queries}")
    click.echo(f"  NDCG@{k}:           {agg.avg_ndcg:.4f}")
    click.echo(f"  Precision@{k}:      {agg.avg_precision_at_k:.4f}")
    click.echo(f"  Recall@{k}:         {agg.avg_recall_at_k:.4f}")
    click.echo(f"  Hit Rate@{k}:       {agg.avg_hit_rate_at_k:.4f}")
    click.echo(f"  MRR:               {agg.avg_mrr:.4f}")
    click.echo(f"  MAP:               {agg.avg_map:.4f}")
    click.echo("=" * 54)
    click.echo(f"\n[SUCCESS] Reports:")
    click.echo(f"   JSON:     {paths['json']}")
    click.echo(f"   Markdown: {paths['markdown']}")


# ─── query ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--query",    required=True, help="Query text to search for.")
@click.option("--faiss-dir", default=None, help="FAISS Index directory.")
@click.option("--model",    default=None,  help="SentenceTransformer model.")
@click.option("--top-k",    type=int, default=5, show_default=True, help="Number of results.")
@click.option("--alpha",    type=float, default=0.7, show_default=True, help="Hybrid alpha.")
@click.option("--no-hybrid", is_flag=True, default=False, help="Disable BM25.")
@click.pass_context
def query(ctx, query, faiss_dir, model, top_k, alpha, no_hybrid):
    """Run a single retrieval query (debug / demo mode) from FAISS."""
    cfg = ctx.obj["config"]
    emb = cfg.get("embeddings", {})
    vs  = cfg.get("vectorstore", {})

    faiss_path = faiss_dir or vs.get("index_dir", "data/indices/faiss")
    model_name = model      or emb.get("model", "BAAI/bge-small-en-v1.5")

    click.echo(f"\n[QUERY] Loading FAISS Vector Store from: {faiss_path}")
    from rag_system.vectorstore.faiss_store import FAISSStore
    from rag_system.retrieval.retriever import HybridRetriever, BM25Index

    # Load FAISS
    metadata_file = os.path.join(faiss_path, "metadata.json")
    dimension = 384
    if os.path.exists(metadata_file):
        import json
        with open(metadata_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
            dimension = meta.get("dimension", 384)

    store = FAISSStore(dimension=dimension)
    store.load(faiss_path)
    chunks = list(store.metadata.values())

    # Build BM25
    bm25 = BM25Index(chunks) if not no_hybrid else None

    ret = cfg.get("retrieval", {})
    hyb = ret.get("hybrid", {})
    fusion_type = hyb.get("fusion_type", "rrf")
    rrf_k       = hyb.get("rrf_k", 60)

    reranker_cfg = ret.get("reranker", {})
    reranker_obj = None
    if reranker_cfg.get("enabled", False):
        rmodel = reranker_cfg.get("model", "BAAI/bge-reranker-large")
        from rag_system.retrieval.reranker import get_reranker
        reranker_obj = get_reranker(model_name=rmodel)

    expander_cfg = ret.get("query_expansion", {})
    expander_obj = None
    expansion_method = "none"
    if expander_cfg.get("enabled", False):
        expansion_method = expander_cfg.get("method", "hyde")
        from rag_system.llm.generator import LLMGenerator
        from rag_system.retrieval.expansion import QueryExpander
        try:
            llm_gen = LLMGenerator()
            expander_obj = QueryExpander(generator=llm_gen)
        except Exception as e:
            click.echo(f"Warning: Failed to initialize LLM generator for query expansion: {e}", err=True)

    use_parent_child = ret.get("use_parent_child", False)
    retrieval_pool_multiplier = ret.get("retrieval_pool_multiplier", 10)
    use_query_expansion = ret.get("use_query_expansion", False)
    query_variants = ret.get("query_variants", 4)
    use_hyde = ret.get("use_hyde", False)
    use_metadata_filtering = ret.get("use_metadata_filtering", True)

    retriever = HybridRetriever(
        vector_store=store,
        bm25_index=bm25,
        model_name=model_name,
        alpha=1.0 if no_hybrid else alpha,
        fusion_type=fusion_type,
        rrf_k=rrf_k,
        reranker=reranker_obj,
        expander=expander_obj,
        expansion_method=expansion_method,
        use_parent_child=use_parent_child,
        retrieval_pool_multiplier=retrieval_pool_multiplier,
        use_query_expansion=use_query_expansion,
        query_variants=query_variants,
        use_hyde=use_hyde,
        use_metadata_filtering=use_metadata_filtering,
    )


    click.echo(f'\n[QUERY] Query: "{query}"\n')
    results = retriever.retrieve(query, top_k=top_k)

    for r in results:
        click.echo(
            f"  [{r.rank:2d}] score={r.score:.4f} "
            f"(sem={r.semantic_score:.3f} bm25={r.bm25_score:.3f}) "
            f"id={r.chunk_id!r} (page={r.page})"
        )
        preview = r.content[:120].replace("\n", " ")
        click.echo(f"       {preview}...")
        click.echo()


# ─── benchmark ───────────────────────────────────────────────────────────────

@cli.command()
@click.option("--pdf", default="data/row/andrew-ng-machine-learning-yearning.pdf", help="Path to PDF.")
@click.option("--eval-dataset", default="rag_system/eval/eval_dataset.json", help="Eval dataset JSON.")
@click.option("--output-dir", default="data/reports/benchmark", help="Output report directory.")
@click.pass_context
def benchmark(ctx, pdf, eval_dataset, output_dir):
    """Compare chunking layouts, embedding models, and hybrid parameters."""
    click.echo(f"\n[BENCHMARK] Starting RAG Evaluation Benchmark Suite")
    click.echo(f"   PDF:          {pdf}")
    click.echo(f"   Eval Dataset: {eval_dataset}")
    click.echo(f"   Output Dir:   {output_dir}\n")

    from rag_system.benchmark.benchmarker import RAGBenchmarker

    benchmarker = RAGBenchmarker(
        pdf_path=pdf,
        eval_dataset_path=eval_dataset,
        output_dir=output_dir,
    )
    benchmarker.run_all()

    click.echo(f"\n[SUCCESS] Benchmark Suite complete! Report saved to: {output_dir}/benchmark_report.md")


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
