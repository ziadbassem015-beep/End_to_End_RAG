"""
rag_system/api/state.py
=======================
In-memory cache manager for FAISS stores, BM25 indices, and retrievers.
Also tracks progress of background ingestion tasks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from rag_system.vectorstore.faiss_store import FAISSStore
from rag_system.retrieval.retriever import BM25Index, HybridRetriever
from rag_system.retrieval.reranker import CrossEncoderReranker, BGEReranker
from rag_system.llm.generator import LLMGenerator

logger = logging.getLogger(__name__)


class APIStateManager:
    """
    State manager for caching heavy models/stores and tracking background tasks.
    """

    def __init__(self) -> None:
        self.retrievers: Dict[str, HybridRetriever] = {}  # key: book_name + retriever_config_hash
        self.generators: Dict[str, LLMGenerator] = {}  # key: api_key + provider + model
        self.tasks: Dict[str, Dict[str, Any]] = {}  # key: task_id, value: {status, progress, book_name, error}

    def get_retriever(
        self,
        book_name: str,
        index_dir: Path,
        embedding_model: str,
        alpha: float = 0.7,
        use_reranker: bool = True,
        reranker_model: str = "Cross-Encoder (ms-marco-MiniLM)"
    ) -> HybridRetriever:
        """
        Get or load the hybrid retriever for a given book.
        Caches in memory to avoid reloading indices from disk.
        """
        cache_key = f"{book_name}||{embedding_model}||{alpha}||{use_reranker}||{reranker_model}"
        if cache_key in self.retrievers:
            logger.info("StateManager: Cache hit for retriever key: %s", cache_key)
            return self.retrievers[cache_key]

        logger.info("StateManager: Loading vector store and BM25 index for %s", book_name)
        
        # Load FAISS Store
        import json
        metadata_file = index_dir / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Index metadata missing for {book_name}. Please re-index.")
            
        with open(metadata_file, "r", encoding="utf-8") as f:
            state = json.load(f)
        dim = state.get("dimension", 384)

        vector_store = FAISSStore(dimension=dim)
        vector_store.load(str(index_dir))

        # Reconstruct BM25 index
        chunks = list(vector_store.metadata.values())
        bm25_index = BM25Index(chunks)

        # Initialize Reranker if requested
        reranker_obj = None
        if use_reranker:
            if "cross" in reranker_model.lower():
                reranker_obj = CrossEncoderReranker()
            else:
                reranker_obj = BGEReranker()

        # Build Hybrid Retriever
        retriever = HybridRetriever(
            vector_store=vector_store,
            bm25_index=bm25_index,
            model_name=embedding_model,
            alpha=alpha,
            reranker=reranker_obj
        )

        self.retrievers[cache_key] = retriever
        return retriever

    def get_generator(
        self,
        api_key: str,
        provider: str,
        model_name: str
    ) -> LLMGenerator:
        """Get or load LLM Generator."""
        cache_key = f"{api_key[-6:]}||{provider}||{model_name}"
        if cache_key in self.generators:
            return self.generators[cache_key]

        generator = LLMGenerator(api_key=api_key, provider=provider, model_name=model_name)
        self.generators[cache_key] = generator
        return generator

    def set_task(self, task_id: str, book_name: str, status: str = "running", progress: int = 0, error: Optional[str] = None) -> None:
        """Update background task ingestion status."""
        self.tasks[task_id] = {
            "status": status,
            "progress": progress,
            "book_name": book_name,
            "error": error
        }

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve task details."""
        return self.tasks.get(task_id)

    def clear_retriever_cache(self) -> None:
        """Clear all loaded retrievers."""
        self.retrievers.clear()
        logger.info("StateManager: Retriever cache cleared.")


# Global Singleton Instance
state_manager = APIStateManager()
