"""
rag_system/retrieval/session_retriever.py
==========================================
Session-scoped retriever combining document and memory retrieval.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_system.retrieval.retriever import (
    BM25Index,
    HybridRetriever,
    RetrievalResult,
)
from rag_system.retrieval.reranker import CrossEncoderReranker, BGEReranker
from rag_system.session.models import Message
from rag_system.vectorstore.faiss_store import FAISSStore
from rag_system.memory.manager import MemoryManager

logger = logging.getLogger(__name__)


@dataclass
class SessionRetrievalResult:
    """Result combining document and memory retrieval."""

    chunk_id: str
    rank: int
    score: float
    content: str
    source: str
    page: int
    section: str
    result_type: str  # "document" or "memory"
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "rank": self.rank,
            "score": self.score,
            "content": self.content,
            "source": self.source,
            "page": self.page,
            "section": self.section,
            "result_type": self.result_type,
            "semantic_score": self.semantic_score,
            "bm25_score": self.bm25_score,
            "rerank_score": self.rerank_score,
        }


class SessionHybridRetriever:
    """
    Session-scoped retriever that combines:
    1. Document retrieval (FAISS + BM25)
    2. Memory retrieval (semantic memories)
    3. Context-aware ranking
    """

    def __init__(
        self,
        session_id: str,
        faiss_index_dir: Path | str,
        memory_manager: MemoryManager,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        alpha: float = 0.7,
        use_reranker: bool = True,
        reranker_model: str = "Cross-Encoder (ms-marco-MiniLM)",
        memory_weight: float = 0.3,
    ):
        """
        Initialize session-scoped retriever.
        
        Args:
            session_id: Session ID
            faiss_index_dir: Path to session FAISS index
            memory_manager: MemoryManager instance for session
            embedding_model: Embedding model name
            alpha: Weight for semantic vs BM25 search
            use_reranker: Whether to use reranking
            reranker_model: Reranker model to use
            memory_weight: Weight for memory results in final ranking
        """
        self.session_id = session_id
        self.faiss_index_dir = Path(faiss_index_dir)
        self.memory_manager = memory_manager
        self.embedding_model = embedding_model
        self.alpha = alpha
        self.use_reranker = use_reranker
        self.reranker_model = reranker_model
        self.memory_weight = memory_weight
        
        # Load document retriever
        self.hybrid_retriever: Optional[HybridRetriever] = None
        self._initialize_document_retriever()
        
        logger.info(
            "SessionHybridRetriever initialized for session %s", session_id
        )

    def _initialize_document_retriever(self) -> None:
        """Load or create document-scoped hybrid retriever."""
        try:
            if not (self.faiss_index_dir / "index.faiss").exists():
                logger.warning(
                    "FAISS index not found for session %s", self.session_id
                )
                return
            
            # Load FAISS store
            import json
            
            metadata_file = self.faiss_index_dir / "metadata.json"
            if not metadata_file.exists():
                logger.warning("Metadata not found for session %s", self.session_id)
                return
            
            with open(metadata_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            
            dim = state.get("dimension", 384)
            vector_store = FAISSStore(dimension=dim)
            vector_store.load(str(self.faiss_index_dir))
            
            # Create BM25 index
            chunks = list(vector_store.metadata.values())
            bm25_index = BM25Index(chunks)
            
            # Initialize reranker
            reranker_obj = None
            if self.use_reranker:
                if "cross" in self.reranker_model.lower():
                    reranker_obj = CrossEncoderReranker()
                else:
                    reranker_obj = BGEReranker()
            
            # Create hybrid retriever
            self.hybrid_retriever = HybridRetriever(
                vector_store=vector_store,
                bm25_index=bm25_index,
                model_name=self.embedding_model,
                alpha=self.alpha,
                reranker=reranker_obj,
            )
            
            logger.info(
                "Document retriever initialized for session %s", self.session_id
            )
        except Exception as e:
            logger.error(
                "Failed to initialize document retriever for session %s: %s",
                self.session_id,
                e,
                exc_info=True,
            )

    def retrieve_documents(
        self, query: str, top_k: int = 5
    ) -> List[SessionRetrievalResult]:
        """
        Retrieve relevant documents from session.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of retrieval results
        """
        if not self.hybrid_retriever:
            logger.warning("Hybrid retriever not initialized for session %s", self.session_id)
            return []
        
        try:
            results = self.hybrid_retriever.retrieve(query, top_k=top_k)
            
            session_results = []
            for rank, result in enumerate(results, 1):
                session_results.append(
                    SessionRetrievalResult(
                        chunk_id=result.chunk_id,
                        rank=rank,
                        score=result.score,
                        content=result.content,
                        source=result.source,
                        page=result.page,
                        section=result.section,
                        result_type="document",
                        semantic_score=result.semantic_score,
                        bm25_score=result.bm25_score,
                        rerank_score=result.rerank_score,
                    )
                )
            
            return session_results
        except Exception as e:
            logger.error(
                "Failed to retrieve documents for session %s: %s",
                self.session_id,
                e,
                exc_info=True,
            )
            return []

    def retrieve_memories(
        self, query: str, top_k: int = 3
    ) -> List[SessionRetrievalResult]:
        """
        Retrieve relevant memories from session.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of memory retrieval results
        """
        if not self.memory_manager:
            return []
        
        try:
            memory_results = self.memory_manager.retrieve_semantic_memories(
                query, top_k=top_k
            )
            
            session_results = []
            for rank, result in enumerate(memory_results, 1):
                session_results.append(
                    SessionRetrievalResult(
                        chunk_id=result["message_id"],
                        rank=rank,
                        score=result["score"],
                        content=result["content"],
                        source=f"memory_{result['role']}",
                        page=0,
                        section="memory",
                        result_type="memory",
                    )
                )
            
            return session_results
        except Exception as e:
            logger.error("Failed to retrieve memories for session %s: %s", self.session_id, e)
            return []

    def retrieve_combined(
        self,
        query: str,
        top_k_documents: int = 5,
        top_k_memories: int = 3,
    ) -> List[SessionRetrievalResult]:
        """
        Retrieve and combine documents and memories.
        
        Args:
            query: Query text
            top_k_documents: Number of document results
            top_k_memories: Number of memory results
            
        Returns:
            Combined and ranked results
        """
        # Retrieve from both sources
        doc_results = self.retrieve_documents(query, top_k=top_k_documents)
        memory_results = self.retrieve_memories(query, top_k=top_k_memories)
        
        # Combine and re-rank
        all_results = doc_results + memory_results
        
        # Weight memory results slightly higher due to conversation context
        for result in all_results:
            if result.result_type == "memory":
                result.score = result.score * (1.0 + self.memory_weight)
        
        # Sort by score
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        # Re-rank
        final_results = []
        for rank, result in enumerate(all_results, 1):
            result.rank = rank
            final_results.append(result)
        
        # Return top results
        return final_results[: top_k_documents + top_k_memories]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        include_memory: bool = True,
    ) -> List[SessionRetrievalResult]:
        """
        Main retrieve method for session.
        
        Args:
            query: Query text
            top_k: Total number of results to return
            include_memory: Whether to include memory results
            
        Returns:
            Retrieved results
        """
        if include_memory:
            return self.retrieve_combined(
                query,
                top_k_documents=int(top_k * 0.7),
                top_k_memories=int(top_k * 0.3),
            )
        else:
            return self.retrieve_documents(query, top_k=top_k)
