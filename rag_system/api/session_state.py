"""
rag_system/api/session_state.py
================================
Session-aware state management for the API.
Extends the basic state manager with session support.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from rag_system.api.state import APIStateManager
from rag_system.session.manager import SessionManager
from rag_system.memory.manager import MemoryManager
from rag_system.retrieval.session_retriever import SessionHybridRetriever

logger = logging.getLogger(__name__)


class SessionAwareStateManager(APIStateManager):
    """
    Extended state manager with session support.
    Manages sessions, session-scoped retrievers, and memory managers.
    """

    def __init__(
        self,
        storage_dir: Path | str = "data/sessions",
        cache_dir: Path | str = "data/cache/embeddings",
    ):
        """
        Initialize session-aware state manager.
        
        Args:
            storage_dir: Base directory for session storage
            cache_dir: Cache directory for embeddings
        """
        super().__init__()
        
        self.storage_dir = Path(storage_dir)
        self.cache_dir = Path(cache_dir)
        
        # Session management
        self.session_manager = SessionManager(storage_dir)
        
        # Session-scoped resources (session_id -> resource)
        self.session_retrievers: Dict[str, SessionHybridRetriever] = {}
        self.session_memory_managers: Dict[str, MemoryManager] = {}
        
        logger.info("SessionAwareStateManager initialized")

    # ─── Session Management Delegation ────────────────────────────────────────

    def create_session(
        self,
        name: str,
        description: Optional[str] = None,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ) -> Dict[str, Any]:
        """Create a new session."""
        session = self.session_manager.create_session(
            name=name,
            description=description,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return {
            "session_id": session.session_id,
            "name": session.name,
            "created_at": session.created_at.isoformat(),
        }

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session details."""
        session = self.session_manager.get_session(session_id)
        if not session:
            return None
        
        return {
            "session_id": session.session_id,
            "name": session.name,
            "description": session.description,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "document_count": len(session.documents),
            "message_count": len(session.chat_history),
        }

    def list_sessions(self) -> list:
        """List all sessions."""
        return self.session_manager.list_sessions()

    def set_active_session(self, session_id: str) -> bool:
        """Set active session."""
        return self.session_manager.set_active_session(session_id)

    def get_active_session_id(self) -> Optional[str]:
        """Get currently active session ID."""
        return self.session_manager.active_session_id

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        # Clean up cached resources
        self.session_retrievers.pop(session_id, None)
        self.session_memory_managers.pop(session_id, None)
        
        # Delete from storage
        return self.session_manager.delete_session(session_id)

    # ─── Session-Scoped Retriever Management ──────────────────────────────────

    def get_session_retriever(
        self,
        session_id: str,
        embedding_model: str,
        alpha: float = 0.7,
        use_reranker: bool = True,
        reranker_model: str = "Cross-Encoder (ms-marco-MiniLM)",
    ) -> Optional[SessionHybridRetriever]:
        """
        Get or create session-scoped retriever.
        
        Args:
            session_id: Session ID
            embedding_model: Embedding model
            alpha: Dense search weight
            use_reranker: Whether to use reranking
            reranker_model: Reranker model name
            
        Returns:
            SessionHybridRetriever or None
        """
        cache_key = f"{session_id}||{embedding_model}||{alpha}||{use_reranker}||{reranker_model}"
        
        if cache_key in self.session_retrievers:
            logger.info("Cache hit for session retriever: %s", session_id)
            return self.session_retrievers[cache_key]
        
        try:
            # Get session
            session = self.session_manager.get_session(session_id)
            if not session:
                logger.warning("Session not found: %s", session_id)
                return None
            
            # Get memory manager
            memory_manager = self.get_memory_manager(session_id)
            
            # Get FAISS index directory
            index_dirs = self.session_manager.index_storage.get_session_index_dir(session_id)
            faiss_dir = self.session_manager.index_storage.get_faiss_index_dir(session_id)
            
            # Create session retriever
            retriever = SessionHybridRetriever(
                session_id=session_id,
                faiss_index_dir=faiss_dir,
                memory_manager=memory_manager,
                embedding_model=embedding_model,
                alpha=alpha,
                use_reranker=use_reranker,
                reranker_model=reranker_model,
            )
            
            self.session_retrievers[cache_key] = retriever
            logger.info("Session retriever created for: %s", session_id)
            return retriever
        except Exception as e:
            logger.error("Failed to create session retriever: %s", e, exc_info=True)
            return None

    # ─── Session-Scoped Memory Management ──────────────────────────────────────

    def get_memory_manager(self, session_id: str) -> MemoryManager:
        """
        Get or create memory manager for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            MemoryManager instance
        """
        if session_id in self.session_memory_managers:
            return self.session_memory_managers[session_id]
        
        try:
            session = self.session_manager.get_session(session_id)
            if not session:
                raise ValueError(f"Session not found: {session_id}")
            
            memory_index_dir = (
                self.session_manager.index_storage.get_memory_index_dir(session_id)
            )
            
            memory_manager = MemoryManager(
                session_id=session_id,
                memory_index_dir=memory_index_dir,
                embedding_model=session.embedding_model,
            )
            
            self.session_memory_managers[session_id] = memory_manager
            logger.info("Memory manager created for session: %s", session_id)
            return memory_manager
        except Exception as e:
            logger.error("Failed to create memory manager: %s", e)
            raise

    def clear_session_retriever_cache(self, session_id: str) -> None:
        """Clear retriever cache for a session."""
        to_remove = [
            k for k in self.session_retrievers.keys() if k.startswith(session_id)
        ]
        for k in to_remove:
            del self.session_retrievers[k]
        logger.info("Session retriever cache cleared for: %s", session_id)

    # ─── Utilities ────────────────────────────────────────────────────────────

    def add_message_to_session(
        self,
        session_id: str,
        role: str,
        content: str,
        source_chunks: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add a message to session chat history."""
        message = self.session_manager.add_message(
            session_id=session_id,
            role=role,
            content=content,
            source_chunks=source_chunks,
            metadata=metadata,
        )
        
        # Add to semantic memory
        memory_manager = self.get_memory_manager(session_id)
        memory_manager.add_to_semantic_memory(message)
        
        return {
            "message_id": message.message_id,
            "role": message.role,
            "timestamp": message.timestamp.isoformat(),
        }

    def get_session_chat_history(self, session_id: str, limit: Optional[int] = None) -> list:
        """Get chat history for a session."""
        messages = self.session_manager.get_chat_history(session_id, limit)
        return [
            {
                "message_id": m.message_id,
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp.isoformat(),
            }
            for m in messages
        ]

    def get_session_memory_context(
        self, session_id: str, query: str
    ) -> Dict[str, Any]:
        """Get memory context for a session query."""
        session = self.session_manager.get_session(session_id)
        if not session:
            return {}
        
        memory_manager = self.get_memory_manager(session_id)
        memory_context = memory_manager.build_memory_context(
            chat_history=session.chat_history,
            query=query,
        )
        
        return {
            "memory_context": memory_context,
            "message_count": len(session.chat_history),
        }
