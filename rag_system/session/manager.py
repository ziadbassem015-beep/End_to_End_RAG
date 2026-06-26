"""
rag_system/session/manager.py
=============================
Session management business logic.
Handles creation, deletion, switching, and persistence of sessions.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag_system.session.models import (
    Session,
    Message,
    DocumentMetadata,
    MemorySummary,
    MemoryEmbedding,
)
from rag_system.session.storage import SessionStorage, SessionIndexStorage

logger = logging.getLogger(__name__)


class SessionManager:
    """
    High-level session management.
    Coordinates session lifecycle, document management, and memory.
    """

    def __init__(
        self,
        storage_dir: Path | str = "data/sessions",
    ):
        """
        Initialize session manager.
        
        Args:
            storage_dir: Root directory for session storage
        """
        self.storage = SessionStorage(storage_dir)
        self.index_storage = SessionIndexStorage(storage_dir)
        self.active_session_id: Optional[str] = None
        self._session_cache: Dict[str, Session] = {}
        logger.info("SessionManager initialized")

    # ─── Session Lifecycle ────────────────────────────────────────────────────

    def create_session(
        self,
        name: str,
        description: Optional[str] = None,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ) -> Session:
        """
        Create a new session.
        
        Args:
            name: Session name
            description: Optional description
            embedding_model: Embedding model to use
            chunk_size: Document chunk size
            chunk_overlap: Chunk overlap
            
        Returns:
            Created session object
        """
        session = Session(
            name=name,
            description=description,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        # Ensure directories
        self.index_storage.get_faiss_index_dir(session.session_id)
        self.index_storage.get_bm25_cache_dir(session.session_id)
        self.index_storage.get_memory_index_dir(session.session_id)
        self.index_storage.get_documents_dir(session.session_id)
        
        # Save to storage
        self.storage.save_session(session)
        self._session_cache[session.session_id] = session
        
        logger.info("Session created: %s (%s)", session.session_id, name)
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get a session by ID (with caching).
        
        Args:
            session_id: Session ID
            
        Returns:
            Session object or None
        """
        if session_id in self._session_cache:
            return self._session_cache[session_id]
        
        session = self.storage.load_session(session_id)
        if session:
            self._session_cache[session_id] = session
        
        return session

    def update_session(
        self,
        session_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Optional[Session]:
        """
        Update session metadata.
        
        Args:
            session_id: Session ID
            name: New name (optional)
            description: New description (optional)
            settings: Updated settings dict (optional)
            
        Returns:
            Updated session or None if not found
        """
        session = self.get_session(session_id)
        if not session:
            return None
        
        if name:
            session.name = name
        if description is not None:
            session.description = description
        if settings:
            session.settings.update(settings)
        
        session.updated_at = datetime.utcnow()
        self.storage.save_session(session)
        
        logger.info("Session updated: %s", session_id)
        return session

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all its data.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if deleted successfully
        """
        # Remove from cache
        self._session_cache.pop(session_id, None)
        
        # Clear active session if it was deleted
        if self.active_session_id == session_id:
            self.active_session_id = None
        
        # Delete from storage
        result = self.storage.delete_session(session_id)
        logger.info("Session deleted: %s", session_id)
        return result

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all sessions with summaries.
        
        Returns:
            List of session info
        """
        return self.storage.list_sessions()

    # ─── Session Switching ────────────────────────────────────────────────────

    def set_active_session(self, session_id: str) -> bool:
        """
        Switch to an active session.
        
        Args:
            session_id: Session ID to activate
            
        Returns:
            True if session exists and was activated
        """
        session = self.get_session(session_id)
        if not session:
            logger.warning("Cannot activate non-existent session: %s", session_id)
            return False
        
        self.active_session_id = session_id
        session.last_accessed = datetime.utcnow()
        self.storage.save_session(session)
        
        logger.info("Active session switched to: %s", session_id)
        return True

    def get_active_session(self) -> Optional[Session]:
        """Get the currently active session."""
        if not self.active_session_id:
            return None
        return self.get_session(self.active_session_id)

    # ─── Document Management ──────────────────────────────────────────────────

    def add_document_to_session(
        self,
        session_id: str,
        filename: str,
        source: str,
        embedding_model: str,
        file_size: int = 0,
        page_count: Optional[int] = None,
    ) -> DocumentMetadata:
        """
        Register a document in a session.
        
        Args:
            session_id: Session ID
            filename: Original filename
            source: Source path/identifier
            embedding_model: Model used for embeddings
            file_size: File size in bytes
            page_count: Page count if applicable
            
        Returns:
            Document metadata
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        doc = DocumentMetadata(
            filename=filename,
            source=source,
            embedding_model=embedding_model,
            file_size=file_size,
            page_count=page_count,
            upload_date=datetime.utcnow(),
        )
        
        session.documents.append(doc)
        self.storage.save_session(session)
        
        logger.info("Document added to session %s: %s", session_id, filename)
        return doc

    def get_session_documents(self, session_id: str) -> List[DocumentMetadata]:
        """Get all documents in a session."""
        session = self.get_session(session_id)
        if not session:
            return []
        return session.documents

    def remove_document_from_session(
        self, session_id: str, document_id: str
    ) -> bool:
        """
        Remove a document from a session.
        
        Args:
            session_id: Session ID
            document_id: Document ID to remove
            
        Returns:
            True if removed
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        initial_count = len(session.documents)
        session.documents = [d for d in session.documents if d.document_id != document_id]
        
        if len(session.documents) < initial_count:
            self.storage.save_session(session)
            logger.info("Document removed from session %s: %s", session_id, document_id)
            return True
        
        return False

    # ─── Chat History Management ──────────────────────────────────────────────

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        source_chunks: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """
        Add a message to session chat history.
        
        Args:
            session_id: Session ID
            role: Message role ('user' or 'assistant')
            content: Message content
            source_chunks: Retrieved chunks if applicable
            metadata: Additional metadata
            
        Returns:
            Created message object
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        message = Message(
            role=role,
            content=content,
            source_chunks=source_chunks,
            metadata=metadata or {},
        )
        
        session.chat_history.append(message)
        self.storage.save_session(session)
        
        return message

    def get_chat_history(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[Message]:
        """
        Get chat history for a session.
        
        Args:
            session_id: Session ID
            limit: Max messages to return (most recent)
            
        Returns:
            List of messages
        """
        return self.storage.get_chat_history(session_id, limit)

    def clear_chat_history(self, session_id: str) -> bool:
        """
        Clear all chat history and memory for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if successful
        """
        try:
            self.storage.clear_chat_history(session_id)
            logger.info("Chat history cleared for session %s", session_id)
            return True
        except Exception as e:
            logger.error("Failed to clear chat history: %s", e)
            return False

    # ─── Memory Management ────────────────────────────────────────────────────

    def save_memory_summary(
        self, session_id: str, summary: MemorySummary
    ) -> None:
        """Save a memory summary."""
        self.storage.save_memory_summary(session_id, summary)
        # Clear cache so next load gets updated data
        self._session_cache.pop(session_id, None)

    def save_memory_embedding(
        self, session_id: str, embedding: MemoryEmbedding
    ) -> None:
        """Save a memory embedding."""
        self.storage.save_memory_embedding(session_id, embedding)
        # Clear cache
        self._session_cache.pop(session_id, None)

    def get_memory_summaries(self, session_id: str) -> List[MemorySummary]:
        """Get all memory summaries for a session."""
        return self.storage.get_memory_summaries(session_id)

    def get_memory_embeddings(self, session_id: str) -> List[MemoryEmbedding]:
        """Get all memory embeddings for a session."""
        return self.storage.get_memory_embeddings(session_id)

    # ─── Utilities ────────────────────────────────────────────────────────────

    def export_session(self, session_id: str, export_path: Path | str) -> bool:
        """Export a session for backup/sharing."""
        return self.storage.export_session(session_id, export_path)

    def get_session_storage_info(self, session_id: str) -> Dict[str, Any]:
        """
        Get storage information about a session.
        
        Returns:
            Dictionary with storage stats
        """
        session = self.get_session(session_id)
        if not session:
            return {}
        
        size_bytes = self.storage.get_storage_size(session_id)
        
        return {
            "session_id": session_id,
            "name": session.name,
            "total_size_bytes": size_bytes,
            "total_size_mb": round(size_bytes / (1024 * 1024), 2),
            "document_count": len(session.documents),
            "message_count": len(session.chat_history),
            "memory_summaries": len(session.memory_summaries),
            "memory_embeddings": len(session.memory_embeddings),
        }

    def clear_session_indices(self, session_id: str) -> bool:
        """Clear all indices (FAISS, BM25, memory) for a session."""
        return self.index_storage.clear_session_indices(session_id)

    def get_index_directories(self, session_id: str) -> Dict[str, Path]:
        """Get all index directories for a session."""
        return {
            "faiss": self.index_storage.get_faiss_index_dir(session_id),
            "bm25": self.index_storage.get_bm25_cache_dir(session_id),
            "memory": self.index_storage.get_memory_index_dir(session_id),
            "documents": self.index_storage.get_documents_dir(session_id),
            "uploads": self.index_storage.get_uploads_dir(session_id),
        }
