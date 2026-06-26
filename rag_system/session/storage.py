"""
rag_system/session/storage.py
=============================
Persistent storage for sessions, documents, and memory.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag_system.session.models import Session, Message, MemorySummary, MemoryEmbedding

logger = logging.getLogger(__name__)


class SessionStorage:
    """
    Persistent storage layer for sessions.
    Stores everything on disk using JSON + binary files.
    """

    def __init__(self, base_dir: Path | str = "data/sessions"):
        """
        Initialize storage.
        
        Args:
            base_dir: Root directory for session data
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info("SessionStorage initialized at: %s", self.base_dir)

    def _get_session_dir(self, session_id: str) -> Path:
        """Get the directory for a session."""
        return self.base_dir / session_id

    def _ensure_session_dir(self, session_id: str) -> Path:
        """Ensure session directory exists."""
        session_dir = self._get_session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    # ─── Session Management ───────────────────────────────────────────────────

    def save_session(self, session: Session) -> None:
        """
        Save a session to disk.
        
        Args:
            session: Session object to save
        """
        session_dir = self._ensure_session_dir(session.session_id)
        
        # Update timestamp
        session.updated_at = datetime.utcnow()
        
        # Save main session metadata
        session_file = session_dir / "session.json"
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, indent=2, default=str)
        
        logger.info("Session saved: %s", session.session_id)

    def load_session(self, session_id: str) -> Optional[Session]:
        """
        Load a session from disk.
        
        Args:
            session_id: ID of session to load
            
        Returns:
            Session object or None if not found
        """
        session_file = self._get_session_dir(session_id) / "session.json"
        
        if not session_file.exists():
            logger.warning("Session not found: %s", session_id)
            return None
        
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            session = Session.from_dict(data)
            session.last_accessed = datetime.utcnow()
            return session
        except Exception as e:
            logger.error("Failed to load session %s: %s", session_id, e, exc_info=True)
            return None

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all its data.
        
        Args:
            session_id: ID of session to delete
            
        Returns:
            True if deleted successfully
        """
        session_dir = self._get_session_dir(session_id)
        
        if not session_dir.exists():
            logger.warning("Session directory not found: %s", session_id)
            return False
        
        try:
            import shutil
            shutil.rmtree(session_dir)
            logger.info("Session deleted: %s", session_id)
            return True
        except Exception as e:
            logger.error("Failed to delete session %s: %s", session_id, e, exc_info=True)
            return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all sessions with basic info.
        
        Returns:
            List of session summaries
        """
        sessions = []
        
        if not self.base_dir.exists():
            return sessions
        
        for session_dir in self.base_dir.iterdir():
            if not session_dir.is_dir():
                continue
            
            session = self.load_session(session_dir.name)
            if session:
                sessions.append({
                    "session_id": session.session_id,
                    "name": session.name,
                    "description": session.description,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "last_accessed": session.last_accessed.isoformat() if session.last_accessed else None,
                    "document_count": len(session.documents),
                    "message_count": len(session.chat_history),
                })
        
        # Sort by last accessed (most recent first)
        sessions.sort(
            key=lambda s: s.get("last_accessed") or s.get("updated_at"),
            reverse=True
        )
        
        return sessions

    # ─── Chat History Management ──────────────────────────────────────────────

    def append_message(self, session_id: str, message: Message) -> None:
        """
        Append a message to session chat history.
        
        Args:
            session_id: Session ID
            message: Message to append
        """
        session = self.load_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        session.chat_history.append(message)
        self.save_session(session)
        logger.info("Message added to session %s", session_id)

    def get_chat_history(self, session_id: str, limit: Optional[int] = None) -> List[Message]:
        """
        Get chat history for a session.
        
        Args:
            session_id: Session ID
            limit: Maximum number of recent messages to return
            
        Returns:
            List of messages
        """
        session = self.load_session(session_id)
        if not session:
            return []
        
        history = session.chat_history
        if limit and len(history) > limit:
            history = history[-limit:]
        
        return history

    def clear_chat_history(self, session_id: str) -> None:
        """
        Clear all messages from a session.
        
        Args:
            session_id: Session ID
        """
        session = self.load_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        session.chat_history = []
        session.memory_summaries = []
        session.memory_embeddings = []
        self.save_session(session)
        logger.info("Chat history cleared for session %s", session_id)

    # ─── Memory Management ────────────────────────────────────────────────────

    def save_memory_summary(self, session_id: str, summary: MemorySummary) -> None:
        """
        Save a memory summary.
        
        Args:
            session_id: Session ID
            summary: Memory summary to save
        """
        session = self.load_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        session.memory_summaries.append(summary)
        self.save_session(session)
        logger.info("Memory summary saved for session %s", session_id)

    def save_memory_embedding(self, session_id: str, embedding: MemoryEmbedding) -> None:
        """
        Save a memory embedding.
        
        Args:
            session_id: Session ID
            embedding: Memory embedding to save
        """
        session = self.load_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        session.memory_embeddings.append(embedding)
        self.save_session(session)

    def get_memory_summaries(self, session_id: str) -> List[MemorySummary]:
        """Get all memory summaries for a session."""
        session = self.load_session(session_id)
        if not session:
            return []
        return session.memory_summaries

    def get_memory_embeddings(self, session_id: str) -> List[MemoryEmbedding]:
        """Get all memory embeddings for a session."""
        session = self.load_session(session_id)
        if not session:
            return []
        return session.memory_embeddings

    # ─── Utility Methods ──────────────────────────────────────────────────────

    def export_session(self, session_id: str, export_path: Path | str) -> bool:
        """
        Export session as a JSON file for backup/sharing.
        
        Args:
            session_id: Session ID
            export_path: Path to export to
            
        Returns:
            True if successful
        """
        session = self.load_session(session_id)
        if not session:
            return False
        
        try:
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, indent=2, default=str)
            
            logger.info("Session exported: %s -> %s", session_id, export_path)
            return True
        except Exception as e:
            logger.error("Failed to export session %s: %s", session_id, e, exc_info=True)
            return False

    def get_storage_size(self, session_id: str) -> int:
        """
        Get total storage size of a session in bytes.
        
        Args:
            session_id: Session ID
            
        Returns:
            Size in bytes
        """
        session_dir = self._get_session_dir(session_id)
        
        if not session_dir.exists():
            return 0
        
        total_size = 0
        for file_path in session_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size


class SessionIndexStorage:
    """
    Manages per-session vector store and BM25 indices.
    """

    def __init__(self, base_dir: Path | str = "data/sessions"):
        """
        Initialize index storage.
        
        Args:
            base_dir: Root directory for session data
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_session_index_dir(self, session_id: str) -> Path:
        """Get the index directory for a session."""
        index_dir = self.base_dir / session_id / "indices"
        index_dir.mkdir(parents=True, exist_ok=True)
        return index_dir

    def get_faiss_index_dir(self, session_id: str) -> Path:
        """Get FAISS index directory for a session."""
        faiss_dir = self.get_session_index_dir(session_id) / "faiss"
        faiss_dir.mkdir(parents=True, exist_ok=True)
        return faiss_dir

    def get_bm25_cache_dir(self, session_id: str) -> Path:
        """Get BM25 cache directory for a session."""
        bm25_dir = self.get_session_index_dir(session_id) / "bm25"
        bm25_dir.mkdir(parents=True, exist_ok=True)
        return bm25_dir

    def get_memory_index_dir(self, session_id: str) -> Path:
        """Get memory semantic index directory for a session."""
        memory_dir = self.get_session_index_dir(session_id) / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        return memory_dir

    def get_documents_dir(self, session_id: str) -> Path:
        """Get documents directory for a session."""
        docs_dir = self.base_dir / session_id / "documents"
        docs_dir.mkdir(parents=True, exist_ok=True)
        return docs_dir

    def get_uploads_dir(self, session_id: str) -> Path:
        """Get temporary uploads directory for a session."""
        uploads_dir = self.base_dir / session_id / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        return uploads_dir

    def clear_session_indices(self, session_id: str) -> bool:
        """
        Clear all indices for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if successful
        """
        try:
            import shutil
            index_dir = self.get_session_index_dir(session_id)
            if index_dir.exists():
                shutil.rmtree(index_dir)
            logger.info("Session indices cleared: %s", session_id)
            return True
        except Exception as e:
            logger.error("Failed to clear indices for session %s: %s", session_id, e, exc_info=True)
            return False
