"""
rag_system/session/models.py
============================
Data models for session management, messages, and memory.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata for a document within a session."""
    
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    source: str
    page_count: Optional[int] = None
    upload_date: datetime
    file_size: int
    embedding_model: str
    chunk_count: int = 0


class Message(BaseModel):
    """A single message in the conversation history."""
    
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str = Field(..., description="'user' or 'assistant'")
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_chunks: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemorySummary(BaseModel):
    """Long-term memory summary created periodically."""
    
    summary_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    message_range: tuple = Field(..., description="(start_msg_id, end_msg_id)")
    topics: List[str] = Field(default_factory=list)
    key_concepts: List[str] = Field(default_factory=list)
    key_facts: List[str] = Field(default_factory=list)
    unresolved_questions: List[str] = Field(default_factory=list)
    summary_text: str


class MemoryEmbedding(BaseModel):
    """Semantic memory stored as embeddings."""
    
    embedding_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message_id: str
    embedding: List[float]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    content_type: str = Field(..., description="'message' | 'summary'")


class Session(BaseModel):
    """Main session model representing a persistent knowledge workspace."""
    
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="User-friendly session name")
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: Optional[datetime] = None
    
    # Documents in this session
    documents: List[DocumentMetadata] = Field(default_factory=list)
    
    # Conversation history
    chat_history: List[Message] = Field(default_factory=list)
    
    # Memory
    memory_summaries: List[MemorySummary] = Field(default_factory=list)
    memory_embeddings: List[MemoryEmbedding] = Field(default_factory=list)
    
    # Configuration
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    chunk_size: int = Field(default=800)
    chunk_overlap: int = Field(default=150)
    
    # Settings
    settings: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "documents": [d.dict() for d in self.documents],
            "chat_history": [m.dict() for m in self.chat_history],
            "memory_summaries": [s.dict() for s in self.memory_summaries],
            "memory_embeddings": [e.dict() for e in self.memory_embeddings],
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "settings": self.settings,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Session:
        """Create from dictionary (e.g., loaded from JSON)."""
        # Parse nested objects
        documents = [DocumentMetadata(**d) for d in data.get("documents", [])]
        chat_history = [Message(**m) for m in data.get("chat_history", [])]
        memory_summaries = [MemorySummary(**s) for s in data.get("memory_summaries", [])]
        memory_embeddings = [MemoryEmbedding(**e) for e in data.get("memory_embeddings", [])]
        
        # Parse timestamps
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        
        last_accessed = data.get("last_accessed")
        if isinstance(last_accessed, str):
            last_accessed = datetime.fromisoformat(last_accessed)
        
        return cls(
            session_id=data.get("session_id", str(uuid.uuid4())),
            name=data.get("name", "Untitled Session"),
            description=data.get("description"),
            created_at=created_at,
            updated_at=updated_at,
            last_accessed=last_accessed,
            documents=documents,
            chat_history=chat_history,
            memory_summaries=memory_summaries,
            memory_embeddings=memory_embeddings,
            embedding_model=data.get("embedding_model", "BAAI/bge-small-en-v1.5"),
            chunk_size=data.get("chunk_size", 800),
            chunk_overlap=data.get("chunk_overlap", 150),
            settings=data.get("settings", {}),
        )


# ─── Request/Response Models for API ─────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    """Request to create a new session."""
    
    name: str = Field(..., description="Session name")
    description: Optional[str] = None
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    chunk_size: int = Field(default=800, ge=100, le=5000)
    chunk_overlap: int = Field(default=150, ge=0, le=1000)


class UpdateSessionRequest(BaseModel):
    """Request to update session metadata."""
    
    name: Optional[str] = None
    description: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None


class SessionResponse(BaseModel):
    """Response model for session info."""
    
    session_id: str
    name: str
    description: Optional[str]
    created_at: str
    updated_at: str
    last_accessed: Optional[str]
    document_count: int
    message_count: int
    embedding_model: str


class SessionDetailResponse(BaseModel):
    """Detailed session response."""
    
    session_id: str
    name: str
    description: Optional[str]
    created_at: str
    updated_at: str
    documents: List[DocumentMetadata]
    chat_history: List[Message]
    memory_summaries: List[MemorySummary]
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
