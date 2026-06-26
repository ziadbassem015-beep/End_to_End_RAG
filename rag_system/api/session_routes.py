"""
rag_system/api/session_routes.py
================================
FastAPI routes for session management and session-scoped operations.
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from rag_system.api.session_state import SessionAwareStateManager
from rag_system.ingestion.loader import PDFLoader
from rag_system.ingestion.chunker import DocumentChunker
from rag_system.embeddings.embedder import Embedder
from rag_system.session.models import (
    CreateSessionRequest,
    UpdateSessionRequest,
    SessionResponse,
    SessionDetailResponse,
)
from rag_system.vectorstore.faiss_store import FAISSStore

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/sessions", tags=["sessions"])

# Shared state manager (initialized in main.py)
session_state_manager: Optional[SessionAwareStateManager] = None

# Directories
UPLOAD_DIR = Path("data/uploads")
CACHE_DIR = Path("data/cache/embeddings")
SESSIONS_DIR = Path("data/sessions")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Request/Response Models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request model for querying a session."""

    query: str = Field(..., description="The user query")
    top_k: int = Field(5, description="Number of passages to retrieve")
    alpha: float = Field(0.7, description="Dense search weight")
    use_reranker: bool = Field(True, description="Whether to use reranking")
    use_memory: bool = Field(True, description="Include memory in retrieval")
    llm_provider: str = Field("github", description="LLM provider")
    llm_key: Optional[str] = Field(None, description="Optional API key override")
    llm_model: Optional[str] = Field(None, description="Optional LLM model name")
    llm_temp: float = Field(0.1, description="Generation temperature")


class QueryResponse(BaseModel):
    """Response model for query."""

    answer: str
    retrieved_chunks: list
    memory_context: Optional[str] = None
    message_id: str


# ─── Background Tasks ────────────────────────────────────────────────────────

def process_document_for_session(
    task_id: str,
    session_id: str,
    file_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    ocr_mode: str,
    embedding_model: str,
):
    """Background task to process document for a session."""
    if not session_state_manager:
        logger.error("State manager not initialized")
        return
    
    try:
        # Get session
        session = session_state_manager.session_manager.get_session(session_id)
        if not session:
            logger.error("Session not found: %s", session_id)
            return
        
        logger.info("Processing document for session %s: %s", session_id, file_path.name)
        
        # Step 1: Load document
        session_state_manager.set_task(task_id, file_path.name, status="loading", progress=10)
        
        ocr_fallback = False
        min_chars_for_ocr = 0.5
        if ocr_mode == "Force OCR":
            ocr_fallback = True
            min_chars_for_ocr = 0.0
        elif ocr_mode == "Auto-detect scanned pages":
            ocr_fallback = True
            min_chars_for_ocr = 0.5
        
        loader = PDFLoader(
            pdf_path=file_path,
            ocr_fallback=ocr_fallback,
            min_chars_for_ocr=min_chars_for_ocr,
        )
        documents = loader.load()
        
        # Step 2: Chunk
        session_state_manager.set_task(task_id, file_path.name, status="chunking", progress=40)
        
        chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = chunker.chunk(documents)
        chunk_dicts = [c.to_dict() for c in chunks]
        
        # Step 3: Embed
        session_state_manager.set_task(task_id, file_path.name, status="embedding", progress=60)
        
        embedder = Embedder(model_name=embedding_model, cache_dir=CACHE_DIR)
        embedded_chunks = embedder.embed(chunk_dicts)
        
        # Step 4: Index
        session_state_manager.set_task(task_id, file_path.name, status="indexing", progress=80)
        
        faiss_dir = session_state_manager.session_manager.index_storage.get_faiss_index_dir(
            session_id
        )
        
        dim = len(embedded_chunks[0]["embedding"]) if embedded_chunks else 384
        vector_store = FAISSStore(dimension=dim)
        vector_store.add_batch(embedded_chunks)
        vector_store.save(str(faiss_dir))
        
        # Step 5: Register document in session
        doc_metadata = session_state_manager.session_manager.add_document_to_session(
            session_id=session_id,
            filename=file_path.name,
            source=str(file_path),
            embedding_model=embedding_model,
            file_size=file_path.stat().st_size,
            page_count=None,
        )
        
        # Clear retriever cache
        session_state_manager.clear_session_retriever_cache(session_id)
        
        session_state_manager.set_task(task_id, file_path.name, status="completed", progress=100)
        logger.info("Document processing completed for session %s", session_id)
        
    except Exception as e:
        logger.error("Document processing failed for session %s: %s", session_id, e, exc_info=True)
        session_state_manager.set_task(
            task_id, file_path.name, status="failed", progress=100, error=str(e)
        )


# ─── Session Management Endpoints ────────────────────────────────────────────

@router.post("")
async def create_session(request: CreateSessionRequest) -> dict:
    """Create a new session."""
    if not session_state_manager:
        raise HTTPException(status_code=500, detail="State manager not initialized")
    
    try:
        result = session_state_manager.create_session(
            name=request.name,
            description=request.description,
            embedding_model=request.embedding_model,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
        return result
    except Exception as e:
        logger.error("Failed to create session: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_sessions() -> dict:
    """List all sessions."""
    if not session_state_manager:
        raise HTTPException(status_code=500, detail="State manager not initialized")
    
    try:
        sessions = session_state_manager.list_sessions()
        return {"sessions": sessions, "count": len(sessions)}
    except Exception as e:
        logger.error("Failed to list sessions: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}")
async def get_session(session_id: str) -> dict:
    """Get session details."""
    if not session_state_manager:
        raise HTTPException(status_code=500, detail="State manager not initialized")
    
    try:
        session = session_state_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get session: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{session_id}")
async def update_session(session_id: str, request: UpdateSessionRequest) -> dict:
    """Update session metadata."""
    if not session_state_manager:
        raise HTTPException(status_code=500, detail="State manager not initialized")
    
    try:
        session = session_state_manager.session_manager.update_session(
            session_id=session_id,
            name=request.name,
            description=request.description,
            settings=request.settings,
        )
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"session_id": session.session_id, "name": session.name}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update session: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{session_id}")
async def delete_session(session_id: str) -> dict:
    """Delete a session and all its data."""
    if not session_state_manager:
        raise HTTPException(status_code=500, detail="State manager not initialized")
    
    try:
        if not session_state_manager.delete_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "deleted", "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete session: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/switch")
async def switch_session(session_id: str) -> dict:
    """Switch to a different session."""
    if not session_state_manager:
        raise HTTPException(status_code=500, detail="State manager not initialized")
    
    try:
        if not session_state_manager.set_active_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "switched", "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to switch session: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Document Upload Endpoints ───────────────────────────────────────────────

@router.post("/{session_id}/upload")
async def upload_document(
    session_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chunk_size: int = Form(800),
    chunk_overlap: int = Form(150),
    ocr_mode: str = Form("Disable OCR"),
):
    """Upload a document to a session."""
    if not session_state_manager:
        raise HTTPException(status_code=500, detail="State manager not initialized")
    
    # Verify session exists
    session = session_state_manager.session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files supported")
    
    try:
        task_id = str(uuid.uuid4())
        temp_file_path = UPLOAD_DIR / file.filename
        
        # Save file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        session_state_manager.set_task(task_id, file.filename, status="starting", progress=0)
        
        # Queue background task
        background_tasks.add_task(
            process_document_for_session,
            task_id=task_id,
            session_id=session_id,
            file_path=temp_file_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            ocr_mode=ocr_mode,
            embedding_model=session.embedding_model,
        )
        
        return {"task_id": task_id, "filename": file.filename, "status": "processing"}
    except Exception as e:
        logger.error("Failed to upload document: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Query Endpoints ────────────────────────────────────────────────────────

@router.post("/{session_id}/query")
async def query_session(session_id: str, request: QueryRequest) -> QueryResponse:
    """Query a session."""
    if not session_state_manager:
        raise HTTPException(status_code=500, detail="State manager not initialized")
    
    try:
        # Verify session exists
        session = session_state_manager.session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get retriever
        retriever = session_state_manager.get_session_retriever(
            session_id=session_id,
            embedding_model=session.embedding_model,
            alpha=request.alpha,
            use_reranker=request.use_reranker,
        )
        
        if not retriever:
            raise HTTPException(
                status_code=500,
                detail="Failed to initialize retriever for session",
            )
        
        # Retrieve
        retrieved_results = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            include_memory=request.use_memory,
        )
        
        # Format passages
        passages = [r.to_dict() for r in retrieved_results]
        
        # Get memory context
        memory_context = ""
        if request.use_memory:
            memory_info = session_state_manager.get_session_memory_context(
                session_id, request.query
            )
            memory_context = memory_info.get("memory_context", "")
        
        # Generate answer
        answer = ""
        api_key = request.llm_key or os.getenv("GITHUB_TOKEN") or os.getenv("OPENAI_API_KEY")
        
        if api_key and api_key.strip():
            try:
                generator = session_state_manager.get_generator(
                    api_key=api_key.strip(),
                    provider=request.llm_provider,
                    model_name=request.llm_model,
                )
                answer = generator.generate_answer(
                    query=request.query,
                    retrieved_chunks=retrieved_results,
                    temperature=request.llm_temp,
                )
            except Exception as e:
                logger.warning("LLM generation failed, using mock: %s", e)
                answer = f"*(LLM failed: {str(e)})*"
        else:
            answer = "*(LLM not configured - showing retrieved passages)*"
        
        # Add message to session
        user_msg = session_state_manager.add_message_to_session(
            session_id=session_id,
            role="user",
            content=request.query,
            source_chunks=passages,
        )
        
        assistant_msg = session_state_manager.add_message_to_session(
            session_id=session_id,
            role="assistant",
            content=answer,
        )
        
        return QueryResponse(
            answer=answer,
            retrieved_chunks=passages,
            memory_context=memory_context,
            message_id=assistant_msg["message_id"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Query failed for session %s: %s", session_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Chat History Endpoints ─────────────────────────────────────────────────

@router.get("/{session_id}/chat-history")
async def get_chat_history(session_id: str, limit: Optional[int] = None) -> dict:
    """Get chat history for a session."""
    if not session_state_manager:
        raise HTTPException(status_code=500, detail="State manager not initialized")
    
    try:
        messages = session_state_manager.get_session_chat_history(session_id, limit)
        return {"messages": messages, "count": len(messages)}
    except Exception as e:
        logger.error("Failed to get chat history: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/clear-chat")
async def clear_chat(session_id: str) -> dict:
    """Clear chat history for a session."""
    if not session_state_manager:
        raise HTTPException(status_code=500, detail="State manager not initialized")
    
    try:
        session_state_manager.session_manager.clear_chat_history(session_id)
        return {"status": "cleared", "session_id": session_id}
    except Exception as e:
        logger.error("Failed to clear chat: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Export Endpoints ────────────────────────────────────────────────────────

@router.post("/{session_id}/export")
async def export_session(session_id: str) -> dict:
    """Export session as JSON."""
    if not session_state_manager:
        raise HTTPException(status_code=500, detail="State manager not initialized")
    
    try:
        export_path = UPLOAD_DIR / f"session_{session_id}_export.json"
        if session_state_manager.session_manager.export_session(session_id, export_path):
            return {
                "status": "exported",
                "session_id": session_id,
                "export_path": str(export_path),
            }
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to export session: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
