"""
rag_system/api/main.py
======================
FastAPI Server exposing RAG pipelines as REST API endpoints.
"""

from __future__ import annotations

import os
import uuid
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Core RAG imports
from rag_system.ingestion.loader import PDFLoader
from rag_system.ingestion.chunker import DocumentChunker
from rag_system.embeddings.embedder import Embedder
from rag_system.vectorstore.faiss_store import FAISSStore
from rag_system.retrieval.retriever import BM25Index
from rag_system.api.state import state_manager
from rag_system.api.session_state import SessionAwareStateManager
from rag_system.api import session_routes

logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="RAG Book QA API",
    description="REST API backend for production-grade Book QA Retrieval-Augmented Generation.",
    version="1.0.0"
)

# Enable CORS for the Node.js / React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Directories Setup ───────────────────────────────────────────────────────
UPLOAD_DIR = Path("data/uploads")
INDEX_BASE_DIR = Path("data/indices/faiss")
CACHE_DIR = Path("data/cache/embeddings")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_BASE_DIR.mkdir(parents=True, exist_ok=True)


# ─── Request Data Models ─────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    book_name: str = Field(..., description="The name of the book index to query.")
    query: str = Field(..., description="The natural language question.")
    top_k: int = Field(5, description="Number of passages to retrieve.")
    alpha: float = Field(0.7, description="Dense search weight (0.0 to 1.0).")
    use_reranker: bool = Field(True, description="Whether to run CrossEncoder reranking.")
    reranker_model: str = Field("Cross-Encoder (ms-marco-MiniLM)", description="Reranker model type.")
    
    llm_provider: str = Field("github", description="LLM provider: 'github' or 'openai'.")
    llm_key: Optional[str] = Field(None, description="Optional API key override.")
    llm_model: Optional[str] = Field(None, description="Optional LLM model name override.")
    llm_temp: float = Field(0.1, description="Generation temperature.")


# ─── Ingestion Background Task ───────────────────────────────────────────────

def run_ingestion_pipeline(
    task_id: str,
    file_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    ocr_mode: str,
    embedding_model: str
):
    """Asynchronous background worker to process the book."""
    book_name = file_path.name
    book_dir_name = file_path.stem.replace(" ", "_")
    book_index_dir = INDEX_BASE_DIR / book_dir_name

    try:
        # Step 1: Set OCR settings
        state_manager.set_task(task_id, book_name, status="loading", progress=10)
        ocr_fallback = False
        min_chars_for_ocr = 0.5
        if ocr_mode == "Force OCR":
            ocr_fallback = True
            min_chars_for_ocr = 0.0
        elif ocr_mode == "Auto-detect scanned pages":
            ocr_fallback = True
            min_chars_for_ocr = 0.5

        # Step 2: Load Document
        loader = PDFLoader(
            pdf_path=file_path,
            ocr_fallback=ocr_fallback,
            min_chars_for_ocr=min_chars_for_ocr
        )
        documents = loader.load()

        # Step 3: Chunking
        state_manager.set_task(task_id, book_name, status="chunking", progress=40)
        chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = chunker.chunk(documents)
        chunk_dicts = [c.to_dict() for c in chunks]

        # Step 4: Embedding
        state_manager.set_task(task_id, book_name, status="embedding", progress=60)
        embedder = Embedder(
            model_name=embedding_model,
            cache_dir=CACHE_DIR
        )
        embedded_chunks = embedder.embed(chunk_dicts)

        # Step 5: FAISS Storage
        state_manager.set_task(task_id, book_name, status="indexing", progress=80)
        dim = len(embedded_chunks[0]["embedding"]) if embedded_chunks else 384
        vector_store = FAISSStore(dimension=dim)
        vector_store.add_batch(embedded_chunks)
        
        # Save FAISS index under the book-specific folder
        book_index_dir.mkdir(parents=True, exist_ok=True)
        vector_store.save(str(book_index_dir))

        # Clear retriever cache so the new book is loaded fresh if queried
        state_manager.clear_retriever_cache()

        state_manager.set_task(task_id, book_name, status="completed", progress=100)
        logger.info("Ingestion completed successfully for book: %s", book_name)

    except Exception as e:
        logger.error("Async ingestion failed for task %s: %s", task_id, e, exc_info=True)
        state_manager.set_task(task_id, book_name, status="failed", progress=100, error=str(e))


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_book(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chunk_size: int = Form(800),
    chunk_overlap: int = Form(150),
    ocr_mode: str = Form("Disable OCR"),
    embedding_model: str = Form("BAAI/bge-small-en-v1.5")
):
    """
    Upload a book PDF/TXT and start the asynchronous ingestion pipeline.
    Returns a task ID to poll for status.
    """
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    task_id = str(uuid.uuid4())
    temp_file_path = UPLOAD_DIR / file.filename

    # Save uploaded file
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    state_manager.set_task(task_id, file.filename, status="starting", progress=0)

    # Dispatch background task
    background_tasks.add_task(
        run_ingestion_pipeline,
        task_id=task_id,
        file_path=temp_file_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        ocr_mode=ocr_mode,
        embedding_model=embedding_model
    )

    return {"task_id": task_id, "book_name": file.filename, "status": "starting"}


@app.get("/api/status/{task_id}")
async def get_task_status(task_id: str):
    """Poll the status and progress of a background book processing task."""
    task = state_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    return task


@app.get("/api/books")
async def get_indexed_books():
    """List all books currently indexed and ready to query."""
    if not INDEX_BASE_DIR.exists():
        return {"books": []}
    
    books = []
    # Scan subdirectories under INDEX_BASE_DIR
    for folder in INDEX_BASE_DIR.iterdir():
        if folder.is_dir() and (folder / "index.faiss").exists() and (folder / "metadata.json").exists():
            # Extract actual book filename from FAISS metadata
            try:
                import json
                with open(folder / "metadata.json", "r", encoding="utf-8") as f:
                    meta = json.load(f)
                chunks = list(meta.get("metadata", {}).values())
                book_name = chunks[0].get("source", folder.name) if chunks else folder.name
                embedding_model = chunks[0].get("embedding_model", "unknown") if chunks else "unknown"
                books.append({
                    "id": folder.name,
                    "name": book_name,
                    "chunk_count": len(chunks),
                    "embedding_model": embedding_model
                })
            except Exception:
                books.append({
                    "id": folder.name,
                    "name": folder.name.replace("_", " "),
                    "chunk_count": 0,
                    "embedding_model": "unknown"
                })
    return {"books": books}


@app.post("/api/query")
async def query_book(request: QueryRequest):
    """
    Retrieve relevant passages from the active book index,
    and generate an answer using LLMs (GitHub Models or OpenAI).
    """
    book_index_dir = INDEX_BASE_DIR / request.book_name
    if not book_index_dir.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Book index '{request.book_name}' not found. Please upload it first."
        )

    # 1. Resolve active embedding model from metadata to avoid mismatch
    try:
        import json
        with open(book_index_dir / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        chunks = list(meta.get("metadata", {}).values())
        embedding_model = chunks[0].get("embedding_model", "BAAI/bge-small-en-v1.5") if chunks else "BAAI/bge-small-en-v1.5"
    except Exception:
        embedding_model = "BAAI/bge-small-en-v1.5"

    # 2. Load Retriever
    try:
        retriever = state_manager.get_retriever(
            book_name=request.book_name,
            index_dir=book_index_dir,
            embedding_model=embedding_model,
            alpha=request.alpha,
            use_reranker=request.use_reranker,
            reranker_model=request.reranker_model
        )
    except Exception as e:
        logger.error("Failed to load retriever for query: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load retriever: {str(e)}")

    # 3. Retrieve Passages
    try:
        retrieved_results = retriever.retrieve(request.query, top_k=request.top_k)
    except Exception as e:
        logger.error("Failed to retrieve passages: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed during retrieval: {str(e)}")

    # Format output passages
    passages = []
    for r in retrieved_results:
        passages.append({
            "chunk_id": r.chunk_id,
            "rank": r.rank,
            "score": r.score,
            "content": r.content,
            "source": r.source,
            "page": r.page,
            "semantic_score": r.semantic_score,
            "bm25_score": r.bm25_score,
            "rerank_score": r.rerank_score
        })

    # 4. LLM Answering
    answer = ""
    api_key = request.llm_key or os.getenv("GITHUB_TOKEN") or os.getenv("OPENAI_API_KEY")
    is_mock = request.llm_provider.lower() == "mock"
    
    if is_mock or (api_key and api_key.strip()):
        try:
            generator = state_manager.get_generator(
                api_key=api_key.strip() if api_key else "mock",
                provider=request.llm_provider,
                model_name=request.llm_model
            )
            answer = generator.generate_answer(
                query=request.query,
                retrieved_chunks=retrieved_results,
                temperature=request.llm_temp
            )
        except Exception as e:
            logger.error("Failed to generate LLM response, falling back to Mock/Offline mode: %s", e)
            try:
                mock_generator = state_manager.get_generator(
                    api_key="mock",
                    provider="mock",
                    model_name="mock-generator"
                )
                fallback_answer = mock_generator.generate_answer(
                    query=request.query,
                    retrieved_chunks=retrieved_results,
                    temperature=request.llm_temp
                )
                answer = (
                    f"*(LLM Provider '{request.llm_provider}' failed with authentication/permission error: {str(e)}. "
                    f"Fell back to Offline Mock Mode.)*\n\n"
                    f"{fallback_answer}"
                )
            except Exception as fallback_err:
                answer = f"*(Error generating answer from LLM: {str(e)}. Fallback also failed: {str(fallback_err)})*"
    else:
        answer = "*(LLM Answering skipped because no API key was configured. Showing retrieved passages.)*"

    return {"answer": answer, "retrieved_chunks": passages}


@app.post("/api/clear")
async def clear_system():
    """Clear all book indices, uploads, and in-memory caches."""
    try:
        if INDEX_BASE_DIR.exists():
            shutil.rmtree(INDEX_BASE_DIR)
            INDEX_BASE_DIR.mkdir(parents=True, exist_ok=True)
        if UPLOAD_DIR.exists():
            shutil.rmtree(UPLOAD_DIR)
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        state_manager.clear_retriever_cache()
        return {"status": "success", "message": "System database and caches cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear system: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
