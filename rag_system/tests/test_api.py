"""
rag_system/tests/test_api.py
============================
Unit tests for the FastAPI backend REST endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from pathlib import Path

from rag_system.api.main import app
from rag_system.api.state import state_manager

client = TestClient(app)


def test_get_indexed_books_empty():
    """Test getting books list when no indices exist."""
    with patch("pathlib.Path.exists", return_value=False):
        response = client.get("/api/books")
        assert response.status_code == 200
        assert response.json() == {"books": []}


def test_get_task_status_not_found():
    """Test polling status for an invalid/non-existent task ID."""
    response = client.get("/api/status/invalid-uuid-value")
    assert response.status_code == 404
    assert "detail" in response.json()


def test_clear_system_success():
    """Test clearing indices and uploads."""
    with patch("shutil.rmtree") as mock_rmtree, \
         patch("pathlib.Path.mkdir") as mock_mkdir:
        response = client.post("/api/clear")
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert mock_rmtree.call_count == 2  # INDEX_BASE_DIR and UPLOAD_DIR


@patch("fastapi.BackgroundTasks.add_task")
def test_upload_book_success(mock_add_task):
    """Test successful book file upload initiates background processing."""
    file_content = b"PDF dummy content"
    
    with patch("shutil.copyfileobj"), \
         patch("builtins.open", create=True):
        
        response = client.post(
            "/api/upload",
            files={"file": ("test_book.pdf", file_content, "application/pdf")},
            data={
                "chunk_size": 800,
                "chunk_overlap": 150,
                "ocr_mode": "Disable OCR",
                "embedding_model": "BAAI/bge-small-en-v1.5"
            }
        )
        
        assert response.status_code == 200
        res_json = response.json()
        assert "task_id" in res_json
        assert res_json["book_name"] == "test_book.pdf"
        assert res_json["status"] == "starting"
        
        mock_add_task.assert_called_once()


@patch("rag_system.api.state.state_manager.get_retriever")
@patch("rag_system.api.state.state_manager.get_generator")
def test_query_book_success(mock_get_generator, mock_get_retriever):
    """Test querying an indexed book and returning an LLM answer."""
    # Setup mocks
    mock_retriever = MagicMock()
    mock_retrieved_result = MagicMock()
    mock_retrieved_result.chunk_id = "DOC_P001_C001"
    mock_retrieved_result.rank = 1
    mock_retrieved_result.score = 0.95
    mock_retrieved_result.content = "FAISS is a fast vector store."
    mock_retrieved_result.source = "test_book.pdf"
    mock_retrieved_result.page = 1
    mock_retrieved_result.semantic_score = 0.95
    mock_retrieved_result.bm25_score = 0.0
    mock_retrieved_result.rerank_score = 0.0
    
    mock_get_retriever.return_value = mock_retriever
    mock_retriever.retrieve.return_value = [mock_retrieved_result]
    
    mock_generator = MagicMock()
    mock_get_generator.return_value = mock_generator
    mock_generator.generate_answer.return_value = "FAISS is indeed fast."

    with patch("pathlib.Path.exists", return_value=True), \
         patch("builtins.open", create=True) as mock_open:
        
        # Mock reading metadata file
        mock_open.return_value.__enter__.return_value.read.return_value = (
            '{"dimension": 384, "metadata": {}}'
        )

        response = client.post(
            "/api/query",
            json={
                "book_name": "test_book",
                "query": "Is FAISS fast?",
                "top_k": 5,
                "alpha": 0.7,
                "use_reranker": True,
                "reranker_model": "Cross-Encoder (ms-marco-MiniLM)",
                "llm_provider": "github",
                "llm_key": "dummy_token"
            }
        )

        assert response.status_code == 200
        res_json = response.json()
        assert res_json["answer"] == "FAISS is indeed fast."
        assert len(res_json["retrieved_chunks"]) == 1
        assert res_json["retrieved_chunks"][0]["chunk_id"] == "DOC_P001_C001"
        assert res_json["retrieved_chunks"][0]["content"] == "FAISS is a fast vector store."
