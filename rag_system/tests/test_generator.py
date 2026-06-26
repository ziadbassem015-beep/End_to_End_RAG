"""
rag_system/tests/test_generator.py
==================================
Unit tests for the LLMGenerator module.
"""

import pytest
from unittest.mock import MagicMock, patch
from rag_system.llm.generator import LLMGenerator
from rag_system.retrieval.retriever import RetrievalResult


def test_generator_no_key_raises_error():
    """Verify that ValueError is raised if no API key is set in environment or passed."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="No API Key found"):
            LLMGenerator()


def test_generator_provider_auto_detect():
    """Verify provider auto-detection works for GitHub vs OpenAI keys."""
    # Test Github token
    with patch("openai.OpenAI"):
        gen_gh = LLMGenerator(api_key="github_pat_1234abcd5678efgh")
        assert gen_gh.provider == "github"
        assert gen_gh.base_url == "https://models.inference.ai.azure.com"

        # Test OpenAI key
        gen_oa = LLMGenerator(api_key="sk-proj-somekeyvalue")
        assert gen_oa.provider == "openai"
        assert gen_oa.base_url == "https://api.openai.com/v1"


@patch("openai.OpenAI")
def test_generate_answer_success(mock_openai):
    """Test successful response generation with context formatting."""
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    # Mock chat completion return structure
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    
    mock_client.chat.completions.create.return_value = mock_response
    mock_response.choices = [mock_choice]
    mock_choice.message = mock_message
    mock_message.content = "According to the context [1], RAG is great."

    gen = LLMGenerator(api_key="sk-test-key")
    
    # Mock retrieved chunks
    chunks = [
        RetrievalResult(
            chunk_id="DOC_P001_C001",
            rank=1,
            score=0.9,
            content="RAG systems combine retrieval and LLM generation.",
            source="test_book.pdf",
            page=1,
            section="Introduction"
        )
    ]

    answer = gen.generate_answer(query="What is RAG?", retrieved_chunks=chunks)
    assert "RAG is great" in answer
    mock_client.chat.completions.create.assert_called_once()
