"""
rag_system/llm/generator.py
===========================
LLM Generator for answering questions based on retrieved document context.
Supports both GitHub Models API and standard OpenAI API.
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()



class LLMGenerator:
    """
    Handles connection to LLM providers (GitHub Models or OpenAI)
    and generates answers based on retrieved context.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,  # "github" or "openai"
        model_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the LLM client.

        Args:
            api_key: The API key to use. If None, it will look up GITHUB_TOKEN or OPENAI_API_KEY.
            provider: Explicit provider choice. If None, it will auto-detect.
            model_name: The model to use. Defaults depend on the provider.
        """
        # Support Mock/Offline mode directly
        if provider and provider.lower() == "mock":
            self.provider = "mock"
            self.model_name = "mock-generator"
            logger.info("LLMGenerator initialized in Mock/Offline mode.")
            return

        # Try to resolve API Key and Provider
        resolved_key = api_key or os.getenv("GITHUB_TOKEN") or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No API Key found. Please configure GITHUB_TOKEN or OPENAI_API_KEY in your environment, "
                "or provide it explicitly."
            )

        # Detect provider
        if provider:
            self.provider = provider.lower()
        else:
            # Auto-detect based on key prefix or variable presence
            if api_key:
                if api_key.startswith("ghp_") or api_key.startswith("github_pat_") or len(api_key) > 40:
                    self.provider = "github"
                else:
                    self.provider = "openai"
            elif os.getenv("GITHUB_TOKEN"):
                self.provider = "github"
            else:
                self.provider = "openai"

        self.api_key = resolved_key

        # Set default model names
        if self.provider == "github":
            self.base_url = "https://models.inference.ai.azure.com"
            self.model_name = model_name or os.getenv("GITHUB_METADATA_MODEL", "gpt-4o")
        else:
            self.base_url = "https://api.openai.com/v1"
            self.model_name = model_name or "gpt-4o-mini"

        logger.info(
            "LLMGenerator initialized. Provider: %s, Model: %s",
            self.provider,
            self.model_name,
        )

        # Initialize OpenAI Client
        try:
            from openai import OpenAI
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        except ImportError as exc:
            raise ImportError(
                "LLM generator requires the 'openai' package. Install it with pip install openai"
            ) from exc

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Any],
        custom_system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> str:
        """
        Generate an answer to the query using the retrieved chunks as context.

        Args:
            query: The user's question.
            retrieved_chunks: List of RetrievalResult objects containing context.
            custom_system_prompt: Override the default system instructions.
            temperature: Generation temperature (default 0.0 for deterministic answers).
            max_tokens: Maximum tokens in response.

        Returns:
            The generated response from the LLM.
        """
        if self.provider == "mock":
            if not retrieved_chunks:
                return "No relevant context was found in the book to answer your question."
            top_chunk = retrieved_chunks[0]
            content = getattr(top_chunk, "content", "")
            source = getattr(top_chunk, "source", "Unknown Source")
            page = getattr(top_chunk, "page", "Unknown Page")
            return (
                f"[MOCK LLM RESPONSE - OFFLINE MODE]\n\n"
                f"Based on the retrieved context from page {page} of '{source}':\n\n"
                f"\"...{content.strip()}...\"\n\n"
                f"*(This answer is simulated since the system is running in Mock Mode.)*"
            )

        if not retrieved_chunks:
            return "No relevant context was found in the book to answer your question."

        # Format context
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source = getattr(chunk, "source", "Unknown Source")
            page = getattr(chunk, "page", "Unknown Page")
            content = getattr(chunk, "content", "")
            context_blocks.append(
                f"[{i}] Source: {source} (Page {page})\n{content.strip()}"
            )

        context_text = "\n\n---\n\n".join(context_blocks)

        # Default system prompt
        system_prompt = custom_system_prompt or (
            "You are a helpful and precise assistant. You are answering questions based on the provided book context.\n"
            "Analyze the context carefully to answer the user's question as accurately and concisely as possible.\n"
            "Rules:\n"
            "1. Answer the question using ONLY the provided context.\n"
            "2. If the context does not contain the answer, state that you cannot find the answer in the book.\n"
            "3. Cite the context sources (e.g., [1], [2]) when referencing details in your answer.\n"
            "4. Do NOT make up any facts or extrapolate beyond what is explicitly mentioned in the context."
        )

        user_content = (
            f"Context:\n"
            f"======================\n"
            f"{context_text}\n"
            f"======================\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        logger.info("Sending prompt to LLM (provider: %s, model: %s)...", self.provider, self.model_name)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            answer = response.choices[0].message.content
            return answer or "No response from model."
        except Exception as e:
            logger.error("Failed to generate LLM response: %s", e)
            return f"Error generating answer from LLM: {str(e)}"
