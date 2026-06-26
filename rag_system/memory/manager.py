"""
rag_system/memory/manager.py
=============================
Hybrid memory system for session-scoped conversations.
Implements short-term, long-term, and semantic memory.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_system.session.models import Message, MemorySummary, MemoryEmbedding
from rag_system.vectorstore.faiss_store import FAISSStore

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Manages multi-level memory for a session:
    - Short-term: Recent messages (kept in context)
    - Long-term: Periodic summaries of conversations
    - Semantic: Embeddings of key information for retrieval
    """

    def __init__(
        self,
        session_id: str,
        memory_index_dir: Path | str,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        short_term_limit: int = 30,
        summarization_threshold: int = 50,
    ):
        """
        Initialize memory manager.
        
        Args:
            session_id: Session ID
            memory_index_dir: Directory to store memory indices
            embedding_model: Model for semantic embeddings
            short_term_limit: Max messages in short-term memory
            summarization_threshold: Create summary after this many messages
        """
        self.session_id = session_id
        self.memory_index_dir = Path(memory_index_dir)
        self.memory_index_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedding_model = embedding_model
        self.short_term_limit = short_term_limit
        self.summarization_threshold = summarization_threshold
        
        # Load embedding model
        try:
            self.embedder = SentenceTransformer(embedding_model)
        except Exception as e:
            logger.warning("Failed to load embedding model %s: %s", embedding_model, e)
            self.embedder = None
        
        # Initialize semantic memory index
        self.memory_vector_store: Optional[FAISSStore] = None
        self._initialize_memory_index()
        
        logger.info("MemoryManager initialized for session %s", session_id)

    def _initialize_memory_index(self) -> None:
        """Initialize or load memory vector store."""
        try:
            index_path = self.memory_index_dir / "memory_index"
            index_metadata_path = self.memory_index_dir / "memory_metadata.json"
            
            if (index_path / "index.faiss").exists():
                # Load existing index
                self.memory_vector_store = FAISSStore(dimension=384)
                self.memory_vector_store.load(str(index_path))
                logger.info("Memory index loaded for session %s", self.session_id)
            else:
                # Create new index
                self.memory_vector_store = FAISSStore(dimension=384)
                logger.info("New memory index created for session %s", self.session_id)
        except Exception as e:
            logger.error("Failed to initialize memory index: %s", e)
            self.memory_vector_store = FAISSStore(dimension=384)

    # ─── Short-Term Memory ───────────────────────────────────────────────────

    def get_short_term_memory(self, messages: List[Message]) -> List[Message]:
        """
        Get recent messages for short-term memory.
        
        Args:
            messages: Full chat history
            
        Returns:
            Recent messages (up to short_term_limit)
        """
        if len(messages) <= self.short_term_limit:
            return messages
        
        # Return last N messages
        return messages[-self.short_term_limit :]

    def format_short_term_context(self, messages: List[Message]) -> str:
        """
        Format short-term memory for inclusion in prompt.
        
        Args:
            messages: Recent messages
            
        Returns:
            Formatted context string
        """
        if not messages:
            return ""
        
        context_parts = ["Recent conversation history:"]
        for msg in messages[-10:]:  # Last 10 messages
            context_parts.append(f"\n{msg.role.upper()}: {msg.content[:200]}...")
        
        return "\n".join(context_parts)

    # ─── Long-Term Memory ────────────────────────────────────────────────────

    def should_create_summary(self, messages: List[Message]) -> bool:
        """
        Determine if a summary should be created.
        
        Args:
            messages: Full chat history
            
        Returns:
            True if summary needed
        """
        # Summary after each N new messages
        return len(messages) % self.summarization_threshold == 0 and len(messages) > 0

    def create_summary(
        self,
        messages: List[Message],
        summary_text: str,
    ) -> MemorySummary:
        """
        Create a long-term memory summary.
        
        Args:
            messages: Messages to summarize
            summary_text: Pre-generated summary text
            
        Returns:
            MemorySummary object
        """
        # Extract key information from summary
        topics = self._extract_topics(summary_text)
        concepts = self._extract_concepts(summary_text)
        facts = self._extract_facts(summary_text)
        questions = self._extract_questions(summary_text)
        
        summary = MemorySummary(
            message_range=(
                messages[0].message_id if messages else "",
                messages[-1].message_id if messages else "",
            ),
            topics=topics,
            key_concepts=concepts,
            key_facts=facts,
            unresolved_questions=questions,
            summary_text=summary_text,
        )
        
        logger.info(
            "Summary created for session %s with %d messages",
            self.session_id,
            len(messages),
        )
        return summary

    def get_relevant_summaries(
        self, query: str, top_k: int = 3
    ) -> List[MemorySummary]:
        """
        Retrieve relevant memory summaries for a query.
        This is placeholder - actual implementation would search summaries.
        
        Args:
            query: Query text
            top_k: Number of summaries to return
            
        Returns:
            Relevant summaries (not yet implemented)
        """
        # TODO: Implement semantic search over summaries
        return []

    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from summary text."""
        # Simple heuristic: look for key phrases
        keywords = [
            "discussed",
            "covered",
            "explored",
            "analyzed",
            "examined",
        ]
        topics = []
        for keyword in keywords:
            if keyword in text.lower():
                # Extract surrounding text as topic
                idx = text.lower().find(keyword)
                if idx != -1:
                    start = max(0, idx - 50)
                    end = min(len(text), idx + 100)
                    topics.append(text[start:end].strip())
        return topics[:5]

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from summary."""
        # Simple extraction of capitalized phrases
        words = text.split()
        concepts = []
        for i, word in enumerate(words):
            if word and word[0].isupper() and len(word) > 3:
                if i + 1 < len(words) and words[i + 1][0].isupper():
                    concepts.append(f"{word} {words[i+1]}")
                else:
                    concepts.append(word)
        return list(set(concepts))[:10]

    def _extract_facts(self, text: str) -> List[str]:
        """Extract key facts from summary."""
        sentences = text.split(". ")
        # Return sentences longer than 20 chars as facts
        facts = [s.strip() for s in sentences if len(s) > 20]
        return facts[:10]

    def _extract_questions(self, text: str) -> List[str]:
        """Extract unresolved questions from summary."""
        sentences = text.split(". ")
        questions = [s.strip() for s in sentences if "?" in s]
        return questions[:5]

    # ─── Semantic Memory ─────────────────────────────────────────────────────

    def embed_message(self, message: Message) -> Optional[np.ndarray]:
        """
        Embed a message for semantic memory.
        
        Args:
            message: Message to embed
            
        Returns:
            Embedding vector or None
        """
        if not self.embedder:
            return None
        
        try:
            embedding = self.embedder.encode(message.content, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error("Failed to embed message: %s", e)
            return None

    def add_to_semantic_memory(self, message: Message) -> Optional[MemoryEmbedding]:
        """
        Add message to semantic memory index.
        
        Args:
            message: Message to add
            
        Returns:
            MemoryEmbedding object or None
        """
        embedding = self.embed_message(message)
        if embedding is None:
            return None
        
        # Add to vector store
        if self.memory_vector_store:
            chunk_data = {
                "chunk_id": message.message_id,
                "content": message.content,
                "embedding": embedding.tolist(),
                "role": message.role,
                "timestamp": message.timestamp.isoformat(),
            }
            
            self.memory_vector_store.add([chunk_data])
            
            # Save index
            self._save_memory_index()
            
            memory_embedding = MemoryEmbedding(
                message_id=message.message_id,
                embedding=embedding.tolist(),
                content_type="message",
            )
            
            return memory_embedding
        
        return None

    def retrieve_semantic_memories(
        self, query: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant messages from semantic memory.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of relevant memory chunks
        """
        if not self.memory_vector_store or not self.embedder:
            return []
        
        try:
            # Embed query
            query_embedding = self.embedder.encode(query, convert_to_numpy=True)
            
            # Search in memory index
            results = self.memory_vector_store.search(
                query_embedding.tolist(),
                top_k=top_k
            )
            
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "message_id": result.chunk_id,
                    "content": result.content,
                    "role": result.metadata.get("role", "user"),
                    "score": result.score,
                })
            
            return formatted_results
        except Exception as e:
            logger.error("Failed to retrieve semantic memories: %s", e)
            return []

    def _save_memory_index(self) -> None:
        """Save memory vector store to disk."""
        if not self.memory_vector_store:
            return
        
        try:
            index_path = self.memory_index_dir / "memory_index"
            self.memory_vector_store.save(str(index_path))
        except Exception as e:
            logger.error("Failed to save memory index: %s", e)

    # ─── Memory Context Builder ──────────────────────────────────────────────

    def build_memory_context(
        self,
        chat_history: List[Message],
        query: str,
        include_summaries: bool = True,
        include_semantic: bool = True,
    ) -> str:
        """
        Build comprehensive memory context for LLM prompt.
        
        Args:
            chat_history: Full chat history
            query: Current user query
            include_summaries: Include long-term summaries
            include_semantic: Include semantic memory results
            
        Returns:
            Formatted memory context
        """
        context_parts = []
        
        # Add short-term memory
        short_term = self.get_short_term_memory(chat_history)
        if short_term:
            context_parts.append("### Recent Conversation (Short-term Memory)")
            for msg in short_term[-5:]:
                context_parts.append(f"{msg.role.upper()}: {msg.content[:150]}")
        
        # Add semantic memory results
        if include_semantic:
            semantic_results = self.retrieve_semantic_memories(query, top_k=3)
            if semantic_results:
                context_parts.append("\n### Relevant Past Discussions (Semantic Memory)")
                for result in semantic_results:
                    context_parts.append(f"- {result['role'].upper()}: {result['content'][:100]}")
        
        # Add relevant summaries
        if include_summaries:
            summaries = self.get_relevant_summaries(query)
            if summaries:
                context_parts.append("\n### Session Summaries (Long-term Memory)")
                for summary in summaries:
                    context_parts.append(f"- {summary.summary_text[:200]}")
        
        return "\n".join(context_parts)

    # ─── Utilities ───────────────────────────────────────────────────────────

    def clear_memory(self) -> bool:
        """Clear all memory for the session."""
        try:
            import shutil
            if self.memory_index_dir.exists():
                shutil.rmtree(self.memory_index_dir)
            self._initialize_memory_index()
            logger.info("Memory cleared for session %s", self.session_id)
            return True
        except Exception as e:
            logger.error("Failed to clear memory: %s", e)
            return False

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage."""
        stats = {
            "session_id": self.session_id,
            "has_embedder": self.embedder is not None,
            "memory_index_exists": self.memory_vector_store is not None,
        }
        
        if self.memory_vector_store:
            try:
                stats["indexed_items"] = len(self.memory_vector_store.metadata)
            except Exception:
                stats["indexed_items"] = 0
        
        return stats
