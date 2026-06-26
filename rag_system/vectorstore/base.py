"""
vectorstore/base.py
===================
Abstract base class representing the VectorStore interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class VectorStore(ABC):
    """
    Abstract interface for a Vector Database store in the RAG framework.
    All vector store backends (FAISS, ChromaDB, PGVector, Qdrant) must implement
    this interface.
    """

    @abstractmethod
    def add(self, record: Dict[str, Any]) -> None:
        """
        Add a single record to the vector store.

        Each record must contain the following keys:
            - chunk_id: str
            - content: str
            - source: str
            - page: int
            - section: str
            - embedding: List[float]
        """
        pass

    @abstractmethod
    def add_batch(self, records: List[Dict[str, Any]]) -> None:
        """
        Add a batch of records to the vector store.
        Each record must contain: chunk_id, content, source, page, section, embedding.
        """
        pass

    @abstractmethod
    def search(self, query_vector: List[float], top_k: int, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search the vector store for top_k most similar records.

        Args:
            query_vector: A list of floats representing the query embedding.
            top_k: Number of nearest neighbors to retrieve.
            filter_dict: Optional dictionary of key-value metadata filters.

        Returns:
            A list of dictionaries representing matching records. Each dict should
            contain: chunk_id, content, source, page, section, embedding, and score
            (similarity score).
        """
        pass

    @abstractmethod
    def delete(self, chunk_id: str) -> bool:
        """
        Delete a record by its chunk_id.

        Returns:
            True if the record was successfully found and deleted, False otherwise.
        """
        pass

    @abstractmethod
    def update(self, chunk_id: str, record: Dict[str, Any]) -> bool:
        """
        Update a record by chunk_id.

        Returns:
            True if the record was found and updated, False otherwise.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """
        Return the total number of records currently stored.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Persist the vector store index and companion metadata to the specified directory/path.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the vector store index and companion metadata from the specified directory/path.
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if the vector store is healthy and operational.
        """
        pass
