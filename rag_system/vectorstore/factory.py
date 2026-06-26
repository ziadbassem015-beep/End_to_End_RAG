"""
vectorstore/factory.py
======================
Factory class to instantiate VectorStore implementations.
"""

from typing import Any, Dict
from rag_system.vectorstore.base import VectorStore
from rag_system.vectorstore.faiss_store import FAISSStore


class VectorStoreFactory:
    """
    Factory for producing VectorStore instances.
    """

    @staticmethod
    def get_vector_store(store_type: str = "faiss", **kwargs: Any) -> VectorStore:
        """
        Produce a VectorStore implementation.

        Args:
            store_type: Type of store, e.g., 'faiss'.
            kwargs: Extra parameters to pass to the store constructor (e.g. dimension).

        Returns:
            An instance implementing the VectorStore interface.
        """
        store_type = store_type.lower().strip()
        if store_type == "faiss":
            dimension = kwargs.get("dimension", 384)
            return FAISSStore(dimension=dimension)
        else:
            raise ValueError(
                f"Unsupported vector store type: '{store_type}'. "
                f"Currently supported types: 'faiss'."
            )
