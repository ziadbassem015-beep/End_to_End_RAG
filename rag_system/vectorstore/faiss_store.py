"""
vectorstore/faiss_store.py
==========================
FAISS-backed implementation of the VectorStore interface.
Uses IndexFlatIP (dot product/cosine similarity on normalized vectors) wrapped in IndexIDMap.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional
import numpy as np
import faiss

from rag_system.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)


class FAISSStore(VectorStore):
    """
    FAISS-based vector store implementation.
    """

    def __init__(self, dimension: int = 384) -> None:
        """
        Args:
            dimension: Dimension of the vectors. Default 384 matches all-MiniLM-L6-v2.
                       BGE-small also uses 384. E5-small also uses 384.
        """
        self.dimension = dimension
        self._counter = 0
        self.chunk_id_to_int_id: Dict[str, int] = {}
        self.int_id_to_chunk_id: Dict[int, str] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

        # IndexFlatIP calculates dot product. L2-normalized vector dot product is equivalent to cosine similarity.
        self.quantizer = faiss.IndexFlatIP(self.dimension)
        # Wrap in IndexIDMap to assign arbitrary integer IDs to vectors and support removal.
        self.index = faiss.IndexIDMap(self.quantizer)

        logger.info("Initialized FAISSStore with dimension=%d", dimension)

    def add(self, record: Dict[str, Any]) -> None:
        """
        Add a single record to the FAISS vector store.
        """
        required = ["chunk_id", "content", "source", "page", "section", "embedding"]
        for key in required:
            if key not in record:
                raise ValueError(f"Missing required key '{key}' in vector record.")

        chunk_id = record["chunk_id"]
        embedding = np.array(record["embedding"], dtype=np.float32)

        if len(embedding) != self.dimension:
            # Reinitialize index if dimensions mismatch and index is empty
            if self.count() == 0:
                self.dimension = len(embedding)
                self.quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIDMap(self.quantizer)
                logger.warning("FAISSStore: adjusted index dimension to %d", self.dimension)
            else:
                raise ValueError(
                    f"Dimension mismatch: got {len(embedding)}, expected {self.dimension}"
                )

        # L2-normalize the vector for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 1e-10:
            embedding = embedding / norm

        # If chunk already exists, remove it first (update behavior)
        if chunk_id in self.chunk_id_to_int_id:
            self.delete(chunk_id)

        int_id = self._counter
        self._counter += 1

        # Add to index
        self.index.add_with_ids(
            embedding.reshape(1, -1),
            np.array([int_id], dtype=np.int64)
        )

        # Store metadata
        self.chunk_id_to_int_id[chunk_id] = int_id
        self.int_id_to_chunk_id[int_id] = chunk_id
        
        meta_entry = {
            "chunk_id": chunk_id,
            "content": record["content"],
            "source": record["source"],
            "page": record["page"],
            "section": record["section"],
            "embedding": record["embedding"]
        }
        if "child_content" in record:
            meta_entry["child_content"] = record["child_content"]
            
        # Copy extra metadata fields if present
        if "metadata" in record and isinstance(record["metadata"], dict):
            meta_entry["metadata"] = record["metadata"]
            for k, v in record["metadata"].items():
                if k not in meta_entry:
                    meta_entry[k] = v
                    
        # Support root level custom keys
        for k in ["parent_id", "child_id", "is_front_matter"]:
            if k in record and k not in meta_entry:
                meta_entry[k] = record[k]

        self.metadata[chunk_id] = meta_entry


    def add_batch(self, records: List[Dict[str, Any]]) -> None:
        """
        Add a batch of records to the vector store.
        """
        for record in records:
            self.add(record)

    def search(self, query_vector: List[float], top_k: int, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search the vector store for top_k most similar records, optionally filtering by metadata.
        """
        if self.count() == 0:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(q_vec)
        if norm > 1e-10:
            q_vec = q_vec / norm

        search_k = self.count() if filter_dict else min(top_k, self.count())
        if search_k == 0:
            return []

        scores, indices = self.index.search(q_vec.reshape(1, -1), search_k)

        results = []
        for score, idx_val in zip(scores[0], indices[0]):
            idx_val = int(idx_val)
            if idx_val == -1:
                continue
            chunk_id = self.int_id_to_chunk_id.get(idx_val)
            if chunk_id and chunk_id in self.metadata:
                rec = dict(self.metadata[chunk_id])

                # Apply filter
                if filter_dict:
                    match = True
                    for fkey, fval in filter_dict.items():
                        val = rec.get(fkey, rec.get("metadata", {}).get(fkey))
                        if val != fval:
                            match = False
                            break
                    if not match:
                        continue

                rec["score"] = float(score)
                results.append(rec)
                if len(results) >= top_k:
                    break

        return results

    def delete(self, chunk_id: str) -> bool:
        """
        Delete a record by chunk_id.
        """
        if chunk_id not in self.chunk_id_to_int_id:
            return False

        int_id = self.chunk_id_to_int_id[chunk_id]

        # Call remove_ids on FAISS IndexIDMap
        self.index.remove_ids(np.array([int_id], dtype=np.int64))

        del self.chunk_id_to_int_id[chunk_id]
        del self.int_id_to_chunk_id[int_id]
        del self.metadata[chunk_id]

        return True

    def update(self, chunk_id: str, record: Dict[str, Any]) -> bool:
        """
        Update a record by chunk_id.
        """
        if chunk_id not in self.chunk_id_to_int_id:
            return False

        self.delete(chunk_id)
        # Ensure the record has the same chunk_id as specified
        record = dict(record)
        record["chunk_id"] = chunk_id
        self.add(record)
        return True

    def count(self) -> int:
        """
        Return the total number of records.
        """
        return len(self.metadata)

    def save(self, path: str) -> None:
        """
        Save FAISS index and metadata to path directory.
        """
        os.makedirs(path, exist_ok=True)
        index_file = os.path.join(path, "index.faiss")
        metadata_file = os.path.join(path, "metadata.json")

        faiss.write_index(self.index, index_file)

        state = {
            "chunk_id_to_int_id": self.chunk_id_to_int_id,
            "int_id_to_chunk_id": {str(k): v for k, v in self.int_id_to_chunk_id.items()},
            "metadata": self.metadata,
            "counter": self._counter,
            "dimension": self.dimension
        }

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

        logger.info("Saved FAISS index and metadata to: %s", path)

    def load(self, path: str) -> None:
        """
        Load FAISS index and metadata from path directory.
        """
        index_file = os.path.join(path, "index.faiss")
        metadata_file = os.path.join(path, "metadata.json")

        if not os.path.exists(index_file) or not os.path.exists(metadata_file):
            raise FileNotFoundError(f"FAISS index or metadata file not found at: {path}")

        self.index = faiss.read_index(index_file)

        with open(metadata_file, "r", encoding="utf-8") as f:
            state = json.load(f)

        self.chunk_id_to_int_id = state["chunk_id_to_int_id"]
        self.int_id_to_chunk_id = {int(k): v for k, v in state["int_id_to_chunk_id"].items()}
        self.metadata = state["metadata"]
        self._counter = state["counter"]
        self.dimension = state["dimension"]

        # Also rebuild quantizer reference if possible
        # note: read_index recreates the self.index IndexIDMap directly.
        logger.info("Loaded FAISS index and metadata from: %s", path)

    def health_check(self) -> bool:
        """
        Check if FAISS store is healthy.
        """
        try:
            if self.index is None:
                return False
            if self.index.ntotal != len(self.metadata):
                return False
            if self.index.ntotal > 0:
                dummy = np.zeros(self.dimension, dtype=np.float32)
                self.index.search(dummy.reshape(1, -1), 1)
            return True
        except Exception as e:
            logger.error("FAISSStore health check failed: %s", e)
            return False
