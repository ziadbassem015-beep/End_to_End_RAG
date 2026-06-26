"""
embeddings/embedder.py
======================
Production-grade embedding pipeline with disk caching.
Supports BAAI/bge-small-en-v1.5 as default, MiniLM, and E5-small.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ─── Supported Models ────────────────────────────────────────────────────────

SUPPORTED_MODELS = {
    "bge":     "BAAI/bge-small-en-v1.5",
    "minilm":  "sentence-transformers/all-MiniLM-L6-v2",
    "e5":      "intfloat/e5-small-v2",
}


# ─── Cache ───────────────────────────────────────────────────────────────────

class EmbeddingCache:
    """
    Disk-based cache for embeddings.
    Cache key: SHA-256(model_name + chunk_text)
    Cache value: embedding vector stored as JSON
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    def _key_path(self, model: str, text: str) -> Path:
        raw = f"{model}||{text}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        # Shard into subdirs to avoid filesystem limits
        return self.cache_dir / digest[:2] / f"{digest}.json"

    def get(self, model: str, text: str) -> Optional[List[float]]:
        path = self._key_path(model, text)
        if path.exists():
            self._hits += 1
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("Failed to read cache file %s: %s", path, e)
                return None
        self._misses += 1
        return None

    def set(self, model: str, text: str, embedding: List[float]) -> None:
        path = self._key_path(model, text)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(json.dumps(embedding), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to write cache file %s: %s", path, e)

    @property
    def stats(self) -> Dict[str, int]:
        return {"hits": self._hits, "misses": self._misses}


# ─── Embedder ────────────────────────────────────────────────────────────────

class Embedder:
    """
    Batch-embed document chunks using sentence-transformers.
    Default model: BAAI/bge-small-en-v1.5
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 32,
        normalize: bool = True,
        cache_dir: Optional[str | Path] = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.cache: Optional[EmbeddingCache] = (
            EmbeddingCache(cache_dir) if cache_dir else None
        )

        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        logger.info("Model loaded successfully. normalize=%s batch_size=%d", normalize, batch_size)

    def embed(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of chunk dicts.
        """
        self._validate_input(chunks)

        to_embed: List[Tuple[int, str, str]] = []  # (original_index, chunk_id, text)
        results: Dict[str, List[float]] = {}

        for idx, chunk in enumerate(chunks):
            chunk_id = chunk["chunk_id"]
            text = self._prepare_text(chunk)

            if self.cache:
                cached = self.cache.get(self.model_name, text)
                if cached is not None:
                    results[chunk_id] = cached
                    continue

            to_embed.append((idx, chunk_id, text))

        logger.info(
            "Embedding: %d chunks total (%d cache hits, %d to compute)",
            len(chunks),
            len(chunks) - len(to_embed),
            len(to_embed),
        )

        # Batch-encode uncached chunks
        if to_embed:
            texts = [t for _, _, t in to_embed]
            embeddings = self._batch_encode(texts)

            for (idx, chunk_id, text), embedding in zip(to_embed, embeddings):
                vec = embedding.tolist()
                results[chunk_id] = vec
                if self.cache:
                    self.cache.set(self.model_name, text, vec)

        if self.cache:
            logger.info("Embedding Cache stats: %s", self.cache.stats)

        # Build output - preserve structure with top-level metadata fields AND nested metadata
        embedded: List[Dict[str, Any]] = []
        expected_dim = len(next(iter(results.values()))) if results else 384

        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            if chunk_id not in results:
                raise RuntimeError(f"Embedding missing for chunk_id={chunk_id!r}")

            vec = results[chunk_id]

            if len(vec) != expected_dim:
                raise ValueError(
                    f"Dimension mismatch: chunk {chunk_id} has {len(vec)} dims, expected {expected_dim}"
                )

            # Metadata resolution
            meta_dict = chunk.get("metadata", {})
            source = meta_dict.get("source", "")
            page = int(meta_dict.get("page", meta_dict.get("page_number", 1)))
            section = meta_dict.get("section", "")

            # Merge all extra metadata keys
            full_meta = {
                "source": source,
                "page": page,
                "section": section,
            }
            for k, v in meta_dict.items():
                full_meta[k] = v

            rec = {
                "chunk_id": chunk_id,
                "content": chunk["content"],
                "source": source,
                "page": page,
                "section": section,
                "metadata": full_meta,
                "embedding": vec,
                "embedding_model": self.model_name,
                "embedding_dimensions": len(vec),
            }
            # Copy specific top-level metadata values as well if present
            for k in ["parent_id", "child_id", "is_front_matter"]:
                if k in meta_dict:
                    rec[k] = meta_dict[k]

            if "child_content" in chunk:
                rec["child_content"] = chunk["child_content"]
            embedded.append(rec)

        self._validate_output(chunks, embedded)
        logger.info(
            "Embedding generation complete: %d records, dim=%d, normalized=%s",
            len(embedded), expected_dim, self.normalize,
        )
        return embedded

    def save(self, embedded_chunks: List[Dict[str, Any]], output_path: str | Path) -> Path:
        """Save embedded chunks to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(embedded_chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved %d embedded chunks -> %s", len(embedded_chunks), output_path)
        return output_path

    @staticmethod
    def load(embeddings_path: str | Path) -> List[Dict[str, Any]]:
        """Load embedded chunks from JSON."""
        path = Path(embeddings_path)
        if not path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Embeddings file must contain a JSON array.")
        logger.info("Loaded %d embedded chunks from %s", len(data), path)
        return data

    def _batch_encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts in batches with sentence-transformers."""
        all_embeddings: List[np.ndarray] = []

        batches = [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]

        for batch_idx, batch in enumerate(
            tqdm(batches, desc="Encoding Chunks", unit="batch"), 1
        ):
            emb = self._model.encode(
                batch,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            all_embeddings.append(emb)

        return np.vstack(all_embeddings) if all_embeddings else np.array([])

    @staticmethod
    def _prepare_text(chunk: Dict[str, Any]) -> str:
        """Clean chunk text for embedding."""
        text = chunk.get("child_content", chunk.get("content", ""))
        return " ".join(text.split())

    @staticmethod
    def _validate_input(chunks: List[Dict[str, Any]]) -> None:
        """Ensure all chunks have required fields."""
        for i, chunk in enumerate(chunks):
            if "chunk_id" not in chunk:
                raise ValueError(f"Chunk at index {i} is missing 'chunk_id'")
            if "content" not in chunk:
                raise ValueError(f"Chunk {chunk.get('chunk_id')} is missing 'content'")

        ids = [c["chunk_id"] for c in chunks]
        if len(ids) != len(set(ids)):
            from collections import Counter
            dupes = [cid for cid, cnt in Counter(ids).items() if cnt > 1]
            raise ValueError(f"Duplicate chunk_ids found in input: {dupes}")

    @staticmethod
    def _validate_output(
        original: List[Dict[str, Any]],
        embedded: List[Dict[str, Any]],
    ) -> None:
        """Verify that chunk_ids and ordering are preserved after embedding."""
        if len(original) != len(embedded):
            raise RuntimeError(
                f"Output count mismatch: input={len(original)}, output={len(embedded)}"
            )
        for orig, emb in zip(original, embedded):
            if orig["chunk_id"] != emb["chunk_id"]:
                raise RuntimeError(
                    f"chunk_id mismatch: expected {orig['chunk_id']!r}, got {emb['chunk_id']!r}"
                )
