"""
scripts/migrate_json_to_faiss.py
===============================
Migration script to load legacy JSON embeddings and index them into a FAISS VectorStore.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add root folder to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_system.embeddings.embedder import Embedder
from rag_system.vectorstore.faiss_store import FAISSStore


def migrate(json_path: str, faiss_out_dir: str) -> None:
    """Migrate JSON embeddings into a FAISS index directory."""
    print(f"🚀 Starting migration from JSON embeddings: {json_path}")

    # Check existence
    if not os.path.exists(json_path):
        print(f"❌ Error: JSON embeddings file not found at {json_path}")
        sys.exit(1)

    # Load JSON
    try:
        chunks = Embedder.load(json_path)
    except Exception as e:
        print(f"❌ Error loading JSON embeddings: {e}")
        sys.exit(1)

    if not chunks:
        print("⚠️  Warning: No chunks found in JSON embeddings.")
        return

    # Extract dimension
    first_chunk = chunks[0]
    emb = first_chunk.get("embedding")
    if not emb:
        print("❌ Error: Chunks do not contain 'embedding' vectors.")
        sys.exit(1)
        
    dimension = len(emb)
    print(f"📦 Loaded {len(chunks)} embedded chunks. Vector dimension: {dimension}")

    # Initialize FAISS Store
    store = FAISSStore(dimension=dimension)

    # Build migration records conforming to strict metadata schema
    records = []
    for idx, chunk in enumerate(chunks):
        cid = chunk.get("chunk_id", f"MIGRATED_{idx:04d}")
        content = chunk.get("content", "")
        
        # Parse metadata
        meta = chunk.get("metadata", {})
        source = chunk.get("source", meta.get("source", "unknown"))
        page = int(chunk.get("page", meta.get("page", meta.get("page_number", 1))))
        section = chunk.get("section", meta.get("section", ""))
        embedding = chunk["embedding"]

        records.append({
            "chunk_id": cid,
            "content": content,
            "source": source,
            "page": page,
            "section": section,
            "embedding": embedding,
        })

    # Add to store
    print("Indexing vectors into FAISS...")
    store.add_batch(records)

    # Save
    print(f"Saving FAISS index and metadata to: {faiss_out_dir}")
    store.save(faiss_out_dir)

    # Health check
    if store.health_check():
        print(f"✅ Migration successful! Saved FAISS store containing {store.count()} records.")
    else:
        print("❌ Error: FAISS store health check failed after migration.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate JSON embeddings to FAISS store.")
    parser.add_argument(
        "--input",
        default="rag_system/output/embeddings.json",
        help="Path to legacy input JSON embeddings file."
    )
    parser.add_argument(
        "--output-dir",
        default="rag_system/output/faiss_index",
        help="Path to output FAISS index directory."
    )
    args = parser.parse_args()
    migrate(args.input, args.output_dir)
