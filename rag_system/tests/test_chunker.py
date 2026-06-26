"""
tests/test_chunker.py
=====================
Tests for the DocumentChunker.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List

import pytest
from langchain_core.documents import Document

from rag_system.ingestion.chunker import DocumentChunker, Chunk, make_chunk_id, derive_prefix


def make_docs(texts: List[str], source: str = "test.pdf", page_start: int = 1) -> List[Document]:
    """Create fake LangChain Documents for testing."""
    docs = []
    for i, text in enumerate(texts):
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": source,
                    "file_path": f"/data/{source}",
                    "page_number": page_start + i,
                    "loader": "PyPDFLoader",
                },
            )
        )
    return docs


SAMPLE_TEXT = (
    "Machine learning is a branch of artificial intelligence. "
    "It enables systems to learn from data and improve over time "
    "without being explicitly programmed. This has applications in "
    "image recognition, natural language processing, and many other fields. "
    "Deep learning is a subset of machine learning that uses neural networks."
)

LONG_TEXT = SAMPLE_TEXT * 10  # ~1,500 chars


class TestPrefixDerivation:
    def test_known_document(self):
        assert derive_prefix("andrew-ng-machine-learning-yearning.pdf") == "MLY"
        assert derive_prefix("ml-yearning.pdf") == "MLY"

    def test_multi_word_document(self):
        assert derive_prefix("my-awesome-document.pdf") == "MAD"

    def test_single_word_document(self):
        assert derive_prefix("document.pdf") == "DOC"


class TestMakeChunkId:
    def test_deterministic(self):
        id1 = make_chunk_id("MLY", 12, 3)
        id2 = make_chunk_id("MLY", 12, 3)
        assert id1 == id2

    def test_format_correct(self):
        cid = make_chunk_id("MLY", 12, 3)
        assert cid == "MLY_P012_C003"


class TestDocumentChunker:
    def test_basic_chunking_produces_chunks(self):
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
        docs = make_docs([LONG_TEXT])
        chunks = chunker.chunk(docs)
        assert len(chunks) > 0

    def test_all_chunks_meet_min_length(self):
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20, min_chunk_length=50)
        docs = make_docs([LONG_TEXT])
        chunks = chunker.chunk(docs)
        for chunk in chunks:
            assert len(chunk.content) >= 50

    def test_no_duplicate_chunk_ids(self):
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
        docs = make_docs([LONG_TEXT, LONG_TEXT[:500]])
        chunks = chunker.chunk(docs)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found!"

    def test_chunk_ids_are_deterministic(self):
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
        docs = make_docs([LONG_TEXT])

        chunks_run1 = chunker.chunk(docs)
        chunks_run2 = chunker.chunk(docs)

        ids1 = [c.chunk_id for c in chunks_run1]
        ids2 = [c.chunk_id for c in chunks_run2]
        assert ids1 == ids2, "Chunk IDs are not deterministic!"

    def test_metadata_is_populated(self):
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        docs = make_docs([LONG_TEXT], source="test.pdf", page_start=3)
        chunks = chunker.chunk(docs)
        for chunk in chunks:
            assert chunk.metadata.source == "test.pdf"
            assert chunk.metadata.page == 3
            assert chunk.metadata.section != ""

    def test_section_inferred_from_first_line(self):
        text = "Introduction to Machine Learning\n\nThis chapter covers basic concepts."
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
        docs = make_docs([text])
        chunks = chunker.chunk(docs)
        assert chunks[0].metadata.section == "Introduction to Machine Learning"

    def test_duplicate_content_deduplicated(self):
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        identical = SAMPLE_TEXT
        docs = make_docs([identical, identical])
        chunks = chunker.chunk(docs)
        contents = [c.content for c in chunks]
        assert len(contents) == len(set(contents)), "Duplicate chunks not deduplicated!"

    def test_short_chunks_filtered(self):
        short_text = "Too short."
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50, min_chunk_length=50)
        docs = make_docs([short_text])
        chunks = chunker.chunk(docs)
        for chunk in chunks:
            assert len(chunk.content) >= 50


class TestChunkerSaveLoad:
    def test_save_and_load_round_trip(self):
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
        docs = make_docs([LONG_TEXT])
        chunks = chunker.chunk(docs)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "chunks.json"
            chunker.save(chunks, out_path)

            assert out_path.exists()
            loaded = DocumentChunker.load(out_path)
            assert isinstance(loaded, list)
            assert len(loaded) == len(chunks)

            saved_ids = [c.chunk_id for c in chunks]
            loaded_ids = [c["chunk_id"] for c in loaded]
            assert saved_ids == loaded_ids

    def test_saved_format_has_required_fields(self):
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
        docs = make_docs([LONG_TEXT])
        chunks = chunker.chunk(docs)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "chunks.json"
            chunker.save(chunks, out_path)

            data = json.loads(out_path.read_text(encoding="utf-8"))
            for item in data:
                assert "chunk_id" in item
                assert "content" in item
                assert "metadata" in item
                meta = item["metadata"]
                assert "source" in meta
                assert "page" in meta
                assert "section" in meta
