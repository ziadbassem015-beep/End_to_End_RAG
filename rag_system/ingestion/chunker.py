"""
ingestion/chunker.py
====================
Document chunker that splits pages into validated, deduplicated, and ID-stable chunks.
Generates human-readable deterministic IDs (e.g., MLY_P012_C003).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Strict metadata schema required by the RAG framework."""
    source: str
    page: int
    section: str
    parent_id: Optional[str] = None
    child_id: Optional[str] = None
    is_front_matter: bool = False


@dataclass
class Chunk:
    """
    A single document chunk with deterministic human-readable ID.
    """
    chunk_id: str
    content: str
    metadata: ChunkMetadata
    child_content: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": asdict(self.metadata),
        }
        if self.child_content is not None:
            d["child_content"] = self.child_content
        return d


def derive_prefix(source_name: str) -> str:
    """
    Derive a deterministic 3-4 letter uppercase prefix from the source file name.
    E.g., 'andrew-ng-machine-learning-yearning.pdf' -> 'MLY'
    """
    base = os.path.splitext(os.path.basename(source_name))[0]
    
    # Custom rule for Andrew Ng's Machine Learning Yearning
    if "machine-learning-yearning" in base.lower() or "ml-yearning" in base.lower():
        return "MLY"
        
    # Standard initial extractor
    cleaned = re.sub(r'[^a-zA-Z\s_-]', '', base)
    words = re.split(r'[\s_-]+', cleaned)
    words = [w for w in words if w]
    
    if len(words) >= 3:
        # Take first letter of each word
        return "".join(w[0].upper() for w in words)[:4]
    elif len(base) >= 3:
        # Take first 3 letters
        return base[:3].upper()
    return "DOC"


def make_chunk_id(prefix: str, page_number: int, position_index: int) -> str:
    """
    Generate a human-readable deterministic chunk ID.
    Format: PREFIX_P{page_number:03d}_C{position_index:03d}
    E.g.: MLY_P012_C003
    """
    return f"{prefix}_P{page_number:03d}_C{position_index:03d}"


def check_front_matter(content: str, page: int) -> bool:
    """
    Detect if the content represents a front matter page (copyright page, TOC, ISBN, etc.).
    """
    content_lower = content.lower()
    if page <= 6:
        # Check copyright indicators or TOC indicators
        copyright_indicators = [
            "copyright", "all rights reserved", "isbn", "published by", 
            "deeplearning.ai", "table of contents", "contents"
        ]
        if any(ind in content_lower for ind in copyright_indicators):
            return True
        
        # Check dotted line density indicating table of contents
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if lines:
            dotted_lines = sum(
                1 for line in lines
                if "..." in line or "···" in line or re.search(r'\s+\d+$', line)
            )
            if dotted_lines / len(lines) > 0.4:
                return True
    return False


class DocumentChunker:
    """
    Split loaded documents into validated, ID-stable chunks.
    Supports Parent-Child hierarchical split.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        min_chunk_length: int = 50,
        separators: Optional[List[str]] = None,
        child_chunk_size: int = 0,
        child_chunk_overlap: int = 0,
        use_parent_child: bool = False,
        parent_size: int = 1600,
        parent_overlap: int = 300,
        child_size: int = 400,
        child_overlap: int = 100,
    ) -> None:
        # If parent-child mode is active, override standard parameters
        if use_parent_child:
            self.chunk_size = parent_size
            self.chunk_overlap = parent_overlap
            self.child_chunk_size = child_size
            self.child_chunk_overlap = child_overlap
        else:
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.child_chunk_size = child_chunk_size
            self.child_chunk_overlap = child_chunk_overlap

        self.min_chunk_length = min_chunk_length
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )

        self._child_splitter = None
        if self.child_chunk_size > 0:
            self._child_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.child_chunk_size,
                chunk_overlap=self.child_chunk_overlap,
                separators=self.separators,
                length_function=len,
            )

        logger.info(
            "DocumentChunker: size=%d overlap=%d child_size=%d child_overlap=%d min_len=%d",
            self.chunk_size,
            self.chunk_overlap,
            self.child_chunk_size,
            self.child_chunk_overlap,
            self.min_chunk_length,
        )

    def chunk(self, docs: List[Document]) -> List[Chunk]:
        """
        Split a list of loaded pages into validated Chunk objects.
        Supports Parent-Child chunking if child_chunk_size > 0 or parent-child is explicitly configured.
        """
        raw_splits = self._splitter.split_documents(docs)
        logger.info("Splitter produced %d raw splits from %d pages", len(raw_splits), len(docs))

        chunks: List[Chunk] = []
        seen_content_hashes: Set[str] = set()

        # Track per-page position for deterministic IDs
        parent_position_counter: Dict[tuple, int] = {}
        child_position_counter: Dict[tuple, int] = {}

        for parent_split in raw_splits:
            parent_content = parent_split.page_content.strip()
            if not parent_content:
                continue

            meta = parent_split.metadata
            source = meta.get("source", "unknown")
            source_filename = os.path.basename(source)
            page_number = int(meta.get("page_number", meta.get("page", 1)) or 1)
            if page_number < 1:
                page_number = 1

            page_key = (source_filename, page_number)
            is_front = check_front_matter(parent_content, page_number)
            prefix = derive_prefix(source_filename)

            # Generate unique parent ID
            p_pos = parent_position_counter.get(page_key, 1)
            parent_position_counter[page_key] = p_pos + 1
            parent_id = f"{prefix}_P{page_number:03d}_PAR{p_pos:03d}"

            if self._child_splitter is not None:
                child_texts = self._child_splitter.split_text(parent_content)
                for child_text in child_texts:
                    child_text = child_text.strip()
                    if not child_text:
                        continue
                    if len(child_text) < self.min_chunk_length:
                        continue

                    content_hash = hashlib.md5(child_text.encode("utf-8")).hexdigest()
                    if content_hash in seen_content_hashes:
                        continue
                    seen_content_hashes.add(content_hash)

                    c_pos = child_position_counter.get(page_key, 1)
                    child_position_counter[page_key] = c_pos + 1
                    child_id = make_chunk_id(prefix, page_number, c_pos)

                    section = self._infer_section(child_text) or self._infer_section(parent_content)

                    chunk = Chunk(
                        chunk_id=child_id,
                        content=parent_content,
                        metadata=ChunkMetadata(
                            source=source_filename,
                            page=page_number,
                            section=section,
                            parent_id=parent_id,
                            child_id=child_id,
                            is_front_matter=is_front,
                        ),
                        child_content=child_text,
                    )
                    chunks.append(chunk)
            else:
                if len(parent_content) < self.min_chunk_length:
                    continue
                content_hash = hashlib.md5(parent_content.encode("utf-8")).hexdigest()
                if content_hash in seen_content_hashes:
                    continue
                seen_content_hashes.add(content_hash)

                c_pos = child_position_counter.get(page_key, 1)
                child_position_counter[page_key] = c_pos + 1
                child_id = make_chunk_id(prefix, page_number, c_pos)
                section = self._infer_section(parent_content)

                chunk = Chunk(
                    chunk_id=child_id,
                    content=parent_content,
                    metadata=ChunkMetadata(
                        source=source_filename,
                        page=page_number,
                        section=section,
                        parent_id=parent_id,
                        child_id=child_id,
                        is_front_matter=is_front,
                    ),
                    child_content=None,
                )
                chunks.append(chunk)

        logger.info(
            "Chunking complete: %d valid chunks",
            len(chunks),
        )

        self._log_quality_report(chunks)
        return chunks

    def save(self, chunks: List[Chunk], output_path: str | Path) -> Path:
        """
        Save chunks to a JSON file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [c.to_dict() for c in chunks]
        output_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved %d chunks -> %s", len(chunks), output_path)
        return output_path

    @staticmethod
    def load(chunks_path: str | Path) -> List[dict]:
        """
        Load chunks from a saved JSON file.
        """
        path = Path(chunks_path)
        if not path.exists():
            raise FileNotFoundError(f"Chunks file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Chunks file must contain a JSON array.")
        logger.info("Loaded %d chunks from %s", len(data), path)
        return data

    @staticmethod
    def _infer_section(content: str) -> str:
        """
        Infer the section name from the first non-empty line of the chunk.
        """
        for line in content.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped[:100]
        return ""

    def _log_quality_report(self, chunks: List[Chunk]) -> None:
        """Log a quality summary of the chunking output."""
        if not chunks:
            logger.warning("No chunks produced!")
            return

        lengths = [len(c.content) for c in chunks]
        avg_len = sum(lengths) / len(lengths)
        min_len = min(lengths)
        max_len = max(lengths)

        logger.info(
            "Chunk quality: count=%d avg_chars=%.0f min=%d max=%d",
            len(chunks), avg_len, min_len, max_len,
        )

