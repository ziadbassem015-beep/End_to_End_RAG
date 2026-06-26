"""
ingestion/loader.py
===================
Production-grade PDF loader with OCR fallback.

Responsibilities:
- Load PDF pages via PyPDFLoader
- Detect weak text extraction and trigger OCR
- Normalize page metadata
- Return typed Document objects
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)


# ─── Data Contract ───────────────────────────────────────────────────────────

class LoadedPage:
    """Thin wrapper around a single PDF page with normalized metadata."""

    __slots__ = ("page_content", "metadata")

    def __init__(self, content: str, metadata: dict) -> None:
        self.page_content: str = content
        self.metadata: dict = metadata

    def __repr__(self) -> str:
        return (
            f"<LoadedPage page={self.metadata.get('page_number')} "
            f"chars={len(self.page_content)}>"
        )


# ─── Loader ──────────────────────────────────────────────────────────────────

class PDFLoader:
    """Load a PDF file into a list of page-level documents."""

    # A page with fewer characters than this is considered "weak" (likely image-only)
    MIN_CHARS_PER_PAGE: int = 40

    def __init__(
        self,
        pdf_path: str | Path,
        ocr_fallback: bool = True,
        min_chars_for_ocr: float = 0.5,
    ) -> None:
        """
        Args:
            pdf_path:         Path to the PDF file.
            ocr_fallback:     Whether to run OCR when text extraction is poor.
            min_chars_for_ocr: Fraction of weak pages that triggers OCR (0.5 = 50%).
        """
        self.pdf_path = Path(pdf_path)
        self.ocr_fallback = ocr_fallback
        self.min_chars_for_ocr = min_chars_for_ocr

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

    # ── Public ───────────────────────────────────────────────────────────────

    def load(self) -> List[Document]:
        """
        Load the PDF and return one LangChain Document per page.

        Each document has the following metadata keys:
            source       — PDF filename
            file_path    — Absolute path to the PDF
            page_number  — 1-indexed page number
            loader       — Loader name used ("PyPDFLoader" or "OCR")
        """
        logger.info("Loading PDF: %s", self.pdf_path)
        docs = self._load_text()

        if self.ocr_fallback and self._needs_ocr(docs):
            logger.warning(
                "Text extraction produced weak content — attempting OCR fallback."
            )
            docs = self._load_ocr()

        logger.info("Loaded %d pages from %s", len(docs), self.pdf_path.name)
        return docs

    # ── Private ──────────────────────────────────────────────────────────────

    def _load_text(self) -> List[Document]:
        """Use PyPDFLoader for standard text extraction."""
        loader = PyPDFLoader(str(self.pdf_path))
        raw_docs = loader.load()
        return [self._normalize_metadata(doc, loader_name="PyPDFLoader") for doc in raw_docs]

    def _load_ocr(self) -> List[Document]:
        """Fallback to OCR-based extraction."""
        try:
            from rag_system.ingestion.ocr import ocr_pdf_pages
        except ImportError as exc:
            raise ImportError(
                "OCR fallback requires 'pytesseract' and 'Pillow'. "
                "Install them or set ocr_fallback=False."
            ) from exc

        raw_docs = ocr_pdf_pages(str(self.pdf_path))
        return [self._normalize_metadata(doc, loader_name="OCR") for doc in raw_docs]

    def _normalize_metadata(self, doc: Document, loader_name: str) -> Document:
        """Standardize metadata keys across all loader backends."""
        # PyPDFLoader uses zero-indexed 'page'; convert to 1-indexed 'page_number'
        raw_page = int(doc.metadata.get("page", 0))
        doc.metadata.update(
            {
                "source": self.pdf_path.name,
                "file_path": str(self.pdf_path.resolve()),
                "page_number": raw_page + 1,
                "loader": loader_name,
            }
        )
        return doc

    def _needs_ocr(self, docs: List[Document]) -> bool:
        """Return True if too many pages have very little extracted text."""
        if not docs:
            return True
        weak = sum(
            1 for d in docs if len(d.page_content.strip()) < self.MIN_CHARS_PER_PAGE
        )
        ratio = weak / len(docs)
        logger.debug("OCR check: %d/%d weak pages (%.0f%%)", weak, len(docs), ratio * 100)
        return ratio > self.min_chars_for_ocr


# ─── Convenience function ────────────────────────────────────────────────────

def load_pdf(
    pdf_path: str | Path,
    ocr_fallback: bool = True,
) -> List[Document]:
    """Convenience wrapper around PDFLoader.load()."""
    return PDFLoader(pdf_path, ocr_fallback=ocr_fallback).load()
