"""
rag_system/ingestion/ocr.py
===========================
OCR processing for scanned PDF files using pypdfium2 and pytesseract.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def ocr_pdf_pages(file_path: str | Path, scale: int = 2) -> List[Document]:
    """
    Render PDF pages to PIL images and perform OCR using pytesseract.

    Args:
        file_path: Path to the PDF file.
        scale: Resolution scale factor for rendering pages (higher means better OCR, but slower).

    Returns:
        List of LangChain Document objects, one per page, containing OCR'ed text.
    """
    try:
        import pypdfium2 as pdfium
        from PIL import Image
        import pytesseract
    except ImportError as exc:
        raise ImportError(
            "OCR requires 'pypdfium2', 'pytesseract', and 'Pillow' packages to be installed."
        ) from exc

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found for OCR: {path}")

    logger.info("Initializing OCR for scanned PDF: %s", path.name)
    pdf = pdfium.PdfDocument(str(path))
    docs: List[Document] = []

    for i, page in enumerate(pdf):
        page_num = i + 1
        logger.info("Rendering and running OCR on page %d/%d...", page_num, len(pdf))
        try:
            # Render page to bitmap and convert to PIL Image
            bitmap = page.render(scale=scale).to_pil()
            # Perform text extraction
            text = pytesseract.image_to_string(bitmap)
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path.name,
                        "file_path": str(path.resolve()),
                        "page_number": page_num,
                        "loader": "OCR",
                    },
                )
            )
        except Exception as e:
            logger.error("Failed to run OCR on page %d: %s", page_num, e)
            docs.append(
                Document(
                    page_content="",
                    metadata={
                        "source": path.name,
                        "file_path": str(path.resolve()),
                        "page_number": page_num,
                        "loader": "OCR_FAILED",
                    },
                )
            )

    return docs
