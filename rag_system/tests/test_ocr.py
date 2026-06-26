"""
rag_system/tests/test_ocr.py
============================
Unit tests for the OCR ingestion module.
"""

import pytest
from unittest.mock import MagicMock, patch
from rag_system.ingestion.ocr import ocr_pdf_pages


@patch("pypdfium2.PdfDocument")
@patch("pytesseract.image_to_string")
def test_ocr_pdf_pages_success(mock_image_to_string, mock_pdf_document):
    """Test successful OCR processing of a mock PDF page."""
    # Setup mock structure
    mock_pdf = MagicMock()
    mock_page = MagicMock()
    mock_bitmap = MagicMock()
    mock_pil_img = MagicMock()

    mock_pdf_document.return_value = mock_pdf
    mock_pdf.__len__.return_value = 1
    mock_pdf.__iter__.return_value = [mock_page]

    mock_page.render.return_value = mock_bitmap
    mock_bitmap.to_pil.return_value = mock_pil_img
    mock_image_to_string.return_value = "Hello World from OCR!"

    with patch("pathlib.Path.exists", return_value=True):
        docs = ocr_pdf_pages("dummy_scanned.pdf")

        assert len(docs) == 1
        assert docs[0].page_content == "Hello World from OCR!"
        assert docs[0].metadata["page_number"] == 1
        assert docs[0].metadata["loader"] == "OCR"


@patch("pypdfium2.PdfDocument")
@patch("pytesseract.image_to_string")
def test_ocr_pdf_pages_failure(mock_image_to_string, mock_pdf_document):
    """Test OCR error handling when rendering fails on a page."""
    mock_pdf = MagicMock()
    mock_page = MagicMock()

    mock_pdf_document.return_value = mock_pdf
    mock_pdf.__len__.return_value = 1
    mock_pdf.__iter__.return_value = [mock_page]

    # Force page render to raise an exception
    mock_page.render.side_effect = RuntimeError("Failed to render page")

    with patch("pathlib.Path.exists", return_value=True):
        docs = ocr_pdf_pages("dummy_scanned.pdf")

        assert len(docs) == 1
        assert docs[0].page_content == ""
        assert docs[0].metadata["page_number"] == 1
        assert docs[0].metadata["loader"] == "OCR_FAILED"
