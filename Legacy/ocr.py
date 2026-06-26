from pathlib import Path
from typing import List
from PIL import Image
import pytesseract
import pypdfium2 as pdfium
from langchain_core.documents import Document


def ocr_pdf_pages(file_path: str, scale: int = 2) -> List[Document]:
    """
    OCR scanned PDF pages using pypdfium2 rendering + pytesseract.
    Requires tesseract installed in the OS.
    Ubuntu: sudo apt-get install -y tesseract-ocr
    """
    path = Path(file_path)
    pdf = pdfium.PdfDocument(str(path))
    docs: List[Document] = []

    for i, page in enumerate(pdf):
        bitmap = page.render(scale=scale).to_pil()
        text = pytesseract.image_to_string(bitmap)
        docs.append(Document(
            page_content=text,
            metadata={
                "source": path.name,
                "file_path": str(path),
                "page_number": i + 1,
                "loader": "OCR",
            }
        ))
    return docs


def needs_ocr(docs: List[Document], min_chars_per_page: int = 40) -> bool:
    """
    If most pages have almost no extracted text, assume scanned PDF.
    """
    if not docs:
        return True
    weak_pages = sum(len(d.page_content.strip()) < min_chars_per_page for d in docs)
    return weak_pages / len(docs) > 0.5