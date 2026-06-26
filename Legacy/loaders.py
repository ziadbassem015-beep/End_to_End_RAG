from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

DEFAULT_PDF_PATH = "data/row/andrew-ng-machine-learning-yearning.pdf"


def load_pdf_pages(file_path: str) -> List[Document]:
    """
    Load text-based PDF page by page.
    Each page becomes a LangChain Document with page metadata.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    loader = PyPDFLoader(str(path))
    docs = loader.load()

    for doc in docs:
        # PyPDFLoader returns page zero-indexed
        page = int(doc.metadata.get("page", 0)) + 1
        doc.metadata.update({
            "source": path.name,
            "file_path": str(path),
            "page_number": page,
            "loader": "PyPDFLoader",
        })
    return docs


if __name__ == "__main__":
    from metadata import add_page_descriptions

    def needs_ocr(docs: List[Document], min_chars_per_page: int = 40) -> bool:
        if not docs:
            return True
        weak_pages = sum(len(doc.page_content.strip()) < min_chars_per_page for doc in docs)
        return weak_pages / len(docs) > 0.5

    docs = load_pdf_pages(DEFAULT_PDF_PATH)
    if needs_ocr(docs):
        try:
            from ocr import ocr_pdf_pages
        except ModuleNotFoundError as exc:
            missing_package = exc.name
            raise ModuleNotFoundError(
                f"Text extraction found little content and OCR needs '{missing_package}'. "
                f"Install OCR dependencies or add '{missing_package}' to pyproject.toml."
            ) from exc
        docs = ocr_pdf_pages(DEFAULT_PDF_PATH)

    docs = add_page_descriptions(docs)
    first_text_doc = next((doc for doc in docs if doc.page_content.strip()), docs[0])

    print(f"Total pages loaded: {len(docs)}")
    print(f"Loader used: {first_text_doc.metadata.get('loader')}")
    print(f"Preview page: {first_text_doc.metadata.get('page_number')}")
    print(first_text_doc.page_content[:500].strip())
    print(first_text_doc.metadata)
