import os
import json
from dotenv import load_dotenv

from loaders import load_pdf_pages
from metadata import add_page_descriptions

load_dotenv("/mnt/d/End_to_End_Agent/2.env")

PDF_PATH = "/mnt/d/End_to_End_Agent/data/row/andrew-ng-machine-learning-yearning.pdf"
OUTPUT_PATH = "output_metadata.json"


def main():
    print("Loading PDF...")
    docs = load_pdf_pages(PDF_PATH)

    print(f"Loaded {len(docs)} pages")

    print("Generating metadata summaries...")
    docs = add_page_descriptions(docs)

    print("Done processing all pages")

    # preview أول 3 صفحات
    for i, doc in enumerate(docs[:3]):
        print("-" * 50)
        print(f"Page {i + 1} description:")
        print(doc.metadata.get("page_description"))

    # 💾 Save JSON output
    results = []

    for i, doc in enumerate(docs):
        results.append({
            "page": i + 1,
            "page_description": doc.metadata.get("page_description"),
            "content_preview": doc.page_content[:300]
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved JSON → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()