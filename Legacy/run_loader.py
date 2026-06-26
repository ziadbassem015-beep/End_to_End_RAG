from loaders import load_pdf_pages

pdf_path = "data/row/andrew-ng-machine-learning-yearning.pdf"

docs = load_pdf_pages(pdf_path)

print(f"Total pages loaded: {len(docs)}\n")

# اطبع أول صفحة للتأكد
print(docs[0].page_content[:500])
print(docs[0].metadata)
