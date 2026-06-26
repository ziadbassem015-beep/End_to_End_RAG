from typing import List
import os
import requests
from dotenv import load_dotenv
from langchain_core.documents import Document
import time

load_dotenv("/mnt/d/End_to_End_Agent/2.env")

github_token = os.getenv("GITHUB_TOKEN")

if not github_token:
    raise ValueError("GITHUB_TOKEN not found in environment (check 2.env file)")

headers = {
    "Authorization": f"token {github_token}"
}

response = requests.get(
    "https://api.github.com/user",
    headers=headers
)

print(response.json())


# 🔥 بدل Gemini: summarizer بسيط
def describe_page(text: str) -> str:
    clean = " ".join(text.replace("\n", " ").split())

    if not clean:
        return "Empty or OCR-unreadable page"

    # simple heuristic summary
    words = clean.split()

    if len(words) < 20:
        return clean[:200]

    return " ".join(words[:25]) + "..."


def add_page_descriptions(docs: List[Document]) -> List[Document]:
    for idx, doc in enumerate(docs):
        try:
            summary = describe_page(doc.page_content)

            doc.metadata["page_description"] = summary
            print(f"Processed page {idx + 1}")

            time.sleep(0.2)  # small delay (no API anymore)

        except Exception as e:
            print(f"Failed on page {idx + 1}: {e}")
            doc.metadata["page_description"] = "Metadata generation failed"

    return docs