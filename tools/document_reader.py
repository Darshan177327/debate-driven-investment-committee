
import pymupdf
from pathlib import Path


def extract_pdf_pages(pdf_path):

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages = []

    with pymupdf.open(str(path)) as document:

        for page_number, page in enumerate(document, start=1):

            pages.append({
                "page": page_number,
                "text": page.get_text("text")
            })

    return pages


def search_document(pages, keyword):

    results = []

    for page in pages:

        if keyword.lower() in page["text"].lower():

            results.append({
                "page": page["page"],
                "passage": page["text"]
            })

    return results
