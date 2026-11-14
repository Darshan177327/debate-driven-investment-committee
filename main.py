
from tools.document_reader import (
    extract_pdf_pages,
    search_document
)

print("=== INVESTMENT RESEARCH TOOL ===")

pdf_path = input("Enter PDF file path: ")

try:
    pages = extract_pdf_pages(pdf_path)

    print("Total pages:", len(pages))

    keyword = input("Enter keyword: ")

    results = search_document(pages, keyword)

    for result in results:
        print("\nPage:", result["page"])
        print(result["passage"][:500])

    if not results:
        print("No matching information found.")

except Exception as error:
    print("Error:", error)
