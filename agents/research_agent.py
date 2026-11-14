
from agents.gemini_client import generate_response
from tools.document_reader import extract_pdf_pages, search_document

PDF_PATH = "documents/sample_reports/JIOFIN_Annual_Report_FY25.pdf"


def find_report_evidence(keyword):
    """Tool 1: Extract PDF pages and find matching evidence."""

    pages = extract_pdf_pages(PDF_PATH)

    results = search_document(pages, keyword)

    evidence = []

    for result in results:
        evidence.append({
            "page": result["page"],
            "passage": result["passage"][:1500]
        })

    return evidence


def analyze_report(question, keyword):
    """Research Agent: Answer using evidence from the annual report."""

    evidence = find_report_evidence(keyword)

    if not evidence:
        return "No matching evidence found in the annual report."

    evidence_text = ""

    for item in evidence[:8]:
        evidence_text += (
            f"\nPage {item['page']}:\n"
            f"{item['passage']}\n"
        )

    prompt = f"""
You are the Research Agent of a Debate-Driven Investment Committee.

Your task is to answer the research question using ONLY the
evidence provided from the annual report.

Research question:
{question}

Evidence from the annual report:
{evidence_text}

Instructions:
1. Use only the provided evidence.
2. Do not invent financial figures.
3. Mention the page number for every important fact.
4. Clearly state when the evidence is insufficient.
5. Distinguish reported facts from your interpretation.

Provide a structured research report.
"""

    return generate_response(prompt)


if __name__ == "__main__":
    question = input("Enter your research question: ")
    keyword = input("Enter a keyword to search in the report: ")

    print("\n=== RESEARCH AGENT REPORT ===\n")
    print(analyze_report(question, keyword))
