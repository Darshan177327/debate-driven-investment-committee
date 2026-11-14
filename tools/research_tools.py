
from pathlib import Path

from tools.multi_company_rag import (
    search_company_documents,
    get_company_collection,
)


# ==================================================
# CONFIGURATION
# ==================================================

DEFAULT_COMPANY = "Jio Financial Services"

_active_company = DEFAULT_COMPANY


# ==================================================
# ACTIVE COMPANY MANAGEMENT
# ==================================================

def set_active_company(company_name: str) -> str:
    """
    Select the company whose documents the agents will analyze.

    Call this before starting an investment committee run.
    """

    global _active_company

    if not isinstance(company_name, str) or not company_name.strip():
        raise ValueError("A valid company name is required.")

    company_name = company_name.strip()

    collection = get_company_collection(company_name)

    if collection.count() == 0:
        raise ValueError(
            f"No documents indexed for {company_name}. "
            "Upload and index documents first."
        )

    _active_company = company_name

    print(
        f"\n[DOCUMENT CONTEXT] Active company: {_active_company}"
    )

    print(
        f"[DOCUMENT CONTEXT] Indexed chunks: {collection.count()}"
    )

    return _active_company


def get_active_company() -> str:
    """Return the currently selected company."""

    return _active_company


# ==================================================
# TOOL 1: KEYWORD SEARCH
# ==================================================

def search_annual_report(keyword: str) -> str:
    """
    Search indexed company documents for an exact keyword.

    Searches the selected company's ChromaDB collection.
    """

    print(
        f"\n[TOOL CALLED] search_annual_report: {keyword}"
    )

    if not isinstance(keyword, str) or not keyword.strip():
        return "Invalid keyword. Please provide a search term."

    try:
        collection = get_company_collection(_active_company)

        if collection.count() == 0:
            return (
                f"No indexed documents found for {_active_company}."
            )

        results = collection.get(
            include=["documents", "metadatas"]
        )

        matches = []
        search_term = keyword.strip().casefold()

        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []

        for document, metadata in zip(documents, metadatas):

            if search_term in document.casefold():

                matches.append({
                    "text": document,
                    "metadata": metadata
                })

        if not matches:
            return (
                f"No exact keyword matches found for "
                f"'{keyword}' in {_active_company}'s documents."
            )

        output = []

        for index, match in enumerate(matches[:10], start=1):

            metadata = match["metadata"]

            output.append(
                f"RESULT {index}\n"
                f"COMPANY: {_active_company}\n"
                f"SOURCE FILE: {metadata.get('filename', 'Unknown')}\n"
                f"CHUNK: {metadata.get('chunk_index', 'Unknown')}\n"
                f"EVIDENCE:\n{match['text'][:1800]}"
            )

        print(
            f"[TOOL RESULT] Found {len(matches)} matching chunks."
        )

        return "\n\n".join(output)

    except Exception as error:

        print(f"[TOOL ERROR] {error}")

        return f"Unable to search company documents: {error}"


# ==================================================
# TOOL 2: READ DOCUMENT CHUNK
# ==================================================

def read_annual_report_page(page_number: int) -> str:
    """
    Legacy compatibility function.

    The current multi-company index stores chunk numbers,
    not original PDF page numbers.

    This function therefore does not pretend that a chunk
    number is an original PDF page number.
    """

    print(
        f"\n[TOOL CALLED] read_annual_report_page: {page_number}"
    )

    return (
        "Original PDF page retrieval is not yet available "
        "in the new multi-company index. "
        "The indexed documents currently retain filename "
        "and chunk number, but not original page numbers. "
        "Do not treat a chunk number as a PDF page number. "
        "Use semantic_search_annual_report to retrieve evidence."
    )


# ==================================================
# TOOL 3: SEMANTIC RAG SEARCH
# ==================================================

def semantic_search_annual_report(query: str) -> str:
    """
    Search the active company's documents using local embeddings.

    Uses FastEmbed and ChromaDB.
    Does not consume Gemini embedding API quota.
    """

    print(
        f"\n[TOOL CALLED] semantic_search_annual_report: {query}"
    )

    if not isinstance(query, str) or not query.strip():
        return "Invalid search query."

    try:

        results = search_company_documents(
            company_name=_active_company,
            query=query.strip(),
            top_k=5
        )

        if not results:
            return (
                f"No relevant evidence found for {_active_company}."
            )

        output = []

        for index, result in enumerate(results, start=1):

            metadata = result["metadata"]

            output.append(
                f"RESULT {index}\n"
                f"COMPANY: {_active_company}\n"
                f"SOURCE FILE: {metadata.get('filename', 'Unknown')}\n"
                f"CHUNK: {metadata.get('chunk_index', 'Unknown')}\n"
                f"DISTANCE: {result['distance']:.4f}\n"
                f"EVIDENCE:\n{result['text']}"
            )

        print(
            f"[TOOL RESULT] Retrieved {len(results)} relevant chunks."
        )

        return "\n\n".join(output)

    except Exception as error:

        print(f"[RAG TOOL ERROR] {error}")

        return f"Company document search failed: {error}"


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    print("\nCOMPANY RESEARCH TOOLS")
    print("=" * 45)

    print("Active company:", get_active_company())

    try:

        set_active_company("Jio Financial Services")

        result = semantic_search_annual_report(
            "What was the consolidated profit after tax in FY2025?"
        )

        print("\nSEARCH RESULT:\n")
        print(result)

    except Exception as error:

        print("[TEST ERROR]", error)
