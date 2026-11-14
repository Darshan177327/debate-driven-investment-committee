
import os
import time
import argparse
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

from tools.document_reader import extract_pdf_pages


# ==================================================
# 1. CONFIGURATION
# ==================================================

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = (
    PROJECT_ROOT
    / "documents"
    / "sample_reports"
    / "JIOFIN_Annual_Report_FY25.pdf"
)

CHROMA_PATH = PROJECT_ROOT / "memory" / "vector_store"

COLLECTION_NAME = "jiofin_annual_report"

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# Keep requests below the reported free-tier limit.
BATCH_SIZE = 20
EMBEDDING_PAUSE_SECONDS = 15

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found. Check your .env file."
    )

gemini_client = genai.Client(api_key=API_KEY)

CHROMA_PATH.mkdir(parents=True, exist_ok=True)

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)


# ==================================================
# 2. TEXT CHUNKING
# ==================================================

def create_text_chunks(pages):
    """
    Split PDF pages into overlapping chunks.

    Every chunk keeps its original PDF page number.
    """

    chunks = []

    for page in pages:

        page_number = page["page"]
        page_text = page["text"]

        if not page_text or not page_text.strip():
            continue

        start = 0
        chunk_index = 0

        while start < len(page_text):

            end = start + CHUNK_SIZE

            chunk_text = page_text[start:end].strip()

            if chunk_text:

                chunks.append({
                    "id": f"page_{page_number}_chunk_{chunk_index}",
                    "text": chunk_text,
                    "page": page_number,
                    "chunk_index": chunk_index
                })

            if end >= len(page_text):
                break

            start = end - CHUNK_OVERLAP
            chunk_index += 1

    return chunks


# ==================================================
# 3. GEMINI EMBEDDING GENERATION
# ==================================================

def generate_embeddings(texts, task_type):
    """
    Generate Gemini embeddings.

    Document chunks use RETRIEVAL_DOCUMENT.
    Search queries use RETRIEVAL_QUERY.

    The caller controls batch size and request pacing.
    """

    if not texts:
        return []

    response = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIMENSIONS
        )
    )

    if not response.embeddings:
        raise RuntimeError(
            "Gemini returned no embeddings."
        )

    if len(response.embeddings) != len(texts):
        raise RuntimeError(
            "Embedding count does not match input count."
        )

    all_embeddings = []

    for embedding in response.embeddings:

        values = embedding.values

        if not values or len(values) != EMBEDDING_DIMENSIONS:
            raise RuntimeError(
                "Unexpected embedding dimensions."
            )

        all_embeddings.append(values)

    return all_embeddings


# ==================================================
# 4. CHROMADB COLLECTION
# ==================================================

def get_collection():
    """
    Get or create the persistent ChromaDB collection.
    """

    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "Jio Financial Services FY2025 Annual Report",
            "hnsw:space": "cosine"
        }
    )


# ==================================================
# 5. INDEX ANNUAL REPORT
# ==================================================

def index_annual_report(force_rebuild=False):
    """
    Extract, chunk, embed and store the annual report.

    Each successful batch is saved immediately.

    If indexing is interrupted, the next run skips
    already saved chunks and continues.
    """

    print("\n===== ANNUAL REPORT RAG INDEXING =====")

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Annual report not found: {PDF_PATH}"
        )

    # Extract PDF pages.
    print("[RAG] Extracting PDF pages...")

    pages = extract_pdf_pages(str(PDF_PATH))

    print(f"[RAG] Extracted {len(pages)} pages.")

    # Create chunks.
    print("[RAG] Creating overlapping text chunks...")

    chunks = create_text_chunks(pages)

    if not chunks:
        raise RuntimeError(
            "No text chunks were extracted from the PDF."
        )

    print(f"[RAG] Created {len(chunks)} text chunks.")

    # Rebuild only this project's collection when requested.
    if force_rebuild:

        try:
            chroma_client.delete_collection(
                name=COLLECTION_NAME
            )

            print("[RAG] Previous collection deleted.")

        except Exception:
            pass

    collection = get_collection()

    existing_count = collection.count()

    if existing_count == len(chunks):

        print("[RAG] All chunks are already indexed.")
        print(f"[RAG] Stored vectors: {existing_count}")

        return existing_count

    print(
        f"[RAG] Existing chunks: "
        f"{existing_count}/{len(chunks)}"
    )

    # Process one batch at a time.
    total_batches = (
        len(chunks) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    for batch_number, start in enumerate(
        range(0, len(chunks), BATCH_SIZE),
        start=1
    ):

        batch = chunks[start:start + BATCH_SIZE]

        batch_ids = [
            chunk["id"]
            for chunk in batch
        ]

        # Find chunks already saved in earlier runs.
        existing = collection.get(
            ids=batch_ids
        )

        existing_ids = set(existing["ids"])

        pending = [
            chunk
            for chunk in batch
            if chunk["id"] not in existing_ids
        ]

        if not pending:

            print(
                f"[RAG] Batch {batch_number}/{total_batches}: "
                "already saved, skipping."
            )

            continue

        print(
            f"\n[RAG] Processing batch "
            f"{batch_number}/{total_batches}"
        )

        print(
            f"[RAG] Generating embeddings for "
            f"{len(pending)} chunks..."
        )

        texts = [
            chunk["text"]
            for chunk in pending
        ]

        # Generate embeddings for this batch.
        embeddings = generate_embeddings(
            texts,
            task_type="RETRIEVAL_DOCUMENT"
        )

        # Save immediately after successful embedding.
        collection.add(
            ids=[
                chunk["id"]
                for chunk in pending
            ],

            documents=texts,

            embeddings=embeddings,

            metadatas=[
                {
                    "page": chunk["page"],
                    "chunk_index": chunk["chunk_index"],
                    "source": PDF_PATH.name
                }
                for chunk in pending
            ]
        )

        stored_count = collection.count()

        print(
            f"[CHROMADB] Saved successfully. "
            f"Total stored: {stored_count}/{len(chunks)}"
        )

        # Wait before next API request.
        if batch_number < total_batches:

            print(
                f"[RAG] Waiting "
                f"{EMBEDDING_PAUSE_SECONDS} seconds "
                "before the next batch..."
            )

            time.sleep(EMBEDDING_PAUSE_SECONDS)

    final_count = collection.count()

    print("\n===== INDEXING SUMMARY =====")
    print("PDF pages:", len(pages))
    print("Expected chunks:", len(chunks))
    print("Stored vectors:", final_count)
    print("Storage location:", CHROMA_PATH)

    if final_count != len(chunks):

        raise RuntimeError(
            f"Index incomplete: "
            f"{final_count}/{len(chunks)} chunks stored."
        )

    print("\n[RAG] Annual report indexing completed successfully.")

    return final_count


# ==================================================
# 6. SEMANTIC SEARCH
# ==================================================

def semantic_search(query, top_k=5):
    """
    Retrieve relevant annual-report passages
    using semantic similarity.

    Returns:
        page
        chunk_index
        distance
        text
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError(
            "Search query cannot be empty."
        )

    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or top_k < 1
    ):
        raise ValueError(
            "top_k must be a positive integer."
        )

    collection = get_collection()

    stored_count = collection.count()

    if stored_count == 0:

        raise RuntimeError(
            "RAG index is empty. "
            "Run index_annual_report() first."
        )

    print(f"\n[RAG SEARCH] Query: {query}")

    # Embed the user's question.
    query_embedding = generate_embeddings(
        [query],
        task_type="RETRIEVAL_QUERY"
    )[0]

    # Search ChromaDB.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, stored_count),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    matches = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        matches.append({
            "page": metadata["page"],
            "chunk_index": metadata["chunk_index"],
            "distance": distance,
            "text": document
        })

    print(
        f"[RAG SEARCH] Retrieved "
        f"{len(matches)} passages."
    )

    return matches


# ==================================================
# 7. COMMAND-LINE EXECUTION
# ==================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Jio Financial Services Annual Report RAG System"
        )
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and rebuild the RAG index."
    )

    parser.add_argument(
        "--query",
        type=str,
        default=(
            "What was the company's total income "
            "and profit in FY2025?"
        )
    )

    args = parser.parse_args()

    # Build or resume the index.
    index_annual_report(
        force_rebuild=args.rebuild
    )

    # Perform semantic search.
    print("\n===== SEMANTIC SEARCH RESULTS =====")

    results = semantic_search(
        args.query,
        top_k=5
    )

    for number, result in enumerate(results, start=1):

        print(f"\n--- RESULT {number} ---")

        print("PDF Page:", result["page"])

        print(
            "Distance:",
            round(result["distance"], 4)
        )

        print("Text:")
        print(result["text"][:1000])
