
import re
import hashlib
from pathlib import Path

import chromadb
from fastembed import TextEmbedding

from tools.universal_document_processor import extract_document


# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VECTOR_STORE_PATH = (
    PROJECT_ROOT / "memory" / "vector_store"
)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
BATCH_SIZE = 32

VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

print("[RAG] Loading local embedding model...")

embedding_model = TextEmbedding(
    model_name=MODEL_NAME
)

print("[RAG] Local embedding model ready.")

chroma_client = chromadb.PersistentClient(
    path=str(VECTOR_STORE_PATH)
)


# ==================================================
# COMPANY COLLECTIONS
# ==================================================

def make_collection_name(company_name: str) -> str:
    """Create a unique collection for each company."""

    if not company_name or not company_name.strip():
        raise ValueError("Company name cannot be empty.")

    slug = re.sub(
        r"[^a-z0-9]+",
        "_",
        company_name.lower()
    ).strip("_")[:35]

    suffix = hashlib.sha256(
        company_name.strip().lower().encode("utf-8")
    ).hexdigest()[:8]

    return f"company_v2_{slug}_{suffix}"


def get_company_collection(company_name: str):
    """Get or create a company-specific 384D collection."""

    name = make_collection_name(company_name)

    collection = chroma_client.get_or_create_collection(
        name=name,
        metadata={
            "company_name": company_name.strip(),
            "embedding_model": MODEL_NAME,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "version": "2"
        }
    )

    # Prevent accidental mixing of embedding dimensions.
    metadata = collection.metadata or {}

    if metadata.get("embedding_dimension") != EMBEDDING_DIMENSION:
        raise ValueError(
            "Collection embedding dimension mismatch. "
            "Use a new collection before indexing."
        )

    return collection


# ==================================================
# TEXT CHUNKING
# ==================================================

def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[str]:

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap must be non-negative and smaller than chunk size."
        )

    chunks = []
    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        if end < len(text):
            boundary = text.rfind(" ", start, end)

            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# ==================================================
# LOCAL EMBEDDINGS
# ==================================================

def create_document_embeddings(
    texts: list[str]
) -> list[list[float]]:
    """Generate document vectors locally."""

    if not texts:
        return []

    vectors = list(
        embedding_model.embed(
            texts,
            batch_size=BATCH_SIZE
        )
    )

    embeddings = [
        vector.tolist() for vector in vectors
    ]

    if len(embeddings) != len(texts):
        raise RuntimeError(
            "Embedding count does not match document count."
        )

    return embeddings


def create_query_embedding(query: str) -> list[float]:
    """Generate a local search-query vector."""

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    # BGE models use an instruction prefix for queries.
    query_text = (
        "Represent this sentence for searching relevant passages: "
        + query.strip()
    )

    vectors = list(
        embedding_model.query_embed(query_text)
    )

    if not vectors:
        raise RuntimeError("Could not generate query embedding.")

    return vectors[0].tolist()


# ==================================================
# DOCUMENT INDEXING
# ==================================================

def index_document(
    company_name: str,
    file_path: str
) -> dict:
    """Extract, chunk and index a document locally."""

    path = Path(file_path)

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(
            f"Document not found: {path}"
        )

    print("\n[INDEXING] Document:", path.name)

    document = extract_document(str(path))

    full_text = document.get("full_text", "")

    if not full_text or not full_text.strip():
        raise ValueError(
            f"No readable text found in {path.name}"
        )

    chunks = split_text(full_text)

    if not chunks:
        raise ValueError("No chunks generated.")

    collection = get_company_collection(company_name)

    file_hash = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    # Skip exact duplicate document content.
    existing = collection.get(
        where={"document_hash": file_hash},
        include=["metadatas"]
    )

    if existing["ids"]:
        print("[SKIPPED] This document is already indexed.")

        return {
            "company": company_name,
            "filename": path.name,
            "chunks_indexed": 0,
            "status": "Already indexed",
            "collection": collection.name
        }

    ids = []
    metadatas = []

    for index, chunk in enumerate(chunks):

        chunk_id = hashlib.sha256(
            f"{file_hash}_{index}".encode("utf-8")
        ).hexdigest()

        ids.append(chunk_id)

        metadatas.append({
            "company": company_name.strip(),
            "filename": path.name,
            "file_type": path.suffix.lower(),
            "document_hash": file_hash,
            "chunk_index": index,
            "total_chunks": len(chunks)
        })

    print("[INDEXING] Total chunks:", len(chunks))
    print("[INDEXING] Generating local embeddings...")

    for start in range(0, len(chunks), BATCH_SIZE):

        end = min(start + BATCH_SIZE, len(chunks))

        batch_texts = chunks[start:end]

        batch_embeddings = create_document_embeddings(
            batch_texts
        )

        collection.add(
            ids=ids[start:end],
            documents=batch_texts,
            embeddings=batch_embeddings,
            metadatas=metadatas[start:end]
        )

        print(
            f"[INDEXING] {end}/{len(chunks)} chunks stored"
        )

    result = {
        "company": company_name,
        "filename": path.name,
        "chunks_indexed": len(chunks),
        "characters": len(full_text),
        "collection": collection.name,
        "embedding_model": MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "status": "Successfully indexed"
    }

    print("\n[SUCCESS] Document indexing completed.")

    return result


# ==================================================
# COMPANY-SPECIFIC SEARCH
# ==================================================

def search_company_documents(
    company_name: str,
    query: str,
    top_k: int = 5
) -> list[dict]:
    """Search only the selected company's documents."""

    collection = get_company_collection(company_name)

    total_chunks = collection.count()

    if total_chunks == 0:
        return []

    query_vector = create_query_embedding(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, total_chunks),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    matches = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for text, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):
        matches.append({
            "text": text,
            "metadata": metadata,
            "distance": distance
        })

    return matches


# ==================================================
# COMPANY DOCUMENT MANAGEMENT
# ==================================================

def list_indexed_companies() -> list[str]:
    """List companies with v2 collections."""

    companies = []

    for collection in chroma_client.list_collections():

        metadata = collection.metadata or {}

        if (
            metadata.get("version") == "2"
            and metadata.get("embedding_dimension")
            == EMBEDDING_DIMENSION
        ):
            company = metadata.get("company_name")

            if company:
                companies.append(company)

    return sorted(set(companies))


def get_company_document_count(
    company_name: str
) -> int:
    """Return total indexed chunks for a company."""

    return get_company_collection(company_name).count()


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    print("\nMULTI-COMPANY LOCAL RAG ENGINE")
    print("=" * 45)

    print("Embedding model:", MODEL_NAME)
    print("Dimensions:", EMBEDDING_DIMENSION)
    print("Storage:", VECTOR_STORE_PATH)

    print("\nIndexed companies:")

    companies = list_indexed_companies()

    if companies:
        for company in companies:
            print(
                f"- {company}: "
                f"{get_company_document_count(company)} chunks"
            )
    else:
        print("No company documents indexed yet.")

    print("\nLOCAL RAG ENGINE READY")
