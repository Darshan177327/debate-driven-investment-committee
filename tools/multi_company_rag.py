import re
import hashlib
import shutil
import tempfile
from pathlib import Path
import chromadb
from fastembed import TextEmbedding

from tools.universal_document_processor import extract_document


# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Original database containing your indexed company documents.
SOURCE_VECTOR_STORE_PATH = (
    PROJECT_ROOT / "memory" / "vector_store"
)

# Use a writable temporary database on Streamlit Cloud.
# Streamlit Cloud normally runs applications from /mount/src.
if Path("/mount/src").exists():

    VECTOR_STORE_PATH = (
        Path(tempfile.gettempdir())
        / "debate_driven_investment_committee"
        / "vector_store"
    )

    if not VECTOR_STORE_PATH.exists():

        VECTOR_STORE_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if SOURCE_VECTOR_STORE_PATH.exists():

            shutil.copytree(
                SOURCE_VECTOR_STORE_PATH,
                VECTOR_STORE_PATH
            )

        else:

            VECTOR_STORE_PATH.mkdir(
                parents=True,
                exist_ok=True
            )

else:

    # Preserve the existing Windows database.
    VECTOR_STORE_PATH = SOURCE_VECTOR_STORE_PATH

VECTOR_STORE_PATH.mkdir(
    parents=True,
    exist_ok=True
)

print("[RAG] Database location:", VECTOR_STORE_PATH)

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
    file_path: str,
    progress_callback=None,
) -> dict:
    """Extract, chunk, embed and index a document.

    progress_callback receives truthful stage events. Existing callers remain
    compatible because the callback is optional.
    """

    def report(stage: str, message: str, *, completed=None, total=None):
        if progress_callback is not None:
            progress_callback({
                "stage": stage,
                "message": message,
                "completed": completed,
                "total": total,
            })

    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")

    print("\n[INDEXING] Document:", path.name)
    report("extract", "Extracting text and tables from the document")
    document = extract_document(str(path))
    full_text = document.get("full_text", "")
    if not full_text or not full_text.strip():
        raise ValueError(f"No readable text found in {path.name}")
    report("extract_done", f"Extraction complete · {len(full_text):,} characters")

    report("chunk", "Splitting extracted text into overlapping chunks")
    chunks = split_text(full_text)
    if not chunks:
        raise ValueError("No chunks generated.")
    report("chunk_done", f"Chunking complete · {len(chunks):,} chunks created", completed=len(chunks), total=len(chunks))

    collection = get_company_collection(company_name)
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    existing = collection.get(where={"document_hash": file_hash}, include=["metadatas"])
    if existing["ids"]:
        print("[SKIPPED] This document is already indexed.")
        report("duplicate", "This exact document is already indexed; no duplicate chunks were added")
        return {
            "company": company_name,
            "filename": path.name,
            "chunks_indexed": 0,
            "status": "Already indexed",
            "collection": collection.name,
        }

    ids = []
    metadatas = []
    for index, chunk in enumerate(chunks):
        chunk_id = hashlib.sha256(f"{file_hash}_{index}".encode("utf-8")).hexdigest()
        ids.append(chunk_id)
        metadatas.append({
            "company": company_name.strip(),
            "filename": path.name,
            "file_type": path.suffix.lower(),
            "document_hash": file_hash,
            "chunk_index": index,
            "total_chunks": len(chunks),
        })

    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    embedded_count = 0
    stored_count = 0
    for batch_number, start in enumerate(range(0, len(chunks), BATCH_SIZE), start=1):
        end = min(start + BATCH_SIZE, len(chunks))
        batch_texts = chunks[start:end]

        report("embedding", f"Generating embeddings · batch {batch_number}/{total_batches}", completed=embedded_count, total=len(chunks))
        batch_embeddings = create_document_embeddings(batch_texts)
        embedded_count += len(batch_embeddings)
        report("embedding_done", f"Embeddings generated · {embedded_count:,}/{len(chunks):,} chunks", completed=embedded_count, total=len(chunks))

        report("storage", f"Writing batch {batch_number}/{total_batches} to ChromaDB", completed=stored_count, total=len(chunks))
        collection.add(
            ids=ids[start:end],
            documents=batch_texts,
            embeddings=batch_embeddings,
            metadatas=metadatas[start:end],
        )
        stored_count += len(batch_texts)
        report("storage_done", f"Stored in ChromaDB · {stored_count:,}/{len(chunks):,} chunks", completed=stored_count, total=len(chunks))
        print(f"[INDEXING] {stored_count}/{len(chunks)} chunks stored")

    result = {
        "company": company_name,
        "filename": path.name,
        "chunks_indexed": stored_count,
        "characters": len(full_text),
        "collection": collection.name,
        "embedding_model": MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "status": "Successfully indexed",
    }
    report("complete", f"Indexing complete · {stored_count:,} chunks stored", completed=stored_count, total=len(chunks))
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


def get_all_company_chunks(company_name: str, batch_size: int = 500) -> list[dict]:
    """Read all indexed chunks for one company without changing ChromaDB."""
    collection = get_company_collection(company_name)
    total = collection.count()
    if total <= 0:
        return []
    output = []
    for offset in range(0, total, batch_size):
        result = collection.get(
            limit=min(batch_size, total - offset),
            offset=offset,
            include=["documents", "metadatas"],
        )
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        for text, metadata in zip(documents, metadatas):
            output.append({"text": text or "", "metadata": metadata or {}})
    # Stable ordering helps adjacent chunk context and repeatable results.
    output.sort(key=lambda item: (
        str((item.get("metadata") or {}).get("filename", "")),
        int((item.get("metadata") or {}).get("chunk_index", 0)),
    ))
    return output


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
