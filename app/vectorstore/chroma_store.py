"""
Chroma vector store.

Two collections:
  child_chunks  – small embeddings for precise vector search
  parent_chunks – full-context text, fetched after a child match
                  stored with embedding=None (we only do ID lookups here)
"""

import chromadb
from app.config import CHROMA_DIR, CHROMA_CHILD_COLLECTION, CHROMA_PARENT_COLLECTION

_client = chromadb.PersistentClient(path=CHROMA_DIR)

child_collection  = _client.get_or_create_collection(
    name=CHROMA_CHILD_COLLECTION,
    metadata={"hnsw:space": "cosine"},
)
parent_collection = _client.get_or_create_collection(
    name=CHROMA_PARENT_COLLECTION,
)


# ── Write ─────────────────────────────────────────────────────────────────────

def add_children(ids, documents, embeddings, metadatas):
    child_collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def add_parents(ids, documents, metadatas):
    """Parents are stored by ID for lookup – no embedding needed."""
    parent_collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )


# ── Read ──────────────────────────────────────────────────────────────────────

def query_children(query_embedding: list[float], n_results: int = 20) -> dict:
    return child_collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )


def get_parents_by_ids(parent_ids: list[str]) -> list[dict]:
    """
    Fetch parent chunks by their IDs.
    Returns list of {id, text, metadata} dicts, preserving the input order.
    """
    if not parent_ids:
        return []

    # Deduplicate while preserving order
    seen = set()
    unique_ids = []
    for pid in parent_ids:
        if pid not in seen:
            seen.add(pid)
            unique_ids.append(pid)

    result = parent_collection.get(
        ids=unique_ids,
        include=["documents", "metadatas"],
    )

    id_to_doc = {
        rid: {"id": rid, "text": doc, "metadata": meta}
        for rid, doc, meta in zip(
            result["ids"], result["documents"], result["metadatas"]
        )
    }

    return [id_to_doc[pid] for pid in unique_ids if pid in id_to_doc]
