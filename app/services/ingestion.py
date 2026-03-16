"""
Ingestion service.

Pipeline:
  1. Load file → TextUnits  (one per page / section)
  2. Semantic chunk each unit → (parents, children)
  3. Embed child chunks with bge-large
  4. Store children in Chroma child collection
  5. Store parents in Chroma parent collection (no embedding)
  6. Add children to BM25 index
"""

import asyncio
from fastapi import UploadFile

from app.loaders import extract_text
from app.utils.chunker import chunk_document
from app.utils.embedder import embed_documents
from app.vectorstore import add_children, add_parents, get_bm25_index


async def ingest_file(file: UploadFile) -> dict:
    # 1. Extract raw text units
    units = await extract_text(file)

    all_parents = []
    all_children = []

    # 2. Chunk every unit into parent + child pairs
    for unit in units:
        parents, children = chunk_document(unit["text"], unit["metadata"])
        all_parents.extend(parents)
        all_children.extend(children)

    if not all_children:
        return {
            "file_name": file.filename,
            "parents_added": 0,
            "children_added": 0,
        }

    # 3. Embed children (CPU-bound – run in thread pool)
    loop = asyncio.get_event_loop()
    child_texts = [c["text"] for c in all_children]
    child_embeddings = await loop.run_in_executor(None, embed_documents, child_texts)

    # 4. Store children in Chroma
    add_children(
        ids        = [c["id"]       for c in all_children],
        documents  = child_texts,
        embeddings = child_embeddings,
        metadatas  = [c["metadata"] for c in all_children],
    )

    # 5. Store parents in Chroma (ID lookup only, no embedding)
    add_parents(
        ids       = [p["id"]       for p in all_parents],
        documents = [p["text"]     for p in all_parents],
        metadatas = [p["metadata"] for p in all_parents],
    )

    # 6. Add children to BM25 index
    bm25 = get_bm25_index()
    bm25.add_documents([
        {"id": c["id"], "text": c["text"], "metadata": c["metadata"]}
        for c in all_children
    ])

    return {
        "file_name":      file.filename,
        "parents_added":  len(all_parents),
        "children_added": len(all_children),
    }
