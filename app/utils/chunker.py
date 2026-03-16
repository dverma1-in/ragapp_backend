"""
Semantic chunker implementing the parent-document retrieval pattern.

Strategy
--------
1. Split text into *parent* chunks at natural semantic boundaries
   (double newlines / paragraph breaks).  Parents are large (~1200 chars)
   and contain full context – they are what the LLM eventually reads.

2. Further split each parent into small *child* chunks (~200 chars).
   Children are what gets embedded and stored in the vector index for
   precise semantic matching.

3. Every child carries a `parent_id` in its metadata so we can fetch
   the parent once a child matches a query.

This gives us the best of both worlds: precision at retrieval time
(small embeddings) and rich context at generation time (large parents).
"""

import re
import uuid
from typing import List, Dict, Any, Tuple
from app.config import CHILD_CHUNK_SIZE, PARENT_CHUNK_SIZE, CHUNK_OVERLAP


def _split_by_paragraphs(text: str, max_size: int, overlap: int) -> List[str]:
    """
    Split text preferring paragraph / sentence boundaries.
    Only falls back to hard character cuts when a paragraph exceeds max_size.
    """
    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Try paragraph-level split first
    raw_paragraphs = re.split(r"\n{2,}", text)

    chunks: List[str] = []
    current = ""

    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue

        # Para fits in current chunk
        if len(current) + len(para) + 1 <= max_size:
            current = (current + "\n\n" + para).strip()

        else:
            # Flush current chunk
            if current:
                chunks.append(current)

            # Para itself is too big – split by sentences
            if len(para) > max_size:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                sentence_buffer = ""
                for sent in sentences:
                    if len(sentence_buffer) + len(sent) + 1 <= max_size:
                        sentence_buffer = (sentence_buffer + " " + sent).strip()
                    else:
                        if sentence_buffer:
                            chunks.append(sentence_buffer)
                        sentence_buffer = sent
                if sentence_buffer:
                    current = sentence_buffer
                else:
                    current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    # Apply overlap between consecutive chunks
    if overlap > 0 and len(chunks) > 1:
        overlapped: List[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:]
            overlapped.append((tail + " " + chunks[i]).strip())
        return overlapped

    return chunks


def build_parent_chunks(text: str, base_metadata: Dict[str, Any]) -> List[Dict]:
    """Return list of parent chunk dicts with their own unique IDs."""
    raw = _split_by_paragraphs(text, PARENT_CHUNK_SIZE, overlap=0)
    parents = []
    for chunk_text in raw:
        if not chunk_text.strip():
            continue
        parents.append({
            "id": str(uuid.uuid4()),
            "text": chunk_text,
            "metadata": {**base_metadata},
        })
    return parents


def build_child_chunks(parent: Dict) -> List[Dict]:
    """
    Split a parent chunk into small child chunks.
    Each child stores the parent_id so we can look it up later.
    """
    raw = _split_by_paragraphs(parent["text"], CHILD_CHUNK_SIZE, CHUNK_OVERLAP)
    children = []
    for child_text in raw:
        if not child_text.strip():
            continue
        children.append({
            "id": str(uuid.uuid4()),
            "text": child_text,
            "metadata": {
                **parent["metadata"],
                "parent_id": parent["id"],
            },
        })
    return children


def chunk_document(
    text: str,
    base_metadata: Dict[str, Any],
) -> Tuple[List[Dict], List[Dict]]:
    """
    Main entry point.

    Returns
    -------
    parents : list of parent chunk dicts  {id, text, metadata}
    children: list of child chunk dicts   {id, text, metadata}  (includes parent_id)
    """
    parents = build_parent_chunks(text, base_metadata)
    children = []
    for parent in parents:
        children.extend(build_child_chunks(parent))
    return parents, children
