"""
Agentic retrieval loop.

After the first retrieval the LLM checks whether the retrieved context
is *sufficient* to answer the user's question well.  If not, it
produces a reformulated search query and we retrieve again.

This loop runs for at most MAX_RETRIEVAL_HOPS iterations.

Sufficiency check response schema (JSON):
{
  "sufficient": true | false,
  "reason":     "brief explanation",
  "new_query":  "reformulated query to try next"  // only when sufficient=false
}
"""

import asyncio
from typing import List, Dict, Any

from app.LLM import chat_with_gemini_json
from app.utils.embedder import embed_query, embed_queries
from app.utils.retriever import hybrid_retrieve
from app.utils.reranker import rerank
from app.vectorstore.chroma_store import get_parents_by_ids
from app.config import MAX_RETRIEVAL_HOPS, RERANK_TOP_N


_SUFFICIENCY_PROMPT = """\
You are a retrieval quality evaluator for a RAG system.

User question:
{query}

Retrieved context chunks:
{context}

Evaluate whether the retrieved chunks contain enough information to answer the user's question accurately and completely.

Respond ONLY with this JSON object and nothing else:
{{
  "sufficient": <true if the context is sufficient, false if not>,
  "reason":     "<one sentence explaining your decision>",
  "new_query":  "<a better search query to find missing information — only required when sufficient is false, otherwise use empty string>"
}}

Be strict: if key facts needed to answer the question are missing, respond with sufficient=false.
If the information simply does not exist in the knowledge base (the question is out of scope), set sufficient=true and note that in your answer generation — do NOT keep looping.
"""


async def _single_retrieval(query_variants: List[str]) -> List[Dict[str, Any]]:
    """
    Embed all query variants, run hybrid retrieval, rerank, fetch parents.
    Returns a list of parent-chunk dicts ready for the LLM.
    """
    loop = asyncio.get_event_loop()

    # Embed all variants in a thread (CPU-bound)
    embeddings = await loop.run_in_executor(None, embed_queries, query_variants)

    # Hybrid retrieve children
    child_candidates = hybrid_retrieve(
        query_embeddings=embeddings,
        query_texts=query_variants,
    )

    if not child_candidates:
        return []

    # Re-rank using the primary (rewritten) query
    primary_query = query_variants[0]
    reranked_children = await loop.run_in_executor(
        None, rerank, primary_query, child_candidates
    )

    # Expand children → parents
    parent_ids = [c["metadata"].get("parent_id") for c in reranked_children if c.get("metadata", {}).get("parent_id")]
    parent_chunks = get_parents_by_ids(parent_ids)

    # If no parent found for some reason, fall back to child text
    if not parent_chunks:
        parent_chunks = [{"id": c["id"], "text": c["text"], "metadata": c.get("metadata", {})} for c in reranked_children]

    return parent_chunks


async def _check_sufficiency(query: str, context_chunks: List[Dict]) -> Dict:
    """Ask the LLM if the retrieved context is sufficient."""
    context_text = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{c['text']}" for i, c in enumerate(context_chunks)
    )
    prompt = _SUFFICIENCY_PROMPT.format(query=query, context=context_text)
    result = await chat_with_gemini_json(prompt)

    # Ensure safe defaults
    if not isinstance(result.get("sufficient"), bool):
        result["sufficient"] = True  # default: don't loop forever
    if not isinstance(result.get("new_query"), str):
        result["new_query"] = query

    return result


async def agentic_retrieve(
    primary_query: str,
    all_query_variants: List[str],
) -> tuple[List[Dict[str, Any]], List[str]]:
    """
    Main entry point for the agentic loop.

    Parameters
    ----------
    primary_query       : the rewritten/resolved query (used for sufficiency check)
    all_query_variants  : [rewritten, para1, para2, hyde]

    Returns
    -------
    context_chunks : list of parent-chunk dicts to send to the LLM
    sources        : deduplicated source labels for citations
    """
    current_variants = all_query_variants
    best_chunks: List[Dict] = []

    for hop in range(MAX_RETRIEVAL_HOPS):
        chunks = await _single_retrieval(current_variants)

        if not chunks:
            # Nothing found at all – stop early
            break

        best_chunks = chunks

        # On the last hop, skip the sufficiency check – just use what we have
        if hop == MAX_RETRIEVAL_HOPS - 1:
            break

        check = await _check_sufficiency(primary_query, chunks)

        if check["sufficient"]:
            break

        # Not sufficient – reformulate and try again
        new_query = check.get("new_query", primary_query).strip() or primary_query
        loop = asyncio.get_event_loop()
        new_embedding = await loop.run_in_executor(None, embed_query, new_query)
        current_variants = [new_query]  # focused single-query retry

    # Build sources list
    sources: List[str] = []
    seen_sources = set()
    for chunk in best_chunks:
        meta = chunk.get("metadata", {})
        file_name = meta.get("file_name", "unknown")
        page = meta.get("page")
        label = f"{file_name} (page {page})" if page else file_name
        if label not in seen_sources:
            seen_sources.add(label)
            sources.append(label)

    return best_chunks, sources
