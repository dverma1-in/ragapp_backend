"""
Hybrid retriever combining dense (Chroma) and sparse (BM25) search.

Fusion strategy: Reciprocal Rank Fusion (RRF)
----------------------------------------------
RRF score for a document d across ranked lists R:
    RRF(d) = Σ  1 / (k + rank_i(d))
where k=60 is a smoothing constant that prevents top-1 items from
dominating.  Documents not appearing in a list are simply not scored
for that list.  The merged list is sorted descending by RRF score.

Per-query result set is then deduped by chunk ID before being passed
to the re-ranker.
"""

from typing import List, Dict, Any
from app.vectorstore.chroma_store import query_children
from app.vectorstore.bm25_store import get_bm25_index
from app.config import RETRIEVAL_TOP_K

RRF_K = 60  # standard constant for RRF


def _rrf_merge(
    dense_results: List[Dict],
    sparse_results: List[Dict],
    top_k: int,
) -> List[Dict]:
    """Merge two ranked lists with RRF and return top_k unique docs."""
    scores: Dict[str, float] = {}
    docs_by_id: Dict[str, Dict] = {}

    for rank, doc in enumerate(dense_results):
        did = doc["id"]
        scores[did] = scores.get(did, 0.0) + 1.0 / (RRF_K + rank + 1)
        docs_by_id[did] = doc

    for rank, doc in enumerate(sparse_results):
        did = doc["id"]
        scores[did] = scores.get(did, 0.0) + 1.0 / (RRF_K + rank + 1)
        docs_by_id[did] = doc

    ranked_ids = sorted(scores, key=lambda d: scores[d], reverse=True)

    results = []
    for did in ranked_ids[:top_k]:
        doc = docs_by_id[did].copy()
        doc["rrf_score"] = scores[did]
        results.append(doc)

    return results


def _parse_dense(raw: dict) -> List[Dict]:
    """Convert raw Chroma query result to a flat list of dicts."""
    docs       = raw.get("documents", [[]])[0]
    metadatas  = raw.get("metadatas",  [[]])[0]
    distances  = raw.get("distances",  [[]])[0]
    ids        = raw.get("ids",         [[]])[0]

    results = []
    for doc_id, text, meta, dist in zip(ids, docs, metadatas, distances):
        results.append({
            "id":       doc_id,
            "text":     text,
            "metadata": meta,
            "dense_distance": dist,
        })
    return results


def hybrid_retrieve(
    query_embeddings: List[List[float]],
    query_texts: List[str],
    top_k: int = RETRIEVAL_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Run dense + sparse search for every query variant, merge with RRF.

    Parameters
    ----------
    query_embeddings : one embedding per query variant
    query_texts      : matching plain-text versions (for BM25)
    top_k            : total candidates to return before re-ranking

    Returns
    -------
    Deduplicated list of child-chunk dicts sorted by RRF score.
    """
    bm25 = get_bm25_index()
    all_dense:  List[Dict] = []
    all_sparse: List[Dict] = []

    per_query_k = max(top_k, 10)

    for embedding, text in zip(query_embeddings, query_texts):
        # Dense
        raw_dense = query_children(embedding, n_results=per_query_k)
        all_dense.extend(_parse_dense(raw_dense))

        # Sparse
        sparse_hits = bm25.search(text, n_results=per_query_k)
        all_sparse.extend(sparse_hits)

    # Deduplicate within each list before fusion (keep first occurrence)
    def dedup(docs: List[Dict]) -> List[Dict]:
        seen = set()
        out = []
        for d in docs:
            if d["id"] not in seen:
                seen.add(d["id"])
                out.append(d)
        return out

    merged = _rrf_merge(dedup(all_dense), dedup(all_sparse), top_k=top_k)
    return merged
