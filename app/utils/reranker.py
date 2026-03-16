"""
Cross-encoder re-ranker using BAAI/bge-reranker-base.

A cross-encoder reads the (query, document) pair *together* – unlike a
bi-encoder which embeds them separately – so it can model fine-grained
relevance interactions.  It's slower than vector search but we only run
it on the small RETRIEVAL_TOP_K candidate set, not the whole corpus.
"""

from functools import lru_cache
from typing import List, Tuple

from sentence_transformers import CrossEncoder
from app.config import RERANKER_MODEL, RERANK_TOP_N


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL)


def rerank(query: str, candidates: List[dict]) -> List[dict]:
    """
    Score every candidate against the query and return the top-N
    sorted by relevance (highest first).

    Parameters
    ----------
    query      : the user's (possibly rewritten) query
    candidates : list of dicts with at least a "text" key

    Returns
    -------
    Top RERANK_TOP_N candidates with an added "rerank_score" field.
    """
    if not candidates:
        return []

    reranker = _get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores: List[float] = reranker.predict(pairs).tolist()

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = score

    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:RERANK_TOP_N]
