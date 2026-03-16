"""
Embedding utility.

Uses BAAI/bge-large-en-v1.5 locally via sentence-transformers.
The model is loaded once at import time (singleton) so the 1.3 GB
weights are not reloaded on every request.

bge-large expects a query prefix for retrieval tasks:
  - At *index* time  : plain text (no prefix)
  - At *query* time  : "Represent this sentence: <query>"
"""

from functools import lru_cache
from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL

_QUERY_PREFIX = "Represent this sentence: "


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed document chunks (no prefix)."""
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single query (with bge prefix for better retrieval quality)."""
    model = _get_model()
    prefixed = _QUERY_PREFIX + text
    return model.encode(prefixed, normalize_embeddings=True, convert_to_numpy=True).tolist()


def embed_queries(texts: list[str]) -> list[list[float]]:
    """Embed multiple queries (used for multi-query expansion)."""
    return [embed_query(t) for t in texts]
