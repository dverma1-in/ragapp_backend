"""
BM25 keyword index backed by rank_bm25.

The index is rebuilt from all stored child chunks and persisted as a
pickle file so it survives server restarts without needing to re-ingest.

Thread-safety note: ingestion is expected to be sequential (one file at
a time).  For concurrent ingestion in production, wrap mutations in a
threading.Lock.
"""

import os
import pickle
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi
from app.config import BM25_INDEX_PATH


class BM25Index:
    def __init__(self):
        self._corpus_docs: List[Dict[str, Any]] = []   # [{id, text, metadata}]
        self._tokenized: List[List[str]] = []
        self._bm25: BM25Okapi | None = None

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self):
        os.makedirs(os.path.dirname(BM25_INDEX_PATH), exist_ok=True)
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump({
                "corpus": self._corpus_docs,
                "tokenized": self._tokenized,
            }, f)

    def load(self):
        if not os.path.exists(BM25_INDEX_PATH):
            return
        with open(BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        self._corpus_docs = data["corpus"]
        self._tokenized   = data["tokenized"]
        if self._tokenized:
            self._bm25 = BM25Okapi(self._tokenized)

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add_documents(self, docs: List[Dict[str, Any]]):
        """
        Add new documents to the index.
        Each doc must have {id, text, metadata}.
        """
        for doc in docs:
            tokens = doc["text"].lower().split()
            self._corpus_docs.append(doc)
            self._tokenized.append(tokens)
        self._bm25 = BM25Okapi(self._tokenized)
        self.save()

    # ── Query ─────────────────────────────────────────────────────────────────

    def search(self, query: str, n_results: int = 20) -> List[Dict[str, Any]]:
        """Return top-n docs sorted by BM25 score (highest first)."""
        if self._bm25 is None or not self._corpus_docs:
            return []

        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)

        scored = sorted(
            zip(scores, self._corpus_docs),
            key=lambda x: x[0],
            reverse=True,
        )
        return [
            {**doc, "bm25_score": float(score)}
            for score, doc in scored[:n_results]
            if score > 0
        ]


# Module-level singleton – loaded once at import time
_index = BM25Index()
_index.load()


def get_bm25_index() -> BM25Index:
    return _index
