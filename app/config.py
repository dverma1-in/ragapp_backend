import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────
#GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY = "AIzaSyC9L38FQ3BnO40Ci-AKFXR0W6ysx-AoymQ"

# ── Embedding & re-ranking models (local) ────────────────────────────────────
EMBEDDING_MODEL  = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL   = "BAAI/bge-reranker-base"

# ── Vector store ─────────────────────────────────────────────────────────────
CHROMA_DIR             = "data/chroma"
CHROMA_CHILD_COLLECTION  = "child_chunks"   # small chunks – used for vector search
CHROMA_PARENT_COLLECTION = "parent_chunks"  # large chunks – fetched after match

# ── BM25 ──────────────────────────────────────────────────────────────────────
BM25_INDEX_PATH = "data/bm25_index.pkl"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHILD_CHUNK_SIZE  = 200   # characters – small, precise for embedding
PARENT_CHUNK_SIZE = 1200  # characters – large, rich context for the LLM
CHUNK_OVERLAP     = 40

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K    = 20   # candidates from hybrid search before re-ranking
RERANK_TOP_N       = 5    # final chunks sent to LLM after re-ranking

# ── Agentic loop ──────────────────────────────────────────────────────────────
MAX_RETRIEVAL_HOPS = 3    # maximum re-retrieval attempts before giving up

# ── Conversation memory ───────────────────────────────────────────────────────
MEMORY_WINDOW = 6         # number of recent turns kept in context

os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)
