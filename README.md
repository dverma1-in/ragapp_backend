# Advanced RAG Agent v2

A production-grade Retrieval-Augmented Generation (RAG) backend built with FastAPI.  
Every technique below is a deliberate upgrade over the naive fixed-chunk + single-query baseline.

---

## Architecture overview

```
INGESTION
  File upload
    → Semantic chunker  (paragraph-aware, parent + child splits)
    → bge-large embedder  (child chunks only)
    → Chroma child collection  (vector index)
    → Chroma parent collection  (full-context store, ID lookup)
    → BM25 index  (keyword index, persisted to disk)

QUERY
  User question + session_id
    → Conversation memory  (sliding window of past turns)
    → Query transformer  (rewrite + 2 paraphrases + HyDE)
    ↓  4 query variants
    → Hybrid retriever  (Chroma dense + BM25 sparse, per variant)
    → RRF merge  (Reciprocal Rank Fusion across all variants)
    → Cross-encoder re-ranker  (top-20 → top-5)
    → Parent-doc fetch  (matched child → large parent chunk)
    → Agentic sufficiency loop  (up to 3 hops if context is weak)
    → Prompt builder  (history + numbered sources + question)
    → Gemini 2.5 Flash  (answer generation)
    → Memory update  (save turn)
    → Response  {answer, sources, session_id}
```

---

## Techniques explained

### 1. Semantic chunking with parent-document retrieval

**Problem with fixed-size chunking:**  
Splitting at a fixed character count breaks mid-sentence and mid-concept.  
A 600-char chunk about "annual revenue" might have its definition in the chunk before and its figures in the chunk after.

**What we do instead:**  
- Split at paragraph and sentence boundaries, respecting natural semantic units.
- Two tiers per document:
  - **Child chunks** (~200 chars) — small, precise, embedded for vector search.
  - **Parent chunks** (~1200 chars) — large, context-rich, fetched when a child matches.
- Children store a `parent_id`. When a child is retrieved, we swap it for its parent before sending context to the LLM.

**Why it works:**  
The embedding captures meaning precisely from the small child. The LLM gets the large parent with full surrounding context. Best of both worlds.

---

### 2. Hybrid search (dense + BM25) with Reciprocal Rank Fusion

**Problem with pure vector search:**  
Embeddings capture semantic meaning but can miss exact terminology. Searching for "myocardial infarction" might not retrieve a chunk that only says "heart attack", and vice versa.

**What we do:**  
- **Dense search** via Chroma cosine similarity (semantic meaning).
- **Sparse search** via BM25 (exact keyword matching — the same algorithm that powers classic search engines like Elasticsearch).
- Both run in parallel for every query variant.
- Results are merged with **Reciprocal Rank Fusion (RRF)**:
  ```
  RRF_score(doc) = Σ  1 / (60 + rank_in_list)
  ```
  A document appearing at rank 3 in the dense list AND rank 5 in the sparse list scores higher than one that appears only in one list — even if it ranked #1 in that list. The constant 60 prevents top-1 items from completely dominating.

**Why it works:**  
The two retrieval methods have complementary failure modes. RRF robustly combines them without needing to tune weights.

---

### 3. Cross-encoder re-ranking

**Problem with bi-encoder retrieval:**  
When embedding a query and document separately (bi-encoder), the model never sees them together — it can only compare their individual representations. This misses subtle relevance interactions.

**What we do:**  
After hybrid retrieval gives us ~20 candidates, a **cross-encoder** (`bge-reranker-base`) reads each `(query, chunk)` pair *together* and outputs a relevance score. We keep the top 5.

**Why it works:**  
Cross-encoders are ~10× more accurate than bi-encoders at relevance scoring. They're too slow to run on the whole corpus, but running them on 20 pre-filtered candidates is fast (under 1 second on CPU).

---

### 4. Multi-query expansion + HyDE

**Problem with single-query retrieval:**  
If the user asks a vague question or uses vocabulary different from the document, a single embedding query might miss all the relevant chunks.

**What we do:**  
Before any retrieval, the LLM transforms the query into 4 variants:
1. **Rewritten query** — resolves pronouns/references using conversation history ("what about its revenue?" → "what is Apple's annual revenue?")
2. **Paraphrase 1** — different vocabulary, same intent
3. **Paraphrase 2** — another alternative phrasing
4. **HyDE** (Hypothetical Document Embedding) — the LLM generates a *fake ideal answer* as if it were in the document. We embed this fake answer and search with it.

All 4 variants hit the retriever in parallel. Results are merged via RRF.

**Why HyDE works:**  
Question embeddings and answer embeddings live in different regions of the embedding space. By embedding a hypothetical answer, we search from the answer's neighbourhood — much closer to where the actual document chunks live. Counterintuitive but consistently effective.

---

### 5. Agentic retrieval loop

**Problem with one-shot retrieval:**  
A single retrieval pass might find the wrong chunks, especially for complex or multi-part questions. There's no way to course-correct.

**What we do:**  
After retrieving and re-ranking, the LLM acts as a **sufficiency evaluator**:
```json
{
  "sufficient": false,
  "reason": "Found context about Q3 revenue but question also asks about Q4.",
  "new_query": "Q4 2023 revenue figures annual report"
}
```
If `sufficient=false`, we reformulate the query and retrieve again. This loop runs up to `MAX_RETRIEVAL_HOPS=3` times.

If the LLM determines the topic is simply not in the knowledge base, it sets `sufficient=true` and the answer generation step tells the user honestly that the information isn't available.

**Why it works:**  
Complex questions often need bridging between multiple topics. The agent can discover mid-retrieval that it's missing a key piece and go find it.

---

### 6. Conversation memory

**Problem with stateless chat:**  
Without memory, "what about its profit margin?" after asking about a company is an impossible query — there's no "it" to resolve.

**What we do:**  
- Each request carries a `session_id`.
- We maintain a sliding window of the last `MEMORY_WINDOW=6` turns (user + assistant alternating) per session.
- The conversation history is injected into:
  1. The **query transformer prompt** — so the rewriter can resolve references.
  2. The **final answer prompt** — so the LLM can maintain coherent conversation.

Sessions are stored in-process (Python dict). For multi-server production deployments, replace with Redis.

---

## Setup

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 4. Run the server
uvicorn app.main:app --reload
```

On first run the two local models will download automatically:
- `BAAI/bge-large-en-v1.5` (~1.3 GB) — embedding model
- `BAAI/bge-reranker-base` (~280 MB) — cross-encoder re-ranker

---

## API reference

### `POST /upload`
Upload a PDF or TXT file to ingest into the knowledge base.

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@document.pdf"
```

Response:
```json
{
  "file_name": "document.pdf",
  "parents_added": 42,
  "children_added": 187,
  "status": "success"
}
```

---

### `POST /chat`
Ask a question. Include `session_id` on follow-up messages to maintain context.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What was the revenue in Q3?", "session_id": "my-session-123"}'
```

Response:
```json
{
  "answer": "According to the report (Source 1), Q3 revenue was $4.2B...",
  "sources": ["annual_report.pdf (page 12)"],
  "session_id": "my-session-123"
}
```

Follow-up (pass the same session_id):
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How does that compare to Q4?", "session_id": "my-session-123"}'
```

---

### `DELETE /chat/session/{session_id}`
Clear conversation memory for a session.

---

### `GET /chat/sessions`
List all active session IDs (debug endpoint).

---

## Configuration (`app/config.py`)

| Setting | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Local embedding model |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Local cross-encoder |
| `CHILD_CHUNK_SIZE` | `200` | Max chars per child chunk |
| `PARENT_CHUNK_SIZE` | `1200` | Max chars per parent chunk |
| `RETRIEVAL_TOP_K` | `20` | Candidates before re-ranking |
| `RERANK_TOP_N` | `5` | Chunks sent to LLM after re-ranking |
| `MAX_RETRIEVAL_HOPS` | `3` | Agentic loop max iterations |
| `MEMORY_WINDOW` | `6` | Conversation turns kept in context |

---

## Project structure

```
app/
├── main.py                   FastAPI app, router registration
├── config.py                 All settings in one place
├── LLM/
│   └── gemini_client.py      Gemini 2.5 Flash (text + JSON mode)
├── loaders/
│   ├── pdf.py                PyMuPDF page-by-page extraction
│   └── txt.py                UTF-8 / Latin-1 text loading
├── utils/
│   ├── chunker.py            Semantic chunker, parent-doc pattern
│   ├── embedder.py           bge-large singleton, query prefix
│   ├── reranker.py           bge-reranker cross-encoder singleton
│   ├── retriever.py          Hybrid search + RRF merge
│   └── prompt_builder.py     Final answer prompt assembly
├── vectorstore/
│   ├── chroma_store.py       Child + parent Chroma collections
│   └── bm25_store.py         BM25 index, persisted to disk
├── agent/
│   ├── query_transformer.py  Multi-query + HyDE expansion
│   └── loop.py               Agentic sufficiency check loop
├── memory/
│   └── conversation.py       Per-session sliding window memory
├── services/
│   ├── ingestion.py          Ingest pipeline orchestration
│   └── answer.py             Answer pipeline orchestration
├── routes/
│   ├── upload.py             POST /upload
│   └── chat.py               POST /chat, session management
└── schemas/
    └── schemas.py            Pydantic request/response models
```
