from fastapi import FastAPI
from app.routes import upload_router, chat_router
from app.middlewares.middleware import exception_handling_middleware

app = FastAPI(
    title="Advanced RAG Agent",
    description=(
        "Semantic chunking · Hybrid search (dense + BM25) · RRF · "
        "Cross-encoder re-ranking · Parent-document retrieval · "
        "Multi-query + HyDE expansion · Agentic sufficiency loop · "
        "Conversation memory"
    ),
    version="2.0.0",
)

app.middleware("http")(exception_handling_middleware)

app.include_router(upload_router)
app.include_router(chat_router)


@app.get("/", tags=["health"])
def health():
    return {"status": "ok", "version": "2.0.0"}
