from fastapi import APIRouter
from app.schemas import ChatRequest, ChatResponse
from app.services import generate_answer
from app.memory import delete_session, list_sessions

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    answer, sources = await generate_answer(
        query=request.query,
        session_id=request.session_id,
    )
    return ChatResponse(
        answer=answer,
        sources=sources,
        session_id=request.session_id,
    )


@router.delete("/session/{session_id}", tags=["memory"])
async def clear_session(session_id: str):
    """Clear conversation memory for a session."""
    delete_session(session_id)
    return {"status": "cleared", "session_id": session_id}


@router.get("/sessions", tags=["memory"])
async def get_sessions():
    """List all active session IDs (useful for debugging)."""
    return {"sessions": list_sessions()}
