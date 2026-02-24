from fastapi import APIRouter
from app.schemas import ChatRequest, ChatResponse
from app.services import generate_answer

router = APIRouter(prefix="/chat", tags=['chat'])

@router.get("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    answer, sources = generate_answer(request.query)
    return ChatResponse(answer=answer, sources=sources) 