from pydantic import BaseModel, Field
from typing import List
import uuid


class ChatRequest(BaseModel):
    query: str
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description=(
            "Conversation session identifier. "
            "Pass the same ID on follow-up questions to maintain context. "
            "Omit (or send a new UUID) to start a fresh conversation."
        ),
    )


class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []
    session_id: str


class UploadResponse(BaseModel):
    file_name: str
    parents_added: int
    children_added: int
    status: str = "success"
