from pydantic import BaseModel, Field
from typing import Optional

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message")
    session_id: Optional[str] = Field(None, description="Client session id for chat memory")

class ChatResponse(BaseModel):
    request_id: str
    session_id: str
    reply: str
