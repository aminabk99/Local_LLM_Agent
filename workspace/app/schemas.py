from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message")
    session_id: Optional[str] = Field(default=None, description="Conversation session id (optional)")


class ChatResponse(BaseModel):
    request_id: str
    session_id: str
    reply: str

    # ✅ New: real JSON returned when a tool is used (Swagger will render it cleanly)
    tool_result: Optional[Dict[str, Any]] = None
