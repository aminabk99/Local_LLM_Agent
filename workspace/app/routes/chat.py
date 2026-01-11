from fastapi import APIRouter, HTTPException
from uuid import uuid4

from app.schemas import ChatRequest, ChatResponse
from app.services.ollama_client import generate_reply
from app.services.memory import store

router = APIRouter(tags=["chat"])

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the conversation history to respond naturally. "
    "If the user refers to something they said earlier, use that context."
)

def build_prompt(session_id: str, user_msg: str) -> str:
    history = store.get_history(session_id)

    lines = [f"SYSTEM:\n{SYSTEM_PROMPT}\n"]

    # Add history turns
    for m in history:
        role = m.role.upper()
        lines.append(f"{role}:\n{m.content}\n")

    # Add current user message
    lines.append(f"USER:\n{user_msg}\nASSISTANT:\n")
    return "\n".join(lines)

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    msg = req.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Use provided session_id, or create a new one
    session_id = req.session_id or str(uuid4())
    request_id = str(uuid4())

    # Save user message to memory
    store.append(session_id, "user", msg)

    # Build prompt including memory
    prompt = build_prompt(session_id, msg)

    try:
        reply_text = generate_reply(prompt)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {str(e)}")

    # Save assistant reply to memory
    store.append(session_id, "assistant", reply_text)

    return ChatResponse(request_id=request_id, session_id=session_id, reply=reply_text)
