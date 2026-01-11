from fastapi import APIRouter, HTTPException
from uuid import uuid4

from app.schemas import ChatRequest, ChatResponse
from app.services.ollama_client import generate_reply
from app.services.memory import store

router = APIRouter(tags=["chat"])

def build_prompt(session_id: str, user_msg: str) -> str:
    history = store.get_history(session_id)

    # Simple memory prompt (user + assistant turns)
    convo_lines = [
        "SYSTEM:\nYou are a helpful assistant. Use the conversation history to respond naturally.\n"
    ]

    for m in history:
        convo_lines.append(f"{m.role.upper()}:\n{m.content}\n")

    convo_lines.append(f"USER:\n{user_msg}\nASSISTANT:\n")
    return "\n".join(convo_lines)

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    msg = req.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session_id = req.session_id or str(uuid4())
    request_id = str(uuid4())

    # Save user message
    store.append(session_id, "user", msg)

    # Build prompt including history
    prompt = build_prompt(session_id, msg)

    try:
        reply_text = generate_reply(prompt)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {str(e)}")

    # Save assistant reply
    store.append(session_id, "assistant", reply_text)

    return ChatResponse(request_id=request_id, session_id=session_id, reply=reply_text)
