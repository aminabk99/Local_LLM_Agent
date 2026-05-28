"""
AI_LLM — Local LLM Coding Agent + Chat API
FastAPI server with /chat endpoint, tool calling, and dark-themed web UI.
Powered by Ollama (qwen2.5-coder:7b) running fully locally.
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import uuid

from agent.agent import run_agent

app = FastAPI(title="AI_LLM Coding Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Per-session in-memory chat history
chat_sessions: dict[str, list[dict]] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str = ""


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    steps_taken: int


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the dark-themed web UI."""
    with open("agent/ui.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    Accepts a user message, runs it through the local LLM agent,
    and returns the agent's final response.
    """
    session_id = request.session_id or str(uuid.uuid4())

    if session_id not in chat_sessions:
        chat_sessions[session_id] = []

    history = chat_sessions[session_id]
    history.append({"role": "user", "content": request.message})

    reply, steps = run_agent(request.message, history, workspace="workspace")

    history.append({"role": "assistant", "content": reply})
    chat_sessions[session_id] = history

    return ChatResponse(reply=reply, session_id=session_id, steps_taken=steps)


@app.get("/history/{session_id}")
async def get_history(session_id: str):
    """Return full chat history for a session."""
    history = chat_sessions.get(session_id, [])
    return JSONResponse(content={"session_id": session_id, "history": history})


@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    """Clear chat history for a session."""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    return JSONResponse(content={"message": "Session cleared."})


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
