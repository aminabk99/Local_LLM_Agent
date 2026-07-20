"""
Local_LLM_Agent -- Local LLM Coding Agent + Chat API
FastAPI server with /chat endpoint, tool calling, and dark-themed web UI.
Powered by Ollama (qwen2.5-coder:7b) running fully locally.
"""

import os
import uuid

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from agent.agent import run_agent

app = FastAPI(title="Local_LLM_Agent Coding Agent", version="1.1.0")

# Default to localhost only. The agent can run shell commands, so a wide-open
# CORS policy on a machine exposed to a network is a real risk. Override with
# AGENT_CORS_ORIGINS="https://example.com,https://foo.com" if you know you need it.
_origins = [o.strip() for o in os.getenv(
    "AGENT_CORS_ORIGINS",
    "http://localhost:8000,http://127.0.0.1:8000",
).split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
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
    ui_path = os.path.join(os.path.dirname(__file__), "agent", "ui.html")
    with open(ui_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Run a user message through the local LLM agent and return its final reply."""
    session_id = request.session_id or str(uuid.uuid4())
    history = chat_sessions.setdefault(session_id, [])

    # Pass PRIOR turns as conversation context, then record this turn.
    reply, steps = run_agent(request.message, history=list(history), workspace="workspace")

    history.append({"role": "user", "content": request.message})
    history.append({"role": "assistant", "content": reply})

    return ChatResponse(reply=reply, session_id=session_id, steps_taken=steps)


@app.get("/history/{session_id}")
async def get_history(session_id: str):
    return JSONResponse(content={"session_id": session_id, "history": chat_sessions.get(session_id, [])})


@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    chat_sessions.pop(session_id, None)
    return JSONResponse(content={"message": "Session cleared."})


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
