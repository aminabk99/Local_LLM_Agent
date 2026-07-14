"""
FastAPI SSE server for Local_LLM_Agent real-time streaming.

Run:
    uvicorn streaming.sse_server:app --host 0.0.0.0 --port 8001 --reload

Endpoints
---------
POST /stream/run
    Stream a complete agent run as Server-Sent Events.

    Request:
        {
            "task":       "Write a hello world script",  // required
            "image_b64":  "<base64 screenshot>",         // optional: inject CI error context
            "model":      "qwen2.5-coder:7b"            // optional
        }

    SSE event stream:
        event: start
        data: {"type": "start", "task": "...", "model": "..."}

        event: step_start
        data: {"type": "step_start", "step": 1, "max_steps": 20}

        event: token
        data: {"type": "token", "text": "{", "step": 1, "elapsed_ms": 142.3}

        event: action
        data: {"type": "action", "action": {"action": "write_file", ...}, "step": 1, ...}

        event: tool_start
        data: {"type": "tool_start", "tool": "write_file", "step": 1}

        event: tool_done
        data: {"type": "tool_done", "tool": "write_file", "result": "Written: hello.py", ...}

        event: finish
        data: {"type": "finish", "result": "Done.", "total_steps": 3, "total_ms": 12410.5}

POST /stream/multimodal
    Same as /stream/run but accepts an image_b64 field.
    llava extracts error context from the screenshot and injects it into the task.

GET /health
    {"status": "ok", "streaming": true}
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from streaming.config import AGENT_MODEL

app = FastAPI(
    title="Local_LLM_Agent Streaming API",
    description="Real-time streaming agent with SSE — text + image (multimodal) tasks",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    task: str
    model: str = AGENT_MODEL
    image_b64: Optional[str] = None   # base64 screenshot for multimodal tasks


def _sse(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def _agent_sse_generator(task: str, model: str, image_b64: Optional[str] = None):
    """Async generator yielding SSE frames for the full agent run."""
    from streaming.stream_agent import run_streaming_agent

    # Optional: inject image context into task
    if image_b64:
        yield _sse("vision_start", {"type": "vision_start", "message": "Analysing screenshot..."})
        import asyncio as _asyncio
        from streaming.multimodal import inject_image_context
        loop = _asyncio.get_event_loop()
        task = await loop.run_in_executor(None, inject_image_context, task, image_b64)
        yield _sse("vision_done", {"type": "vision_done", "augmented_task": task})

    async for event in run_streaming_agent(task, model=model):
        yield _sse(event["type"], event)
        if event["type"] in ("finish", "error"):
            yield _sse("close", {})
            break


@app.post("/stream/run")
async def stream_run(req: RunRequest):
    """Stream a complete agent run as Server-Sent Events."""
    if not req.task.strip():
        raise HTTPException(status_code=400, detail="task cannot be empty")

    return StreamingResponse(
        _agent_sse_generator(req.task, req.model, req.image_b64),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/health")
def health():
    return {"status": "ok", "streaming": True, "model": AGENT_MODEL}


# ---------------------------------------------------------------------------
# Example client (run with: python -m streaming.sse_server client)
# ---------------------------------------------------------------------------
async def _demo_client():
    """Demo: connect to the SSE endpoint and print events."""
    import httpx

    print("Connecting to SSE stream...")
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            "http://localhost:8001/stream/run",
            json={"task": "Write a Python hello world script and run it."},
        ) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data:"):
                    data = json.loads(line[5:].strip())
                    etype = data.get("type", "?")
                    if etype == "token":
                        print(data["text"], end="", flush=True)
                    elif etype == "tool_done":
                        print(f"\n[Tool: {data['tool']}] {data['result'][:80]}")
                    elif etype == "finish":
                        print(f"\n\nDone in {data['total_steps']} steps, {data['total_ms']/1000:.1f}s")
                    elif etype == "error":
                        print(f"\nError: {data['message']}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        asyncio.run(_demo_client())
    else:
        import uvicorn
        uvicorn.run("streaming.sse_server:app", host="0.0.0.0", port=8001, reload=True)
