"""
Streaming agent loop with per-step SSE events.

Wraps the existing CodingAgent logic but emits streaming events at each step:
  - token events as the LLM generates the action
  - step_start / step_done events for each tool call
  - finish event when the agent completes

Usage (called by sse_server.py)
---------------------------------
    async for event in run_streaming_agent(task, model):
        yield sse_frame(event)
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator, Optional

from agent.tools import Workspace, run_command
from streaming.stream_llm import stream_next_action
from streaming.config import AGENT_MODEL, MAX_STEPS, STEP_TIMEOUT_S


async def run_streaming_agent(
    task: str,
    model: str = AGENT_MODEL,
    workspace_dir: Optional[str] = None,
    require_approval: bool = False,
) -> AsyncIterator[dict]:
    """
    Async generator yielding structured events for each stage of the agent loop.

    Events:
      {"type": "start",      "task": str, "model": str}
      {"type": "step_start", "step": int, "max_steps": int}
      {"type": "token",      "text": str, "step": int, "elapsed_ms": float}
      {"type": "action",     "action": dict, "step": int, "elapsed_ms": float}
      {"type": "tool_start", "tool": str, "step": int}
      {"type": "tool_done",  "tool": str, "result": str, "step": int, "elapsed_ms": float}
      {"type": "step_timeout","step": int}
      {"type": "finish",     "result": str, "total_steps": int, "total_ms": float}
      {"type": "error",      "message": str}
    """
    ws = Workspace(base_dir=workspace_dir or ".")
    state: dict = {"task": task, "history": []}
    agent_t0 = time.perf_counter()

    yield {"type": "start", "task": task, "model": model}

    for step in range(1, MAX_STEPS + 1):
        yield {"type": "step_start", "step": step, "max_steps": MAX_STEPS}

        # Stream LLM tokens for this step
        action_dict = None
        step_elapsed = 0.0

        async for event in stream_next_action(state, model=model, timeout_s=STEP_TIMEOUT_S):
            if event["type"] == "token":
                yield {**event, "step": step}
            elif event["type"] == "action":
                action_dict = event["action"]
                step_elapsed = event["elapsed_ms"]
                yield {"type": "action", "action": action_dict, "step": step,
                       "elapsed_ms": step_elapsed}
                break
            elif event["type"] == "timeout":
                yield {"type": "step_timeout", "step": step}
                action_dict = {"action": "finish", "final": "Step timed out — stopping agent."}
                break
            elif event["type"] == "error":
                yield {"type": "error", "message": event["message"]}
                return

        if action_dict is None:
            yield {"type": "error", "message": "No action received from LLM"}
            return

        action = action_dict.get("action")

        # Handle finish
        if action == "finish":
            total_ms = round((time.perf_counter() - agent_t0) * 1000, 1)
            yield {
                "type": "finish",
                "result": action_dict.get("final", "Done."),
                "total_steps": step,
                "total_ms": total_ms,
            }
            return

        # Execute tool
        tool_t0 = time.perf_counter()
        yield {"type": "tool_start", "tool": action, "step": step}

        try:
            if action == "list_files":
                res = ws.list_files()
                result_text = res.output
            elif action == "read_file":
                res = ws.read_text(action_dict.get("path", ""))
                result_text = res.output
            elif action == "write_file":
                res = ws.write_text(action_dict.get("path", ""), action_dict.get("content", ""))
                result_text = res.output
            elif action == "run":
                res = run_command(action_dict.get("cmd", "echo 'no command'"))
                result_text = res.output
            else:
                result_text = f"Unknown action: {action}"
        except Exception as exc:
            result_text = f"Tool error: {exc}"

        tool_ms = round((time.perf_counter() - tool_t0) * 1000, 1)
        state["history"].append({"tool": action, "result": result_text})

        yield {
            "type": "tool_done",
            "tool": action,
            "result": result_text[:500],   # truncate long output in the event stream
            "step": step,
            "elapsed_ms": tool_ms,
        }

    total_ms = round((time.perf_counter() - agent_t0) * 1000, 1)
    yield {
        "type": "finish",
        "result": f"Stopped: reached max steps ({MAX_STEPS}).",
        "total_steps": MAX_STEPS,
        "total_ms": total_ms,
    }
