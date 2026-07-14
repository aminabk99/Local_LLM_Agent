"""
Streaming Ollama LLM for the agent action loop.

Drop-in async replacement for OllamaLLM.next_action().
Instead of waiting for the full response, yields tokens as they arrive,
then parses and returns the final action dict.

Usage
-----
    async for event in stream_next_action(state):
        if event["type"] == "token":
            print(event["text"], end="", flush=True)
        elif event["type"] == "action":
            action = event["action"]  # the parsed JSON dict
            break
        elif event["type"] == "error":
            ...
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncIterator

import httpx

from agent.prompts import SYSTEM_PROMPT, build_user_prompt
from streaming.config import OLLAMA_BASE_URL, AGENT_MODEL, STEP_TIMEOUT_S


def _parse_action(text: str) -> dict:
    """Parse JSON action from raw model output (mirrors OllamaLLM logic)."""
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {"action": "finish", "final": f"[Parse error] Model output was not valid JSON:\n{text}"}


async def stream_next_action(
    state: dict,
    model: str = AGENT_MODEL,
    timeout_s: float = STEP_TIMEOUT_S,
) -> AsyncIterator[dict]:
    """
    Async generator yielding streaming events for one agent step.

    Events:
      {"type": "token",  "text": str,   "elapsed_ms": float}
      {"type": "action", "action": dict, "elapsed_ms": float, "raw": str}
      {"type": "timeout","elapsed_ms": float}
      {"type": "error",  "message": str, "elapsed_ms": float}
    """
    payload = {
        "model": model,
        "prompt": build_user_prompt(state["task"], state["history"]),
        "system": SYSTEM_PROMPT,
        "stream": True,
        "options": {"temperature": 0.2, "num_predict": 300},
    }

    t0 = time.perf_counter()
    collected: list[str] = []

    try:
        async with httpx.AsyncClient(timeout=timeout_s + 5) as client:
            async with client.stream("POST", f"{OLLAMA_BASE_URL}/api/generate", json=payload) as resp:
                resp.raise_for_status()

                async def _gen():
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            collected.append(token)
                            elapsed = (time.perf_counter() - t0) * 1000
                            yield {"type": "token", "text": token, "elapsed_ms": round(elapsed, 1)}
                        if chunk.get("done"):
                            break

                try:
                    async for event in asyncio.wait_for(_exhaust(_gen()), timeout=timeout_s):
                        yield event
                except asyncio.TimeoutError:
                    elapsed = (time.perf_counter() - t0) * 1000
                    yield {"type": "timeout", "elapsed_ms": round(elapsed, 1)}
                    return

        raw = "".join(collected)
        action = _parse_action(raw)
        elapsed = (time.perf_counter() - t0) * 1000
        yield {"type": "action", "action": action, "elapsed_ms": round(elapsed, 1), "raw": raw}

    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        yield {"type": "error", "message": str(exc), "elapsed_ms": round(elapsed, 1)}


async def _exhaust(gen):
    async for item in gen:
        yield item
