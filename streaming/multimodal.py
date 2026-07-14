"""
Multimodal task injection for Local_LLM_Agent.

When the user provides a screenshot of an error (e.g. a browser console
error, a terminal traceback, a test failure) alongside the task description,
this module sends the image to llava first to extract visible error text,
then injects that context into the agent's task string.

Result: the agent loop gets a richer task like:
  "Fix the TypeError: 'NoneType' object is not iterable
   [from screenshot: line 42 of app.py, variable 'results' is None]"

Usage (via the /stream/run endpoint)
--------------------------------------
    POST /stream/run
    {
        "task": "Fix the failing tests",
        "image_b64": "<base64 screenshot>",  // optional
        "model": "qwen2.5-coder:7b"          // optional
    }
"""

from __future__ import annotations

import base64

import httpx

from streaming.config import OLLAMA_BASE_URL, VISION_MODEL

_MAX_IMAGE_BYTES = 5 * 1024 * 1024

_VISION_PROMPT = (
    "This is a screenshot showing an error, test failure, or bug. "
    "Extract all error messages, exception types, line numbers, and stack trace "
    "information visible in the image. Be concise and precise. "
    "Format as: '<ErrorType>: <message> at <file>:<line>' per line. "
    "If no errors are visible, say 'No errors visible.'"
)


def extract_error_from_image(image_b64: str) -> str:
    """
    Send a screenshot to llava and return extracted error context.
    Synchronous — run in a thread executor inside async contexts.
    """
    try:
        img_bytes = base64.b64decode(image_b64)
    except Exception:
        return "[Invalid image data]"

    if len(img_bytes) > _MAX_IMAGE_BYTES:
        return f"[Image too large: {len(img_bytes)//1024//1024}MB > 5MB limit]"

    payload = {
        "model": VISION_MODEL,
        "prompt": _VISION_PROMPT,
        "images": [image_b64],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 200},
    }

    try:
        r = httpx.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=60.0)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as exc:
        return f"[Vision model unavailable: {exc}]"


def inject_image_context(task: str, image_b64: str) -> str:
    """
    Extract error context from the image and prepend it to the task string.
    Returns the augmented task string.
    """
    error_context = extract_error_from_image(image_b64)
    return (
        f"{task}\n\n"
        f"[Screenshot context — errors visible in the provided image:]\n"
        f"{error_context}"
    )
