from __future__ import annotations

from typing import Dict, Any, List
import json
import os
import requests

from agent.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
    build_reflection_prompt,
)


class OllamaLLM:
    """Talks to a local Ollama server via /api/generate.

    Reliability features:
    - format="json": Ollama constrains decoding to valid JSON, so the model
      cannot emit markdown fences or prose prefixes in the first place. This is
      the structural fix for the "chatty model" problem; the {..} slice parser
      below is kept only as a belt-and-braces fallback.
    - keep_alive: the model stays resident between the up-to-20 agent steps
      instead of unloading and re-warming each call.
    - 600s timeout to survive the cold first-load of a 7B model on CPU.
    - num_predict cap to keep single-action responses short.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str = "http://localhost:11434",
        keep_alive: str = "30m",
    ):
        self.model = model or os.getenv("AGENT_MODEL", "qwen2.5-coder:7b")
        self.base_url = base_url.rstrip("/")
        self.keep_alive = keep_alive

    def _generate(self, prompt: str, *, system: str, as_json: bool) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0.2, "num_predict": 300},
        }
        if as_json:
            payload["format"] = "json"
        r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=600)
        r.raise_for_status()
        return r.json().get("response", "").strip()

    def next_action(
        self,
        state: Dict[str, Any],
        conversation: List[dict] | None = None,
    ) -> Dict[str, Any]:
        prompt = build_user_prompt(state["task"], state["history"], conversation)
        try:
            text = self._generate(prompt, system=SYSTEM_PROMPT, as_json=True)
        except requests.exceptions.ReadTimeout:
            return {
                "action": "finish",
                "final": (
                    "Ollama API call timed out. The model may still be loading or "
                    "your machine is slow.\nTry:\n"
                    "1) `ollama ps` to check if the model is running\n"
                    "2) a smaller model, e.g. AGENT_MODEL=qwen2.5-coder:3b\n"
                    "3) re-run after the model warms up"
                ),
            }
        except Exception as e:
            return {"action": "finish", "final": f"Failed to call Ollama API: {type(e).__name__}: {e}"}

        return self._parse_json(text)

    def reflect(self, task: str, history: List[dict], error: str) -> str:
        """Reflexion self-critique after a failed command. Plain text, best-effort."""
        prompt = build_reflection_prompt(task, history, error)
        try:
            return self._generate(prompt, system="You are a concise debugging assistant.", as_json=False)
        except Exception:
            return "(reflection unavailable)"

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        # With format="json" this almost always succeeds on the first try.
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        # Fallback: slice the first {...} span.
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                obj = json.loads(text[start:end + 1])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        return {
            "action": "finish",
            "final": (
                "Model did not return valid JSON.\n\n"
                f"Raw output:\n{text}\n\n"
                "Tip: ensure the Ollama version supports format=json, or lower temperature."
            ),
        }
