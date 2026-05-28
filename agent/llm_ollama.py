from __future__ import annotations

from typing import Dict, Any
import json
import requests

from agent.prompts import SYSTEM_PROMPT, build_user_prompt


class OllamaLLM:
    """
    Talks to a local Ollama server (default: http://localhost:11434) using /api/generate.

    Fixes included:
    - Much longer timeout (600s) to avoid ReadTimeout on first model warm-up
    - num_predict cap to keep responses short and fast (important for JSON-only agents)
    - Slightly lower temperature for more consistent JSON formatting
    """

    def __init__(self, model: str = "qwen2.5-coder:7b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def next_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "prompt": build_user_prompt(state["task"], state["history"]),
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 300,   # caps output tokens (prevents rambling + helps avoid timeouts)
            },
        }

        # Longer timeout to handle slow first-run model loading
        try:
            r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=600)
            r.raise_for_status()
        except requests.exceptions.ReadTimeout:
            return {
                "action": "finish",
                "final": (
                    "Ollama API call timed out. The model may still be loading or your machine is slow.\n"
                    "Try:\n"
                    "1) Run `ollama ps` to check if the model is running\n"
                    "2) Use a smaller model like `qwen2.5-coder:3b`\n"
                    "3) Re-run the agent after the model warms up"
                ),
            }
        except Exception as e:
            return {"action": "finish", "final": f"Failed to call Ollama API: {type(e).__name__}: {e}"}

        text = r.json().get("response", "").strip()

        # Strict JSON parse, with a recovery attempt if the model adds extra text
        try:
            obj = json.loads(text)
            if not isinstance(obj, dict):
                raise ValueError("Model returned JSON but not an object.")
            return obj
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    obj = json.loads(text[start:end + 1])
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    pass

            # If we can't recover JSON, stop and show raw output to you
            return {
                "action": "finish",
                "final": (
                    "Model did not return valid JSON.\n\n"
                    f"Raw output:\n{text}\n\n"
                    "Tip: If this keeps happening, we can tighten the prompt or reduce temperature further."
                ),
            }
