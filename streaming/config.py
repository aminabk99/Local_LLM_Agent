"""Streaming configuration for Local_LLM_Agent."""
import os
from pathlib import Path

OLLAMA_BASE_URL: str  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
AGENT_MODEL:    str   = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
VISION_MODEL:   str   = os.getenv("OLLAMA_VISION_MODEL", "llava")

STEP_TIMEOUT_S: float = float(os.getenv("STEP_TIMEOUT_S", "30"))   # per agent step
MAX_STEPS:      int   = int(os.getenv("MAX_STEPS", "20"))

LATENCY_LOG: Path = Path(__file__).parent / "latency_log.jsonl"
