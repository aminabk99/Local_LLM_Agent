"""Thin entry point: build a CodingAgent from env config and run one task.

Kept as a stable function for the FastAPI server (`main.py`). All loop logic
lives in `agent.core.CodingAgent` -- this module no longer duplicates it.
"""

from __future__ import annotations

import os

from agent.core import CodingAgent, AgentConfig
from agent.llm_ollama import OllamaLLM
from agent.tools import Workspace


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def run_agent(
    task: str,
    history: list[dict] | None = None,
    workspace: str = "workspace",
) -> tuple[str, int]:
    """Run the coding agent on a task.

    `history` is the prior conversation ([{role, content}, ...]) so multi-turn
    sessions actually carry context into the loop.
    Returns (final_reply, steps_taken).
    """
    llm = OllamaLLM()
    ws = Workspace(workspace_dir=workspace)
    config = AgentConfig(
        max_steps=int(os.getenv("AGENT_MAX_STEPS", "20")),
        require_approval=_env_bool("AGENT_REQUIRE_APPROVAL", False),
        allow_shell=_env_bool("AGENT_ALLOW_SHELL", True),
        reflect_on_failure=_env_bool("AGENT_REFLECT", True),
    )
    agent = CodingAgent(llm=llm, workspace=ws, config=config)
    return agent.run(task, conversation=history or [])
