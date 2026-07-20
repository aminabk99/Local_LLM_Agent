"""The single agent loop (ReAct + Reflexion).

This is the one place the action loop lives. `agent.agent.run_agent` is a thin
wrapper around it so the FastAPI server and the CLI share identical behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from rich.console import Console

from agent.tools import Workspace, run_command

console = Console()


@dataclass
class AgentConfig:
    max_steps: int = 20
    require_approval: bool = False   # CLI human-in-the-loop for write/run
    allow_shell: bool = True         # gate the `run` action entirely
    reflect_on_failure: bool = True  # Reflexion step after a failed command


class CodingAgent:
    def __init__(self, llm, workspace: Workspace, config: AgentConfig | None = None):
        self.llm = llm
        self.ws = workspace
        self.config = config or AgentConfig()

    def _approve(self, action_desc: str) -> bool:
        if not self.config.require_approval:
            return True
        console.print(f"\n[bold yellow]Approve action?[/bold yellow] {action_desc}")
        try:
            return input("Type 'y' to approve: ").strip().lower() == "y"
        except EOFError:
            return False

    def run(self, task: str, conversation: List[dict] | None = None) -> Tuple[str, int]:
        state: Dict[str, Any] = {"task": task, "history": []}
        steps = 0

        for step in range(1, self.config.max_steps + 1):
            steps = step
            msg = self.llm.next_action(state, conversation)
            action = msg.get("action")

            if action == "finish":
                return msg.get("final", "Done."), steps

            if action == "list_files":
                res = self.ws.list_files()
                state["history"].append({"tool": "list_files", "result": res.output})

            elif action == "read_file":
                res = self.ws.read_text(msg.get("path", ""))
                state["history"].append({"tool": "read_file", "result": res.output})

            elif action == "write_file":
                if self._approve(f"write_file {msg.get('path')}"):
                    res = self.ws.write_text(msg.get("path", ""), msg.get("content", ""))
                    state["history"].append({"tool": "write_file", "result": res.output})
                else:
                    state["history"].append({"tool": "write_file", "result": "Skipped (not approved)."})

            elif action == "run":
                if not self.config.allow_shell:
                    state["history"].append({"tool": "run", "result": "Blocked: shell execution is disabled."})
                elif self._approve(f"run {msg.get('cmd')}"):
                    res = run_command(msg.get("cmd", ""), cwd=str(self.ws.root))
                    state["history"].append({"tool": "run", "result": res.output})
                    if not res.ok and self.config.reflect_on_failure:
                        note = self.llm.reflect(task, state["history"], res.output)
                        state["history"].append({"tool": "reflection", "result": note})
                else:
                    state["history"].append({"tool": "run", "result": "Skipped (not approved)."})

            else:
                state["history"].append({"tool": "error", "result": f"Unknown action: {msg}"})

        return "Stopped: max steps reached.", steps
