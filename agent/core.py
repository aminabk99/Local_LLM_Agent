from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
from rich.console import Console
from agent.tools import Workspace, run_command

console = Console()

@dataclass
class AgentConfig:
    max_steps: int = 20
    require_approval: bool = True

class CodingAgent:
    def __init__(self, llm, workspace: Workspace, config: AgentConfig = AgentConfig()):
        self.llm = llm
        self.ws = workspace
        self.config = config

    def _approve(self, action_desc: str) -> bool:
        if not self.config.require_approval:
            return True
        console.print(f"\n[bold yellow]Approve action?[/bold yellow] {action_desc}")
        return input("Type 'y' to approve: ").strip().lower() == "y"

    def run(self, user_task: str) -> str:
        state: Dict[str, Any] = {"task": user_task, "history": []}

        for step in range(1, self.config.max_steps + 1):
            console.print(f"\n[bold cyan]Step {step}/{self.config.max_steps}[/bold cyan]")
            msg = self.llm.next_action(state)
            action = msg.get("action")

            if action == "finish":
                return msg.get("final", "Done.")

            if action == "list_files":
                res = self.ws.list_files()
                state["history"].append({"tool": "list_files", "result": res.output})

            elif action == "write_file":
                if self._approve(f"write_file {msg['path']}"):
                    res = self.ws.write_text(msg["path"], msg["content"])
                    state["history"].append({"tool": "write_file", "result": res.output})

            elif action == "run":
                if self._approve(f"run {msg['cmd']}"):
                    res = run_command(msg["cmd"])
                    state["history"].append({"tool": "run", "result": res.output})

            else:
                state["history"].append({"tool": "error", "result": f"Unknown action: {msg}"})

        return "Stopped: max steps reached."

