from __future__ import annotations
from agent.core import CodingAgent, AgentConfig
from agent.llm_ollama import OllamaLLM
from agent.tools import Workspace


def run_agent(task: str, history: list[dict], workspace: str = "workspace") -> tuple[str, int]:
    """
    Run the coding agent on a task.
    Returns (final_reply, steps_taken).
    """
    llm = OllamaLLM(model="qwen2.5-coder:7b")
    ws = Workspace(workspace_dir=workspace)
    config = AgentConfig(max_steps=20, require_approval=False)
    agent = CodingAgent(llm=llm, workspace=ws, config=config)

    steps = 0
    state = {"task": task, "history": []}

    for step in range(1, config.max_steps + 1):
        steps = step
        msg = llm.next_action(state)
        action = msg.get("action")

        if action == "finish":
            return msg.get("final", "Done."), steps

        elif action == "list_files":
            res = ws.list_files()
            state["history"].append({"tool": "list_files", "result": res.output})

        elif action == "read_file":
            res = ws.read_text(msg.get("path", ""))
            state["history"].append({"tool": "read_file", "result": res.output})

        elif action == "write_file":
            res = ws.write_text(msg.get("path", ""), msg.get("content", ""))
            state["history"].append({"tool": "write_file", "result": res.output})

        elif action == "run":
            from agent.tools import run_command
            res = run_command(msg.get("cmd", ""))
            state["history"].append({"tool": "run", "result": res.output})

        else:
            state["history"].append({"tool": "error", "result": f"Unknown action: {msg}"})

    return "Stopped: max steps reached.", steps
