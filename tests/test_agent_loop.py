from agent.core import CodingAgent, AgentConfig
from agent.tools import Workspace


class ScriptedLLM:
    """Returns a fixed sequence of actions; records reflect() calls."""
    def __init__(self, actions):
        self.actions = list(actions)
        self.reflections = 0

    def next_action(self, state, conversation=None):
        return self.actions.pop(0) if self.actions else {"action": "finish", "final": "done"}

    def reflect(self, task, history, error):
        self.reflections += 1
        return "It failed because the file was missing; I'll create it first."


def test_write_then_finish(tmp_path):
    ws = Workspace(str(tmp_path / "ws"))
    llm = ScriptedLLM([
        {"action": "write_file", "path": "hi.txt", "content": "yo"},
        {"action": "finish", "final": "wrote hi.txt"},
    ])
    agent = CodingAgent(llm, ws, AgentConfig(require_approval=False))
    reply, steps = agent.run("make a file")
    assert reply == "wrote hi.txt" and steps == 2
    assert ws.read_text("hi.txt").output == "yo"


def test_reflexion_fires_on_failed_command(tmp_path):
    ws = Workspace(str(tmp_path / "ws"))
    llm = ScriptedLLM([
        {"action": "run", "cmd": "false"},        # exits non-zero -> reflect
        {"action": "finish", "final": "gave up gracefully"},
    ])
    agent = CodingAgent(llm, ws, AgentConfig(reflect_on_failure=True))
    reply, steps = agent.run("run something that fails")
    assert llm.reflections == 1
    assert reply == "gave up gracefully"


def test_shell_can_be_disabled(tmp_path):
    ws = Workspace(str(tmp_path / "ws"))
    llm = ScriptedLLM([
        {"action": "run", "cmd": "echo hi"},
        {"action": "finish", "final": "ok"},
    ])
    agent = CodingAgent(llm, ws, AgentConfig(allow_shell=False))
    agent.run("try shell")
    # no reflection, command was blocked not executed
    assert llm.reflections == 0


def test_max_steps_guard(tmp_path):
    ws = Workspace(str(tmp_path / "ws"))
    llm = ScriptedLLM([{"action": "list_files"} for _ in range(50)])
    agent = CodingAgent(llm, ws, AgentConfig(max_steps=3))
    reply, steps = agent.run("loop forever")
    assert steps == 3 and "max steps" in reply
