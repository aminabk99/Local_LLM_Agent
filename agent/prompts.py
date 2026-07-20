"""Prompt construction for the ReAct-style JSON action loop.

The loop follows the ReAct pattern (Yao et al., 2023, "ReAct: Synergizing
Reasoning and Acting in Language Models"): at each step the model observes the
tool history and emits a single grounded action. On failure the agent runs a
Reflexion step (Shinn et al., 2023) whose verbal self-critique is fed back in as
an observation before the next action.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are an AI Software Engineer & Coding Agent.

You MUST respond with ONE valid JSON object only. No markdown, no prose.

You can choose ONE action per step from:
- "finish"      (requires: final)
- "list_files"
- "read_file"   (requires: path)
- "write_file"  (requires: path, content)
- "run"         (requires: cmd)

Rules:
- You may ONLY read/write files inside the workspace.
- Prefer small, incremental edits.
- After writing code, run a command to verify it.
- If a step fails, read the error and REFLECTION note, then fix and retry.
- When the task is done, use "finish" with a short summary in "final".
"""

# Keep individual observations from blowing up the context window on long runs.
_MAX_RESULT_CHARS = 1500


def _trim(text: str) -> str:
    text = str(text)
    if len(text) > _MAX_RESULT_CHARS:
        return text[:_MAX_RESULT_CHARS] + f"\n... (trimmed, {len(text)} chars)"
    return text


def build_user_prompt(
    task: str,
    history: list[dict],
    conversation: list[dict] | None = None,
) -> str:
    convo_text = ""
    for turn in (conversation or [])[-6:]:
        role = turn.get("role", "?")
        convo_text += f"\n{role.upper()}: {_trim(turn.get('content', ''))}"

    hist_text = ""
    for item in history[-12:]:
        tool = item.get("tool")
        hist_text += f"\nTOOL: {tool}\nRESULT:\n{_trim(item.get('result'))}\n"

    convo_block = f"CONVERSATION SO FAR:{convo_text}\n\n" if convo_text else ""

    return f"""{convo_block}TASK:
{task}

RECENT TOOL HISTORY:
{hist_text}

Now choose the next single action as JSON only.
"""


def build_reflection_prompt(task: str, history: list[dict], error: str) -> str:
    """Reflexion: ask the model to diagnose a failure in plain language.

    The result is fed back into the loop as an observation, not executed.
    """
    hist_text = "".join(
        f"\nTOOL: {i.get('tool')}\nRESULT:\n{_trim(i.get('result'))}\n"
        for i in history[-6:]
    )
    return f"""You are debugging your own failed action.

TASK:
{task}

RECENT STEPS:
{hist_text}

The last command FAILED with:
{_trim(error)}

In 1-3 sentences, explain the likely cause and what you will do differently on
the next step. Plain text only, no JSON.
"""
