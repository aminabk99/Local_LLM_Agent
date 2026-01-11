from uuid import uuid4
import json

from fastapi import APIRouter, HTTPException

from app.schemas import ChatRequest, ChatResponse
from app.services.memory import store
from app.services.ollama_client import generate_reply
from app.services.tools import TOOL_SPECS, execute_tool_call, extract_tool_call

router = APIRouter(tags=["chat"])

SYSTEM_AGENT_PROMPT = """You are a coding agent running inside a FastAPI server.

You have access to tools. Use them ONLY when needed.

Tool calling rules:
- If you need a tool, output ONLY valid JSON (no extra text, no backticks) in this exact shape:
  {{
    "tool_call": {{
      "name": "<tool_name>",
      "args": {{ ... }}
    }}
  }}

- If you do NOT need a tool, respond normally with the final answer (no JSON).

Available tools:
{tools}

Safety:
- You can only read/write files inside the workspace directory.
- You can only run allowlisted commands via run_command.
"""

def try_direct_tool(msg: str):
    """
    Fast-path: if the user explicitly asks to use a tool, run it immediately
    WITHOUT calling the model (prevents slow/timeout LLM calls).
    """
    m = msg.lower().strip()

    if "use the list_files tool" in m or m.startswith("list_files"):
        parts = msg.split()
        if len(parts) >= 2 and parts[0].lower().startswith("list_files"):
            return ("list_files", {"rel_dir": parts[1]})
        return ("list_files", {"rel_dir": "app"})

    if "use the read_file tool" in m or m.startswith("read_file"):
        parts = msg.split()
        if len(parts) >= 2:
            return ("read_file", {"rel_path": parts[1]})
        return ("read_file", {"rel_path": "app/main.py"})

    if "use the write_file tool" in m or m.startswith("write_file"):
        lines = msg.splitlines()
        first = lines[0].strip()
        parts = first.split(maxsplit=1)
        if len(parts) >= 2:
            rel_path = parts[1].strip()
            content = "\n".join(lines[1:]) if len(lines) > 1 else ""
            return ("write_file", {"rel_path": rel_path, "content": content})
        return None

    if "run python_compileall" in m or "use the run_command tool" in m:
        return ("run_command", {"command_key": "python_compileall"})

    return None

def build_prompt(session_id: str, user_msg: str) -> str:
    history = store.get_history(session_id)

    tools_text = "\n".join(
        f"- {t['name']}: {t['description']} args={t['args_schema']}"
        for t in TOOL_SPECS
    )
    sys = SYSTEM_AGENT_PROMPT.format(tools=tools_text)

    lines = [f"SYSTEM:\n{sys}\n"]
    for m in history:
        lines.append(f"{m.role.upper()}:\n{m.content}\n")
    lines.append(f"USER:\n{user_msg}\nASSISTANT:\n")
    return "\n".join(lines)

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    msg = req.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session_id = req.session_id or str(uuid4())
    request_id = str(uuid4())

    store.append(session_id, "user", msg)

    # ✅ Fast-path: direct tool execution (no LLM call)
    direct = try_direct_tool(msg)
    if direct:
        tool_name, args = direct
        tool_result = execute_tool_call(tool_name, args)

        store.append(
            session_id,
            "tool",
            f"{tool_name} result:\n{json.dumps(tool_result, indent=2)}",
        )

        # ✅ Return tool_result as REAL JSON (not string)
        return ChatResponse(
            request_id=request_id,
            session_id=session_id,
            reply=f"Tool '{tool_name}' executed.",
            tool_result=tool_result,
        )

    # Agent loop (LLM decides tool-call or final answer)
    tool_calls_used = 0
    while True:
        prompt = build_prompt(session_id, msg)

        try:
            assistant_text = generate_reply(prompt)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Ollama error: {str(e)}")

        is_tool, obj = extract_tool_call(assistant_text)

        if not is_tool:
            store.append(session_id, "assistant", assistant_text)
            return ChatResponse(
                request_id=request_id,
                session_id=session_id,
                reply=assistant_text,
                tool_result=None,
            )

        tool_calls_used += 1
        if tool_calls_used > 4:
            final = "Tool limit reached. Please ask in smaller steps."
            store.append(session_id, "assistant", final)
            return ChatResponse(request_id=request_id, session_id=session_id, reply=final)

        tc = obj["tool_call"]
        tool_name = tc["name"]
        args = tc.get("args", {})

        store.append(session_id, "assistant", assistant_text)

        tool_result = execute_tool_call(tool_name, args)

        store.append(
            session_id,
            "tool",
            f"{tool_name} result:\n{json.dumps(tool_result, indent=2)}",
        )

        # Let the model read the tool result and produce final answer
