from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

ALLOWED_COMMANDS: dict[str, list[str]] = {
    "python_compileall": ["python", "-m", "compileall", "app"],
}

def _safe_path(rel_path: str) -> Path:
    p = (WORKSPACE_ROOT / rel_path).resolve()
    if not str(p).startswith(str(WORKSPACE_ROOT.resolve())):
        raise ValueError("Unsafe path (outside workspace).")
    return p

def list_files(rel_dir: str = "app") -> Dict[str, Any]:
    """
    List files under rel_dir, excluding __pycache__ and *.pyc,
    and normalize paths to forward slashes for nicer JSON.
    """
    d = _safe_path(rel_dir)
    if not d.exists() or not d.is_dir():
        return {"ok": False, "error": f"Directory not found: {rel_dir}"}

    files: list[str] = []
    for p in d.rglob("*"):
        if not p.is_file():
            continue

        # Skip cache files
        if "__pycache__" in p.parts:
            continue
        if p.suffix.lower() == ".pyc":
            continue

        rel = p.relative_to(WORKSPACE_ROOT)
        # Make it look clean in JSON/UI
        files.append(rel.as_posix())

    files.sort()
    return {"ok": True, "files": files}

def read_file(rel_path: str) -> Dict[str, Any]:
    p = _safe_path(rel_path)
    if not p.exists() or not p.is_file():
        return {"ok": False, "error": f"File not found: {rel_path}"}
    content = p.read_text(encoding="utf-8", errors="replace")
    return {"ok": True, "path": Path(rel_path).as_posix(), "content": content}

def write_file(rel_path: str, content: str) -> Dict[str, Any]:
    p = _safe_path(rel_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return {"ok": True, "path": Path(rel_path).as_posix(), "bytes": len(content.encode("utf-8"))}

def run_command(command_key: str) -> Dict[str, Any]:
    if command_key not in ALLOWED_COMMANDS:
        return {"ok": False, "error": f"Command not allowed: {command_key}"}

    cmd = ALLOWED_COMMANDS[command_key]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
            shell=False,
        )
        return {
            "ok": proc.returncode == 0,
            "command": cmd,
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "")[-8000:],
            "stderr": (proc.stderr or "")[-8000:],
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Command timed out."}

TOOL_SPECS = [
    {
        "name": "list_files",
        "description": "List files recursively under a directory inside workspace (default: app).",
        "args_schema": {"rel_dir": "string"},
    },
    {
        "name": "read_file",
        "description": "Read a text file inside workspace (example: app/main.py).",
        "args_schema": {"rel_path": "string"},
    },
    {
        "name": "write_file",
        "description": "Write/overwrite a text file inside workspace.",
        "args_schema": {"rel_path": "string", "content": "string"},
    },
    {
        "name": "run_command",
        "description": "Run an allowlisted command. Allowed: python_compileall",
        "args_schema": {"command_key": "string"},
    },
]

def execute_tool_call(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if tool_name == "list_files":
            rel_dir = args.get("rel_dir", "app")
            return list_files(rel_dir)

        if tool_name == "read_file":
            rel_path = args.get("rel_path")
            if not rel_path:
                return {"ok": False, "error": "Missing required arg: rel_path"}
            return read_file(rel_path)

        if tool_name == "write_file":
            rel_path = args.get("rel_path")
            if not rel_path:
                return {"ok": False, "error": "Missing required arg: rel_path"}
            content = args.get("content", "")
            return write_file(rel_path, content)

        if tool_name == "run_command":
            command_key = args.get("command_key")
            if not command_key:
                return {"ok": False, "error": "Missing required arg: command_key"}
            return run_command(command_key)

        return {"ok": False, "error": f"Unknown tool: {tool_name}"}

    except Exception as e:
        return {"ok": False, "error": str(e)}

def extract_tool_call(text: str) -> Tuple[bool, Dict[str, Any]]:
    if not text:
        return (False, {})

    s = text.strip()

    if s.lower().startswith("json"):
        s = s[4:].strip()

    if "{" in s:
        s = s[s.find("{"):].strip()
    else:
        return (False, {})

    try:
        obj = json.loads(s)
    except Exception:
        return (False, {})

    tc = obj.get("tool_call")
    if isinstance(tc, dict) and "name" in tc and "args" in tc and isinstance(tc["args"], dict):
        return (True, obj)

    return (False, {})

