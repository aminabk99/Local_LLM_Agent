from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import subprocess
import textwrap

@dataclass
class ToolResult:
    ok: bool
    output: str

class Workspace:
    """Safety: restrict all file operations to workspace folder only."""
    def __init__(self, workspace_dir: str = "workspace"):
        self.root = Path(workspace_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _safe_path(self, rel_path: str) -> Path:
        p = (self.root / rel_path).resolve()
        if self.root not in p.parents and p != self.root:
            raise ValueError("Path escapes workspace.")
        return p

    def list_files(self) -> ToolResult:
        try:
            files = []
            for p in self.root.rglob("*"):
                if p.is_file():
                    files.append(str(p.relative_to(self.root)))
            return ToolResult(True, "\n".join(sorted(files)) if files else "(no files)")
        except Exception as e:
            return ToolResult(False, f"{type(e).__name__}: {e}")

    def read_text(self, rel_path: str) -> ToolResult:
        try:
            p = self._safe_path(rel_path)
            return ToolResult(True, p.read_text(encoding="utf-8"))
        except Exception as e:
            return ToolResult(False, f"{type(e).__name__}: {e}")

    def write_text(self, rel_path: str, content: str) -> ToolResult:
        try:
            p = self._safe_path(rel_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return ToolResult(True, f"Wrote {rel_path} ({len(content)} chars)")
        except Exception as e:
            return ToolResult(False, f"{type(e).__name__}: {e}")

def run_command(cmd: str, cwd: str = "workspace", timeout_sec: int = 120) -> ToolResult:
    """Run shell commands in workspace. Used for tests/build/debug."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        out = textwrap.dedent(f"""
        exit_code: {result.returncode}
        --- stdout ---
        {result.stdout.strip()}
        --- stderr ---
        {result.stderr.strip()}
        """).strip()
        return ToolResult(result.returncode == 0, out)
    except Exception as e:
        return ToolResult(False, f"{type(e).__name__}: {e}")

