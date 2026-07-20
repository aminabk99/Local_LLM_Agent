from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import textwrap

# Commands that could damage the host outside the sandbox. run_command uses a
# shell, so a determined prompt can still try to escape the workspace (e.g.
# `cd .. && ...`). This denylist is defense-in-depth, not a real jail -- see the
# Security section of the README for the honest threat model.
_DESTRUCTIVE = [
    r"\brm\s+-rf\s+/",       # rm -rf /  (and /root, /home, etc.)
    r"\brm\s+-rf\s+~",
    r":\(\)\s*\{.*\};:",     # fork bomb
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r">\s*/dev/sd",
    r"\bshutdown\b|\breboot\b",
    r"\bchmod\s+-R\s+777\s+/",
]


@dataclass
class ToolResult:
    ok: bool
    output: str


class Workspace:
    """Safety: restrict all file operations to the workspace folder only."""

    def __init__(self, workspace_dir: str = "workspace"):
        self.root = Path(workspace_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _safe_path(self, rel_path: str) -> Path:
        p = (self.root / rel_path).resolve()
        # Path.is_relative_to (3.9+) is the correct, symlink-aware check.
        if p != self.root and not p.is_relative_to(self.root):
            raise ValueError("Path escapes workspace.")
        return p

    def list_files(self) -> ToolResult:
        try:
            files = [
                str(p.relative_to(self.root))
                for p in self.root.rglob("*")
                if p.is_file()
            ]
            return ToolResult(True, "\n".join(sorted(files)) if files else "(no files)")
        except Exception as e:
            return ToolResult(False, f"{type(e).__name__}: {e}")

    def read_text(self, rel_path: str, max_chars: int = 20_000) -> ToolResult:
        try:
            p = self._safe_path(rel_path)
            text = p.read_text(encoding="utf-8")
            if len(text) > max_chars:
                text = text[:max_chars] + f"\n... (truncated, {len(text)} chars total)"
            return ToolResult(True, text)
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


def is_destructive(cmd: str) -> bool:
    """True if the command matches a known host-damaging pattern."""
    return any(re.search(pat, cmd) for pat in _DESTRUCTIVE)


def run_command(cmd: str, cwd: str = "workspace", timeout_sec: int = 120) -> ToolResult:
    """Run a shell command inside the workspace. Used for tests/build/debug.

    The cwd is resolved so it points at the same absolute folder the Workspace
    uses regardless of where the server was launched from.
    """
    if is_destructive(cmd):
        return ToolResult(False, f"Blocked: command matches a destructive pattern: {cmd!r}")

    root = Path(cwd).resolve()
    root.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=str(root),
            capture_output=True, text=True, timeout=timeout_sec,
        )
        out = textwrap.dedent(f"""
        exit_code: {result.returncode}
        --- stdout ---
        {result.stdout.strip()}
        --- stderr ---
        {result.stderr.strip()}
        """).strip()
        return ToolResult(result.returncode == 0, out)
    except subprocess.TimeoutExpired as e:
        return ToolResult(False, f"TimeoutExpired after {timeout_sec}s: {e}")
    except Exception as e:
        return ToolResult(False, f"{type(e).__name__}: {e}")
