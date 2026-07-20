import pytest
from agent.tools import Workspace, run_command, is_destructive


def test_write_read_list_roundtrip(tmp_path):
    ws = Workspace(str(tmp_path / "ws"))
    assert ws.write_text("a/b.txt", "hello").ok
    r = ws.read_text("a/b.txt")
    assert r.ok and r.output == "hello"
    assert "a/b.txt" in ws.list_files().output


def test_path_traversal_is_blocked(tmp_path):
    ws = Workspace(str(tmp_path / "ws"))
    with pytest.raises(ValueError):
        ws._safe_path("../escape.txt")
    # The public API surfaces the failure as ok=False, not a crash.
    assert ws.read_text("../../etc/passwd").ok is False


def test_read_truncates_large_files(tmp_path):
    ws = Workspace(str(tmp_path / "ws"))
    ws.write_text("big.txt", "x" * 50_000)
    out = ws.read_text("big.txt", max_chars=1000).output
    assert "truncated" in out and len(out) < 2000


def test_destructive_commands_blocked():
    assert is_destructive("rm -rf /")
    assert is_destructive("sudo rm -rf ~")
    assert not is_destructive("pytest -q")


def test_run_command_executes_in_workspace(tmp_path):
    ws = Workspace(str(tmp_path / "ws"))
    res = run_command("echo hi", cwd=str(ws.root))
    assert res.ok and "hi" in res.output


def test_run_command_refuses_destructive(tmp_path):
    ws = Workspace(str(tmp_path / "ws"))
    res = run_command("rm -rf /", cwd=str(ws.root))
    assert res.ok is False and "Blocked" in res.output
