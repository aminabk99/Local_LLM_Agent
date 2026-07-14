#!/usr/bin/env python3
"""
Generate the DPO (Direct Preference Optimisation) dataset.

Each example is a (prompt, chosen, rejected) triple:
  - prompt   = system + task + history (same format as SFT)
  - chosen   = the correct JSON-only response
  - rejected = a plausible but bad response (one of several failure modes)

Failure modes covered:
  1. Markdown fence wrapping  — JSON inside ```json ... ```
  2. Prose prefix             — "Sure! I'll do that. {json}"
  3. Prose suffix             — "{json}\n\nI've now written the file..."
  4. Wrong action keys        — missing required key (e.g. no "path" in read_file)
  5. Multiple actions         — model outputs two JSON objects instead of one
  6. Natural language only    — no JSON at all, just a sentence

Output: finetune/data/dpo_dataset.jsonl  (one JSON object per line)

Usage:
    python -m finetune.data.generate_dpo
"""

from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(7)

SYSTEM_PROMPT = """You are an AI Software Engineer & Coding Agent.

You MUST respond with ONE valid JSON object only. No markdown.

You can choose ONE action per step from:
- "finish"
- "list_files"
- "read_file"   (requires: path)
- "write_file"  (requires: path, content)
- "run"         (requires: cmd)

Rules:
- You may ONLY read/write files inside the workspace.
- Prefer small, incremental edits.
- After writing code, run a command to verify.
- If you see an error, fix it and try again.
"""


def _prompt(task: str, history: list[dict] | None = None) -> str:
    hist_text = ""
    if history:
        for item in history:
            hist_text += f"\nTOOL: {item['tool']}\nRESULT:\n{item['result']}\n"
    return (
        f"TASK:\n{task}\n\n"
        f"RECENT TOOL HISTORY:\n{hist_text or '(none yet)'}\n\n"
        "Now choose the next single action as JSON only."
    )


def _ex(task: str, chosen: dict, rejected: str, history: list[dict] | None = None) -> dict:
    return {
        "system": SYSTEM_PROMPT,
        "prompt": _prompt(task, history),
        "chosen": json.dumps(chosen),
        "rejected": rejected,
    }


# ---------------------------------------------------------------------------
# Failure mode 1: Markdown fence wrapping (15 examples)
# ---------------------------------------------------------------------------
MARKDOWN_FENCE_EXAMPLES = [
    _ex(
        "Create a hello world Python script.",
        chosen={"action": "write_file", "path": "hello.py", "content": "print('Hello, World!')\n"},
        rejected='```json\n{"action": "write_file", "path": "hello.py", "content": "print(\'Hello, World!\')\\n"}\n```',
    ),
    _ex(
        "List files in the workspace.",
        chosen={"action": "list_files"},
        rejected='```json\n{"action": "list_files"}\n```',
    ),
    _ex(
        "Run the test suite.",
        chosen={"action": "run", "cmd": "pytest tests/ -v"},
        rejected='```\n{"action": "run", "cmd": "pytest tests/ -v"}\n```',
    ),
    _ex(
        "Read main.py before editing it.",
        chosen={"action": "read_file", "path": "main.py"},
        rejected='```json\n{"action": "read_file", "path": "main.py"}\n```\n',
    ),
    _ex(
        "Finish — the task is done.",
        chosen={"action": "finish", "final": "Task completed successfully."},
        rejected='```json\n{"action": "finish", "final": "Task completed successfully."}\n```',
    ),
    _ex(
        "Install requirements.",
        chosen={"action": "run", "cmd": "pip install -r requirements.txt"},
        rejected='Here is the action:\n```json\n{"action": "run", "cmd": "pip install -r requirements.txt"}\n```',
    ),
    _ex(
        "Write a .gitignore.",
        chosen={"action": "write_file", "path": ".gitignore", "content": "__pycache__/\n*.pyc\n.env\n"},
        rejected='```python\n{"action": "write_file", "path": ".gitignore", "content": "__pycache__/\\n*.pyc\\n.env\\n"}\n```',
    ),
    _ex(
        "Read config.json.",
        chosen={"action": "read_file", "path": "config.json"},
        rejected='Action:\n```\n{"action": "read_file", "path": "config.json"}\n```',
    ),
    _ex(
        "Run mypy on the codebase.",
        chosen={"action": "run", "cmd": "mypy . --strict"},
        rejected='```json\n{\n  "action": "run",\n  "cmd": "mypy . --strict"\n}\n```',
    ),
    _ex(
        "Write a simple calculator.",
        chosen={"action": "write_file", "path": "calc.py", "content": "def add(a, b): return a + b\n"},
        rejected='```json\n{"action": "write_file", "path": "calc.py", "content": "def add(a, b): return a + b\\n"}\n```\n',
    ),
    _ex(
        "Check what files exist.",
        chosen={"action": "list_files"},
        rejected='Response:\n```json\n{"action": "list_files"}\n```',
    ),
    _ex(
        "Run the script.",
        chosen={"action": "run", "cmd": "python main.py"},
        rejected='```\n{"action": "run", "cmd": "python main.py"}\n```',
        history=[{"tool": "write_file", "result": "Written: main.py"}],
    ),
    _ex(
        "Create a README.",
        chosen={"action": "write_file", "path": "README.md", "content": "# Project\n\nA simple project.\n"},
        rejected='```markdown\n{"action": "write_file", "path": "README.md", "content": "# Project\\n\\nA simple project.\\n"}\n```',
    ),
    _ex(
        "Finish the task.",
        chosen={"action": "finish", "final": "Done. All files created and verified."},
        rejected='My response:\n```json\n{"action": "finish", "final": "Done. All files created and verified."}\n```',
    ),
    _ex(
        "Run pytest with coverage.",
        chosen={"action": "run", "cmd": "pytest --cov=. -v"},
        rejected='```sh\n{"action": "run", "cmd": "pytest --cov=. -v"}\n```',
    ),
]


# ---------------------------------------------------------------------------
# Failure mode 2: Prose prefix (15 examples)
# ---------------------------------------------------------------------------
PROSE_PREFIX_EXAMPLES = [
    _ex(
        "Write a hello world script.",
        chosen={"action": "write_file", "path": "hello.py", "content": "print('Hello, World!')\n"},
        rejected='Sure! I\'ll create a simple hello world script for you. {"action": "write_file", "path": "hello.py", "content": "print(\'Hello, World!\')\\n"}',
    ),
    _ex(
        "List the workspace files.",
        chosen={"action": "list_files"},
        rejected='I\'ll start by checking what files exist in the workspace. {"action": "list_files"}',
    ),
    _ex(
        "Run the tests.",
        chosen={"action": "run", "cmd": "pytest -v"},
        rejected='To run the tests, I\'ll use pytest. {"action": "run", "cmd": "pytest -v"}',
    ),
    _ex(
        "Read utils.py.",
        chosen={"action": "read_file", "path": "utils.py"},
        rejected='Let me read the utils.py file first to understand its contents. {"action": "read_file", "path": "utils.py"}',
    ),
    _ex(
        "Write a requirements file.",
        chosen={"action": "write_file", "path": "requirements.txt", "content": "flask\nrequests\n"},
        rejected='Great task! Here\'s the action to create a requirements.txt: {"action": "write_file", "path": "requirements.txt", "content": "flask\\nrequests\\n"}',
    ),
    _ex(
        "Install dependencies.",
        chosen={"action": "run", "cmd": "pip install -r requirements.txt"},
        rejected='Of course! I will install the dependencies using pip. {"action": "run", "cmd": "pip install -r requirements.txt"}',
    ),
    _ex(
        "Read the config file.",
        chosen={"action": "read_file", "path": "config.json"},
        rejected='Before making changes, let me read config.json. {"action": "read_file", "path": "config.json"}',
    ),
    _ex(
        "Finish the task.",
        chosen={"action": "finish", "final": "Completed. All tests pass."},
        rejected='The task is complete! {"action": "finish", "final": "Completed. All tests pass."}',
    ),
    _ex(
        "Write an empty __init__.py.",
        chosen={"action": "write_file", "path": "__init__.py", "content": ""},
        rejected='I\'ll create an empty __init__.py file. {"action": "write_file", "path": "__init__.py", "content": ""}',
    ),
    _ex(
        "Run the main script.",
        chosen={"action": "run", "cmd": "python main.py"},
        rejected='Now I\'ll execute the main script to verify it works. {"action": "run", "cmd": "python main.py"}',
        history=[{"tool": "write_file", "result": "Written: main.py"}],
    ),
    _ex(
        "List files to start the task.",
        chosen={"action": "list_files"},
        rejected='As a first step, I should see what\'s available. {"action": "list_files"}',
    ),
    _ex(
        "Create a setup.py.",
        chosen={"action": "write_file", "path": "setup.py", "content": "from setuptools import setup\nsetup(name='myproject')\n"},
        rejected='To package this project, I\'ll write a setup.py. {"action": "write_file", "path": "setup.py", "content": "from setuptools import setup\\nsetup(name=\'myproject\')\\n"}',
    ),
    _ex(
        "Format code with black.",
        chosen={"action": "run", "cmd": "black ."},
        rejected='I\'ll format all Python files using black. {"action": "run", "cmd": "black ."}',
    ),
    _ex(
        "Read the README.",
        chosen={"action": "read_file", "path": "README.md"},
        rejected='Let me check the README for instructions. {"action": "read_file", "path": "README.md"}',
    ),
    _ex(
        "Finish — tests are passing.",
        chosen={"action": "finish", "final": "All 5 tests pass. Task done."},
        rejected='All tests are passing now, so I can finish. {"action": "finish", "final": "All 5 tests pass. Task done."}',
    ),
]


# ---------------------------------------------------------------------------
# Failure mode 3: Wrong / missing keys (15 examples)
# ---------------------------------------------------------------------------
WRONG_KEYS_EXAMPLES = [
    _ex(
        "Read main.py.",
        chosen={"action": "read_file", "path": "main.py"},
        rejected='{"action": "read_file"}',  # missing path
    ),
    _ex(
        "Write a hello world script.",
        chosen={"action": "write_file", "path": "hello.py", "content": "print('Hello, World!')\n"},
        rejected='{"action": "write_file", "path": "hello.py"}',  # missing content
    ),
    _ex(
        "Run pytest.",
        chosen={"action": "run", "cmd": "pytest -v"},
        rejected='{"action": "run"}',  # missing cmd
    ),
    _ex(
        "Finish the task.",
        chosen={"action": "finish", "final": "Done."},
        rejected='{"action": "finish"}',  # missing final
    ),
    _ex(
        "Write a config file.",
        chosen={"action": "write_file", "path": "config.json", "content": "{}"},
        rejected='{"action": "write", "file": "config.json", "text": "{}"}',  # wrong key names
    ),
    _ex(
        "Read utils.py.",
        chosen={"action": "read_file", "path": "utils.py"},
        rejected='{"action": "read", "filename": "utils.py"}',  # wrong action name
    ),
    _ex(
        "Run the test suite.",
        chosen={"action": "run", "cmd": "pytest"},
        rejected='{"action": "execute", "command": "pytest"}',  # wrong key names
    ),
    _ex(
        "List files.",
        chosen={"action": "list_files"},
        rejected='{"action": "ls"}',  # wrong action name
    ),
    _ex(
        "Write requirements.txt.",
        chosen={"action": "write_file", "path": "requirements.txt", "content": "flask\n"},
        rejected='{"action": "write_file", "content": "flask\\n"}',  # missing path
    ),
    _ex(
        "Read config.json.",
        chosen={"action": "read_file", "path": "config.json"},
        rejected='{"action": "read_file", "file": "config.json"}',  # wrong key (file vs path)
    ),
    _ex(
        "Run pip install.",
        chosen={"action": "run", "cmd": "pip install flask"},
        rejected='{"action": "run", "command": "pip install flask"}',  # wrong key (command vs cmd)
    ),
    _ex(
        "Create main.py with a main function.",
        chosen={"action": "write_file", "path": "main.py", "content": "def main():\n    pass\n"},
        rejected='{"action": "create_file", "path": "main.py", "content": "def main():\\n    pass\\n"}',  # wrong action
    ),
    _ex(
        "Finish the task with a summary.",
        chosen={"action": "finish", "final": "Task complete."},
        rejected='{"action": "done", "message": "Task complete."}',  # wrong action and key
    ),
    _ex(
        "Read the error log.",
        chosen={"action": "read_file", "path": "error.log"},
        rejected='{"type": "read_file", "path": "error.log"}',  # 'type' instead of 'action'
    ),
    _ex(
        "Run black on the project.",
        chosen={"action": "run", "cmd": "black ."},
        rejected='{"action": "run", "cmd": "black .", "description": "Format Python files"}',  # extra key — minor but inconsistent
    ),
]


# ---------------------------------------------------------------------------
# Failure mode 4: Natural language only (15 examples)
# ---------------------------------------------------------------------------
NATURAL_LANGUAGE_EXAMPLES = [
    _ex(
        "Write a hello world script.",
        chosen={"action": "write_file", "path": "hello.py", "content": "print('Hello, World!')\n"},
        rejected="I would create a file called hello.py containing a print statement for Hello World.",
    ),
    _ex(
        "List the files.",
        chosen={"action": "list_files"},
        rejected="I'll check what files are in the workspace.",
    ),
    _ex(
        "Run the test suite.",
        chosen={"action": "run", "cmd": "pytest -v"},
        rejected="The next step is to run pytest to verify the tests pass.",
    ),
    _ex(
        "Read app.py.",
        chosen={"action": "read_file", "path": "app.py"},
        rejected="I should read app.py to understand its current content.",
    ),
    _ex(
        "Finish the task.",
        chosen={"action": "finish", "final": "Done."},
        rejected="The task appears to be complete now.",
    ),
    _ex(
        "Create a config file.",
        chosen={"action": "write_file", "path": "config.json", "content": "{}\n"},
        rejected="I would write a config.json file with an empty JSON object.",
    ),
    _ex(
        "Install dependencies.",
        chosen={"action": "run", "cmd": "pip install -r requirements.txt"},
        rejected="We should install the packages listed in requirements.txt using pip.",
    ),
    _ex(
        "Read the README.",
        chosen={"action": "read_file", "path": "README.md"},
        rejected="Let me check the README file for context about this project.",
    ),
    _ex(
        "Write a .gitignore.",
        chosen={"action": "write_file", "path": ".gitignore", "content": "__pycache__/\n*.pyc\n"},
        rejected="I'll create a .gitignore with standard Python patterns.",
    ),
    _ex(
        "Run the script after writing it.",
        chosen={"action": "run", "cmd": "python script.py"},
        rejected="Now that the script is written, I need to execute it.",
        history=[{"tool": "write_file", "result": "Written: script.py"}],
    ),
    _ex(
        "Start by listing files.",
        chosen={"action": "list_files"},
        rejected="First, I should look at what files already exist in the project directory.",
    ),
    _ex(
        "Finish after successful tests.",
        chosen={"action": "finish", "final": "All tests pass."},
        rejected="Since the tests are passing, the task is complete.",
        history=[{"tool": "run", "result": "5 passed in 0.1s"}],
    ),
    _ex(
        "Fix the syntax error.",
        chosen={"action": "read_file", "path": "buggy.py"},
        rejected="I need to look at the file with the syntax error before I can fix it.",
    ),
    _ex(
        "Format the code.",
        chosen={"action": "run", "cmd": "black ."},
        rejected="Black should be used to format all the Python files in the project.",
    ),
    _ex(
        "Check if the server starts.",
        chosen={"action": "run", "cmd": "python -c \"from main import app; print('OK')\""},
        rejected="I would import the app from main.py to check that it initializes without errors.",
    ),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def generate() -> list[dict]:
    all_examples = (
        MARKDOWN_FENCE_EXAMPLES
        + PROSE_PREFIX_EXAMPLES
        + WRONG_KEYS_EXAMPLES
        + NATURAL_LANGUAGE_EXAMPLES
    )
    random.shuffle(all_examples)
    return all_examples


def main() -> None:
    from finetune.config import DPO_DATA_PATH
    examples = generate()
    DPO_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DPO_DATA_PATH.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"DPO dataset: {len(examples)} examples → {DPO_DATA_PATH}")
    print(f"  Markdown fence:    {len(MARKDOWN_FENCE_EXAMPLES)}")
    print(f"  Prose prefix:      {len(PROSE_PREFIX_EXAMPLES)}")
    print(f"  Wrong keys:        {len(WRONG_KEYS_EXAMPLES)}")
    print(f"  Natural lang only: {len(NATURAL_LANGUAGE_EXAMPLES)}")


if __name__ == "__main__":
    main()
