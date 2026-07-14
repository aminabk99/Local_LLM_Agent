#!/usr/bin/env python3
"""
Generate the SFT (Supervised Fine-Tuning) dataset for the JSON action loop.

Each example is a (prompt, completion) pair:
  - prompt     = system + task description + tool history so far
  - completion = the single correct JSON action (NO markdown, NO explanation)

The dataset is designed to teach the model:
  1. Always output one JSON object per response
  2. Include exactly the right keys for each action type
  3. Never wrap JSON in markdown fences or add prose

Covers all 5 action types:
  finish, list_files, read_file, write_file, run

Output: finetune/data/sft_dataset.jsonl  (one JSON object per line)

Usage:
    python -m finetune.data.generate_sft
"""

from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(42)

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


def _ex(task: str, action: dict, history: list[dict] | None = None) -> dict:
    """Build one SFT example."""
    return {
        "system": SYSTEM_PROMPT,
        "prompt": _prompt(task, history),
        "completion": json.dumps(action),
    }


# ---------------------------------------------------------------------------
# finish examples (15)
# ---------------------------------------------------------------------------
FINISH_EXAMPLES = [
    _ex(
        "Write a hello world script and run it.",
        {"action": "finish", "final": "Done. hello.py written and executed successfully — output was 'Hello, World!'."},
        history=[
            {"tool": "write_file", "result": "Written: hello.py"},
            {"tool": "run", "result": "Hello, World!\n"},
        ],
    ),
    _ex(
        "Create a simple calculator module with add, subtract, multiply, divide functions.",
        {"action": "finish", "final": "calculator.py created with add, subtract, multiply, and divide functions. All four operations tested and verified."},
        history=[
            {"tool": "write_file", "result": "Written: calculator.py"},
            {"tool": "run", "result": "All tests passed.\n"},
        ],
    ),
    _ex(
        "Read config.json and print the 'debug' field value.",
        {"action": "finish", "final": "Read config.json. The 'debug' field is set to true."},
        history=[
            {"tool": "read_file", "result": '{"debug": true, "port": 8080}'},
        ],
    ),
    _ex(
        "List the files in the workspace.",
        {"action": "finish", "final": "Files in workspace: main.py, utils.py, requirements.txt, README.md"},
        history=[
            {"tool": "list_files", "result": "main.py\nutils.py\nrequirements.txt\nREADME.md"},
        ],
    ),
    _ex(
        "Fix the syntax error in app.py and verify it runs.",
        {"action": "finish", "final": "Fixed syntax error on line 12 (missing colon after if statement). app.py now runs without errors."},
        history=[
            {"tool": "read_file", "result": "def greet(name)\n    return f'Hello {name}'"},
            {"tool": "write_file", "result": "Written: app.py"},
            {"tool": "run", "result": ""},
        ],
    ),
    _ex(
        "Create a requirements.txt with flask and requests.",
        {"action": "finish", "final": "requirements.txt created with flask and requests listed."},
        history=[
            {"tool": "write_file", "result": "Written: requirements.txt"},
        ],
    ),
    _ex(
        "Write a function that reverses a string and test it.",
        {"action": "finish", "final": "reverse_string() written in utils.py and tested. 'hello' → 'olleh' verified."},
        history=[
            {"tool": "write_file", "result": "Written: utils.py"},
            {"tool": "run", "result": "olleh\n"},
        ],
    ),
    _ex(
        "Add a docstring to every function in helpers.py.",
        {"action": "finish", "final": "Added docstrings to all 4 functions in helpers.py: parse_date, format_output, validate_input, retry."},
        history=[
            {"tool": "read_file", "result": "def parse_date(s): ..."},
            {"tool": "write_file", "result": "Written: helpers.py"},
        ],
    ),
    _ex(
        "Install dependencies and run the test suite.",
        {"action": "finish", "final": "pip install completed. pytest ran 12 tests — all passed."},
        history=[
            {"tool": "run", "result": "Successfully installed flask-3.0.0"},
            {"tool": "run", "result": "12 passed in 0.43s"},
        ],
    ),
    _ex(
        "Rename all occurrences of 'foo' to 'bar' in main.py.",
        {"action": "finish", "final": "Replaced 7 occurrences of 'foo' with 'bar' in main.py."},
        history=[
            {"tool": "read_file", "result": "def foo():\n    return foo_value"},
            {"tool": "write_file", "result": "Written: main.py"},
        ],
    ),
    _ex(
        "Create an empty __init__.py for the utils package.",
        {"action": "finish", "final": "Created utils/__init__.py as an empty file."},
        history=[
            {"tool": "write_file", "result": "Written: utils/__init__.py"},
        ],
    ),
    _ex(
        "Check if tests pass after refactoring.",
        {"action": "finish", "final": "All 8 tests pass after refactoring. No regressions."},
        history=[
            {"tool": "run", "result": "8 passed in 0.31s"},
        ],
    ),
    _ex(
        "Generate a .gitignore for a Python project.",
        {"action": "finish", "final": ".gitignore created with standard Python patterns: __pycache__/, *.pyc, .env, venv/, dist/, .pytest_cache/."},
        history=[
            {"tool": "write_file", "result": "Written: .gitignore"},
        ],
    ),
    _ex(
        "Write a script to count lines in all .py files.",
        {"action": "finish", "final": "count_lines.py written and executed. Found 3 Python files totalling 142 lines."},
        history=[
            {"tool": "write_file", "result": "Written: count_lines.py"},
            {"tool": "run", "result": "main.py: 87 lines\nutils.py: 34 lines\ntest_main.py: 21 lines\nTotal: 142"},
        ],
    ),
    _ex(
        "Add type hints to all function signatures in parser.py.",
        {"action": "finish", "final": "Added type hints to 5 functions in parser.py. Verified with mypy — no errors."},
        history=[
            {"tool": "read_file", "result": "def parse(text): ..."},
            {"tool": "write_file", "result": "Written: parser.py"},
            {"tool": "run", "result": "Success: no issues found in 1 source file"},
        ],
    ),
]


# ---------------------------------------------------------------------------
# list_files examples (10)
# ---------------------------------------------------------------------------
LIST_FILES_EXAMPLES = [
    _ex("I need to understand what's in the workspace before starting.", {"action": "list_files"}),
    _ex("First, see what files exist.", {"action": "list_files"}),
    _ex("Check what Python files are available.", {"action": "list_files"}),
    _ex("Before writing anything, survey the existing files.", {"action": "list_files"}),
    _ex("List all files to find the main entry point.", {"action": "list_files"}),
    _ex(
        "Fix the import error in the project.",
        {"action": "list_files"},
        history=[{"tool": "run", "result": "ModuleNotFoundError: No module named 'utils'"}],
    ),
    _ex(
        "Add a new feature to the existing codebase.",
        {"action": "list_files"},
        history=[],
    ),
    _ex("What files do we have to work with?", {"action": "list_files"}),
    _ex("Start by auditing the project structure.", {"action": "list_files"}),
    _ex(
        "Update the configuration file.",
        {"action": "list_files"},
        history=[],
    ),
]


# ---------------------------------------------------------------------------
# read_file examples (15)
# ---------------------------------------------------------------------------
READ_FILE_EXAMPLES = [
    _ex(
        "Fix the bug in main.py.",
        {"action": "read_file", "path": "main.py"},
        history=[{"tool": "list_files", "result": "main.py\nutils.py\ntests.py"}],
    ),
    _ex(
        "Add error handling to the database module.",
        {"action": "read_file", "path": "database.py"},
        history=[{"tool": "list_files", "result": "database.py\nmodels.py\napp.py"}],
    ),
    _ex(
        "Update the config values in settings.py.",
        {"action": "read_file", "path": "settings.py"},
        history=[{"tool": "list_files", "result": "settings.py\nmain.py\nrequirements.txt"}],
    ),
    _ex(
        "Read what's currently in requirements.txt before updating it.",
        {"action": "read_file", "path": "requirements.txt"},
        history=[{"tool": "list_files", "result": "requirements.txt\napp.py"}],
    ),
    _ex(
        "The test is failing. Read the test file to understand what it expects.",
        {"action": "read_file", "path": "test_app.py"},
        history=[
            {"tool": "run", "result": "FAILED test_app.py::test_calculate - AssertionError"},
        ],
    ),
    _ex(
        "Refactor utils.py to use dataclasses.",
        {"action": "read_file", "path": "utils.py"},
        history=[{"tool": "list_files", "result": "utils.py\nmain.py"}],
    ),
    _ex(
        "Add logging to every function in helpers.py.",
        {"action": "read_file", "path": "helpers.py"},
        history=[{"tool": "list_files", "result": "helpers.py\napp.py"}],
    ),
    _ex(
        "Optimize the slow function in data_processor.py.",
        {"action": "read_file", "path": "data_processor.py"},
        history=[{"tool": "list_files", "result": "data_processor.py\napp.py"}],
    ),
    _ex(
        "Review parser.py before making changes.",
        {"action": "read_file", "path": "parser.py"},
        history=[{"tool": "list_files", "result": "parser.py\ntokenizer.py\nmain.py"}],
    ),
    _ex(
        "Read the README to understand the project structure.",
        {"action": "read_file", "path": "README.md"},
        history=[{"tool": "list_files", "result": "README.md\nmain.py\napp.py"}],
    ),
    _ex(
        "Fix the ValueError in api.py.",
        {"action": "read_file", "path": "api.py"},
        history=[
            {"tool": "run", "result": "ValueError: invalid literal for int() with base 10: 'abc'\n  File 'api.py', line 34"},
        ],
    ),
    _ex(
        "Add type hints to model.py.",
        {"action": "read_file", "path": "model.py"},
        history=[{"tool": "list_files", "result": "model.py\ntrain.py\neval.py"}],
    ),
    _ex(
        "Check what the current version number is in version.py.",
        {"action": "read_file", "path": "version.py"},
        history=[{"tool": "list_files", "result": "version.py\nmain.py"}],
    ),
    _ex(
        "Read config.json to understand the current settings.",
        {"action": "read_file", "path": "config.json"},
        history=[{"tool": "list_files", "result": "config.json\nmain.py\napp.py"}],
    ),
    _ex(
        "Understand the existing schema before adding a new field.",
        {"action": "read_file", "path": "schema.py"},
        history=[{"tool": "list_files", "result": "schema.py\nmodels.py"}],
    ),
]


# ---------------------------------------------------------------------------
# write_file examples (20)
# ---------------------------------------------------------------------------
WRITE_FILE_EXAMPLES = [
    _ex(
        "Create a hello world Python script.",
        {"action": "write_file", "path": "hello.py", "content": "print('Hello, World!')\n"},
    ),
    _ex(
        "Write a requirements.txt with flask and requests.",
        {"action": "write_file", "path": "requirements.txt", "content": "flask>=3.0.0\nrequests>=2.31.0\n"},
    ),
    _ex(
        "Create a .gitignore for a Python project.",
        {"action": "write_file", "path": ".gitignore", "content": "__pycache__/\n*.pyc\n*.pyo\n.env\nvenv/\n.venv/\ndist/\nbuild/\n*.egg-info/\n.pytest_cache/\n.mypy_cache/\n"},
    ),
    _ex(
        "Write a simple add function to utils.py.",
        {"action": "write_file", "path": "utils.py", "content": "def add(a: int, b: int) -> int:\n    \"\"\"Return the sum of a and b.\"\"\"\n    return a + b\n"},
    ),
    _ex(
        "Create a config.json with debug=false and port=8080.",
        {"action": "write_file", "path": "config.json", "content": '{\n  "debug": false,\n  "port": 8080\n}\n'},
    ),
    _ex(
        "Fix the missing colon syntax error in app.py.",
        {"action": "write_file", "path": "app.py", "content": "def greet(name: str) -> str:\n    return f'Hello {name}'\n"},
        history=[
            {"tool": "read_file", "result": "def greet(name)\n    return f'Hello {name}'"},
        ],
    ),
    _ex(
        "Write a pytest test for the add function in utils.py.",
        {"action": "write_file", "path": "test_utils.py", "content": "from utils import add\n\ndef test_add():\n    assert add(2, 3) == 5\n    assert add(-1, 1) == 0\n    assert add(0, 0) == 0\n"},
    ),
    _ex(
        "Create an empty __init__.py for the package.",
        {"action": "write_file", "path": "__init__.py", "content": ""},
    ),
    _ex(
        "Write a simple FastAPI app with a /health endpoint.",
        {"action": "write_file", "path": "main.py", "content": "from fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get('/health')\ndef health():\n    return {'status': 'ok'}\n"},
    ),
    _ex(
        "Create a Makefile with install and test targets.",
        {"action": "write_file", "path": "Makefile", "content": ".PHONY: install test\n\ninstall:\n\tpip install -r requirements.txt\n\ntest:\n\tpytest tests/ -v\n"},
    ),
    _ex(
        "Write a version.py with __version__ = '1.0.0'.",
        {"action": "write_file", "path": "version.py", "content": "__version__ = '1.0.0'\n"},
    ),
    _ex(
        "Update main.py to add logging at the start.",
        {"action": "write_file", "path": "main.py", "content": "import logging\n\nlogging.basicConfig(level=logging.INFO)\nlogger = logging.getLogger(__name__)\n\ndef main():\n    logger.info('Starting...')\n"},
        history=[
            {"tool": "read_file", "result": "def main():\n    pass\n"},
        ],
    ),
    _ex(
        "Write a dataclass for a User with name and email fields.",
        {"action": "write_file", "path": "models.py", "content": "from dataclasses import dataclass\n\n@dataclass\nclass User:\n    name: str\n    email: str\n"},
    ),
    _ex(
        "Create a script that prints numbers 1 to 10.",
        {"action": "write_file", "path": "count.py", "content": "for i in range(1, 11):\n    print(i)\n"},
    ),
    _ex(
        "Write a reverse_string function in utils.py.",
        {"action": "write_file", "path": "utils.py", "content": "def reverse_string(s: str) -> str:\n    \"\"\"Return the reverse of string s.\"\"\"\n    return s[::-1]\n"},
    ),
    _ex(
        "Create a setup.py for the project.",
        {"action": "write_file", "path": "setup.py", "content": "from setuptools import setup, find_packages\n\nsetup(\n    name='myproject',\n    version='0.1.0',\n    packages=find_packages(),\n    python_requires='>=3.9',\n)\n"},
    ),
    _ex(
        "Add type hints to the existing greet function.",
        {"action": "write_file", "path": "app.py", "content": "def greet(name: str) -> str:\n    return f'Hello, {name}!'\n"},
        history=[
            {"tool": "read_file", "result": "def greet(name):\n    return f'Hello, {name}!'"},
        ],
    ),
    _ex(
        "Write a context manager for timing code blocks.",
        {"action": "write_file", "path": "timer.py", "content": "import time\nfrom contextlib import contextmanager\n\n@contextmanager\ndef timer(label: str = 'block'):\n    t0 = time.perf_counter()\n    yield\n    elapsed = time.perf_counter() - t0\n    print(f'{label}: {elapsed:.3f}s')\n"},
    ),
    _ex(
        "Create a retry decorator with exponential backoff.",
        {"action": "write_file", "path": "retry.py", "content": "import time\nfrom functools import wraps\n\ndef retry(max_attempts: int = 3, delay: float = 1.0):\n    def decorator(fn):\n        @wraps(fn)\n        def wrapper(*args, **kwargs):\n            for attempt in range(1, max_attempts + 1):\n                try:\n                    return fn(*args, **kwargs)\n                except Exception as e:\n                    if attempt == max_attempts:\n                        raise\n                    time.sleep(delay * 2 ** (attempt - 1))\n        return wrapper\n    return decorator\n"},
    ),
    _ex(
        "Write a simple LRU cache implementation.",
        {"action": "write_file", "path": "cache.py", "content": "from functools import lru_cache\n\n@lru_cache(maxsize=128)\ndef fibonacci(n: int) -> int:\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n"},
    ),
]


# ---------------------------------------------------------------------------
# run examples (20)
# ---------------------------------------------------------------------------
RUN_EXAMPLES = [
    _ex(
        "Run the hello world script.",
        {"action": "run", "cmd": "python hello.py"},
        history=[{"tool": "write_file", "result": "Written: hello.py"}],
    ),
    _ex(
        "Install the dependencies from requirements.txt.",
        {"action": "run", "cmd": "pip install -r requirements.txt"},
        history=[{"tool": "write_file", "result": "Written: requirements.txt"}],
    ),
    _ex(
        "Run the test suite with pytest.",
        {"action": "run", "cmd": "pytest test_utils.py -v"},
        history=[{"tool": "write_file", "result": "Written: test_utils.py"}],
    ),
    _ex(
        "Check the Python version.",
        {"action": "run", "cmd": "python --version"},
    ),
    _ex(
        "Run mypy on utils.py to check type hints.",
        {"action": "run", "cmd": "mypy utils.py --strict"},
        history=[{"tool": "write_file", "result": "Written: utils.py"}],
    ),
    _ex(
        "Format main.py with black.",
        {"action": "run", "cmd": "black main.py"},
        history=[{"tool": "write_file", "result": "Written: main.py"}],
    ),
    _ex(
        "Run flake8 to check for style issues.",
        {"action": "run", "cmd": "flake8 . --max-line-length 100"},
    ),
    _ex(
        "Count the lines of code in all Python files.",
        {"action": "run", "cmd": "find . -name '*.py' | xargs wc -l"},
    ),
    _ex(
        "Start the FastAPI server on port 8000.",
        {"action": "run", "cmd": "uvicorn main:app --host 0.0.0.0 --port 8000 --reload"},
        history=[{"tool": "write_file", "result": "Written: main.py"}],
    ),
    _ex(
        "Run the data processing script.",
        {"action": "run", "cmd": "python data_processor.py"},
        history=[{"tool": "write_file", "result": "Written: data_processor.py"}],
    ),
    _ex(
        "Check which packages are installed.",
        {"action": "run", "cmd": "pip list"},
    ),
    _ex(
        "Run all tests and show coverage.",
        {"action": "run", "cmd": "pytest tests/ --cov=. --cov-report=term-missing"},
    ),
    _ex(
        "Verify the script exits cleanly.",
        {"action": "run", "cmd": "python main.py && echo 'Exit code 0'"},
        history=[{"tool": "write_file", "result": "Written: main.py"}],
    ),
    _ex(
        "Sort imports with isort.",
        {"action": "run", "cmd": "isort . --profile black"},
    ),
    _ex(
        "Run the benchmarks.",
        {"action": "run", "cmd": "python -m benchmark.run_benchmark"},
    ),
    _ex(
        "After fixing the IndexError, verify the fix.",
        {"action": "run", "cmd": "python app.py"},
        history=[
            {"tool": "read_file", "result": "IndexError on line 22"},
            {"tool": "write_file", "result": "Written: app.py"},
        ],
    ),
    _ex(
        "Install torch for the fine-tuning script.",
        {"action": "run", "cmd": "pip install torch transformers peft trl --quiet"},
    ),
    _ex(
        "Run the training script.",
        {"action": "run", "cmd": "python -m finetune.train_sft"},
    ),
    _ex(
        "Generate the SFT dataset.",
        {"action": "run", "cmd": "python -m finetune.data.generate_sft"},
    ),
    _ex(
        "Run a quick sanity check on the model output.",
        {"action": "run", "cmd": "python -c \"from model.inference import classify; print(classify('AADSTS50126'))\""},
    ),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def generate() -> list[dict]:
    all_examples = (
        FINISH_EXAMPLES
        + LIST_FILES_EXAMPLES
        + READ_FILE_EXAMPLES
        + WRITE_FILE_EXAMPLES
        + RUN_EXAMPLES
    )
    random.shuffle(all_examples)
    return all_examples


def main() -> None:
    from finetune.config import SFT_DATA_PATH
    examples = generate()
    SFT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SFT_DATA_PATH.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"SFT dataset: {len(examples)} examples → {SFT_DATA_PATH}")
    action_counts: dict[str, int] = {}
    for ex in examples:
        action = json.loads(ex["completion"]).get("action", "?")
        action_counts[action] = action_counts.get(action, 0) + 1
    for action, count in sorted(action_counts.items()):
        print(f"  {action:<14} {count} examples")


if __name__ == "__main__":
    main()
