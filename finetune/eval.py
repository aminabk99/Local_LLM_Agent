#!/usr/bin/env python3
"""
Fine-tuning evaluation: JSON parse rate before vs. after LoRA/DPO.

Loads the base model and the fine-tuned adapter, runs both on the same
set of agent prompts, and measures:

  1. JSON parse rate       — fraction of responses that are valid parseable JSON
  2. Action validity rate  — JSON is valid AND action key is a known action type
  3. Key completeness rate — all required keys present for the chosen action

These three metrics map directly to the benchmark's json_parse_rate and
show the fine-tuning delta clearly for portfolio purposes.

Usage:
    # Evaluate base model vs. SFT adapter
    python -m finetune.eval --adapter sft

    # Evaluate base model vs. DPO adapter (recommended — best results)
    python -m finetune.eval --adapter dpo

    # Both
    python -m finetune.eval --adapter both
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REQUIRED_KEYS: dict[str, list[str]] = {
    "finish":     ["final"],
    "list_files": [],
    "read_file":  ["path"],
    "write_file": ["path", "content"],
    "run":        ["cmd"],
}

VALID_ACTIONS = set(REQUIRED_KEYS.keys())

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

# 20 eval prompts (not in the training set)
EVAL_PROMPTS = [
    "Write a Python script that prints the Fibonacci sequence up to 100.",
    "List the files in the workspace to get started.",
    "Read the file main.py to understand the current code.",
    "Run the existing test file with pytest.",
    "Write a function that checks if a number is prime.",
    "Create a requirements.txt for a machine learning project (numpy, pandas, scikit-learn).",
    "Install the packages from requirements.txt.",
    "The tests are passing. Summarise what was done and finish.",
    "Read config.yaml before modifying it.",
    "Write a script that sorts a list of numbers and prints the result.",
    "Check what Python files are available before making changes.",
    "Run flake8 to find style issues.",
    "Write a simple class called Stack with push and pop methods.",
    "Create a __init__.py to make the directory a Python package.",
    "Run the main script and check for errors.",
    "Write a decorator that logs function calls to stdout.",
    "Read utils.py to see what helper functions already exist.",
    "Finish the task — all tests pass and code is formatted.",
    "Write a context manager for opening files safely.",
    "Run mypy on the codebase to check type annotations.",
]


def _check_deps() -> None:
    missing = []
    for pkg in ("torch", "transformers", "peft"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing: {', '.join(missing)}\npip install torch transformers peft")
        sys.exit(1)


def _try_parse_json(text: str) -> dict | None:
    """Try to parse JSON from raw model output, with markdown-fence recovery."""
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # Recovery: find first { ... }
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
    return None


def _score(obj: dict | None) -> tuple[bool, bool, bool]:
    """Returns (json_valid, action_valid, keys_complete)."""
    if obj is None:
        return False, False, False
    action = obj.get("action")
    if action not in VALID_ACTIONS:
        return True, False, False
    required = REQUIRED_KEYS[action]
    keys_ok = all(k in obj for k in required)
    return True, True, keys_ok


def _build_prompt(task: str) -> str:
    return (
        f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
        f"<|user|>\n"
        f"TASK:\n{task}\n\n"
        "RECENT TOOL HISTORY:\n(none yet)\n\n"
        "Now choose the next single action as JSON only."
        "</s>\n"
        "<|assistant|>\n"
    )


def _eval_model(model, tokenizer, device, label: str) -> dict:
    """Run all eval prompts on a model and return metrics."""
    import torch

    json_valid = 0
    action_valid = 0
    keys_complete = 0
    results = []

    print(f"\n{'─'*60}")
    print(f" Evaluating: {label}")
    print(f"{'─'*60}")

    for i, task in enumerate(EVAL_PROMPTS):
        prompt = _build_prompt(task)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        obj = _try_parse_json(text)
        jv, av, kc = _score(obj)

        json_valid    += int(jv)
        action_valid  += int(av)
        keys_complete += int(kc)

        status = "✓" if kc else ("~" if jv else "✗")
        print(f"  [{status}] {i+1:2d}. {task[:55]:<55} → {text[:40]}")

        results.append({
            "task": task,
            "output": text,
            "json_valid": jv,
            "action_valid": av,
            "keys_complete": kc,
        })

    n = len(EVAL_PROMPTS)
    return {
        "label": label,
        "n": n,
        "json_parse_rate":    round(json_valid / n, 3),
        "action_valid_rate":  round(action_valid / n, 3),
        "key_complete_rate":  round(keys_complete / n, 3),
        "results": results,
    }


def _print_comparison(scores: list[dict]) -> None:
    print(f"\n{'='*65}")
    print("  FINE-TUNING EVAL RESULTS")
    print(f"{'='*65}")
    print(f"{'Model':<22} {'JSON Parse':>12} {'Action Valid':>14} {'Keys Complete':>14}")
    print("─" * 65)
    for s in scores:
        print(
            f"  {s['label']:<20} "
            f"{s['json_parse_rate']:>10.1%} "
            f"{s['action_valid_rate']:>14.1%} "
            f"{s['key_complete_rate']:>14.1%}"
        )
    print("─" * 65)
    if len(scores) >= 2:
        base = scores[0]
        best = scores[-1]
        delta_jp = best["json_parse_rate"] - base["json_parse_rate"]
        delta_kc = best["key_complete_rate"] - base["key_complete_rate"]
        sign_jp = "+" if delta_jp >= 0 else ""
        sign_kc = "+" if delta_kc >= 0 else ""
        print(
            f"\n  Delta ({best['label']} vs {base['label']}):\n"
            f"    JSON parse rate:   {sign_jp}{delta_jp:.1%}\n"
            f"    Key complete rate: {sign_kc}{delta_kc:.1%}"
        )
    print(f"{'='*65}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval fine-tuned adapter vs. base model")
    parser.add_argument("--adapter", choices=["sft", "dpo", "both"], default="dpo",
                        help="Which adapter to evaluate (default: dpo)")
    parser.add_argument("--output", default="finetune/eval_results.json",
                        help="Where to save detailed results JSON")
    args = parser.parse_args()

    _check_deps()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    from finetune.config import BASE_MODEL, SFT_ADAPTER, DPO_ADAPTER

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Device: {device} | dtype: {dtype}")
    print(f"Base model: {BASE_MODEL}\n")

    # ------------------------------------------------------------------
    # Load tokenizer + base model
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, device_map=device, trust_remote_code=True
    )
    base_model.eval()

    all_scores = []

    # Eval base model
    all_scores.append(_eval_model(base_model, tokenizer, device, "Base (TinyLlama)"))

    # Eval SFT adapter
    if args.adapter in ("sft", "both"):
        if not SFT_ADAPTER.exists():
            print(f"SFT adapter not found at {SFT_ADAPTER}. Run train_sft.py first.")
        else:
            sft_model = PeftModel.from_pretrained(base_model, str(SFT_ADAPTER))
            sft_model.eval()
            all_scores.append(_eval_model(sft_model, tokenizer, device, "SFT (LoRA r=16)"))

    # Eval DPO adapter
    if args.adapter in ("dpo", "both"):
        if not DPO_ADAPTER.exists():
            print(f"DPO adapter not found at {DPO_ADAPTER}. Run train_dpo.py first.")
        else:
            dpo_model = PeftModel.from_pretrained(base_model, str(DPO_ADAPTER))
            dpo_model.eval()
            all_scores.append(_eval_model(dpo_model, tokenizer, device, "DPO (SFT→DPO)"))

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    _print_comparison(all_scores)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_scores, f, indent=2)
    print(f"Detailed results saved → {out_path}")


if __name__ == "__main__":
    main()
