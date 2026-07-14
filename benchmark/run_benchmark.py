#!/usr/bin/env python3
"""
Local_LLM_Agent — Multi-Model Inference Benchmark
===================================================
Runs a fixed coding task suite through 3 local SLMs via Ollama and records:

  tokens/sec        — from Ollama eval_count / eval_duration
  TTFT              — from Ollama prompt_eval_duration
  JSON parse rate   — critical for this agent (LLM must emit valid JSON actions)
  Task success      — did the response contain meaningful code/reasoning?

Models benchmarked
------------------
  tinyllama        1.1B params  ~638 MB   speed champion
  phi3:mini        3.8B params  ~2.3 GB   balanced
  qwen2.5-coder:7b 7B   params  ~4.7 GB   coding quality champion (already in use)

Why these three?
  - tinyllama vs phi3:mini vs qwen2.5-coder shows the tradeoff curve clearly
  - qwen2.5-coder is already the production model — benchmark shows WHY it was chosen
  - All run 100%% locally, zero data leaves the machine

Usage
-----
    python -m benchmark.run_benchmark
    python -m benchmark.run_benchmark --models tinyllama phi3:mini
    python -m benchmark.run_benchmark --output benchmark/results/run_001.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODELS  = ["tinyllama", "phi3:mini", "qwen2.5-coder:7b"]
TASKS_FILE      = Path(__file__).parent / "tasks.json"

SYSTEM_PROMPT = (
    "You are a precise coding assistant. "
    "When asked for a JSON action, respond with valid JSON only — no markdown, no explanation. "
    "When asked for code, write clean, working Python."
)


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def _pull_model(model: str) -> None:
    print(f"  Checking / pulling {model} …", end="", flush=True)
    with httpx.stream(
        "POST",
        f"{OLLAMA_BASE_URL}/api/pull",
        json={"name": model, "stream": True},
        timeout=600,
    ) as r:
        for line in r.iter_lines():
            if line and json.loads(line).get("status") == "success":
                print(" done.")
                return
    print(" done.")


def _generate(model: str, prompt: str) -> dict:
    payload = {
        "model":   model,
        "system":  SYSTEM_PROMPT,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": 0.1, "num_predict": 400},
    }
    resp = httpx.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------

def _check_json_parseable(text: str) -> bool:
    """Try to extract and parse a JSON object from the response."""
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if not match:
        return False
    try:
        json.loads(match.group())
        return True
    except json.JSONDecodeError:
        return False


def _check_task_success(task_id: str, response: str) -> bool:
    """Heuristic success check per task type."""
    r = response.lower()
    checks = {
        "hello_world":       lambda: "def greet" in r and "return" in r,
        "json_parse":        lambda: "def " in r and "json" in r,
        "file_read":         lambda: _check_json_parseable(response),
        "error_handling":    lambda: "def safe_divide" in r and "except" in r,
        "privacy_reasoning": lambda: len(response.split()) > 30,
    }
    fn = checks.get(task_id)
    return fn() if fn else len(response) > 20


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _extract_metrics(response: dict, wall_time: float) -> dict:
    ns = 1e9
    eval_count    = response.get("eval_count", 0)
    eval_dur_ns   = response.get("eval_duration", 1)
    prompt_dur_ns = response.get("prompt_eval_duration", 0)
    total_dur_ns  = response.get("total_duration", 0)

    return {
        "tokens_generated": eval_count,
        "tokens_per_sec":   round(eval_count / (eval_dur_ns / ns), 2) if eval_dur_ns else 0.0,
        "ttft_s":           round(prompt_dur_ns / ns, 3),
        "generation_s":     round(eval_dur_ns / ns, 3),
        "total_s":          round(total_dur_ns / ns, 3) if total_dur_ns else round(wall_time, 3),
        "wall_time_s":      round(wall_time, 3),
    }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    models: list[str] = DEFAULT_MODELS,
    pull: bool = True,
) -> dict:
    with open(TASKS_FILE) as fh:
        tasks = json.load(fh)["tasks"]

    if pull:
        print("\nEnsuring models are available …")
        for model in models:
            _pull_model(model)

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models":    models,
        "tasks":     [t["id"] for t in tasks],
        "runs":      [],
    }

    for model in models:
        print(f"\n{'='*56}")
        print(f"Model: {model}")
        print(f"{'='*56}")

        model_result = {
            "model":          model,
            "task_results":   [],
            "mean_tokens_per_sec": 0.0,
            "mean_ttft_s":         0.0,
            "mean_total_s":        0.0,
            "json_parse_rate":     0.0,
            "task_success_rate":   0.0,
        }

        for task in tasks:
            print(f"  [{task['id']}] … ", end="", flush=True)
            try:
                t0       = time.perf_counter()
                response = _generate(model, task["prompt"])
                wall     = time.perf_counter() - t0

                answer      = response.get("response", "").strip()
                metrics     = _extract_metrics(response, wall)
                json_ok     = _check_json_parseable(answer) if "json" in task["id"] or "file_read" in task["id"] else None
                task_ok     = _check_task_success(task["id"], answer)

                result = {
                    "task_id":       task["id"],
                    "category":      task["category"],
                    "answer":        answer[:400],
                    "json_parseable": json_ok,
                    "task_success":  task_ok,
                    **metrics,
                }
                model_result["task_results"].append(result)

                print(
                    f"{metrics['tokens_per_sec']:.1f} tok/s  "
                    f"TTFT {metrics['ttft_s']:.2f}s  "
                    f"{'✓' if task_ok else '✗'} success"
                )

            except Exception as exc:
                print(f"ERROR: {exc}")
                model_result["task_results"].append({
                    "task_id": task["id"], "category": task["category"],
                    "error": str(exc), "task_success": False,
                    "json_parseable": False, "tokens_per_sec": 0,
                    "ttft_s": 0, "total_s": 0, "tokens_generated": 0, "wall_time_s": 0,
                })

        # Aggregate
        ok = [r for r in model_result["task_results"] if "error" not in r]
        json_tasks = [r for r in ok if r.get("json_parseable") is not None]

        if ok:
            model_result["mean_tokens_per_sec"] = round(
                sum(r["tokens_per_sec"] for r in ok) / len(ok), 2)
            model_result["mean_ttft_s"] = round(
                sum(r["ttft_s"] for r in ok) / len(ok), 3)
            model_result["mean_total_s"] = round(
                sum(r["total_s"] for r in ok) / len(ok), 3)
            model_result["task_success_rate"] = round(
                sum(1 for r in ok if r["task_success"]) / len(ok), 3)
        if json_tasks:
            model_result["json_parse_rate"] = round(
                sum(1 for r in json_tasks if r["json_parseable"]) / len(json_tasks), 3)

        results["runs"].append(model_result)
        print(f"\n  Summary → {model_result['mean_tokens_per_sec']} tok/s  "
              f"success {model_result['task_success_rate']:.0%}  "
              f"JSON parse {model_result['json_parse_rate']:.0%}")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Local_LLM_Agent multi-model benchmark")
    parser.add_argument("--models",   nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--no-pull",  action="store_true")
    parser.add_argument("--output",   default=None)
    args = parser.parse_args()

    results  = run_benchmark(args.models, pull=not args.no_pull)
    out_path = args.output or (
        f"benchmark/results/benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")

    from benchmark.report import print_summary_table
    print_summary_table(results)


if __name__ == "__main__":
    main()
