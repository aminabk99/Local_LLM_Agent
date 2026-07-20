<div align="center">

# 🤖 AI LLM Coding Agent
### A Fully Local AI Agent That Writes, Reads, and Executes Code Autonomously

A Python coding agent powered by **Ollama + qwen2.5-coder:7b** that takes a task, breaks it into tool calls, and autonomously writes files, reads code, and runs shell commands — all in a structured **JSON action loop** with up to 20 agent steps. Exposed via a **FastAPI** `/chat` endpoint with a dark-themed web UI.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-qwen2.5--coder:7b-black?style=for-the-badge)
![Local](https://img.shields.io/badge/100%25_Local-No_API_Key-4CAF50?style=for-the-badge&logo=homeassistant&logoColor=white)

</div>

---

## How It Works

1. You send a task to the `/chat` endpoint — e.g. *"Write a Python script that scrapes headlines"*
2. The agent enters a **JSON action loop** — each step the LLM returns one action as a JSON object
3. The agent executes that action using one of four tools, appends the result to history, and loops
4. After up to **20 steps** the agent calls `finish` and returns the final response
5. All file operations are **sandboxed to the `workspace/` folder** — nothing outside can be touched

**Tools available:** 📁 `list_files` · 📖 `read_file` · ✏️ `write_file` · ⚙️ `run_command`

---

## The AI layer

The agent isn't just a prompt in a loop — it implements two published agent techniques and leans on constrained decoding for reliability.

### ReAct — reason, act, observe (Yao et al., 2023)
> Yao, S. et al. *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR 2023.

Each step the model observes the tool history and emits **one** grounded action as JSON; the agent executes it, appends the observation, and loops. That observe→act→observe cycle is ReAct: the model reasons over concrete tool results rather than hallucinating a whole plan up front. Implemented as the single loop in `agent/core.py`.

### Reflexion — verbal self-correction on failure (Shinn et al., 2023)
> Shinn, N. et al. *Reflexion: Language Agents with Verbal Reinforcement Learning.* NeurIPS 2023.

When a `run` command exits non-zero, the agent doesn't just retry blindly. It makes a short **reflection** call — "why did that fail, what will you change?" — and injects that self-critique back into the history as an observation before the next action. So a failed test run becomes a diagnosis the next step can act on. Implemented as `OllamaLLM.reflect()`, gated by `AGENT_REFLECT` (default on).

### Structured decoding — the real JSON fix
Every step calls `json.loads()` on the model output, so a single malformed object breaks the loop. Rather than rely on cleanup, the agent asks Ollama for `format="json"`, which **constrains decoding to valid JSON** — the model structurally cannot emit markdown fences or prose prefixes. The `{...}`-slice recovery parser is kept only as a fallback. (The `finetune/` track below is the alternative approach: teach a small model the format via LoRA/DPO instead of constraining it.)

---

---

## Setup

**Requirements:** Python 3.11+ · [Ollama](https://ollama.com)

**1. Clone & install**
```bash
git clone https://github.com/aminabk99/Local_LLM_Agent
cd Local_LLM_Agent
python -m venv venv && source venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

**2. Pull the model** (~4.7GB)
```bash
ollama pull qwen2.5-coder:7b
```

**3. Start Ollama**
```bash
ollama serve
```

**4. Run the server**
```bash
python main.py
```

Open `http://localhost:8000` in your browser to use the chat UI, or hit `/docs` for the interactive API docs.

---

## Project Structure

```
AI_LLM/
├── main.py                  # FastAPI server — /chat endpoint, session history, web UI
├── requirements.txt         # Python dependencies
├── agent/
│   ├── agent.py             # Agent runner — orchestrates the JSON action loop
│   ├── core.py              # CodingAgent class with step logic and approval gate
│   ├── llm_ollama.py        # Ollama API client with JSON recovery and 600s timeout
│   ├── prompts.py           # System prompt, ReAct prompt builder, Reflexion prompt
│   ├── tools.py             # Workspace (list/read/write) + run_command + destructive denylist
│   └── ui.html              # Dark-themed web UI served at /
├── tests/                   # pytest: sandbox traversal, JSON recovery, agent loop (no Ollama needed)
├── .github/workflows/       # CI — runs pytest on every push
└── workspace/               # Sandboxed folder — agent reads/writes only here
```

**Config via environment variables:** `AGENT_MODEL` (default `qwen2.5-coder:7b`), `AGENT_MAX_STEPS` (20), `AGENT_ALLOW_SHELL` (true), `AGENT_REQUIRE_APPROVAL` (false), `AGENT_REFLECT` (true), `AGENT_CORS_ORIGINS` (localhost only).

---

## API

**POST `/chat`**
```json
{
  "message": "Write a FastAPI hello world app",
  "session_id": ""
}
```
Returns:
```json
{
  "reply": "Done! I wrote main.py with a /hello endpoint...",
  "session_id": "abc-123",
  "steps_taken": 4
}
```

Sessions are multi-turn: prior turns are passed back into the agent as conversation context, so follow-up messages build on earlier ones.

**GET `/history/{session_id}`** — retrieve full chat history for a session

**DELETE `/history/{session_id}`** — clear a session

---

## Hardest Part
**Getting reliable JSON out of a chatty model.** `qwen2.5-coder:7b` would wrap its JSON in markdown fences or prefix it with prose, and every step calls `json.loads()`, so one bad object breaks the loop. The fix is layered: Ollama's `format="json"` constrains decoding to valid JSON at the source, a `{`…`}`-slice parser recovers anything that still slips through, and if both fail the agent finishes gracefully with the raw output rather than crashing. The `finetune/` track takes the orthogonal route — teaching a small model the format directly with LoRA + DPO.

## Most Interesting
**The 600s timeout** — on first run, Ollama has to load the full 7B model into memory which can take 30–60 seconds on a CPU. Without the extended timeout the agent would crash before the model even responded. A short retry message explains what happened if it still times out, rather than just throwing an exception.

---

## Security

- **File operations are jailed to `workspace/`** — paths are resolved and checked with `is_relative_to`, and traversal attempts (`../…`) raise a `ValueError`. This part is a real boundary and is covered by tests.
- **Shell execution is best-effort, not a sandbox — stated honestly.** `run_command` uses a real shell, so a command like `cd .. && …` can technically reach outside `workspace/`. Mitigations: a denylist blocks obviously destructive commands (`rm -rf /`, fork bombs, `mkfs`, …), and shell can be turned off entirely with `AGENT_ALLOW_SHELL=false`. For untrusted input, disable shell or run the whole thing in a container.
- **Web exposure is locked down by default.** CORS defaults to localhost only (`AGENT_CORS_ORIGINS`), and the server binds to `127.0.0.1`. Set `AGENT_REQUIRE_APPROVAL=true` to gate every write/run behind a manual `y/n` when running the CLI.
- **No API keys, no cloud calls** — the model runs entirely on your machine.

---

<div align="center">
  <sub>Built by <a href="https://github.com/aminabk99">Amina Bilal</a> · <a href="https://linkedin.com/in/amina-bilal-926340382">LinkedIn</a></sub>
</div>

---

## Fine-Tuning with LoRA & DPO (alternative track)

**In the shipped agent, JSON reliability is handled at inference by Ollama's `format="json"` constrained decoding — not by fine-tuning.** This section is a separate, self-contained experiment: *can a tiny model be taught the action format directly, so it emits clean JSON without constrained decoding?* It's here as a learning exercise and a portfolio piece, not a dependency of the running agent.

The JSON action loop is a precision formatting task: exactly one JSON object per step, right action key, required fields present. Any deviation breaks the loop — which makes it a clean target for LoRA SFT + DPO.

### What's in `finetune/`

| File | Purpose |
|------|---------|
| `config.py` | Base model, LoRA rank/alpha, training hyperparameters |
| `data/generate_sft.py` | Generates 80 supervised (prompt → JSON) examples |
| `data/generate_dpo.py` | Generates 60 preference pairs (chosen: clean JSON vs. rejected: broken output) |
| `train_sft.py` | LoRA SFT using TRL's SFTTrainer — teaches output format |
| `train_dpo.py` | DPO from SFT checkpoint — reinforces preference via contrastive signal |
| `eval.py` | Measures JSON parse rate, action validity, and key completeness before vs. after |

**Base model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` — same family as the Ollama `tinyllama` used in benchmarking, small enough to fine-tune on a single consumer GPU (6–8 GB VRAM) or in Google Colab.

**LoRA config:** rank r=16, alpha=32 (scaling=2.0), targeting all projection matrices (q/k/v/o + gate/up/down).

### Dataset

| Split | Examples | Covers |
|-------|----------|--------|
| SFT | 80 | All 5 action types: finish, list\_files, read\_file, write\_file, run |
| DPO chosen | 60 | Clean single JSON objects |
| DPO rejected | 60 | Markdown-wrapped (15), prose prefix (15), wrong keys (15), natural language only (15) |

### Quick start

```bash
# 1. Install fine-tuning dependencies
pip install torch transformers peft trl datasets accelerate

# 2. Generate datasets
python -m finetune.data.generate_sft
python -m finetune.data.generate_dpo

# 3. SFT training (~20 min on a single GPU)
python -m finetune.train_sft

# 4. DPO training (~10 min on a single GPU)
python -m finetune.train_dpo

# 5. Evaluate: JSON parse rate before vs. after
python -m finetune.eval --adapter both
```

### How it's evaluated

`finetune/eval.py` runs the base model and the fine-tuned adapter on the same agent prompts and reports three rates:

| Metric | Meaning |
|--------|---------|
| JSON parse rate | fraction of responses that are valid, parseable JSON |
| Action validity | JSON is valid **and** the `action` key is a known action |
| Key completeness | all required keys are present for the chosen action |

> **No results are checked into this repo.** Training an adapter needs a GPU, and no adapter or eval run is committed here, so I'm not quoting parse-rate numbers I haven't produced — run `python -m finetune.eval --adapter both` to generate real before/after figures on your own hardware. JSON parse rate is the metric that matters because every agent step calls `json.loads()`, so a failed parse is a broken step. (For context, the local `benchmark/` run only exercises JSON formatting on a single task, so it isn't a parse-*rate* — it's one datapoint, not a benchmark of the format.)

### Running with the fine-tuned adapter in Ollama

After fine-tuning, convert the adapter to GGUF and push to Ollama:

```bash
# Merge LoRA weights into base model
python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
model = PeftModel.from_pretrained(base, 'finetune/adapters/dpo')
merged = model.merge_and_unload()
merged.save_pretrained('finetune/merged')
AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0').save_pretrained('finetune/merged')
print('Merged model saved to finetune/merged/')
"

# Convert to GGUF (requires llama.cpp)
python llama.cpp/convert_hf_to_gguf.py finetune/merged --outfile finetune/tinyllama-json-agent.gguf

# Register with Ollama
echo 'FROM ./finetune/tinyllama-json-agent.gguf' > Modelfile
ollama create tinyllama-json-agent -f Modelfile

# Use in the agent
python main.py --model tinyllama-json-agent
```

---

## Real-Time Streaming & Multimodal Agent 

The agent loop now streams token-by-token via Server-Sent Events, and accepts CI error screenshots as a second input modality.

### Architecture

```
User task (+ optional screenshot)
           │
           ▼
POST /stream/run          — streams every agent step in real time
POST /stream/run (image)  — llava extracts errors from screenshot → injects into task
           │
  ┌────────┴──────────────────────────────────┐
  │  step_start → token → token → action      │
  │  → tool_start → tool_done → step_start... │
  │  → finish                                  │
  └────────────────────────────────────────────┘
         Server-Sent Events (one frame per event)
```

### Quick start

```bash
# Install streaming deps
pip install httpx uvicorn fastapi

# Start the streaming server (separate from the CLI agent)
uvicorn streaming.sse_server:app --host 0.0.0.0 --port 8001 --reload

# Demo client (in another terminal)
python -m streaming.sse_server client
```

### Text-only task

```python
import httpx, json

async with httpx.AsyncClient(timeout=300) as client:
    async with client.stream(
        "POST", "http://localhost:8001/stream/run",
        json={"task": "Write a hello world script and run it."}
    ) as resp:
        async for line in resp.aiter_lines():
            if line.startswith("data:"):
                event = json.loads(line[5:])
                if event["type"] == "token":
                    print(event["text"], end="", flush=True)
                elif event["type"] == "tool_done":
                    print(f"\n[{event['tool']}] {event['result'][:80]}")
                elif event["type"] == "finish":
                    print(f"\nDone in {event['total_steps']} steps")
```

### Multimodal task (with error screenshot)

```python
import base64

with open("terminal_error.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

async with client.stream(
    "POST", "http://localhost:8001/stream/run",
    json={"task": "Fix the failing tests", "image_b64": image_b64}
) as resp:
    async for line in resp.aiter_lines():
        event = json.loads(line[5:])
        if event["type"] == "vision_done":
            print(f"[Screenshot context] {event['augmented_task'][:200]}")
        elif event["type"] == "token":
            print(event["text"], end="", flush=True)
```

llava reads the screenshot, extracts visible error messages/tracebacks, and injects them into the task description so the agent has full context without you having to manually copy error text.

### SSE event reference

| Event type | Key fields | When |
|-----------|-----------|------|
| `start` | task, model | Agent loop begins |
| `step_start` | step, max_steps | Each new step |
| `token` | text, step, elapsed_ms | Each LLM token |
| `action` | action dict, step | LLM output parsed |
| `tool_start` | tool, step | Before tool execution |
| `tool_done` | tool, result, elapsed_ms | After tool returns |
| `step_timeout` | step | Step exceeded timeout |
| `finish` | result, total_steps, total_ms | Agent completes |
| `vision_start` | — | Screenshot analysis begins |
| `vision_done` | augmented_task | Screenshot analysis complete |

### What's in `streaming/`

| File | Purpose |
|------|---------|
| `config.py` | Step timeout, model, max steps |
| `stream_llm.py` | Ollama streaming for one action step — yields token events |
| `stream_agent.py` | Full agent loop emitting step/tool/finish events |
| `multimodal.py` | llava screenshot → extracted error context → task injection |
| `sse_server.py` | FastAPI app: `POST /stream/run`, `GET /health` |
